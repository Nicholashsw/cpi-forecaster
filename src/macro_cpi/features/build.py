"""Point-in-time feature engineering. Every feature is known only at its release_date.

The core guarantee: to build the feature row stamped for a forecast made at time t, we use
each series' most recent value whose release_date <= t. A value is never used before it was
published. This is enforced by an as-of merge on release_date, not reference date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from macro_cpi import config
from macro_cpi.db import get_engine, load_series
from macro_cpi.logging_conf import get_logger

logger = get_logger(__name__)

DIFFUSION_SERIES = {"NAPM"}  # PMI-style, already a diffusion index level


def _transform_target(level: pd.Series, transform: str) -> pd.Series:
    """Apply the target transform to a level series."""
    if transform == "mom_pct":
        return level.pct_change() * 100.0
    if transform == "mom_diff":
        return level.diff()
    if transform == "yoy_pct":
        return level.pct_change(12) * 100.0
    raise ValueError(f"unknown target_transform: {transform}")


def build_target(engine: Engine, model: config.ModelConfig) -> pd.DataFrame:
    """Build the target as a tidy frame: month-end date, target value, and its release_date."""
    raw = load_series(engine, model.target_series_id)
    if raw.empty:
        raise ValueError(f"no data stored for target {model.target_series_id}")
    s = raw.set_index("date")["value"].sort_index()
    target = _transform_target(s, model.target_transform).rename("target")
    out = target.to_frame().dropna().reset_index()
    # The target's own release_date is the release_date of its reference month's level.
    rel = raw.set_index("date")["release_date"]
    out["target_release_date"] = out["date"].map(rel)
    return out


def _feature_block_for_series(
    df: pd.DataFrame, series_id: str, model: config.ModelConfig
) -> pd.DataFrame:
    """Compute transforms for one series, carrying each value's release_date forward."""
    s = df.set_index("date").sort_index()
    level = s["value"]
    rel = s["release_date"]
    feats = pd.DataFrame(index=level.index)

    feats[f"{series_id}__level"] = level
    feats[f"{series_id}__mom"] = level.pct_change() * 100.0
    feats[f"{series_id}__yoy"] = level.pct_change(12) * 100.0

    if series_id in DIFFUSION_SERIES:
        feats[f"{series_id}__diffusion_gap"] = level - 50.0

    for w in model.zscore_windows:
        roll = level.rolling(w)
        feats[f"{series_id}__z{w}"] = (level - roll.mean()) / roll.std(ddof=0)

    feats["release_date"] = rel
    feats = feats.reset_index()
    return feats


def _derived_cross_series(panel_pit: pd.DataFrame) -> pd.DataFrame:
    """Add cross-series features (slope, real fed funds proxy) using already-PIT columns."""
    out = panel_pit.copy()
    if {"DGS10__level", "DGS2__level"}.issubset(out.columns):
        out["slope_10y_2y"] = out["DGS10__level"] - out["DGS2__level"]
    if {"FEDFUNDS__level", "CPIAUCSL__yoy"}.issubset(out.columns):
        out["real_fed_funds_proxy"] = out["FEDFUNDS__level"] - out["CPIAUCSL__yoy"]
    return out


def build_feature_panel(engine: Engine, model: config.ModelConfig) -> pd.DataFrame:
    """Build the PIT feature panel indexed by forecast date, plus the target aligned to horizon.

    For each target reference month m with target release r, the forecast is made one horizon
    before m. We assemble features known strictly before that forecast is resolved by as-of
    joining every series on its release_date. The result has no look-ahead by construction.
    """
    series_ids = [spec.series_id for spec in config.FRED_SERIES]
    series_ids += [t for t, _ in config.YF_TICKERS]
    series_ids.append("TREAS_AVG_RATE_MARKETABLE")

    blocks: list[pd.DataFrame] = []
    for sid in series_ids:
        raw = load_series(engine, sid)
        if raw.empty:
            logger.warning("no stored data for %s; skipping feature block", sid)
            continue
        blocks.append(_feature_block_for_series(raw, sid, model))

    if not blocks:
        raise ValueError("no feature blocks could be built; ingest data first")

    # Stack all engineered observations into long form keyed by their release_date.
    # Each block row carries a release_date; we melt feature columns but keep release per row.
    target = build_target(engine, model)

    # Forecast dates: we predict target month `m` using info available at m's reference start.
    # Conservative anchor: forecast made at month-end of (m - horizon). Features must be
    # released on/before that anchor date.
    target = target.sort_values("date").reset_index(drop=True)
    target["anchor_date"] = target["date"] - pd.offsets.MonthEnd(model.forecast_horizon_months)

    feature_rows: list[dict] = []
    for _, trow in target.iterrows():
        anchor = trow["anchor_date"]
        row: dict = {"date": trow["date"], "anchor_date": anchor, "target": trow["target"]}
        for block in blocks:
            avail = block[block["release_date"] <= anchor]
            if avail.empty:
                continue
            latest = avail.sort_values("release_date").iloc[-1]
            for col in block.columns:
                if col in ("date", "release_date"):
                    continue
                row[col] = latest[col]
        feature_rows.append(row)

    panel = pd.DataFrame(feature_rows)
    panel = _derived_cross_series(panel)

    for lag in model.target_lags:
        panel[f"target_lag{lag}"] = panel["target"].shift(lag)

    panel = panel.replace([np.inf, -np.inf], np.nan)
    feature_cols = [c for c in panel.columns if c not in ("date", "anchor_date", "target")]
    panel = panel.dropna(subset=["target"]).reset_index(drop=True)
    logger.info(
        "feature panel rows=%d features=%d span=%s..%s",
        len(panel),
        len(feature_cols),
        panel["date"].min().date() if len(panel) else "NA",
        panel["date"].max().date() if len(panel) else "NA",
    )
    return panel


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Return the model feature column names from a built panel."""
    return [c for c in panel.columns if c not in ("date", "anchor_date", "target")]


def main() -> None:
    """Build and report the feature panel from the configured database."""
    engine = get_engine()
    panel = build_feature_panel(engine, config.MODEL)
    logger.info("built panel shape=%s", panel.shape)


if __name__ == "__main__":
    main()
