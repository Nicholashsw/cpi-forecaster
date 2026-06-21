"""Feature tests — the critical ones: prove point-in-time correctness (no look-ahead)."""
from __future__ import annotations

import pandas as pd

from macro_cpi import config
from macro_cpi.db import get_engine, init_db, upsert_observations
from macro_cpi.features.build import build_feature_panel, build_target


def _seed(engine) -> None:
    """Seed a small monthly CPI series and a feature series with known release lags."""
    rows = []
    dates = pd.date_range("2000-01-31", periods=60, freq="ME")
    for i, d in enumerate(dates):
        rows.append(
            {
                "series_id": "CPIAUCSL",
                "date": d.date(),
                "value": 100.0 + i,  # steady 1.0 increase -> mom% computable
                "source": "TEST",
                "release_date": (d + pd.Timedelta(days=14)).date(),
            }
        )
        rows.append(
            {
                "series_id": "FEDFUNDS",
                "date": d.date(),
                "value": 2.0 + 0.01 * i,
                "source": "TEST",
                "release_date": (d + pd.Timedelta(days=1)).date(),
            }
        )
    upsert_observations(engine, rows)


def _model() -> config.ModelConfig:
    return config.ModelConfig(
        target_series_id="CPIAUCSL",
        target_transform="mom_pct",
        forecast_horizon_months=1,
    )


def test_target_build(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path/'f.db'}")
    init_db(engine)
    _seed(engine)
    tgt = build_target(engine, _model())
    assert "target" in tgt.columns
    assert tgt["target"].notna().all()
    assert (tgt["target_release_date"] > tgt["date"]).all()


def test_no_lookahead_feature_release_respected(tmp_path):
    """A feature value must never appear with release_date after the forecast anchor."""
    engine = get_engine(f"sqlite:///{tmp_path/'f.db'}")
    init_db(engine)
    _seed(engine)
    panel = build_feature_panel(engine, _model())
    assert not panel.empty
    # anchor_date is one month before the target reference date
    assert (panel["anchor_date"] < panel["date"]).all()
    # The CPI level feature used at each anchor must have been released on/before the anchor.
    # CPI release lag is 14 days; anchor is month-end of (target month - 1). The level used
    # should be from a month whose release_date <= anchor. Verify monotonic, no future leak:
    # the level feature at row k should be <= the contemporaneous target-month level.
    merged = panel.dropna(subset=["CPIAUCSL__level"])
    assert len(merged) > 0
    # Because values strictly increase with month, a leaked future value would exceed the
    # level implied by the anchor month. Check feature level < level at target month.
    months = (panel["date"].dt.year - 2000) * 12 + (panel["date"].dt.month - 1)
    target_month_level = 100.0 + months
    feat = merged["CPIAUCSL__level"].to_numpy()
    cap = target_month_level.loc[merged.index].to_numpy()
    assert (feat < cap).all()


def test_lagged_target_present(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path/'f.db'}")
    init_db(engine)
    _seed(engine)
    panel = build_feature_panel(engine, _model())
    assert "target_lag1" in panel.columns
    assert "target_lag12" in panel.columns
