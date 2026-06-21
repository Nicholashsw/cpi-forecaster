"""Shared ingestion contract: every source module exposes fetch/validate/store."""
from __future__ import annotations

import pandas as pd

from macro_cpi.db import Observation
from macro_cpi.logging_conf import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = {"series_id", "date", "value", "source", "release_date"}


def validate_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Validate a tidy observation frame: columns, types, no future release dates; log gaps."""
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"[{source}] missing columns: {missing}")
    if df.empty:
        raise ValueError(f"[{source}] produced zero rows")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["release_date"] = pd.to_datetime(df["release_date"])

    if (df["release_date"] < df["date"]).any():
        raise ValueError(f"[{source}] release_date precedes reference date for some rows")

    for series_id, grp in df.groupby("series_id"):
        n_null = int(grp["value"].isna().sum())
        gaps = _count_month_gaps(grp["date"])
        logger.info(
            "[%s] series=%s rows=%d nulls=%d month_gaps=%d span=%s..%s",
            source,
            series_id,
            len(grp),
            n_null,
            gaps,
            grp["date"].min().date(),
            grp["date"].max().date(),
        )
    return df


def _count_month_gaps(dates: pd.Series) -> int:
    """Count missing months in an otherwise monthly series (0 if not monthly-regular)."""
    d = pd.to_datetime(dates).sort_values().dt.to_period("M")
    if len(d) < 2:
        return 0
    full = pd.period_range(d.iloc[0], d.iloc[-1], freq="M")
    return int(len(full) - d.nunique())


def to_observation_rows(df: pd.DataFrame) -> list[dict]:
    """Convert a validated frame to row dicts ready for upsert into the observations table."""
    cols = ["series_id", "date", "value", "source", "release_date"]
    out = []
    for rec in df[cols].to_dict("records"):
        rec["date"] = pd.Timestamp(rec["date"]).date()
        rec["release_date"] = pd.Timestamp(rec["release_date"]).date()
        rec["value"] = None if pd.isna(rec["value"]) else float(rec["value"])
        out.append(rec)
    return out


# Sanity reference so linters keep the import; Observation is the storage target.
_STORAGE_TARGET = Observation
