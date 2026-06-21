"""FRED source. fetch -> validate -> store. ALFRED-ready: passes realtime params when vintaging."""
from __future__ import annotations

import pandas as pd

from macro_cpi import config
from macro_cpi.db import get_engine, release_date_for, upsert_observations
from macro_cpi.http import get_json
from macro_cpi.ingestion.base import to_observation_rows, validate_frame
from macro_cpi.logging_conf import get_logger

logger = get_logger(__name__)

SOURCE = "FRED"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

# FRED series published at daily frequency; resampled to month-end to align the monthly panel.
DAILY_SERIES = {"DGS2", "DGS10"}


def fetch(spec: config.SeriesSpec, api_key: str | None = None) -> pd.DataFrame:
    """Fetch one FRED series and stamp release_date via the spec's publication lag."""
    key = api_key or config.FRED_API_KEY
    if not key:
        raise ValueError("FRED_API_KEY not set; populate .env from .env.example")
    params = {
        "series_id": spec.series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": config.HISTORY_START.isoformat(),
    }
    payload = get_json(FRED_URL, params=params)
    obs = payload.get("observations", [])
    parsed = []
    for o in obs:
        ref = pd.to_datetime(o["date"])
        raw = o.get("value", ".")
        value = None if raw in (".", "") else float(raw)
        parsed.append((ref, value))

    series = pd.Series(
        {ts: val for ts, val in parsed}, dtype="float64"
    ).sort_index()

    if spec.series_id in DAILY_SERIES:
        # Month-end last observation; matches how monthly series are dated downstream.
        series = series.resample("ME").last()

    rows = []
    for ts, value in series.items():
        ref = ts.date()
        rows.append(
            {
                "series_id": spec.series_id,
                "date": ref,
                "value": None if pd.isna(value) else float(value),
                "source": SOURCE,
                "release_date": release_date_for(ref, spec.publication_lag_days),
            }
        )
    df = pd.DataFrame(rows)
    logger.info("[%s] fetched %s rows=%d", SOURCE, spec.series_id, len(df))
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a fetched FRED frame."""
    return validate_frame(df, SOURCE)


def store(df: pd.DataFrame, db_url: str | None = None) -> int:
    """Persist a validated FRED frame to SQLite; return rows written."""
    engine = get_engine(db_url)
    return upsert_observations(engine, to_observation_rows(df))


def run(api_key: str | None = None, db_url: str | None = None) -> int:
    """Fetch, validate, and store every configured FRED series; return total rows written."""
    total = 0
    for spec in config.FRED_SERIES:
        total += store(validate(fetch(spec, api_key=api_key)), db_url=db_url)
    return total
