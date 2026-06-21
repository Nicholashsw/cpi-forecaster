"""yfinance source. Daily closes resampled to month-end; release_date = month-end + market lag."""
from __future__ import annotations

import pandas as pd

from macro_cpi import config
from macro_cpi.db import get_engine, release_date_for, upsert_observations
from macro_cpi.ingestion.base import to_observation_rows, validate_frame
from macro_cpi.logging_conf import get_logger

logger = get_logger(__name__)

SOURCE = "YFINANCE"


def fetch(ticker: str, label: str, downloader=None) -> pd.DataFrame:  # noqa: ANN001
    """Fetch a ticker's daily close, resample to month-end, stamp market publication lag."""
    if downloader is None:
        import yfinance as yf

        downloader = yf.download
    raw = downloader(
        ticker,
        start=config.HISTORY_START.isoformat(),
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    if raw is None or len(raw) == 0:
        raise ValueError(f"[{SOURCE}] no data for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # multiindex when single ticker passed as list
        close = close.iloc[:, 0]
    monthly = close.resample("ME").last().dropna()
    rows = [
        {
            "series_id": ticker,
            "date": ts.date(),
            "value": float(val),
            "source": SOURCE,
            "release_date": release_date_for(ts.date(), config.MARKET_PUBLICATION_LAG_DAYS),
        }
        for ts, val in monthly.items()
    ]
    df = pd.DataFrame(rows)
    logger.info("[%s] fetched %s (%s) rows=%d", SOURCE, ticker, label, len(df))
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a fetched market frame."""
    return validate_frame(df, SOURCE)


def store(df: pd.DataFrame, db_url: str | None = None) -> int:
    """Persist a validated market frame; return rows written."""
    engine = get_engine(db_url)
    return upsert_observations(engine, to_observation_rows(df))


def run(db_url: str | None = None) -> int:
    """Fetch, validate, and store every configured market ticker; return total rows written."""
    total = 0
    for ticker, label in config.YF_TICKERS:
        total += store(validate(fetch(ticker, label)), db_url=db_url)
    return total
