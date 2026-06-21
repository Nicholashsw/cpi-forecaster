"""Market ingestion tests using an injected fake downloader — no live yfinance call."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_cpi.ingestion import market


def _fake_downloader(ticker, **kwargs):  # noqa: ANN001, ANN003
    idx = pd.date_range("2020-01-01", periods=90, freq="D")
    return pd.DataFrame({"Close": np.linspace(100, 110, len(idx))}, index=idx)


def test_fetch_resamples_to_month_end():
    df = market.fetch("SPY", "S&P 500 ETF", downloader=_fake_downloader)
    # 90 daily points across Jan–Mar -> 3 month-end rows
    assert len(df) == 3
    assert (df["series_id"] == "SPY").all()
    assert df["value"].notna().all()


def test_release_date_uses_market_lag():
    df = market.fetch("SPY", "label", downloader=_fake_downloader)
    delta = (pd.Timestamp(df.iloc[0]["release_date"]) - pd.Timestamp(df.iloc[0]["date"])).days
    assert delta == market.config.MARKET_PUBLICATION_LAG_DAYS


def test_validate_passes():
    df = market.validate(market.fetch("SPY", "label", downloader=_fake_downloader))
    assert not df.empty
