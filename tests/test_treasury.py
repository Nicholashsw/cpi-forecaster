"""Treasury ingestion tests with mocked HTTP."""
from __future__ import annotations

import pandas as pd

from macro_cpi.ingestion import treasury

FAKE = {
    "data": [
        {"record_date": "2021-01-31", "avg_interest_rate_amt": "1.701"},
        {"record_date": "2021-02-28", "avg_interest_rate_amt": "1.688"},
    ]
}


def test_fetch_parses(monkeypatch):
    monkeypatch.setattr(treasury, "get_json", lambda url, params=None: FAKE)
    df = treasury.fetch(api_base="https://example.test")
    assert len(df) == 2
    assert (df["series_id"] == "TREAS_AVG_RATE_MARKETABLE").all()
    assert df["value"].iloc[0] == 1.701


def test_release_lag(monkeypatch):
    monkeypatch.setattr(treasury, "get_json", lambda url, params=None: FAKE)
    df = treasury.validate(treasury.fetch(api_base="https://example.test"))
    delta = (pd.Timestamp(df.iloc[0]["release_date"]) - pd.Timestamp(df.iloc[0]["date"])).days
    assert delta == treasury.config.TREASURY_PUBLICATION_LAG_DAYS
