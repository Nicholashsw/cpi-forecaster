"""FRED ingestion tests with mocked HTTP — no live network."""
from __future__ import annotations

import pandas as pd

from macro_cpi import config
from macro_cpi.db import get_engine, init_db, load_series
from macro_cpi.ingestion import fred

SPEC = config.SeriesSpec("CPIAUCSL", "CPI", 14)

FAKE_PAYLOAD = {
    "observations": [
        {"date": "2020-01-01", "value": "257.971"},
        {"date": "2020-02-01", "value": "258.678"},
        {"date": "2020-03-01", "value": "."},  # missing value handled as null
    ]
}


def test_fetch_parses_and_stamps_release_date(monkeypatch):
    monkeypatch.setattr(fred, "get_json", lambda url, params=None: FAKE_PAYLOAD)
    df = fred.fetch(SPEC, api_key="dummy")
    assert list(df["series_id"].unique()) == ["CPIAUCSL"]
    assert len(df) == 3
    assert df["value"].isna().sum() == 1
    # release_date = reference + 14 days
    first = df.iloc[0]
    assert (pd.Timestamp(first["release_date"]) - pd.Timestamp(first["date"])).days == 14


def test_fetch_requires_key(monkeypatch):
    monkeypatch.setattr(config, "FRED_API_KEY", None)
    try:
        fred.fetch(SPEC, api_key=None)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_validate_and_store_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(fred, "get_json", lambda url, params=None: FAKE_PAYLOAD)
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = get_engine(db_url)
    init_db(engine)
    df = fred.validate(fred.fetch(SPEC, api_key="dummy"))
    written = fred.store(df, db_url=db_url)
    assert written == 3
    loaded = load_series(engine, "CPIAUCSL")
    assert len(loaded) == 3
    assert loaded["release_date"].notna().all()
