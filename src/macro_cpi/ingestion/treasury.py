"""US Treasury fiscal data source. Pulls average interest rate on total marketable debt."""
from __future__ import annotations

import pandas as pd

from macro_cpi import config
from macro_cpi.db import get_engine, release_date_for, upsert_observations
from macro_cpi.http import get_json
from macro_cpi.ingestion.base import to_observation_rows, validate_frame
from macro_cpi.logging_conf import get_logger

logger = get_logger(__name__)

SOURCE = "TREASURY"
SERIES_ID = "TREAS_AVG_RATE_MARKETABLE"
ENDPOINT = "/v2/accounting/od/avg_interest_rates"


def fetch(api_base: str | None = None) -> pd.DataFrame:
    """Fetch monthly avg interest rate on Total Marketable debt; stamp Treasury publication lag."""
    base = api_base or config.TREASURY_API_BASE
    url = f"{base}{ENDPOINT}"
    params = {
        "filter": "security_type_desc:eq:Marketable,security_desc:eq:Total Marketable",
        "fields": "record_date,avg_interest_rate_amt",
        "sort": "record_date",
        "page[size]": "10000",
    }
    payload = get_json(url, params=params)
    data = payload.get("data", [])
    rows = []
    for d in data:
        ref = pd.to_datetime(d["record_date"]).date()
        raw = d.get("avg_interest_rate_amt")
        value = None if raw in (None, "", "null") else float(raw)
        rows.append(
            {
                "series_id": SERIES_ID,
                "date": ref,
                "value": value,
                "source": SOURCE,
                "release_date": release_date_for(ref, config.TREASURY_PUBLICATION_LAG_DAYS),
            }
        )
    df = pd.DataFrame(rows)
    logger.info("[%s] fetched %s rows=%d", SOURCE, SERIES_ID, len(df))
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a fetched Treasury frame."""
    return validate_frame(df, SOURCE)


def store(df: pd.DataFrame, db_url: str | None = None) -> int:
    """Persist a validated Treasury frame; return rows written."""
    engine = get_engine(db_url)
    return upsert_observations(engine, to_observation_rows(df))


def run(api_base: str | None = None, db_url: str | None = None) -> int:
    """Fetch, validate, and store Treasury data; return total rows written."""
    return store(validate(fetch(api_base)), db_url=db_url)
