"""Central configuration: every tunable lives here or in .env. No magic numbers elsewhere."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from macro_cpi import catalog

load_dotenv()

# --- Paths (no hardcoded absolute paths anywhere; everything relative to project root) ---
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
DB_PATH: Path = DATA_DIR / os.getenv("DB_FILENAME", "macro.db")
DB_URL: str = os.getenv("DB_URL", f"sqlite:///{DB_PATH}")

# --- Secrets ---
FRED_API_KEY: str | None = os.getenv("FRED_API_KEY")

# --- HTTP / retry ---
HTTP_TIMEOUT_SECS: float = float(os.getenv("HTTP_TIMEOUT_SECS", "30"))
HTTP_MAX_RETRIES: int = int(os.getenv("HTTP_MAX_RETRIES", "4"))
HTTP_BACKOFF_BASE_SECS: float = float(os.getenv("HTTP_BACKOFF_BASE_SECS", "1.0"))

# --- Sample window ---
HISTORY_START: date = date.fromisoformat(os.getenv("HISTORY_START", "1990-01-01"))


# Compatibility alias: SeriesSpec is the lightweight tuple-style record that ingestion
# modules historically consumed. The catalog is the authoritative source.
@dataclass(frozen=True)
class SeriesSpec:
    """One macro series: its source-side id, human label, and publication lag in days."""

    series_id: str
    label: str
    publication_lag_days: int


def _spec(entry: catalog.DataSeries) -> SeriesSpec:
    """Adapt a catalog record into the legacy SeriesSpec ingestion shape."""
    return SeriesSpec(entry.series_id, entry.label, entry.publication_lag_days)


# Derived views consumed by the ingestion modules.
FRED_SERIES: tuple[SeriesSpec, ...] = tuple(_spec(s) for s in catalog.FRED)
YF_TICKERS: tuple[tuple[str, str], ...] = tuple((s.series_id, s.label) for s in catalog.YFINANCE)

MARKET_PUBLICATION_LAG_DAYS: int = 1

TREASURY_API_BASE: str = os.getenv(
    "TREASURY_API_BASE",
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service",
)
TREASURY_PUBLICATION_LAG_DAYS: int = catalog.TREASURY[0].publication_lag_days


@dataclass(frozen=True)
class ModelConfig:
    """Target definition, feature, and walk-forward validation settings."""

    target_series_id: str = os.getenv("TARGET_SERIES_ID", "CPIAUCSL")
    target_transform: str = os.getenv("TARGET_TRANSFORM", "mom_pct")  # mom_pct | mom_diff | yoy_pct
    forecast_horizon_months: int = int(os.getenv("FORECAST_HORIZON_MONTHS", "1"))

    zscore_windows: tuple[int, ...] = (12, 24)
    target_lags: tuple[int, ...] = (1, 2, 3, 12)

    min_train_months: int = int(os.getenv("MIN_TRAIN_MONTHS", "120"))  # 10y before first OOS fold
    ridge_alpha: float = float(os.getenv("RIDGE_ALPHA", "1.0"))
    lgbm_params: dict = field(
        default_factory=lambda: {
            "n_estimators": int(os.getenv("LGBM_N_ESTIMATORS", "300")),
            "learning_rate": float(os.getenv("LGBM_LEARNING_RATE", "0.03")),
            "num_leaves": int(os.getenv("LGBM_NUM_LEAVES", "15")),
            "min_child_samples": int(os.getenv("LGBM_MIN_CHILD_SAMPLES", "10")),
            "subsample": float(os.getenv("LGBM_SUBSAMPLE", "0.8")),
            "colsample_bytree": float(os.getenv("LGBM_COLSAMPLE", "0.8")),
            "random_state": int(os.getenv("RANDOM_STATE", "42")),
            "verbosity": -1,
        }
    )


MODEL = ModelConfig()
