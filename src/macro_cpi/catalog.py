"""Data catalog: the authoritative registry of every series this project ingests.

Every entry documents the source, FRED/API identifier, frequency, publication lag, units,
and a direct link to the source page. Config and ingestion modules derive their views from
this catalog so there is no duplication — to add or retire a series, edit it here only.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataSeries:
    """Provenance record for one macro or market series. Frozen for hashability."""

    series_id: str
    label: str
    source: str            # "FRED" | "YFINANCE" | "TREASURY"
    frequency: str         # "M" monthly, "D" daily (resampled to month-end at ingestion)
    publication_lag_days: int
    units: str
    description: str
    source_url: str
    notes: str = ""


# ---- FRED macroeconomic series -----------------------------------------------------------
FRED: tuple[DataSeries, ...] = (
    DataSeries(
        "CPIAUCSL", "CPI-U all items SA", "FRED", "M", 14,
        "Index 1982-84=100",
        "Consumer Price Index for All Urban Consumers, all items, seasonally adjusted. "
        "Primary forecasting target. Released around the 10th-15th of the following month.",
        "https://fred.stlouisfed.org/series/CPIAUCSL",
        notes="Among the least-revised major US macro series after first print.",
    ),
    DataSeries(
        "CPILFESL", "Core CPI SA", "FRED", "M", 14,
        "Index 1982-84=100",
        "CPI-U less food and energy, seasonally adjusted. Less noisy than headline; "
        "often used as a forecasting target in its own right.",
        "https://fred.stlouisfed.org/series/CPILFESL",
    ),
    DataSeries(
        "PCEPI", "PCE price index", "FRED", "M", 30,
        "Index 2017=100",
        "Personal Consumption Expenditures price index. The Fed's preferred inflation gauge.",
        "https://fred.stlouisfed.org/series/PCEPI",
    ),
    DataSeries(
        "PAYEMS", "Nonfarm payrolls", "FRED", "M", 7,
        "Thousands of persons",
        "Total nonfarm payroll employment. Released first Friday of the following month.",
        "https://fred.stlouisfed.org/series/PAYEMS",
        notes="Heavily revised in subsequent prints; lag-table PIT is an approximation.",
    ),
    DataSeries(
        "UNRATE", "Unemployment rate", "FRED", "M", 7,
        "Percent",
        "Civilian unemployment rate, seasonally adjusted. Released with NFP.",
        "https://fred.stlouisfed.org/series/UNRATE",
    ),
    DataSeries(
        "RSAFS", "Retail sales", "FRED", "M", 16,
        "Millions of dollars",
        "Advance retail sales: retail trade and food services, total, seasonally adjusted.",
        "https://fred.stlouisfed.org/series/RSAFS",
    ),
    DataSeries(
        "INDPRO", "Industrial production", "FRED", "M", 16,
        "Index 2017=100",
        "Industrial production index, seasonally adjusted.",
        "https://fred.stlouisfed.org/series/INDPRO",
    ),
    DataSeries(
        "FEDFUNDS", "Effective fed funds rate", "FRED", "M", 1,
        "Percent",
        "Effective federal funds rate, monthly average.",
        "https://fred.stlouisfed.org/series/FEDFUNDS",
    ),
    DataSeries(
        "DGS2", "2-year Treasury yield", "FRED", "D", 1,
        "Percent",
        "Constant-maturity 2-year Treasury yield. Daily series resampled to month-end.",
        "https://fred.stlouisfed.org/series/DGS2",
    ),
    DataSeries(
        "DGS10", "10-year Treasury yield", "FRED", "D", 1,
        "Percent",
        "Constant-maturity 10-year Treasury yield. Daily series resampled to month-end.",
        "https://fred.stlouisfed.org/series/DGS10",
    ),
)

# ---- Market series (yfinance) ------------------------------------------------------------
YFINANCE: tuple[DataSeries, ...] = (
    DataSeries(
        "SPY", "S&P 500 ETF", "YFINANCE", "D", 1,
        "USD per share",
        "SPDR S&P 500 ETF adjusted close. Proxy for US equity total return.",
        "https://finance.yahoo.com/quote/SPY",
    ),
    DataSeries(
        "DX-Y.NYB", "US Dollar Index", "YFINANCE", "D", 1,
        "Index",
        "ICE US Dollar Index (DXY) — USD against a basket of major currencies.",
        "https://finance.yahoo.com/quote/DX-Y.NYB",
    ),
    DataSeries(
        "CL=F", "WTI crude futures", "YFINANCE", "D", 1,
        "USD per barrel",
        "Front-month WTI crude oil futures. Energy is a major CPI driver via gasoline.",
        "https://finance.yahoo.com/quote/CL=F",
    ),
    DataSeries(
        "GC=F", "Gold futures", "YFINANCE", "D", 1,
        "USD per troy ounce",
        "Front-month gold futures. Real-rate and inflation-expectations proxy.",
        "https://finance.yahoo.com/quote/GC=F",
    ),
)

# ---- US Treasury fiscal data API ---------------------------------------------------------
TREASURY: tuple[DataSeries, ...] = (
    DataSeries(
        "TREAS_AVG_RATE_MARKETABLE", "Avg interest rate on marketable debt",
        "TREASURY", "M", 30, "Percent",
        "Weighted average interest rate on US Treasury total marketable debt outstanding. "
        "Slow-moving funding-cost proxy.",
        "https://fiscaldata.treasury.gov/datasets/average-interest-rates-treasury-securities/",
    ),
)

# ---- Master view --------------------------------------------------------------------------
ALL_SERIES: tuple[DataSeries, ...] = FRED + YFINANCE + TREASURY


def by_source(source: str) -> tuple[DataSeries, ...]:
    """Return all catalog entries for a given source."""
    return tuple(s for s in ALL_SERIES if s.source == source)


def get(series_id: str) -> DataSeries:
    """Look up a single series by id; raise if not in the catalog."""
    for s in ALL_SERIES:
        if s.series_id == series_id:
            return s
    raise KeyError(f"series_id {series_id!r} not in catalog")
