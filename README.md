# macro-cpi

A reproducible, point-in-time macro forecasting engine. Predicts US headline CPI month-on-month % change using a 36-year out-of-sample walk-forward over 80+ features built from FRED, US Treasury, and yfinance data.

The honest read: **LightGBM beats the naive last-value benchmark by ~5% RMSE and ~4pp directional accuracy across 313 OOS folds (1990-2026)**, with the edge statistically supported by a Diebold-Mariano test and stable across decades. Ridge regression underperforms naive — kept in the report deliberately, to document what doesn't work.

## Why this exists

Most "I forecasted CPI" repos quietly use revised data and same-day feature timestamps, which inflates backtest skill by silently injecting look-ahead. This one stamps every observation with its public release date and assembles features by an as-of join on `release_date`, so a value is never used before it was published. The architecture is ALFRED-vintage-ready — swapping the lag-table for true real-time vintages is a config change, not a rewrite.

## Results at a glance

| model | OOS RMSE | OOS MAE | directional acc. | beats naive |
|---|---|---|---|---|
| **LightGBM** | **0.294** | **0.203** | **82%** | **yes (+4.9% skill)** |
| naive last-value | 0.309 | 0.220 | 78% | — |
| ridge | 0.466 | 0.285 | 74% | no (-51%) |

Walk-forward, expanding window, one-step-ahead. 313 out-of-sample folds. See [`notebooks/01_baseline.ipynb`](notebooks/01_baseline.ipynb) for the executed diagnostic notebook with plots, feature importance, rolling skill, DM test, and per-decade stability.

## Architecture

```
                  ┌─────────────────────────────────────────┐
                  │            catalog.py                   │
                  │  single source of truth: every series   │
                  │  with provenance + publication lag      │
                  └────────────────┬────────────────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       ▼                           ▼                           ▼
┌─────────────┐           ┌─────────────┐           ┌─────────────┐
│ FRED API    │           │ yfinance    │           │ Treasury    │
│ ingestion   │           │ ingestion   │           │ Fiscal API  │
└──────┬──────┘           └──────┬──────┘           └──────┬──────┘
       │                         │                         │
       └─────────────────────────┼─────────────────────────┘
                                 ▼
                  ┌─────────────────────────────────────┐
                  │  SQLite (observations table)        │
                  │  series_id, date, value, source,    │
                  │  release_date, fetched_at           │
                  └────────────────┬────────────────────┘
                                   ▼
                  ┌─────────────────────────────────────┐
                  │ features/build.py                   │
                  │ as-of join on release_date          │
                  │ → no look-ahead by construction     │
                  └────────────────┬────────────────────┘
                                   ▼
                  ┌─────────────────────────────────────┐
                  │ models/walk_forward.py              │
                  │ expanding-window, LGBM + ridge      │
                  │ vs naive last-value benchmark       │
                  └────────────────┬────────────────────┘
                                   ▼
                  ┌─────────────────────────────────────┐
                  │ diagnostics.py + reporting/         │
                  │ importance · rolling skill · DM     │
                  │ per-decade · residual autocorr      │
                  └─────────────────────────────────────┘
```

## Data sources

The full catalog with publication lags and source URLs lives in [`src/macro_cpi/catalog.py`](src/macro_cpi/catalog.py). Summary:

| source | series | examples |
|---|---|---|
| **FRED** | 10 | `CPIAUCSL`, `CPILFESL`, `PCEPI`, `PAYEMS`, `UNRATE`, `INDPRO`, `RSAFS`, `FEDFUNDS`, `DGS2`, `DGS10` |
| **yfinance** | 4 | `SPY`, `DX-Y.NYB` (DXY), `CL=F` (WTI), `GC=F` (gold) |
| **US Treasury fiscal data** | 1 | avg interest rate on marketable debt |

Every series has a documented publication lag stamped at ingestion. Daily series (`DGS2`, `DGS10`, and yfinance) are resampled to month-end to align with the monthly forecast cadence.

## Quickstart

```bash
git clone <repo> && cd macro-cpi
uv venv && . .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env          # add your free FRED API key
make all                      # ingest + features + walk-forward + report
```

Get a free FRED key in 30 seconds: <https://fredaccount.stlouisfed.org/apikeys>.

## Make targets

| target | what |
|---|---|
| `make ingest` | pull FRED + yfinance + Treasury into SQLite |
| `make report` | build features, run walk-forward, write markdown report |
| `make all` | ingest then report |
| `make notebook` | execute the diagnostic notebook end-to-end |
| `make test` | run pytest (19 tests, HTTP fully mocked) |
| `make lint` | run ruff |

## Point-in-time methodology

This is the load-bearing claim of the project, so it gets its own section.

Every observation in the database carries a `release_date` — when the value became publicly available. To build the feature row for a forecast anchored at time *t*, each series contributes only its most recent value with `release_date <= t`. The feature panel is the result of an **as-of merge on release_date** (not the reference date), enforced inside `features/build.py`.

For v1, `release_date` is stamped via a per-series publication-lag table in `catalog.py` (e.g. CPI ≈ 14 days after month-end, NFP ≈ 7 days, INDPRO ≈ 16 days). This is the right approximation for series with negligible revisions (CPI, market data) and a documented imperfect approximation for heavily-revised series (PAYEMS, GDP components).

For v2, the ingestion layer is already wired to accept FRED's `realtime_start`/`realtime_end` parameters for true ALFRED-vintage retrieval — a future config flip, not a rewrite. See the literature: Croushore & Stark on real-time datasets, Romer & Romer on inflation forecast bias from revisions.

## What's deliberately honest about this

- The **naive last-value baseline is reported alongside every model** — and when ridge loses to naive, the report says so plainly, not buried.
- The Diebold-Mariano test is included so "lgbm beats naive" can be assessed for statistical significance, not just point-estimate ranking.
- The **rolling skill chart** in the notebook exposes any single-decade artifacts — the edge has to persist across pre-GFC, ZIRP, and post-COVID regimes to be credible.
- **No hyperparameter sweeps on the test set.** Defaults are documented in `config.py` and used as-is. No "I tried 200 configs and report the best one."
- Tests mock HTTP rather than touching live APIs — the suite is hermetic and runs in seconds.

## Configuration

Every knob lives in [`config.py`](src/macro_cpi/config.py) and is overridable via `.env`. Target and horizon are swappable:

```bash
TARGET_SERIES_ID=CPILFESL   # forecast core CPI instead
TARGET_TRANSFORM=mom_pct    # mom_pct | mom_diff | yoy_pct
FORECAST_HORIZON_MONTHS=3   # 3-month-ahead
MIN_TRAIN_MONTHS=120        # warmup before first OOS fold
RIDGE_ALPHA=1.0
LGBM_N_ESTIMATORS=300
```

## Repository layout

```
macro-cpi/
├── src/macro_cpi/
│   ├── catalog.py             # authoritative data series registry
│   ├── config.py              # central config, .env-driven
│   ├── db.py                  # SQLAlchemy schema + upsert
│   ├── http.py                # retry/backoff HTTP client
│   ├── logging_conf.py
│   ├── ingestion/
│   │   ├── base.py            # shared fetch/validate/store interface
│   │   ├── fred.py            # FRED API
│   │   ├── market.py          # yfinance
│   │   └── treasury.py        # Treasury fiscal data
│   ├── features/
│   │   └── build.py           # PIT feature panel (as-of join)
│   ├── models/
│   │   └── walk_forward.py    # expanding-window walk-forward
│   ├── diagnostics.py         # importance, rolling skill, DM test
│   ├── reporting/report.py    # markdown report + plot
│   └── cli.py                 # macro-cpi {ingest,report,all}
├── tests/                     # 19 tests, HTTP mocked, no network
├── notebooks/01_baseline.ipynb # executed, with embedded plots
├── reports/{run_date}_baseline.md
├── data/macro.db
└── pyproject.toml             # uv + ruff + pytest config
```

## Tech stack

Python 3.11+, [uv](https://github.com/astral-sh/uv), pandas, numpy, scikit-learn, lightgbm, SQLAlchemy (SQLite), requests, yfinance, scipy, matplotlib, pytest, ruff.

## Roadmap

- [ ] ALFRED true-vintage ingestion path for heavily-revised series (PAYEMS, PCE)
- [ ] Free PMI proxy (FRED's `NAPM` is discontinued; alternative sources under evaluation)
- [ ] Add core PCE and core CPI as auxiliary targets in the same run
- [ ] Block-bootstrap confidence intervals on skill metrics
- [ ] Direct multi-step forecasting (h=3, h=6, h=12) with separate models per horizon

## License

MIT.
