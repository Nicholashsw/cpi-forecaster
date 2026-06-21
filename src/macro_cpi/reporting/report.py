"""Stage 5 reporting: render a markdown run report with a predicted-vs-actual plot."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from macro_cpi import config  # noqa: E402
from macro_cpi.features.build import feature_columns  # noqa: E402
from macro_cpi.logging_conf import get_logger  # noqa: E402
from macro_cpi.models.walk_forward import WalkForwardResult  # noqa: E402

logger = get_logger(__name__)


def _plot_pred_vs_actual(result: WalkForwardResult, out_path: Path) -> None:
    """Save a predicted-vs-actual line plot across the walk-forward folds."""
    folds = result.folds
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(folds["date"], folds["y_true"], label="actual", linewidth=1.8)
    ax.plot(folds["date"], folds["pred_lgbm"], label="lgbm", alpha=0.8)
    ax.plot(folds["date"], folds["pred_ridge"], label="ridge", alpha=0.8)
    ax.plot(folds["date"], folds["pred_naive"], label="naive", linestyle="--", alpha=0.6)
    ax.set_title("Out-of-sample predicted vs actual (walk-forward)")
    ax.set_xlabel("forecast date")
    ax.set_ylabel(config.MODEL.target_transform)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _metrics_table(metrics: dict[str, dict[str, float]]) -> str:
    """Render the summary metrics dict as a markdown table."""
    header = "| model | RMSE | MAE | dir. acc. | naive RMSE | skill vs naive | beats naive |\n"
    header += "|---|---|---|---|---|---|---|\n"
    lines = []
    for name, m in metrics.items():
        lines.append(
            f"| {name} | {m['rmse']:.4f} | {m['mae']:.4f} | "
            f"{m['directional_accuracy']:.2%} | {m['naive_rmse']:.4f} | "
            f"{m['rmse_skill_vs_naive']:.2%} | {m['beats_naive']} |"
        )
    return header + "\n".join(lines)


def write_report(
    result: WalkForwardResult,
    panel,  # noqa: ANN001
    model: config.ModelConfig,
    reports_dir: Path | None = None,
    run_date: date | None = None,
) -> Path:
    """Write a markdown report and plot to reports/{run_date}_baseline.md; return its path."""
    reports_dir = reports_dir or config.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    run_date = run_date or date.today()

    plot_path = reports_dir / f"{run_date.isoformat()}_pred_vs_actual.png"
    _plot_pred_vs_actual(result, plot_path)

    feat_cols = feature_columns(panel)
    best = min(("ridge", "lgbm"), key=lambda k: result.metrics[k]["rmse"])
    verdict = (
        f"Best model **{best}** beats the naive last-value benchmark."
        if result.metrics[best]["beats_naive"]
        else f"**No model beats the naive last-value benchmark.** Best was {best}, "
        f"still worse than naive. Reported plainly as required."
    )

    folds = result.folds
    fold_lines = "\n".join(
        f"| {r.date.date()} | {r.y_true:.4f} | {r.pred_ridge:.4f} | "
        f"{r.pred_lgbm:.4f} | {r.pred_naive:.4f} |"
        for r in folds.itertuples()
    )

    md = f"""# Macro CPI Baseline Report — {run_date.isoformat()}

## Target
- Series: `{model.target_series_id}`
- Transform: `{model.target_transform}`
- Horizon: {model.forecast_horizon_months} month(s)

## Verdict
{verdict}

## Summary metrics (out-of-sample, walk-forward)
{_metrics_table(result.metrics)}

## Data coverage
- Forecast rows: {len(panel)}
- Walk-forward folds: {len(folds)}
- Span: {panel['date'].min().date()} .. {panel['date'].max().date()}

## Features ({len(feat_cols)})
{', '.join(f'`{c}`' for c in feat_cols)}

## Model parameters
- Ridge alpha: {model.ridge_alpha}
- LightGBM: {model.lgbm_params}
- Min train months before first OOS fold: {model.min_train_months}
- Validation: expanding-window walk-forward, one-step-ahead, no k-fold

## Point-in-time methodology
Features are assembled by as-of join on each series' `release_date`; a value is never used
before its publication. Release dates are stamped via the per-series publication-lag table in
`config.py`. Architecture is ALFRED-vintage-ready (realtime params) for a later upgrade.

## Predicted vs actual
![predicted vs actual]({plot_path.name})

## Fold-by-fold results
| date | actual | ridge | lgbm | naive |
|---|---|---|---|---|
{fold_lines}
"""
    out_path = reports_dir / f"{run_date.isoformat()}_baseline.md"
    out_path.write_text(md)
    logger.info("wrote report %s", out_path)
    return out_path
