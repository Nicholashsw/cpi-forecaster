"""Models and expanding-window walk-forward validation. No k-fold. Naive last-value benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from macro_cpi import config
from macro_cpi.features.build import feature_columns
from macro_cpi.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class FoldResult:
    """One walk-forward step: the forecast date, prediction, actual, and naive baseline."""

    date: pd.Timestamp
    y_true: float
    pred_ridge: float
    pred_lgbm: float
    pred_naive: float


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward output: per-fold table and summary metrics per model."""

    folds: pd.DataFrame
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)


def _make_ridge(model: config.ModelConfig) -> Pipeline:
    """Build a standardized ridge regression pipeline."""
    return Pipeline(
        [("scale", StandardScaler()), ("ridge", Ridge(alpha=model.ridge_alpha))]
    )


def _make_lgbm(model: config.ModelConfig) -> LGBMRegressor:
    """Build a LightGBM regressor from configured params."""
    return LGBMRegressor(**model.lgbm_params)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, directional accuracy, and skill vs naive last-value."""
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    naive_rmse = float(np.sqrt(np.mean((y_naive - y_true) ** 2)))
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    # Skill: positive means better (lower error) than naive.
    skill = float(1.0 - rmse / naive_rmse) if naive_rmse > 0 else float("nan")
    beats_naive = bool(rmse < naive_rmse)
    return {
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": dir_acc,
        "naive_rmse": naive_rmse,
        "rmse_skill_vs_naive": skill,
        "beats_naive": beats_naive,
    }


def walk_forward(panel: pd.DataFrame, model: config.ModelConfig) -> WalkForwardResult:
    """Run expanding-window walk-forward; train on all past rows, predict the next, step forward."""
    panel = panel.sort_values("date").reset_index(drop=True)
    feat_cols = feature_columns(panel)

    start = model.min_train_months
    if len(panel) <= start:
        raise ValueError(
            f"not enough rows ({len(panel)}) for min_train_months={start}; "
            "lower MIN_TRAIN_MONTHS or extend history"
        )

    fold_rows: list[FoldResult] = []
    for i in range(start, len(panel)):
        train = panel.iloc[:i]
        test = panel.iloc[i : i + 1]

        x_train = train[feat_cols].to_numpy()
        y_train = train["target"].to_numpy()
        x_test = test[feat_cols].to_numpy()
        y_true = float(test["target"].iloc[0])

        # Impute train-mean for any residual NaN, fit on train stats only (no leakage).
        col_mean = np.nanmean(x_train, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        x_train = np.where(np.isnan(x_train), col_mean, x_train)
        x_test = np.where(np.isnan(x_test), col_mean, x_test)

        # Wrap in named frames so estimators see consistent feature names (no sklearn warning).
        x_train_df = pd.DataFrame(x_train, columns=feat_cols)
        x_test_df = pd.DataFrame(x_test, columns=feat_cols)

        ridge = _make_ridge(model).fit(x_train_df, y_train)
        lgbm = _make_lgbm(model).fit(x_train_df, y_train)

        # Naive = last observed target value as of the forecast (previous row's target).
        naive = float(train["target"].iloc[-1])

        fold_rows.append(
            FoldResult(
                date=test["date"].iloc[0],
                y_true=y_true,
                pred_ridge=float(ridge.predict(x_test_df)[0]),
                pred_lgbm=float(lgbm.predict(x_test_df)[0]),
                pred_naive=naive,
            )
        )

    folds = pd.DataFrame([f.__dict__ for f in fold_rows])
    y_true = folds["y_true"].to_numpy()
    y_naive = folds["pred_naive"].to_numpy()
    metrics = {
        "ridge": _metrics(y_true, folds["pred_ridge"].to_numpy(), y_naive),
        "lgbm": _metrics(y_true, folds["pred_lgbm"].to_numpy(), y_naive),
        "naive": _metrics(y_true, y_naive, y_naive),
    }

    for name, m in metrics.items():
        logger.info(
            "%s | rmse=%.4f mae=%.4f dir_acc=%.2f beats_naive=%s",
            name,
            m["rmse"],
            m["mae"],
            m["directional_accuracy"],
            m["beats_naive"],
        )
    return WalkForwardResult(folds=folds, metrics=metrics)
