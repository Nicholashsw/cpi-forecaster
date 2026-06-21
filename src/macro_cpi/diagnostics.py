"""Diagnostics. The numbers that separate a real signal study from a single headline RMSE.

What's here:
- Feature importance (LightGBM gain) on a refit-on-all-data model
- Rolling out-of-sample skill vs naive (does the edge persist or come from one regime?)
- Diebold-Mariano test for statistical significance of forecast improvement vs naive
- Residual autocorrelation (is information left on the table?)
- Per-decade performance breakdown (stability over time)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats

from macro_cpi import config
from macro_cpi.features.build import feature_columns


@dataclass
class DMTestResult:
    """Diebold-Mariano test outcome comparing a candidate forecast to a benchmark."""

    statistic: float
    p_value: float
    favors_candidate: bool


def feature_importance(panel: pd.DataFrame, model: config.ModelConfig) -> pd.DataFrame:
    """Fit LightGBM on the full panel and return a sorted gain-based importance table."""
    feat_cols = feature_columns(panel)
    x = panel[feat_cols].copy()
    col_mean = x.mean(numeric_only=True)
    x = x.fillna(col_mean).fillna(0.0)
    y = panel["target"].to_numpy()
    lgbm = LGBMRegressor(**model.lgbm_params).fit(x, y)
    imp = pd.DataFrame(
        {"feature": feat_cols, "gain": lgbm.booster_.feature_importance(importance_type="gain")}
    )
    return imp.sort_values("gain", ascending=False).reset_index(drop=True)


def rolling_skill(folds: pd.DataFrame, window: int = 36) -> pd.DataFrame:
    """Rolling out-of-sample RMSE skill vs naive: positive means the model beats naive."""
    df = folds.copy().reset_index(drop=True)
    err_model = (df["pred_lgbm"] - df["y_true"]) ** 2
    err_naive = (df["pred_naive"] - df["y_true"]) ** 2
    roll_model = err_model.rolling(window).mean() ** 0.5
    roll_naive = err_naive.rolling(window).mean() ** 0.5
    df["rolling_rmse_lgbm"] = roll_model
    df["rolling_rmse_naive"] = roll_naive
    df["rolling_skill_vs_naive"] = 1.0 - roll_model / roll_naive
    return df[["date", "rolling_rmse_lgbm", "rolling_rmse_naive", "rolling_skill_vs_naive"]]


def diebold_mariano(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, h: int = 1
) -> DMTestResult:
    """Diebold-Mariano test (squared-error loss). H0: equal accuracy. Positive stat favors A."""
    e_a = (pred_a - y_true) ** 2
    e_b = (pred_b - y_true) ** 2
    d = e_b - e_a  # positive => A has lower squared error
    n = len(d)
    mean_d = float(np.mean(d))
    # Newey-West style variance with h-1 lags; h=1 -> simple variance.
    var_d = float(np.var(d, ddof=1))
    for lag in range(1, h):
        cov = float(np.cov(d[:-lag], d[lag:], ddof=1)[0, 1])
        var_d += 2 * cov
    if var_d <= 0 or n == 0:
        return DMTestResult(float("nan"), float("nan"), False)
    dm_stat = mean_d / np.sqrt(var_d / n)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    return DMTestResult(statistic=float(dm_stat), p_value=float(p), favors_candidate=dm_stat > 0)


def residual_autocorr(folds: pd.DataFrame, lags: int = 6) -> pd.DataFrame:
    """Residual autocorrelations for each model up to a given lag count."""
    out = []
    for col in ("pred_ridge", "pred_lgbm", "pred_naive"):
        resid = (folds[col] - folds["y_true"]).to_numpy()
        for lag in range(1, lags + 1):
            r = float(np.corrcoef(resid[:-lag], resid[lag:])[0, 1])
            out.append({"model": col.replace("pred_", ""), "lag": lag, "autocorr": r})
    return pd.DataFrame(out)


def per_decade_metrics(folds: pd.DataFrame) -> pd.DataFrame:
    """RMSE by decade for each model + naive, to expose regime-dependence of the edge."""
    df = folds.copy()
    df["decade"] = (df["date"].dt.year // 10) * 10
    rows = []
    for decade, grp in df.groupby("decade"):
        for col in ("pred_ridge", "pred_lgbm", "pred_naive"):
            err = grp[col] - grp["y_true"]
            rmse = float(np.sqrt(np.mean(err**2)))
            rows.append(
                {"decade": int(decade), "model": col.replace("pred_", ""),
                 "rmse": rmse, "n": len(grp)}
            )
    return pd.DataFrame(rows)
