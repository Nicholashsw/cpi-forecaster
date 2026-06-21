"""Tests for the diagnostics module: importance, rolling skill, DM, residuals, per-decade."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_cpi import config
from macro_cpi.diagnostics import (
    diebold_mariano,
    feature_importance,
    per_decade_metrics,
    residual_autocorr,
    rolling_skill,
)


def _synthetic_panel(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-31", periods=n, freq="ME")
    x = rng.normal(size=n)
    target = 0.5 * x + rng.normal(scale=0.1, size=n)
    return pd.DataFrame(
        {
            "date": dates,
            "anchor_date": dates - pd.offsets.MonthEnd(1),
            "target": target,
            "feat_x": x,
            "target_lag1": pd.Series(target).shift(1),
        }
    )


def _synthetic_folds(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.date_range("2010-01-31", periods=n, freq="ME")
    y = rng.normal(size=n)
    return pd.DataFrame(
        {
            "date": dates,
            "y_true": y,
            "pred_lgbm": y + rng.normal(scale=0.5, size=n),
            "pred_ridge": y + rng.normal(scale=0.8, size=n),
            "pred_naive": np.concatenate([[0.0], y[:-1]]),
        }
    )


def test_feature_importance_returns_sorted_frame():
    imp = feature_importance(_synthetic_panel(), config.ModelConfig(min_train_months=120))
    assert {"feature", "gain"}.issubset(imp.columns)
    assert imp["gain"].is_monotonic_decreasing


def test_rolling_skill_columns_present():
    out = rolling_skill(_synthetic_folds(), window=24)
    expected = {"rolling_rmse_lgbm", "rolling_rmse_naive", "rolling_skill_vs_naive"}
    assert expected.issubset(out.columns)


def test_diebold_mariano_detects_clear_difference():
    rng = np.random.default_rng(7)
    n = 500
    y = rng.normal(size=n)
    good = y + rng.normal(scale=0.1, size=n)   # near perfect
    bad = y + rng.normal(scale=2.0, size=n)    # noisy
    res = diebold_mariano(y, good, bad)
    assert res.favors_candidate
    assert res.p_value < 0.01


def test_residual_autocorr_shape():
    df = residual_autocorr(_synthetic_folds(), lags=4)
    assert set(df["model"].unique()) == {"ridge", "lgbm", "naive"}
    assert df["lag"].max() == 4


def test_per_decade_metrics():
    df = per_decade_metrics(_synthetic_folds())
    assert "decade" in df.columns
    assert (df["rmse"] >= 0).all()
