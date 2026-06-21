"""Tests for HTTP retry behavior and walk-forward validation mechanics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_cpi import config
from macro_cpi.http import FetchError, get_json
from macro_cpi.models.walk_forward import walk_forward


def test_get_json_retries_then_raises(monkeypatch):
    calls = {"n": 0}

    def boom(*a, **k):  # noqa: ANN002, ANN003
        calls["n"] += 1
        raise __import__("requests").RequestException("down")

    monkeypatch.setattr("macro_cpi.http.requests.get", boom)
    monkeypatch.setattr(config, "HTTP_BACKOFF_BASE_SECS", 0.0)
    monkeypatch.setattr(config, "HTTP_MAX_RETRIES", 3)
    with pytest.raises(FetchError):
        get_json("https://example.test")
    assert calls["n"] == 3


def _synthetic_panel(n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2005-01-31", periods=n, freq="ME")
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


def test_walk_forward_runs_and_reports_skill():
    model = config.ModelConfig(min_train_months=120)
    panel = _synthetic_panel()
    result = walk_forward(panel, model)
    assert len(result.folds) == len(panel) - 120
    for name in ("ridge", "lgbm", "naive"):
        assert "rmse" in result.metrics[name]
    # On a learnable signal, ridge should beat naive.
    assert result.metrics["ridge"]["beats_naive"]


def test_walk_forward_errors_when_insufficient_history():
    model = config.ModelConfig(min_train_months=500)
    with pytest.raises(ValueError):
        walk_forward(_synthetic_panel(160), model)
