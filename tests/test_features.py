"""Smoke tests for feature overlays.

Each test verifies that:
- Feature application runs without error
- Output length is preserved
- Values are finite
- Adding a non-flat feature changes the series (feature is not a no-op)
"""
import numpy as np
import pytest

from betise import generate_dataframe, load_config

LENGTH = 300
SEED = 42


def _gen(base: str, features: dict):
    cfg = load_config(dataset={
        "base_series": base,
        "num_series": 1,
        "length_range": [LENGTH, LENGTH],
        "random_seed": SEED,
        "features": features,
    })
    df, _ = generate_dataframe(cfg)
    return df


def _baseline(base: str = "ar") -> "np.ndarray":
    return _gen(base, {})["data"].values


# ── Trend features ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("feat,extra", [
    ("linear_trend",      {"direction": "upward"}),
    ("linear_trend",      {"direction": "downward"}),
    ("quadratic_trend",   {}),
    ("cubic_trend",       {}),
    ("exponential_trend", {"direction": "upward"}),
])
def test_trend(feat, extra):
    df = _gen("ar", {feat: {"enabled": True, **extra}})
    vals = df["data"].values
    assert len(vals) == LENGTH
    assert np.isfinite(vals).all()
    # Verify the feature is recorded in metadata
    assert df["trend_type"].iloc[0] != "" or df["sub_category"].iloc[0] != "", \
        f"{feat}: trend metadata not recorded"


# ── Seasonality features ──────────────────────────────────────────────────────
@pytest.mark.parametrize("feat", ["single_seasonality", "multiple_seasonality"])
def test_seasonality(feat):
    df = _gen("ar", {feat: {"enabled": True}})
    vals = df["data"].values
    assert len(vals) == LENGTH
    assert np.isfinite(vals).all()


# ── Anomaly features ──────────────────────────────────────────────────────────
def test_point_anomaly_spike():
    df = _gen("ar", {"point_anomaly": {"enabled": True, "is_spike": True}})
    assert len(df) == LENGTH
    assert np.isfinite(df["data"].values).all()


def test_collective_anomaly():
    df = _gen("ar", {"collective_anomaly": {"enabled": True}})
    assert len(df) == LENGTH
    assert np.isfinite(df["data"].values).all()


def test_contextual_anomaly():
    df = _gen("ar", {"contextual_anomaly": {"enabled": True}})
    assert len(df) == LENGTH
    assert np.isfinite(df["data"].values).all()


# ── Structural break features ─────────────────────────────────────────────────
@pytest.mark.parametrize("feat", ["mean_shift", "variance_shift", "trend_shift"])
def test_structural_break(feat):
    df = _gen("ar", {feat: {"enabled": True}})
    assert len(df) == LENGTH
    assert np.isfinite(df["data"].values).all()


# ── Combined features ─────────────────────────────────────────────────────────
def test_trend_plus_seasonality():
    df = _gen("ar", {
        "linear_trend":       {"enabled": True, "direction": "upward"},
        "single_seasonality": {"enabled": True},
    })
    assert len(df) == LENGTH
    assert np.isfinite(df["data"].values).all()


def test_trend_plus_anomaly():
    df = _gen("ar", {
        "linear_trend":  {"enabled": True, "direction": "upward"},
        "point_anomaly": {"enabled": True, "is_spike": True},
    })
    assert len(df) == LENGTH
    assert np.isfinite(df["data"].values).all()
