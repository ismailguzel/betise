"""Smoke tests for base series generation.

Each test verifies that:
- The generator runs without error
- The returned series has the expected length
- Values are finite (no NaN / Inf)
"""
import numpy as np
import pytest

from betise import generate_dataframe, load_config

LENGTH = 200
SEED = 42


def _gen(base: str) -> "pd.DataFrame":
    cfg = load_config(dataset={
        "base_series": base,
        "num_series": 1,
        "length_range": [LENGTH, LENGTH],
        "random_seed": SEED,
    })
    df, _ = generate_dataframe(cfg)
    return df


def _check(df, base: str):
    assert len(df) == LENGTH, f"{base}: expected {LENGTH} rows, got {len(df)}"
    assert df["data"].notna().all(), f"{base}: contains NaN"
    assert np.isfinite(df["data"].values).all(), f"{base}: contains Inf"
    assert (df["base_series"] == base).all(), f"{base}: wrong base_series column"


# ── Stationary ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("base", ["ar", "ma", "arma", "white_noise"])
def test_stationary(base):
    _check(_gen(base), base)


# ── Stochastic trend ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("base", ["random_walk", "random_walk_drift", "ari", "ima", "arima"])
def test_stochastic(base):
    _check(_gen(base), base)


# ── Seasonal ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("base", ["sarma", "sarima"])
def test_seasonal(base):
    _check(_gen(base), base)


# ── Volatility ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("base", ["arch", "garch", "egarch", "aparch"])
def test_volatility(base):
    _check(_gen(base), base)


# ── Multiple series ───────────────────────────────────────────────────────────
def test_multiple_series():
    cfg = load_config(dataset={
        "base_series": "ar",
        "num_series": 5,
        "length_range": [LENGTH, LENGTH],
        "random_seed": SEED,
    })
    df, _ = generate_dataframe(cfg)
    assert df["series_id"].nunique() == 5
    assert len(df) == 5 * LENGTH


# ── Variable length ───────────────────────────────────────────────────────────
def test_variable_length():
    cfg = load_config(dataset={
        "base_series": "ar",
        "num_series": 10,
        "length_range": [100, 300],
        "random_seed": SEED,
    })
    df, _ = generate_dataframe(cfg)
    lengths = df.groupby("series_id").size()
    assert (lengths >= 100).all()
    assert (lengths <= 300).all()


# ── Reproducibility ───────────────────────────────────────────────────────────
def test_reproducibility():
    cfg = load_config(dataset={
        "base_series": "ar",
        "num_series": 1,
        "length_range": [LENGTH, LENGTH],
        "random_seed": SEED,
    })
    df1, _ = generate_dataframe(cfg)
    df2, _ = generate_dataframe(cfg)
    np.testing.assert_array_equal(df1["data"].values, df2["data"].values)
