"""End-to-end pipeline tests: config → generate → save → load."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from betise import generate_dataframe, load_config, run

LENGTH = 150
SEED = 1


# ── load_config ───────────────────────────────────────────────────────────────
def test_load_config_defaults():
    cfg = load_config()
    assert "params" in cfg
    assert "dataset" in cfg


def test_load_config_override():
    cfg = load_config(dataset={"base_series": "ma", "num_series": 3})
    assert cfg["dataset"]["base_series"] == "ma"
    assert cfg["dataset"]["num_series"] == 3


def test_load_config_deep_merge():
    cfg = load_config(dataset={
        "features": {"linear_trend": {"enabled": True, "direction": "upward"}}
    })
    assert cfg["dataset"]["features"]["linear_trend"]["enabled"] is True
    # Other features should still be present from default
    assert "point_anomaly" in cfg["dataset"]["features"]


# ── generate_dataframe ────────────────────────────────────────────────────────
def test_generate_dataframe_shape():
    cfg = load_config(dataset={
        "base_series": "ar",
        "num_series": 3,
        "length_range": [LENGTH, LENGTH],
        "random_seed": SEED,
    })
    df, ctx = generate_dataframe(cfg)
    assert df["series_id"].nunique() == 3
    assert len(df) == 3 * LENGTH
    assert "data" in df.columns
    assert "time" in df.columns
    assert "series_id" in df.columns


def test_generate_dataframe_metadata_columns():
    cfg = load_config(dataset={
        "base_series": "ar",
        "num_series": 1,
        "length_range": [LENGTH, LENGTH],
        "random_seed": SEED,
    })
    df, _ = generate_dataframe(cfg)
    expected_cols = {"series_id", "time", "data", "primary_category",
                     "sub_category", "base_series", "is_stationary"}
    assert expected_cols.issubset(df.columns)


# ── run (save to parquet) ─────────────────────────────────────────────────────
def test_run_saves_parquet():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_config(dataset={
            "base_series": "ar",
            "num_series": 2,
            "length_range": [LENGTH, LENGTH],
            "random_seed": SEED,
            "output_dir": tmpdir,
            "output_name": "test_output.parquet",
        })
        run(cfg)
        out_path = Path(tmpdir) / "test_output.parquet"
        assert out_path.exists(), "Parquet file was not created"


def test_run_parquet_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_config(dataset={
            "base_series": "ar",
            "num_series": 3,
            "length_range": [LENGTH, LENGTH],
            "random_seed": SEED,
            "output_dir": tmpdir,
            "output_name": "round_trip.parquet",
        })
        df_orig, _ = generate_dataframe(cfg)
        run(cfg)

        df_loaded = pd.read_parquet(Path(tmpdir) / "round_trip.parquet")
        assert df_loaded["series_id"].nunique() == 3
        assert len(df_loaded) == 3 * LENGTH
        assert np.isfinite(df_loaded["data"].values).all()
