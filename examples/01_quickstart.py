"""
BeTiSe Quickstart
=================

Three common usage patterns in increasing complexity:

  1. Single series, in-memory  — generate_dataframe(), no file written
  2. Single series, to disk    — run(), parquet saved automatically
  3. Feature combination       — trend + anomaly on the same series

Run:
    python examples/01_quickstart.py
"""

import pandas as pd
from betise import generate_dataframe, run
from betise.config import load_config


# ── 1. Single series, in-memory ───────────────────────────────────────────────
#
# generate_dataframe() returns a (DataFrame, context) tuple — nothing is
# written to disk. Useful for exploration, unit tests, or embedding in a
# larger pipeline.

cfg = load_config(
    dataset={
        "base_series":  "arma",
        "num_series":   1,
        "length_range": [300, 300],
        "random_seed":  42,
        "features": {
            "linear_trend": {"enabled": False},
        },
    }
)

df, ctx = generate_dataframe(cfg)

print("=== 1. In-memory single series ===")
print(f"  Rows       : {len(df)}")
print(f"  Columns    : {list(df.columns)}")
print(f"  Base type  : {ctx['metadata']['base_series']}")
print(f"  Series ID  : {df['series_id'].unique()[0]}")
print()


# ── 2. Single series, saved to disk ──────────────────────────────────────────
#
# run() is the same pipeline as generate_dataframe() but also persists the
# result to a parquet file at output_dir/output_name.

cfg = load_config(
    dataset={
        "base_series":  "ar",
        "num_series":   3,
        "length_range": [200, 400],
        "random_seed":  7,
        "output_dir":   "generated-dataset/quickstart",
        "output_name":  "ar_small.parquet",
        "features": {
            "linear_trend": {"enabled": False},
        },
    }
)

run(cfg)

result = pd.read_parquet("generated-dataset/quickstart/ar_small.parquet")
print("=== 2. Saved to disk ===")
print(f"  Parquet rows   : {len(result)}")
print(f"  Unique series  : {result['series_id'].nunique()}")
print()


# ── 3. Feature combination: trend + point anomaly ─────────────────────────────
#
# Multiple features can be active at the same time. They are applied in a
# fixed order: volatility → seasonality → trend → structural break → anomaly.
# Here we add a linear upward trend and a single point anomaly to an AR base.

cfg = load_config(
    dataset={
        "base_series":  "ar",
        "num_series":   1,
        "length_range": [500, 500],
        "random_seed":  123,
        "output_dir":   "generated-dataset/quickstart",
        "output_name":  "ar_trend_anomaly.parquet",
        "features": {
            "linear_trend": {
                "enabled":   True,
                "direction": "upward",
            },
            "point_anomaly": {
                "enabled":   True,
                "mode":      "single",
                "location":  "random",
            },
        },
    }
)

df, ctx = generate_dataframe(cfg)

print("=== 3. Feature combination: linear trend + point anomaly ===")
print(f"  Series length    : {len(df)}")
print(f"  Primary category : {df['primary_category'].unique()[0]}")
print(f"  Sub category     : {df['sub_category'].unique()[0]}")
print(f"  Has anomaly flag : {'anomaly_location' in df.columns}")
print()

print("Done. Output written to generated-dataset/quickstart/")
