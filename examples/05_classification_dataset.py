"""
BeTiSe Classification Dataset
==============================

Generates a balanced, labeled dataset for time series classification tasks.
Each series is assigned a primary_category label (the class) and a sub_category
label (the fine-grained type within that class).

Class definitions are read from configs/classification_config.json so you can
adjust counts and configurations without modifying this script.

Default configuration (14,000 series total, fixed length = 1,000):

  Class             Sub-types                              Series
  ─────────────────────────────────────────────────────────────────
  stationary        ar, ma, arma, white_noise              500 × 4 = 2,000
  stochastic        rw, rw_drift, ari, ima, arima          400 × 5 = 2,000
  trend             linear↑, linear↓, quadratic,
                    cubic, exponential                     400 × 5 = 2,000
  seasonality       sarma, sarima, single, multiple        500 × 4 = 2,000
  volatility        arch, garch, egarch, aparch            500 × 4 = 2,000
  anomaly           point, collective, contextual      667/667/666 = 2,000
  structural_break  mean_shift, variance_shift,
                    trend_shift                        667/667/666 = 2,000
  ─────────────────────────────────────────────────────────────────
  Total                                                         14,000

Design notes:
  - Fixed length ensures X is a proper (n_series, length) matrix after pivot.
  - Series IDs are globally unique and randomly shuffled across classes to
    prevent any ordering bias during model training.
  - Object-dtype metadata columns are cast to str to ensure PyArrow parquet
    compatibility when concatenating series from different base types.

Output:
  generated-dataset/classification/
    stationary/           <- one parquet per sub-type (intermediate files)
    stochastic/
    trend/
    seasonality/
    volatility/
    anomaly/
    structural_break/
    betise_classification.parquet  <- merged, shuffled, ready for ML

Run:
    python examples/05_classification_dataset.py

Load the result:
    python examples/06_load_and_use.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from betise import generate_dataframe
from betise.config import load_config

# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG_FILE = Path(__file__).parent / "configs" / "classification_config.json"
with open(CONFIG_FILE) as f:
    CLASS_CFG = json.load(f)

OUTPUT_ROOT  = CLASS_CFG["output_root"]
OUTPUT_NAME  = CLASS_CFG["output_name"]
FIXED_LENGTH = CLASS_CFG["fixed_length"]
RANDOM_SEED  = CLASS_CFG["random_seed"]
CLASSES      = CLASS_CFG["classes"]

ALL_FEATURES_OFF = {
    "linear_trend":         {"enabled": False},
    "quadratic_trend":      {"enabled": False},
    "cubic_trend":          {"enabled": False},
    "exponential_trend":    {"enabled": False},
    "arch":                 {"enabled": False},
    "garch":                {"enabled": False},
    "egarch":               {"enabled": False},
    "aparch":               {"enabled": False},
    "single_seasonality":   {"enabled": False},
    "multiple_seasonality": {"enabled": False},
    "sarma":                {"enabled": False},
    "sarima":               {"enabled": False},
    "mean_shift":           {"enabled": False},
    "variance_shift":       {"enabled": False},
    "trend_shift":          {"enabled": False},
    "point_anomaly":        {"enabled": False},
    "collective_anomaly":   {"enabled": False},
    "contextual_anomaly":   {"enabled": False},
}

# ── Generation ────────────────────────────────────────────────────────────────

all_frames    = []
series_offset = 0
class_counts  = {}

for class_name, scenarios in CLASSES.items():
    print(f"\n[{class_name}]")
    class_total = 0

    for i, scenario in enumerate(scenarios):
        base  = scenario["base_series"]
        n     = scenario["n"]
        feats = {**ALL_FEATURES_OFF, **scenario["features"]}
        seed  = RANDOM_SEED + i  # vary seed per sub-type for diversity

        # Build sub-label from the first enabled feature key + direction (if present)
        enabled = [k for k, v in scenario["features"].items() if v.get("enabled")]
        if enabled:
            key       = enabled[0]
            direction = scenario["features"][key].get("direction")
            sublabel  = f"{key}_{direction}" if direction else key
        else:
            sublabel = base

        cfg = load_config(
            dataset={
                "base_series":  base,
                "num_series":   n,
                "length_range": [FIXED_LENGTH, FIXED_LENGTH],
                "random_seed":  seed,
                "output_dir":   os.path.join(OUTPUT_ROOT, class_name),
                "output_name":  f"{sublabel}.parquet",
                "features":     feats,
            }
        )

        df, _ = generate_dataframe(cfg)

        # Offset series_id globally so IDs are unique across all classes
        df["series_id"] = df["series_id"] + series_offset
        series_offset   = df["series_id"].max()

        all_frames.append(df)
        class_total += df["series_id"].nunique()
        print(f"  {sublabel:30s}  base={base:20s}  n={n}")

    class_counts[class_name] = class_total

# ── Merge & shuffle ───────────────────────────────────────────────────────────

print("\nMerging and shuffling...")
combined = pd.concat(all_frames, ignore_index=True)

# Shuffle at series level (not row level) to preserve time-series continuity
rng        = np.random.default_rng(RANDOM_SEED)
unique_ids = combined["series_id"].unique()
shuffled   = rng.permutation(unique_ids)
id_map     = {old: new + 1 for new, old in enumerate(shuffled)}
combined["series_id"] = combined["series_id"].map(id_map)
combined = combined.sort_values(["series_id", "time"]).reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────────────

out_path = Path(OUTPUT_ROOT) / OUTPUT_NAME
out_path.parent.mkdir(parents=True, exist_ok=True)

# Cast object columns to str for consistent PyArrow schema across base types
for col in combined.select_dtypes(include=["object", "string"]).columns:
    combined[col] = combined[col].astype(str)

combined.to_parquet(out_path, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────

total = combined["series_id"].nunique()
print(f"\n{'─' * 55}")
print(f"  {'Class':<20} {'Series':>8}  {'%':>6}")
print(f"{'─' * 55}")
for cls, count in class_counts.items():
    print(f"  {cls:<20} {count:>8,}  {count / total * 100:>5.1f}%")
print(f"{'─' * 55}")
print(f"  {'TOTAL':<20} {total:>8,}  100.0%")
print(f"\nSaved → {out_path}")
print(f"Class label column : primary_category")
print(f"Sub-type column    : sub_category")
print(f"\nNext step: python examples/06_load_and_use.py")
