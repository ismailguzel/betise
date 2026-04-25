"""
BeTiSe Large-Scale Pretraining Dataset
=======================================

Generates a large, fixed-length dataset suitable for pretraining foundation
models or running large-scale statistical experiments.

Default configuration:
  15 base types × 5,000 series × length = 1,000 = 75,000 series total

Primary category breakdown:
  stationary  : ar, ma, arma, white_noise             (4 types × 5k = 20,000)
  stochastic  : random_walk, random_walk_drift,
                ari, ima, arima                        (5 types × 5k = 25,000)
  seasonality : sarma, sarima                          (2 types × 5k = 10,000)
  volatility  : arch, garch, egarch, aparch            (4 types × 5k = 20,000)
  ──────────────────────────────────────────────────────────────────────
  Total                                                           75,000

Generation is chunked to keep memory usage constant regardless of total size.
Each chunk is written as a separate parquet file. Adjust the constants below
to scale up or down.

Output layout:
  generated-dataset/pretraining/
    ar/
      chunk_0000.parquet   (CHUNK_SIZE series)
      chunk_0001.parquet
      ...
    ma/  arma/  ...

Scaling guide:
  SERIES_TOTAL = 5_000   → 75k series total  (default)
  SERIES_TOTAL = 20_000  → 300k series total
  CHUNK_SIZE             → tune to fit available RAM (500 is conservative)

Run:
    python examples/04_pretraining_dataset.py
"""

import os
from betise import run
from betise.config import load_config

OUTPUT_ROOT  = "generated-dataset/pretraining"
SERIES_TOTAL = 5_000    # series per base type
FIXED_LENGTH = 1_000    # all series have the same length
CHUNK_SIZE   = 500      # series per parquet file (tune to available RAM)
RANDOM_SEED  = 42

BASE_TYPES = [
    # stationary
    "ar", "ma", "arma", "white_noise",
    # stochastic
    "random_walk", "random_walk_drift", "ari", "ima", "arima",
    # seasonal (base)
    "sarma", "sarima",
    # volatility (base)
    "arch", "garch", "egarch", "aparch",
]

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

assert SERIES_TOTAL % CHUNK_SIZE == 0, "SERIES_TOTAL must be divisible by CHUNK_SIZE"
num_chunks  = SERIES_TOTAL // CHUNK_SIZE
grand_total = 0

for base in BASE_TYPES:
    out_dir = os.path.join(OUTPUT_ROOT, base)
    print(f"\n[{base}] {num_chunks} chunks × {CHUNK_SIZE} series = {SERIES_TOTAL:,} total")

    for chunk_idx in range(num_chunks):
        # Each chunk gets a different seed for diversity while staying reproducible
        seed     = RANDOM_SEED + chunk_idx
        out_name = f"chunk_{chunk_idx:04d}.parquet"

        cfg = load_config(
            dataset={
                "base_series":  base,
                "num_series":   CHUNK_SIZE,
                "length_range": [FIXED_LENGTH, FIXED_LENGTH],
                "random_seed":  seed,
                "output_dir":   out_dir,
                "output_name":  out_name,
                "features":     ALL_FEATURES_OFF,
            }
        )
        run(cfg)
        grand_total += CHUNK_SIZE
        print(f"  chunk {chunk_idx + 1:>3}/{num_chunks} → {out_name}")

print(f"\nDone. Grand total: {grand_total:,} series")
print(f"Output: {OUTPUT_ROOT}/")
print("Tip: load any file with pd.read_parquet() — see 06_load_and_use.py")
