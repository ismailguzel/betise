"""
BeTiSe Benchmark Dataset
=========================

Generates a balanced benchmark covering all 15 base process types across three
series-length buckets. No feature overlays — pure base processes only. Use this
as a reproducible baseline for comparing models or algorithms across process
types and lengths.

Output layout:
  generated-dataset/benchmark/
    ar/
      ar_short.parquet    (11 series, length 100–200)
      ar_medium.parquet   (11 series, length 300–600)
      ar_long.parquet     (11 series, length 700–1500)
    ma/  ...
    (one subfolder per base type)

Total: 15 types × 3 buckets × 11 series = 495 series

Run:
    python examples/02_benchmark_dataset.py
"""

import os
from betise import run
from betise.config import load_config

OUTPUT_ROOT = "generated-dataset/benchmark"
RANDOM_SEED = 42

# (label, length_range, num_series)
LENGTH_BUCKETS = [
    ("short",  [100, 200],   11),
    ("medium", [300, 600],   11),
    ("long",   [700, 1500],  11),
]

BASE_TYPES = [
    "ar",
    "ma",
    "arma",
    "white_noise",
    "random_walk",
    "random_walk_drift",
    "ari",
    "ima",
    "arima",
    "sarma",
    "sarima",
    "arch",
    "garch",
    "egarch",
    "aparch",
]

ALL_FEATURES_OFF = {
    "linear_trend":       {"enabled": False},
    "quadratic_trend":    {"enabled": False},
    "cubic_trend":        {"enabled": False},
    "exponential_trend":  {"enabled": False},
    "arch":               {"enabled": False},
    "garch":              {"enabled": False},
    "egarch":             {"enabled": False},
    "aparch":             {"enabled": False},
    "single_seasonality": {"enabled": False},
    "multiple_seasonality": {"enabled": False},
    "sarma":              {"enabled": False},
    "sarima":             {"enabled": False},
    "mean_shift":         {"enabled": False},
    "variance_shift":     {"enabled": False},
    "trend_shift":        {"enabled": False},
    "point_anomaly":      {"enabled": False},
    "collective_anomaly": {"enabled": False},
    "contextual_anomaly": {"enabled": False},
}

total = 0
for base in BASE_TYPES:
    for bucket_label, length_range, n in LENGTH_BUCKETS:
        out_name = f"{base}_{bucket_label}.parquet"
        cfg = load_config(
            dataset={
                "base_series":   base,
                "num_series":    n,
                "length_range":  length_range,
                "random_seed":   RANDOM_SEED,
                "output_dir":    os.path.join(OUTPUT_ROOT, base),
                "output_name":   out_name,
                "features":      ALL_FEATURES_OFF,
            }
        )
        print(f"Generating {base:20s} | {bucket_label:6s} | n={n} | lengths={length_range}")
        run(cfg)
        total += n

print(f"\nDone. Total series generated: {total}")
print(f"Output: {OUTPUT_ROOT}/")
print("Tip: load any file with pd.read_parquet() — see 06_load_and_use.py")
