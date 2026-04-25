"""
BeTiSe Feature Suite
====================

Comprehensive dataset covering all feature types layered on top of base
processes. Organised into six phases so each combination can be loaded and
studied independently.

Phase breakdown:
  1 — Pure base processes      : 15 bases × 3 lengths × 22 series       =   990
  2 — Base + deterministic trend:  8 bases × 5 trends × 3 lengths × 9  = 1,080
  3 — Base + anomaly           :  6 bases × 3 types × 3 lengths × 11   =   594
  4 — Base + structural break  :  6 bases × 3 types × 3 lengths × 11   =   594
  5 — Base + seasonality       :  4 bases × 2 types × 3 lengths × 11   =   264
  6 — Stochastic + volatility  :  5 bases × 4 types × 3 lengths × 11   =   660
  ─────────────────────────────────────────────────────────────────────
  Total                                                                  ≈ 4,182

Output layout:
  generated-dataset/feature_suite/
    phase1_pure/<base>/short.parquet  medium.parquet  long.parquet
    phase2_trend/<trend>/<base>/...
    phase3_anomaly/<type>/<base>/...
    phase4_break/<type>/<base>/...
    phase5_seasonality/<type>/<base>/...
    phase6_volatility/<type>/<base>/...

Run:
    python examples/03_feature_suite.py
"""

import os
from betise import run
from betise.config import load_config

OUTPUT_ROOT = "generated-dataset/feature_suite"
RANDOM_SEED = 42

LENGTH_BUCKETS = [
    ("short",  [100, 200]),
    ("medium", [300, 600]),
    ("long",   [700, 1500]),
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


def _run(base, feature_overrides, length_range, n, subfolder, filename):
    features = {**ALL_FEATURES_OFF, **feature_overrides}
    cfg = load_config(
        dataset={
            "base_series":  base,
            "num_series":   n,
            "length_range": length_range,
            "random_seed":  RANDOM_SEED,
            "output_dir":   os.path.join(OUTPUT_ROOT, subfolder),
            "output_name":  filename,
            "features":     features,
        }
    )
    run(cfg)


total = 0

# ── Phase 1: Pure base processes ─────────────────────────────────────────────
print("=== Phase 1: Pure base processes ===")
N = 22
BASE_TYPES = [
    "ar", "ma", "arma", "white_noise",
    "random_walk", "random_walk_drift", "ari", "ima", "arima",
    "sarma", "sarima",
    "arch", "garch", "egarch", "aparch",
]
for base in BASE_TYPES:
    for bucket_label, length_range in LENGTH_BUCKETS:
        print(f"  {base:20s} | {bucket_label}")
        _run(
            base=base,
            feature_overrides={},
            length_range=length_range,
            n=N,
            subfolder=f"phase1_pure/{base}",
            filename=f"{bucket_label}.parquet",
        )
        total += N

# ── Phase 2: Base + deterministic trend ──────────────────────────────────────
print("\n=== Phase 2: Base + deterministic trend ===")
N = 9
TREND_BASES = ["ar", "ma", "arma", "white_noise", "random_walk", "ari", "ima", "arima"]
TRENDS = [
    ("linear_up",    {"linear_trend":      {"enabled": True, "direction": "upward"}}),
    ("linear_down",  {"linear_trend":      {"enabled": True, "direction": "downward"}}),
    ("quadratic",    {"quadratic_trend":   {"enabled": True, "direction": "upward"}}),
    ("cubic",        {"cubic_trend":       {"enabled": True, "direction": "upward"}}),
    ("exponential",  {"exponential_trend": {"enabled": True, "direction": "upward"}}),
]
for base in TREND_BASES:
    for trend_name, feat in TRENDS:
        for bucket_label, length_range in LENGTH_BUCKETS:
            print(f"  {base:20s} + {trend_name:14s} | {bucket_label}")
            _run(
                base=base,
                feature_overrides=feat,
                length_range=length_range,
                n=N,
                subfolder=f"phase2_trend/{trend_name}/{base}",
                filename=f"{bucket_label}.parquet",
            )
            total += N

# ── Phase 3: Base + anomaly ───────────────────────────────────────────────────
print("\n=== Phase 3: Base + anomaly ===")
N = 11
ANOMALY_BASES = ["ar", "ma", "arma", "white_noise", "random_walk", "arima"]
ANOMALIES = [
    ("point",      {"point_anomaly":      {"enabled": True, "mode": "single", "location": "random"}}),
    ("collective", {"collective_anomaly": {"enabled": True, "mode": "single", "location": "random"}}),
    ("contextual", {"contextual_anomaly": {"enabled": True, "mode": "single", "location": "random"}}),
]
for base in ANOMALY_BASES:
    for anomaly_name, feat in ANOMALIES:
        for bucket_label, length_range in LENGTH_BUCKETS:
            print(f"  {base:20s} + anomaly:{anomaly_name:12s} | {bucket_label}")
            _run(
                base=base,
                feature_overrides=feat,
                length_range=length_range,
                n=N,
                subfolder=f"phase3_anomaly/{anomaly_name}/{base}",
                filename=f"{bucket_label}.parquet",
            )
            total += N

# ── Phase 4: Base + structural break ─────────────────────────────────────────
print("\n=== Phase 4: Base + structural break ===")
N = 11
BREAK_BASES = ["ar", "ma", "arma", "white_noise", "random_walk", "arima"]
BREAKS = [
    ("mean_shift",     {"mean_shift":     {"enabled": True, "mode": "single", "direction": "up"}}),
    ("variance_shift", {"variance_shift": {"enabled": True, "mode": "single", "direction": "up"}}),
    ("trend_shift",    {"trend_shift":    {"enabled": True, "mode": "single", "change_type": "direction_change"}}),
]
for base in BREAK_BASES:
    for break_name, feat in BREAKS:
        for bucket_label, length_range in LENGTH_BUCKETS:
            print(f"  {base:20s} + break:{break_name:16s} | {bucket_label}")
            _run(
                base=base,
                feature_overrides=feat,
                length_range=length_range,
                n=N,
                subfolder=f"phase4_break/{break_name}/{base}",
                filename=f"{bucket_label}.parquet",
            )
            total += N

# ── Phase 5: Base + seasonality ──────────────────────────────────────────────
print("\n=== Phase 5: Base + seasonality ===")
N = 11
SEASONAL_BASES = ["ar", "ma", "arma", "white_noise"]
SEASONAL_FEATURES = [
    ("single",   {"single_seasonality":   {"enabled": True}}),
    ("multiple", {"multiple_seasonality": {"enabled": True, "num_components": 2}}),
]
for base in SEASONAL_BASES:
    for seasonal_name, feat in SEASONAL_FEATURES:
        for bucket_label, length_range in LENGTH_BUCKETS:
            print(f"  {base:20s} + seasonality:{seasonal_name:8s} | {bucket_label}")
            _run(
                base=base,
                feature_overrides=feat,
                length_range=length_range,
                n=N,
                subfolder=f"phase5_seasonality/{seasonal_name}/{base}",
                filename=f"{bucket_label}.parquet",
            )
            total += N

# ── Phase 6: Stochastic base + volatility overlay ────────────────────────────
print("\n=== Phase 6: Stochastic + volatility overlay ===")
N = 11
STOCHASTIC_BASES = ["ar", "ma", "arma", "random_walk", "arima"]
VOL_FEATURES = [
    ("arch",   {"arch":   {"enabled": True}}),
    ("garch",  {"garch":  {"enabled": True}}),
    ("egarch", {"egarch": {"enabled": True}}),
    ("aparch", {"aparch": {"enabled": True}}),
]
for base in STOCHASTIC_BASES:
    for vol_name, feat in VOL_FEATURES:
        for bucket_label, length_range in LENGTH_BUCKETS:
            print(f"  {base:20s} + vol:{vol_name:8s} | {bucket_label}")
            _run(
                base=base,
                feature_overrides=feat,
                length_range=length_range,
                n=N,
                subfolder=f"phase6_volatility/{vol_name}/{base}",
                filename=f"{bucket_label}.parquet",
            )
            total += N

print(f"\nDone. Total series generated: {total:,}")
print(f"Output: {OUTPUT_ROOT}/")
print("Tip: load any file with pd.read_parquet() — see 06_load_and_use.py")
