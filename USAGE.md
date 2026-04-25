# Usage Guide

## Table of Contents

1. [Base Series](#1-base-series)
2. [Features (Stackable Properties)](#2-features-stackable-properties)
3. [Generating a Single Series](#3-generating-a-single-series)
4. [Multiple Series — Single Characteristic](#4-multiple-series--single-characteristic)
5. [Multiple Characteristics — Combinations](#5-multiple-characteristics--combinations)
6. [Mixed Batch Generation](#6-mixed-batch-generation)
7. [Classification Dataset](#7-classification-dataset)
8. [Output Format](#8-output-format)

---

## 1. Base Series

Every generation starts with a **base series**. The base defines the core statistical process of the series.

| Group | `base_series` value | Description |
|-------|---------------------|-------------|
| **Stationary** | `"ar"` | AutoRegressive — AR(p) process |
| | `"ma"` | Moving Average — MA(q) process |
| | `"arma"` | ARMA(p,q) process |
| | `"white_noise"` | Pure white noise |
| **Stochastic Trend** | `"random_walk"` | Random walk, I(d) |
| | `"random_walk_drift"` | Random walk with drift |
| | `"ari"` | ARI(p,d) — AR with integration |
| | `"ima"` | IMA(d,q) — MA with integration |
| | `"arima"` | ARIMA(p,d,q) |
| **Seasonal** | `"sarma"` | Seasonal ARMA |
| | `"sarima"` | Seasonal ARIMA |
| **Volatility** | `"arch"` | ARCH process |
| | `"garch"` | GARCH(1,1) |
| | `"egarch"` | Exponential GARCH |
| | `"aparch"` | Asymmetric Power ARCH |

> **Rule:** Exactly **one** base is selected per generation run via `dataset.base_series`.

---

## 2. Features (Stackable Properties)

Features are applied **on top of** the selected base series. Multiple features can be active simultaneously.

### Deterministic Trend

| Feature | `feature_name` | Parameters |
|---------|----------------|------------|
| Linear trend | `"linear_trend"` | `direction`: `"upward"` / `"downward"` |
| Quadratic trend | `"quadratic_trend"` | `direction`, `location`: `"beginning"` / `"center"` / `"end"` |
| Cubic trend | `"cubic_trend"` | `direction`, `location` |
| Exponential trend | `"exponential_trend"` | `direction` |

### Seasonality (as a feature)

| Feature | `feature_name` | Parameters |
|---------|----------------|------------|
| Single seasonality | `"single_seasonality"` | — |
| Multiple seasonality | `"multiple_seasonality"` | `num_components`: int |
| SARMA overlay | `"sarma"` | — |
| SARIMA overlay | `"sarima"` | — |

> **Note:** `sarma` and `sarima` can be used as both base and feature. When used as a feature, they **replace** the base series output.

### Volatility (as a feature)

| Feature | `feature_name` |
|---------|----------------|
| ARCH | `"arch"` |
| GARCH | `"garch"` |
| EGARCH | `"egarch"` |
| APARCH | `"aparch"` |

> **Note:** Volatility features also replace the base series output — equivalent to selecting them as the base.

### Anomaly

| Feature | `feature_name` | Parameters |
|---------|----------------|------------|
| Point anomaly | `"point_anomaly"` | `mode`: `"single"` / `"multiple"`, `location`: `"beginning"` / `"middle"` / `"end"` / `"random"` |
| Collective anomaly | `"collective_anomaly"` | `mode`, `location`, `num_anomalies` |
| Contextual anomaly | `"contextual_anomaly"` | `mode`, `location`, `num_anomalies` |

### Structural Break

| Feature | `feature_name` | Parameters |
|---------|----------------|------------|
| Mean shift | `"mean_shift"` | `mode`: `"single"` / `"multiple"`, `direction`: `"up"` / `"down"`, `num_breaks` |
| Variance shift | `"variance_shift"` | `mode`, `direction`, `num_breaks` |
| Trend shift | `"trend_shift"` | `mode`, `direction`, `change_type`: `"direction_change"` / `"magnitude_change"`, `num_breaks` |

---

## 3. Generating a Single Series

Set `num_series=1`. You can also use `TimeSeriesGenerator` directly without the config pipeline.

### Via config (recommended)

```python
from betise import generate_dataframe
from betise.config import load_config

cfg = load_config(dataset={
    "base_series":  "ar",
    "num_series":   1,
    "length_range": [500, 500],   # fixed length: set min == max
    "random_seed":  42,
    "output_dir":   "output",
    "output_name":  "single_series.parquet",
    "features": {
        "linear_trend": {"enabled": False},
        "mean_shift":   {"enabled": False},
        # all other features are already False by default
    },
})

df, ctx = generate_dataframe(cfg)
print(df[["series_id", "time", "data"]].head())
```

### Directly via TimeSeriesGenerator

```python
from betise import TimeSeriesGenerator

ts = TimeSeriesGenerator(length=500)

# Pure AR series
df, info = ts.generate_ar_series(length=500)
print(df["data"].values)
print(info)  # {"ar_order": 2, "ar_coefs": [...]}
```

---

## 4. Multiple Series — Single Characteristic

Increase `num_series` to generate N series of the same type. Each series is generated with independently sampled parameters (reproducible under the same seed).

```python
from betise import run
from betise.config import load_config

# 100 pure AR series, length varies between 200 and 600
cfg = load_config(dataset={
    "base_series":  "ar",
    "num_series":   100,
    "length_range": [200, 600],
    "random_seed":  42,
    "output_dir":   "output/ar",
    "output_name":  "ar_100.parquet",
})
run(cfg)
```

```python
# 50 ARIMA series, fixed length 1000
cfg = load_config(dataset={
    "base_series":  "arima",
    "num_series":   50,
    "length_range": [1000, 1000],
    "random_seed":  42,
    "output_dir":   "output/arima",
    "output_name":  "arima_50.parquet",
})
run(cfg)
```

---

## 5. Multiple Characteristics — Combinations

### Dual combination: AR + linear trend

```python
cfg = load_config(dataset={
    "base_series":  "ar",
    "num_series":   50,
    "length_range": [300, 700],
    "random_seed":  42,
    "output_dir":   "output/combinations",
    "output_name":  "ar_linear_trend.parquet",
    "features": {
        "linear_trend": {"enabled": True, "direction": "upward"},
    },
})
run(cfg)
```

### Triple combination: ARIMA + trend + anomaly

```python
cfg = load_config(dataset={
    "base_series":  "arima",
    "num_series":   50,
    "length_range": [500, 1000],
    "random_seed":  42,
    "output_dir":   "output/combinations",
    "output_name":  "arima_trend_anomaly.parquet",
    "features": {
        "linear_trend":  {"enabled": True, "direction": "upward"},
        "point_anomaly": {"enabled": True, "mode": "single", "location": "random"},
    },
})
run(cfg)
```

### Four features: AR + seasonality + structural break + anomaly

```python
cfg = load_config(dataset={
    "base_series":  "ar",
    "num_series":   30,
    "length_range": [600, 1200],
    "random_seed":  42,
    "output_dir":   "output/combinations",
    "output_name":  "ar_complex.parquet",
    "features": {
        "single_seasonality": {"enabled": True},
        "linear_trend":       {"enabled": True, "direction": "upward"},
        "mean_shift":         {"enabled": True, "mode": "single", "direction": "down"},
        "point_anomaly":      {"enabled": True, "mode": "single", "location": "random"},
    },
})
run(cfg)
```

> **Feature application order** (always applied in this sequence, regardless of config order):
> `volatility → seasonality → trend → structural_break → anomaly`

---

## 6. Mixed Batch Generation

To collect different types and counts into a single file, loop over `generate_dataframe` and concatenate the DataFrames.

### Example: 5 AR + 3 ARIMA + 7 GARCH — single parquet

```python
import pandas as pd
from betise import generate_dataframe
from betise.config import load_config

SCENARIOS = [
    {"base_series": "ar",    "num_series": 5},
    {"base_series": "arima", "num_series": 3},
    {"base_series": "garch", "num_series": 7},
]

frames = []
series_offset = 0

for scenario in SCENARIOS:
    cfg = load_config(dataset={
        **scenario,
        "length_range": [300, 600],
        "random_seed":  42,
    })
    df, _ = generate_dataframe(cfg)

    # Offset series_id to avoid collisions across batches
    df["series_id"] = df["series_id"] + series_offset
    series_offset = df["series_id"].max()

    frames.append(df)

combined = pd.concat(frames, ignore_index=True)
combined.to_parquet("output/mixed_batch.parquet", index=False)
print(f"Total: {combined['series_id'].nunique()} series")
```

### Mixed combinations: single, dual, and triple features

```python
import pandas as pd
from betise import generate_dataframe
from betise.config import load_config

SCENARIOS = [
    # Single: base only
    {
        "label": "ar_pure",
        "dataset": {"base_series": "ar", "num_series": 20},
    },
    # Dual: base + trend
    {
        "label": "ar_linear",
        "dataset": {
            "base_series": "ar",
            "num_series": 15,
            "features": {
                "linear_trend": {"enabled": True, "direction": "upward"},
            },
        },
    },
    # Dual: base + anomaly
    {
        "label": "arima_anomaly",
        "dataset": {
            "base_series": "arima",
            "num_series": 10,
            "features": {
                "point_anomaly": {"enabled": True, "mode": "single", "location": "random"},
            },
        },
    },
    # Triple: base + trend + structural break
    {
        "label": "ar_trend_break",
        "dataset": {
            "base_series": "ar",
            "num_series": 10,
            "features": {
                "linear_trend": {"enabled": True, "direction": "upward"},
                "mean_shift":   {"enabled": True, "mode": "single", "direction": "down"},
            },
        },
    },
    # Triple: base + seasonality + anomaly
    {
        "label": "arima_seasonal_anomaly",
        "dataset": {
            "base_series": "arima",
            "num_series": 10,
            "features": {
                "single_seasonality": {"enabled": True},
                "point_anomaly":      {"enabled": True, "mode": "single", "location": "random"},
            },
        },
    },
]

frames = []
series_offset = 0

for scenario in SCENARIOS:
    cfg = load_config(dataset={
        "length_range": [300, 700],
        "random_seed":  42,
        **scenario["dataset"],
    })
    df, _ = generate_dataframe(cfg)
    df["series_id"] = df["series_id"] + series_offset
    series_offset = df["series_id"].max()
    frames.append(df)
    print(f"  {scenario['label']:30s} → {df['series_id'].nunique()} series added")

combined = pd.concat(frames, ignore_index=True)
combined.to_parquet("output/mixed_combinations.parquet", index=False)

total = combined["series_id"].nunique()
print(f"\nTotal {total} series → output/mixed_combinations.parquet")
```

### Large-scale mixed generation: N series per type

```python
import pandas as pd
from betise import generate_dataframe
from betise.config import load_config

BASE_TYPES = [
    "ar", "ma", "arma", "white_noise",
    "random_walk", "random_walk_drift", "ari", "ima", "arima",
    "sarma", "sarima",
    "arch", "garch", "egarch", "aparch",
]

N_PER_TYPE   = 100
FIXED_LENGTH = 1000

frames = []
series_offset = 0

for base in BASE_TYPES:
    cfg = load_config(dataset={
        "base_series":  base,
        "num_series":   N_PER_TYPE,
        "length_range": [FIXED_LENGTH, FIXED_LENGTH],
        "random_seed":  42,
    })
    df, _ = generate_dataframe(cfg)
    df["series_id"] = df["series_id"] + series_offset
    series_offset = df["series_id"].max()
    frames.append(df)
    print(f"  {base:20s} → {N_PER_TYPE} series")

combined = pd.concat(frames, ignore_index=True)
combined.to_parquet("output/all_types.parquet", index=False)

total = combined["series_id"].nunique()
print(f"\nTotal {total} series ({len(BASE_TYPES)} types × {N_PER_TYPE})")
```

---

## 7. Classification Dataset

Generate a balanced multi-class dataset where `primary_category` is the class label.

### Class layout (7 classes × 2,000 = 14,000 series)

| Class | Sub-types | Split |
|-------|-----------|-------|
| `stationary` | ar, ma, arma, white_noise | 500 × 4 |
| `stochastic` | random_walk, rw_drift, ari, ima, arima | 400 × 5 |
| `trend` | linear↑/↓, quadratic, cubic, exponential | 400 × 5 |
| `seasonality` | sarma, sarima, single, multiple | 500 × 4 |
| `volatility` | arch, garch, egarch, aparch | 500 × 4 |
| `anomaly` | point, collective, contextual | ~667 × 3 |
| `structural_break` | mean_shift, variance_shift, trend_shift | ~667 × 3 |

### Generate

```bash
python examples/05_classification_dataset.py
```

Output:
```
generated-dataset/classification/
  stationary/          ← one parquet per sub-type
  stochastic/
  trend/
  ...
  betise_classification.parquet   ← merged, shuffled, ready for ML
```

The class config is defined in `examples/configs/classification_config.json` — edit `total_per_class`, `fixed_length`, or the sub-type splits there.

### Load for ML (numpy / PyTorch)

```bash
python examples/06_load_and_use.py
```

```python
import numpy as np
import pandas as pd

df = pd.read_parquet("generated-dataset/classification/betise_classification.parquet")

# Pivot to (n_series, length) matrix
X = df.pivot(index="series_id", columns="time", values="data").values.astype("float32")

# Class labels
CLASS_NAMES = sorted(df["primary_category"].unique())
label_map   = {c: i for i, c in enumerate(CLASS_NAMES)}
y = (
    df.drop_duplicates("series_id")
    .set_index("series_id")["primary_category"]
    .map(label_map)
    .values
)

print(X.shape)   # (14000, 1000)
print(y.shape)   # (14000,)
```

For PyTorch — shape `(n, 1, length)` ready for Conv1d / LSTM:

```python
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(1)  # (n, 1, length)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

loader = DataLoader(TimeSeriesDataset(X, y), batch_size=64, shuffle=True)
batch_X, batch_y = next(iter(loader))
# batch_X: (64, 1, 1000)   batch_y: (64,)
```

---

## 8. Output Format

`generate_dataframe` returns a `(DataFrame, context)` tuple. The DataFrame is in **long format** — one row per time step.

```python
df, ctx = generate_dataframe(cfg)

# Core columns
df[["series_id", "time", "data", "label"]].head()

# Metadata columns
df[["is_stationary", "primary_category", "sub_category",
    "base_series", "order", "base_coefs",
    "difference", "drift_value",
    "trend_type", "trend_slope",
    "anomaly_type", "anomaly_indices",
    "break_type", "break_indices"]].head()
```

| Column | Description |
|--------|-------------|
| `series_id` | Series identifier |
| `time` | Time index (0, 1, 2, …) |
| `data` | Series value |
| `label` | Type label (e.g. `ar`, `arima__linear_trend:upward__point_anomaly:single`) |
| `is_stationary` | 1 = stationary, 0 = non-stationary |
| `primary_category` | `stationary` / `stochastic` / `trend` / `seasonality` / `volatility` / `anomaly` / `structural_break` |
| `sub_category` | Sub-type (e.g. `ar`, `arima`, `linear_trend`) |
| `difference` | Integration order d (stochastic series) |
| `drift_value` | Drift coefficient (`random_walk_drift` only) |
| `seasonal_ar_order` | Seasonal AR order P (SARMA/SARIMA) |
| `seasonal_ma_order` | Seasonal MA order Q (SARMA/SARIMA) |
| `seasonal_difference` | Seasonal differencing order D (SARMA/SARIMA) |
| `arima_ar_order` | ARIMA component AR order (ARIMA-GARCH only) |
| `arima_ma_order` | ARIMA component MA order (ARIMA-GARCH only) |
| `arima_diff` | ARIMA component differencing order (ARIMA-GARCH only) |
| `anomaly_indices` | Anomaly positions (list) |
| `break_indices` | Break point positions (list) |

### Reading from parquet

```python
import pandas as pd

df = pd.read_parquet("output/mixed_combinations.parquet")

# Filter by type
ar_series     = df[df["base_series"] == "ar"]
trend_series  = df[df["primary_category"] == "trend"]
nonstationary = df[df["is_stationary"] == 0]

# Extract a single series
series_1 = df[df["series_id"] == 1]["data"].values
```
