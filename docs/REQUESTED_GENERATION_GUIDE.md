# BeTiSe – Requested Dataset Generation Guide

This guide describes how to generate a specified number of synthetic time series from one exact BeTiSe combination.

This interface is intended for routine dataset generation tasks. The full scenario-generation workflow remains available separately for exhaustive validation and coverage analysis.

## 1. Basic Workflow

Create or copy a requested-generation configuration file, for example:

```text
examples/configs/requested_generation_template.json
```

Then run:

```powershell
python -m betise.run_scenario_generation examples/configs/requested_generation_template.json
```

BeTiSe will:

1. validate the requested combination using the canonical combination rules,
2. construct the available categorical variants,
3. select categorical recipes according to the requested mode,
4. generate the requested number of numerical realizations,
5. save the generated series as Parquet shards,
6. create a generation summary.

---

## 2. Configuration Example

```json
{
  "mode": "requested",

  "output_dir": "generated-dataset/my_task",

  "base_components": [
    "arch"
  ],

  "features": [
    "point_anomaly",
    "mean_shift"
  ],

  "num_series": 100,

  "categorical_mode": "sampled",

  "variants_per_type": 10,

  "length_range": [
    300,
    500
  ],

  "seed": 42,

  "shard_size": 100
}
```

---

## 3. Configuration Fields

### `mode`

For exact-combination generation, always use:

```json
"mode": "requested"
```

This tells BeTiSe to generate only the combination explicitly specified in the configuration.

---

### `output_dir`

Directory where the generated dataset will be stored.

Example:

```json
"output_dir": "generated-dataset/task_01"
```

Use a different output directory for different generation tasks to avoid mixing datasets.

---

### `base_components`

Defines the mathematical base components of the requested time series.

Example:

```json
"base_components": [
  "ar",
  "garch"
]
```

Available canonical base types include:

```text
Stationary
- ar
- ma
- arma
- white_noise

Stochastic
- random_walk
- random_walk_drift
- ari
- ima
- arima

Seasonal
- single_seasonality
- multiple_seasonality
- sarma
- sarima

Volatility
- arch
- garch
- egarch
- aparch

Fractional
- arfima
```

Not every base combination is valid. The canonical BeTiSe rule system automatically checks the requested combination before generation.

---

### `features`

Defines optional overlay features.

Available overlays include:

```text
Trend
- linear_trend
- quadratic_trend
- cubic_trend
- exponential_trend
- damped_trend

Structural Break
- mean_shift
- variance_shift
- trend_shift

Anomaly
- point_anomaly
- collective_anomaly
- contextual_anomaly
```

Example:

```json
"features": [
  "linear_trend",
  "point_anomaly"
]
```

Feature order in the configuration does not determine execution order.

BeTiSe applies the canonical order:

```text
trend
→ structural break
→ anomaly
```

---

## 4. Number of Series

`num_series` defines the exact final number of generated time series.

Example:

```json
"num_series": 1000
```

BeTiSe will generate exactly 1000 series from the requested combination.

This value is independent from the number of categorical recipes.

---

## 5. Categorical Generation

Categorical parameters describe structural variations such as:

* anomaly location,
* spike/drop direction,
* single/multiple anomalies,
* break direction,
* break count,
* trend direction,
* anomaly shape,
* trend-shift type.

There are two categorical generation modes.

### Sampled Mode

```json
"categorical_mode": "sampled",
"variants_per_type": 10
```

BeTiSe randomly selects up to 10 unique categorical recipes from the complete categorical space of the requested combination.

The selection is controlled by `seed`, so the same configuration and seed reproduce the same categorical selection.

For example, if a combination contains 63 possible categorical recipes:

```text
categorical_mode = sampled
variants_per_type = 10
```

means:

```text
63 possible recipes
       ↓
10 recipes sampled
       ↓
num_series distributed across these recipes
```

If:

```json
"num_series": 100
```

the final output still contains exactly 100 time series.

---

### All Mode

To use every possible categorical recipe:

```json
"categorical_mode": "all"
```

In this mode, `variants_per_type` does not restrict the categorical space.

For example:

```text
point_anomaly = 7 variants
mean_shift    = 9 variants

7 × 9 = 63 categorical recipes
```

With:

```json
"categorical_mode": "all",
"num_series": 100
```

all 63 recipes will be represented, and the 100 numerical realizations will be distributed across them.

If `num_series` is smaller than the number of available categorical recipes, generation is rejected because every recipe cannot be represented at least once.

---

## 6. Numerical Realizations

Categorical recipe and actual time series are not the same thing.

A categorical recipe defines the structural configuration.

For example:

```text
point anomaly:
    single
    beginning
    spike

mean shift:
    single
    middle
    upward
```

Multiple actual time series may be generated from the same categorical recipe.

The numerical parameters of each realization are sampled from the parameter definitions in:

```text
betise/config/params.json
```

Therefore, series generated from the same categorical recipe are still independent numerical realizations.

---

## 7. Series Length

```json
"length_range": [
  300,
  500
]
```

For each generated time series, BeTiSe samples a length between the specified minimum and maximum values.

To force every series to have the same length:

```json
"length_range": [
  500,
  500
]
```

---

## 8. Reproducibility

```json
"seed": 42
```

The seed controls reproducible random generation.

Using the same configuration and the same seed reproduces the same generation procedure, including categorical sampling.

Use different seeds when generating independent dataset batches.

---

## 9. Parquet Shards

```json
"shard_size": 100
```

This determines how many complete time series are stored in each Parquet shard.

For example:

```text
num_series = 1000
shard_size = 100
```

produces approximately:

```text
10 Parquet shards
```

Shard size refers to the number of time series, not the number of dataframe rows.

---

## 10. Output Structure

A requested generation produces an output structure similar to:

```text
generated-dataset/
└── my_task/
    ├── request_summary.json
    └── 3-way/
        ├── part-00000.parquet
        ├── part-00001.parquet
        └── ...
```

`request_summary.json` records information such as:

* requested base components,
* requested features,
* combination size,
* requested number of series,
* categorical mode,
* number of categorical recipes used,
* length range,
* seed,
* shard size,
* total generated series,
* number of written shards.

---

## 11. Invalid Combinations

Users do not need to manually determine whether every requested combination is valid.

BeTiSe validates the requested composition using its canonical rule system before series generation.

For example:

```json
"base_components": [
  "garch"
],
"features": [
  "variance_shift"
]
```

is rejected because variance shift is incompatible with a volatility base under the current BeTiSe rules.

Generation should not proceed for invalid combinations.

---

## 12. Recommended Workflow for Assigned Generation Tasks

For each assigned task:

1. Copy `requested_generation_template.json`.
2. Give the copied file a task-specific name.
3. Set `output_dir`.
4. Set `base_components`.
5. Set `features`.
6. Set `num_series`.
7. Select `sampled` or `all` categorical generation.
8. Set an appropriate seed.
9. Run the generation command.
10. Check `request_summary.json` after generation.

Example:

```powershell
python -m betise.run_scenario_generation examples/configs/task_01.json
```

Before large generation jobs, it is recommended to perform a small smoke test such as:

```json
"num_series": 10,
"shard_size": 5
```

and verify that the requested combination and output structure are correct.
