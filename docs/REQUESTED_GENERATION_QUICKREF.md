# BeTiSe — Requested Generation

`requested` mode generates an exact number of time series from one specified combination of base components and overlay features. Every series uses that combination; categorical settings and numerical parameters may vary.

## 1. Configure and run

Activate the project's Python environment and run commands from the repository root. Replace `betise/config/generation_config.json` with:

```json
{
  "mode": "requested",
  "output_dir": "generated-dataset/request_ar_point_mean",
  "base_components": ["ar"],
  "features": ["point_anomaly", "mean_shift"],
  "num_series": 12,
  "categorical_mode": "sampled",
  "variants_per_type": 3,
  "length_category": "medium",
  "seed": 42,
  "shard_size": 4
}
```

Run:

```powershell
python -m betise.run_scenario_generation
```

To use a config at another location, pass its path:

```powershell
python -m betise.run_scenario_generation path/to/config.json
```

The example produces **12 series**, distributed across **3 categorical recipes**, with lengths between **300 and 500**. Use a different `output_dir` for each run to keep outputs separate.

## 2. Config fields

| Field | Meaning |
| --- | --- |
| `mode` | Set to `"requested"`. |
| `output_dir` | Output directory, relative to the working directory or absolute. |
| `base_components` | Nonempty list of canonical base names. |
| `features` | List of overlay names; use `[]` for base-only generation. |
| `num_series` | Exact total number of series to generate. Positive integer. |
| `categorical_mode` | `"sampled"` or `"all"`; see below. |
| `variants_per_type` | Positive integer limiting recipe selection in sampled mode. |
| `length_category` **or** `length` | Provide exactly one length setting. |
| `seed` | Integer random seed. |
| `shard_size` | Maximum number of complete series per Parquet file, not rows. Positive integer. |

`min_size`, `max_size`, `series_per_recipe`, and `max_recipes` belong to full mode. In requested mode, the component lists define the combination and `num_series` defines its output count.

## 3. Component names and validation

| Base family | Names for `base_components` |
| --- | --- |
| Stationary | `ar`, `ma`, `arma`, `white_noise` |
| Stochastic | `random_walk`, `random_walk_drift`, `ari`, `ima`, `arima` |
| Fractional | `arfima` |
| Volatility | `arch`, `garch`, `egarch`, `aparch` |
| Seasonality | `single_seasonality`, `multiple_seasonality`, `sarma`, `sarima` |

| Overlay family | Names for `features` |
| --- | --- |
| Trend | `linear_trend`, `quadratic_trend`, `cubic_trend`, `exponential_trend`, `damped_trend` |
| Break | `mean_shift`, `variance_shift`, `trend_shift` |
| Anomaly | `point_anomaly`, `collective_anomaly`, `contextual_anomaly` |

The canonical rules validate every combination. Use at most one subtype per base family and one deterministic trend. `trend_shift` requires `linear_trend`; `contextual_anomaly` requires `single_seasonality` or `multiple_seasonality`; `variance_shift` cannot accompany a volatility base. Other base compatibility rules are also enforced.

## 4. Length and short rules

| Length category | Inclusive range |
| --- | --- |
| `short` | 50–100 |
| `medium` | 300–500 |
| `long` | 1000–10000 |

For exact-length generation, replace `"length_category": "medium"` with `"length": 55`. Every generated series then has exactly 55 rows.

**Short rules apply to both `length_category="short"` and exact lengths 50–100.** The following combinations are rejected:

- Two or more different shift subtypes.
- Collective anomaly plus any shift.
- Contextual anomaly plus any shift.
- Collective plus contextual anomaly.

For short series, collective/contextual anomalies and all shifts use only **single-event categorical recipes**. Point anomalies may still use multiple-event recipes.

For medium/long series, multiple-event recipes remain available. When two or more dense feature types (collective/contextual anomalies or shifts) coexist, each has at most two events; this is a per-feature limit.

## 5. Recipes and series counts

A **recipe** fixes categorical choices such as location, direction, shape, and event count. A **series** is one numerical realization of that recipe.

- `"sampled"`: select up to `variants_per_type` recipes, also limited by the available recipes and `num_series`.
- `"all"`: use every allowed recipe. `num_series` must be at least the recipe count; otherwise the request is rejected. `variants_per_type` does not limit this mode.

The total is always `num_series`. Realizations are distributed as evenly as possible across the selected recipes.

Example: `base_components=["ar"]`, `features=["mean_shift"]`, `length=55`, `categorical_mode="all"`, and `num_series=6` produces all **6 allowed single-event recipes**, one series each. With medium length, this combination has **8 recipes**, so all mode requires at least 8 series.

## 6. Output and errors

For the first example, output contains:

- `generated-dataset/request_ar_point_mean/3-way/part-00000.parquet` and subsequent shards.
- `generated-dataset/request_ar_point_mean/request_summary.json` with the actual series/recipe counts and length range.

The folder name counts base components plus features: one base and two features gives `3-way`. Each series remains complete inside one shard. Rows include `series_id`, canonical context, and localization labels.

Read a shard with:

```python
import pandas as pd

df = pd.read_parquet(
    "generated-dataset/request_ar_point_mean/3-way/part-00000.parquet"
)
print(df.groupby("series_id").size())
```

The CLI prints the config summary. A rejected short combination ends with:

```text
Error: Requested combination is not allowed for the selected series length.
```

Expected validation errors exit with status 1 and no traceback. A successful run prints its final summary; check that `total_series` equals `num_series`.
