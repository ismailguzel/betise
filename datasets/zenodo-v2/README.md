# Generate the BeTiSe benchmark

These native BeTiSe configs generate 120,000 new series: 40,000 per length group,
with 20,000 stationary and 20,000 non-stationary series per group under the
generation-label convention. Each of the 120 configs generates 1,000 series.
This is a new realization of the benchmark, not the original archived numerical series.

Use an activated Python 3.11 environment. From the repository root:

```bash
python -m pip install betise==0.6.1
python -m betise.run_scenario_generation datasets/zenodo-v2/configs/short/short__ar__batch01.json
```

To run all configs:

```bash
for config in datasets/zenodo-v2/configs/*/*.json; do
    python -m betise.run_scenario_generation "$config" || break
done
```

Lengths are short (50–100), medium (300–500), and long (1,000–10,000).
The three short configs for single seasonality, multiple seasonality, and contextual
anomaly use exactly 100 time steps so the library's default seasonal periods fit.
No library changes or parameter overrides are needed.

Outputs are Parquet shards under `generated/zenodo-v2/`, separated by config.
Use `(config filename, series_id)` as the global identifier when combining them.
Rerunning a config can overwrite its output; there is no custom resume runner.
