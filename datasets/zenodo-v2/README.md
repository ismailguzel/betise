# Generate the BeTiSe benchmark

Generate 120,000 new series with BeTiSe 0.6.1: 40,000 each at short (50–100),
medium (300–500) and long (1,000–10,000) lengths. Each length group contains
20,000 stationary and 20,000 non-stationary series under the generation-label convention.

Use Python 3.11 in an activated virtual environment. From the repository root:

```bash
python -m pip install -r datasets/zenodo-v2/requirements.txt
```

Try a small run first (240 series):

```bash
python datasets/zenodo-v2/prepare.py --profile pilot --output generated/pilot
python datasets/zenodo-v2/run.py generated/pilot
```

Generate the full dataset:

```bash
python datasets/zenodo-v2/prepare.py --profile candidate --output generated/dataset
python datasets/zenodo-v2/run.py generated/dataset
```

Settings are in [scope.json](scope.json). Outputs are Parquet shards with metadata;
logs and checksums are saved alongside them. Rerun the same command to skip completed
jobs. Use `(job_id, series_id)` as the global series identifier when merging jobs.

The managed runner applies the configured short-seasonal periods. This produces a
new realization of the benchmark, not the original archived numerical series.
