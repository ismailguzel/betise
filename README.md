# BeTiSe — Benchmark Time Series Generator

A modular Python library for generating synthetic time series datasets with rich, reproducible metadata.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18513505.svg)](https://doi.org/10.5281/zenodo.18513505)

## Overview

BeTiSe provides a comprehensive toolkit for generating synthetic time series data with configurable statistical properties. It is designed for researchers, data scientists, and ML practitioners who need reproducible, well-documented time series datasets for benchmarking, model training, or educational purposes.

## Published Dataset

A large-scale benchmark dataset generated with this library has been published on Zenodo.

- **Dataset Name**: BeTiSe: A Benchmark Time Series Dataset for Stationarity and Structural Analysis
- **DOI**: [10.5281/zenodo.18513505](https://doi.org/10.5281/zenodo.18513505)
- **Size**: 120,000 univariate time series (23.8 GB)
- **Conference**: Submitted to [ITISE 2026](https://itise.ugr.es/)

Access: [https://zenodo.org/records/18513505](https://zenodo.org/records/18513505)

## Installation

```bash
pip install betise
```

Or install from source:

```bash
git clone https://github.com/ismailguzel/betise.git
cd betise
pip install -e .
```

For gallery scripts and the introductory notebook, install visualization dependencies too:

```bash
pip install "betise[viz]"
```

## Quick Start

```python
from betise import generate_dataframe, load_config

# In-memory — no file written
cfg = load_config(dataset={"base_series": "arma", "num_series": 5, "length_range": [300, 500]})
df, ctx = generate_dataframe(cfg)

# Save to parquet
from betise import run

cfg = load_config(dataset={
    "base_series":  "ar",
    "num_series":   10,
    "length_range": [200, 500],
    "output_dir":   "output",
    "output_name":  "ar_demo.parquet",
    "features": {
        "linear_trend": {"enabled": True, "direction": "upward"},
    },
})
run(cfg)
```

### Load generated data

```python
import pandas as pd

df = pd.read_parquet("output/ar_demo.parquet")
print(df[["series_id", "time", "data", "primary_category", "sub_category"]].head())
```

For full loading examples (numpy, sklearn, PyTorch) see `examples/06_load_and_use.py`.

## Series Types

| Category | Base types |
|---|---|
| Stationary | `ar`, `ma`, `arma`, `white_noise` |
| Stochastic | `random_walk`, `random_walk_drift`, `ari`, `ima`, `arima` |
| Seasonal | `sarma`, `sarima` |
| Volatility | `arch`, `garch`, `egarch`, `aparch` |

Feature overlays (trend, seasonality, anomaly, structural break) can be combined on top of any base type. See [USAGE.md](USAGE.md) for the full feature reference.

## Examples

```
examples/
├── 00_introduction.ipynb          # Interactive getting-started notebook
├── 01_quickstart.py               # In-memory generation, save to disk, feature combinations
├── 02_benchmark_dataset.py        # All base types × 3 length buckets (~495 series)
├── 03_feature_suite.py            # All base types × all feature types, phased (~4,200 series)
├── 04_pretraining_dataset.py      # Large-scale fixed-length dataset (default 75k, scalable)
├── 05_classification_dataset.py   # Balanced 7-class ML dataset (14,000 series)
├── 06_load_and_use.py             # Load parquet → numpy / sklearn / PyTorch
├── 07_feature_gallery.py          # PDF gallery: all 15 base types + all 12 features
├── 08_combinations_gallery.py     # PDF gallery: every base × feature combination (573 plots)
├── configs/
│   └── classification_config.json # Class / sub-type config for script 05
└── data/
    └── combinations.csv           # Combination definitions for script 08
```

Run any example:

```bash
python examples/01_quickstart.py
python examples/07_feature_gallery.py   # produces feature_gallery.pdf
python examples/08_combinations_gallery.py  # produces combinations_gallery.pdf
```

## Project Structure

```
betise/
├── betise/
│   ├── __init__.py                 # Public API: run, generate_dataframe, load_config
│   ├── dataset_generation.py       # generate_dataframe() / run() pipeline
│   ├── config/
│   │   ├── __init__.py             # load_config() with deep merge
│   │   ├── dataset.json            # Default dataset settings
│   │   └── params.json             # Default process parameters
│   ├── core/
│   │   ├── generator.py            # TimeSeriesGenerator
│   │   └── metadata.py             # create_metadata_record()
│   └── utils/
│       └── helpers.py              # Internal helpers
├── examples/                       # Ready-to-run scenarios (see above)
├── tests/                          # Test suite
├── USAGE.md                        # Full feature & config reference
├── CHANGELOG.md
├── pyproject.toml
└── requirements.txt
```

## Reproducibility

Default seed is 42. ARCH/GARCH models may show minor non-determinism (~1–2%) due to upstream library behaviour.

## Dependencies

| Package | Min version | Purpose |
|---|---|---|
| `numpy` | 1.21 | Array operations |
| `pandas` | 1.3 | DataFrame output |
| `statsmodels` | 0.13 | ARIMA/SARIMA generation |
| `arch` | 5.0 | ARCH/GARCH generation |
| `pyarrow` | 7.0 | Parquet I/O |

Optional: `matplotlib>=3.4`, `tqdm>=4.60` for gallery scripts and the notebook.

## Citation

If you use BeTiSe or the published dataset in your research, please cite:

```bibtex
@dataset{betise2026,
  author    = {Gür, Kerem and Yazıcı, Pınar Cemre and Erkaya, Pelin and Türkmen, Yağmur and Baytak, Berke and Güzel, İsmail and Karagöz, Pınar and Yozgatlıgil, Ceylan}},
  title     = {{BeTiSe: A Benchmark Time Series Dataset for Stationarity
                and Structural Analysis}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18513505},
  url       = {https://doi.org/10.5281/zenodo.18513505}
}
```

## Funding

- **TÜBİTAK** — Grant No. 124F095
- **METU** Scientific Research Projects — Grant No. GAP-109-2023-11361

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).

---

**Version**: 0.2.0 | **License**: MIT
