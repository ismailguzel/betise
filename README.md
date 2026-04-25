# BeTiSe — Benchmark Time Series Generator

A Python library for generating synthetic time series datasets with configurable statistical properties and rich metadata.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18513505.svg)](https://doi.org/10.5281/zenodo.18513505)

## Installation

```bash
pip install betise
```

## Quick Start

```python
from betise import generate_dataframe, load_config

cfg = load_config(dataset={"base_series": "arma", "num_series": 5, "length_range": [300, 500]})
df, ctx = generate_dataframe(cfg)
print(df[["series_id", "time", "data", "primary_category"]].head())
```

Save to parquet and add feature overlays:

```python
from betise import run, load_config

cfg = load_config(dataset={
    "base_series":  "ar",
    "num_series":   100,
    "length_range": [300, 700],
    "output_dir":   "output",
    "output_name":  "ar_trend.parquet",
    "features": {
        "linear_trend":       {"enabled": True, "direction": "upward"},
        "single_seasonality": {"enabled": True},
        "point_anomaly":      {"enabled": True, "is_spike": True},
    },
})
run(cfg)
```

## Series Types

| Category | Base types |
|---|---|
| Stationary | `ar`, `ma`, `arma`, `white_noise` |
| Stochastic trend | `random_walk`, `random_walk_drift`, `ari`, `ima`, `arima` |
| Seasonal | `sarma`, `sarima` |
| Volatility | `arch`, `garch`, `egarch`, `aparch` |

## Feature Overlays

Multiple features can be stacked on top of any base type:

| Category | Features |
|---|---|
| Trend | `linear_trend`, `quadratic_trend`, `cubic_trend`, `exponential_trend` |
| Seasonality | `single_seasonality`, `multiple_seasonality` |
| Anomaly | `point_anomaly`, `collective_anomaly`, `contextual_anomaly` |
| Structural break | `mean_shift`, `variance_shift`, `trend_shift` |

## Published Dataset

A large-scale benchmark dataset (120,000 series, 23.8 GB) generated with BeTiSe is available on Zenodo:

- **DOI**: [10.5281/zenodo.18513505](https://doi.org/10.5281/zenodo.18513505)
- **Conference**: Submitted to [ITISE 2026](https://itise.ugr.es/)

## Documentation & Examples

Full usage guide, config reference, and ready-to-run examples are on GitHub:  
**[github.com/ismailguzel/betise](https://github.com/ismailguzel/betise)**

## Citation

```bibtex
@dataset{betise2026,
  author    = {Yazıcı, Pınar Cemre and Erkaya, Pelin and
               Türkmen, Yağmur and Güzel, İsmail and
               Karagöz, Pınar and Yozgatlıgil, Ceylan},
  title     = {{BeTiSe: A Benchmark Time Series Dataset for Stationarity
                and Structural Analysis}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18513505},
  url       = {https://doi.org/10.5281/zenodo.18513505}
}
```

## Contact

**İsmail Güzel** — ismailgzel@gmail.com

## License

MIT — see [LICENSE](https://github.com/ismailguzel/betise/blob/main/LICENSE).
