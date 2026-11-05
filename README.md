# BeTiSe — Benchmark Time Series Generator

A Python library for generating synthetic time series datasets with configurable statistical properties and rich metadata.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper DOI](https://img.shields.io/badge/Paper%20DOI-10.3390%2Fcmsf2026016024-blue)](https://doi.org/10.3390/cmsf2026016024)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18513505.svg)](https://doi.org/10.5281/zenodo.18513505)

## Installation

```bash
pip install betise
```

## Quick Start

```python
from betise import generate_dataframe, load_config

cfg = load_config(dataset={"base_series": "arma", "num_series": 5, "length_range": [300, 500]})
df, ctx = generate_dataframe(cfg)
print(df[["series_id", "time", "data", "base_components"]].head())
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
| Seasonal | `sarma`, `sarima`, `single_seasonality`, `multiple_seasonality` |
| Volatility | `arch`, `garch`, `egarch`, `aparch` |
| Fractional | `arfima` |

## Feature Overlays

Multiple features can be stacked on top of any base type:

| Category | Features |
|---|---|
| Trend | `linear_trend`, `quadratic_trend`, `cubic_trend`, `exponential_trend`,`damped_trend` |
| Anomaly | `point_anomaly`, `collective_anomaly`, `contextual_anomaly` |
| Structural break | `mean_shift`, `variance_shift`, `trend_shift` |

## Publication & Dataset

The BeTiSe paper is published in *Computer Sciences & Mathematics Forum*, 2026, 16(1), Article 24:

**[BeTiSe: A Comprehensive Benchmark Dataset for Univariate Time Series Stationarity and Structural Analysis](https://doi.org/10.3390/cmsf2026016024)**

A large-scale benchmark dataset (120,000 series, 23.8 GB) generated with BeTiSe is available on Zenodo:

- **Dataset DOI**: [10.5281/zenodo.18513505](https://doi.org/10.5281/zenodo.18513505)

## Documentation & Examples

Full usage guide, config reference, and ready-to-run examples are on GitHub:  
**[github.com/ismailguzel/betise](https://github.com/ismailguzel/betise)**

| Document | Contents |
|----------|----------|
| [USAGE.md](https://github.com/ismailguzel/betise/blob/main/USAGE.md) | Base series, feature overlays, output format, and the canonical catalog |
| [docs/REQUESTED_GENERATION_QUICKREF.md](https://github.com/ismailguzel/betise/blob/main/docs/REQUESTED_GENERATION_QUICKREF.md) | Generating N series from one exact combination — short reference |
| [docs/REQUESTED_GENERATION_GUIDE.md](https://github.com/ismailguzel/betise/blob/main/docs/REQUESTED_GENERATION_GUIDE.md) | The same workflow in full detail |
| [docs/combination-rules/](https://github.com/ismailguzel/betise/tree/main/docs/combination-rules/) | Which base/feature combinations are valid, per base family |
| [APPENDIX.md](https://github.com/ismailguzel/betise/blob/main/APPENDIX.md) | Paper appendix — metadata schema, generation templates, LLM prompts, example figures |

## Reproduction Package

[Download the reproduction package](reproduction-packages/betise-reproduction-v1.1.0.zip)
or follow the [generation instructions](datasets/zenodo-v2/README.md) to generate
120,000 series with BeTiSe 0.6.1.

## Citation

Please cite the paper when using BeTiSe:

Gür, K., Yazıcı, P., Erkaya, P., Türkmen, Y., Baytak, B., Güzel, İ., Karagöz, P., & Yozgatlıgil, C. (2026). BeTiSe: A comprehensive benchmark dataset for univariate time series stationarity and structural analysis. *Computer Sciences & Mathematics Forum, 16*(1), Article 24. https://doi.org/10.3390/cmsf2026016024

```bibtex
@article{gur2026betise,
  author  = {Gür, Kerem R and Yazıcı, P. Cemre and Erkaya, Pelin and Türkmen, Yağmur and Baytak, Berke and Güzel, İsmail and Karagöz, Pınar and Yozgatlıgil, Ceylan},
  title   = {{BeTiSe}: A Comprehensive Benchmark Dataset for Univariate Time Series Stationarity and Structural Analysis},
  journal = {Computer Sciences & Mathematics Forum},
  year    = {2026},
  volume  = {16},
  number  = {1},
  pages   = {24},
  doi     = {10.3390/cmsf2026016024},
  url     = {https://doi.org/10.3390/cmsf2026016024}
}
```

If you use the published benchmark dataset, please also cite its Zenodo record:

```bibtex
@dataset{betise2026,
  author    = {Gür, Kerem R and Yazıcı, P. Cemre and Erkaya, Pelin and Türkmen, Yağmur and Baytak, Berke and Güzel, İsmail and Karagöz, Pınar and Yozgatlıgil, Ceylan},
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
