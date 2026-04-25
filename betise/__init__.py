"""BeTiSe — Benchmark Time Series Generator.

Generate synthetic time series with configurable statistical properties:
stationary processes (AR, MA, ARMA), stochastic-trend processes (ARIMA family),
seasonal models (SARMA, SARIMA), and volatility models (ARCH/GARCH family).
Feature overlays — trend, seasonality, anomaly, structural break — can be
stacked on top of any base process.

Quick start
-----------
>>> from betise import generate_dataframe
>>> from betise.config import load_config
>>>
>>> cfg = load_config(dataset={"base_series": "arma", "num_series": 5})
>>> df, ctx = generate_dataframe(cfg)

Save to disk
------------
>>> from betise import run
>>> cfg = load_config(dataset={
...     "base_series": "ar",
...     "num_series": 100,
...     "output_dir": "output",
...     "output_name": "ar_dataset.parquet",
... })
>>> run(cfg)
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("betise")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.2.0"

__author__ = "Pınar Cemre Yazıcı, Pelin Erkaya, Yağmur Türkmen, İsmail Güzel"
__license__ = "MIT"

from .core.generator import TimeSeriesGenerator
from .core.metadata import (
    create_metadata_record,
    attach_metadata_columns_to_df,
    get_metadata_columns_defaults,
    make_json_serializable,
)
from .dataset_generation import run, generate_dataframe
from .config import load_config

__all__ = [
    # Primary API
    "run",
    "generate_dataframe",
    "load_config",
    # Core class
    "TimeSeriesGenerator",
    # Metadata helpers
    "create_metadata_record",
    "attach_metadata_columns_to_df",
    "get_metadata_columns_defaults",
    "make_json_serializable",
    # Package info
    "__version__",
    "__author__",
    "__license__",
]
