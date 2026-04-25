"""Configuration loader for the generation pipeline."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base (override wins on scalar conflicts)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    config_dir: Optional[str] = None,
    *,
    dataset: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load generation configuration, with optional in-memory overrides.

    Parameters
    ----------
    config_dir : str, optional
        Directory containing ``params.json`` and ``dataset.json``.
        Defaults to the built-in config directory shipped with the package.
    dataset : dict, optional
        In-memory overrides for the dataset config (deep-merged over the
        file defaults). Useful for scripted generation without editing files.
    params : dict, optional
        In-memory overrides for the params config.

    Returns
    -------
    dict
        ``{"params": {...}, "dataset": {...}}`` ready for generate_dataframe().

    Examples
    --------
    # Default config only
    cfg = load_config()

    # In-memory override (no file changes needed)
    cfg = load_config(dataset={
        "base_series": "arima",
        "num_series": 50,
        "length_range": [300, 600],
        "random_seed": 7,
        "features": {
            "linear_trend": {"enabled": True, "direction": "upward"},
        },
    })

    # Load from a custom directory then override
    cfg = load_config("path/to/my_configs/", dataset={"num_series": 5})
    """
    base_dir = Path(config_dir) if config_dir else Path(__file__).resolve().parent

    cfg_params  = _load_json_config(base_dir / "params.json")
    cfg_dataset = _load_json_config(base_dir / "dataset.json")

    if params is not None:
        cfg_params = _deep_merge(cfg_params, params)

    if dataset is not None:
        cfg_dataset = _deep_merge(cfg_dataset, dataset)

    return {"params": cfg_params, "dataset": cfg_dataset}
