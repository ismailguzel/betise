"""Configuration loaders for BeTiSe generation pipelines."""

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


def _deep_merge(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """Recursively merge override into base; override wins on conflicts."""

    result = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(
                result[key],
                value,
            )
        else:
            result[key] = copy.deepcopy(
                value
            )

    return result


def load_config(
    config_dir: Optional[str] = None,
    *,
    dataset: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load the legacy/simple BeTiSe dataset configuration.

    This intentionally continues to load ``dataset.json`` so existing examples
    and the legacy single-root-base pipeline remain backward compatible.
    """

    base_dir = (
        Path(config_dir)
        if config_dir
        else Path(__file__).resolve().parent
    )

    cfg_params = _load_json_config(
        base_dir / "params.json"
    )

    cfg_dataset = _load_json_config(
        base_dir / "dataset.json"
    )

    if params is not None:
        cfg_params = _deep_merge(
            cfg_params,
            params,
        )

    if dataset is not None:
        cfg_dataset = _deep_merge(
            cfg_dataset,
            dataset,
        )

    return {
        "params": cfg_params,
        "dataset": cfg_dataset,
    }


def load_full_dataset_config(
    config_dir: Optional[str] = None,
    *,
    full_dataset: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load the canonical multi-base full-dataset configuration.

    Reads:
        params.json
        full_dataset.json

    The legacy ``dataset.json`` is deliberately not touched.

    Returns
    -------
    dict
        {
            "params": {...},
            "full_dataset": {...},
        }
    """

    base_dir = (
        Path(config_dir)
        if config_dir
        else Path(__file__).resolve().parent
    )

    cfg_params = _load_json_config(
        base_dir / "params.json"
    )

    cfg_full = _load_json_config(
        base_dir / "full_dataset.json"
    )

    if params is not None:
        cfg_params = _deep_merge(
            cfg_params,
            params,
        )

    if full_dataset is not None:
        cfg_full = _deep_merge(
            cfg_full,
            full_dataset,
        )

    _validate_full_dataset_schema(
        cfg_full
    )

    return {
        "params": cfg_params,
        "full_dataset": cfg_full,
    }


def _validate_full_dataset_schema(
    cfg: Dict[str, Any],
) -> None:
    """Lightweight structural validation for ``full_dataset.json``."""

    if not isinstance(cfg, dict):
        raise TypeError(
            "full_dataset.json must contain a JSON object."
        )

    required_top_level = {
        "random_seed",
        "output",
        "defaults",
        "feature_defaults",
        "compositions",
    }

    missing = sorted(
        required_top_level - set(cfg)
    )

    if missing:
        raise ValueError(
            "full_dataset.json is missing required fields: "
            f"{missing}"
        )

    if not isinstance(
        cfg["compositions"],
        list,
    ):
        raise TypeError(
            "'compositions' must be a list."
        )

    for index, composition in enumerate(
        cfg["compositions"]
    ):
        if not isinstance(composition, dict):
            raise TypeError(
                f"Composition #{index} must be an object."
            )

        for field in (
            "id",
            "name",
            "base_components",
            "features",
        ):
            if field not in composition:
                raise ValueError(
                    f"Composition #{index} is missing '{field}'."
                )

        if not isinstance(
            composition["base_components"],
            list,
        ):
            raise TypeError(
                f"{composition['id']}: base_components must be a list."
            )

        if not isinstance(
            composition["features"],
            list,
        ):
            raise TypeError(
                f"{composition['id']}: features must be a list."
            )
