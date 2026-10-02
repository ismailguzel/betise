"""
Scenario-driven BeTiSe time-series generation.

This module connects:

    rules.py
        ↓
    scenario_builder.py
        ↓
    categorical variants
        ↓
    params.json
        ↓
    full_dataset_generation.generate_full_series()

It does not define combination validity.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pandas as pd

from betise.config import load_config
from betise.full_dataset_generation import (
    generate_full_series,
)

from betise.scenario_builder import (
    enumerate_type_scenarios,
    iter_materialized_scenarios,
    to_generation_composition,
    build_type_scenario,
)


# ============================================================================
# LENGTH SAMPLING
# ============================================================================

def _sample_length(
    length_range,
) -> int:
    """
    Sample one series length from [low, high].
    """

    if len(length_range) != 2:
        raise ValueError(
            "length_range must contain "
            "[low, high]."
        )

    low = int(
        length_range[0]
    )

    high = int(
        length_range[1]
    )

    if low <= 0:
        raise ValueError(
            "Minimum length must be positive."
        )

    if high < low:
        raise ValueError(
            "Maximum length must be >= minimum length."
        )

    return int(
        np.random.randint(
            low,
            high + 1,
        )
    )


# ============================================================================
# SERIES ITERATOR
# ============================================================================

def iter_generated_series(
    *,
    min_size: int = 1,
    max_size: int = 5,
    categorical_mode: str = "sampled",
    variants_per_type: int = 1,
    series_per_recipe: int = 1,
    length_range=(300, 500),
    seed: int = 42,
    max_recipes: Optional[int] = None,
) -> Iterator[
    tuple[
        pd.DataFrame,
        Dict[str, Any],
    ]
]:
    """
    Generate actual BeTiSe time series.

    Parameters
    ----------
    min_size, max_size:
        Allowed total combination size.

    categorical_mode:
        Categorical variant selection mode:

            "all"
                Use every categorical variant combination
                available for each type-level scenario.

            "sampled"
                Select a limited number of categorical recipes
                for each type-level scenario.

    variants_per_type:
        Number of categorical recipes selected per type-level
        scenario when categorical_mode="sampled".

        If a type scenario has fewer possible categorical recipes
        than this value, all available recipes are used.

    series_per_recipe:
        Number of independent numerical realizations generated
        from each categorical recipe.

    length_range:
        [minimum_length, maximum_length]

    seed:
        Reproducibility seed.

    max_recipes:
        Optional safety/debug limit.

        Example:
            max_recipes=10

        means:
            process only the first 10 categorical recipes.

        This is especially useful for smoke tests.

    Yields
    ------
    (dataframe, generation_context)
    """

    if series_per_recipe <= 0:
        raise ValueError(
            "series_per_recipe must be positive."
        )

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    # ------------------------------------------------------------
    # Numerical parameter configuration
    # ------------------------------------------------------------

    cfg = load_config()

    params_cfg = cfg[
        "params"
    ]

    # full_dataset_generation only needs
    # feature_defaults from full_cfg here.
    full_cfg = {
        "feature_defaults": {}
    }

    # ------------------------------------------------------------
    # Type-level scenario space
    # ------------------------------------------------------------

    type_scenarios = (
        enumerate_type_scenarios(
            min_size=min_size,
            max_size=max_size,
        )
    )

    # ------------------------------------------------------------
    # Categorical materialization
    # ------------------------------------------------------------

    materialized_iterator = (
        iter_materialized_scenarios(
        type_scenarios=type_scenarios,
        categorical_mode=categorical_mode,
        variants_per_type=variants_per_type,
        seed=seed,
        )
    )

    series_id = 1
    recipe_index = 0

    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------

    for materialized in (
        materialized_iterator
    ):

        if (
            max_recipes is not None
            and recipe_index
            >= max_recipes
        ):
            break

        recipe_index += 1

        composition = (
            to_generation_composition(
                materialized
            )
        )

        for realization_index in range(
            series_per_recipe
        ):

            length = _sample_length(
                length_range
            )

            dataframe = (
                generate_full_series(
                    composition=composition,
                    full_cfg=full_cfg,
                    params_cfg=params_cfg,
                    series_id=series_id,
                    length=length,
                )
            )

            context = {
                "series_id": (
                    series_id
                ),

                "recipe_index": (
                    recipe_index
                ),

                "realization_index": (
                    realization_index
                ),

                "type_scenario_id": (
                    materialized[
                        "scenario_id"
                    ]
                ),

                "materialized_scenario_id": (
                    materialized[
                        "materialized_scenario_id"
                    ]
                ),

                "combination_size": (
                    materialized[
                        "combination_size"
                    ]
                ),

                "base_components": list(
                    materialized[
                        "base_components"
                    ]
                ),

                "feature_components": list(
                    materialized[
                        "feature_components"
                    ]
                ),

                "categorical_variant_ids": dict(
                    materialized.get(
                        "categorical_variant_ids",
                        {},
                    )
                ),

                "feature_overrides": dict(
                    materialized.get(
                        "feature_overrides",
                        {},
                    )
                ),

                "length": (
                    length
                ),

                "seed": (
                    seed
                ),
            }

            yield (
                dataframe,
                context,
            )

            series_id += 1

def iter_requested_series(
    *,
    base_components,
    feature_components=(),
    num_series: int,
    categorical_mode: str = "sampled",
    variants_per_type: int = 1,
    length_range=(300, 500),
    seed: int = 42,
) -> Iterator[
    tuple[
        pd.DataFrame,
        Dict[str, Any],
    ]
]:
    """
    Generate an exact number of time series from one explicitly
    requested canonical BeTiSe combination.

    Example
    -------
    arch
        + mean_shift
        + point_anomaly

    num_series=100

    produces exactly 100 actual time series.

    categorical_mode controls how categorical recipes are selected,
    while num_series controls the final number of numerical
    realizations.
    """

    if num_series <= 0:
        raise ValueError(
            "num_series must be positive."
        )

    if variants_per_type <= 0:
        raise ValueError(
            "variants_per_type must be positive."
        )

    categorical_mode = (
        categorical_mode.lower()
    )

    if categorical_mode not in {
        "all",
        "sampled",
    }:
        raise ValueError(
            "categorical_mode must be "
            "'all' or 'sampled'."
        )

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    # ------------------------------------------------------------
    # Numerical parameters
    # ------------------------------------------------------------

    cfg = load_config()

    params_cfg = cfg[
        "params"
    ]

    full_cfg = {
        "feature_defaults": {}
    }

    # ------------------------------------------------------------
    # Build and validate exactly ONE requested type scenario.
    # ------------------------------------------------------------

    scenario = (
        build_type_scenario(
            base_components=base_components,
            feature_components=feature_components,
        )
    )

    # ------------------------------------------------------------
    # Categorical recipe selection
    # ------------------------------------------------------------

    if categorical_mode == "sampled":

        # Never select more recipes than actual series,
        # because every selected recipe should produce at
        # least one realization.
        effective_variants = min(
            variants_per_type,
            num_series,
        )

    else:

        effective_variants = (
            variants_per_type
        )

    materialized_recipes = list(
        iter_materialized_scenarios(
            type_scenarios=[
                scenario
            ],
            categorical_mode=categorical_mode,
            seed=seed,
            variants_per_type=effective_variants,
        )
    )

    if not materialized_recipes:
        raise RuntimeError(
            "Requested scenario produced zero "
            "categorical recipes."
        )

    # Keep recipe ordering deterministic.
    materialized_recipes.sort(
        key=lambda recipe: (
            recipe[
                "materialized_scenario_id"
            ]
        )
    )

    recipe_count = len(
        materialized_recipes
    )

    # In "all" mode, every categorical recipe must appear
    # at least once. Therefore num_series cannot be smaller
    # than the number of recipes.
    if (
        categorical_mode == "all"
        and recipe_count > num_series
    ):
        raise ValueError(
            "categorical_mode='all' requires "
            "num_series to be at least the number "
            "of categorical recipes. "
            f"Requested num_series={num_series}, "
            f"but this combination has "
            f"{recipe_count} recipes."
        )

    # ------------------------------------------------------------
    # Exact allocation
    #
    # Example:
    #   100 series / 6 recipes
    #
    #   17, 17, 17, 17, 16, 16
    #
    # Total is always exactly 100.
    # ------------------------------------------------------------

    base_count = (
        num_series
        //
        recipe_count
    )

    remainder = (
        num_series
        %
        recipe_count
    )

    series_id = 1

    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------

    for recipe_position, materialized in enumerate(
        materialized_recipes
    ):

        realization_count = (
            base_count
            +
            (
                1
                if recipe_position
                < remainder
                else 0
            )
        )

        composition = (
            to_generation_composition(
                materialized
            )
        )

        for realization_index in range(
            realization_count
        ):

            length = _sample_length(
                length_range
            )

            dataframe = (
                generate_full_series(
                    composition=composition,
                    full_cfg=full_cfg,
                    params_cfg=params_cfg,
                    series_id=series_id,
                    length=length,
                )
            )

            context = {
                "series_id": (
                    series_id
                ),

                "recipe_index": (
                    recipe_position + 1
                ),

                "realization_index": (
                    realization_index
                ),

                "type_scenario_id": (
                    materialized[
                        "scenario_id"
                    ]
                ),

                "materialized_scenario_id": (
                    materialized[
                        "materialized_scenario_id"
                    ]
                ),

                "combination_size": (
                    materialized[
                        "combination_size"
                    ]
                ),

                "base_components": list(
                    materialized[
                        "base_components"
                    ]
                ),

                "feature_components": list(
                    materialized[
                        "feature_components"
                    ]
                ),

                "categorical_variant_ids": dict(
                    materialized.get(
                        "categorical_variant_ids",
                        {},
                    )
                ),

                "feature_overrides": dict(
                    materialized.get(
                        "feature_overrides",
                        {},
                    )
                ),

                "length": (
                    length
                ),

                "seed": (
                    seed
                ),

                "requested_num_series": (
                    num_series
                ),
            }

            yield (
                dataframe,
                context,
            )

            series_id += 1