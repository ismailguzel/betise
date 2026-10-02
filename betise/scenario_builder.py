"""
Programmatic scenario builder for BeTiSe.

This module does NOT define combination validity.
The canonical source of truth is betise.core.rules.

Its responsibility is only to:
1. enumerate candidate base/feature combinations,
2. validate them through rules.py,
3. return valid scenarios for a requested combination size.

Combination size is defined as:

    len(base_components) + len(feature_components)

Examples
--------
1-way:
    ar

2-way:
    ar + garch

3-way:
    ar + linear_trend + point_anomaly

4-way:
    arima + garch + single_seasonality + linear_trend

5-way:
    arfima + garch + single_seasonality
    + linear_trend + collective_anomaly
"""

from itertools import combinations, product
from typing import Any, Dict, List, Tuple

from betise.core.rules import (
    CANONICAL_BASE_SERIES,
    OVERLAY_FEATURES,
    validate_requested_combination,
)

import json
from pathlib import Path
import random

# ============================================================================
# BASE COMPOSITIONS
# ============================================================================

def enumerate_valid_base_compositions(
    max_size: int = 5,
) -> List[Tuple[str, ...]]:
    """
    Enumerate every valid canonical base composition.

    Candidate base sets are created automatically from
    CANONICAL_BASE_SERIES and checked through rules.py.

    Parameters
    ----------
    max_size:
        Maximum number of base components to consider.

        Note:
        Under the current rule system, valid base compositions
        naturally stop before this if larger compositions are invalid.

    Returns
    -------
    list of tuple
        Valid base-component tuples.
    """

    if max_size < 1:
        raise ValueError(
            "max_size must be at least 1."
        )

    base_candidates = sorted(
        CANONICAL_BASE_SERIES
    )

    valid_compositions = []

    max_base_count = min(
        max_size,
        len(base_candidates),
    )

    for base_count in range(
        1,
        max_base_count + 1,
    ):
        for candidate in combinations(
            base_candidates,
            base_count,
        ):
            report = validate_requested_combination(
                base_components=candidate,
                feature_components=(),
            )

            if report.valid:
                valid_compositions.append(
                    tuple(
                        report.base_components
                    )
                )

    return valid_compositions


# ============================================================================
# TYPE-LEVEL SCENARIOS
# ============================================================================

def enumerate_type_scenarios(
    min_size: int = 1,
    max_size: int = 5,
) -> List[Dict[str, Any]]:
    """
    Enumerate every valid BeTiSe type-level scenario.

    A scenario consists of:
        - one or more mathematical base components,
        - zero or more overlay features.

    Combination size is:

        number of base components
        +
        number of overlay features

    No validity logic is duplicated here.
    Every candidate is checked by
    validate_requested_combination().

    Parameters
    ----------
    min_size:
        Minimum total combination size.

    max_size:
        Maximum total combination size.

    Returns
    -------
    list of dict
        Valid canonical scenario definitions.
    """

    if min_size < 1:
        raise ValueError(
            "min_size must be at least 1."
        )

    if max_size < min_size:
        raise ValueError(
            "max_size must be greater than "
            "or equal to min_size."
        )

    valid_base_compositions = (
        enumerate_valid_base_compositions(
            max_size=max_size
        )
    )

    overlay_candidates = sorted(
        OVERLAY_FEATURES
    )

    scenarios = []

    for base_components in valid_base_compositions:

        base_count = len(
            base_components
        )

        # How many overlays do we need at minimum
        # to reach min_size?
        min_feature_count = max(
            0,
            min_size - base_count,
        )

        # Do not exceed requested total size.
        max_feature_count = min(
            len(overlay_candidates),
            max_size - base_count,
        )

        if max_feature_count < 0:
            continue

        for feature_count in range(
            min_feature_count,
            max_feature_count + 1,
        ):

            for feature_components in combinations(
                overlay_candidates,
                feature_count,
            ):

                report = (
                    validate_requested_combination(
                        base_components=(
                            base_components
                        ),
                        feature_components=(
                            feature_components
                        ),
                    )
                )

                if not report.valid:
                    continue

                combination_size = (
                    len(
                        report.base_components
                    )
                    +
                    len(
                        report.feature_components
                    )
                )

                if not (
                    min_size
                    <= combination_size
                    <= max_size
                ):
                    continue

                scenario_name = "__".join(
                    report.base_components
                    +
                    report.feature_components
                )

                scenario_id = (
                    f"{combination_size}way__"
                    f"{scenario_name}"
                )

                scenarios.append(
                    {
                        "scenario_id": (
                            scenario_id
                        ),

                        "name": (
                            scenario_name
                        ),

                        "combination_size": (
                            combination_size
                        ),

                        "base_components": list(
                            report.base_components
                        ),

                        "base_families": list(
                            report.base_families
                        ),

                        "feature_components": list(
                            report.feature_components
                        ),

                        "feature_families": list(
                            report.feature_families
                        ),

                        "composition_steps": list(
                            report.composition_steps
                        ),
                    }
                )

    # Deterministic ordering.
    scenarios.sort(
        key=lambda scenario: (
            scenario[
                "combination_size"
            ],
            scenario[
                "scenario_id"
            ],
        )
    )

    return scenarios

def build_type_scenario(
    base_components,
    feature_components=(),
) -> Dict[str, Any]:
    """
    Build one exact canonical type-level scenario.

    Unlike enumerate_type_scenarios(), this function does not
    enumerate the complete scenario space.

    It validates only the combination explicitly requested
    by the user.

    Parameters
    ----------
    base_components:
        Requested mathematical base components.

        Example:
            ["arch"]

    feature_components:
        Requested overlay features.

        Example:
            ["point_anomaly", "mean_shift"]

    Returns
    -------
    dict
        Canonical type-level scenario.

    Raises
    ------
    ValueError
        If the requested combination is invalid according
        to the canonical rule system.
    """

    report = validate_requested_combination(
        base_components=base_components,
        feature_components=feature_components,
    )

    report.raise_for_errors()

    combination_size = (
        len(report.base_components)
        +
        len(report.feature_components)
    )

    scenario_name = "__".join(
        report.base_components
        +
        report.feature_components
    )

    scenario_id = (
        f"{combination_size}way__"
        f"{scenario_name}"
    )

    return {
        "scenario_id": scenario_id,
        "name": scenario_name,
        "combination_size": combination_size,

        "base_components": list(
            report.base_components
        ),

        "base_families": list(
            report.base_families
        ),

        "feature_components": list(
            report.feature_components
        ),

        "feature_families": list(
            report.feature_families
        ),

        "composition_steps": list(
            report.composition_steps
        ),
    }


# ============================================================================
# SUMMARY
# ============================================================================

def count_type_scenarios(
    min_size: int = 1,
    max_size: int = 5,
) -> Dict[int, int]:
    """
    Return the number of valid scenarios for each size.
    """

    scenarios = enumerate_type_scenarios(
        min_size=min_size,
        max_size=max_size,
    )

    counts = {
        size: 0
        for size in range(
            min_size,
            max_size + 1,
        )
    }

    for scenario in scenarios:
        size = scenario[
            "combination_size"
        ]

        counts[size] += 1

    return counts


# ============================================================================
# CATEGORICAL PARAMETER LOADING
# ============================================================================

def load_categorical_params(
    config_dir=None,
) -> Dict[str, Any]:
    """
    Load categorical_params.json.

    Parameters
    ----------
    config_dir:
        Optional config directory.

        If omitted, the default path is:
            betise/config/categorical_params.json

    Returns
    -------
    dict
        Parsed categorical configuration.
    """

    if config_dir is None:
        config_dir = (
            Path(__file__).resolve().parent
            / "config"
        )
    else:
        config_dir = Path(
            config_dir
        )

    config_path = (
        config_dir
        / "categorical_params.json"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            "categorical_params.json "
            f"was not found at: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    return config


# ============================================================================
# CATEGORICAL AXIS EXPANSION
# ============================================================================

def _expand_axes(
    axes: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    """
    Expand categorical axes using Cartesian product.

    Example
    -------
    {
        "direction": ["up", "down"],
        "location": ["left", "center", "right"]
    }

    becomes six dictionaries.
    """

    if not axes:
        return [{}]

    keys = list(
        axes.keys()
    )

    value_lists = [
        axes[key]
        for key in keys
    ]

    expanded = []

    for values in product(
        *value_lists
    ):
        expanded.append(
            dict(
                zip(
                    keys,
                    values,
                )
            )
        )

    return expanded


# ============================================================================
# FEATURE VARIANT EXPANSION
# ============================================================================

def build_feature_variants(
    categorical_config=None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Expand compact categorical feature definitions
    into concrete variants.

    Returns
    -------
    dict

    Example
    -------
    {
        "linear_trend": [
            {
                "variant_id": "...",
                "params": {
                    "direction": "up"
                }
            },
            ...
        ]
    }
    """

    if categorical_config is None:
        categorical_config = (
            load_categorical_params()
        )

    feature_configs = (
        categorical_config.get(
            "features",
            {},
        )
    )

    all_variants = {}

    for feature_name, feature_config in (
        feature_configs.items()
    ):

        feature_variants = []

        # ------------------------------------------------------------
        # Simple feature:
        #
        # "linear_trend": {
        #     "axes": {...}
        # }
        # ------------------------------------------------------------

        if "cases" not in feature_config:

            fixed_params = (
                feature_config.get(
                    "fixed",
                    {},
                )
            )

            axes = (
                feature_config.get(
                    "axes",
                    {},
                )
            )

            axis_variants = (
                _expand_axes(
                    axes
                )
            )

            for axis_params in axis_variants:

                params = {
                    **fixed_params,
                    **axis_params,
                }

                feature_variants.append(
                    {
                        "variant_id": (
                            _make_variant_id(
                                feature_name,
                                params,
                            )
                        ),
                        "params": params,
                    }
                )

        # ------------------------------------------------------------
        # Feature with cases:
        #
        # "mean_shift": {
        #     "cases": {
        #         "single": {...},
        #         "multiple": {...}
        #     }
        # }
        # ------------------------------------------------------------

        else:

            for case_name, case_config in (
                feature_config[
                    "cases"
                ].items()
            ):

                fixed_params = (
                    case_config.get(
                        "fixed",
                        {},
                    )
                )

                axes = (
                    case_config.get(
                        "axes",
                        {},
                    )
                )

                axis_variants = (
                    _expand_axes(
                        axes
                    )
                )

                for axis_params in (
                    axis_variants
                ):

                    params = {
                        **fixed_params,
                        **axis_params,
                    }

                    feature_variants.append(
                        {
                            "variant_id": (
                                _make_variant_id(
                                    feature_name,
                                    params,
                                )
                            ),
                            "case": case_name,
                            "params": params,
                        }
                    )

        all_variants[
            feature_name
        ] = feature_variants

    return all_variants


# ============================================================================
# VARIANT IDS
# ============================================================================

def _make_variant_id(
    feature_name: str,
    params: Dict[str, Any],
) -> str:
    """
    Create a deterministic readable categorical variant ID.
    """

    parts = [
        feature_name
    ]

    for key in sorted(
        params.keys()
    ):

        value = params[key]

        if isinstance(
            value,
            bool,
        ):
            value = str(
                value
            ).lower()

        parts.append(
            f"{key}-{value}"
        )

    return "__".join(
        parts
    )


# ============================================================================
# CATEGORICAL SUMMARY
# ============================================================================

def count_feature_variants(
    categorical_config=None,
) -> Dict[str, int]:
    """
    Return categorical variant count per feature.
    """

    variants = (
        build_feature_variants(
            categorical_config
        )
    )

    return {
        feature_name: len(
            feature_variants
        )
        for (
            feature_name,
            feature_variants
        ) in variants.items()
    }

# ============================================================================
# SCENARIO CATEGORICAL EXPANSION
# ============================================================================

def count_scenario_categorical_variants(
    scenario: Dict[str, Any],
    feature_variants=None,
) -> int:
    """
    Count how many categorical realizations belong to one type-level scenario.

    Examples
    --------
    ar
        -> 1

    ar + linear_trend
        -> 2

    ar + quadratic_trend
        -> 6

    ar + linear_trend + mean_shift
        -> 2 * 9 = 18
    """

    if feature_variants is None:
        feature_variants = (
            build_feature_variants()
        )

    features = scenario.get(
        "feature_components",
        [],
    )

    # Base-only scenario:
    # no categorical overlay variation.
    if not features:
        return 1

    total = 1

    for feature_name in features:

        if feature_name not in feature_variants:
            raise KeyError(
                "No categorical definition found for "
                f"feature: {feature_name}"
            )

        total *= len(
            feature_variants[
                feature_name
            ]
        )

    return total

def _materialize_variant_combination(
    scenario: Dict[str, Any],
    active_features: List[str],
    variant_combination,
) -> Dict[str, Any]:
    """
    Build one concrete categorical scenario from a type-level scenario
    and one selected variant for each active feature.
    """

    feature_overrides = {}
    categorical_variant_ids = {}

    for (
        feature_name,
        variant,
    ) in zip(
        active_features,
        variant_combination,
    ):

        feature_overrides[
            feature_name
        ] = dict(
            variant[
                "params"
            ]
        )

        categorical_variant_ids[
            feature_name
        ] = variant[
            "variant_id"
        ]

    materialized = dict(
        scenario
    )

    materialized[
        "feature_overrides"
    ] = feature_overrides

    materialized[
        "categorical_variant_ids"
    ] = categorical_variant_ids

    if categorical_variant_ids:

        variant_suffix = "__".join(
            categorical_variant_ids[
                feature_name
            ]
            for feature_name
            in active_features
        )

        materialized[
            "materialized_scenario_id"
        ] = (
            scenario[
                "scenario_id"
            ]
            + "__"
            + variant_suffix
        )

    else:

        materialized[
            "materialized_scenario_id"
        ] = scenario[
            "scenario_id"
        ]

    return materialized

def iter_materialized_scenarios(
    type_scenarios=None,
    categorical_config=None,
    categorical_mode: str = "sampled",
    seed: int = 42,
    variants_per_type: int = 1,
):
    """
    Lazily expand type-level scenarios into categorical recipes.

    Parameters
    ----------
    type_scenarios:
        Type-level scenarios produced by enumerate_type_scenarios().

    categorical_config:
        Parsed categorical_params.json configuration.

    categorical_mode:
        "all"
            Use every categorical combination for each type scenario.

        "sampled"
            Select a limited number of categorical recipes
            for each type scenario.

    seed:
        Random seed used for categorical selection.

    variants_per_type:
        Number of categorical recipes selected for each type scenario
        when categorical_mode="sampled".

        If a type has fewer possible categorical recipes than this value,
        all available recipes are used.

    Yields
    ------
    dict
        Materialized categorical scenario.
    """

    categorical_mode = (
        categorical_mode.lower()
    )

    valid_modes = {
        "all",
        "sampled",
    }

    if categorical_mode not in valid_modes:
        raise ValueError(
            "Unknown categorical_mode: "
            f"{categorical_mode}. "
            "Expected 'all' or 'sampled'."
        )

    if variants_per_type <= 0:
        raise ValueError(
            "variants_per_type must be positive."
        )

    if type_scenarios is None:
        type_scenarios = (
            enumerate_type_scenarios()
        )

    feature_variants = (
        build_feature_variants(
            categorical_config
        )
    )

    rng = random.Random(
        seed
    )

    for scenario in type_scenarios:

        active_features = scenario.get(
            "feature_components",
            [],
        )

        # ============================================================
        # BASE-ONLY SCENARIO
        # ============================================================

        if not active_features:

            yield (
                _materialize_variant_combination(
                    scenario,
                    [],
                    [],
                )
            )

            continue

        # ============================================================
        # COLLECT VARIANT LISTS
        # ============================================================

        variant_lists = []

        for feature_name in active_features:

            if feature_name not in feature_variants:
                raise KeyError(
                    "No categorical definition found "
                    f"for feature: {feature_name}"
                )

            variants = (
                feature_variants[
                    feature_name
                ]
            )

            if not variants:
                raise ValueError(
                    "Feature has zero categorical variants: "
                    f"{feature_name}"
                )

            variant_lists.append(
                variants
            )

        # ============================================================
        # ALL
        # ============================================================

        if categorical_mode == "all":

            selected_combinations = (
                product(
                    *variant_lists
                )
            )

            for variant_combination in (
                selected_combinations
            ):

                yield (
                    _materialize_variant_combination(
                        scenario,
                        active_features,
                        variant_combination,
                    )
                )

            continue

        # ============================================================
        # SAMPLED
        # ============================================================

        total_possible = 1

        for variants in variant_lists:
            total_possible *= len(
                variants
            )

        target_count = min(
            variants_per_type,
            total_possible,
        )

        # Keep samples unique inside the same type scenario.
        selected_index_combinations = set()

        while (
            len(
                selected_index_combinations
            )
            < target_count
        ):

            index_combination = tuple(
                rng.randrange(
                    len(variants)
                )
                for variants
                in variant_lists
            )

            selected_index_combinations.add(
                index_combination
            )

        for index_combination in (
            selected_index_combinations
        ):

            variant_combination = tuple(
                variant_lists[
                    feature_index
                ][
                    variant_index
                ]
                for (
                    feature_index,
                    variant_index,
                ) in enumerate(
                    index_combination
                )
            )

            yield (
                _materialize_variant_combination(
                    scenario,
                    active_features,
                    variant_combination,
                )
            )

# ============================================================================
# EXPANDED SCENARIO COUNTS
# ============================================================================

def count_materialized_scenarios(
    min_size: int = 1,
    max_size: int = 5,
) -> Dict[int, int]:
    """
    Count categorical scenario realizations without materializing them.

    This uses multiplication of feature variant counts instead of creating
    the full Cartesian scenario space.
    """

    type_scenarios = (
        enumerate_type_scenarios(
            min_size=min_size,
            max_size=max_size,
        )
    )

    feature_variants = (
        build_feature_variants()
    )

    counts = {
        size: 0
        for size in range(
            min_size,
            max_size + 1,
        )
    }

    for scenario in type_scenarios:

        size = scenario[
            "combination_size"
        ]

        variant_count = (
            count_scenario_categorical_variants(
                scenario,
                feature_variants=(
                    feature_variants
                ),
            )
        )

        counts[
            size
        ] += variant_count

    return counts

# ============================================================================
# GENERATION ADAPTER
# ============================================================================

def _normalize_feature_override_for_generation(
    feature_name: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert categorical scenario parameters into the config format
    expected by dataset_generation.apply_feature().

    categorical_params.json describes semantic variants.
    This adapter translates those variants to generator arguments.
    """

    normalized = dict(params)

    # ------------------------------------------------------------------
    # Mean / variance shift
    #
    # For multiple shifts, the existing generator already samples
    # random directions internally.
    # ------------------------------------------------------------------

    if feature_name in {
        "mean_shift",
        "variance_shift",
    }:
        if normalized.get("mode") == "multiple":
            normalized.pop(
                "direction_strategy",
                None,
            )
            normalized.pop(
                "location",
                None,
            )

    # ------------------------------------------------------------------
    # Point anomaly
    #
    # Multiple point anomalies already determine their count
    # automatically in the generator.
    # ------------------------------------------------------------------

    elif feature_name == "point_anomaly":

        normalized.pop(
            "count_strategy",
            None,
        )

        if normalized.get("mode") == "multiple":
            normalized.pop(
                "location",
                None,
            )
            normalized.pop(
                "num_anomalies",
                None,
            )
            normalized.pop(
                "is_spike",
                None,
            )

    # ------------------------------------------------------------------
    # Collective anomaly
    #
    # categorical_params.json uses shape_strategy.
    # apply_feature() currently expects anomaly_shapes.
    # ------------------------------------------------------------------

    elif feature_name == "collective_anomaly":

        shape_strategy = normalized.pop(
            "shape_strategy",
            None,
        )

        if shape_strategy is not None:
            normalized[
                "anomaly_shapes"
            ] = shape_strategy

        if normalized.get("mode") == "multiple":
            normalized.pop(
                "location",
                None,
            )

    # ------------------------------------------------------------------
    # Contextual anomaly
    # ------------------------------------------------------------------

    elif feature_name == "contextual_anomaly":

        if normalized.get("mode") == "multiple":
            normalized.pop(
                "location",
                None,
            )

    # ------------------------------------------------------------------
    # Trend shift
    # ------------------------------------------------------------------

    elif feature_name == "trend_shift":

        if normalized.get("mode") == "multiple":
            normalized.pop(
                "location",
                None,
            )

    return normalized


def to_generation_composition(
    materialized_scenario: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert a materialized scenario into the composition schema expected by
    full_dataset_generation.generate_full_series().

    This is the boundary between:

        scenario definition

    and:

        actual time-series generation.
    """

    feature_overrides = {}

    for feature_name, params in (
        materialized_scenario.get(
            "feature_overrides",
            {},
        ).items()
    ):

        feature_overrides[
            feature_name
        ] = (
            _normalize_feature_override_for_generation(
                feature_name,
                params,
            )
        )

    return {
        "id": materialized_scenario[
            "materialized_scenario_id"
        ],

        "name": materialized_scenario[
            "name"
        ],

        "group": (
            f"{materialized_scenario['combination_size']}-way"
        ),

        "base_components": list(
            materialized_scenario[
                "base_components"
            ]
        ),

        # IMPORTANT:
        # generate_full_series() currently expects "features",
        # not "feature_components".
        "features": list(
            materialized_scenario[
                "feature_components"
            ]
        ),

        "feature_overrides": (
            feature_overrides
        ),

        "combination_size": (
            materialized_scenario[
                "combination_size"
            ]
        ),

        "type_scenario_id": (
            materialized_scenario[
                "scenario_id"
            ]
        ),

        "categorical_variant_ids": dict(
            materialized_scenario.get(
                "categorical_variant_ids",
                {},
            )
        ),
    }


def count_categorical_recipes(
    min_size: int = 1,
    max_size: int = 5,
    categorical_mode: str = "sampled",
    variants_per_type: int = 1,
) -> Dict[int, int]:
    """
    Count how many categorical recipes would be produced.

    Type-level scenarios are always fully enumerated.

    categorical_mode="all":
        Count the full categorical Cartesian product.

    categorical_mode="sampled":
        Count up to variants_per_type recipes per type scenario.
    """

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

    if variants_per_type <= 0:
        raise ValueError(
            "variants_per_type must be positive."
        )

    scenarios = (
        enumerate_type_scenarios(
            min_size=min_size,
            max_size=max_size,
        )
    )

    feature_variants = (
        build_feature_variants()
    )

    counts = {
        size: 0
        for size in range(
            min_size,
            max_size + 1,
        )
    }

    for scenario in scenarios:

        size = scenario[
            "combination_size"
        ]

        active_features = scenario[
            "feature_components"
        ]

        # Base-only scenario has exactly one categorical recipe.
        if not active_features:

            recipe_count = 1

        else:

            total_possible = (
                count_scenario_categorical_variants(
                    scenario,
                    feature_variants=(
                        feature_variants
                    ),
                )
            )

            if categorical_mode == "all":

                recipe_count = (
                    total_possible
                )

            else:

                recipe_count = min(
                    variants_per_type,
                    total_possible,
                )

        counts[
            size
        ] += recipe_count

    return counts