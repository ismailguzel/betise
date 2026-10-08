"""

User-facing runner for scenario-driven BeTiSe dataset generation.



Usage

-----

python -m betise.run_scenario_generation



Optional:

python -m betise.run_scenario_generation path/to/generation_config.json

"""


from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from betise.scenario_builder import (count_categorical_recipes)
from betise.scenario_output import (generate_dataset_to_parquet,generate_requested_dataset_to_parquet,)



LENGTH_PRESETS = {
    "short": (50, 100),
    "medium": (300, 500),
    "long": (1000, 10000),}


# ============================================================================
# CONFIG LOADING
# ============================================================================

def load_generation_config(config_path=None,
) -> Dict[str, Any]:

    """
    Load generation_config.json.
    """

    if config_path is None:
        config_path = (Path(__file__).resolve().parent / "config" / "generation_config.json")

    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError("Generation config was not found: "f"{config_path}")

    with config_path.open("r",encoding="utf-8") as file:
        config = json.load(file)

    return config


def resolve_length_settings(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve user-facing length settings.

    Exactly one of these must be provided:

        length_category:
            "short", "medium", or "long"

        length:
            exact positive integer series length
    """

    length_category = config.get(
        "length_category"
    )

    length = config.get(
        "length"
    )

    has_category = (
        length_category is not None
    )

    has_length = (
        length is not None
    )

    if has_category and has_length:
        raise ValueError(
            "Use either 'length_category' "
            "or 'length', not both."
        )

    if not has_category and not has_length:
        raise ValueError(
            "Either 'length_category' "
            "or 'length' must be provided."
        )

    # ---------------------------------------------------------
    # Named length category
    # ---------------------------------------------------------

    if has_category:

        category = str(
            length_category
        ).lower()

        if category not in LENGTH_PRESETS:
            raise ValueError(
                f"Unknown length_category: {category}. "
                "Expected 'short', 'medium', or 'long'."
            )

        return {
            "length_category": category,
            "length": None,
            "length_range": (
                LENGTH_PRESETS[
                    category
                ]
            ),
        }

    # ---------------------------------------------------------
    # Exact numeric length
    # ---------------------------------------------------------

    if (
        not isinstance(length, int)
        or isinstance(length, bool)
    ):
        raise ValueError(
            "'length' must be a positive integer."
        )

    exact_length = length

    if exact_length <= 0:
        raise ValueError(
            "'length' must be a positive integer."
        )

    return {
        "length_category": None,
        "length": exact_length,
        "length_range": (
            exact_length,
            exact_length,
        ),
    }


def resolve_length_range(
    config: Dict[str, Any],
):
    """
    Backward-compatible helper returning only the numeric range.
    """

    return resolve_length_settings(
        config
    )["length_range"]


def validate_requested_generation_config(
    config: Dict[str, Any],
) -> None:

    """

    Validate one user-requested exact combination.

    """

    required = {
        "output_dir",
        "base_components",
        "features",
        "num_series",
        "categorical_mode",
        "variants_per_type",
        "seed",
        "shard_size",}

    missing = (required - set(config))

    if missing:
        raise ValueError(f"Requested generation config is missing: {sorted(missing)}")



    if not isinstance(config["base_components"],list,):

        raise TypeError("base_components must be a list.")

    if not config["base_components"]:

        raise ValueError("base_components cannot be empty.")

    if not isinstance(config["features"],list,):

        raise TypeError("features must be a list.")

    if int(config["num_series"]) <= 0:

        raise ValueError("num_series must be positive.")

    categorical_mode = str(config["categorical_mode"]).lower()

    if categorical_mode not in {"all","sampled",}:

        raise ValueError("categorical_mode must be 'all' or 'sampled'.")

    if int(config["variants_per_type"]) <= 0:

        raise ValueError("variants_per_type must be positive.")

    if int(config["shard_size"]) <= 0:

        raise ValueError("shard_size must be positive.")

    resolve_length_settings(config)


# ============================================================================
# CONFIG VALIDATION
# ============================================================================

def validate_generation_config(config: Dict[str, Any],
) -> None:

    """
    Validate user-facing generation settings.
    """

    required = {
        "output_dir",
        "min_size",
        "max_size",
        "categorical_mode",
        "variants_per_type",
        "series_per_recipe",
        "seed",
        "shard_size",}


    missing = (required - set(config))

    if missing:
        raise ValueError(f"generation_config.json is missing: {sorted(missing)}")

    min_size = int(config["min_size"])
    max_size = int(config["max_size"])

    if min_size < 1:
        raise ValueError("min_size must be >= 1.")

    if max_size < min_size:
        raise ValueError("max_size must be >= min_size.")

    if max_size > 5:
        raise ValueError("Current canonical generation supports combination sizes up to 5.")

    categorical_mode = str(config["categorical_mode"]).lower()

    if categorical_mode not in {"all","sampled",}:

        raise ValueError("categorical_mode must be 'all' or 'sampled'.")


    if int(config["variants_per_type"]) <= 0:

        raise ValueError("variants_per_type must be positive.")

    if int(config["series_per_recipe"]) <= 0:

        raise ValueError("series_per_recipe must be positive.")

    if int(config["shard_size"]) <= 0:

        raise ValueError("shard_size must be positive.")

    resolve_length_settings(config)

    max_recipes = config.get("max_recipes")

    if (max_recipes is not None and int(max_recipes) <= 0):

        raise ValueError("max_recipes must be null or a positive integer.")


# ============================================================================
# GENERATION ESTIMATE
# ============================================================================

def build_generation_estimate(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Estimate recipe and series counts under the selected length policy."""

    length_settings = resolve_length_settings(config)

    recipe_counts = count_categorical_recipes(
        min_size=int(config["min_size"]),
        max_size=int(config["max_size"]),
        categorical_mode=str(config["categorical_mode"]),
        variants_per_type=int(config["variants_per_type"]),
        length_category=length_settings["length_category"],
        length=length_settings["length"],
    )

    total_recipes = sum(recipe_counts.values())
    max_recipes = config.get("max_recipes")

    if max_recipes is not None:
        effective_recipes = min(
            total_recipes,
            int(max_recipes),
        )
    else:
        effective_recipes = total_recipes

    total_series = (
        effective_recipes
        * int(config["series_per_recipe"])
    )

    return {
        "recipe_counts": recipe_counts,
        "total_available_recipes": total_recipes,
        "effective_recipes": effective_recipes,
        "estimated_series": total_series,
    }

# ============================================================================
# RUNNER
# ============================================================================

def run_generation(config_path=None,
):

    """
    Load config, validate it, show generation estimate, and start dataset generation.
    """

    config = (load_generation_config(config_path))

    mode = str(config.get("mode","full",)).lower()

    if mode == "requested":
        validate_requested_generation_config(config)
        length_settings = resolve_length_settings(config)

        length_settings = (
            resolve_length_settings(
                config
            )
        )

        length_range = (
            length_settings[
                "length_range"
            ]
        )

        print("=" * 72)

        print("BeTiSe REQUESTED GENERATION")

        print("=" * 72)

        print("Base components: ",config["base_components"],)

        print("Features: ",config["features"],)

        print("Requested series: ",config["num_series"],)

        print("Categorical mode: ",config["categorical_mode"],)

        print("Variants per type: ",config["variants_per_type"],)

        print(
            "Length category: ",
            length_settings[
                "length_category"
            ],
        )

        print(
            "Exact length: ",
            length_settings[
                "length"
            ],
        )

        print("Length range: ",length_range,)

        print("Output directory: ",config["output_dir"],)

        print("=" * 72)


        summary = generate_requested_dataset_to_parquet(
            output_dir=config["output_dir"],
            base_components=config["base_components"],
            feature_components=config["features"],
            num_series=int(config["num_series"]),
            categorical_mode=str(config["categorical_mode"]),
            variants_per_type=int(config["variants_per_type"]),
            length_range=length_range,
            length_category=length_settings["length_category"],
            exact_length=length_settings["length"],
            seed=int(config["seed"]),
            shard_size=int(config["shard_size"]),
        )

        print()

        print("=" * 72)

        print("REQUESTED GENERATION COMPLETE")

        print("=" * 72)

        print(json.dumps(summary,indent=2,ensure_ascii=False,))

        return summary

    validate_generation_config(config)

    length_settings = (
        resolve_length_settings(
            config
        )
    )

    length_range = (
        length_settings[
            "length_range"
        ]
    )

    estimate = (build_generation_estimate(config))



    print("=" * 72)

    print("BeTiSe SCENARIO GENERATION")

    print("=" * 72)

    print("Categorical mode: ",config["categorical_mode"],)

    print("Variants per type: ",config["variants_per_type"],)

    print("Combination sizes: ", f"{config['min_size']}–{config['max_size']}",)

    print("Available recipes: ",estimate["total_available_recipes"],)

    print("Recipes to generate: ",estimate["effective_recipes"],)

    print("Series per recipe: ",config["series_per_recipe"],)

    print("Estimated series: ",estimate["estimated_series"],)

    print(
        "Length category: ",
        length_settings[
            "length_category"
        ],
    )

    print(
        "Exact length: ",
        length_settings[
            "length"
        ],
    )

    print("Length range: ",length_range,)

    print("Output directory: ",config["output_dir"],)

    print("=" * 72)

    # ------------------------------------------------------------
    # Safety guard
    # ------------------------------------------------------------

    if (config["categorical_mode"] == "all" and estimate[ "estimated_series"] > 100000
        and not config.get("allow_large_run",False,)):

        raise RuntimeError(
            "Large exhaustive generation blocked."
            f"This configuration would generate approximately {estimate['estimated_series']} series. "
            "Set allow_large_run=true in generation_config.json if this is intentional.")


    # ------------------------------------------------------------
    # Actual generation
    # ------------------------------------------------------------

    summary = generate_dataset_to_parquet(
        output_dir=config["output_dir"],
        min_size=int(config["min_size"]),
        max_size=int(config["max_size"]),
        categorical_mode=str(config["categorical_mode"]),
        variants_per_type=int(config["variants_per_type"]),
        series_per_recipe=int(config["series_per_recipe"]),
        length_range=length_range,
        length_category=length_settings["length_category"],
        exact_length=length_settings["length"],
        seed=int(config["seed"]),
        shard_size=int(config["shard_size"]),
        max_recipes=config.get("max_recipes"),
    )


    print()

    print("=" * 72)

    print("GENERATION COMPLETE")

    print("=" * 72)

    print(json.dumps(summary,indent=2,ensure_ascii=False,))
    return summary


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    custom_config = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        run_generation(custom_config)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)