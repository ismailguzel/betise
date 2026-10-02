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

from betise.scenario_builder import (count_categorical_recipes,)
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


def resolve_length_range(
    config: Dict[str, Any],
):

    """

    Resolve the user-facing length category to a numerical generation range.

    """

    if "length" not in config:
        raise ValueError("Generation config must contain 'length'. Expected one of: 'short', 'medium', 'long'.")


    length_name = str(config["length"]).lower()


    if length_name not in LENGTH_PRESETS:
        raise ValueError(f"Unknown length category: {length_name}. Expected 'short', 'medium', or 'long'.")


    return LENGTH_PRESETS[length_name]


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
        "length",
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

    resolve_length_range(config)


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
        "length",
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

    resolve_length_range(config)

    max_recipes = config.get("max_recipes")

    if (max_recipes is not None and int(max_recipes) <= 0):

        raise ValueError("max_recipes must be null or a positive integer.")


# ============================================================================
# GENERATION ESTIMATE
# ============================================================================

def build_generation_estimate(config: Dict[str, Any],
) -> Dict[str, Any]:

    """
    Estimate number of recipes and actual series before generation.
    """

    recipe_counts = (
        count_categorical_recipes(min_size=int(config["min_size"]),
            max_size=int(config["max_size"]),
            categorical_mode=str(config["categorical_mode"]),
            variants_per_type=int(config["variants_per_type"]),
        )
    )


    total_recipes = sum(recipe_counts.values())

    max_recipes = config.get("max_recipes")

    if max_recipes is not None:
        effective_recipes = min(total_recipes,int(max_recipes),)

    else:
        effective_recipes = (total_recipes)

    total_series = (effective_recipes * int(config["series_per_recipe"]))

    return {"recipe_counts": (recipe_counts),
            "total_available_recipes": (total_recipes),
            "effective_recipes": (effective_recipes),
            "estimated_series": (total_series),}


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

        length_range = (resolve_length_range(config))

        print("=" * 72)

        print("BeTiSe REQUESTED GENERATION")

        print("=" * 72)

        print("Base components: ",config["base_components"],)

        print("Features: ",config["features"],)

        print("Requested series: ",config["num_series"],)

        print("Categorical mode: ",config["categorical_mode"],)

        print("Variants per type: ",config["variants_per_type"],)

        print("Length category: ",config["length"],)

        print("Length range: ",length_range,)

        print("Output directory: ",config["output_dir"],)

        print("=" * 72)


        summary = (generate_requested_dataset_to_parquet(output_dir=(config["output_dir"]),
                base_components=(config["base_components"]),
                feature_components=(config["features"]),
                num_series=int(config["num_series"]),
                categorical_mode=str(config["categorical_mode"]),
                variants_per_type=int(config["variants_per_type"]),
                length_range=length_range, seed=int(config["seed"]),
                shard_size=int(config["shard_size"]),))

        print()

        print("=" * 72)

        print("REQUESTED GENERATION COMPLETE")

        print("=" * 72)

        print(json.dumps(summary,indent=2,ensure_ascii=False,))

        return summary

    validate_generation_config(config)

    length_range = (resolve_length_range(config))

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

    print("Length category: ",config["length"],)

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

    summary = (generate_dataset_to_parquet(
            output_dir=(config["output_dir"]),
            min_size=int(config["min_size"]),
            max_size=int(config["max_size"]),
            categorical_mode=str(config["categorical_mode"]),
            variants_per_type=int(config["variants_per_type"]),
            series_per_recipe=int(config["series_per_recipe"]),
            length_range=length_range,seed=int(config["seed"]),
            shard_size=int(config["shard_size"]),
            max_recipes=(config.get("max_recipes")),))


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

    custom_config = (sys.argv[1] if len(sys.argv) > 1 else None)

    run_generation(custom_config)