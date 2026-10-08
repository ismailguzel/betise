"""
Parquet output utilities for scenario-driven BeTiSe generation.

Responsibilities
----------------
1. consume iter_generated_series(),
2. attach scenario-level bookkeeping metadata,
3. write data incrementally,
4. partition output by combination size,
5. avoid holding the whole dataset in memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from betise.scenario_generation import (
    iter_generated_series,
    iter_requested_series,
)




# ============================================================================
# METADATA HELPERS
# ============================================================================

def _json_cell(value: Any) -> str:
    """
    Serialize lists/dicts into a stable JSON string for parquet metadata cells.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
    )


def _attach_scenario_context(
    dataframe: pd.DataFrame,
    context: Dict[str, Any],
) -> pd.DataFrame:
    """
    Attach scenario-generation bookkeeping fields to every row
    of one generated time series.
    """

    df = dataframe.copy()

    df[
        "type_scenario_id"
    ] = context[
        "type_scenario_id"
    ]

    df[
        "materialized_scenario_id"
    ] = context[
        "materialized_scenario_id"
    ]

    df[
        "combination_size"
    ] = context[
        "combination_size"
    ]

    df[
        "recipe_index"
    ] = context[
        "recipe_index"
    ]

    df[
        "realization_index"
    ] = context[
        "realization_index"
    ]

    df[
        "generation_seed"
    ] = context[
        "seed"
    ]

    df[
        "categorical_variant_ids"
    ] = _json_cell(
        context.get(
            "categorical_variant_ids",
            {},
        )
    )

    df[
        "feature_overrides"
    ] = _json_cell(
        context.get(
            "feature_overrides",
            {},
        )
    )

    return df


# ============================================================================
# SHARD WRITER
# ============================================================================

def _write_shard(
    frames: List[pd.DataFrame],
    output_dir: Path,
    combination_size: int,
    shard_index: int,
) -> Path:
    """
    Write one parquet shard.
    """

    if not frames:
        raise ValueError(
            "Cannot write an empty shard."
        )

    size_dir = (
        output_dir
        / f"{combination_size}-way"
    )

    size_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Preserve the complete schema across all generated series.
    all_columns = list(
        dict.fromkeys(
            column
            for frame in frames
            for column in frame.columns
        )
    )

    # Pandas warns when concat receives columns that are entirely NA
    # in some frames. Such columns contribute no values, so remove them
    # temporarily from those individual frames.
    concat_frames = [
        frame.dropna(
            axis=1,
            how="all",
        )
        for frame in frames
    ]

    combined = pd.concat(
        concat_frames,
        ignore_index=True,
    )

    # Restore columns that were entirely NA across the whole shard.
    for column in all_columns:
        if column not in combined.columns:
            combined[column] = pd.NA

    # Restore deterministic column ordering.
    combined = combined.reindex(
        columns=all_columns
    )

    output_path = (
        size_dir
        / f"part-{shard_index:05d}.parquet"
    )

    combined.to_parquet(
        output_path,
        index=False,
    )

    return output_path


# ============================================================================
# MAIN DATASET WRITER
# ============================================================================

def generate_dataset_to_parquet(
    *,
    output_dir="generated-dataset/scenario_dataset",
    min_size: int = 1,
    max_size: int = 5,
    categorical_mode: str = "sampled",
    variants_per_type: int = 1,
    series_per_recipe: int = 1,
    length_range=(300, 500),
    length_category=None,
    exact_length=None,
    seed: int = 42,
    shard_size: int = 100,
    max_recipes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate scenario-driven BeTiSe data and write parquet shards.

    Parameters
    ----------
    output_dir:
        Root output directory.

    min_size, max_size:
        Allowed total combination size.

    policy:
        exhaustive, balanced, or sampled.

    samples_per_type:
        Number of categorical recipes selected per type scenario
        when policy="sampled".

    series_per_recipe:
        Number of numerical realizations generated from each
        categorical recipe.

    length_range:
        Inclusive series length interval.

    seed:
        Reproducibility seed.

    shard_size:
        Number of TIME SERIES stored in one parquet shard.

        Note:
        This is not number of dataframe rows.

    max_recipes:
        Optional debugging/smoke-test limit.

    Returns
    -------
    dict
        Generation summary.
    """

    if shard_size <= 0:
        raise ValueError(
            "shard_size must be positive."
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Separate buffers for each combination size.
    buffers = {
        size: []
        for size in range(
            min_size,
            max_size + 1,
        )
    }

    buffered_series = {
        size: 0
        for size in buffers
    }

    shard_indices = {
        size: 0
        for size in buffers
    }

    generated_series = {
        size: 0
        for size in buffers
    }

    generated_recipes = {
        size: set()
        for size in buffers
    }

    # ------------------------------------------------------------
    # Generation stream
    # ------------------------------------------------------------

    stream = iter_generated_series(
        min_size=min_size,
        max_size=max_size,
        categorical_mode=categorical_mode,
        variants_per_type=variants_per_type,
        series_per_recipe=series_per_recipe,
        length_range=length_range,
        length_category=length_category,
        exact_length=exact_length,
        seed=seed,
        max_recipes=max_recipes,
    )

    for dataframe, context in stream:

        size = int(
            context[
                "combination_size"
            ]
        )

        dataframe = (
            _attach_scenario_context(
                dataframe,
                context,
            )
        )

        buffers[
            size
        ].append(
            dataframe
        )

        buffered_series[
            size
        ] += 1

        generated_series[
            size
        ] += 1

        generated_recipes[
            size
        ].add(
            context[
                "materialized_scenario_id"
            ]
        )

        # --------------------------------------------------------
        # Flush shard
        # --------------------------------------------------------

        if (
            buffered_series[
                size
            ]
            >= shard_size
        ):

            _write_shard(
                frames=buffers[
                    size
                ],
                output_dir=output_dir,
                combination_size=size,
                shard_index=(
                    shard_indices[
                        size
                    ]
                ),
            )

            buffers[
                size
            ] = []

            buffered_series[
                size
            ] = 0

            shard_indices[
                size
            ] += 1

    # ------------------------------------------------------------
    # Flush remaining partial shards
    # ------------------------------------------------------------

    for size in sorted(
        buffers
    ):

        if not buffers[
            size
        ]:
            continue

        _write_shard(
            frames=buffers[
                size
            ],
            output_dir=output_dir,
            combination_size=size,
            shard_index=(
                shard_indices[
                    size
                ]
            ),
        )

        shard_indices[
            size
        ] += 1

        buffers[
            size
        ] = []

        buffered_series[
            size
        ] = 0

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    recipe_counts = {
        size: len(
            recipe_ids
        )
        for (
            size,
            recipe_ids
        ) in generated_recipes.items()
    }

    summary = {
        "min_size": min_size,
        "max_size": max_size,

        "categorical_mode": categorical_mode,
        "variants_per_type": variants_per_type,

        "series_per_recipe": (
            series_per_recipe
        ),

        "length_range": list(
            length_range
        ),

        "seed": seed,

        "shard_size": (
            shard_size
        ),

        "max_recipes": (
            max_recipes
        ),

        "generated_recipe_counts": {
            str(size): count
            for size, count
            in recipe_counts.items()
        },

        "generated_series_counts": {
            str(size): count
            for size, count
            in generated_series.items()
        },

        "shard_counts": {
            str(size): count
            for size, count
            in shard_indices.items()
        },

        "total_recipes": sum(
            recipe_counts.values()
        ),

        "total_series": sum(
            generated_series.values()
        ),
    }

    summary_path = (
        output_dir
        / "generation_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

    return summary

def generate_requested_dataset_to_parquet(
    *,
    output_dir,
    base_components,
    feature_components=(),
    num_series: int,
    categorical_mode: str = "sampled",
    variants_per_type: int = 1,
    length_range=(300, 500),
    length_category=None,
    exact_length=None,
    seed: int = 42,
    shard_size: int = 100,
) -> Dict[str, Any]:
    """
    Generate exactly num_series time series from one explicitly
    requested canonical combination and save them as parquet shards.

    This is the user-facing exact-combination output path.

    It is independent from the full scenario-space generation path.
    """

    if num_series <= 0:
        raise ValueError(
            "num_series must be positive."
        )

    if shard_size <= 0:
        raise ValueError(
            "shard_size must be positive."
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stream = iter_requested_series(
        base_components=base_components,
        feature_components=feature_components,
        num_series=num_series,
        categorical_mode=categorical_mode,
        variants_per_type=variants_per_type,
        length_range=length_range,
        length_category=length_category,
        exact_length=exact_length,
        seed=seed,
    )

    buffer = []

    buffered_series = 0
    shard_index = 0
    generated_series = 0

    generated_recipes = set()

    canonical_bases = None
    canonical_features = None
    combination_size = None

    for dataframe, context in stream:

        if canonical_bases is None:
            canonical_bases = list(
                context[
                    "base_components"
                ]
            )

            canonical_features = list(
                context[
                    "feature_components"
                ]
            )

            combination_size = int(
                context[
                    "combination_size"
                ]
            )

        dataframe = (
            _attach_scenario_context(
                dataframe,
                context,
            )
        )

        buffer.append(
            dataframe
        )

        buffered_series += 1
        generated_series += 1

        generated_recipes.add(
            context[
                "materialized_scenario_id"
            ]
        )

        if (
            buffered_series
            >= shard_size
        ):

            _write_shard(
                frames=buffer,
                output_dir=output_dir,
                combination_size=combination_size,
                shard_index=shard_index,
            )

            buffer = []
            buffered_series = 0
            shard_index += 1

    # ------------------------------------------------------------
    # Remaining partial shard
    # ------------------------------------------------------------

    if buffer:

        _write_shard(
            frames=buffer,
            output_dir=output_dir,
            combination_size=combination_size,
            shard_index=shard_index,
        )

        shard_index += 1

    if generated_series != num_series:
        raise RuntimeError(
            "Requested generation count mismatch: "
            f"requested={num_series}, "
            f"generated={generated_series}"
        )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary = {
        "mode": "requested",

        "base_components": (
            canonical_bases
        ),

        "feature_components": (
            canonical_features
        ),

        "combination_size": (
            combination_size
        ),

        "requested_num_series": (
            num_series
        ),

        "categorical_mode": (
            categorical_mode
        ),

        "variants_per_type": (
            variants_per_type
        ),

        "length_range": list(
            length_range
        ),

        "seed": seed,

        "shard_size": (
            shard_size
        ),

        "total_recipes": len(
            generated_recipes
        ),

        "total_series": (
            generated_series
        ),

        "shard_count": (
            shard_index
        ),
    }

    summary_path = (
        output_dir
        / "request_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

    return summary