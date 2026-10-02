import json
from pathlib import Path

import pandas as pd
import pytest

from betise.scenario_output import (
    generate_dataset_to_parquet,
    generate_requested_dataset_to_parquet,
)

import betise.scenario_generation as scenario_generation


# ============================================================================
# SHARED TEST DATASET
# ============================================================================

@pytest.fixture(scope="module")
def generated_dataset(
    tmp_path_factory,
):
    """
    Generate a small temporary dataset once for this test module.

    Configuration:
        4 categorical recipes
        × 2 realizations per recipe
        = 8 actual time series

    shard_size=3 therefore:
        3 + 3 + 2 series
        = 3 parquet shards
    """

    output_dir = (
        tmp_path_factory.mktemp(
            "scenario_output"
        )
        / "dataset"
    )

    summary = (
        generate_dataset_to_parquet(
            output_dir=output_dir,
            min_size=3,
            max_size=3,
            categorical_mode="sampled",
            variants_per_type=1,
            series_per_recipe=2,
            length_range=(300, 300),
            seed=42,
            shard_size=3,
            max_recipes=4,
        )
    )

    return {
        "output_dir": output_dir,
        "summary": summary,
    }


# ============================================================================
# DIRECTORY / FILE STRUCTURE
# ============================================================================

def test_output_directory_structure(
    generated_dataset,
):
    output_dir = (
        generated_dataset[
            "output_dir"
        ]
    )

    assert output_dir.exists()

    assert (
        output_dir
        / "3-way"
    ).exists()

    assert (
        output_dir
        / "generation_summary.json"
    ).exists()


def test_expected_number_of_shards(
    generated_dataset,
):
    output_dir = (
        generated_dataset[
            "output_dir"
        ]
    )

    shard_files = sorted(
        (
            output_dir
            / "3-way"
        ).glob(
            "part-*.parquet"
        )
    )

    # 8 series with shard_size=3:
    # 3 + 3 + 2
    assert len(
        shard_files
    ) == 3


# ============================================================================
# SUMMARY CONTRACT
# ============================================================================

def test_generation_summary_contents(
    generated_dataset,
):
    output_dir = (
        generated_dataset[
            "output_dir"
        ]
    )

    summary_path = (
        output_dir
        / "generation_summary.json"
    )

    with open(
        summary_path,
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(
            file
        )

    assert summary[
        "min_size"
    ] == 3

    assert summary[
        "max_size"
    ] == 3

    assert summary[
        "categorical_mode"
    ] == "sampled"

    assert summary[
        "variants_per_type"
    ] == 1

    assert summary[
        "series_per_recipe"
    ] == 2

    assert summary[
        "max_recipes"
    ] == 4

    assert summary[
        "generated_recipe_counts"
    ] == {
        "3": 4
    }

    assert summary[
        "generated_series_counts"
    ] == {
        "3": 8
    }

    assert summary[
        "shard_counts"
    ] == {
        "3": 3
    }

    assert summary[
        "total_recipes"
    ] == 4

    assert summary[
        "total_series"
    ] == 8


# ============================================================================
# PARQUET CONTENT
# ============================================================================

def test_parquet_contains_required_context_columns(
    generated_dataset,
):
    output_dir = (
        generated_dataset[
            "output_dir"
        ]
    )

    first_shard = sorted(
        (
            output_dir
            / "3-way"
        ).glob(
            "part-*.parquet"
        )
    )[0]

    dataframe = (
        pd.read_parquet(
            first_shard
        )
    )

    required_columns = {
        "series_id",
        "time",
        "data",
        "type_scenario_id",
        "materialized_scenario_id",
        "combination_size",
        "recipe_index",
        "realization_index",
        "generation_seed",
        "categorical_variant_ids",
        "feature_overrides",
    }

    assert (
        required_columns
        <= set(
            dataframe.columns
        )
    )


def test_parquet_series_count_matches_summary(
    generated_dataset,
):
    output_dir = (
        generated_dataset[
            "output_dir"
        ]
    )

    shard_files = sorted(
        (
            output_dir
            / "3-way"
        ).glob(
            "part-*.parquet"
        )
    )

    all_series_ids = set()

    for shard_file in shard_files:

        dataframe = (
            pd.read_parquet(
                shard_file
            )
        )

        all_series_ids.update(
            dataframe[
                "series_id"
            ].unique()
        )

    assert len(
        all_series_ids
    ) == 8

    assert all_series_ids == {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    }


# ============================================================================
# SHARD SIZE SEMANTICS
# ============================================================================

def test_shard_size_counts_series_not_rows(
    generated_dataset,
):
    """
    shard_size=3 means three complete time series per shard,
    not three dataframe rows.
    """

    output_dir = (
        generated_dataset[
            "output_dir"
        ]
    )

    shard_files = sorted(
        (
            output_dir
            / "3-way"
        ).glob(
            "part-*.parquet"
        )
    )

    series_counts = []

    for shard_file in shard_files:

        dataframe = (
            pd.read_parquet(
                shard_file
            )
        )

        series_counts.append(
            dataframe[
                "series_id"
            ].nunique()
        )

    assert series_counts == [
        3,
        3,
        2,
    ]


# ============================================================================
# SERIES INTEGRITY
# ============================================================================

def test_each_series_is_complete(
    generated_dataset,
):
    """
    Since length_range=(300, 300),
    every saved time series must contain exactly 300 rows.
    """

    output_dir = (
        generated_dataset[
            "output_dir"
        ]
    )

    shard_files = sorted(
        (
            output_dir
            / "3-way"
        ).glob(
            "part-*.parquet"
        )
    )

    combined = pd.concat(
        [
            pd.read_parquet(
                shard_file
            )
            for shard_file
            in shard_files
        ],
        ignore_index=True,
    )

    lengths = (
        combined
        .groupby(
            "series_id"
        )
        .size()
    )

    assert len(
        lengths
    ) == 8

    assert (
        lengths == 300
    ).all()


# ============================================================================
# JSON METADATA CELLS
# ============================================================================

def test_json_context_columns_are_valid_json(
    generated_dataset,
):
    output_dir = (
        generated_dataset[
            "output_dir"
        ]
    )

    first_shard = sorted(
        (
            output_dir
            / "3-way"
        ).glob(
            "part-*.parquet"
        )
    )[0]

    dataframe = (
        pd.read_parquet(
            first_shard
        )
    )

    for column in [
        "categorical_variant_ids",
        "feature_overrides",
    ]:

        values = (
            dataframe[
                column
            ]
            .dropna()
            .unique()
        )

        for value in values:

            parsed = json.loads(
                value
            )

            assert isinstance(
                parsed,
                dict,
            )

def test_requested_output_generates_exact_series_count(
    tmp_path,
):
    output_dir = (
        tmp_path
        / "requested_dataset"
    )

    summary = (
        generate_requested_dataset_to_parquet(
            output_dir=output_dir,
            base_components=[
                "arch",
            ],
            feature_components=[
                "mean_shift",
                "point_anomaly",
            ],
            num_series=7,
            categorical_mode="sampled",
            variants_per_type=3,
            length_range=(300, 300),
            seed=42,
            shard_size=3,
        )
    )

    assert summary[
        "total_series"
    ] == 7

    assert summary[
        "requested_num_series"
    ] == 7

    assert summary[
        "total_recipes"
    ] == 3

    assert summary[
        "shard_count"
    ] == 3


def test_requested_output_contains_only_requested_combination(
    tmp_path,
):
    output_dir = (
        tmp_path
        / "requested_dataset"
    )

    summary = (
        generate_requested_dataset_to_parquet(
            output_dir=output_dir,
            base_components=[
                "arch",
            ],
            feature_components=[
                "point_anomaly",
                "mean_shift",
            ],
            num_series=5,
            categorical_mode="sampled",
            variants_per_type=2,
            length_range=(300, 300),
            seed=42,
            shard_size=10,
        )
    )

    assert summary[
        "base_components"
    ] == [
        "arch"
    ]

    assert summary[
        "feature_components"
    ] == [
        "mean_shift",
        "point_anomaly",
    ]

    assert summary[
        "combination_size"
    ] == 3

    shard_files = list(
        (
            output_dir
            / "3-way"
        ).glob(
            "*.parquet"
        )
    )

    assert len(
        shard_files
    ) == 1

    dataframe = pd.read_parquet(
        shard_files[0]
    )

    assert dataframe[
        "series_id"
    ].nunique() == 5

    assert len(
        dataframe
    ) == (
        5 * 300
    )


def test_requested_output_writes_summary_file(
    tmp_path,
):
    output_dir = (
        tmp_path
        / "requested_dataset"
    )

    generate_requested_dataset_to_parquet(
        output_dir=output_dir,
        base_components=[
            "ar",
        ],
        feature_components=[
            "linear_trend",
        ],
        num_series=4,
        categorical_mode="sampled",
        variants_per_type=2,
        length_range=(300, 300),
        seed=42,
        shard_size=10,
    )

    summary_path = (
        output_dir
        / "request_summary.json"
    )

    assert summary_path.exists()

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        summary = json.load(
            file
        )

    assert summary[
        "mode"
    ] == "requested"

    assert summary[
        "total_series"
    ] == 4

    assert summary[
        "base_components"
    ] == [
        "ar"
    ]

    assert summary[
        "feature_components"
    ] == [
        "linear_trend"
    ]