import numpy as np

from betise.scenario_generation import (
    iter_generated_series,
    iter_requested_series,
)

import betise.scenario_generation as scenario_generation

# ============================================================================
# BASIC GENERATION
# ============================================================================

def test_generated_series_count_respects_max_recipes():
    """
    5 categorical recipes × 1 realization
    must produce exactly 5 series.
    """

    generated = list(
        iter_generated_series(
            min_size=3,
            max_size=3,
            categorical_mode="sampled",
            variants_per_type=1,
            series_per_recipe=1,
            length_range=(300, 320),
            seed=42,
            max_recipes=5,
        )
    )

    assert len(
        generated
    ) == 5


def test_series_per_recipe_creates_multiple_realizations():
    """
    3 recipes × 2 numerical realizations
    must produce 6 actual time series.
    """

    generated = list(
        iter_generated_series(
            min_size=3,
            max_size=3,
            categorical_mode="sampled",
            variants_per_type=1,
            series_per_recipe=2,
            length_range=(300, 300),
            seed=42,
            max_recipes=3,
        )
    )

    assert len(
        generated
    ) == 6


# ============================================================================
# DATAFRAME CONTRACT
# ============================================================================

def test_generated_dataframe_length_and_values():
    generated = list(
        iter_generated_series(
            min_size=2,
            max_size=2,
            categorical_mode="sampled",
            variants_per_type=1,
            series_per_recipe=1,
            length_range=(300, 320),
            seed=42,
            max_recipes=3,
        )
    )

    for dataframe, context in generated:

        assert (
            300
            <= len(dataframe)
            <= 320
        )

        assert len(
            dataframe
        ) == context[
            "length"
        ]

        assert np.all(
            np.isfinite(
                dataframe[
                    "data"
                ].to_numpy()
            )
        )

        assert (
            "series_id"
            in dataframe.columns
        )

        assert (
            "time"
            in dataframe.columns
        )

        assert (
            "data"
            in dataframe.columns
        )


# ============================================================================
# CONTEXT CONTRACT
# ============================================================================

def test_generation_context_matches_combination_size():
    generated = list(
        iter_generated_series(
            min_size=3,
            max_size=3,
            categorical_mode="sampled",
            variants_per_type=1,
            series_per_recipe=1,
            length_range=(300, 300),
            seed=42,
            max_recipes=5,
        )
    )

    for _, context in generated:

        expected_size = (
            len(
                context[
                    "base_components"
                ]
            )
            +
            len(
                context[
                    "feature_components"
                ]
            )
        )

        assert expected_size == 3

        assert (
            context[
                "combination_size"
            ]
            == expected_size
        )


# ============================================================================
# RECIPE VS NUMERICAL REALIZATION
# ============================================================================

def test_same_recipe_produces_different_realizations():
    """
    series_per_recipe=2 means:

        same categorical recipe
        +
        two independent numerical realizations.
    """

    generated = list(
        iter_generated_series(
            min_size=3,
            max_size=3,
            categorical_mode="sampled",
            variants_per_type=1,
            series_per_recipe=2,
            length_range=(300, 300),
            seed=42,
            max_recipes=1,
        )
    )

    assert len(
        generated
    ) == 2

    df_a, context_a = (
        generated[0]
    )

    df_b, context_b = (
        generated[1]
    )

    # Same categorical recipe.
    assert (
        context_a[
            "materialized_scenario_id"
        ]
        ==
        context_b[
            "materialized_scenario_id"
        ]
    )

    assert (
        context_a[
            "categorical_variant_ids"
        ]
        ==
        context_b[
            "categorical_variant_ids"
        ]
    )

    # But different actual series.
    assert (
        context_a[
            "series_id"
        ]
        !=
        context_b[
            "series_id"
        ]
    )

    assert not np.array_equal(
        df_a[
            "data"
        ].to_numpy(),
        df_b[
            "data"
        ].to_numpy(),
    )


# ============================================================================
# SERIES IDS
# ============================================================================

def test_series_ids_are_sequential():
    generated = list(
        iter_generated_series(
            min_size=3,
            max_size=3,
            categorical_mode="sampled",
            variants_per_type=1,
            series_per_recipe=2,
            length_range=(300, 300),
            seed=42,
            max_recipes=3,
        )
    )

    series_ids = [
        context[
            "series_id"
        ]
        for _, context
        in generated
    ]

    assert series_ids == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def test_same_seed_is_reproducible():
    """
    The same generation request with the same seed
    should select the same recipes and generate the same data.
    """

    kwargs = {
        "min_size": 3,
        "max_size": 3,
        "categorical_mode": "sampled",
        "variants_per_type": 1,
        "series_per_recipe": 1,
        "length_range": (300, 300),
        "seed": 42,
        "max_recipes": 3,
    }

    first = list(
        iter_generated_series(
            **kwargs
        )
    )

    second = list(
        iter_generated_series(
            **kwargs
        )
    )

    assert len(
        first
    ) == len(
        second
    )

    for (
        df_a,
        context_a,
    ), (
        df_b,
        context_b,
    ) in zip(
        first,
        second,
    ):

        assert (
            context_a[
                "materialized_scenario_id"
            ]
            ==
            context_b[
                "materialized_scenario_id"
            ]
        )

        assert (
            context_a[
                "categorical_variant_ids"
            ]
            ==
            context_b[
                "categorical_variant_ids"
            ]
        )

        assert np.array_equal(
            df_a[
                "data"
            ].to_numpy(),
            df_b[
                "data"
            ].to_numpy(),
        )

def test_variants_per_type_is_forwarded_to_materializer(
    monkeypatch,
):
    """
    iter_generated_series() must forward variants_per_type
    to iter_materialized_scenarios().
    """

    captured = {}

    def fake_enumerate_type_scenarios(
        *,
        min_size,
        max_size,
        **kwargs,
    ):
        return []

    def fake_iter_materialized_scenarios(
        type_scenarios,
        categorical_mode,
        variants_per_type,
        seed,
        **kwargs,
    ):
        captured[
            "variants_per_type"
        ] = variants_per_type

        return iter(())

    monkeypatch.setattr(
        scenario_generation,
        "enumerate_type_scenarios",
        fake_enumerate_type_scenarios,
    )

    monkeypatch.setattr(
        scenario_generation,
        "iter_materialized_scenarios",
        fake_iter_materialized_scenarios,
    )

    generated = list(
        scenario_generation.iter_generated_series(
            min_size=3,
            max_size=3,
            categorical_mode="sampled",
            variants_per_type=7,
            series_per_recipe=1,
            length_range=(300, 300),
            seed=42,
        )
    )

    assert generated == []

    assert (
        captured[
            "variants_per_type"
        ]
        == 7
    )

def test_requested_generation_produces_exact_series_count():
    generated = list(
        iter_requested_series(
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
        )
    )

    assert len(
        generated
    ) == 7

    series_ids = [
        context[
            "series_id"
        ]
        for _, context
        in generated
    ]

    assert series_ids == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]


def test_requested_generation_uses_only_requested_combination():
    generated = list(
        iter_requested_series(
            base_components=[
                "arch",
            ],
            feature_components=[
                "mean_shift",
                "point_anomaly",
            ],
            num_series=5,
            categorical_mode="sampled",
            variants_per_type=2,
            length_range=(300, 300),
            seed=42,
        )
    )

    for dataframe, context in generated:

        assert context[
            "base_components"
        ] == [
            "arch"
        ]

        assert set(
            context[
                "feature_components"
            ]
        ) == {
            "mean_shift",
            "point_anomaly",
        }

        assert context[
            "combination_size"
        ] == 3

        assert len(
            dataframe
        ) == 300

        assert np.all(
            np.isfinite(
                dataframe[
                    "data"
                ].to_numpy()
            )
        )


def test_requested_generation_distributes_series_across_recipes():
    generated = list(
        iter_requested_series(
            base_components=[
                "ar",
            ],
            feature_components=[
                "quadratic_trend",
            ],
            num_series=7,
            categorical_mode="sampled",
            variants_per_type=3,
            length_range=(300, 300),
            seed=42,
        )
    )

    recipe_counts = {}

    for _, context in generated:

        recipe_id = context[
            "materialized_scenario_id"
        ]

        recipe_counts[
            recipe_id
        ] = (
            recipe_counts.get(
                recipe_id,
                0,
            )
            + 1
        )

    assert len(
        recipe_counts
    ) == 3

    assert sorted(
        recipe_counts.values(),
        reverse=True,
    ) == [
        3,
        2,
        2,
    ]


def test_requested_generation_rejects_invalid_combination():
    import pytest

    with pytest.raises(
        ValueError
    ):
        list(
            iter_requested_series(
                base_components=[
                    "garch",
                ],
                feature_components=[
                    "variance_shift",
                ],
                num_series=5,
                categorical_mode="sampled",
                variants_per_type=1,
                length_range=(300, 300),
                seed=42,
            )
        )