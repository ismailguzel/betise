import numpy as np
import pytest

import betise.scenario_generation as scenario_generation

from betise.scenario_builder import (
    enumerate_type_scenarios,
)


def _find_scenario(
    base_components,
    feature_components,
):
    """
    Find one exact canonical type-level scenario.
    """

    wanted_bases = set(
        base_components
    )

    wanted_features = set(
        feature_components
    )

    combination_size = (
        len(base_components)
        +
        len(feature_components)
    )

    scenarios = (
        enumerate_type_scenarios(
            min_size=combination_size,
            max_size=combination_size,
        )
    )

    for scenario in scenarios:

        if (
            set(
                scenario[
                    "base_components"
                ]
            )
            == wanted_bases
            and
            set(
                scenario[
                    "feature_components"
                ]
            )
            == wanted_features
        ):
            return scenario

    raise AssertionError(
        "Expected valid scenario was not found: "
        f"bases={base_components}, "
        f"features={feature_components}"
    )


@pytest.mark.parametrize(
    "base_components, feature_components",
    [
        (
            ["ar", "garch"],
            [],
        ),
        (
            [
                "arima",
                "multiple_seasonality",
            ],
            [],
        ),
        (
            [
                "sarma",
                "garch",
            ],
            [],
        ),

        (
            [
                "sarima",
                "garch",
            ],
            [],
        ),

        (
            [
                "arfima",
                "garch",
                "single_seasonality",
            ],
            [],
        ),
        (
            [
                "single_seasonality",
            ],
            [
                "contextual_anomaly",
            ],
        ),
        (
            [
                "ar",
            ],
            [
                "linear_trend",
                "trend_shift",
            ],
        ),
        (
            [
                "arch",
            ],
            [
                "mean_shift",
                "point_anomaly",
            ],
        ),
    ],
)
def test_targeted_composition_generates_valid_series(
    monkeypatch,
    base_components,
    feature_components,
):
    """
    Explicitly exercise representative canonical
    composition paths through the real generation engine.
    """

    scenario = _find_scenario(
        base_components,
        feature_components,
    )

    monkeypatch.setattr(
        scenario_generation,
        "enumerate_type_scenarios",
        lambda **kwargs: [
            scenario
        ],
    )

    generated = list(
        scenario_generation.iter_generated_series(
            min_size=(
                len(base_components)
                +
                len(feature_components)
            ),
            max_size=(
                len(base_components)
                +
                len(feature_components)
            ),
            categorical_mode="sampled",
            variants_per_type=1,
            series_per_recipe=1,
            length_range=(300, 300),
            seed=42,
            max_recipes=1,
        )
    )

    assert len(
        generated
    ) == 1

    dataframe, context = (
        generated[0]
    )

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

    assert set(
        context[
            "base_components"
        ]
    ) == set(
        base_components
    )

    assert set(
        context[
            "feature_components"
        ]
    ) == set(
        feature_components
    )

    assert (
        context[
            "combination_size"
        ]
        ==
        (
            len(base_components)
            +
            len(feature_components)
        )
    )