"""Canonical multi-base full-dataset generation for BeTiSe.

This module is deliberately separate from ``dataset_generation.py``.

- ``dataset_generation.py`` remains the legacy/simple pipeline.
- ``full_dataset_generation.py`` consumes ``full_dataset.json`` and supports
  canonical multi-base compositions.

Execution model
---------------
The config is declarative. ``base_components`` order is NOT execution order.

The planner resolves a valid composition into:

    optional volatility innovations
        -> core process
        -> optional external deterministic Fourier seasonality
        -> deterministic trend
        -> structural break
        -> anomaly

Deterministic SARMA/SARIMA are treated as internal seasonal cores because
their Fourier component is already generated inside the base generator.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import inspect
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from betise.config import load_full_dataset_config
from betise.core.generator import TimeSeriesGenerator
from betise.core.metadata import (
    attach_metadata_columns_to_df,
    create_metadata_record,
)
from betise.core.rules import (
    FRACTIONAL_BASE_SERIES,
    SEASONAL_BASE_SERIES,
    STOCHASTIC_BASE_SERIES,
    VOLATILITY_BASE_SERIES,
    base_family,
    validate_label_integrity,
    validate_metadata_contract,
    validate_numerical_output,
    validate_requested_combination,
)
from betise.dataset_generation import (
    _patch_pyarrow_unregister_extension_type,
    _sample_value,
    apply_feature,
    generate_base_series,
    populate_family_metadata,
    update_metadata,
)
from betise.utils.helpers import add_indices_column


EXTERNAL_FOURIER_BASES = {
    "single_seasonality",
    "multiple_seasonality",
}

INTERNAL_SEASONAL_CORES = {
    "sarma",
    "sarima",
}

DYNAMIC_CORE_FAMILIES = {
    "stationary",
    "stochastic",
    "fractional",
}

PRIMARY_LABELS = {
    "stationary": 0,
    "anomaly": 1,
    "trend": 2,
    "stochastic": 3,
    "seasonality": 4,
    "volatility": 5,
    "structural_break": 6,
}


# ============================================================================
# SMALL HELPERS
# ============================================================================

def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> Dict[str, Any]:
    result = deepcopy(dict(base))

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
            result[key] = deepcopy(value)

    return result


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    output = []

    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)

    return output


def _canonical_base_rank(component: str) -> Tuple[int, str]:
    if component in INTERNAL_SEASONAL_CORES:
        return 0, component

    family = base_family(component)

    if family in DYNAMIC_CORE_FAMILIES:
        return 0, component

    if family == "volatility":
        return 1, component

    if component in EXTERNAL_FOURIER_BASES:
        return 2, component

    return 9, component


def canonicalize_base_components(
    base_components: Sequence[str],
) -> List[str]:
    return sorted(
        _unique(base_components),
        key=_canonical_base_rank,
    )


def _sample_length(
    length_range: Sequence[int],
) -> int:
    if len(length_range) != 2:
        raise ValueError(
            "length_range must contain exactly [low, high]."
        )

    low = int(length_range[0])
    high = int(length_range[1])

    if low <= 0 or high < low:
        raise ValueError(
            f"Invalid length_range: {length_range}"
        )

    return int(
        np.random.randint(
            low,
            high + 1,
        )
    )


# ============================================================================
# GENERATION PLANNER
# ============================================================================

def build_base_generation_plan(
    base_components: Sequence[str],
) -> Dict[str, Any]:
    """Validate and resolve declarative base components into execution roles."""

    canonical_components = canonicalize_base_components(
        base_components
    )

    report = validate_requested_combination(
        base_components=canonical_components,
        feature_components=(),
    )
    report.raise_for_errors()

    dynamic_cores = [
        component
        for component in canonical_components
        if base_family(component) in DYNAMIC_CORE_FAMILIES
    ]

    internal_seasonal = [
        component
        for component in canonical_components
        if component in INTERNAL_SEASONAL_CORES
    ]

    volatility = [
        component
        for component in canonical_components
        if component in VOLATILITY_BASE_SERIES
    ]

    external_fourier = [
        component
        for component in canonical_components
        if component in EXTERNAL_FOURIER_BASES
    ]

    if len(dynamic_cores) > 1:
        raise ValueError(
            "Planner received more than one dynamic core: "
            f"{dynamic_cores}"
        )

    if len(internal_seasonal) > 1:
        raise ValueError(
            "Planner received more than one internal seasonal core: "
            f"{internal_seasonal}"
        )

    if dynamic_cores and internal_seasonal:
        raise ValueError(
            "A dynamic core and deterministic SARMA/SARIMA cannot both be "
            "execution cores in the same canonical composition."
        )

    if len(volatility) > 1:
        raise ValueError(
            "Only one volatility component is allowed."
        )

    if len(external_fourier) > 1:
        raise ValueError(
            "Only one external Fourier base subtype is allowed."
        )

    core = (
        dynamic_cores[0]
        if dynamic_cores
        else (
            internal_seasonal[0]
            if internal_seasonal
            else None
        )
    )

    volatility_component = (
        volatility[0]
        if volatility
        else None
    )

    external_fourier_component = (
        external_fourier[0]
        if external_fourier
        else None
    )

    if core is not None:
        primary_base = core
    elif volatility_component is not None:
        primary_base = volatility_component
    elif external_fourier_component is not None:
        primary_base = external_fourier_component
    else:
        raise ValueError(
            "No executable base component was resolved."
        )

    return {
        "base_components": canonical_components,
        "base_families": [
            base_family(component)
            for component in canonical_components
        ],
        "core": core,
        "volatility": volatility_component,
        "external_fourier": external_fourier_component,
        "primary_base": primary_base,
        "composition_steps": list(
            report.composition_steps
        ),
    }


# ============================================================================
# EXTERNAL FOURIER ADAPTER
# ============================================================================

def _choose_external_fourier_parameters(
    ts: TimeSeriesGenerator,
    seasonal_component: str,
    params_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve configured periods/amplitudes before composition."""

    seasonality_cfg = params_cfg.get(
        "seasonality",
        {},
    )

    if seasonal_component == "single_seasonality":
        cfg = seasonality_cfg.get(
            "single_seasonality",
            {},
        )

        period_cfg = cfg.get("period")
        period = None

        if isinstance(period_cfg, list):
            valid = ts.get_valid_calendar_periods(
                allowed_periods=period_cfg
            )

            if not valid:
                raise ValueError(
                    "No valid configured period found for "
                    "single_seasonality."
                )

            period = int(
                random.choice(valid)
            )

        elif period_cfg is not None:
            period = int(period_cfg)

        amplitude_cfg = cfg.get("amplitude")
        amplitude = (
            _sample_value(amplitude_cfg)
            if amplitude_cfg is not None
            else None
        )

        return {
            "kind": "single",
            "period": period,
            "periods": (
                [period]
                if period is not None
                else None
            ),
            "amplitude": amplitude,
            "amplitudes": (
                [amplitude]
                if amplitude is not None
                else None
            ),
            "num_components": 1,
        }

    if seasonal_component == "multiple_seasonality":
        cfg = seasonality_cfg.get(
            "multiple_seasonality",
            {},
        )

        num_components = int(
            cfg.get("num_components", 2)
        )

        period_candidates = cfg.get("periods")
        periods = None

        if period_candidates is not None:
            valid = ts.get_valid_calendar_periods(
                allowed_periods=period_candidates
            )

            if len(valid) < num_components:
                raise ValueError(
                    "multiple_seasonality needs "
                    f"{num_components} valid periods, "
                    f"but only {valid} are available."
                )

            periods = random.sample(
                valid,
                num_components,
            )

        amplitude_cfg = cfg.get("amplitudes")
        amplitudes = None

        if (
            amplitude_cfg is not None
            and periods is not None
        ):
            if (
                isinstance(amplitude_cfg, list)
                and len(amplitude_cfg) == 2
                and all(
                    isinstance(value, (int, float))
                    for value in amplitude_cfg
                )
            ):
                low, high = amplitude_cfg

                amplitudes = [
                    float(
                        np.random.uniform(
                            low,
                            high,
                        )
                    )
                    for _ in periods
                ]

        return {
            "kind": "multiple",
            "period": periods,
            "periods": periods,
            "amplitude": None,
            "amplitudes": amplitudes,
            "num_components": num_components,
        }

    raise ValueError(
        "External Fourier composition only supports "
        "'single_seasonality' and 'multiple_seasonality'."
    )


def _fourier_difference_order(
    background_component: str,
    background_info: Dict[str, Any],
) -> int:
    """Integer differencing order used only for Fourier calibration."""

    if background_component in {
        "random_walk",
        "random_walk_drift",
    }:
        return 1

    if background_component in {
        "ari",
        "ima",
        "arima",
    }:
        return int(
            background_info.get(
                "diff",
                1,
            )
        )

    # ARFIMA d is fractional and MUST NOT be passed as integer difference_order.
    return 0


def _compose_external_fourier(
    ts: TimeSeriesGenerator,
    background_df: pd.DataFrame,
    background_component: str,
    background_info: Dict[str, Any],
    seasonal_component: str,
    params_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Call the generator's validated Fourier-composition utility.

    The recent project version exposes ``compose_with_fourier_seasonality``.
    This adapter intentionally inspects its bound-method signature so the
    orchestrator stays tolerant to small keyword-name changes while preserving
    the generator's own validated calibration logic.
    """

    method = getattr(
        ts,
        "compose_with_fourier_seasonality",
        None,
    )

    if method is None:
        raise AttributeError(
            "TimeSeriesGenerator must define "
            "'compose_with_fourier_seasonality' for canonical multi-base "
            "seasonality composition."
        )

    params = _choose_external_fourier_parameters(
        ts,
        seasonal_component,
        params_cfg,
    )

    difference_order = _fourier_difference_order(
        background_component,
        background_info,
    )

    signature = inspect.signature(
        method
    )

    accepted = set(
        signature.parameters
    )

    kwargs: Dict[str, Any] = {}

    # Semantic selector.
    if "kind" in accepted:
        kwargs["kind"] = params["kind"]
    elif "seasonality_type" in accepted:
        kwargs["seasonality_type"] = params["kind"]
    elif "seasonality_kind" in accepted:
        kwargs["seasonality_kind"] = params["kind"]
    elif "mode" in accepted:
        kwargs["mode"] = params["kind"]

    # Period-related arguments.
    if "period" in accepted:
        kwargs["period"] = params["period"]

    if "periods" in accepted:
        kwargs["periods"] = params["periods"]

    if "num_components" in accepted:
        kwargs["num_components"] = params[
            "num_components"
        ]

    # Amplitude-related arguments.
    if "amplitude" in accepted:
        kwargs["amplitude"] = params[
            "amplitude"
        ]

    if "amplitudes" in accepted:
        kwargs["amplitudes"] = params[
            "amplitudes"
        ]

    # Validated calibration rule.
    if "difference_order" in accepted:
        kwargs["difference_order"] = (
            difference_order
        )

    try:
        result = method(
            background_df,
            **kwargs,
        )
    except TypeError as exc:
        raise TypeError(
            "Could not call compose_with_fourier_seasonality with the "
            "current generator signature. "
            f"Detected signature: {signature}. "
            f"Prepared kwargs: {sorted(kwargs)}"
        ) from exc

    if (
        not isinstance(result, tuple)
        or len(result) != 2
    ):
        raise TypeError(
            "compose_with_fourier_seasonality must return "
            "(dataframe, info)."
        )

    return result


# ============================================================================
# BASE COMPOSITION EXECUTION
# ============================================================================

def _sample_arfima_numseas(
    params_cfg: Dict[str, Any],
) -> int:
    arfima_params = params_cfg.get(
        "base",
        {},
    ).get(
        "arfima",
        {},
    )

    return int(
        _sample_value(
            arfima_params.get(
                "numseas",
                100,
            )
        )
    )


def generate_base_composition(
    ts: TimeSeriesGenerator,
    plan: Dict[str, Any],
    params_cfg: Dict[str, Any],
) -> Tuple[
    pd.DataFrame,
    List[Dict[str, Any]],
]:
    """Execute a validated base-composition plan."""

    core = plan["core"]
    volatility_component = plan["volatility"]
    external_fourier = plan["external_fourier"]

    component_records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Special pure external-Fourier standalone path.
    # ------------------------------------------------------------------

    if (
        core is None
        and volatility_component is None
        and external_fourier is not None
    ):
        df, seasonal_info = generate_base_series(
            ts,
            external_fourier,
            params_cfg,
        )

        component_records.append({
            "name": external_fourier,
            "family": "seasonality",
            "info": seasonal_info,
        })

        return df, component_records

    # ------------------------------------------------------------------
    # Pure volatility standalone OR volatility background for Fourier.
    # ------------------------------------------------------------------

    if core is None and volatility_component is not None:
        df, volatility_info = generate_base_series(
            ts,
            volatility_component,
            params_cfg,
        )

        component_records.append({
            "name": volatility_component,
            "family": "volatility",
            "info": volatility_info,
        })

        if external_fourier is None:
            return df, component_records

        df, seasonal_info = _compose_external_fourier(
            ts=ts,
            background_df=df,
            background_component=volatility_component,
            background_info=volatility_info,
            seasonal_component=external_fourier,
            params_cfg=params_cfg,
        )

        component_records.append({
            "name": external_fourier,
            "family": "seasonality",
            "info": seasonal_info,
        })

        return df, component_records

    # ------------------------------------------------------------------
    # Dynamic/internal-seasonal core, optionally driven by volatility.
    # ------------------------------------------------------------------

    if core is None:
        raise RuntimeError(
            "Planner produced no executable core/background."
        )

    innovations = None
    volatility_info = None
    arfima_numseas = None

    if core == "arfima":
        arfima_numseas = _sample_arfima_numseas(
            params_cfg
        )

    if volatility_component is not None:
        volatility_length = ts.length

        if core == "arfima":
            volatility_length = (
                ts.length
                + int(arfima_numseas)
            )

        volatility_ts = TimeSeriesGenerator(
            length=volatility_length
        )

        innovations, volatility_info = (
            volatility_ts.generate_volatility(
                kind=volatility_component,
                as_innovations=True,
            )
        )

    df, core_info = generate_base_series(
        ts,
        core,
        params_cfg,
        innovations=innovations,
        arfima_numseas=arfima_numseas,
    )

    component_records.append({
        "name": core,
        "family": base_family(core),
        "info": core_info,
    })

    if (
        volatility_component is not None
        and volatility_info is not None
    ):
        component_records.append({
            "name": volatility_component,
            "family": "volatility",
            "info": volatility_info,
        })

    if external_fourier is not None:
        df, seasonal_info = _compose_external_fourier(
            ts=ts,
            background_df=df,
            background_component=core,
            background_info=core_info,
            seasonal_component=external_fourier,
            params_cfg=params_cfg,
        )

        component_records.append({
            "name": external_fourier,
            "family": "seasonality",
            "info": seasonal_info,
        })

    return df, component_records


# ============================================================================
# FEATURE CONFIG RESOLUTION
# ============================================================================

def _resolve_feature_configs(
    full_cfg: Dict[str, Any],
    composition: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    defaults = full_cfg.get(
        "feature_defaults",
        {},
    )

    overrides = composition.get(
        "feature_overrides",
        {},
    )

    resolved = {}

    for feature in composition.get(
        "features",
        [],
    ):
        base_cfg = defaults.get(
            feature,
            {},
        )

        override_cfg = overrides.get(
            feature,
            {},
        )

        resolved[feature] = _deep_merge(
            base_cfg,
            override_cfg,
        )

    return resolved


# ============================================================================
# METADATA
# ============================================================================

def _initial_primary_metadata(
    df: pd.DataFrame,
    primary_base: str,
) -> Tuple[str, int]:
    family = base_family(
        primary_base
    )

    if family == "seasonality":
        return (
            "seasonality",
            PRIMARY_LABELS["seasonality"],
        )

    if family == "volatility":
        return (
            "volatility",
            PRIMARY_LABELS["volatility"],
        )

    if family == "stochastic":
        return (
            "stochastic",
            PRIMARY_LABELS["stochastic"],
        )

    is_stationary = (
        int(df["stationary"].iloc[0])
        if "stationary" in df.columns
        else 1
    )

    if is_stationary == 1:
        return (
            "stationary",
            PRIMARY_LABELS["stationary"],
        )

    return (
        "stochastic",
        PRIMARY_LABELS["stochastic"],
    )


def _create_record(
    *,
    series_id: int,
    length: int,
    label: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    return create_metadata_record(
        series_id=series_id,
        length=length,
        label=label,
        is_stationary=meta.get("is_stationary", 1),
        primary_category=meta.get("primary_category"),
        primary_label=meta.get("primary_label"),
        sub_category=meta.get("sub_category"),
        sub_label=meta.get("sub_label"),
        base_series=meta.get("base_series"),
        base_components=meta.get("base_components"),
        base_families=meta.get("base_families"),
        composition_steps=meta.get("composition_steps"),
        feature_components=meta.get("feature_components"),
        feature_families=meta.get("feature_families"),
        feature_infos=meta.get("feature_infos"),
        ar_order=meta.get("ar_order"),
        ma_order=meta.get("ma_order"),
        ar_coefs=meta.get("ar_coefs"),
        ma_coefs=meta.get("ma_coefs"),
        trend_type=meta.get("trend_type"),
        trend_slope=meta.get("trend_slope"),
        trend_intercept=meta.get("trend_intercept"),
        trend_coef_a=meta.get("trend_coef_a"),
        trend_coef_b=meta.get("trend_coef_b"),
        trend_coef_c=meta.get("trend_coef_c"),
        trend_damping_rate=meta.get("trend_damping_rate"),
        stochastic_type=meta.get("stochastic_type"),
        difference=meta.get("difference"),
        drift_value=meta.get("drift_value"),
        is_seasonal=meta.get("is_seasonal"),
        seasonality_type=meta.get("seasonality_type"),
        seasonality_periods=meta.get("seasonality_periods"),
        seasonality_amplitudes=meta.get("seasonality_amplitudes"),
        seasonal_ar_order=meta.get("seasonal_ar_order"),
        seasonal_ma_order=meta.get("seasonal_ma_order"),
        seasonal_ar_coefs=meta.get("seasonal_ar_coefs"),
        seasonal_ma_coefs=meta.get("seasonal_ma_coefs"),
        seasonal_difference=meta.get("seasonal_difference"),
        seasonality_period_meanings=meta.get(
            "seasonality_period_meanings"
        ),
        num_harmonics=meta.get("num_harmonics"),
        fourier_coefficients=meta.get(
            "fourier_coefficients"
        ),
        seasonal_unit_root=meta.get(
            "seasonal_unit_root"
        ),
        seasonality_scale_factor=meta.get(
            "seasonality_scale_factor"
        ),
        seasonality_strength=meta.get(
            "seasonality_strength"
        ),
        seasonality_period_balance_factors=meta.get(
            "seasonality_period_balance_factors"
        ),
        seasonality_calibration_difference_order=meta.get(
            "seasonality_calibration_difference_order"
        ),
        seasonal_initial_std=meta.get(
            "seasonal_initial_std"
        ),
        volatility_type=meta.get("volatility_type"),
        volatility_alpha=meta.get(
            "volatility_alpha"
        ),
        volatility_beta=meta.get(
            "volatility_beta"
        ),
        volatility_omega=meta.get(
            "volatility_omega"
        ),
        volatility_theta=meta.get(
            "volatility_theta"
        ),
        volatility_lambda=meta.get(
            "volatility_lambda"
        ),
        volatility_gamma=meta.get(
            "volatility_gamma"
        ),
        volatility_delta=meta.get(
            "volatility_delta"
        ),
        fractional_type=meta.get(
            "fractional_type"
        ),
        fractional_integrated=meta.get(
            "fractional_integrated"
        ),
        long_memory=meta.get("long_memory"),
        d_parameter=meta.get("d_parameter"),
        anomaly_type=meta.get("anomaly_type"),
        anomaly_count=meta.get("anomaly_count"),
        anomaly_indices=meta.get(
            "anomaly_indices"
        ),
        break_type=meta.get("break_type"),
        break_count=meta.get("break_count"),
        break_indices=meta.get(
            "break_indices"
        ),
        break_magnitudes=meta.get(
            "break_magnitudes"
        ),
        trend_shift_change_types=meta.get(
            "trend_shift_change_types"
        ),
        location_point=meta.get(
            "location_point"
        ),
        location_collective=meta.get(
            "location_collective"
        ),
        location_mean_shift=meta.get(
            "location_mean_shift"
        ),
        location_variance_shift=meta.get(
            "location_variance_shift"
        ),
        location_trend_shift=meta.get(
            "location_trend_shift"
        ),
        location_contextual=meta.get(
            "location_contextual"
        ),
    )


# ============================================================================
# ONE SERIES
# ============================================================================

def generate_full_series(
    *,
    composition: Dict[str, Any],
    full_cfg: Dict[str, Any],
    params_cfg: Dict[str, Any],
    series_id: int,
    length: int,
) -> pd.DataFrame:
    base_components = canonicalize_base_components(
        composition["base_components"]
    )

    requested_features = list(
        composition.get(
            "features",
            [],
        )
    )

    rule_report = validate_requested_combination(
        base_components=base_components,
        feature_components=requested_features,
    )
    rule_report.raise_for_errors()

    plan = build_base_generation_plan(
        base_components
    )

    # Use one generator object for this final series.
    ts = TimeSeriesGenerator(
        length=length
    )

    df, component_records = generate_base_composition(
        ts=ts,
        plan=plan,
        params_cfg=params_cfg,
    )

    state: Dict[str, Any] = {
        "seasonal_period": None,
        "seasonal_info": None,
    }

    primary_category, primary_label = (
        _initial_primary_metadata(
            df,
            plan["primary_base"],
        )
    )

    meta: Dict[str, Any] = {
        "is_stationary": (
            int(df["stationary"].iloc[0])
            if "stationary" in df.columns
            else 1
        ),
        "is_seasonal": (
            int(df["seasonal"].iloc[0])
            if "seasonal" in df.columns
            else 0
        ),
        "primary_category": primary_category,
        "primary_label": primary_label,
        "sub_category": plan["primary_base"],
        "sub_label": 0,
        "base_series": plan["primary_base"],
        "base_components": list(
            rule_report.base_components
        ),
        "base_families": list(
            rule_report.base_families
        ),
        "composition_steps": list(
            rule_report.composition_steps
        ),
    }

    for component in component_records:
        meta = populate_family_metadata(
            meta=meta,
            component_name=component["name"],
            family=component["family"],
            info=component["info"],
            state=state,
        )

    feature_cfgs = _resolve_feature_configs(
        full_cfg,
        composition,
    )

    feature_records = []

    # rules.py returns canonical order:
    # trend -> structural break -> anomaly.
    for feature_name in rule_report.feature_components:
        feature_cfg = feature_cfgs.get(
            feature_name,
            {},
        )

        df, info = apply_feature(
            ts,
            df,
            feature_name,
            feature_cfg,
            params_cfg,
            state,
        )

        feature_records.append({
            "name": feature_name,
            "family": (
                "trend"
                if feature_name.endswith("_trend")
                and feature_name != "trend_shift"
                else None
            ),
            "info": info,
        })

        meta = update_metadata(
            meta,
            feature_name,
            info,
            feature_cfg,
        )

    # Use rule_report rather than the provisional family value above.
    meta["feature_components"] = list(
        rule_report.feature_components
    )

    meta["feature_families"] = list(
        rule_report.feature_families
    )

    meta["feature_infos"] = {
        record["name"]: record["info"]
        for record in feature_records
    }

    meta["is_stationary"] = (
        int(df["stationary"].iloc[0])
        if "stationary" in df.columns
        else meta.get("is_stationary", 1)
    )

    meta["is_seasonal"] = (
        int(df["seasonal"].iloc[0])
        if "seasonal" in df.columns
        else meta.get("is_seasonal", 0)
    )

    # ------------------------------------------------------------------
    # Runtime validation gates
    # ------------------------------------------------------------------

    gate_5 = validate_numerical_output(
        data=df["data"],
        expected_length=length,
    )

    if not gate_5["passed"]:
        raise ValueError(
            "GATE_5_NUMERICAL_SANITY failed: "
            f"{gate_5['details']}"
        )

    gate_7 = validate_metadata_contract(
        metadata=meta,
        report=rule_report,
    )

    if not gate_7["passed"]:
        raise ValueError(
            "GATE_7_METADATA failed: "
            f"{gate_7['details']}"
        )

    gate_8 = validate_label_integrity(
        dataframe=df,
        feature_components=rule_report.feature_components,
    )

    if not gate_8["passed"]:
        raise ValueError(
            "GATE_8_LABEL_INTEGRITY failed: "
            f"{gate_8['details']}"
        )

    label = composition["name"]

    record = _create_record(
        series_id=series_id,
        length=length,
        label=label,
        meta=meta,
    )

    # Full-dataset-specific bookkeeping. attach_metadata_columns_to_df accepts
    # the metadata record as a mapping, so these become ordinary metadata cols.
    record["composition_id"] = composition["id"]
    record["composition_name"] = composition["name"]
    record["composition_group"] = composition.get(
        "group"
    )

    df_clean = df.drop(
        columns=[
            "stationary",
            "seasonal",
        ],
        errors="ignore",
    )

    return attach_metadata_columns_to_df(
        df_clean,
        record,
    )


# ============================================================================
# FULL DATASET
# ============================================================================

def _select_compositions(
    compositions: Sequence[Dict[str, Any]],
    *,
    composition_ids: Sequence[str] | None = None,
    composition_names: Sequence[str] | None = None,
    max_compositions: int | None = None,
) -> List[Dict[str, Any]]:
    selected = [
        composition
        for composition in compositions
        if composition.get("enabled", True)
    ]

    if composition_ids is not None:
        wanted = set(composition_ids)
        selected = [
            composition
            for composition in selected
            if composition["id"] in wanted
        ]

    if composition_names is not None:
        wanted = set(composition_names)
        selected = [
            composition
            for composition in selected
            if composition["name"] in wanted
        ]

    if max_compositions is not None:
        selected = selected[
            :int(max_compositions)
        ]

    if not selected:
        raise ValueError(
            "No full-dataset compositions matched the requested selection."
        )

    return selected


def _normalize_object_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    numeric_cols = {
        "d_parameter",
        "difference",
        "ar_order",
        "ma_order",
        "fractional_integrated",
        "long_memory",
        "seasonal_initial_std",
        "seasonality_scale_factor",
        "seasonality_strength",
        "seasonality_calibration_difference_order",
    }

    for column in dataframe.select_dtypes(
        include="object"
    ).columns:
        if column in numeric_cols:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )
        else:
            dataframe[column] = (
                dataframe[column].astype(str)
            )

    return dataframe


def generate_full_dataframe(
    cfg: Dict[str, Any],
    *,
    composition_ids: Sequence[str] | None = None,
    composition_names: Sequence[str] | None = None,
    max_compositions: int | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate selected compositions from canonical ``full_dataset.json``.

    With no selector, every enabled composition is generated.
    """

    _patch_pyarrow_unregister_extension_type()

    params_cfg = cfg["params"]
    full_cfg = cfg["full_dataset"]

    seed = int(
        full_cfg.get(
            "random_seed",
            42,
        )
    )

    random.seed(seed)
    np.random.seed(seed)

    defaults = full_cfg.get(
        "defaults",
        {},
    )

    output_cfg = full_cfg.get(
        "output",
        {},
    )

    default_count = int(
        defaults.get(
            "series_per_combination",
            1,
        )
    )

    default_length_range = defaults.get(
        "length_range",
        [300, 500],
    )

    include_indices = bool(
        output_cfg.get(
            "include_indices",
            True,
        )
    )

    compositions = _select_compositions(
        full_cfg["compositions"],
        composition_ids=composition_ids,
        composition_names=composition_names,
        max_compositions=max_compositions,
    )

    all_dfs: List[pd.DataFrame] = []
    series_id = 1

    for composition in compositions:
        count = int(
            composition.get(
                "series_per_combination",
                default_count,
            )
        )

        length_range = composition.get(
            "length_range",
            default_length_range,
        )

        if count <= 0:
            continue

        for _ in range(count):
            length = _sample_length(
                length_range
            )

            df = generate_full_series(
                composition=composition,
                full_cfg=full_cfg,
                params_cfg=params_cfg,
                series_id=series_id,
                length=length,
            )

            if include_indices:
                df = add_indices_column(
                    df
                )

            all_dfs.append(df)
            series_id += 1

    if not all_dfs:
        raise ValueError(
            "Selected compositions produced zero series."
        )

    combined_df = pd.concat(
        all_dfs,
        ignore_index=True,
    )

    combined_df = _normalize_object_columns(
        combined_df
    )

    context = {
        "num_compositions": len(compositions),
        "num_series": series_id - 1,
        "composition_ids": [
            composition["id"]
            for composition in compositions
        ],
        "composition_names": [
            composition["name"]
            for composition in compositions
        ],
        "output_dir": Path(
            output_cfg.get(
                "output_dir",
                "generated-dataset",
            )
        ),
        "output_name": output_cfg.get(
            "output_name",
            "full_dataset.parquet",
        ),
    }

    return combined_df, context


def run_full(
    cfg: Dict[str, Any] | None = None,
    *,
    composition_ids: Sequence[str] | None = None,
    composition_names: Sequence[str] | None = None,
    max_compositions: int | None = None,
) -> None:
    """Generate canonical full dataset and save it to parquet."""

    if cfg is None:
        cfg = load_full_dataset_config()

    dataframe, context = generate_full_dataframe(
        cfg,
        composition_ids=composition_ids,
        composition_names=composition_names,
        max_compositions=max_compositions,
    )

    output_dir: Path = context[
        "output_dir"
    ]

    output_name: str = context[
        "output_name"
    ]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / output_name
    )

    dataframe.to_parquet(
        output_path,
        index=False,
    )

    print("=" * 72)
    print("FULL DATASET GENERATION COMPLETE")
    print(f"Output       : {output_path.resolve()}")
    print(f"Compositions : {context['num_compositions']}")
    print(f"Series       : {context['num_series']}")
    print("=" * 72)


if __name__ == "__main__":
    run_full()
