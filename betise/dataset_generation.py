"""Config-driven dataset generation pipeline.

Public API (re-exported from package root):
    run(cfg)                    -> None          (generate + save to parquet)
    generate_dataframe(cfg)     -> (DataFrame, context)   (in-memory)
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from betise.config import load_config
from betise.core.generator import TimeSeriesGenerator
from betise.core.metadata import (
    attach_metadata_columns_to_df,
    create_metadata_record,
)
from betise.core.rules import (
    STATIONARY_BASE_SERIES,
    STOCHASTIC_BASE_SERIES,
    SEASONAL_BASE_SERIES,
    VOLATILITY_BASE_SERIES,
    FRACTIONAL_BASE_SERIES,
    TREND_FEATURES,
    BREAK_FEATURES,
    ANOMALY_FEATURES,
    base_family as _base_family,
    feature_family as _feature_family,
    promote_legacy_volatility_features,
    validate_requested_combination,
    validate_numerical_output,
    validate_metadata_contract,
    validate_label_integrity,
)
from betise.utils.helpers import add_indices_column

# ── Feature pipeline order ────────────────────────────────────────────────────
# Volatility combinations are handled before base-series generation through the innovation process.
# Remaining overlays are applied afterward in feature order.

FEATURE_ORDER = [
    "arch",
    "garch",
    "egarch",
    "aparch",
    "linear_trend",
    "quadratic_trend",
    "cubic_trend",
    "exponential_trend",
    "damped_trend",
    "mean_shift",
    "variance_shift",
    "trend_shift",
    "point_anomaly",
    "collective_anomaly",
    "contextual_anomaly",
]

# Volatility still lives under dataset.features in the legacy config,
# but canonical rules treat it as a mathematical base-family component.
VOLATILITY_FEATURES = VOLATILITY_BASE_SERIES

    
# ── PyArrow compatibility patch ───────────────────────────────────────────────

def _patch_pyarrow_unregister_extension_type() -> None:
    try:
        import pyarrow as pa
    except Exception:
        return
    if getattr(pa, "_tsgen_ext_patch", False):
        return
    _orig_unreg = getattr(pa, "unregister_extension_type", None)
    _orig_reg   = getattr(pa, "register_extension_type", None)
    if _orig_unreg is None or _orig_reg is None:
        return
    def _safe_unreg(name):
        try: return _orig_unreg(name)
        except Exception: return None
    def _safe_reg(ext_type):
        try: return _orig_reg(ext_type)
        except Exception: return None
    pa.unregister_extension_type = _safe_unreg
    pa.register_extension_type   = _safe_reg
    pa._tsgen_ext_patch = True


# ── Small helpers ─────────────────────────────────────────────────────────────

def _sample_value(val: Any) -> Any:
    if isinstance(val, list) and len(val) == 2 and all(isinstance(x, (int, float)) for x in val):
        lo, hi = val
        if isinstance(lo, int) and isinstance(hi, int):
            return int(np.random.randint(lo, hi + 1))
        return float(np.random.uniform(lo, hi))
    return val


def _parse_sign(direction: str | None) -> int:
    direction = (direction or "up").lower()
    if direction in {"down", "downward", "negative"}:
        return -1
    if direction in {"both", "mixed"}:
        return random.choice([-1, 1])
    return 1


def _resolve_count(configured: int | None, low: int, high: int) -> int:
    if configured is None or configured <= 0:
        return int(np.random.randint(low, high + 1))
    return int(configured)


def _ensure_list(values: Any, expected_len: int, fallback: Any) -> List[Any]:
    if isinstance(values, list):
        if len(values) >= expected_len:
            return values[:expected_len]
        if not values:
            values = [fallback]
        return (values * expected_len)[:expected_len]
    return [fallback] * expected_len


def _validate_current_pipeline_support(base_components: List[str]) -> None:
    """GATE 4: ensure the current orchestrator can generate this valid rule.

    The canonical rule engine already knows more combinations than the current
    legacy config can express. At this stage the dataset orchestrator directly
    supports:

    - one standalone base component
    - root base + one volatility mechanism, when volatility is used as
      innovations (AR/MA/ARMA, stochastic, deterministic SARMA/SARIMA, ARFIMA)

    Additive multi-base Fourier compositions will be wired in the later
    multi-base orchestrator.
    """

    if len(base_components) == 1:
        return

    if len(base_components) == 2:
        families = [_base_family(component) for component in base_components]

        if "volatility" in families:
            non_volatility = next(
                component
                for component in base_components
                if _base_family(component) != "volatility"
            )

            implemented_with_volatility = (
                {"ar", "ma", "arma"}
                | STOCHASTIC_BASE_SERIES
                | {"sarma", "sarima"}
                | FRACTIONAL_BASE_SERIES
            )

            if non_volatility in implemented_with_volatility:
                return

    raise NotImplementedError(
        "The requested combination is canonically valid, but the current "
        "dataset-generation orchestrator does not yet implement this path: "
        f"{base_components}"
    )


# ── Base metadata helper ──────────────────────────────────────────────────────

def _base_metadata(base_series: str, info: Dict[str, Any]) -> Tuple[str, str]:
    """Return (base_coefs, order) as strings for metadata.

    Always returns strings so that concatenating DataFrames from different
    base types does not produce mixed-type object columns (which cause
    PyArrow type errors on to_parquet).
    """
    if base_series in {"white_noise", "random_walk", "random_walk_drift"} \
            | SEASONAL_BASE_SERIES | VOLATILITY_BASE_SERIES | FRACTIONAL_BASE_SERIES:
        return "0", "0"
    if base_series == "ar":
        return f"({info.get('ar_coefs')})", f"({info.get('ar_order')})"
    if base_series == "ma":
        return f"({info.get('ma_coefs')})", f"({info.get('ma_order')})"
    if base_series in {"arma", "arima"}:
        return (
            f"({info.get('ar_coefs')},{info.get('ma_coefs')})",
            f"({info.get('ar_order')},{info.get('ma_order')})",
        )
    if base_series == "ari":
        return f"({info.get('ar_coefs')})", f"({info.get('ar_order')})"
    if base_series == "ima":
        return f"({info.get('ma_coefs')})", f"({info.get('ma_order')})"
    return "0", "0"


# ── Base series generation ────────────────────────────────────────────────────

def generate_base_series(
    ts: TimeSeriesGenerator,
    base_series: str,
    params_cfg: Dict[str, Any],
    innovations=None,
    arfima_numseas=None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    base_params = params_cfg.get("base", {}).get(base_series, {})

    # Stationary: ar, ma, arma, white_noise
    if base_series in {"white_noise", "ar", "ma", "arma"}:
        return ts.generate_stationary_base_series(distribution=base_series, innovations=innovations)

    # Stochastic: random_walk
    if base_series == "random_walk":
        sigma = _sample_value(base_params.get("sigma", 1.0))
        return ts.generate_stochastic_trend(
            kind="rw",
            noise_std=sigma,
            innovations=innovations)

    # Stochastic: random_walk_drift
    if base_series == "random_walk_drift":
        drift = _sample_value(base_params.get("drift", [0.01, 0.1]))
        sigma = _sample_value(base_params.get("sigma", 1.0))
        return ts.generate_stochastic_trend(
            kind="rwd",
            drift=drift,
            noise_std=sigma,
            innovations=innovations)

    # Integrated stochastic: ari, ima, arima
    if base_series == "ari":
        diff = _sample_value(base_params.get("diff", 1))
        return ts.generate_stochastic_trend(
            kind="ari",
            d=diff,
            innovations=innovations)

    if base_series == "ima":
        diff = _sample_value(base_params.get("diff", 1))
        return ts.generate_stochastic_trend(
            kind="ima",
            d=diff,
            innovations=innovations)

    if base_series == "arima":
        diff = _sample_value(base_params.get("diff", 1))
        return ts.generate_stochastic_trend(
            kind="arima",
            d=diff,
            innovations=innovations)

    # Seasonal base: sarma, sarima, single_seasonality, multiple_seasonality
    seasonality_cfg = params_cfg.get("seasonality", {})

    if base_series == "single_seasonality":

        p = seasonality_cfg.get(
            "single_seasonality",
            {}
        )

        period_cfg = p.get("period")

        if isinstance(period_cfg, list):
            valid_periods = ts.get_valid_calendar_periods(
                allowed_periods=period_cfg
            )

            if not valid_periods:
                raise ValueError(
                    "No valid configured period found for "
                    "single_seasonality."
                )

            period = int(
                random.choice(valid_periods)
            )

        else:
            period = (
                int(period_cfg)
                if period_cfg is not None
                else None
            )

        amplitude_cfg = p.get("amplitude")

        amplitude = (
            _sample_value(amplitude_cfg)
            if amplitude_cfg is not None
            else None
        )

        return ts.generate_single_seasonality(
            period=period,
            amplitude=amplitude
        )


    if base_series == "multiple_seasonality":

        p = seasonality_cfg.get(
            "multiple_seasonality",
            {}
        )

        num_components = int(
            p.get("num_components", 2)
        )

        period_candidates = p.get(
            "periods"
        )

        selected_periods = None

        if period_candidates is not None:

            valid_periods = ts.get_valid_calendar_periods(
                allowed_periods=period_candidates
            )

            if len(valid_periods) < num_components:
                raise ValueError(
                    f"multiple_seasonality needs "
                    f"{num_components} valid periods, "
                    f"but only {valid_periods} are available."
                )

            selected_periods = random.sample(
                valid_periods,
                num_components
            )

        amplitudes = None

        amplitude_cfg = p.get(
            "amplitudes"
        )

        if (
            amplitude_cfg is not None
            and selected_periods is not None
        ):

            if (
                isinstance(amplitude_cfg, list)
                and len(amplitude_cfg) == 2
                and all(
                    isinstance(x, (int, float))
                    for x in amplitude_cfg
                )
            ):

                low, high = amplitude_cfg

                amplitudes = [
                    float(
                        np.random.uniform(
                            low,
                            high
                        )
                    )
                    for _ in selected_periods
                ]

        return ts.generate_multiple_seasonality(
            num_components=num_components,
            periods=selected_periods,
            amplitudes=amplitudes
        )


    if base_series == "sarma":

        p = seasonality_cfg.get(
            "sarma",
            {}
        )

        period = p.get("period")

        if isinstance(period, list):
            valid_periods = ts.get_valid_calendar_periods(
                allowed_periods=period
            )

            period = (
                int(random.choice(valid_periods))
                if valid_periods
                else None
            )

        amplitude = (
            _sample_value(p["amplitude"])
            if "amplitude" in p
            else None
        )

        return ts.generate_deterministic_sarma(
            period=period,
            amplitude=amplitude,
            innovations=innovations
        )


    if base_series == "sarima":

        p = seasonality_cfg.get("sarima",{})
        period = p.get("period")

        if isinstance(period, list):
            valid_periods = ts.get_valid_calendar_periods(allowed_periods=period)
            period = (int(random.choice(valid_periods)) if valid_periods else None)

        amplitude = (_sample_value(p["amplitude"]) if "amplitude" in p else None)

        d = int(_sample_value(p.get("diff", 1)))

        return ts.generate_deterministic_sarima(
            period=period,
            amplitude=amplitude,
            d=d,
            innovations=innovations)
    
    # Volatility base: arch, garch, egarch, aparch
    if base_series in VOLATILITY_BASE_SERIES:
        return ts.generate_volatility(kind=base_series)

    # Fractional base: arfima
    if base_series == "arfima":
        d_range = base_params.get("d_range", [0.25, 0.49])
        alpha = _sample_value(base_params.get("alpha", 0))
        numseas = (
            int(arfima_numseas)
            if arfima_numseas is not None
            else int(_sample_value(base_params.get("numseas", 100))))

        return ts.generate_fractional_process(
            kind="arfima",
            d_range=tuple(d_range) if isinstance(d_range, list) else d_range,
            alpha=alpha,
            numseas=numseas,
            innovations=innovations,
        )

    raise ValueError(f"Unsupported base_series: '{base_series}'")


# ── Feature application ───────────────────────────────────────────────────────

def apply_feature(
    ts: TimeSeriesGenerator,
    df: pd.DataFrame,
    feature_name: str,
    feature_cfg: Dict[str, Any],
    params_cfg: Dict[str, Any],
    state: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    info: Dict[str, Any] = {}

    # ── Trend overlays ────────────────────────────────────────────────────────
    if feature_name == "linear_trend":
        p = params_cfg.get("trends", {}).get("linear_trend", {})
        sign = _parse_sign(feature_cfg.get("direction"))
        slope = (_sample_value(p.get("slope")) if "slope" in p else None)
        intercept = _sample_value(p.get("intercept", 1.0))

        df, info = ts.generate_deterministic_trend_linear(
            df,
            sign=sign,
            slope=slope,
            intercept=intercept)

        # Keep the generated linear trend parameters so trend_shift can continue from the same trend.
        state["linear_trend_info"] = info

        return df, info

    if feature_name == "quadratic_trend":
        p = params_cfg.get("trends", {}).get("quadratic_trend", {})
        sign     = _parse_sign(feature_cfg.get("direction"))
        a        = _sample_value(p.get("a")) if "a" in p else None
        b        = _sample_value(p.get("b")) if "b" in p else None
        c        = _sample_value(p.get("c")) if "c" in p else None
        location = feature_cfg.get("location", p.get("location", "center"))
        return ts.generate_deterministic_trend_quadratic(df, sign=sign, a=a, b=b, c=c, location=location)

    if feature_name == "cubic_trend":
        p         = params_cfg.get("trends", {}).get("cubic_trend", {})
        sign      = _parse_sign(feature_cfg.get("direction"))
        amplitude = _sample_value(p.get("amplitude", 10.0))
        location  = feature_cfg.get("location", p.get("location", "center"))
        return ts.generate_deterministic_trend_cubic(df, sign=sign, amplitude=amplitude, location=location)

    if feature_name == "exponential_trend":
        p    = params_cfg.get("trends", {}).get("exponential_trend", {})
        sign = _parse_sign(feature_cfg.get("direction"))
        a    = _sample_value(p.get("a")) if "a" in p else None
        b    = _sample_value(p.get("b")) if "b" in p else None
        return ts.generate_deterministic_trend_exponential(df, sign=sign, a=a, b=b)

    if feature_name == "damped_trend":
        p = params_cfg.get("trends",{}).get("damped_trend",{})
        sign = _parse_sign(feature_cfg.get("direction"))
        a = (_sample_value(p.get("a")) if "a" in p else None)
        if a is not None:
            a = sign * abs(a)
        b = (_sample_value(p.get("b")) if "b" in p else None)
        damping_rate = (_sample_value(p.get("damping_rate")) if "damping_rate" in p else None)
        return ts.generate_deterministic_trend_damped(df,sign=sign,a=a,b=b,damping_rate=damping_rate)

    # ── Structural breaks ─────────────────────────────────────────────────────
    if feature_name in {"mean_shift", "variance_shift"}:
        p            = params_cfg.get("structural_breaks", {}).get(feature_name, {})
        mode         = feature_cfg.get("mode", "single")
        direction    = _parse_sign(feature_cfg.get("direction", "up"))
        num_breaks   = _resolve_count(feature_cfg.get("num_breaks"), 2, 4) if mode == "multiple" else 1
        signs        = [direction] if mode == "single" else [random.choice([-1, 1]) for _ in range(num_breaks)]
        location     = feature_cfg.get("location", "middle") if mode == "single" else None
        scale_factor = _sample_value(p.get("scale_factor", 1.0))
        seasonal_period = state.get("seasonal_period")

        if feature_name == "mean_shift":
            return ts.generate_mean_shift(
                df, signs=signs, location=location,
                num_breaks=num_breaks, scale_factor=scale_factor,
                seasonal_period=seasonal_period)
        return ts.generate_variance_shift(
            df, signs=signs, location=location,
            num_breaks=num_breaks, scale_factor=scale_factor,
            seasonal_period=seasonal_period)

    if feature_name == "trend_shift":
        p = params_cfg.get("structural_breaks",{}).get("trend_shift",{})
        mode = feature_cfg.get("mode","single")
        num_breaks = (_resolve_count(feature_cfg.get("num_breaks"),2,4) if mode == "multiple" else 1)
        location = (feature_cfg.get("location","middle") if mode == "single" else None)
        scale_factor = _sample_value(p.get("scale_factor",1.0))
        # Use the ACTUAL linear trend already applied to the series
        linear_info = state.get("linear_trend_info")
        if linear_info is None:
            raise ValueError("trend_shift requires an existing linear_trend to be applied before it.")
        slope = linear_info.get("slope")
        intercept = linear_info.get("intercept")
        if slope is None or intercept is None:
            raise ValueError("linear_trend_info must contain both 'slope' and 'intercept' for trend_shift.")
        default_change = feature_cfg.get("change_type","direction_change")

        if (mode == "multiple" and default_change == "mixed"):
            valid_change_types = [
                "direction_change",
                "magnitude_change",
                "direction_and_magnitude_change",]

            change_types = [random.choice(valid_change_types) for _ in range(num_breaks)]

        else:
            change_types = _ensure_list(feature_cfg.get("change_types",[default_change]),num_breaks,default_change)

        return ts.generate_trend_shift(
            df,
            slope=slope,
            intercept=intercept,
            location=location,
            num_breaks=num_breaks,
            change_types=change_types,
            seasonal_period=state.get("seasonal_period"),
            scale_factor=scale_factor)

    # ── Anomalies ─────────────────────────────────────────────────────────────
    if feature_name == "point_anomaly":
        p            = {**params_cfg.get("anomalies", {}).get("point_anomaly", {}), **feature_cfg}
        mode         = feature_cfg.get("mode", "single")
        is_spike     = bool(p.get("is_spike", False))
        scale_factor = _sample_value(p.get("scale_factor", 1.0))
        if mode == "multiple":
            return ts.generate_point_anomalies(df, scale_factor=scale_factor)
        location = feature_cfg.get("location", "middle")
        return ts.generate_point_anomaly(df, location=location, scale_factor=scale_factor)

    if feature_name == "collective_anomaly":
        p            = params_cfg.get("anomalies", {}).get("collective_anomaly", {})
        mode         = feature_cfg.get("mode", "single")
        scale_factor = _sample_value(p.get("scale_factor", 1.0))
        location     = feature_cfg.get("location", "middle") if mode == "single" else None
        num_anomalies = _resolve_count(feature_cfg.get("num_anomalies"), 2, 4) if mode == "multiple" else 1
        default_shape = feature_cfg.get("anomaly_shapes", "rectangular")
        anomaly_shapes = _ensure_list(feature_cfg.get("anomaly_shapes"), num_anomalies, default_shape)
        return ts.generate_collective_anomalies(
            df, num_anomalies=num_anomalies, location=location, 
            anomaly_shapes= anomaly_shapes, scale_factor=scale_factor)

    if feature_name == "contextual_anomaly":
        p            = params_cfg.get("anomalies", {}).get(feature_name, {})
        mode         = feature_cfg.get("mode", "single")
        location     = feature_cfg.get("location", "middle") if mode == "single" else None
        num_anomalies = _resolve_count(feature_cfg.get("num_anomalies"), 2, 4) if mode == "multiple" else 1
    
        # Get the seasonal_info from state (populated when a seasonality base/feature was applied)
        seasonal_info = state.get("seasonal_info")
    
        # If no seasonal base was used, contextual anomalies cannot be applied
        if seasonal_info is None:
            raise ValueError(
                "contextual_anomaly requires a seasonal base series or seasonal feature. "
                "Please use base_series='single_seasonality', 'multiple_seasonality', 'sarma', or 'sarima', "
                "or enable a seasonality feature first."
            )
    
        return ts.generate_contextual_anomalies(
            df, 
            seasonal_info=seasonal_info,
            num_anomalies=num_anomalies, 
            location=location)

    return df, info


# ── Metadata helpers ──────────────────────────────────────────────────────────

def populate_family_metadata(
    meta: Dict[str, Any],
    component_name: str,
    family: str,
    info: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:

    if family == "stationary":
        meta["ar_order"] = info.get("ar_order")
        meta["ma_order"] = info.get("ma_order")
        meta["ar_coefs"] = info.get("ar_coefs")
        meta["ma_coefs"] = info.get("ma_coefs")

    elif family == "stochastic":
        meta["stochastic_type"] = component_name
        meta["difference"] = info.get("diff")
        meta["drift_value"] = info.get("drift")
        meta["ar_order"] = info.get("ar_order")
        meta["ma_order"] = info.get("ma_order")
        meta["ar_coefs"] = info.get("ar_coefs")
        meta["ma_coefs"] = info.get("ma_coefs")

    elif family == "volatility":
        meta["volatility_type"] = component_name
        meta["volatility_alpha"] = info.get("alpha")
        meta["volatility_beta"] = info.get("beta")
        meta["volatility_omega"] = info.get("omega")
        meta["volatility_theta"] = info.get("theta")
        meta["volatility_lambda"] = info.get("lambda")
        meta["volatility_gamma"] = info.get("gamma")
        meta["volatility_delta"] = info.get("delta")

    elif family == "seasonality":
        meta["is_seasonal"] = 1
        meta["seasonality_type"] = component_name

        periods = info.get("periods")
        meta["seasonality_periods"] = list(periods) if periods else None
        meta["seasonality_period_meanings"] = info.get("period_meanings")

        amplitudes = info.get("amplitudes")

        if amplitudes is not None:
            if isinstance(amplitudes, (list, tuple, np.ndarray)):
                amplitudes = list(amplitudes)
            else:
                amplitudes = [amplitudes]

        meta["seasonality_amplitudes"] = amplitudes
        meta["num_harmonics"] = info.get("num_harmonics")

        coefficients = info.get("fourier_coefficients")
        if coefficients is None:
            coefficients = info.get("coefficients")

        meta["fourier_coefficients"] = coefficients

        meta["seasonality_scale_factor"] = info.get(
            "fourier_scale_factor",
            info.get("scale_factor")
        )
        meta["seasonality_strength"] = info.get("seasonal_strength")
        meta["seasonality_period_balance_factors"] = info.get(
            "period_balance_factors"
        )
        meta["seasonality_calibration_difference_order"] = info.get(
            "calibration_difference_order",
            info.get("diff", 0)
        )

        meta["seasonal_difference"] = info.get("seasonal_diff")
        meta["seasonal_unit_root"] = info.get("seasonal_unit_root")
        meta["seasonal_initial_std"] = info.get("initial_std")
        meta["seasonal_ar_order"] = info.get("seasonal_ar_order")
        meta["seasonal_ma_order"] = info.get("seasonal_ma_order")
        meta["seasonal_ar_coefs"] = info.get("seasonal_ar_coefs")
        meta["seasonal_ma_coefs"] = info.get("seasonal_ma_coefs")

        # SARMA / SARIMA internally contain AR/MA information.
        if info.get("ar_order") is not None:
            meta["ar_order"] = info.get("ar_order")
        if info.get("ma_order") is not None:
            meta["ma_order"] = info.get("ma_order")
        if info.get("ar_coefs") is not None:
            meta["ar_coefs"] = info.get("ar_coefs")
        if info.get("ma_coefs") is not None:
            meta["ma_coefs"] = info.get("ma_coefs")
        if info.get("diff") is not None:
            meta["difference"] = info.get("diff")

        state["seasonal_info"] = info

        if periods:
            state["seasonal_period"] = (
                int(periods[0])
                if len(periods) == 1
                else [int(p) for p in periods]
            )

    elif family == "fractional":
        meta["fractional_type"] = component_name
        meta["d_parameter"] = info.get("d")
        meta["ar_order"] = info.get("p")
        meta["ma_order"] = info.get("q")
        meta["fractional_integrated"] = 1
        meta["long_memory"] = int(0 < info.get("d", 0) < 0.5)

    return meta

def _set_primary(
    meta: Dict[str, Any],
    category: str, label: int,
    sub_category: str, sub_label: int,
) -> None:
    meta["primary_category"] = category
    meta["primary_label"]    = label
    meta["sub_category"]     = sub_category
    meta["sub_label"]        = sub_label


def update_metadata(
    meta: Dict[str, Any],
    feature_name: str,
    info: Dict[str, Any],
    feature_cfg: Dict[str, Any],
) -> Dict[str, Any]:

    if feature_name in VOLATILITY_FEATURES:
        _set_primary(meta, "volatility", 5, feature_name, 0)
        meta["volatility_type"]  = feature_name
        meta["volatility_alpha"] = info.get("alpha")
        meta["volatility_beta"]  = info.get("beta")
        meta["volatility_omega"] = info.get("omega")
        meta["volatility_theta"] = info.get("theta")
        meta["volatility_lambda"] = info.get("lambda")
        meta["volatility_gamma"] = info.get("gamma")
        meta["volatility_delta"] = info.get("delta")
        meta["is_stationary"]    = 0
        return meta


    if feature_name in TREND_FEATURES:
        sub_map = {
            "linear_trend":      0,
            "quadratic_trend":   1,
            "cubic_trend":       2,
            "exponential_trend": 3,
            "damped_trend": 4,
        }
        _set_primary(meta, "trend", 2, feature_name, sub_map.get(feature_name, 0))
        meta["is_stationary"]   = 0
        meta["trend_type"]      = feature_name
        meta["trend_slope"]     = info.get("slope")
        meta["trend_intercept"] = info.get("intercept")
        meta["trend_coef_a"]    = info.get("a")
        meta["trend_coef_b"]    = info.get("b")
        meta["trend_coef_c"]    = info.get("c")
        meta["trend_damping_rate"] = info.get("damping_rate")
        return meta

    if feature_name in BREAK_FEATURES:
        sub_map = {"mean_shift": 0, "variance_shift": 1, "trend_shift": 2}
        _set_primary(meta, "structural_break", 6, feature_name, sub_map.get(feature_name, 0))
        meta["is_stationary"] = 0
        meta["break_type"]    = info.get("subtype", feature_name)
        meta["break_count"]   = info.get("num_breaks")
        indices = info.get("shift_indices") or info.get("starts")
        if indices is not None: meta["break_indices"] = indices
        if "shift_magnitudes" in info: meta["break_magnitudes"]         = info.get("shift_magnitudes")
        if "shift_types"      in info: meta["trend_shift_change_types"] = info.get("shift_types")
        loc = feature_cfg.get("location")
        if   feature_name == "mean_shift":     meta["location_mean_shift"]     = loc
        elif feature_name == "variance_shift": meta["location_variance_shift"] = loc
        else:                                  meta["location_trend_shift"]    = loc
        return meta

    if feature_name in ANOMALY_FEATURES:
        sub_map = {"point_anomaly": 0, "collective_anomaly": 1, "contextual_anomaly": 2}
        _set_primary(meta, "anomaly", 1, feature_name, sub_map.get(feature_name, 0))
        meta["is_stationary"] = 0
        meta["anomaly_type"]  = info.get("subtype", feature_name)
        meta["anomaly_count"] = info.get("num_anomalies", 1)
        if "anomaly_indices" in info:
            meta["anomaly_indices"] = info.get("anomaly_indices")
        if "starts" in info and "ends" in info:
            meta["anomaly_indices"] = [info.get("starts"), info.get("ends")]
        loc = info.get("location", feature_cfg.get("location"))
        if   feature_name == "point_anomaly":      meta["location_point"]      = loc
        elif feature_name == "collective_anomaly": meta["location_collective"] = loc
        else:                                      meta["location_contextual"] = loc
        return meta

    return meta


# ── Label builder ─────────────────────────────────────────────────────────────

def _build_label(
    base_series: str,
    enabled_features: List[str],
    feature_cfgs: Dict[str, Any],
) -> str:
    parts = [base_series]
    for name in enabled_features:
        cfg    = feature_cfgs.get(name, {})
        suffix = None
        if name in TREND_FEATURES:
            suffix = cfg.get("direction")
        elif name in BREAK_FEATURES | ANOMALY_FEATURES:
            suffix = cfg.get("mode")
        parts.append(f"{name}:{suffix}" if suffix else name)
    return "__".join(parts)


# ── Main entry points ─────────────────────────────────────────────────────────

def generate_dataframe(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate time series dataset and return as (DataFrame, context).

    Nothing is written to disk. Use run() to also save the result.
    """
    _patch_pyarrow_unregister_extension_type()

    params_cfg  = cfg["params"]
    dataset_cfg = cfg["dataset"]

    seed = dataset_cfg.get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    output_dir      = Path(dataset_cfg.get("output_dir", "generated-dataset"))
    output_name     = dataset_cfg.get("output_name", "dataset.parquet")
    num_series      = int(dataset_cfg.get("num_series", 10))
    length_range    = dataset_cfg.get("length_range", [300, 500])
    include_indices = bool(dataset_cfg.get("include_indices", True))
    base_series     = dataset_cfg.get("base_series", "ar")

    feature_cfgs     = dataset_cfg.get("features", {})
    enabled_features = [n for n in FEATURE_ORDER if feature_cfgs.get(n, {}).get("enabled")]

    # -----------------------------------------------------------------
    # Canonical pre-generation validation (Gates 1-4)
    # -----------------------------------------------------------------
    requested_base_components, requested_overlay_features = (
        promote_legacy_volatility_features(
            base_components=[base_series],
            enabled_features=enabled_features,
        )
    )

    rule_report = validate_requested_combination(
        base_components=requested_base_components,
        feature_components=requested_overlay_features,
    )
    rule_report.raise_for_errors()

    _validate_current_pipeline_support(
        rule_report.base_components
    )

    all_dfs: List[pd.DataFrame] = []
    label = _build_label(base_series, enabled_features, feature_cfgs)

    for i in range(num_series):
        length = int(
            np.random.randint(
                int(length_range[0]),
                int(length_range[1]) + 1
            )
        )

        ts = TimeSeriesGenerator(
            length=length
        )

        arfima_numseas = None

        if base_series == "arfima":
            arfima_params = params_cfg.get("base", {}).get("arfima", {})
            arfima_numseas = int(
                _sample_value(
                    arfima_params.get("numseas", 100)
                )
            )
        feature_records = []

        # ---------------------------------------------------------
        # Prepare volatility innovations BEFORE base generation
        # ---------------------------------------------------------

        volatility_features = [
            name
            for name in enabled_features
            if name in VOLATILITY_FEATURES
        ]

        if len(volatility_features) > 1:
            raise ValueError(
                "Only one volatility mechanism can be enabled "
                "for a series."
            )

        volatility_feature = (
            volatility_features[0]
            if volatility_features
            else None
        )

        volatility_info = None
        innovations = None

        if volatility_feature is not None:

            if base_series == "white_noise":
                raise ValueError(
                    "White Noise + Volatility is not allowed."
                )

            if base_series == "arfima":
                volatility_ts = TimeSeriesGenerator(
                    length=length + arfima_numseas
                )

                innovations, volatility_info = (
                    volatility_ts.generate_volatility(
                        kind=volatility_feature,
                        as_innovations=True
                    )
                )

            elif (
                base_series in {"ar", "ma", "arma"}
                or base_series in STOCHASTIC_BASE_SERIES
                or base_series in {"sarma", "sarima"}
            ):
                innovations, volatility_info = (
                    ts.generate_volatility(
                        kind=volatility_feature,
                        as_innovations=True
                    )
                )

            elif base_series in VOLATILITY_BASE_SERIES:
                raise ValueError(
                    "A volatility base series cannot be combined "
                    "with another volatility feature."
                )

            else:
                raise NotImplementedError(
                    f"{base_series} + {volatility_feature} "
                    "has not yet been implemented in the "
                    "dataset generation pipeline."
                )

        # ---------------------------------------------------------
        # Generate the base series
        # ---------------------------------------------------------

        df, base_info = generate_base_series(
            ts,
            base_series,
            params_cfg,
            innovations=innovations,
            arfima_numseas=arfima_numseas,
        )

        components = [
            {
                "name": base_series,
                "family": _base_family(base_series),
                "info": base_info,
            }
        ]

        if volatility_feature is not None and volatility_info is not None:
            components.append({
                "name": volatility_feature,
                "family": "volatility",
                "info": volatility_info,
            })

        state: Dict[str, Any] = {
            "seasonal_period": None,
            "seasonal_info": None
        }

        base_coefs, base_order = _base_metadata(base_series, base_info)

        # ── Determine initial primary category from base type ─────────────────
        if base_series in SEASONAL_BASE_SERIES:
            primary_category = "seasonality"
            primary_label    = 4
        elif base_series in VOLATILITY_BASE_SERIES:
            primary_category = "volatility"
            primary_label    = 5
        elif base_series in STOCHASTIC_BASE_SERIES:
            primary_category = "stochastic"
            primary_label    = 3
        else:
            is_stat          = int(df["stationary"].iloc[0]) if "stationary" in df.columns else 1
            primary_category = "stationary" if is_stat == 1 else "stochastic"
            primary_label    = 0 if is_stat == 1 else 3

        meta: Dict[str, Any] = {
            "is_stationary": int(df["stationary"].iloc[0]) if "stationary" in df.columns else 1,
            "is_seasonal": int(df["seasonal"].iloc[0]) if "seasonal" in df.columns else 0,
            "primary_category": primary_category,
            "primary_label": primary_label,
            "sub_category": base_series,
            "sub_label": 0,
            "base_series": base_series,
            "base_components": list(rule_report.base_components),
            "base_families": list(rule_report.base_families),
            "composition_steps": list(rule_report.composition_steps),
        }

        # Populate metadata for every base-family component.
        for component in components:
            meta = populate_family_metadata(
                meta=meta,
                component_name=component["name"],
                family=component["family"],
                info=component["info"],
                state=state,
            )

        # ── Apply canonical overlay pipeline ──────────────────────────────────
        # Volatility has already been promoted to base_components by rules.py.
        # The rule report orders true overlays as:
        # deterministic trend -> structural break -> anomaly.
        for feature_name in rule_report.feature_components:
            feature_cfg_item = feature_cfgs.get(
                feature_name,
                {}
            )

            df, info = apply_feature(
                ts,
                df,
                feature_name,
                feature_cfg_item,
                params_cfg,
                state
            )

            feature_records.append({
                "name": feature_name,
                "family": _feature_family(feature_name),
                "info": info,
            })

            meta = update_metadata(
                meta,
                feature_name,
                info,
                feature_cfg_item
            )

        meta["feature_components"] = [feature["name"] for feature in feature_records]
        meta["feature_families"] = [feature["family"] for feature in feature_records]
        meta["feature_infos"] = {feature["name"]: feature["info"] for feature in feature_records}

        # ── Final stationarity / seasonality flags from df columns ────────────
        meta["is_stationary"] = int(df["stationary"].iloc[0]) if "stationary" in df.columns else meta.get("is_stationary", 1)
        meta["is_seasonal"]   = int(df["seasonal"].iloc[0])   if "seasonal"   in df.columns else meta.get("is_seasonal",   0)

        # -----------------------------------------------------------------
        # Runtime validation gates
        # -----------------------------------------------------------------
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

        series_id = i + 1
        df_clean  = df.drop(columns=["stationary", "seasonal"], errors="ignore")

        record = create_metadata_record(
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
            seasonality_period_meanings=meta.get("seasonality_period_meanings"),
            num_harmonics=meta.get("num_harmonics"),
            fourier_coefficients=meta.get("fourier_coefficients"),
            seasonal_unit_root=meta.get("seasonal_unit_root"),
            seasonality_scale_factor=meta.get("seasonality_scale_factor"),
            seasonality_strength=meta.get("seasonality_strength"),
            seasonality_period_balance_factors=meta.get("seasonality_period_balance_factors"),
            seasonality_calibration_difference_order=meta.get("seasonality_calibration_difference_order"),
            seasonal_initial_std=meta.get("seasonal_initial_std"),
            volatility_type=meta.get("volatility_type"),
            volatility_alpha=meta.get("volatility_alpha"),
            volatility_beta=meta.get("volatility_beta"),
            volatility_omega=meta.get("volatility_omega"),
            volatility_theta=meta.get("volatility_theta"),
            volatility_lambda=meta.get("volatility_lambda"),
            volatility_gamma=meta.get("volatility_gamma"),
            volatility_delta=meta.get("volatility_delta"),
            fractional_type=meta.get("fractional_type"),
            fractional_integrated=meta.get("fractional_integrated"),
            long_memory=meta.get("long_memory"),
            d_parameter=meta.get("d_parameter"),
            anomaly_type=meta.get("anomaly_type"),
            anomaly_count=meta.get("anomaly_count"),
            anomaly_indices=meta.get("anomaly_indices"),
            break_type=meta.get("break_type"),
            break_count=meta.get("break_count"),
            break_indices=meta.get("break_indices"),
            break_magnitudes=meta.get("break_magnitudes"),
            trend_shift_change_types=meta.get("trend_shift_change_types"),
            location_point=meta.get("location_point"),
            location_collective=meta.get("location_collective"),
            location_mean_shift=meta.get("location_mean_shift"),
            location_variance_shift=meta.get("location_variance_shift"),
            location_trend_shift=meta.get("location_trend_shift"),
            location_contextual=meta.get("location_contextual"),
        )

        df_meta = attach_metadata_columns_to_df(df_clean, record)
        if include_indices:
            df_meta = add_indices_column(df_meta)
        all_dfs.append(df_meta)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Normalize object columns: mixed int/str/None across base types causes
    # PyArrow type errors. Cast all object-dtype columns to str consistently,
    # but preserve numeric types for fractional parameters.
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
    for _col in combined_df.select_dtypes(include="object").columns:
        if _col in numeric_cols:
            # Convert to numeric, coercing errors to NaN
            combined_df[_col] = pd.to_numeric(combined_df[_col], errors='coerce')
        else:
            combined_df[_col] = combined_df[_col].astype(str)

    context = {
        "dataset_cfg": dataset_cfg,
        "output_dir": output_dir,
        "output_name": output_name,
        "num_series": num_series,
        "base_series": base_series,
        "enabled_features": enabled_features,
        "base_components": list(rule_report.base_components),
        "base_families": list(rule_report.base_families),
        "overlay_features": list(rule_report.feature_components),
        "composition_steps": list(rule_report.composition_steps),
        "rule_warnings": list(rule_report.warnings),
        "label": label,
        "metadata": {
            "base_series": base_series,
            "base_components": list(rule_report.base_components),
            "base_families": list(rule_report.base_families),
        },
    }
    return combined_df, context


def run(cfg: Dict[str, Any]) -> None:
    """Generate dataset and save to parquet at output_dir/output_name."""
    combined_df, context = generate_dataframe(cfg)
    output_dir:  Path = context["output_dir"]
    output_name: str  = context["output_name"]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    combined_df.to_parquet(output_path, index=False)

    print("=" * 70)
    print("DATASET GENERATION COMPLETE")
    print(f"Output   : {output_path.resolve()}")
    print(f"Series   : {context['num_series']}")
    print(f"Base     : {context['base_series']}")
    print(f"Features : {context['enabled_features'] or ['<none>']}")
    print("=" * 70)


if __name__ == "__main__":
    run(load_config())
