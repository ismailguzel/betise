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
from betise.utils.helpers import add_indices_column

# ── Feature pipeline order ────────────────────────────────────────────────────
# Applied in this exact sequence: volatility → seasonality → trend → break → anomaly

FEATURE_ORDER = [
    "arch",
    "garch",
    "egarch",
    "aparch",
    "single_seasonality",
    "multiple_seasonality",
    "sarma",
    "sarima",
    "linear_trend",
    "quadratic_trend",
    "cubic_trend",
    "exponential_trend",
    "mean_shift",
    "variance_shift",
    "trend_shift",
    "point_anomaly",
    "collective_anomaly",
    "contextual_anomaly",
]

VOLATILITY_FEATURES   = {"arch", "garch", "egarch", "aparch"}
SEASONALITY_FEATURES  = {"single_seasonality", "multiple_seasonality", "sarma", "sarima"}
TREND_FEATURES        = {"linear_trend", "quadratic_trend", "cubic_trend", "exponential_trend"}
BREAK_FEATURES        = {"mean_shift", "variance_shift", "trend_shift"}
ANOMALY_FEATURES      = {"point_anomaly", "collective_anomaly", "contextual_anomaly"}

# Base-series category sets (used for metadata and initial classification)
STOCHASTIC_BASE_SERIES = {"random_walk", "random_walk_drift", "ari", "ima", "arima"}
SEASONAL_BASE_SERIES   = {"sarma", "sarima"}
VOLATILITY_BASE_SERIES = {"arch", "garch", "egarch", "aparch"}


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


# ── Base metadata helper ──────────────────────────────────────────────────────

def _base_metadata(base_series: str, info: Dict[str, Any]) -> Tuple[str, str]:
    """Return (base_coefs, order) as strings for metadata.

    Always returns strings so that concatenating DataFrames from different
    base types does not produce mixed-type object columns (which cause
    PyArrow type errors on to_parquet).
    """
    if base_series in {"white_noise", "random_walk", "random_walk_drift"} \
            | SEASONAL_BASE_SERIES | VOLATILITY_BASE_SERIES:
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
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    base_params = params_cfg.get("base", {}).get(base_series, {})

    # Stationary: ar, ma, arma, white_noise
    if base_series in {"white_noise", "ar", "ma", "arma"}:
        return ts.generate_stationary_base_series(distribution=base_series)

    # Stochastic: random_walk
    if base_series == "random_walk":
        sigma = _sample_value(base_params.get("sigma", 1.0))
        return ts.generate_stochastic_trend(kind="rw", noise_std=sigma)

    # Stochastic: random_walk_drift
    if base_series == "random_walk_drift":
        drift = _sample_value(base_params.get("drift", [0.01, 0.1]))
        sigma = _sample_value(base_params.get("sigma", 1.0))
        return ts.generate_stochastic_trend(kind="rwd", drift=drift, noise_std=sigma)

    # Integrated: ari, ima, arima
    if base_series == "ari":
        return ts.generate_stochastic_trend(kind="ari")
    if base_series == "ima":
        return ts.generate_stochastic_trend(kind="ima")
    if base_series == "arima":
        return ts.generate_stochastic_trend(kind="arima")

    # Seasonal base: sarma, sarima
    if base_series == "sarma":
        series, info = ts.generate_sarma_series(ts.length)
        if series is None:
            raise RuntimeError("SARMA base generation failed")
        df = pd.DataFrame({
            "time":       np.arange(ts.length),
            "data":       series,
            "stationary": np.zeros(ts.length, dtype=int),
            "seasonal":   np.ones(ts.length, dtype=int),
        })
        return df, info

    if base_series == "sarima":
        series, info = ts.generate_sarima_series(ts.length)
        if series is None:
            raise RuntimeError("SARIMA base generation failed")
        df = pd.DataFrame({
            "time":       np.arange(ts.length),
            "data":       series,
            "stationary": np.zeros(ts.length, dtype=int),
            "seasonal":   np.ones(ts.length, dtype=int),
        })
        return df, info

    # Volatility base: arch, garch, egarch, aparch
    if base_series in VOLATILITY_BASE_SERIES:
        return ts.generate_volatility(kind=base_series)

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

    # ── Volatility overlay ────────────────────────────────────────────────────
    if feature_name in VOLATILITY_FEATURES:
        vol_df, info = ts.generate_volatility(kind=feature_name)
        df.loc[:, "data"] = vol_df["data"].values
        return df, info

    # ── Seasonality overlay ───────────────────────────────────────────────────
    if feature_name == "single_seasonality":
        p = params_cfg.get("seasonality", {}).get("single_seasonality", {})
        period    = _sample_value(p.get("period"))    if "period"    in p else None
        amplitude = _sample_value(p.get("amplitude")) if "amplitude" in p else None
        df, info = ts.generate_single_seasonality(df, period=period, amplitude=amplitude)
        state["seasonal_period"] = info.get("period")
        return df, info

    if feature_name == "multiple_seasonality":
        p = params_cfg.get("seasonality", {}).get("multiple_seasonality", {})
        num_components = int(feature_cfg.get("num_components", p.get("num_components", 2)))
        periods_pool   = p.get("periods")
        periods = None
        if isinstance(periods_pool, list) and periods_pool:
            size    = min(num_components, len(periods_pool))
            periods = random.sample(periods_pool, size)
        amp_cfg = p.get("amplitudes")
        amplitudes = None
        if amp_cfg is not None and periods is not None:
            amplitudes = [_sample_value(amp_cfg) for _ in range(len(periods))]
        df, info = ts.generate_multiple_seasonality(
            df, num_components=num_components, periods=periods, amplitudes=amplitudes)
        periods_meta = info.get("periods") or []
        state["seasonal_period"] = periods_meta[0] if periods_meta else None
        return df, info

    if feature_name == "sarma":
        series, info = ts.generate_sarma_series(ts.length)
        if series is None:
            raise RuntimeError("SARMA generation failed")
        # Series is already z-normalized inside generator — do not normalize again
        df.loc[:, "data"]     = series
        df.loc[:, "seasonal"] = 1
        state["seasonal_period"] = info.get("period")
        return df, info

    if feature_name == "sarima":
        series, info = ts.generate_sarima_series(ts.length)
        if series is None:
            raise RuntimeError("SARIMA generation failed")
        # Series is already z-normalized inside generator — do not normalize again
        df.loc[:, "data"]     = series
        df.loc[:, "seasonal"] = 1
        state["seasonal_period"] = info.get("period")
        return df, info

    # ── Trend overlays ────────────────────────────────────────────────────────
    if feature_name == "linear_trend":
        p = params_cfg.get("trends", {}).get("linear_trend", {})
        sign      = _parse_sign(feature_cfg.get("direction"))
        slope     = _sample_value(p.get("slope"))     if "slope"     in p else None
        intercept = _sample_value(p.get("intercept", 1.0))
        return ts.generate_deterministic_trend_linear(df, sign=sign, slope=slope, intercept=intercept)

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
        p            = params_cfg.get("structural_breaks", {}).get("trend_shift", {})
        mode         = feature_cfg.get("mode", "single")
        sign         = _parse_sign(feature_cfg.get("direction", "up"))
        num_breaks   = _resolve_count(feature_cfg.get("num_breaks"), 2, 4) if mode == "multiple" else 1
        location     = feature_cfg.get("location", "middle") if mode == "single" else None
        scale_factor = _sample_value(p.get("scale_factor", 1.0))
        default_change = feature_cfg.get("change_type", "direction_change")
        change_types = _ensure_list(
            feature_cfg.get("change_types", [default_change]), num_breaks, default_change)
        return ts.generate_trend_shift(
            df, sign=sign, location=location, num_breaks=num_breaks,
            change_types=change_types, seasonal_period=state.get("seasonal_period"),
            scale_factor=scale_factor)

    # ── Anomalies ─────────────────────────────────────────────────────────────
    if feature_name == "point_anomaly":
        p            = {**params_cfg.get("anomalies", {}).get("point_anomaly", {}), **feature_cfg}
        mode         = feature_cfg.get("mode", "single")
        is_spike     = bool(p.get("is_spike", False))
        scale_factor = _sample_value(p.get("scale_factor", 1.0))
        if mode == "multiple":
            return ts.generate_point_anomalies(df, scale_factor=scale_factor, is_spike=is_spike)
        location = feature_cfg.get("location", "middle")
        return ts.generate_point_anomaly(df, location=location, scale_factor=scale_factor, is_spike=is_spike)

    if feature_name in {"collective_anomaly", "contextual_anomaly"}:
        p            = params_cfg.get("anomalies", {}).get(feature_name, {})
        mode         = feature_cfg.get("mode", "single")
        scale_factor = _sample_value(p.get("scale_factor", 1.0))
        location     = feature_cfg.get("location", "middle") if mode == "single" else None
        num_anomalies = _resolve_count(feature_cfg.get("num_anomalies"), 2, 4) if mode == "multiple" else 1

        if feature_name == "collective_anomaly":
            return ts.generate_collective_anomalies(
                df, num_anomalies=num_anomalies, location=location, scale_factor=scale_factor)
        return ts.generate_contextual_anomalies(
            df, num_anomalies=num_anomalies, location=location,
            scale_factor=scale_factor, seasonal_period=state.get("seasonal_period"))

    return df, info


# ── Metadata helpers ──────────────────────────────────────────────────────────

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

    if feature_name in SEASONALITY_FEATURES:
        sub_map = {
            "single_seasonality":   0,
            "multiple_seasonality": 1,
            "sarma":  2,
            "sarima": 3,
        }
        _set_primary(meta, "seasonality", 4, feature_name, sub_map.get(feature_name, 0))
        meta["is_stationary"]   = 0
        meta["is_seasonal"]     = 1
        meta["seasonality_type"] = feature_name
        if "period"     in info: meta["seasonality_periods"]   = [info.get("period")]
        if "periods"    in info: meta["seasonality_periods"]   = info.get("periods")
        if "amplitude"  in info: meta["seasonality_amplitudes"] = [info.get("amplitude")]
        if "amplitudes" in info: meta["seasonality_amplitudes"] = info.get("amplitudes")
        # Seasonal model orders (populated for sarma / sarima overlays)
        meta["seasonal_ar_order"]   = info.get("seasonal_ar_order")
        meta["seasonal_ma_order"]   = info.get("seasonal_ma_order")
        meta["seasonal_difference"] = info.get("seasonal_diff")
        return meta

    if feature_name in TREND_FEATURES:
        sub_map = {
            "linear_trend":      0,
            "quadratic_trend":   1,
            "cubic_trend":       2,
            "exponential_trend": 3,
        }
        _set_primary(meta, "trend", 2, feature_name, sub_map.get(feature_name, 0))
        meta["is_stationary"]   = 0
        meta["trend_type"]      = feature_name
        meta["trend_slope"]     = info.get("slope")
        meta["trend_intercept"] = info.get("intercept")
        meta["trend_coef_a"]    = info.get("a")
        meta["trend_coef_b"]    = info.get("b")
        meta["trend_coef_c"]    = info.get("c")
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

    all_dfs: List[pd.DataFrame] = []
    label = _build_label(base_series, enabled_features, feature_cfgs)

    for i in range(num_series):
        length = int(np.random.randint(int(length_range[0]), int(length_range[1]) + 1))
        ts     = TimeSeriesGenerator(length=length)

        df, base_info = generate_base_series(ts, base_series, params_cfg)
        state: Dict[str, Any] = {"seasonal_period": None}

        base_coefs, base_order = _base_metadata(base_series, base_info)

        # ── Extract stochastic / ARIMA base parameters ────────────────────────
        drift_value    = base_info.get("drift")
        arima_ar_order = base_info.get("ar_order")
        arima_ma_order = base_info.get("ma_order")
        arima_diff     = base_info.get("diff")

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
            "is_stationary":   int(df["stationary"].iloc[0]) if "stationary" in df.columns else 1,
            "is_seasonal":     int(df["seasonal"].iloc[0])   if "seasonal"   in df.columns else 0,
            "primary_category": primary_category,
            "primary_label":    primary_label,
            "sub_category":     base_series,
            "sub_label":        0,
            "base_series":      base_series,
            "order":            base_order,
            "base_coefs":       base_coefs,
        }

        # ── Populate seasonality metadata when sarma/sarima is the base ───────
        if base_series in SEASONAL_BASE_SERIES:
            meta["seasonality_type"]    = base_series
            meta["is_seasonal"]         = 1
            meta["seasonality_periods"] = [base_info.get("period")] if base_info.get("period") else None
            meta["seasonal_ar_order"]   = base_info.get("seasonal_ar_order")
            meta["seasonal_ma_order"]   = base_info.get("seasonal_ma_order")
            meta["seasonal_difference"] = base_info.get("seasonal_diff")

        # ── Populate volatility metadata when arch/garch is the base ──────────
        if base_series in VOLATILITY_BASE_SERIES:
            meta["volatility_type"]   = base_series
            meta["volatility_alpha"]  = base_info.get("alpha")
            meta["volatility_beta"]   = base_info.get("beta")
            meta["volatility_omega"]  = base_info.get("omega")
            meta["volatility_theta"]  = base_info.get("theta")
            meta["volatility_lambda"] = base_info.get("lambda")
            meta["volatility_gamma"]  = base_info.get("gamma")
            meta["volatility_delta"]  = base_info.get("delta")

        # ── Apply feature pipeline ────────────────────────────────────────────
        for feature_name in enabled_features:
            feature_cfg_item = feature_cfgs.get(feature_name, {})
            df, info = apply_feature(ts, df, feature_name, feature_cfg_item, params_cfg, state)
            meta = update_metadata(meta, feature_name, info, feature_cfg_item)

        # ── Final stationarity / seasonality flags from df columns ────────────
        meta["is_stationary"] = int(df["stationary"].iloc[0]) if "stationary" in df.columns else meta.get("is_stationary", 1)
        meta["is_seasonal"]   = int(df["seasonal"].iloc[0])   if "seasonal"   in df.columns else meta.get("is_seasonal",   0)

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
            order=meta.get("order"),
            base_coefs=meta.get("base_coefs"),
            trend_type=meta.get("trend_type"),
            trend_slope=meta.get("trend_slope"),
            trend_intercept=meta.get("trend_intercept"),
            trend_coef_a=meta.get("trend_coef_a"),
            trend_coef_b=meta.get("trend_coef_b"),
            trend_coef_c=meta.get("trend_coef_c"),
            stochastic_type=base_series if base_series in STOCHASTIC_BASE_SERIES else None,
            drift_value=drift_value,
            arima_ar_order=arima_ar_order,
            arima_ma_order=arima_ma_order,
            arima_diff=arima_diff,
            is_seasonal=meta.get("is_seasonal"),
            seasonality_type=meta.get("seasonality_type"),
            seasonality_periods=meta.get("seasonality_periods"),
            seasonality_amplitudes=meta.get("seasonality_amplitudes"),
            seasonal_ar_order=meta.get("seasonal_ar_order"),
            seasonal_ma_order=meta.get("seasonal_ma_order"),
            seasonal_difference=meta.get("seasonal_difference"),
            volatility_type=meta.get("volatility_type"),
            volatility_alpha=meta.get("volatility_alpha"),
            volatility_beta=meta.get("volatility_beta"),
            volatility_omega=meta.get("volatility_omega"),
            volatility_theta=meta.get("volatility_theta"),
            volatility_lambda=meta.get("volatility_lambda"),
            volatility_gamma=meta.get("volatility_gamma"),
            volatility_delta=meta.get("volatility_delta"),
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
    # PyArrow type errors. Cast all object-dtype columns to str consistently.
    for _col in combined_df.select_dtypes(include="object").columns:
        combined_df[_col] = combined_df[_col].astype(str)

    context = {
        "dataset_cfg":     dataset_cfg,
        "output_dir":      output_dir,
        "output_name":     output_name,
        "num_series":      num_series,
        "base_series":     base_series,
        "enabled_features": enabled_features,
        "label":           label,
        "metadata":        {"base_series": base_series},
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
