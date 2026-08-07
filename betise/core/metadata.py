"""
Metadata management for seasonal time series datasets.

This module creates metadata records from the info dictionaries returned by:
- generate_single_seasonality
- generate_multiple_seasonality
- generate_sarima_series
- generate_deterministic_sarma_series
"""

import json
import numpy as np
import pandas as pd


def create_metadata_record(
    # === CORE ===
    series_id,
    length,
    label,
    is_stationary=0,

    # === SEASONAL INFO ===
    series_type=None,          # info["type"] -> "seasonal"
    subtype=None,              # info["subtype"] -> single/multiple/SARIMA/SARMA
    periods=None,
    period_meanings=None,

    # === FOURIER / SEASONALITY PARAMETERS ===
    amplitude=None,            # single, SARIMA, SARMA
    amplitudes=None,           # multiple
    noise_std=None,
    scale_factor=None,
    num_harmonics=None,
    coefficients=None,         # single, multiple, SARIMA
    fourier_coefficients=None, # SARMA

    # === SARIMA-SPECIFIC ===
    diff=None,
    seasonal_diff=None,
    unit_root=None,
    initial_std=None,

    # === SARMA-SPECIFIC ===
    ar_order=None,
    ma_order=None,
    seasonal_ar_order=None,
    seasonal_ma_order=None,
    # === VOLATILITY ===
    volatility_type=None,
    volatility_alpha=None,
    volatility_beta=None,
    volatility_omega=None,
    volatility_theta=None,
    volatility_lambda=None,
    volatility_gamma=None,
    volatility_delta=None,
    # === FRACTIONAL ===
    fractional_type=None,
    fractional_integrated=None,
    long_memory=None,
    d_parameter=None,
    #ar_order=None,
    #ma_order=None,
    # === ANOMALY ===
    anomaly_type=None,
    anomaly_shapes=None,
    anomaly_count=None,
    anomaly_indices=None,
    anomaly_magnitudes=None,
    # === BREAK ===
    break_type=None,
    break_count=None,
    break_indices=None,
    break_magnitudes=None,
    break_directions=None,
    trend_shift_change_types=None,
    # === LOCATION ===
    location_point=None,
    location_collective=None,
    location_mean_shift=None,
    location_variance_shift=None,
    location_trend_shift=None,
    location_contextual=None,
    # === NOISE & ETC ===
    noise_type=None,
    #noise_std=None,
    sampling_frequency=None,
):
    """
    Create metadata record for seasonal time series.
    """

    record = {
        # === Core ===
        "series_id": series_id,
        "length": length,
        "label": label,
        "is_stationary": is_stationary,

        # === Seasonal identity ===
        "type": series_type,
        "subtype": subtype,
        "periods": periods,
        "period_meanings": period_meanings,

        # === Fourier / seasonality parameters ===
        "amplitude": amplitude,
        "amplitudes": amplitudes,
        "noise_std": noise_std,
        "scale_factor": scale_factor,
        "num_harmonics": num_harmonics,
        "coefficients": coefficients,
        "fourier_coefficients": fourier_coefficients,

        # === SARIMA-specific ===
        "diff": diff,
        "seasonal_diff": seasonal_diff,
        "unit_root": unit_root,
        "initial_std": initial_std,

        # === SARMA-specific ===
        "ar_order": ar_order,
        "ma_order": ma_order,
        "seasonal_ar_order": seasonal_ar_order,
        "seasonal_ma_order": seasonal_ma_order,
        # === Volatility ===
        "volatility_type": volatility_type,
        "volatility_alpha": volatility_alpha,
        "volatility_beta": volatility_beta,
        "volatility_omega": volatility_omega,
        "volatility_theta": volatility_theta,
        "volatility_lambda": volatility_lambda,
        "volatility_gamma": volatility_gamma,
        "volatility_delta": volatility_delta,
        # === Fractional ===
        "fractional_type": fractional_type,
        "fractional_integrated": fractional_integrated,
        "long_memory": long_memory,
        "d_parameter": d_parameter,
        "ar_order": ar_order,
        "ma_order": ma_order,
        # === Anomaly ===
        "anomaly_type": anomaly_type,
        "anomaly_count": anomaly_count,
        "anomaly_indices": anomaly_indices,
        "anomaly_magnitudes": anomaly_magnitudes,
        "anomaly_shapes": anomaly_shapes,
        # === Break ===
        "break_type": break_type,
        "break_count": break_count,
        "break_indices": break_indices,
        "break_magnitudes": break_magnitudes,
        "break_directions": break_directions,
        "trend_shift_change_types": trend_shift_change_types,
        # === Location ===
        "location_point": location_point,
        "location_collective": location_collective,
        "location_mean_shift": location_mean_shift,
        "location_variance_shift": location_variance_shift,
        "location_trend_shift": location_trend_shift,
        "location_contextual": location_contextual,
        # === Noise & ETC ===
        "noise_type": noise_type,
        "noise_std": noise_std,
        "sampling_frequency": sampling_frequency,
    }
    return record


def make_json_serializable(obj):
    """
    Convert numpy objects to JSON-serializable Python objects.
    """

    if obj is None:
        return None

    if isinstance(obj, (np.integer, np.int_, np.int64, np.int32)):
        return int(obj)

    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)

    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return [make_json_serializable(x) for x in obj.tolist()]

    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]

    if isinstance(obj, set):
        return [make_json_serializable(x) for x in sorted(obj)]

    if isinstance(obj, dict):
        return {
            str(make_json_serializable(k)): make_json_serializable(v)
            for k, v in obj.items()
        }

    return obj


def metadata_value_to_cell(value):
    """
    Convert metadata values into dataframe-cell-friendly values.

    Scalars stay as scalars.
    Lists/dicts become JSON strings.
    """

    value = make_json_serializable(value)

    if value is None:
        return None

    if isinstance(value, (int, float, str, bool)):
        return value

    return json.dumps(value, ensure_ascii=False)


def get_metadata_columns_defaults():
    """
    Get metadata column names and default values.
    """

    dummy = create_metadata_record(
        series_id=0,
        length=0,
        label="",
        is_stationary=0
    )

    return list(dummy.keys()), dummy


def attach_metadata_columns_to_df(df, metadata_record):
    """
    Attach metadata columns to a generated time series dataframe.
    """

    df = df.copy()

    metadata_cols, default_record = get_metadata_columns_defaults()

    for col in metadata_cols:
        val = metadata_record.get(col, default_record[col])
        df[col] = metadata_value_to_cell(val)

    df["label"] = metadata_record["label"]

    core_cols = ["series_id", "time", "data"]

    optional_series_cols = [
        "seasonal_diff"
    ]

    meta_cols = [
        col for col in metadata_cols
        if col not in core_cols + ["label"] and col in df.columns
    ]

    final_cols_order = (
        core_cols
        + [col for col in optional_series_cols if col in df.columns]
        + meta_cols
        + ["label"]
    )

    final_cols_in_df = [
        col for col in final_cols_order
        if col in df.columns
    ]

    df = df[final_cols_in_df]

    return df