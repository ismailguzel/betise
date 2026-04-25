"""
Metadata management for time series datasets.

This module contains functions for creating and managing metadata records
that describe the characteristics of generated time series.
"""

import numpy as np
import pandas as pd


def create_metadata_record(
    # === CORE ===
    series_id,
    length,
    label,
    is_stationary=1,
    # === NEW: HIERARCHY ===
    primary_category=None,  # "stationary" | "trend" | "volatility" | ...
    primary_label=None,
    sub_category=None,   # "linear_up" | "arch" | ...
    sub_label=None,
    # === BASE ===
    base_series=None,
    base_process_type=None,
    order=None,
    base_coefs=None,
    # === TREND ===
    trend_type=None,
    trend_slope=None,
    trend_intercept=None,
    trend_coef_a=None,
    trend_coef_b=None,
    trend_coef_c=None,
    trend_damping_rate=None,
    # === STOCHASTIC ===
    stochastic_type=None,
    difference=None,
    drift_value=None,
    arima_ar_order=None,
    arima_ma_order=None,
    arima_diff=None,
    # === SEASONALITY ===
    is_seasonal=None,
    seasonality_type=None,
    seasonality_periods=None,
    seasonality_amplitudes=None,
    seasonality_from_base=None,
    seasonal_difference=None,
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
    # === ANOMALY ===
    anomaly_type=None,
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
    noise_std=None,
    sampling_frequency=None,
):
    """
    Create a non-redundant, hierarchical metadata record for a time series.
    """
    record = {
        "series_id": series_id,
        "length": length,
        "label": label,
        "is_stationary": is_stationary,
        "primary_category": primary_category,
        "primary_label": primary_label,
        "sub_category": sub_category,
        "sub_label": sub_label,
        # === Base Process ===
        "base_series": base_series,
        "base_process_type": base_process_type,
        "order": order,
        "base_coefs": base_coefs,
        # === Trend ===
        "trend_type": trend_type,
        "trend_slope": trend_slope,
        "trend_intercept": trend_intercept,
        "trend_coef_a": trend_coef_a,
        "trend_coef_b": trend_coef_b,
        "trend_coef_c": trend_coef_c,
        "trend_damping_rate": trend_damping_rate,
        # === Stochastic ===
        "stochastic_type": stochastic_type,
        "difference": difference,
        "drift_value": drift_value,
        "arima_ar_order": arima_ar_order,
        "arima_ma_order": arima_ma_order,
        "arima_diff": arima_diff,
        # === Seasonality ===
        "is_seasonal" : is_seasonal,
        "seasonality_type": seasonality_type,
        "seasonality_periods": seasonality_periods,
        "seasonality_amplitudes": seasonality_amplitudes,
        "seasonality_from_base": seasonality_from_base,
        "seasonal_difference": seasonal_difference,
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
        # === Anomaly ===
        "anomaly_type": anomaly_type,
        "anomaly_count": anomaly_count,
        "anomaly_indices": anomaly_indices,
        "anomaly_magnitudes": anomaly_magnitudes,
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
    Convert numpy types to Python native types for JSON serialization.

    Parameters
    ----------
    obj : any
        Object to convert

    Returns
    -------
    any
        JSON-serializable version of the object
    """
    if isinstance(obj, (np.integer, np.int_, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    return str(obj)


def get_metadata_columns_defaults():
    """
    Get metadata column names and default values.

    Returns
    -------
    tuple
        (column_names, default_record)
    """
    dummy = create_metadata_record(series_id=0, length=0, label="", is_stationary=1)
    return list(dummy.keys()), dummy


def attach_metadata_columns_to_df(df, metadata_record):
    """
    Attach metadata columns to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data
    metadata_record : dict
        Metadata record to attach

    Returns
    -------
    pd.DataFrame
        DataFrame with metadata columns attached
    """
    metadata_cols, default_record = get_metadata_columns_defaults()

    for col in metadata_cols:
        val = metadata_record.get(col, default_record[col])

        if isinstance(val, (int, float, str)):
            df[col] = val
        else:
            df[col] = str(val)

    df['label'] = metadata_record['label']
    
    core_cols = ['series_id', 'time', 'data', 'label']
    meta_cols = [col for col in metadata_cols if col not in core_cols and col in df.columns]
    final_cols_order = ['series_id', 'time', 'data'] + meta_cols + ['label']
    
    final_cols_in_df = [col for col in final_cols_order if col in df.columns]
    
    df = df[final_cols_in_df]
    return df

