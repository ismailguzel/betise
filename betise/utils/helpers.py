"""
Helper utilities for dataset generation.

This module contains utility functions used across the dataset generation package.
"""

import os
import pandas as pd
import ast
import re
import numpy as np


def save_and_cleanup_grouped(all_dfs, folder, count, label):
    """
    Save combined DataFrame to Parquet and cleanup empty directory.

    This function is similar to save_and_cleanup but it also groups the data by series_id and aggregates the data points and labels into lists. 
    
    This is useful for saving the data in a more compact format where each row corresponds to a single time series with its associated labels.

    Parameters
    ----------
    all_dfs : list of pd.DataFrame
        List of DataFrames to combine
    folder : str
        Folder path where data was temporarily stored
    count : int
        Number of series generated
    label : str
        Label for the dataset

    Returns
    -------
    None
    """
    if not all_dfs:
        print(f"No data generated for {folder}")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    out = (
    combined_df
    .sort_values(["series_id", "time"])
    .groupby("series_id", as_index=False)
    .agg(
        data_points=("data", list),
        primary_label=("primary_label", "first"),
        sub_label=("sub_label", "first"),
        is_stationary=("is_stationary", "first"),
        is_seasonal=("is_seasonal", "first"),

)
)

    category_label = os.path.basename(folder) 
    parent_folder = os.path.dirname(folder)   
    
    output_filename = f"{category_label}.parquet"
    output_path = os.path.join(parent_folder, output_filename)

    out.to_parquet(output_path, index=False)

    try:
        os.rmdir(folder)
    except OSError as e:
        print(f"Warning: Could not remove empty directory {folder}: {e}")

    print(f"{count} '{label}' series saved in ONE file: '{output_path}'")


def save_and_cleanup(all_dfs, folder, count, label):
    """
    Save combined DataFrame to Parquet and cleanup empty directory.

    Parameters
    ----------
    all_dfs : list of pd.DataFrame
        List of DataFrames to combine
    folder : str
        Folder path where data was temporarily stored
    count : int
        Number of series generated
    label : str
        Label for the dataset

    Returns
    -------
    None
    """
    if not all_dfs:
        print(f"No data generated for {folder}")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    category_label = os.path.basename(folder) 
    parent_folder = os.path.dirname(folder)   
    
    output_filename = f"{category_label}.parquet"
    output_path = os.path.join(parent_folder, output_filename)

    combined_df.to_parquet(output_path, index=False)

    try:
        os.rmdir(folder)
    except OSError as e:
        print(f"Warning: Could not remove empty directory {folder}: {e}")

    print(f"{count} '{label}' series saved in ONE file: '{output_path}'")


def parse_indices(val):
    """
    Returns Python ints (and preserves nesting if present).

    Handles:
      - [1,2,3]
      - [[s1,s2,...],[e1,e2,...]]
      - strings of the above
      - strings like "[np.int64(64), np.int64(193)]"
      - strings like "[[np.int64(1)], [np.int64(2)]]"
    """
    if val is None:
        return []
    if isinstance(val, float) and pd.isna(val):
        return []

    # already list-like
    if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
        val = list(val)
        # nested list case: [[...],[...]]
        if len(val) == 2 and all(isinstance(x, (list, tuple, np.ndarray, pd.Series)) for x in val):
            return [[int(v) for v in list(val[0])], [int(v) for v in list(val[1])]]
        # flat list case: [...]
        return [int(v) for v in val]

    # string case
    if isinstance(val, str):
        s = val.strip()

        # turn np.int64(64) / numpy.int64(64) into 64 (keeps brackets/commas intact)
        s = re.sub(r"(?:np|numpy)\.int\d+\((-?\d+)\)", r"\1", s)

        try:
            obj = ast.literal_eval(s)
        except Exception:
            # fallback: just extract standalone integers (won't match the 64 in 'int64')
            nums = re.findall(r"\b-?\d+\b", s)
            return [int(n) for n in nums]

        # preserve nesting if present
        if isinstance(obj, (list, tuple)):
            if len(obj) == 2 and all(isinstance(x, (list, tuple)) for x in obj):
                return [[int(v) for v in obj[0]], [int(v) for v in obj[1]]]
            return [int(v) for v in obj]

        # single number
        return [int(obj)]

    # fallback
    return [int(val)]


def unpack_interval_indices(val):
    """
    Return start/end index lists from nested or flattened interval encodings.

    Supports values such as:
      - [[s1, s2], [e1, e2]]
      - [s1, e1]
      - [s1, s2, e1, e2]
      - string forms of the above
    """
    parsed = parse_indices(val)

    if not parsed:
        return [], []

    if (
        isinstance(parsed, list)
        and len(parsed) == 2
        and all(isinstance(part, list) for part in parsed)
    ):
        return [int(v) for v in parsed[0]], [int(v) for v in parsed[1]]

    if isinstance(parsed, list) and all(isinstance(v, int) for v in parsed):
        if len(parsed) == 2:
            return [int(parsed[0])], [int(parsed[1])]
        if len(parsed) % 2 == 0:
            midpoint = len(parsed) // 2
            return [int(v) for v in parsed[:midpoint]], [int(v) for v in parsed[midpoint:]]

    raise ValueError(f"Could not unpack interval indices from value: {val!r}")


def add_indices_column(df):
    """
    Add the indices of the anomalies and structural breaks in the DataFrame.
    
    The function can be used both in generation step and after loading the data from parquet files. 
    
    It checks the primary and sub labels to determine which type of anomaly or break is present and then parses the corresponding indices to mark them in new columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame

    Returns
    -------
    df: pd.DataFrame
        DataFrame with an additional 'indices' column containing indices of the break/anomaly points

    """
    df = df.copy()
    df.loc[:, 'point_anomaly_indices'] = 0
    df.loc[:, 'collective_anomaly_indices'] = 0
    df.loc[:, 'contextual_anomaly_indices'] = 0
    df.loc[:, 'mean_shift_indices'] = 0
    df.loc[:, 'var_shift_indices'] = 0
    df.loc[:, 'trend_shift_indices'] = 0
    if df['primary_label'].iloc[0] == 1 and df['sub_label'].iloc[0] == 0:
        point_anomaly_indices = df['anomaly_indices'][0]
        point_anomaly_indices = parse_indices(point_anomaly_indices)
        for point_anomaly_index in point_anomaly_indices:
            df.loc[point_anomaly_index, 'point_anomaly_indices'] = 1
    elif df['primary_label'].iloc[0] == 1 and df['sub_label'].iloc[0] == 1:
        collective_anomaly_indices = df['anomaly_indices'][0]
        starts, ends = unpack_interval_indices(collective_anomaly_indices)

        for s, e in zip(starts, ends):
            s = max(0, int(s))
            e = min(len(df) - 1, int(e))
            if s > e:
                s, e = e, s
            df.iloc[s:e+1, df.columns.get_loc("collective_anomaly_indices")] = 1

    elif df['primary_label'].iloc[0] == 1 and df['sub_label'].iloc[0] == 2:
        contextual_anomaly_indices = df['anomaly_indices'][0]
        starts, ends = unpack_interval_indices(contextual_anomaly_indices)

        for s, e in zip(starts, ends):
            s = max(0, int(s))
            e = min(len(df) - 1, int(e))
            if s > e:
                s, e = e, s
            df.iloc[s:e+1, df.columns.get_loc("contextual_anomaly_indices")] = 1
    elif df['primary_label'].iloc[0] == 6 and df['sub_label'].iloc[0] == 0:
        mean_shift_indices = df['break_indices'][0]
        mean_shift_indices = parse_indices(mean_shift_indices)
        for idx, break_point in enumerate(mean_shift_indices, start=1):
            df.iloc[break_point:, df.columns.get_loc('mean_shift_indices')] = idx
    elif df['primary_label'].iloc[0] == 6 and df['sub_label'].iloc[0] == 1:
        var_shift_indices = df['break_indices'][0]
        var_shift_indices = parse_indices(var_shift_indices)
        for idx, break_point in enumerate(var_shift_indices, start=1):
            df.iloc[break_point:, df.columns.get_loc('var_shift_indices')] = idx
    elif df['primary_label'].iloc[0] == 6 and df['sub_label'].iloc[0] == 2:
        trend_shift_indices = df['break_indices'][0]
        trend_shift_indices = parse_indices(trend_shift_indices)
        for idx, break_point in enumerate(trend_shift_indices, start=1):
            df.iloc[break_point:, df.columns.get_loc('trend_shift_indices')] = idx
    else:
        pass

    return df


def get_length_label(length_range):
    """
    Get a label for the length range.

    Parameters
    ----------
    length_range : tuple
        (min, max) length range

    Returns
    -------
    str
        'short', 'medium', or 'long'
    """
    if length_range == (50, 100):
        return "short"
    elif length_range == (300, 500):
        return "medium"
    else:
        return "long"

