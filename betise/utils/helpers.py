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
    Add legacy *_indices columns from canonical localization labels.

    Localization labels are the source of truth.
    If a feature is not present, its indices column remains zero.
    """

    df = df.copy()

    label_to_indices = {
        "point_anom_label": "point_anomaly_indices",
        "collect_anom_label": "collective_anomaly_indices",
        "context_anom_label": "contextual_anomaly_indices",
        "mean_shift_label": "mean_shift_indices",
        "variance_shift_label": "var_shift_indices",
        "trend_shift_label": "trend_shift_indices",
    }

    for label_col, indices_col in label_to_indices.items():

        if label_col in df.columns:
            df[indices_col] = df[label_col]
        else:
            df[indices_col] = 0

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

