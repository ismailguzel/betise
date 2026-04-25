"""
Core components for time series generation.

This module contains the fundamental classes and utilities for generating
time series data.
"""

from .metadata import (
    create_metadata_record,
    attach_metadata_columns_to_df,
    get_metadata_columns_defaults,
    make_json_serializable
)

__all__ = [
    'create_metadata_record',
    'attach_metadata_columns_to_df',
    'get_metadata_columns_defaults',
    'make_json_serializable'
]

