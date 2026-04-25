"""
BeTiSe — Loading and Using Generated Datasets
==============================================

Shows how to work with any parquet file produced by BeTiSe, from basic
inspection to ML-ready arrays.

Sections:
  1. Load & inspect                    — overview of columns and class balance
  2. Filter by metadata               — select series by type or feature
  3. Convert to numpy (X, y)          — pivot to (n_series, length) matrix
  4. Stratified train/test split       — scikit-learn compatible
  5. PyTorch Dataset & DataLoader      — Conv1d / LSTM ready tensors

By default this script loads the classification dataset produced by
05_classification_dataset.py. Change PARQUET_PATH to load any other file.

Run:
    python examples/06_load_and_use.py
"""

import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

# Classification dataset (multi-class, fixed length)
PARQUET_PATH = "generated-dataset/classification/betise_classification.parquet"

# To load a benchmark or feature-suite file instead, point here:
# PARQUET_PATH = "generated-dataset/benchmark/ar/ar_short.parquet"


# ── 1. Load & inspect ─────────────────────────────────────────────────────────

df = pd.read_parquet(PARQUET_PATH)

print("=== 1. Dataset overview ===")
print(f"  Total rows     : {len(df):,}")
print(f"  Unique series  : {df['series_id'].nunique():,}")
lengths = df.groupby("series_id").size()
print(f"  Series lengths : min={lengths.min()}  max={lengths.max()}  mean={lengths.mean():.1f}")
print(f"\n  Columns: {list(df.columns)}")
print()

if "primary_category" in df.columns:
    print("  Class distribution (primary_category):")
    dist = (
        df.drop_duplicates("series_id")["primary_category"]
        .value_counts()
        .sort_index()
    )
    for cls, cnt in dist.items():
        print(f"    {cls:<22} {cnt:>6,}")
    print()


# ── 2. Filter by metadata ─────────────────────────────────────────────────────
#
# Every series carries rich metadata columns. You can filter on any of them
# to build task-specific subsets.

print("=== 2. Filter examples ===")

# All AR series
ar_series_ids = df.loc[df["base_series"] == "ar", "series_id"].unique()
print(f"  AR base series count        : {len(ar_series_ids):,}")

# Series with a linear trend (sub_category == "trend" and label contains linear)
if "sub_category" in df.columns:
    trend_ids = df.loc[df["sub_category"] == "trend", "series_id"].unique()
    print(f"  Trend sub-category count    : {len(trend_ids):,}")

# Series that have an anomaly (anomaly_location is populated)
if "anomaly_location" in df.columns:
    anomaly_ids = df.loc[
        df["anomaly_location"].notna() & (df["anomaly_location"] != "None"),
        "series_id"
    ].unique()
    print(f"  Series with anomaly label   : {len(anomaly_ids):,}")

print()


# ── 3. Convert to numpy (X, y) ────────────────────────────────────────────────
#
# This section assumes all series have the same length (e.g. classification
# dataset with fixed_length=1000). For variable-length datasets use padding or
# truncation before pivoting.

if "primary_category" not in df.columns:
    print("Skipping numpy conversion (no primary_category column in this file).")
else:
    print("=== 3. Numpy conversion ===")

    CLASS_NAMES  = sorted(df["primary_category"].unique())
    CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
    IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}
    print(f"  Classes: {CLASS_TO_IDX}")

    # One row per series, columns = time steps
    series_meta = (
        df.drop_duplicates("series_id")
        [["series_id", "primary_category", "sub_category"]]
        .set_index("series_id")
    )

    values = (
        df[["series_id", "time", "data"]]
        .pivot(index="series_id", columns="time", values="data")
    )

    X = values.values.astype(np.float32)        # (n_series, length)
    y = (
        series_meta
        .loc[values.index, "primary_category"]
        .map(CLASS_TO_IDX)
        .values
    )

    print(f"  X shape : {X.shape}   dtype: {X.dtype}")
    print(f"  y shape : {y.shape}   unique labels: {np.unique(y)}")
    print()


# ── 4. Stratified train / test split ─────────────────────────────────────────

    print("=== 4. Train / test split ===")
    try:
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        print(f"  Train: {X_train.shape}   Test: {X_test.shape}")
        print(f"  Test class counts: {np.bincount(y_test)}")
        print()

    except ImportError:
        print("  scikit-learn not installed — skipping split example.")
        X_train, X_test, y_train, y_test = X, X, y, y
        print()


# ── 5. PyTorch Dataset & DataLoader ──────────────────────────────────────────

    print("=== 5. PyTorch DataLoader ===")
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader

        class TimeSeriesDataset(Dataset):
            """Wraps (X, y) arrays as a PyTorch Dataset.

            X is reshaped to (n, 1, length) so it is immediately compatible with
            Conv1d layers. For LSTM / Transformer models transpose to (n, length, 1).
            """

            def __init__(self, X: np.ndarray, y: np.ndarray):
                self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (n, 1, L)
                self.y = torch.tensor(y, dtype=torch.long)

            def __len__(self) -> int:
                return len(self.y)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        train_ds = TimeSeriesDataset(X_train, y_train)
        test_ds  = TimeSeriesDataset(X_test,  y_test)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=0)

        batch_X, batch_y = next(iter(train_loader))
        print(f"  Batch X : {tuple(batch_X.shape)}   (batch, channels, length)")
        print(f"  Batch y : {tuple(batch_y.shape)}")
        print(f"  Class map: {IDX_TO_CLASS}")
        print()

    except ImportError:
        print("  PyTorch not installed — skipping DataLoader example.")
        print()

print("Done.")
