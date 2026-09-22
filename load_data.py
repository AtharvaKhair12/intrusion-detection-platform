"""
load_data.py — CICIDS2017 CSV loader and cleaner.

Reads every CSV in data/raw/, standardises column names (strip
whitespace, lowercase, spaces → underscores), replaces inf/-inf with
NaN, drops affected rows, and writes a single combined parquet file to
data/processed/cicids2017.parquet.

Data-quality note (inf handling)
--------------------------------
CICIDS2017 contains inf/-inf values primarily in the "Flow Bytes/s" and
"Flow Packets/s" columns.  These arise from division-by-zero when flow
duration is 0.  We **drop** those rows rather than impute because:
  1. The values are fundamentally undefined (division by zero), so any
     imputed number would be fabricated.
  2. The affected rows are a tiny fraction of the ~2.8 M total records —
     removing them has negligible impact on dataset size or class balance.
  3. Dropping keeps the pipeline simple and avoids introducing silent
     data artefacts that could confuse the Isolation Forest.
"""

import os
import glob
import numpy as np
import pandas as pd

RAW_DIR = os.path.join("data", "raw", "archive")
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "cicids2017.parquet")


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, lowercase, and replace spaces with underscores."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def load_and_combine(raw_dir: str) -> pd.DataFrame:
    """Read all CSVs in *raw_dir* into a single DataFrame."""
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir!r}.  "
            "Download the CICIDS2017 dataset and place the CSVs there."
        )

    frames = []
    for path in csv_files:
        print(f"  Reading {os.path.basename(path)} ...")
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        df = standardise_columns(df)
        frames.append(df)
        print(f"    -> {len(df):,} rows, {len(df.columns)} columns")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined: {len(combined):,} rows, {len(combined.columns)} columns")
    return combined


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/-inf with NaN, then drop rows that contain any NaN."""
    n_before = len(df)

    # Count inf values per column for logging
    inf_mask = df.isin([np.inf, -np.inf])
    inf_counts = inf_mask.sum()
    inf_cols = inf_counts[inf_counts > 0]
    if not inf_cols.empty:
        print("\nInf values found per column:")
        for col, count in inf_cols.items():
            print(f"  {col}: {count:,}")

    # Replace inf → NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Count total NaN rows (including any that were already NaN)
    nan_rows = df.isna().any(axis=1).sum()
    print(f"\nRows with any NaN (including former inf): {nan_rows:,}")

    # Drop
    df = df.dropna()
    n_after = len(df)
    print(f"Rows dropped: {n_before - n_after:,}  ({(n_before - n_after) / n_before * 100:.2f}%)")
    print(f"Rows remaining: {n_after:,}")
    return df.reset_index(drop=True)


def main():
    print("=" * 60)
    print("CICIDS2017 — Load & Clean")
    print("=" * 60)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"\nLoading CSVs from {RAW_DIR}/")
    df = load_and_combine(RAW_DIR)

    print("\n--- Cleaning ---")
    df = clean(df)

    print(f"\nSaving parquet -> {OUTPUT_PATH}")
    df.to_parquet(OUTPUT_PATH, index=False)
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Done. File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
