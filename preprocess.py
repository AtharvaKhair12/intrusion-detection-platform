"""
preprocess.py — Feature selection, scaling, and train/test splitting
for Isolation Forest anomaly detection on CICIDS2017.

Pipeline:
  1. Load the cleaned parquet (output of load_data.py).
  2. Drop non-numeric / metadata columns (timestamps, IPs, ports, label).
  3. Scale all remaining numeric features with StandardScaler.
  4. Split:
       • Train  — 80 % of BENIGN samples (Isolation Forest trains on
                  "normal" traffic only).
       • Test   — remaining 20 % BENIGN + ALL attack samples, with
                  binary labels (0 = benign, 1 = attack).
  5. Save to data/processed/:
       • X_train_benign.npy   — scaled benign-only training features
       • X_test.npy           — scaled mixed test features
       • y_test.npy           — binary labels for the test set
       • scaler.joblib        — fitted StandardScaler (reused at inference)
       • feature_names.json   — ordered list of feature column names
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PARQUET_PATH = os.path.join("data", "processed", "cicids2017.parquet")
PROCESSED_DIR = os.path.join("data", "processed")

LABEL_COL = "label"

# Columns to explicitly drop even if they look numeric.
# These are metadata, not flow features.
DROP_COLS = {
    "destination_port",   # categorical (well-known ports), not a flow stat
    "fwd_header_length.1",  # duplicate column in some CICIDS2017 versions
}


def select_numeric_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return only numeric flow-feature columns (no label, no metadata)."""
    # Start with all numeric columns
    numeric = df.select_dtypes(include=[np.number])

    # Drop metadata columns that slipped through
    to_drop = [c for c in DROP_COLS if c in numeric.columns]
    if to_drop:
        numeric = numeric.drop(columns=to_drop)

    feature_names = list(numeric.columns)
    print(f"Selected {len(feature_names)} numeric features")
    return numeric, feature_names


def main():
    print("=" * 60)
    print("CICIDS2017 — Preprocess for Isolation Forest")
    print("=" * 60)

    # --- Load ---------------------------------------------------------------
    print(f"\nLoading {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # --- Labels -------------------------------------------------------------
    labels = df[LABEL_COL].str.strip()
    is_benign = labels == "BENIGN"
    # Binary: 0 = benign, 1 = attack
    y_binary = (~is_benign).astype(int).values

    n_benign = is_benign.sum()
    n_attack = (~is_benign).sum()
    print(f"\n  Benign : {n_benign:,}")
    print(f"  Attack : {n_attack:,}")

    # --- Feature selection --------------------------------------------------
    X_df, feature_names = select_numeric_features(df)
    X = X_df.values

    # --- Train / test split -------------------------------------------------
    # Isolate benign rows; split them 80/20
    benign_idx = np.where(is_benign)[0]
    attack_idx = np.where(~is_benign)[0]

    benign_train_idx, benign_test_idx = train_test_split(
        benign_idx, test_size=0.2, random_state=42
    )

    # Test set = 20 % benign + all attack
    test_idx = np.concatenate([benign_test_idx, attack_idx])
    np.random.seed(42)
    np.random.shuffle(test_idx)

    X_train_raw = X[benign_train_idx]
    X_test_raw = X[test_idx]
    y_test = y_binary[test_idx]

    print(f"\n  Train (benign-only) : {X_train_raw.shape[0]:,} samples")
    print(f"  Test  (mixed)       : {X_test_raw.shape[0]:,} samples")
    print(f"     > benign in test  : {(y_test == 0).sum():,}")
    print(f"     > attack in test  : {(y_test == 1).sum():,}")

    # --- Scaling ------------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    print(f"\n  Features scaled with StandardScaler")

    # --- Save ---------------------------------------------------------------
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    np.save(os.path.join(PROCESSED_DIR, "X_train_benign.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.joblib"))

    with open(os.path.join(PROCESSED_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"\n  Saved to {PROCESSED_DIR}/:")
    print(f"    X_train_benign.npy  {X_train.shape}")
    print(f"    X_test.npy          {X_test.shape}")
    print(f"    y_test.npy          {y_test.shape}")
    print(f"    scaler.joblib")
    print(f"    feature_names.json  ({len(feature_names)} features)")
    print("\nDone.")


if __name__ == "__main__":
    main()
