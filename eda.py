"""
eda.py — Exploratory Data Analysis for CICIDS2017.

Reads the cleaned parquet from data/processed/cicids2017.parquet and
generates eda_summary.md in the repo root containing:
  • Class balance  (BENIGN vs each attack type — counts and %)
  • Missing-value counts per column
  • Basic summary statistics (numeric columns)
"""

import os
import pandas as pd

PARQUET_PATH = os.path.join("data", "processed", "cicids2017.parquet")
OUTPUT_PATH = "eda_summary.md"

# The label column in CICIDS2017 (after our name standardisation)
LABEL_COL = "label"


def class_balance(df: pd.DataFrame) -> str:
    """Return a markdown table of class counts and percentages."""
    counts = df[LABEL_COL].value_counts()
    total = len(df)
    lines = [
        "## Class Balance\n",
        "| Label | Count | % of Total |",
        "|-------|------:|-----------:|",
    ]
    for label, count in counts.items():
        pct = count / total * 100
        lines.append(f"| {label} | {count:,} | {pct:.4f}% |")
    lines.append(f"| **Total** | **{total:,}** | **100%** |")
    lines.append("")
    return "\n".join(lines)


def missing_values(df: pd.DataFrame) -> str:
    """Return a markdown table of per-column missing-value counts."""
    mv = df.isna().sum()
    mv_nonzero = mv[mv > 0]

    lines = ["## Missing Values\n"]
    if mv_nonzero.empty:
        lines.append("No missing values in any column after cleaning.\n")
    else:
        lines.extend([
            "| Column | Missing Count |",
            "|--------|-------------:|",
        ])
        for col, count in mv_nonzero.items():
            lines.append(f"| {col} | {count:,} |")
        lines.append("")
    return "\n".join(lines)


def summary_statistics(df: pd.DataFrame) -> str:
    """Return basic summary statistics for numeric columns."""
    desc = df.describe().T
    # Round for readability
    desc = desc.round(4)

    lines = [
        "## Summary Statistics (numeric columns)\n",
        "| Column | count | mean | std | min | 25% | 50% | 75% | max |",
        "|--------|------:|-----:|----:|----:|----:|----:|----:|----:|",
    ]
    for col in desc.index:
        row = desc.loc[col]
        lines.append(
            f"| {col} "
            f"| {row['count']:.0f} "
            f"| {row['mean']:.4f} "
            f"| {row['std']:.4f} "
            f"| {row['min']:.4f} "
            f"| {row['25%']:.4f} "
            f"| {row['50%']:.4f} "
            f"| {row['75%']:.4f} "
            f"| {row['max']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    print(f"Loading {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns\n")

    sections = [
        "# CICIDS2017 — Exploratory Data Analysis\n",
        f"Dataset: **{len(df):,}** rows × **{len(df.columns)}** columns "
        f"(after cleaning).\n",
        class_balance(df),
        missing_values(df),
        summary_statistics(df),
    ]

    report = "\n".join(sections)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"EDA report saved -> {OUTPUT_PATH}")
    # Also print the class balance to stdout for quick review
    print("\n" + class_balance(df))


if __name__ == "__main__":
    main()
