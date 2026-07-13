"""
analyze_correlations.py

Compute and verify correlations between model metrics.

Usage:
    python analyze_correlations.py --csv model_metrics.csv
"""

import argparse
import pandas as pd


def load_df(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Ensure numeric columns
    numeric_cols = [
        "Parameters",
        "RAM (bytes)",
        "Last Time (ms)",
        "Maximum Time (ms)",
        "ST size (bytes)"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_latency_column(df):
    if "Last Time (ms)" in df.columns:
        return "Last Time (ms)"
    elif "Maximum Time (ms)" in df.columns:
        return "Maximum Time (ms)"
    else:
        raise ValueError("No latency column found")


def compute_and_print(df):
    latency_col = get_latency_column(df)

    print("\n=== Correlation Analysis ===\n")

    def corr(a, b, name):
        value = df[[a, b]].dropna().corr().iloc[0, 1]
        print(f"{name:<40}: {value:.4f}")
        return value

    # Core correlations
    c_param_ram = corr("Parameters", "RAM (bytes)", "Parameters ↔ RAM")
    c_param_lat = corr("Parameters", latency_col, "Parameters ↔ Latency")
    c_lat_ram   = corr(latency_col, "RAM (bytes)", "Latency ↔ RAM")

    # Optional ST size
    if "ST size (bytes)" in df.columns:
        c_param_st = corr("Parameters", "ST size (bytes)", "Parameters ↔ ST size")
        c_lat_st   = corr(latency_col, "ST size (bytes)", "Latency ↔ ST size")
    else:
        print("\n(ST size not present in dataset)")

    print("\n=== Interpretation ===\n")

    print(f"Parameters ↔ RAM: {c_param_ram:.4f}")
    if c_param_ram > 0.99:
        print("→ Near-perfect linear scaling (memory dominated by parameters)\n")

    print(f"Parameters ↔ Latency: {c_param_lat:.4f}")
    if c_param_lat > 0.8:
        print("→ Strong scaling, but not purely linear (architecture effects present)\n")

    print(f"Latency ↔ RAM: {c_lat_ram:.4f}")
    if c_lat_ram > 0.8:
        print("→ Memory and compute scale together, but not perfectly\n")

    print("Done.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="model_metrics.csv")
    args = parser.parse_args()

    df = load_df(args.csv)
    compute_and_print(df)


if __name__ == "__main__":
    main()