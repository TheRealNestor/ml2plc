"""Analyze model memory cost models from model_metrics.csv.

This script computes:
- An overall affine fit: M = alpha * P + beta
- An overall proportional fit through the origin: M = alpha * P
- Per-architecture affine fits
- Relative fixed-cost shifts at the minimum and maximum parameter counts
- Memory values reported in both decimal kB and binary KiB

Usage:
    python analyze_memory_costs.py --csv model_metrics.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import linregress
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "scipy is required for this analysis. Install it with `pip install scipy`."
    ) from exc


PARAM_COL = "Parameters"
MEMORY_COL = "RAM (bytes)"
MODEL_COL = "Model"
DECIMAL_KB = 1000.0
BINARY_KIB = 1024.0
PLOT_DEFAULT_DIR = Path("plots") / "memory_costs"


@dataclass(frozen=True)
class AffineFit:
    """Container for affine fit statistics."""

    alpha: float
    beta: float
    r2: float
    pearson_r: float | None = None


def load_metrics(path: str) -> pd.DataFrame:
    """Load the metrics CSV and normalize the relevant columns."""

    df = pd.read_csv(path)
    df.columns = [column.strip() for column in df.columns]

    required = {MODEL_COL, PARAM_COL, MEMORY_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df[PARAM_COL] = pd.to_numeric(df[PARAM_COL], errors="coerce")
    df[MEMORY_COL] = pd.to_numeric(df[MEMORY_COL], errors="coerce")
    df = df.dropna(subset=[MODEL_COL, PARAM_COL, MEMORY_COL]).copy()

    df["Architecture"] = df[MODEL_COL].astype(str).str.extract(r"^([A-Za-z]+)", expand=False)
    df["Architecture"] = df["Architecture"].fillna(df[MODEL_COL].astype(str)).str.upper()

    return df


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the coefficient of determination for a fitted model."""

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def fit_affine(x: np.ndarray, y: np.ndarray) -> AffineFit:
    """Fit M = alpha * P + beta using scipy.stats.linregress."""

    result = linregress(x, y)
    return AffineFit(
        alpha=float(result.slope),
        beta=float(result.intercept),
        r2=float(result.rvalue**2),
        pearson_r=float(result.rvalue),
    )


def fit_origin(x: np.ndarray, y: np.ndarray) -> AffineFit:
    """Fit M = alpha * P with the intercept fixed at zero."""

    denominator = float(np.sum(x**2))
    if denominator == 0.0:
        raise ValueError("Cannot fit a proportional model when all parameter counts are zero.")

    alpha = float(np.sum(x * y) / denominator)
    y_pred = alpha * x
    return AffineFit(alpha=alpha, beta=0.0, r2=float(r_squared(y, y_pred)))


def print_fit(title: str, fit: AffineFit) -> None:
    """Print a fit in a readable format."""

    if title:
        print(title)
    print(
        "  Marginal cost (alpha): "
        f"{fit.alpha:.6f} bytes/parameter "
        f"({fit.alpha / DECIMAL_KB:.6f} kB/parameter, "
        f"{fit.alpha / BINARY_KIB:.6f} KiB/parameter)"
    )
    print(
        f"  Fixed cost (beta):     {fit.beta:.2f} bytes "
        f"({fit.beta / DECIMAL_KB:.2f} kB, {fit.beta / BINARY_KIB:.2f} KiB)"
    )
    print(f"  R^2:                   {fit.r2:.6f}")
    if fit.pearson_r is not None:
        print(f"  Pearson r:             {fit.pearson_r:.6f}")


def format_memory(value_bytes: float) -> str:
    """Format a memory value in bytes, decimal kB, and binary KiB."""

    return (
        f"{value_bytes:.2f} bytes "
        f"({value_bytes / DECIMAL_KB:.2f} kB, {value_bytes / BINARY_KIB:.2f} KiB)"
    )


def prepare_plot_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a plotting frame with helper columns and stable ordering."""

    plot_df = df.copy()
    plot_df["RAM (decimal kB)"] = plot_df[MEMORY_COL] / DECIMAL_KB
    plot_df["RAM (binary KiB)"] = plot_df[MEMORY_COL] / BINARY_KIB
    return plot_df.sort_values(["Architecture", PARAM_COL])


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save a figure as both PNG and PDF."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_cost_decomposition(df: pd.DataFrame, overall_fit: AffineFit, output_dir: Path) -> None:
    """Plot fixed and variable memory cost components against measured RAM."""

    plot_df = prepare_plot_frame(df)
    x_min = float(plot_df[PARAM_COL].min())
    x_max = float(plot_df[PARAM_COL].max())
    x_line = np.linspace(x_min, x_max, 400)

    fixed_kb = overall_fit.beta / DECIMAL_KB
    variable_kb = (overall_fit.alpha * x_line) / DECIMAL_KB
    total_kb = variable_kb + fixed_kb
    crossover = overall_fit.beta / overall_fit.alpha if overall_fit.alpha != 0.0 else np.nan

    measured_plotted = False

    fig, ax = plt.subplots(figsize=(8.4, 5.6))

    fixed_band = ax.fill_between(
        x_line,
        0.0,
        fixed_kb,
        color="#B9B9B9",
        alpha=0.90,
        label=f"Fixed overhead (~{fixed_kb:.2f} KB)",
    )
    variable_band = ax.fill_between(
        x_line,
        fixed_kb,
        total_kb,
        color="#4C72B0",
        alpha=0.88,
        label=f"Parameter storage (~{overall_fit.alpha:.2f} B/param)",
    )
    total_line, = ax.plot(x_line, total_kb, color="#325C9E", linewidth=0.0)

    for architecture, group in plot_df.groupby("Architecture", sort=True):
        ax.scatter(
            group[PARAM_COL],
            group[MEMORY_COL] / DECIMAL_KB,
            s=72,
            color="black",
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
            label="Measured RAM" if not measured_plotted else None,
        )
        measured_plotted = True

    if np.isfinite(crossover):
        ax.axvline(crossover, color="black", linestyle=":", linewidth=1.8)
        ax.annotate(
            f"crossover\n~{crossover:.0f} params",
            xy=(crossover, fixed_kb * 0.35),
            xytext=(crossover * 1.05, max(0.8, fixed_kb * 0.28)),
            ha="left",
            va="center",
            fontsize=12,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.0),
        )

    ax.set_xlim(x_min * 0.95, x_max * 1.02)
    ax.set_ylim(0.0, max(total_kb) * 1.06)
    ax.set_xlabel("P (parameters)")
    ax.set_ylabel("RAM (KB)")
    ax.set_title("Fixed vs. parameter-driven memory cost")
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(loc="upper left", frameon=True, framealpha=0.90)

    save_figure(fig, output_dir, "memory_cost_decomposition")


def plot_family_fit_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot per-family marginal and fixed costs as a compact summary."""

    rows = []
    for architecture, group in df.groupby("Architecture", sort=True):
        x = group[PARAM_COL].to_numpy(dtype=float)
        y = group[MEMORY_COL].to_numpy(dtype=float)
        if len(group) < 2:
            continue
        fit = fit_affine(x, y)
        rows.append(
            {
                "Architecture": architecture,
                "Alpha": fit.alpha,
                "Beta": fit.beta,
                "R2": fit.r2,
            }
        )

    if not rows:
        return

    fit_df = pd.DataFrame(rows).sort_values("Architecture")
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.3), sharex=False)

    axes[0].barh(fit_df["Architecture"], fit_df["Alpha"] / DECIMAL_KB, color="#4C72B0")
    axes[0].set_title("Marginal cost")
    axes[0].set_xlabel("kB / 1k parameters")
    axes[0].grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.4)

    axes[1].barh(fit_df["Architecture"], fit_df["Beta"] / DECIMAL_KB, color="#B9B9B9")
    axes[1].set_title("Fixed cost")
    axes[1].set_xlabel("kB")
    axes[1].grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.4)

    axes[2].barh(fit_df["Architecture"], fit_df["R2"], color="#54A24B")
    axes[2].set_title("Fit quality")
    axes[2].set_xlabel("R²")
    axes[2].set_xlim(0.0, 1.05)
    axes[2].grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.4)

    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    axes[2].invert_yaxis()

    fig.suptitle("Per-family affine fit summary", y=1.03, fontsize=14)
    save_figure(fig, output_dir, "family_fit_summary")


def generate_plots(df: pd.DataFrame, overall_fit: AffineFit, output_dir: str | Path = PLOT_DEFAULT_DIR) -> None:
    """Generate a small set of diagnostic plots for the memory-cost model."""

    output_path = Path(output_dir)
    plot_cost_decomposition(df, overall_fit, output_path)
    plot_family_fit_summary(df, output_path)


def compute_fits(df: pd.DataFrame) -> tuple[AffineFit, AffineFit]:
    """Compute the overall affine and proportional fits for a dataset."""

    x = df[PARAM_COL].to_numpy(dtype=float)
    y = df[MEMORY_COL].to_numpy(dtype=float)
    return fit_affine(x, y), fit_origin(x, y)


def print_analysis(df: pd.DataFrame, overall_fit: AffineFit, origin_fit: AffineFit) -> None:
    """Print the textual analysis for a prepared dataframe."""

    print("=== Overall Affine Fit ===")
    print_fit("", overall_fit)

    print("\n=== Overall Proportional Fit Through the Origin ===")
    print(f"  Marginal cost (alpha): {origin_fit.alpha:.6f} bytes/parameter")
    print(f"  R^2:                   {origin_fit.r2:.6f}")

    print_architecture_fits(df)
    print_relative_cost_shifts(df, overall_fit)


def print_architecture_fits(df: pd.DataFrame) -> None:
    """Compute and print affine fits per architecture family."""

    print("\n=== Architecture-Specific Affine Fits ===")
    for architecture, group in df.groupby("Architecture", sort=True):
        x = group[PARAM_COL].to_numpy(dtype=float)
        y = group[MEMORY_COL].to_numpy(dtype=float)
        if len(group) < 2:
            print(f"\n{architecture}:")
            print("  Not enough points for a regression fit.")
            continue

        fit = fit_affine(x, y)
        print(f"\n{architecture}:")
        print(
            "  Marginal cost (alpha): "
            f"{fit.alpha:.6f} bytes/parameter "
            f"({fit.alpha / DECIMAL_KB:.6f} kB/parameter, "
            f"{fit.alpha / BINARY_KIB:.6f} KiB/parameter)"
        )
        print(
            f"  Fixed cost (beta):     {fit.beta:.2f} bytes "
            f"({fit.beta / DECIMAL_KB:.2f} kB, {fit.beta / BINARY_KIB:.2f} KiB)"
        )
        print(f"  R^2:                   {fit.r2:.6f}")


def print_relative_cost_shifts(df: pd.DataFrame, overall_fit: AffineFit) -> None:
    """Report fixed-cost ratios at the observed parameter extremes."""

    print("\n=== Relative Cost Shifts ===")

    p_min = float(df[PARAM_COL].min())
    p_max = float(df[PARAM_COL].max())

    for label, p_value in (("P_min", p_min), ("P_max", p_max)):
        total_estimated = overall_fit.alpha * p_value + overall_fit.beta
        fixed_share = (overall_fit.beta / total_estimated) * 100.0 if total_estimated != 0.0 else np.nan
        print(f"{label}: {p_value:.0f} parameters")
        print(f"  Estimated memory: {format_memory(total_estimated)}")
        fixed_cost = overall_fit.beta
        variable_cost = overall_fit.alpha * p_value
        print(f"  Fixed cost: {format_memory(fixed_cost)}")
        print(f"  Variable cost: {format_memory(variable_cost)}")
        print(f"  Fixed-cost share: {fixed_share:.2f}%")

    if overall_fit.alpha == 0.0:
        print("  Intersection point: undefined (alpha is zero)")
    else:
        intersection = overall_fit.beta / overall_fit.alpha
        print(f"  Intersection point (beta = alpha * P): P = {intersection:.2f} parameters")


def analyze(path: str) -> None:
    """Run the full memory-cost analysis and print the results."""

    df = load_metrics(path)
    overall_fit, origin_fit = compute_fits(df)
    print_analysis(df, overall_fit, origin_fit)


def analyze_and_plot(path: str, output_dir: str | Path = PLOT_DEFAULT_DIR) -> None:
    """Run the analysis and save plots for the same dataset."""

    df = load_metrics(path)
    overall_fit, origin_fit = compute_fits(df)
    print_analysis(df, overall_fit, origin_fit)
    generate_plots(df, overall_fit, output_dir)
    print(f"\nSaved plots to {Path(output_dir)}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Analyze memory cost models from model_metrics.csv")
    parser.add_argument("--csv", default="model_metrics.csv", help="Path to the metrics CSV file")
    parser.add_argument(
        "--plots-outdir",
        default=None,
        help="If set, generate plots and save them to this directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plots_outdir:
        analyze_and_plot(args.csv, args.plots_outdir)
    else:
        analyze(args.csv)


if __name__ == "__main__":
    main()