"""Analyze latency trends from model_metrics.csv.

This script computes:
- An overall affine fit: T = alpha * P + beta
- An overall proportional fit through the origin: T = alpha * P
- Per-architecture affine fits
- Relative latency shifts at the minimum and maximum parameter counts
- A normalized execution-cost plot styled after the supplied reference

Usage:
    python analyze_latency.py --csv model_metrics.csv
    python analyze_latency.py --csv model_metrics.csv --plots-outdir plots/latency_analysis
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter

try:
    from scipy.stats import linregress
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "scipy is required for this analysis. Install it with `pip install scipy`."
    ) from exc


PARAM_COL = "Parameters"
MODEL_COL = "Model"
LAST_TIME_COL = "Last Time (ms)"
MAX_TIME_COL = "Maximum Time (ms)"
MILLISECONDS_TO_MICROSECONDS = 1000.0
PLOT_DEFAULT_DIR = Path("plots") / "latency_analysis"

ARCH_MARKERS = {
    "CNN": "o",
    "GRU": "s",
    "LSTM": "^",
    "MLP": "D",
    "RESNET": "P",
}


plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


@dataclass(frozen=True)
class AffineFit:
    """Container for affine fit statistics."""

    alpha: float
    beta: float
    r2: float
    pearson_r: float | None = None


def extract_architecture(model_name: str) -> str:
    """Extract the architecture family prefix from a model name."""

    match = re.match(r"^([A-Za-z]+)", str(model_name))
    return match.group(1).upper() if match else "OTHER"


def select_time_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    """Pick the latency column to analyze."""

    if preferred is not None:
        if preferred not in df.columns:
            raise ValueError(f"Requested time column not found: {preferred}")
        return preferred

    if LAST_TIME_COL in df.columns:
        return LAST_TIME_COL
    if MAX_TIME_COL in df.columns:
        return MAX_TIME_COL

    raise ValueError(f"No latency column found. Expected {LAST_TIME_COL!r} or {MAX_TIME_COL!r}.")


def load_metrics(path: str, time_column: str | None = None) -> tuple[pd.DataFrame, str]:
    """Load the metrics CSV and normalize relevant columns."""

    df = pd.read_csv(path)
    df.columns = [column.strip() for column in df.columns]

    required = {MODEL_COL, PARAM_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    time_col = select_time_column(df, time_column)

    df[PARAM_COL] = pd.to_numeric(df[PARAM_COL], errors="coerce")
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[MODEL_COL, PARAM_COL, time_col]).copy()
    df = df[df[PARAM_COL] > 0].copy()

    df["Architecture"] = df[MODEL_COL].apply(extract_architecture)
    df["Latency (ms)"] = df[time_col]
    df["Latency (us)"] = df[time_col] * MILLISECONDS_TO_MICROSECONDS
    df["Latency per parameter (us/param)"] = df["Latency (us)"] / df[PARAM_COL]

    return df, time_col


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the coefficient of determination for a fitted model."""

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def fit_affine(x: np.ndarray, y: np.ndarray) -> AffineFit:
    """Fit T = alpha * P + beta using scipy.stats.linregress."""

    result = linregress(x, y)
    return AffineFit(
        alpha=float(result.slope),
        beta=float(result.intercept),
        r2=float(result.rvalue**2),
        pearson_r=float(result.rvalue),
    )


def fit_origin(x: np.ndarray, y: np.ndarray) -> AffineFit:
    """Fit T = alpha * P with intercept fixed at zero."""

    denominator = float(np.sum(x**2))
    if denominator == 0.0:
        raise ValueError("Cannot fit a proportional model when all parameter counts are zero.")

    alpha = float(np.sum(x * y) / denominator)
    y_pred = alpha * x
    return AffineFit(alpha=alpha, beta=0.0, r2=float(r_squared(y, y_pred)))


def format_time(value_ms: float) -> str:
    """Format latency in milliseconds and microseconds."""

    return f"{value_ms:.2f} ms ({value_ms * MILLISECONDS_TO_MICROSECONDS:.2f} µs)"


def print_fit(title: str, fit: AffineFit) -> None:
    """Print a fit in a readable format."""

    if title:
        print(title)
    print(
        f"  Marginal cost (alpha): {fit.alpha:.6f} ms/parameter "
        f"({fit.alpha * MILLISECONDS_TO_MICROSECONDS:.2f} µs/parameter)"
    )
    print(
        f"  Fixed cost (beta):     {fit.beta:.4f} ms "
        f"({fit.beta * MILLISECONDS_TO_MICROSECONDS:.2f} µs)"
    )
    print(f"  R^2:                   {fit.r2:.6f}")
    if fit.pearson_r is not None:
        print(f"  Pearson r:             {fit.pearson_r:.6f}")


def print_architecture_fits(df: pd.DataFrame) -> None:
    """Compute and print affine fits per architecture family."""

    print("\n=== Architecture-Specific Affine Fits ===")
    for architecture, group in df.groupby("Architecture", sort=True):
        x = group[PARAM_COL].to_numpy(dtype=float)
        y = group["Latency (ms)"].to_numpy(dtype=float)
        if len(group) < 2:
            print(f"\n{architecture}:")
            print("  Not enough points for a regression fit.")
            continue

        fit = fit_affine(x, y)
        print(f"\n{architecture}:")
        print(
            f"  Marginal cost (alpha): {fit.alpha:.6f} ms/parameter "
            f"({fit.alpha * MILLISECONDS_TO_MICROSECONDS:.2f} µs/parameter)"
        )
        print(
            f"  Fixed cost (beta):     {fit.beta:.4f} ms "
            f"({fit.beta * MILLISECONDS_TO_MICROSECONDS:.2f} µs)"
        )
        print(f"  R^2:                   {fit.r2:.6f}")


def print_relative_latency_shifts(df: pd.DataFrame, overall_fit: AffineFit) -> None:
    """Report fixed/variable latency shares at the observed parameter extremes."""

    print("\n=== Relative Latency Shifts ===")

    p_min = float(df[PARAM_COL].min())
    p_max = float(df[PARAM_COL].max())

    for label, p_value in (("P_min", p_min), ("P_max", p_max)):
        total_estimated_ms = overall_fit.alpha * p_value + overall_fit.beta
        fixed_share = (overall_fit.beta / total_estimated_ms) * 100.0 if total_estimated_ms != 0.0 else np.nan
        variable_ms = overall_fit.alpha * p_value
        print(f"{label}: {p_value:.0f} parameters")
        print(f"  Estimated latency: {format_time(total_estimated_ms)}")
        print(f"  Fixed latency:     {format_time(overall_fit.beta)}")
        print(f"  Variable latency:  {format_time(variable_ms)}")
        print(f"  Fixed-latency share: {fixed_share:.2f}%")

    if overall_fit.alpha == 0.0:
        print("  Intersection point: undefined (alpha is zero)")
    else:
        intersection = overall_fit.beta / overall_fit.alpha
        print(f"  Intersection point (beta = alpha * P): P = {intersection:.2f} parameters")


def print_per_parameter_summary(df: pd.DataFrame) -> None:
    """Summarize normalized latency by architecture."""

    print("\n=== Per-Parameter Execution Cost ===")
    summary = (
        df.groupby("Architecture")["Latency per parameter (us/param)"]
        .agg(["count", "mean", "median", "std"])
        .sort_index()
    )

    for architecture, row in summary.iterrows():
        print(
            f"{architecture:<10} "
            f"n={int(row['count']):<2d} "
            f"mean={row['mean']:.2f} µs/param "
            f"median={row['median']:.2f} µs/param "
            f"std={0.0 if np.isnan(row['std']) else row['std']:.2f}"
        )

    overall_mean = float(df["Latency per parameter (us/param)"].mean())
    overall_median = float(df["Latency per parameter (us/param)"].median())
    print(f"Overall mean:   {overall_mean:.2f} µs/param")
    print(f"Overall median:  {overall_median:.2f} µs/param")


def print_analysis(df: pd.DataFrame, overall_fit: AffineFit, origin_fit: AffineFit, time_col: str) -> None:
    """Print the textual analysis for a prepared dataframe."""

    print(f"=== Latency Analysis ({time_col}) ===")
    print_fit("", overall_fit)

    print("\n=== Overall Proportional Fit Through the Origin ===")
    print(
        f"  Marginal cost (alpha): {origin_fit.alpha:.6f} ms/parameter "
        f"({origin_fit.alpha * MILLISECONDS_TO_MICROSECONDS:.2f} µs/parameter)"
    )
    print(f"  R^2:                   {origin_fit.r2:.6f}")

    print_architecture_fits(df)
    print_relative_latency_shifts(df, overall_fit)
    print_per_parameter_summary(df)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save a figure as both PNG and PDF."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_execution_cost_by_architecture(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot normalized execution cost by architecture, styled like the reference figure."""

    plot_df = df.sort_values(["Architecture", PARAM_COL]).copy()
    per_param = plot_df["Latency per parameter (us/param)"]
    lower_band = float(per_param.quantile(0.25))
    upper_band = float(per_param.quantile(0.75))
    center = float(per_param.median())
    x_min = float(plot_df[PARAM_COL].min())
    x_max = float(plot_df[PARAM_COL].max())

    fig, ax = plt.subplots(figsize=(8.0, 5.7))
    colors = plt.get_cmap("tab10")

    for index, (architecture, group) in enumerate(plot_df.groupby("Architecture", sort=True)):
        marker = ARCH_MARKERS.get(architecture, "o")
        ax.scatter(
            group[PARAM_COL],
            group["Latency per parameter (us/param)"],
            s=70,
            marker=marker,
            color=colors(index % 10),
            edgecolors="white",
            linewidths=0.5,
            alpha=0.95,
            label=architecture,
            zorder=4,
        )

    ax.axhspan(lower_band, upper_band, color="0.85", alpha=0.75, label=f"Band (~{center:.1f} µs/param)")
    ax.axhline(center, color="0.35", linestyle=":", linewidth=1.8, zorder=3)

    ax.set_xscale("log")
    ax.set_xlim(x_min * 0.95, x_max * 1.05)
    ax.set_xlabel("P (parameters, log scale)")
    ax.set_ylabel("Execution cost (µs / parameter)")
    ax.set_title("Per-parameter execution cost by architecture")
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92)

    save_figure(fig, output_dir, "execution_cost_by_architecture")


def plot_latency_vs_params(df: pd.DataFrame, overall_fit: AffineFit, output_dir: Path) -> None:
    """Plot total latency versus parameters with the fitted affine model."""

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    colors = plt.get_cmap("tab10")

    df_sorted = df.sort_values(["Architecture", PARAM_COL])
    for index, (architecture, group) in enumerate(df_sorted.groupby("Architecture", sort=True)):
        marker = ARCH_MARKERS.get(architecture, "o")
        ax.plot(
            group[PARAM_COL],
            group["Latency (ms)"],
            marker=marker,
            markersize=8,
            linewidth=1.8,
            color=colors(index % 10),
            alpha=0.85,
            label=architecture,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Time vs Model Size")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)

    save_figure(fig, output_dir, "latency_vs_parameters")


def plot_family_fit_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot per-family marginal costs, fixed costs, and fit quality."""

    rows = []
    for architecture, group in df.groupby("Architecture", sort=True):
        x = group[PARAM_COL].to_numpy(dtype=float)
        y = group["Latency (ms)"].to_numpy(dtype=float)
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
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.5), sharey=True)

    axes[0].barh(fit_df["Architecture"], fit_df["Alpha"] / MILLISECONDS_TO_MICROSECONDS, color="#4C72B0")
    axes[0].set_title("Marginal cost")
    axes[0].set_xlabel("ms / parameter")

    axes[1].barh(fit_df["Architecture"], fit_df["Beta"], color="#B9B9B9")
    axes[1].set_title("Fixed cost")
    axes[1].set_xlabel("ms")

    axes[2].barh(fit_df["Architecture"], fit_df["R2"], color="#54A24B")
    axes[2].set_title("Fit quality")
    axes[2].set_xlabel("R²")
    axes[2].set_xlim(0.0, 1.05)

    for axis in axes:
        axis.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
        axis.invert_yaxis()

    fig.suptitle("Per-family affine fit summary", y=1.03, fontsize=14)
    save_figure(fig, output_dir, "family_fit_summary")


def generate_plots(df: pd.DataFrame, overall_fit: AffineFit, output_dir: str | Path = PLOT_DEFAULT_DIR) -> None:
    """Generate the full latency plot set."""

    output_path = Path(output_dir)
    plot_execution_cost_by_architecture(df, output_path)
    plot_latency_vs_params(df, overall_fit, output_path)
    plot_family_fit_summary(df, output_path)


def compute_fits(df: pd.DataFrame) -> tuple[AffineFit, AffineFit]:
    """Compute overall affine and proportional fits for latency."""

    x = df[PARAM_COL].to_numpy(dtype=float)
    y = df["Latency (ms)"].to_numpy(dtype=float)
    return fit_affine(x, y), fit_origin(x, y)


def analyze(path: str, time_column: str | None = None) -> tuple[pd.DataFrame, str, AffineFit, AffineFit]:
    """Run the full latency analysis and print the results."""

    df, time_col = load_metrics(path, time_column)
    overall_fit, origin_fit = compute_fits(df)
    print_analysis(df, overall_fit, origin_fit, time_col)
    return df, time_col, overall_fit, origin_fit


def analyze_and_plot(path: str, output_dir: str | Path = PLOT_DEFAULT_DIR, time_column: str | None = None) -> None:
    """Run the analysis and save plots for the same dataset."""

    df, _, overall_fit, _ = analyze(path, time_column=time_column)
    generate_plots(df, overall_fit, output_dir)
    print(f"\nSaved plots to {Path(output_dir)}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Analyze latency trends from model_metrics.csv")
    parser.add_argument("--csv", default="model_metrics.csv", help="Path to the metrics CSV file")
    parser.add_argument(
        "--time-column",
        default=None,
        help=f"Latency column to analyze (default: {LAST_TIME_COL} if present, otherwise {MAX_TIME_COL})",
    )
    parser.add_argument(
        "--plots-outdir",
        default=None,
        help="If set, generate plots and save them to this directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plots_outdir:
        analyze_and_plot(args.csv, args.plots_outdir, time_column=args.time_column)
    else:
        analyze(args.csv, time_column=args.time_column)


if __name__ == "__main__":
    main()