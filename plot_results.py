"""
plot_results_pub.py

Publication-quality plotting of model metrics.

Improvements:
- Clean academic style (less clutter)
- Better color palette and marker visibility
- Log-scale formatting improvements
- Optional trend lines per architecture
- Saves both PNG and PDF (vector graphics)
- RAM plotted in KB instead of bytes
- Linear RAM axis with clean tick spacing (more readable for small ranges)
"""

from pathlib import Path
import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import LogLocator

# -----------------------------
# Global style (publication-ready)
# -----------------------------
sns.set_theme(style="ticks")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
})

PALETTE = sns.color_palette("colorblind")

# -----------------------------
# Helpers
# -----------------------------
def extract_arch(name: str) -> str:
    m = re.match(r"([A-Za-z]+)", str(name))
    return m.group(1).upper() if m else "OTHER"


def prepare_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    for col in ["Parameters", "Last Time (ms)", "Maximum Time (ms)", "RAM (bytes)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["arch"] = df["Model"].apply(extract_arch)
    return df


def add_trend_lines(ax, df, x_col, y_col):
    """Add simple log-log regression lines per architecture"""
    for arch, sub in df.groupby("arch"):
        if len(sub) < 2:
            continue

        x = np.log10(sub[x_col])
        y = np.log10(sub[y_col])

        coeffs = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = np.polyval(coeffs, x_fit)

        ax.plot(10**x_fit, 10**y_fit, linestyle="--", linewidth=1, alpha=0.6)


# -----------------------------
# Plot functions
# -----------------------------
def plot_time_vs_params(df, outpath, cycle_time_ms=None, trend=False):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    time_col = "Last Time (ms)" if "Last Time (ms)" in df.columns else "Maximum Time (ms)"

    if cycle_time_ms:
        df["y_plot"] = df[time_col] / cycle_time_ms
        ylabel = "Time / cycle time"
    else:
        df["y_plot"] = df[time_col]
        ylabel = "Inference time (ms)"

    df_sorted = df.sort_values(["arch", "Parameters"])

    for arch, sub in df_sorted.groupby("arch"):
        ax.plot(
            sub["Parameters"],
            sub["y_plot"],
            marker="o",
            markersize=6,
            linewidth=1.2,
            alpha=0.8,
            label=arch
        )

    if trend:
        add_trend_lines(ax, df_sorted, "Parameters", "y_plot")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\mathcal{P}$ (parameters)")
    ax.set_ylabel(ylabel)
    ax.set_title("Inference Time vs Model Size")

    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))

    sns.despine()
    fig.tight_layout()

    fig.savefig(outpath.with_suffix(".png"))
    fig.savefig(outpath.with_suffix(".pdf"))
    plt.close(fig)


def plot_ram_vs_params(df, outpath, trend=False):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    df_sorted = df.sort_values(["arch", "Parameters"])

    # Convert bytes → KB
    df_sorted["RAM (KB)"] = df_sorted["RAM (bytes)"] / 1024.0

    for arch, sub in df_sorted.groupby("arch"):
        ax.plot(
            sub["Parameters"],
            sub["RAM (KB)"],
            marker="o",
            markersize=6,
            linewidth=1.2,
            alpha=0.75,
            label=arch
        )

    if trend:
        add_trend_lines(ax, df_sorted, "Parameters", "RAM (KB)")

    ax.set_xscale("log")
    ax.set_yscale("linear")

    ax.set_xlabel(r"$\mathcal{P}$ (parameters)")
    ax.set_ylabel("RAM (KB)")
    ax.set_title("Memory Usage vs Model Size")

    # Clean linear ticks
    y_min = df_sorted["RAM (KB)"].min()
    y_max = df_sorted["RAM (KB)"].max()

    step = 5  # KB

    # Start ticks at nearest lower multiple of step
    tick_start = step * np.floor(y_min / step)
    tick_end = step * np.ceil(y_max / step)

    ticks = np.arange(tick_start, tick_end + step, step)

    ax.set_ylim(tick_start, tick_end)
    ax.set_yticks(ticks)

    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))

    sns.despine()
    fig.tight_layout()

    fig.savefig(outpath.with_suffix(".png"))
    fig.savefig(outpath.with_suffix(".pdf"))
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="model_metrics.csv")
    parser.add_argument("--outdir", default="plots")
    parser.add_argument("--cycle-time-ms", type=float, default=None)
    parser.add_argument("--trend", action="store_true", help="Add trend lines")
    args = parser.parse_args()

    df = prepare_df(Path(args.csv))
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    plot_time_vs_params(
        df,
        outdir / "time_vs_params",
        cycle_time_ms=args.cycle_time_ms,
        trend=args.trend
    )

    plot_ram_vs_params(
        df,
        outdir / "ram_vs_params",
        trend=args.trend
    )

    print(f"Saved plots to {outdir} (PNG + PDF)")


if __name__ == "__main__":
    main()