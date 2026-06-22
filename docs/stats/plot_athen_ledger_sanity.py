#!/usr/bin/env python3
"""Plot the Ledger sanity comparison generated for Athen."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = (
    ROOT
    / "src/matlab/data/"
    "result_wlnm_dir_neg_sweep_train_ratios_10-90_pseudo_properties_stats_Apocrita/"
    "statistical_tests/athen/ledger_sanity"
)
FIGURE_DIR = DATA_DIR / "figures"

METRIC_ORDER = [
    "Connectance",
    "MeanGenerality",
    "MeanVulnerability",
    "MeanTrophicHeight",
]
METRIC_LABELS = {
    "Connectance": "Connectance",
    "MeanGenerality": "Mean generality",
    "MeanVulnerability": "Mean vulnerability",
    "MeanTrophicHeight": "Mean trophic height",
}
METRIC_COLORS = {
    "Connectance": "#0077b6",
    "MeanGenerality": "#d97706",
    "MeanVulnerability": "#7b2cbf",
    "MeanTrophicHeight": "#2a9d8f",
}
ECOSYSTEM_LABELS = {
    "lakes": "Lakes",
    "marine": "Marine",
    "streams": "Streams",
    "terrestrial aboveground": "Terrestrial aboveground",
    "terrestrial belowground": "Terrestrial belowground",
}


def save(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def plot_overall(data: pd.DataFrame) -> None:
    data = data.set_index("Metric").loc[METRIC_ORDER].reset_index()
    y = np.arange(len(data))[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    for yi, (_, row) in zip(y, data.iterrows()):
        margin = row["LedgerRelativeSEMPercent"]
        ax.fill_betweenx(
            [yi - 0.34, yi + 0.34],
            -margin,
            margin,
            color="#b7dfc4",
            alpha=0.65,
        )
        mean = row["MeanRelativeErrorPercent"]
        lower = row["RelativeCI90LowerPercent"]
        upper = row["RelativeCI90UpperPercent"]
        ax.errorbar(
            mean,
            yi,
            xerr=[[mean - lower], [upper - mean]],
            fmt="o",
            color=METRIC_COLORS[row["Metric"]],
            capsize=4,
            markersize=8,
            linewidth=2,
        )
        ax.text(
            margin + 0.4,
            yi + 0.18,
            f"Ledger SEM: +/-{margin:.2f}%",
            fontsize=8.5,
            color="#356447",
        )

    ax.axvline(0, color="#343a40", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([METRIC_LABELS[item] for item in data["Metric"]])
    ax.set_xlabel("Mean relative error (%) with 90% CI")
    ax.set_title("TrainRatio 60: overall reconstruction error vs Ledger relative SEM")
    ax.grid(axis="x", alpha=0.22)
    save(fig, "ledger_train60_overall_relative_error.png")


def plot_ecosystems(data: pd.DataFrame) -> None:
    ecosystems = list(ECOSYSTEM_LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)

    for ax, metric in zip(axes.ravel(), METRIC_ORDER):
        subset = data[data["Metric"] == metric].set_index("EcosystemType").loc[ecosystems]
        y = np.arange(len(subset))[::-1]
        margin = float(subset["LedgerRelativeSEMPercent"].iloc[0])
        ax.axvspan(-margin, margin, color="#b7dfc4", alpha=0.55)
        ax.axvline(0, color="#343a40", linewidth=1)

        for yi, (ecosystem, row) in zip(y, subset.iterrows()):
            mean = row["MeanRelativeErrorPercent"]
            lower = row["RelativeCI90LowerPercent"]
            upper = row["RelativeCI90UpperPercent"]
            ax.errorbar(
                mean,
                yi,
                xerr=[[mean - lower], [upper - mean]],
                fmt="o",
                color=METRIC_COLORS[metric],
                capsize=3,
                linewidth=1.8,
            )

        ax.set_yticks(y)
        ax.set_yticklabels([ECOSYSTEM_LABELS[item] for item in subset.index])
        ax.set_title(f"{METRIC_LABELS[metric]} (Ledger +/-{margin:.2f}%)")
        ax.set_xlabel("Mean relative error (%) with 90% CI")
        ax.grid(axis="x", alpha=0.2)

    fig.suptitle("TrainRatio 60: reconstruction error by ecosystem", y=1.01)
    save(fig, "ledger_train60_ecosystem_relative_error.png")


def plot_heatmap(data: pd.DataFrame) -> None:
    ecosystems = list(ECOSYSTEM_LABELS)
    heat = (
        data.pivot(
            index="EcosystemType",
            columns="Metric",
            values="MeanRelativeErrorPercent",
        )
        .loc[ecosystems, METRIC_ORDER]
    )
    limit = float(np.nanmax(np.abs(heat.to_numpy())))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    image = ax.imshow(heat.to_numpy(), cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(METRIC_ORDER)))
    ax.set_xticklabels([METRIC_LABELS[item] for item in METRIC_ORDER])
    ax.set_yticks(range(len(ecosystems)))
    ax.set_yticklabels([ECOSYSTEM_LABELS[item] for item in ecosystems])
    ax.set_title("TrainRatio 60: mean relative reconstruction error by ecosystem")

    for row_index in range(heat.shape[0]):
        for column_index in range(heat.shape[1]):
            value = heat.iloc[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:+.1f}%",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.5 * limit else "#202020",
                fontweight="bold",
            )

    fig.colorbar(image, ax=ax, label="Mean relative error (%)")
    save(fig, "ledger_train60_ecosystem_error_heatmap.png")


def main() -> None:
    overall = pd.read_csv(DATA_DIR / "ledger_sanity_train60_overall.csv")
    ecosystems = pd.read_csv(DATA_DIR / "ledger_sanity_train60_by_ecosystem.csv")
    plot_overall(overall)
    plot_ecosystems(ecosystems)
    plot_heatmap(ecosystems)


if __name__ == "__main__":
    main()
