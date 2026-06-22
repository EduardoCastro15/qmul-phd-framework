#!/usr/bin/env python3
"""Generate figures for the Athen web-level equivalence analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[2]
ATHEN_DIR = (
    ROOT
    / "src/matlab/data/"
    "result_wlnm_dir_neg_sweep_train_ratios_10-90_pseudo_properties_stats_Apocrita/"
    "statistical_tests/athen"
)
FIGURE_DIR = ATHEN_DIR / "figures"

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


def save(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def selected_train60(eq60: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (eq60["AnalysisPopulation"] == "PrimaryMean20")
        | (eq60["AnalysisPopulation"] == "ConditionalAtLeast10ValidRuns")
        | (
            (eq60["Metric"] == "MeanTrophicHeight")
            & (eq60["AnalysisPopulation"] == "Complete20Sensitivity")
        )
    )
    selected = eq60.loc[mask].copy()
    selected["DisplayMetric"] = selected["Metric"].map(METRIC_LABELS)
    selected.loc[
        selected["AnalysisPopulation"] == "ConditionalAtLeast10ValidRuns",
        "DisplayMetric",
    ] = "Mean trophic height (>=10 valid runs)"
    selected.loc[
        selected["AnalysisPopulation"] == "Complete20Sensitivity",
        "DisplayMetric",
    ] = "Mean trophic height (20/20 sensitivity)"
    return selected


def plot_train60_forest(selected: pd.DataFrame) -> None:
    forest = selected[selected["MarginPercent"] == 20].copy()
    order = [
        "Connectance",
        "Mean generality",
        "Mean vulnerability",
        "Mean trophic height (>=10 valid runs)",
        "Mean trophic height (20/20 sensitivity)",
    ]
    forest["PlotOrder"] = forest["DisplayMetric"].map(
        {name: index for index, name in enumerate(order)}
    )
    forest = forest.sort_values("PlotOrder", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.axvspan(-15, 15, color="#d8f3dc", alpha=0.45, label="+/-15% margin")
    ax.axvline(-10, color="#d97706", linestyle="--", linewidth=1.2)
    ax.axvline(10, color="#d97706", linestyle="--", linewidth=1.2, label="+/-10% margin")
    ax.axvline(-20, color="#6c757d", linestyle=":", linewidth=1.2)
    ax.axvline(20, color="#6c757d", linestyle=":", linewidth=1.2, label="+/-20% margin")
    ax.axvline(0, color="#343a40", linewidth=1)

    for y, (_, row) in enumerate(forest.iterrows()):
        mean = row["MeanRelativeErrorPercent"]
        lower = row["RelativeCI90LowerPercent"]
        upper = row["RelativeCI90UpperPercent"]
        color = METRIC_COLORS[row["Metric"]]
        ax.errorbar(
            mean,
            y,
            xerr=[[mean - lower], [upper - mean]],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=8,
            capsize=4,
            linewidth=2,
        )

    ax.set_yticks(range(len(forest)))
    ax.set_yticklabels(forest["DisplayMetric"])
    ax.set_xlabel("Mean relative error: (mean pseudo - empirical) / empirical (%)")
    ax.set_title("TrainRatio 60: reconstruction bias and 90% confidence intervals")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "athen_train60_relative_error_forest.png")


def plot_train60_sensitivity(selected: pd.DataFrame) -> None:
    order = [
        "Connectance",
        "Mean generality",
        "Mean vulnerability",
        "Mean trophic height (>=10 valid runs)",
        "Mean trophic height (20/20 sensitivity)",
    ]
    heat = selected.pivot(
        index="DisplayMetric", columns="MarginPercent", values="Equivalent"
    ).reindex(order)[[10.0, 15.0, 20.0, 30.0]]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.imshow(
        heat.values,
        cmap=ListedColormap(["#f4b8bd", "#b7dfc4"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels([f"+/-{int(value)}%" for value in heat.columns])
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title("TrainRatio 60: equivalence sensitivity")

    for row_index in range(heat.shape[0]):
        for column_index in range(heat.shape[1]):
            value = int(heat.iloc[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                "Equivalent" if value else "Not equivalent",
                ha="center",
                va="center",
                fontsize=9,
                color="#174c2b" if value else "#721c24",
                fontweight="bold",
            )

    ax.set_xticks(np.arange(-0.5, len(heat.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(heat.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    save(fig, "athen_train60_equivalence_sensitivity.png")


def plot_all_ratios(eq_all: pd.DataFrame) -> None:
    trend = eq_all[
        (eq_all["MarginPercent"] == 20)
        & (eq_all["AnalysisPopulation"] != "Complete20Sensitivity")
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    for ax, metric in zip(axes, METRIC_LABELS):
        data = trend[trend["Metric"] == metric].sort_values("TrainRatio")
        color = METRIC_COLORS[metric]
        ax.axhspan(-15, 15, color="#d8f3dc", alpha=0.35)
        ax.axhline(-10, color="#d97706", linestyle="--", linewidth=1)
        ax.axhline(10, color="#d97706", linestyle="--", linewidth=1)
        ax.axhline(-20, color="#6c757d", linestyle=":", linewidth=1)
        ax.axhline(20, color="#6c757d", linestyle=":", linewidth=1)
        ax.axhline(0, color="#343a40", linewidth=1)
        ax.plot(
            data["TrainRatio"],
            data["MeanRelativeErrorPercent"],
            color=color,
            marker="o",
            linewidth=2,
        )
        ax.fill_between(
            data["TrainRatio"],
            data["RelativeCI90LowerPercent"],
            data["RelativeCI90UpperPercent"],
            color=color,
            alpha=0.18,
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(range(10, 100, 10))
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Relative error (%)")
    axes[2].set_ylabel("Relative error (%)")
    axes[2].set_xlabel("Train ratio (%)")
    axes[3].set_xlabel("Train ratio (%)")
    fig.suptitle(
        "Reconstruction bias across train ratios\n"
        "Shaded area: +/-15%; dashed lines: +/-10%; dotted lines: +/-20%",
        y=1.02,
    )
    save(fig, "athen_relative_error_all_train_ratios.png")


def plot_trophic_validity(validity: pd.DataFrame) -> None:
    data = validity.sort_values("TrainRatio").copy()
    data["PseudoMetricInvalidPercent"] = (
        100
        * data["PseudoMetricInvalidRuns"]
        / (data["PseudoMetricValidRuns"] + data["PseudoMetricInvalidRuns"])
    )
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(data))
    width = 0.65
    ax.bar(x, data["Complete20FoodWebs"], width, label="20/20 valid", color="#2a9d8f")
    ax.bar(
        x,
        data["Partial10To19FoodWebs"],
        width,
        bottom=data["Complete20FoodWebs"],
        label="10-19 valid",
        color="#e9c46a",
    )
    ax.bar(
        x,
        data["Below10ValidFoodWebs"],
        width,
        bottom=data["Complete20FoodWebs"] + data["Partial10To19FoodWebs"],
        label="<10 valid",
        color="#e76f51",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(data["TrainRatio"].astype(int))
    ax.set_xlabel("Train ratio (%)")
    ax.set_ylabel("Number of food webs")
    ax.set_title("Availability of pseudo trophic-height estimates")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.2)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        data["PseudoMetricInvalidPercent"],
        color="#7b2cbf",
        marker="o",
        linewidth=2,
    )
    ax2.set_ylabel("Invalid pseudo runs (%)", color="#7b2cbf")
    ax2.tick_params(axis="y", colors="#7b2cbf")
    save(fig, "athen_trophic_height_validity.png")


def plot_lachlan_audit(audit: pd.DataFrame) -> None:
    data = audit[
        audit["LogEmpiricalTrophicHeight"].notna()
        & audit["LachlanTrophicHeight"].notna()
    ].copy()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(
        data["LachlanTrophicHeight"],
        data["LogEmpiricalTrophicHeight"],
        color="#0077b6",
        alpha=0.65,
        s=30,
    )
    minimum = min(
        data["LachlanTrophicHeight"].min(),
        data["LogEmpiricalTrophicHeight"].min(),
    )
    maximum = max(
        data["LachlanTrophicHeight"].max(),
        data["LogEmpiricalTrophicHeight"].max(),
    )
    ax.plot([minimum, maximum], [minimum, maximum], color="#343a40", linestyle="--")
    ax.set_xlabel("Lachlan mean trophic height")
    ax.set_ylabel("MATLAB-log mean trophic height")
    ax.set_title("Empirical trophic-height audit")
    ax.grid(alpha=0.2)
    save(fig, "athen_trophic_height_lachlan_audit.png")


def plot_minimum_margin(minimum: pd.DataFrame) -> None:
    data = minimum.copy()
    labels = data["Metric"].map(METRIC_LABELS)
    labels = labels.where(
        data["AnalysisPopulation"] != "ConditionalAtLeast10ValidRuns",
        "Mean trophic height (>=10 valid runs)",
    )
    labels = labels.where(
        data["AnalysisPopulation"] != "Complete20Sensitivity",
        "Mean trophic height (20/20 sensitivity)",
    )
    data["DisplayMetric"] = labels
    order = [
        "Connectance",
        "Mean generality",
        "Mean vulnerability",
        "Mean trophic height (>=10 valid runs)",
        "Mean trophic height (20/20 sensitivity)",
    ]
    data["PlotOrder"] = data["DisplayMetric"].map(
        {name: index for index, name in enumerate(order)}
    )
    data = data.sort_values("PlotOrder", ascending=False)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    colors = [
        METRIC_COLORS[metric]
        for metric in data["Metric"]
    ]
    bars = ax.barh(
        data["DisplayMetric"],
        data["MinimumMarginPercentExclusive"],
        color=colors,
        alpha=0.88,
    )
    ax.axvline(15, color="#343a40", linestyle="--", linewidth=1.5, label="15% proposed margin")
    for bar, value in zip(bars, data["MinimumMarginPercentExclusive"]):
        ax.text(
            value + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f">{value:.2f}%",
            va="center",
            fontsize=10,
        )
    ax.set_xlabel("Minimum relative margin required by the 90% CI (%)")
    ax.set_title("TrainRatio 60: descriptive minimum margin required")
    ax.set_xlim(0, max(18, data["MinimumMarginPercentExclusive"].max() + 2))
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False)
    save(fig, "athen_train60_minimum_margin_required.png")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    eq60 = pd.read_csv(ATHEN_DIR / "athen_equivalence_sensitivity_train60.csv")
    eq_all = pd.read_csv(ATHEN_DIR / "athen_equivalence_sensitivity_all_ratios.csv")
    validity = pd.read_csv(ATHEN_DIR / "athen_trophic_height_validity.csv")
    audit = pd.read_csv(ATHEN_DIR / "athen_trophic_height_lachlan_audit.csv")
    minimum = pd.read_csv(ATHEN_DIR / "athen_minimum_margin_required_train60.csv")
    selected = selected_train60(eq60)
    plot_train60_forest(selected)
    plot_train60_sensitivity(selected)
    plot_all_ratios(eq_all)
    plot_trophic_validity(validity)
    plot_lachlan_audit(audit)
    plot_minimum_margin(minimum)


if __name__ == "__main__":
    main()
