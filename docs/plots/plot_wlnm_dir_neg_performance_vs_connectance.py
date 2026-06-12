import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "qmul_phd_framework_mplconfig"),
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "src/matlab/data/result_wlnm_dir_neg_sweep_train_ratios_10-90_pseudo_properties/prediction_scores_logs"
)
FOODWEB_METADATA_CSV = (
    PROJECT_ROOT / "src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv"
)

OUTPUT_PNG = PROJECT_ROOT / "docs/plots/wlnm_dir_neg_performance_vs_empirical_connectance.png"
OUTPUT_PDF = PROJECT_ROOT / "docs/plots/wlnm_dir_neg_performance_vs_empirical_connectance.pdf"

K_TARGET = 10
AGGREGATE_REPEATS = True

METRIC_SPECS = [
    ("ROC_AUC", "ROC-AUC"),
    ("PR_AUC", "PR-AUC"),
    ("F1Score", "F1 score"),
    ("Accuracy", "Accuracy"),
]


def extract_foodweb_name(path: Path) -> str:
    return path.name.split("_results_")[0]


def normalize_train_ratio(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.where(values <= 1.0, values * 100.0, values)


def load_results(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.glob("*_results_*.csv"))
    if not files:
        raise FileNotFoundError(f"No result CSV files found in {results_dir}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["Foodweb"] = extract_foodweb_name(path)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_foodweb_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    required = {"Foodweb", "Connectance", "Nodes", "Edges", "EcosystemType"}
    missing = required.difference(meta.columns)
    if missing:
        raise KeyError(f"Missing required metadata columns: {sorted(missing)}")

    meta = meta[["Foodweb", "Connectance", "Nodes", "Edges", "EcosystemType"]].copy()
    meta["MetadataConnectance"] = pd.to_numeric(meta["Connectance"], errors="coerce")
    meta["Nodes"] = pd.to_numeric(meta["Nodes"], errors="coerce")
    meta["Edges"] = pd.to_numeric(meta["Edges"], errors="coerce")
    return meta.drop(columns=["Connectance"]).drop_duplicates(subset=["Foodweb"])


def prepare_plot_data(results: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    df = results.merge(metadata, on="Foodweb", how="left")

    numeric_cols = [
        "K",
        "TrainRatio",
        "ROC_AUC",
        "PR_AUC",
        "Precision",
        "Recall",
        "F1Score",
        "EmpiricalConnectance",
        "PseudoTP",
        "PseudoFP",
        "PseudoFN",
        "PseudoTN",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["TrainRatio"] = normalize_train_ratio(df["TrainRatio"])
    df["EmpiricalConnectanceForPlot"] = df["MetadataConnectance"].fillna(
        df["EmpiricalConnectance"]
    )

    confusion_total = df[["PseudoTP", "PseudoFP", "PseudoFN", "PseudoTN"]].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Accuracy"] = (df["PseudoTP"] + df["PseudoTN"]) / confusion_total
    df.loc[confusion_total <= 0, "Accuracy"] = np.nan

    if K_TARGET is not None and "K" in df.columns:
        df = df[np.isclose(df["K"], K_TARGET, equal_nan=False)].copy()

    if AGGREGATE_REPEATS:
        group_cols = [
            "Foodweb",
            "K",
            "TrainRatio",
            "EmpiricalConnectanceForPlot",
            "Nodes",
            "Edges",
            "EcosystemType",
        ]
        mean_cols = [metric for metric, _ in METRIC_SPECS]
        df = (
            df.groupby(group_cols, dropna=False, as_index=False)[mean_cols]
            .mean()
            .copy()
        )

    metric_cols = [metric for metric, _ in METRIC_SPECS]
    df = df.dropna(subset=["EmpiricalConnectanceForPlot", "TrainRatio"])
    return df.dropna(subset=metric_cols, how="all")


def binned_median_line(data: pd.DataFrame, metric: str, bins: int = 10) -> pd.DataFrame:
    x = data["EmpiricalConnectanceForPlot"].to_numpy(dtype=float)
    if len(np.unique(x[~np.isnan(x)])) < 3:
        return pd.DataFrame(columns=["x", "y"])

    edges = np.unique(np.quantile(x[~np.isnan(x)], np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return pd.DataFrame(columns=["x", "y"])

    work = data.copy()
    work["ConnectanceBin"] = pd.cut(
        work["EmpiricalConnectanceForPlot"], bins=edges, include_lowest=True
    )
    summary = (
        work.groupby("ConnectanceBin", observed=True)
        .agg(
            x=("EmpiricalConnectanceForPlot", "median"),
            y=(metric, "median"),
            n=(metric, "count"),
        )
        .reset_index(drop=True)
    )
    return summary[summary["n"] >= 5]


def style_axes(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6e2dc", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(colors="#333333", labelsize=9)


def plot_performance_vs_connectance(plot_df: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "figure.titlesize": 13,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), sharex=True)
    axes = axes.ravel()

    train_min = float(plot_df["TrainRatio"].min())
    train_max = float(plot_df["TrainRatio"].max())
    norm = mpl.colors.Normalize(vmin=train_min, vmax=train_max)
    cmap = mpl.colormaps["viridis"]

    for ax, (metric, label) in zip(axes, METRIC_SPECS):
        data = plot_df.dropna(subset=["EmpiricalConnectanceForPlot", metric])

        ax.scatter(
            data["EmpiricalConnectanceForPlot"],
            data[metric],
            c=data["TrainRatio"],
            cmap=cmap,
            norm=norm,
            s=20,
            alpha=0.48,
            linewidth=0.25,
            edgecolor="white",
        )

        med = binned_median_line(data, metric)
        if not med.empty:
            ax.plot(
                med["x"],
                med["y"],
                color="#232323",
                linewidth=2.0,
                marker="o",
                markersize=3.2,
                label="Binned median",
            )

        style_axes(ax, label)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_ylim(0.0, 1.03)

        n_foodwebs = data["Foodweb"].nunique()
        n_points = len(data)
        ax.text(
            0.03,
            0.05,
            f"{n_foodwebs} food webs, {n_points} means",
            transform=ax.transAxes,
            fontsize=8.4,
            color="#555555",
        )

    for ax in axes[2:]:
        ax.set_xlabel("Empirical connectance")

    fig.suptitle("WLNM_dir_neg performance vs empirical connectance", x=0.47, y=0.99)
    fig.subplots_adjust(left=0.085, right=0.90, bottom=0.08, top=0.92, wspace=0.22, hspace=0.30)

    cbar_ax = fig.add_axes([0.925, 0.17, 0.018, 0.67])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Train ratio (%)")
    cbar.outline.set_edgecolor("#777777")
    cbar.ax.tick_params(labelsize=9)

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = load_results(RESULTS_DIR)
    metadata = load_foodweb_metadata(FOODWEB_METADATA_CSV)
    plot_df = prepare_plot_data(results, metadata)

    plot_performance_vs_connectance(plot_df)

    included_foodwebs = plot_df.dropna(subset=["ROC_AUC"])["Foodweb"].nunique()
    included_points = plot_df.dropna(subset=["ROC_AUC"]).shape[0]
    skipped_foodwebs = sorted(set(results["Foodweb"]) - set(plot_df["Foodweb"]))

    print(f"Wrote {OUTPUT_PNG}")
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Included {included_foodwebs} food webs and {included_points} foodweb/train-ratio means.")
    if skipped_foodwebs:
        print("Skipped food webs with no plottable metrics: " + ", ".join(skipped_foodwebs))


if __name__ == "__main__":
    main()
