#!/usr/bin/env python3
"""Generate Supplementary Figure 3 from raw node-ecology repeated CV."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


VERSION = "WLNM_dir_neg_node_ecology_kfold"
EXPECTED_EXPERIMENTS = 100
EXPECTED_FOODWEBS = 290
CVK_ORDER = [3, 5, 10]
METRIC_SPECS = [
    ("ROC_AUC", "ROC-AUC"),
    ("PR_AUC", "PR-AUC"),
    ("F1Score", "F1"),
    ("TestMCC", "MCC"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
    ("TestTSS", "TSS"),
]
METRICS = [metric for metric, _ in METRIC_SPECS]
RESULT_RE = re.compile(
    r"^(?P<foodweb>.+)_results_random_"
    r"wlnm_dir_neg_node_ecology_kfold_cvK(?P<cvk>3|5|10)\.csv$"
)
SHOW_WEB_LEVEL_FLIERS = True


def find_repo_root(start: Optional[Path] = None) -> Path:
    start = (start or Path.cwd()).resolve()
    candidates = [start, *start.parents]
    fallback = Path("/Users/acw792/Developer/qmul-phd-framework")
    if fallback not in candidates:
        candidates.append(fallback)
    for candidate in candidates:
        if (candidate / "src/matlab").is_dir() and (candidate / "docs").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {start}")


def parse_manifest(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def resolve_result_root(repo_root: Path) -> Path:
    configured = os.environ.get("WLNM_NODE_ECOLOGY_KFOLD_RESULT_ROOT", "").strip()
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            root = repo_root / root
        return root.resolve()

    candidates = []
    data_root = repo_root / "src/matlab/data"
    for path in sorted(data_root.glob("result_wlnm_dir_neg_node_ecology_kfold_100x290_*")):
        try:
            manifest = parse_manifest(path / "RUN_MANIFEST.txt")
        except FileNotFoundError:
            continue
        if manifest.get("RunMode") == "full" and manifest.get("Version") == VERSION:
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one full node-ecology k-fold result root; "
            f"found {len(candidates)}. Set WLNM_NODE_ECOLOGY_KFOLD_RESULT_ROOT explicitly."
        )
    return candidates[0].resolve()


def validate_full_manifest(root: Path) -> dict[str, str]:
    manifest = parse_manifest(root / "RUN_MANIFEST.txt")
    expected = {
        "RunMode": "full",
        "Version": VERSION,
        "CvKList": "3,5,10",
        "RepeatedFoldPartitions": "true",
        "NumExperimentsPerCvK": str(EXPECTED_EXPERIMENTS),
        "FoodWebs": str(EXPECTED_FOODWEBS),
        "ExpectedModels": "522000",
        "ExpectedDataRowsCvK3": "300",
        "ExpectedDataRowsCvK5": "500",
        "ExpectedDataRowsCvK10": "1000",
        "PRAUCProtocol": "average_precision_noninterpolated_v1",
        "ClassificationThreshold": "0.50",
        "ComputeEcologicalMetrics": "false",
    }
    mismatches = {
        key: (expected_value, manifest.get(key))
        for key, expected_value in expected.items()
        if manifest.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Unexpected full-run manifest values: {mismatches}")
    return manifest


def load_raw_results(root: Path) -> pd.DataFrame:
    files = sorted((root / "prediction_scores_logs").glob("*.csv"))
    if len(files) != 870:
        raise ValueError(f"Expected 870 prediction CSVs, found {len(files)}")
    required = {
        "Version",
        "K",
        "Threshold",
        "CvK",
        "FoldID",
        "ExperimentID",
        "CvSplitSeed",
        "CvPartitionHash",
        *METRICS,
    }
    frames = []
    for path in files:
        match = RESULT_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected result filename: {path.name}")
        frame = pd.read_csv(path, usecols=lambda column: column in required)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        frame["Foodweb"] = match.group("foodweb")
        filename_cvk = int(match.group("cvk"))
        if not frame["CvK"].eq(filename_cvk).all():
            raise ValueError(f"{path.name}: filename/data CvK mismatch")
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    numeric = [
        "K",
        "Threshold",
        "CvK",
        "FoldID",
        "ExperimentID",
        "CvSplitSeed",
        *METRICS,
    ]
    raw[numeric] = raw[numeric].apply(pd.to_numeric, errors="coerce")
    raw = raw.loc[
        raw["Version"].eq(VERSION)
        & raw["K"].eq(10)
        & np.isclose(raw["Threshold"], 0.50, atol=1e-12)
        & raw["CvK"].isin(CVK_ORDER)
    ].copy()
    if raw.empty or not np.isfinite(raw[numeric].to_numpy(float)).all():
        raise ValueError("Selected raw k-fold values are empty or non-finite")
    probability = [metric for metric in METRICS if metric not in {"TestMCC", "TestTSS"}]
    if ((raw[probability] < 0) | (raw[probability] > 1)).any().any():
        raise ValueError("Probability metrics must be within [0, 1]")
    if ((raw[["TestMCC", "TestTSS"]] < -1) | (raw[["TestMCC", "TestTSS"]] > 1)).any().any():
        raise ValueError("MCC and TSS must be within [-1, 1]")
    return raw


def aggregate_complete_raw_runs(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit_keys = ["Foodweb", "CvK", "ExperimentID"]
    duplicates = raw.duplicated([*unit_keys, "FoldID"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate food-web/CvK/repeat/fold rows were found")

    fold_checks = (
        raw.groupby(unit_keys, as_index=False)
        .agg(
            FoldRows=("FoldID", "size"),
            UniqueFolds=("FoldID", "nunique"),
            PartitionHashes=("CvPartitionHash", "nunique"),
            SplitSeeds=("CvSplitSeed", "nunique"),
        )
    )
    invalid = fold_checks.loc[
        ~fold_checks["FoldRows"].eq(fold_checks["CvK"])
        | ~fold_checks["UniqueFolds"].eq(fold_checks["CvK"])
        | ~fold_checks["PartitionHashes"].eq(1)
        | ~fold_checks["SplitSeeds"].eq(1)
    ]
    if not invalid.empty:
        raise ValueError("Incomplete repeated-CV units:\n" + invalid.head(20).to_string(index=False))

    repeat_means = raw.groupby(unit_keys, as_index=False).agg(
        **{metric: (metric, "mean") for metric in METRICS},
        CvPartitionHash=("CvPartitionHash", "first"),
        CvSplitSeed=("CvSplitSeed", "first"),
    )
    counts = repeat_means.groupby(["Foodweb", "CvK"]).size()
    if not counts.eq(EXPECTED_EXPERIMENTS).all():
        raise ValueError(
            f"Expected {EXPECTED_EXPERIMENTS} complete repeats per web/CvK; "
            f"observed {counts.value_counts().to_dict()}"
        )
    unique_partitions = repeat_means.groupby(["Foodweb", "CvK"])["CvPartitionHash"].nunique()
    if not unique_partitions.eq(EXPECTED_EXPERIMENTS).all():
        raise ValueError("Every repeat must use a distinct fold partition")

    web_means = repeat_means.groupby(["Foodweb", "CvK"], as_index=False).agg(
        **{metric: (metric, "mean") for metric in METRICS},
        RawRepeatedCvUnits=("ExperimentID", "size"),
    )
    coverage = web_means.groupby("CvK")["Foodweb"].nunique()
    if coverage.to_dict() != {3: 290, 5: 290, 10: 290}:
        raise ValueError(f"Expected 290 food webs per CvK; observed {coverage.to_dict()}")
    return repeat_means, web_means


def cvk_label(k: int) -> str:
    return {3: "3-fold (66.7% train)", 5: "5-fold (80% train)", 10: "10-fold (90% train)"}[k]


class HandlerMiniBoxplot(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        facecolor = orig_handle["facecolor"]
        edgecolor = orig_handle.get("edgecolor", "black")
        linewidth = orig_handle.get("linewidth", 0.6)
        cy = ydescent + 0.5 * height - 0.06 * height
        cx = xdescent + 0.5 * width
        box_w = 0.8 * width
        box_h = 0.8 * height
        box_x = cx - box_w / 2
        box_y = cy - box_h / 2
        box = Rectangle(
            (box_x, box_y), box_w, box_h,
            facecolor=facecolor, edgecolor=edgecolor,
            linewidth=linewidth, transform=trans,
        )
        median = Line2D(
            [box_x, box_x + box_w], [cy, cy],
            color=edgecolor, linewidth=linewidth, transform=trans,
        )
        whisker_top = Line2D(
            [cx, cx], [box_y + box_h, box_y + box_h + 0.18 * height],
            color=edgecolor, linewidth=linewidth, transform=trans,
        )
        whisker_bottom = Line2D(
            [cx, cx], [box_y - 0.18 * height, box_y],
            color=edgecolor, linewidth=linewidth, transform=trans,
        )
        return [box, median, whisker_top, whisker_bottom]


def plot_figure(web_means: pd.DataFrame, output_stem: Path) -> None:
    sns.set_context("paper", font_scale=1.15)
    sns.set_style("white")
    palette = {3: "#D8B7E3", 5: "#B07CC6", 10: "#7B3294"}
    fig, ax = plt.subplots(figsize=(12, 5))
    group_centers = 1.0 + np.arange(len(METRIC_SPECS)) * 1.45
    offsets = np.linspace(-0.34, 0.34, len(CVK_ORDER))
    for cv_index, cvk in enumerate(CVK_ORDER):
        data = [
            web_means.loc[web_means["CvK"].eq(cvk), metric].to_numpy(float)
            for metric, _ in METRIC_SPECS
        ]
        positions = [group_centers[index] + offsets[cv_index] for index in range(len(METRIC_SPECS))]
        boxplot = ax.boxplot(
            data,
            positions=positions,
            widths=0.32,
            patch_artist=True,
            manage_ticks=False,
            showfliers=SHOW_WEB_LEVEL_FLIERS,
            boxprops={"edgecolor": "black", "linewidth": 0.8},
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
            flierprops={
                "marker": "o",
                "markersize": 4,
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markeredgewidth": 0.2,
                "alpha": 0.7,
            },
        )
        for box in boxplot["boxes"]:
            box.set_facecolor(palette[cvk])

    handles = [
        {"facecolor": palette[k], "edgecolor": "black", "linewidth": 0.8}
        for k in CVK_ORDER
    ]
    ax.set_xlim(group_centers[0] - 0.8, group_centers[-1] + 0.8)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([label for _, label in METRIC_SPECS], ha="center", fontsize=13)
    ax.set_ylabel("Metric value", fontsize=14)
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=13)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    sns.despine(ax=ax)
    ax.legend(
        handles=handles,
        labels=[cvk_label(k) for k in CVK_ORDER],
        handler_map={dict: HandlerMiniBoxplot()},
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=3,
        frameon=False,
        fontsize=12,
        handlelength=1.5,
        handleheight=0.9,
        handletextpad=0.6,
        borderpad=0.2,
        labelspacing=0.45,
        columnspacing=1.2,
    )
    plt.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        print(f"[SAVED] {path}")
    plt.show()


def write_summaries(repeat_means: pd.DataFrame, web_means: pd.DataFrame, output_stem: Path) -> None:
    long = web_means.melt(
        id_vars=["Foodweb", "CvK", "RawRepeatedCvUnits"],
        value_vars=METRICS,
        var_name="Metric",
        value_name="Value",
    )
    summary = long.groupby(["CvK", "Metric"], as_index=False).agg(
        FoodWebs=("Foodweb", "nunique"),
        Mean=("Value", "mean"),
        Median=("Value", "median"),
        Q1=("Value", lambda values: values.quantile(0.25)),
        Q3=("Value", lambda values: values.quantile(0.75)),
        Minimum=("Value", "min"),
        Maximum=("Value", "max"),
    )
    counts = repeat_means.groupby(["Foodweb", "CvK"], as_index=False).agg(
        CompleteRepeatedCvUnits=("ExperimentID", "size"),
        UniquePartitionHashes=("CvPartitionHash", "nunique"),
        UniqueSplitSeeds=("CvSplitSeed", "nunique"),
    )
    summary_path = output_stem.parent / f"{output_stem.name}_summary.csv"
    counts_path = output_stem.parent / f"{output_stem.name}_raw_run_counts.csv"
    summary.to_csv(summary_path, index=False)
    counts.to_csv(counts_path, index=False)
    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {counts_path}")


def main() -> None:
    repo_root = find_repo_root()
    result_root = resolve_result_root(repo_root)
    validate_full_manifest(result_root)
    raw = load_raw_results(result_root)
    repeat_means, web_means = aggregate_complete_raw_runs(raw)
    output_dir = result_root / "figures_raw_unfiltered"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / (
        "supplementary_figure_3_node_ecology_kfold_100repeats_"
        "cvK3-5-10_raw_unfiltered"
    )
    print(f"[INFO] Result root: {result_root}")
    print("[INFO] No Tukey filtering or outlier removal is applied.")
    print(
        f"[INFO] Fold means -> {EXPECTED_EXPERIMENTS}-repeat means "
        "-> 290 food-web boxplots."
    )
    write_summaries(repeat_means, web_means, output_stem)
    plot_figure(web_means, output_stem)


if __name__ == "__main__":
    main()
