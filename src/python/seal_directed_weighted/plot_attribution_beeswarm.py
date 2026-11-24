#!/usr/bin/env python3
"""Create the publication-ready SHAP-like SEAL attribution v3 figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


REQUIRED = {
    "SUMMARY_METADATA.json",
    "node_feature_summary.csv",
    "edge_feature_summary.csv",
    "drnl_summary.csv",
    "node_beeswarm_sample.csv",
    "edge_beeswarm_sample.csv",
    "drnl_beeswarm_sample.csv",
}
VARIANTS = ("directed_node", "directed_ecc")


def cohort_top(summary, variant, count):
    rows = summary.loc[
        summary["ModelVariant"].eq(variant)
        & summary["GroupType"].eq("cohort")
        & summary["GroupValue"].eq("all")
    ].copy()
    if rows.empty:
        raise ValueError("no cohort summary rows for {}".format(variant))
    return rows.sort_values(
        ["MeanAbsoluteNormalized", "Feature"], ascending=[False, True]
    ).head(int(count))["Feature"].tolist()


def stable_sample(rows, limit, seed):
    return rows if len(rows) <= int(limit) else rows.sample(
        n=int(limit), random_state=int(seed), replace=False
    )


def feature_colour(values):
    values = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = np.isfinite(values)
    scaled = np.full(len(values), 0.5, dtype=float)
    if finite.any():
        low, high = np.nanpercentile(values[finite], [5, 95])
        if high > low:
            scaled[finite] = np.clip((values[finite] - low) / (high - low), 0, 1)
    return scaled


def plot_expected_gradients(ax, points, variant, features, title, seed):
    rng = np.random.default_rng(int(seed))
    for y, feature in enumerate(features):
        rows = points.loc[
            points["ModelVariant"].eq(variant) & points["Feature"].eq(feature)
        ]
        rows = stable_sample(rows, 900, seed + y)
        jitter = rng.normal(0, 0.105, len(rows))
        ax.scatter(
            rows["Attribution"], y + jitter, c=feature_colour(rows["Value"]),
            cmap="coolwarm", vmin=0, vmax=1, s=8, alpha=0.50,
            linewidths=0, rasterized=True,
        )
    ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(features)), features, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10, fontweight="semibold")
    ax.set_xlabel("Expected Gradient contribution to link log-odds", fontsize=8.5)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6, alpha=0.7)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    summary_dir = args.summary_dir.resolve()
    observed = {path.name for path in summary_dir.iterdir()} if summary_dir.is_dir() else set()
    missing = sorted(REQUIRED.difference(observed))
    if missing:
        raise FileNotFoundError("attribution summary is missing {}".format(", ".join(missing)))
    with open(summary_dir / "SUMMARY_METADATA.json", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != "seal-directed-node-ecc-attribution-summary-v3":
        raise ValueError("figure requires attribution summary v3")
    for field in ("tukey_filter", "outlier_removal", "channels_combined"):
        if metadata.get(field) is not False:
            raise ValueError("expected {}=false".format(field))

    node_summary = pd.read_csv(summary_dir / "node_feature_summary.csv")
    edge_summary = pd.read_csv(summary_dir / "edge_feature_summary.csv")
    drnl_summary = pd.read_csv(summary_dir / "drnl_summary.csv")
    node_points = pd.read_csv(summary_dir / "node_beeswarm_sample.csv")
    edge_points = pd.read_csv(summary_dir / "edge_beeswarm_sample.csv")
    drnl_points = pd.read_csv(summary_dir / "drnl_beeswarm_sample.csv")
    if set(node_summary["ModelVariant"].unique()) != set(VARIANTS):
        raise ValueError("node summaries do not contain both attribution variants")

    node_features = {
        variant: cohort_top(node_summary, variant, args.top) for variant in VARIANTS
    }
    edge_features = cohort_top(edge_summary, "directed_ecc", args.top)
    drnl_features = []
    for variant in VARIANTS:
        for feature in cohort_top(drnl_summary, variant, min(7, args.top)):
            if feature not in drnl_features:
                drnl_features.append(feature)
    drnl_features = drnl_features[:args.top]

    sns.set_theme(context="paper", style="white", font_scale=1.0)
    figure, axes = plt.subplots(2, 2, figsize=(14.2, 11.2))
    plot_expected_gradients(
        axes[0, 0], node_points, "directed_node", node_features["directed_node"],
        "A  SEAL-D-NodeAttr-DirectedMP — node attributes", 46101,
    )
    plot_expected_gradients(
        axes[0, 1], node_points, "directed_ecc", node_features["directed_ecc"],
        "B  SEAL-D-NodeEdgeAttr-ECC — node attributes", 46102,
    )
    plot_expected_gradients(
        axes[1, 0], edge_points, "directed_ecc", edge_features,
        "C  SEAL-D-NodeEdgeAttr-ECC — edge covariates", 46103,
    )

    drnl_ax = axes[1, 1]
    styles = {
        "directed_node": ("SEAL-D-NodeAttr-DirectedMP", "#2C7FB8", "o", -0.11),
        "directed_ecc": ("SEAL-D-NodeEdgeAttr-ECC", "#8C2D04", "^", 0.11),
    }
    for variant, (label, colour, marker, offset) in styles.items():
        rng = np.random.default_rng(46200 + int(variant == "directed_ecc"))
        for y, feature in enumerate(drnl_features):
            rows = drnl_points.loc[
                drnl_points["ModelVariant"].eq(variant)
                & drnl_points["Feature"].eq(feature)
                & pd.to_numeric(drnl_points["Value"], errors="coerce").eq(1)
            ]
            rows = stable_sample(rows, 700, 46300 + y + int(variant == "directed_ecc") * 100)
            drnl_ax.scatter(
                rows["Attribution"], y + offset + rng.normal(0, 0.050, len(rows)),
                color=colour, marker=marker, s=8, alpha=0.42,
                linewidths=0, rasterized=True,
            )
    drnl_ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    drnl_ax.set_yticks(range(len(drnl_features)), drnl_features, fontsize=7.5)
    drnl_ax.invert_yaxis()
    drnl_ax.set_title("D  Structural DRNL channel", fontsize=10, fontweight="semibold")
    drnl_ax.set_xlabel("Integrated Gradient contribution to link log-odds", fontsize=8.5)
    drnl_ax.grid(axis="x", color="#DDDDDD", linewidth=0.6, alpha=0.7)
    drnl_ax.legend(
        handles=[Line2D([0], [0], marker=value[2], color="none",
                        markerfacecolor=value[1], markersize=6, label=value[0])
                 for value in styles.values()],
        loc="lower right", frameon=False, fontsize=8,
    )
    for axis in axes.flat:
        sns.despine(ax=axis)
    figure.suptitle(
        "SEAL attribution v3: learned feature dependence",
        fontsize=14, fontweight="semibold", y=0.985,
    )
    figure.text(
        0.5, 0.015,
        "Expected/Integrated Gradients (not exact SHAP values). Signed contributions target link log-odds; "
        "positive values support link presence. Node, edge and DRNL channels are not summed. "
        "Attributions describe model reliance, not causal ecological importance.",
        ha="center", va="bottom", fontsize=8.2, color="#4D4D4D", wrap=True,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.965), h_pad=2.5, w_pad=2.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "seal_attribution_v3_shap_like"
    figure.savefig(stem.with_suffix(".png"), dpi=int(args.dpi), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({
        "status": "FIGURE_OK",
        "summary_dir": str(summary_dir),
        "outputs": [str(stem.with_suffix(ext)) for ext in (".png", ".pdf", ".svg")],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
