import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase


sns.set_theme(context="paper", style="white", font_scale=1.18)

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "src/matlab/data/result_wlnm_dir_neg_sweep_train_ratios_10-90_pseudo_properties/prediction_scores_logs"
ECOSYSTEM_CSV = PROJECT_ROOT / "src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv"
TTEST_CSV = PROJECT_ROOT / "src/matlab/data/result_wlnm_dir_neg_sweep_train_ratios_10-90_pseudo_properties/statistical_tests/wlnm_dir_neg_delta_ttests_by_ecosystem.csv"

OUTPUT_PNG = PROJECT_ROOT / "docs/plots/wlnm_dir_neg_ecosystem_metric_boxplots_ttests.png"
OUTPUT_PDF = PROJECT_ROOT / "docs/plots/wlnm_dir_neg_ecosystem_metric_boxplots_ttests.pdf"

NODES_THRESHOLD = 0
TRAIN_RATIO_TARGET = 90
K_TARGET = 10
TRAIN_RATIO_COL = "TrainRatio"
AGGREGATE_PER_FOODWEB = True

TROPHIC_LEVEL_MIN = 1.0
TROPHIC_LEVEL_MAX = 4.0

# Bigger boxes, but wider group spacing keeps pairs readable.
GROUP_GAP = 1.28
TYPE_OFFSET = 0.26
BOX_WIDTH = 0.38

base_ecosystem_order = [
    "lakes",
    "streams",
    "marine",
    "terrestrial aboveground",
    "terrestrial belowground",
]

ecosystem_labels = {
    "lakes": "Lakes",
    "streams": "Streams",
    "marine": "Marine",
    "terrestrial aboveground": "Terrestrial\naboveground",
    "terrestrial belowground": "Terrestrial\nbelowground",
}

paired_specs = [
    ("EmpiricalConnectance", "PseudoConnectance", "Connectance", "Connectance"),
    ("EmpiricalMeanTrophicLevel", "PseudoMeanTrophicLevel", "Average trophic height", "MeanTrophicLevel"),
    ("EmpiricalMeanGenerality", "PseudoMeanGenerality", "Mean generality", "MeanGenerality"),
    ("EmpiricalMeanVulnerability", "PseudoMeanVulnerability", "Mean vulnerability", "MeanVulnerability"),
]

type_palette = {
    "Empirical": "#3A7D44",
    "Pseudo": "#C6923A",
}

panel_letters = list("abcdefghijklmnopqrstuvwxyz")


# ============================================================
# HELPERS
# ============================================================
def extract_foodweb_name(filepath: str) -> str:
    return Path(filepath).name.split("_results_")[0]


def find_col(df: pd.DataFrame, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(f"Could not find any of these columns: {candidates}")


def normalize_fw_label(name: str) -> str:
    name = str(name).strip()
    name = Path(name).stem
    name = name.replace(" ", "_")
    for token in [
        "_results_",
        "_result_",
        "_results",
        "_random",
        "_tax_mass",
        "_tax",
        "_mass",
    ]:
        name = name.replace(token, "_")
    return "_".join(part for part in name.split("_") if part)


def normalize_ecosystem(value: str) -> str:
    return str(value).strip().lower()


def normalize_train_ratio(value):
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return np.nan
    return float(value * 100.0 if value <= 1.0 else value)


def format_ecosystem_label(eco: str) -> str:
    eco_clean = normalize_ecosystem(eco)
    return ecosystem_labels.get(eco_clean, str(eco).strip().capitalize())


def format_pvalue(p_value) -> str:
    if pd.isna(p_value):
        return "p = n/a"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def load_all_results(results_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(results_dir / "*_results_*.csv")))
    if not files:
        raise FileNotFoundError(f"No results CSVs found in: {results_dir}")

    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            df["Foodweb"] = extract_foodweb_name(fp)
            dfs.append(df)
        except Exception as exc:
            print(f"[WARN] Skipping {fp}: {exc}")

    if not dfs:
        raise RuntimeError("No readable result CSVs were loaded.")

    return pd.concat(dfs, ignore_index=True)


def load_ecosystem_metadata(ecosystem_csv: Path) -> pd.DataFrame:
    meta = pd.read_csv(ecosystem_csv)

    foodweb_col = find_col(meta, ["Foodweb", "foodweb"])
    eco_col = find_col(meta, ["Ecosystem", "ecosystem", "EcosystemType", "ecosystem_type"])
    nodes_col = find_col(meta, ["Nodes", "nodes"])

    meta = meta.rename(
        columns={
            foodweb_col: "Foodweb",
            eco_col: "Ecosystem",
            nodes_col: "Nodes",
        }
    )

    meta["Foodweb"] = meta["Foodweb"].astype(str)
    meta["Ecosystem"] = meta["Ecosystem"].apply(normalize_ecosystem)
    meta["Nodes"] = pd.to_numeric(meta["Nodes"], errors="coerce")
    meta["FW_KEY"] = meta["Foodweb"].apply(normalize_fw_label)

    return meta[["Foodweb", "Ecosystem", "Nodes", "FW_KEY"]].drop_duplicates()


def load_ttests(ttest_csv: Path) -> pd.DataFrame:
    if not ttest_csv.is_file():
        raise FileNotFoundError(f"Missing t-test CSV: {ttest_csv}")

    ttests = pd.read_csv(ttest_csv)
    ttests["EcosystemKey"] = ttests["EcosystemType"].apply(normalize_ecosystem)
    ttests["TrainRatioNorm"] = ttests["TrainRatio"].apply(normalize_train_ratio)
    ttests["K"] = pd.to_numeric(ttests["K"], errors="coerce")
    ttests["PValue"] = pd.to_numeric(ttests["PValue"], errors="coerce")
    return ttests


def safe_numeric_filter(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["K", TRAIN_RATIO_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[pd.notna(df[col])]
    return df


def aggregate_foodweb_runs(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    grouped = df.groupby("Foodweb", as_index=False)[numeric_cols].mean()

    eco_map = df[["Foodweb", "Ecosystem"]].drop_duplicates()
    grouped = grouped.merge(eco_map, on="Foodweb", how="left")

    cols = ["Foodweb", "Ecosystem"] + [
        col for col in grouped.columns if col not in ("Foodweb", "Ecosystem")
    ]
    return grouped[cols]


def lookup_ttest_label(ttests: pd.DataFrame, metric: str, ecosystem: str) -> str:
    eco_key = normalize_ecosystem(ecosystem)
    matches = ttests[
        (ttests["Metric"] == metric)
        & (ttests["EcosystemKey"] == eco_key)
        & np.isclose(ttests["TrainRatioNorm"], TRAIN_RATIO_TARGET)
        & np.isclose(ttests["K"], K_TARGET)
    ].copy()

    if matches.empty:
        return "p = n/a"

    row = matches.sort_values("NumFoodWebs", ascending=False).iloc[0]
    return format_pvalue(row["PValue"])


def add_ttest_bracket(ax, x1, x2, y, h, label):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=0.9, clip_on=False)
    ax.text(
        (x1 + x2) / 2,
        y + h,
        label,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="black",
        clip_on=False,
    )


class HandlerMiniBoxplot(HandlerBase):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        facecolor = orig_handle["facecolor"]
        edgecolor = orig_handle.get("edgecolor", "black")
        linewidth = orig_handle.get("linewidth", 0.8)

        cx = xdescent + 0.5 * width
        cy = ydescent + 0.5 * height
        box_w = 0.80 * width
        box_h = 0.70 * height
        box_x = cx - box_w / 2
        box_y = cy - box_h / 2

        box = Rectangle(
            (box_x, box_y),
            box_w,
            box_h,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            transform=trans,
        )
        median = Line2D(
            [box_x, box_x + box_w],
            [cy, cy],
            color=edgecolor,
            linewidth=linewidth,
            transform=trans,
        )
        whisker_top = Line2D(
            [cx, cx],
            [box_y + box_h, box_y + box_h + 0.18 * height],
            color=edgecolor,
            linewidth=linewidth,
            transform=trans,
        )
        whisker_bottom = Line2D(
            [cx, cx],
            [box_y - 0.18 * height, box_y],
            color=edgecolor,
            linewidth=linewidth,
            transform=trans,
        )
        return [box, median, whisker_top, whisker_bottom]


def draw_box(ax, values, position, facecolor):
    bp = ax.boxplot(
        values,
        positions=[position],
        widths=BOX_WIDTH,
        patch_artist=True,
        manage_ticks=False,
        showfliers=True,
        boxprops={"facecolor": facecolor, "edgecolor": "black", "linewidth": 0.9},
        medianprops={"color": "black", "linewidth": 1.1},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        flierprops={
            "marker": "o",
            "markersize": 3.7,
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markeredgewidth": 0.4,
            "linestyle": "none",
        },
    )
    return bp


# ============================================================
# LOAD + PREPARE DATA
# ============================================================
results = load_all_results(RESULTS_DIR)
meta = load_ecosystem_metadata(ECOSYSTEM_CSV)
ttests = load_ttests(TTEST_CSV)

results = safe_numeric_filter(results)

if TRAIN_RATIO_COL not in results.columns:
    raise RuntimeError(f"Column '{TRAIN_RATIO_COL}' not found in result CSVs.")

results = results[results[TRAIN_RATIO_COL].apply(normalize_train_ratio) == TRAIN_RATIO_TARGET].copy()

if results.empty:
    raise RuntimeError(
        f"No rows found for {TRAIN_RATIO_COL} = {TRAIN_RATIO_TARGET}. "
        "Check whether TrainRatio is stored as 90, 90.0, or 0.9."
    )

results["FW_KEY"] = results["Foodweb"].apply(normalize_fw_label)

allowed_fw_keys = set(meta.loc[meta["Nodes"] >= NODES_THRESHOLD, "FW_KEY"].dropna().unique())
results = results[results["FW_KEY"].isin(allowed_fw_keys)].copy()

if results.empty:
    raise ValueError(f"No foodwebs left after filtering Nodes >= {NODES_THRESHOLD}.")

results = results.merge(
    meta[["FW_KEY", "Ecosystem"]].drop_duplicates(),
    on="FW_KEY",
    how="left",
)

required_plot_cols = sorted(
    {"Ecosystem"} | {col for spec in paired_specs for col in spec[:2]}
)
results = results.dropna(subset=required_plot_cols)

plot_df = aggregate_foodweb_runs(results) if AGGREGATE_PER_FOODWEB else results.copy()

print(f"[INFO] Plot rows after aggregation: {len(plot_df)}")
print(f"[INFO] Reading t-tests from: {TTEST_CSV}")


# ============================================================
# PLOT
# ============================================================
n_panels = len(paired_specs)
n_cols = 2
n_rows = math.ceil(n_panels / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(17.5, 10.8), sharey=False)
axes = np.array(axes).reshape(-1)

legend_handles = [
    {"facecolor": type_palette["Empirical"], "edgecolor": "black", "linewidth": 0.9},
    {"facecolor": type_palette["Pseudo"], "edgecolor": "black", "linewidth": 0.9},
]
legend_labels = ["Empirical", "Pseudo"]

for idx, (ax, (emp_col, pseudo_col, ylabel, ttest_metric)) in enumerate(
    zip(axes, paired_specs)
):
    panel_df = plot_df.dropna(subset=[emp_col, pseudo_col, "Ecosystem"]).copy()

    if ylabel == "Average trophic height":
        panel_df = panel_df[
            np.isfinite(panel_df[emp_col])
            & np.isfinite(panel_df[pseudo_col])
            & panel_df[emp_col].between(TROPHIC_LEVEL_MIN, TROPHIC_LEVEL_MAX)
            & panel_df[pseudo_col].between(TROPHIC_LEVEL_MIN, TROPHIC_LEVEL_MAX)
        ].copy()

    if panel_df.empty:
        ax.set_visible(False)
        continue

    present_ecosystems = [
        eco for eco in base_ecosystem_order if eco in panel_df["Ecosystem"].unique()
    ]
    extra_ecosystems = [
        eco for eco in panel_df["Ecosystem"].unique() if eco not in base_ecosystem_order
    ]
    eco_order = present_ecosystems + extra_ecosystems
    base_positions = np.arange(len(eco_order)) * GROUP_GAP

    all_values = []
    bracket_tops = []

    for eco, base_x in zip(eco_order, base_positions):
        eco_df = panel_df[panel_df["Ecosystem"] == eco]
        emp_values = eco_df[emp_col].dropna().to_numpy()
        pseudo_values = eco_df[pseudo_col].dropna().to_numpy()

        if emp_values.size:
            draw_box(ax, emp_values, base_x - TYPE_OFFSET, type_palette["Empirical"])
            all_values.extend(emp_values)

        if pseudo_values.size:
            draw_box(ax, pseudo_values, base_x + TYPE_OFFSET, type_palette["Pseudo"])
            all_values.extend(pseudo_values)

        pair_values = np.concatenate([emp_values, pseudo_values])
        if pair_values.size:
            bracket_tops.append((eco, base_x, np.nanmax(pair_values)))

    all_values = np.asarray(all_values, dtype=float)
    local_min = np.nanmin(all_values)
    local_max = np.nanmax(all_values)
    data_range = local_max - local_min if local_max > local_min else max(local_max, 0.05)

    y_min = max(0.0, local_min - 0.06 * data_range)
    y_max = local_max + 0.28 * data_range
    bracket_h = 0.025 * data_range
    bracket_pad = 0.055 * data_range

    for eco, base_x, pair_max in bracket_tops:
        label = lookup_ttest_label(ttests, ttest_metric, eco)
        add_ttest_bracket(
            ax,
            base_x - TYPE_OFFSET,
            base_x + TYPE_OFFSET,
            pair_max + bracket_pad,
            bracket_h,
            label,
        )

    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    ax.set_title("")
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=14)

    ax.set_xticks(base_positions)
    ax.set_xticklabels([format_ecosystem_label(eco) for eco in eco_order], fontsize=14)
    ax.set_xlim(base_positions[0] - 0.78, base_positions[-1] + 0.78)

    ax.grid(False)
    sns.despine(ax=ax)

    ax.annotate(
        panel_letters[idx],
        xy=(0, 1.035),
        xycoords="axes fraction",
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=19,
        fontweight="bold",
        annotation_clip=False,
    )

    if idx == 0:
        ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            handler_map={dict: HandlerMiniBoxplot()},
            loc="upper right",
            frameon=False,
            fontsize=14,
            handlelength=1.5,
            handleheight=0.9,
            handletextpad=0.6,
            borderpad=0.2,
            labelspacing=0.45,
        )

for j in range(len(paired_specs), len(axes)):
    axes[j].set_visible(False)

fig.tight_layout(pad=1.4, w_pad=2.0, h_pad=2.6)

OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, bbox_inches="tight")

print(f"[INFO] Saved PNG: {OUTPUT_PNG}")
print(f"[INFO] Saved PDF: {OUTPUT_PDF}")

plt.show()
