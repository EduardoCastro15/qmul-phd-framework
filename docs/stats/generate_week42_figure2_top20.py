"""Generate the 20 ranked Figure 2 options used by week-42.ipynb cell 4.

This file is executed with ``runpy.run_path(..., init_globals=globals())``
from the notebook.  It deliberately reuses the notebook's validated data
loaders and its existing plotting constants so the exported alternatives use
the same data semantics and visual style as the single-food-web panels.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REQUIRED_NOTEBOOK_GLOBALS = {
    "REPO_ROOT",
    "PREDICTION_LOG_DIR",
    "ECOLOGICAL_SNAPSHOTS_DIR",
    "BASE_MAT_PATH",
    "K_TARGET",
    "CLASSIFICATION_THRESHOLD",
    "MIN_NODES",
    "SHOW_TITLES",
    "SHOW_TICKS",
    "USE_NUMERIC_AXIS_LABELS",
    "PANEL_XTICK_FONTSIZES",
    "PANEL_YTICK_FONTSIZES",
    "XLAB",
    "YLAB",
    "legend_handles",
    "canonical_foodweb",
    "tax_mass_stem",
    "select_run",
    "local_snapshot_from_row",
    "load_tax_mass",
    "load_snapshot",
    "validate_run",
    "mask_to_df",
}

missing_globals = sorted(REQUIRED_NOTEBOOK_GLOBALS.difference(globals()))
if missing_globals:
    raise RuntimeError(
        "Run this generator from week-42.ipynb cell 4. Missing notebook "
        f"globals: {missing_globals}"
    )


NUM_OPTIONS = 20
MIN_TEST_LINKS_90 = 30
FIGURE_DPI = 300
OUTPUT_DIR = (
    REPO_ROOT
    / "docs/progress_notebooks_logs/2026/figures"
    / "week-42/figure-2-top20-options"
)
SPECIES_KEY_DIR = OUTPUT_DIR / "species_keys"
INDEX_PATH = OUTPUT_DIR / "top20_options_index.csv"


def build_ranking():
    """Rank food webs from typical performance over all 100 runs."""
    suffix = "_tax_mass_results_random_wlnm_dir_neg.csv"
    required_columns = [
        "Version",
        "K",
        "TrainRatio",
        "ExperimentID",
        "Threshold",
        "TestLinks",
        "TestTP",
        "TestFP",
        "TestFN",
        "TestF1Score",
        "TestMCC",
        "EmpiricalNumSpecies",
        "EmpiricalLinks",
    ]
    csv_paths = sorted(PREDICTION_LOG_DIR.glob(f"*{suffix}"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No WLNM_dir_neg result CSVs found in {PREDICTION_LOG_DIR}"
        )

    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(
            csv_path,
            usecols=required_columns,
            low_memory=False,
        )
        mask = (
            frame["Version"].eq("WLNM_dir_neg")
            & frame["K"].eq(K_TARGET)
            & frame["TrainRatio"].isin([60, 90])
            & np.isclose(
                frame["Threshold"],
                CLASSIFICATION_THRESHOLD,
                atol=1e-12,
                rtol=0,
            )
        )
        frame = frame.loc[mask, required_columns].copy()
        frame["Foodweb"] = csv_path.name[: -len(suffix)]
        frames.append(frame)

    scores = pd.concat(frames, ignore_index=True)
    coverage = scores.groupby(["Foodweb", "TrainRatio"]).size()
    if len(coverage) != 2 * len(csv_paths) or not coverage.eq(100).all():
        raise ValueError(
            "Expected exactly 100 runs for every food web at both 60% "
            "and 90%."
        )

    summary = (
        scores.groupby(["Foodweb", "TrainRatio"], as_index=False)
        .agg(
            NumSpecies=("EmpiricalNumSpecies", "first"),
            EmpiricalLinks=("EmpiricalLinks", "first"),
            TestLinks=("TestLinks", "mean"),
            MeanTP=("TestTP", "mean"),
            MeanFP=("TestFP", "mean"),
            MeanFN=("TestFN", "mean"),
            MeanF1=("TestF1Score", "mean"),
            MeanMCC=("TestMCC", "mean"),
        )
    )

    ratio_tables = {}
    for ratio in (60, 90):
        table = summary.loc[
            summary["TrainRatio"].eq(ratio)
        ].drop(columns="TrainRatio")
        ratio_tables[ratio] = table.rename(
            columns={
                column: f"{column}_{ratio}"
                for column in table.columns
                if column != "Foodweb"
            }
        )

    ranking = ratio_tables[60].merge(
        ratio_tables[90],
        on="Foodweb",
        validate="one_to_one",
    )
    if not ranking["NumSpecies_60"].eq(ranking["NumSpecies_90"]).all():
        raise ValueError("Species counts differ between ratios.")

    ranking["CombinedMeanF1"] = (
        ranking["MeanF1_60"] + ranking["MeanF1_90"]
    ) / 2.0
    ranking["WorstRatioMeanF1"] = ranking[
        ["MeanF1_60", "MeanF1_90"]
    ].min(axis=1)
    ranking["CombinedMeanMCC"] = (
        ranking["MeanMCC_60"] + ranking["MeanMCC_90"]
    ) / 2.0
    ranking = ranking.loc[
        ranking["NumSpecies_90"].ge(MIN_NODES)
        & ranking["TestLinks_90"].ge(MIN_TEST_LINKS_90)
    ].copy()
    ranking = (
        ranking.sort_values(
            [
                "CombinedMeanF1",
                "WorstRatioMeanF1",
                "CombinedMeanMCC",
                "Foodweb",
            ],
            ascending=[False, False, False, True],
        )
        .head(NUM_OPTIONS)
        .reset_index(drop=True)
    )
    if len(ranking) != NUM_OPTIONS:
        raise ValueError(
            f"Expected {NUM_OPTIONS} eligible food webs, found {len(ranking)}."
        )
    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))
    return scores, ranking


def representative_experiment(scores, ranking_row):
    """Choose one ExperimentID representative at both ratios."""
    foodweb = ranking_row["Foodweb"]
    subset = scores.loc[scores["Foodweb"].eq(foodweb)].copy()
    wide = subset.pivot(
        index="ExperimentID",
        columns="TrainRatio",
        values=["TestF1Score", "TestFN", "TestLinks"],
    ).dropna()
    if len(wide) != 100:
        raise ValueError(
            f"{foodweb}: expected 100 shared ExperimentIDs, found {len(wide)}."
        )

    f1_distance = (
        (wide[("TestF1Score", 60)] - ranking_row["MeanF1_60"]).abs()
        + (wide[("TestF1Score", 90)] - ranking_row["MeanF1_90"]).abs()
    )
    fn_rate_distance = (
        (
            wide[("TestFN", 60)] / wide[("TestLinks", 60)]
            - ranking_row["MeanFN_60"] / ranking_row["TestLinks_60"]
        ).abs()
        + (
            wide[("TestFN", 90)] / wide[("TestLinks", 90)]
            - ranking_row["MeanFN_90"] / ranking_row["TestLinks_90"]
        ).abs()
    )
    candidates = pd.DataFrame(
        {
            "F1Distance": f1_distance,
            "FNRateDistance": fn_rate_distance,
        }
    ).rename_axis(None)
    candidates["ExperimentID"] = candidates.index.astype(int)
    selected = candidates.sort_values(
        ["F1Distance", "FNRateDistance", "ExperimentID"]
    ).iloc[0]
    return (
        int(selected["ExperimentID"]),
        float(selected["F1Distance"]),
        float(selected["FNRateDistance"]),
    )


tax_mass_cache = {}


def prepare_panel(foodweb, train_ratio, experiment_id, letter):
    panel = {
        "letter": letter,
        "foodweb": foodweb,
        "train_ratio": train_ratio,
        "experiment_id": experiment_id,
    }
    row, results_csv, ratio = select_run(panel)
    snapshot_path = local_snapshot_from_row(row)
    stem = tax_mass_stem(foodweb)
    if stem not in tax_mass_cache:
        tax_mass_cache[stem] = load_tax_mass(foodweb)

    tax_mass_path, base_net, taxonomy, mass = tax_mass_cache[stem]
    masks = load_snapshot(snapshot_path, base_net, taxonomy)
    seed, counts = validate_run(
        row=row,
        panel=panel,
        ratio=ratio,
        masks=masks,
        snapshot_path=snapshot_path,
    )

    mass_mg = np.round(mass * 1000.0, 4)
    sorted_original_idx = np.argsort(mass_mg, kind="stable")
    sorted_species = [taxonomy[index] for index in sorted_original_idx]
    sorted_mass_mg = mass_mg[sorted_original_idx]
    n = len(sorted_species)
    original_to_sorted = np.empty(n, dtype=int)
    original_to_sorted[sorted_original_idx] = np.arange(n)
    species_to_num = {
        species: index + 1 for index, species in enumerate(sorted_species)
    }
    df_train = mask_to_df(
        masks["Train"], original_to_sorted, "Train"
    )
    df_pm = pd.concat(
        [
            mask_to_df(masks["TP"], original_to_sorted, "TP"),
            mask_to_df(masks["FP"], original_to_sorted, "FP"),
            mask_to_df(masks["FN"], original_to_sorted, "FN"),
        ],
        ignore_index=True,
    )
    return {
        "letter": letter,
        "foodweb": canonical_foodweb(foodweb),
        "train_ratio": ratio,
        "experiment_id": int(experiment_id),
        "seed": seed,
        "counts": counts,
        "row": row,
        "results_csv": results_csv,
        "snapshot_path": snapshot_path,
        "tax_mass_path": tax_mass_path,
        "n": n,
        "sorted_species": sorted_species,
        "sorted_mass_mg": sorted_mass_mg,
        "species_to_num": species_to_num,
        "df_train": df_train,
        "df_pm": df_pm,
    }


def render_option(prepared_panels, output_path):
    """Render with the unchanged Figure 2 visual specification."""
    fig, axs = plt.subplots(
        1,
        len(prepared_panels),
        figsize=(12 * len(prepared_panels), 12),
        squeeze=False,
    )
    axs = axs.ravel()

    for panel_idx, panel in enumerate(prepared_panels):
        ax = axs[panel_idx]
        n = panel["n"]
        if not panel["df_train"].empty:
            sns.scatterplot(
                data=panel["df_train"],
                x="PredIdx",
                y="PreyIdx",
                color="green",
                marker="D",
                alpha=0.3,
                s=70,
                legend=False,
                ax=ax,
            )
        if not panel["df_pm"].empty:
            sns.scatterplot(
                data=panel["df_pm"],
                x="PredIdx",
                y="PreyIdx",
                hue="Label",
                style="Label",
                hue_order=["TP", "FP", "FN"],
                style_order=["TP", "FP", "FN"],
                markers={"TP": "o", "FP": "X", "FN": "s"},
                palette={"TP": "blue", "FP": "gray", "FN": "red"},
                s=70,
                legend=False,
                ax=ax,
            )

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(XLAB, fontsize=20)
        ax.set_ylabel(YLAB, fontsize=20)

        if SHOW_TICKS:
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            x_fontsize = PANEL_XTICK_FONTSIZES[panel_idx]
            y_fontsize = PANEL_YTICK_FONTSIZES[panel_idx]
            if USE_NUMERIC_AXIS_LABELS:
                labels = [
                    str(panel["species_to_num"][species])
                    for species in panel["sorted_species"]
                ]
            else:
                labels = panel["sorted_species"]
            ax.set_xticklabels(labels, rotation=90, fontsize=x_fontsize)
            ax.set_yticklabels(labels, fontsize=y_fontsize)
            ax.tick_params(axis="both", which="both", length=3)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis="both", which="both", length=0)

        if SHOW_TITLES:
            ax.set_title(panel["foodweb"], fontsize=12)
        ax.annotate(
            panel["letter"],
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(-10, 8),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=35,
            fontweight="bold",
            annotation_clip=False,
        )
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            prop={"size": 18},
            title="",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)


def main():
    for label, path in {
        "prediction_scores_logs": PREDICTION_LOG_DIR,
        "ecological_snapshots": ECOLOGICAL_SNAPSHOTS_DIR,
        "foodwebs_mat": BASE_MAT_PATH,
    }.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {label} directory: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPECIES_KEY_DIR.mkdir(parents=True, exist_ok=True)
    scores, ranking = build_ranking()
    records = []
    figure_paths = []

    for _, ranking_row in ranking.iterrows():
        rank = int(ranking_row["Rank"])
        foodweb = ranking_row["Foodweb"]
        experiment_id, f1_distance, fn_rate_distance = (
            representative_experiment(scores, ranking_row)
        )
        panels = [
            prepare_panel(foodweb, 90, experiment_id, "a"),
            prepare_panel(foodweb, 60, experiment_id, "b"),
        ]
        if panels[0]["sorted_species"] != panels[1]["sorted_species"]:
            raise ValueError(
                f"{foodweb}: species ordering differs between panels."
            )

        safe_foodweb = re.sub(
            r"[^A-Za-z0-9._-]+", "_", foodweb
        ).strip("_")
        figure_path = OUTPUT_DIR / f"{rank:02d}_{safe_foodweb}.png"
        species_key_path = (
            SPECIES_KEY_DIR
            / f"{rank:02d}_{safe_foodweb}_species_key.csv"
        )
        species_key = pd.DataFrame(
            {
                "Number": np.arange(1, panels[0]["n"] + 1),
                "Species": panels[0]["sorted_species"],
                "Mass_mg": panels[0]["sorted_mass_mg"],
            }
        )
        species_key["Mass_mg"] = species_key["Mass_mg"].map(
            lambda value: f"{value:.4f}"
        )
        species_key.to_csv(
            species_key_path, index=False, encoding="utf-8"
        )
        render_option(panels, figure_path)
        figure_paths.append(figure_path)

        panel_90, panel_60 = panels
        records.append(
            {
                "Rank": rank,
                "Foodweb": foodweb,
                "ExperimentID": experiment_id,
                "SelectionMethod": (
                    "closest_to_100_run_mean_F1_at_60_and_90"
                ),
                "RepresentativeF1Distance": f1_distance,
                "RepresentativeFNRateDistance": fn_rate_distance,
                "NumSpecies": int(ranking_row["NumSpecies_90"]),
                "EmpiricalLinks": int(ranking_row["EmpiricalLinks_90"]),
                "CombinedMeanF1": ranking_row["CombinedMeanF1"],
                "MeanF1_90": ranking_row["MeanF1_90"],
                "MeanTP_90": ranking_row["MeanTP_90"],
                "MeanFP_90": ranking_row["MeanFP_90"],
                "MeanFN_90": ranking_row["MeanFN_90"],
                "MeanF1_60": ranking_row["MeanF1_60"],
                "MeanTP_60": ranking_row["MeanTP_60"],
                "MeanFP_60": ranking_row["MeanFP_60"],
                "MeanFN_60": ranking_row["MeanFN_60"],
                "DisplaySeed_90": panel_90["seed"],
                "DisplayTrain_90": panel_90["counts"]["TrainLinks"],
                "DisplayTP_90": panel_90["counts"]["TestTP"],
                "DisplayFP_90": panel_90["counts"]["TestFP"],
                "DisplayFN_90": panel_90["counts"]["TestFN"],
                "DisplayF1_90": float(panel_90["row"]["TestF1Score"]),
                "DisplaySeed_60": panel_60["seed"],
                "DisplayTrain_60": panel_60["counts"]["TrainLinks"],
                "DisplayTP_60": panel_60["counts"]["TestTP"],
                "DisplayFP_60": panel_60["counts"]["TestFP"],
                "DisplayFN_60": panel_60["counts"]["TestFN"],
                "DisplayF1_60": float(panel_60["row"]["TestF1Score"]),
                "FigureFile": str(figure_path.relative_to(REPO_ROOT)),
                "SpeciesKeyFile": str(
                    species_key_path.relative_to(REPO_ROOT)
                ),
            }
        )
        print(
            f"[TOP20] {rank:02d}/20 {foodweb}: "
            f"ExperimentID={experiment_id}, "
            f"90% TP/FP/FN="
            f"{panel_90['counts']['TestTP']}/"
            f"{panel_90['counts']['TestFP']}/"
            f"{panel_90['counts']['TestFN']}, "
            f"60% TP/FP/FN="
            f"{panel_60['counts']['TestTP']}/"
            f"{panel_60['counts']['TestFP']}/"
            f"{panel_60['counts']['TestFN']}"
        )

    index = pd.DataFrame(records)
    index.to_csv(INDEX_PATH, index=False, encoding="utf-8")
    if (
        len(figure_paths) != NUM_OPTIONS
        or len(set(figure_paths)) != NUM_OPTIONS
        or not all(path.is_file() for path in figure_paths)
    ):
        raise RuntimeError(
            f"Export did not produce exactly {NUM_OPTIONS} validated files."
        )

    print(
        f"\n[DONE] Exported {len(figure_paths)} figures to "
        f"{OUTPUT_DIR.resolve()}"
    )
    print(f"[DONE] Ranking index: {INDEX_PATH.resolve()}")
    print(
        index[
            [
                "Rank",
                "Foodweb",
                "ExperimentID",
                "CombinedMeanF1",
                "DisplayTP_90",
                "DisplayFP_90",
                "DisplayFN_90",
                "DisplayTP_60",
                "DisplayFP_60",
                "DisplayFN_60",
                "FigureFile",
            ]
        ].to_string(index=False)
    )


main()
