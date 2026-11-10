#!/usr/bin/env python3
"""Primary and secondary analysis for the WLNM focal-mass A-D campaign."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon


VARIANTS = {
    "A": ("A_structure_only", "structure_only"),
    "B": ("B_endpoint_masses", "endpoint_masses"),
    "C": ("C_mass_ratio", "mass_ratio"),
    "D": ("D_endpoint_masses_ratio", "endpoint_masses_ratio"),
}
EXPECTED_RATIOS = np.arange(10.0, 91.0, 10.0)
METRICS = ("PR_AUC", "ROC_AUC", "F1Score", "TestFN", "TestFP")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def foodweb_from_path(path: Path) -> str:
    suffix = "_results_random_wlnm_dir_neg_focal_mass.csv"
    require(path.name.endswith(suffix), f"Unexpected result filename: {path.name}")
    return path.name[: -len(suffix)]


def read_variant(parent: Path, code: str) -> pd.DataFrame:
    directory, mode = VARIANTS[code]
    files = sorted((parent / directory / "prediction_scores_logs").glob("*.csv"))
    require(files, f"No prediction CSVs found for variant {code}")
    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        frame.insert(0, "Foodweb", foodweb_from_path(path))
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    required = {
        "Foodweb", "TrainRatio", "ExperimentID", "PositiveSplitHash",
        "NegativeSplitHash", "FocalMassFeatureMode", "ValidMassNodeCount",
        "MissingMassNodeCount", *METRICS,
    }
    missing = required - set(data.columns)
    require(not missing, f"Variant {code} is missing columns: {sorted(missing)}")
    require(set(data["FocalMassFeatureMode"].astype(str)) == {mode}, f"Variant {code} has wrong feature mode")
    data["Variant"] = code
    numeric = ["TrainRatio", "ExperimentID", "ValidMassNodeCount", "MissingMassNodeCount", *METRICS]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="raise")
    require(np.isfinite(data[numeric].to_numpy(dtype=float)).all(), f"Variant {code} has non-finite analysis values")
    return data


def normalized_auc(ratios: np.ndarray, values: np.ndarray) -> float:
    order = np.argsort(ratios)
    x = np.asarray(ratios, dtype=float)[order]
    y = np.asarray(values, dtype=float)[order]
    require(np.array_equal(x, EXPECTED_RATIOS), f"Expected ratios 10-90, got {x.tolist()}")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def holm_adjust(pvalues: list[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def rank_biserial(differences: np.ndarray) -> float:
    d = np.asarray(differences, dtype=float)
    d = d[d != 0]
    if not len(d):
        return 0.0
    ranks = rankdata(np.abs(d), method="average")
    positive = float(ranks[d > 0].sum())
    negative = float(ranks[d < 0].sum())
    return (positive - negative) / (positive + negative)


def bootstrap_median_ci(differences: np.ndarray, seed: int, draws: int = 10_000) -> tuple[float, float]:
    d = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(d, size=(draws, len(d)), replace=True)
    medians = np.median(samples, axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def signed_rank_summary(differences: np.ndarray, seed: int) -> dict[str, float | int]:
    d = np.asarray(differences, dtype=float)
    if np.all(d == 0):
        statistic, pvalue = 0.0, 1.0
    else:
        result = wilcoxon(d, alternative="two-sided", zero_method="wilcox", method="auto")
        statistic, pvalue = float(result.statistic), float(result.pvalue)
    low, high = bootstrap_median_ci(d, seed)
    return {
        "NFoodwebs": int(len(d)),
        "MedianDifference": float(np.median(d)),
        "Bootstrap95Low": low,
        "Bootstrap95High": high,
        "WilcoxonStatistic": statistic,
        "PValueRaw": pvalue,
        "RankBiserial": rank_biserial(d),
    }


def verify_pairing(variants: dict[str, pd.DataFrame]) -> None:
    keys = ["Foodweb", "TrainRatio", "ExperimentID"]
    reference = variants["A"][keys + ["PositiveSplitHash", "NegativeSplitHash"]].copy()
    require(not reference.duplicated(keys).any(), "A contains duplicate paired keys")
    for code, data in variants.items():
        require(not data.duplicated(keys).any(), f"{code} contains duplicate paired keys")
        merged = reference.merge(
            data[keys + ["PositiveSplitHash", "NegativeSplitHash"]],
            on=keys, how="outer", suffixes=("_A", f"_{code}"), indicator=True,
        )
        require((merged["_merge"] == "both").all(), f"{code} keys do not match A")
        require(
            (merged["PositiveSplitHash_A"] == merged[f"PositiveSplitHash_{code}"]).all()
            and (merged["NegativeSplitHash_A"] == merged[f"NegativeSplitHash_{code}"]).all(),
            f"{code} split hashes do not match A",
        )


def build_per_run_aulc(variants: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code, data in variants.items():
        for (foodweb, experiment), group in data.groupby(["Foodweb", "ExperimentID"], sort=True):
            rows.append({
                "Foodweb": foodweb,
                "ExperimentID": int(experiment),
                "Variant": code,
                "NormalizedPRAUC": normalized_auc(
                    group["TrainRatio"].to_numpy(), group["PR_AUC"].to_numpy()
                ),
            })
    return pd.DataFrame(rows)


def primary_comparisons(per_foodweb: pd.DataFrame) -> pd.DataFrame:
    wide = per_foodweb.pivot(index="Foodweb", columns="Variant", values="MeanNormalizedPRAUC")
    comparisons: list[dict[str, object]] = []
    for seed, code in enumerate(("B", "C", "D"), start=4101):
        delta = (wide[code] - wide["A"]).to_numpy()
        row = {"Comparison": f"{code}-A", **signed_rank_summary(delta, seed)}
        comparisons.append(row)
    output = pd.DataFrame(comparisons)
    output["PValueHolm3"] = holm_adjust(output["PValueRaw"].tolist())
    output["Supported"] = (
        (output["MedianDifference"] > 0)
        & (output["Bootstrap95Low"] > 0)
        & (output["PValueHolm3"] < 0.05)
    )
    return output


def secondary_pairwise(per_foodweb: pd.DataFrame) -> pd.DataFrame:
    wide = per_foodweb.pivot(index="Foodweb", columns="Variant", values="MeanNormalizedPRAUC")
    rows = []
    for seed, (left, right) in enumerate((("B", "C"), ("B", "D"), ("C", "D")), start=5101):
        rows.append({"Comparison": f"{left}-{right}", **signed_rank_summary((wide[left] - wide[right]).to_numpy(), seed)})
    return pd.DataFrame(rows)


def secondary_by_ratio(variants: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summaries = []
    for code, data in variants.items():
        grouped = data.groupby(["Foodweb", "TrainRatio"], as_index=False)[list(METRICS)].mean()
        grouped["Variant"] = code
        summaries.append(grouped)
    per_web = pd.concat(summaries, ignore_index=True)
    output = per_web.groupby(["Variant", "TrainRatio"])[list(METRICS)].agg(["mean", "median"])
    output.columns = [f"{metric}{statistic.title()}" for metric, statistic in output.columns]
    return output.reset_index()


def coverage_associations(variants: dict[str, pd.DataFrame], per_foodweb: pd.DataFrame) -> pd.DataFrame:
    base = variants["A"].groupby("Foodweb", as_index=False)[["ValidMassNodeCount", "MissingMassNodeCount"]].first()
    base["MassCoverage"] = base["ValidMassNodeCount"] / (base["ValidMassNodeCount"] + base["MissingMassNodeCount"])
    wide = per_foodweb.pivot(index="Foodweb", columns="Variant", values="MeanNormalizedPRAUC").reset_index()
    joined = base.merge(wide, on="Foodweb", validate="one_to_one")
    rows = []
    for code in ("B", "C", "D"):
        delta = joined[code] - joined["A"]
        result = spearmanr(joined["MassCoverage"], delta)
        rows.append({
            "Comparison": f"{code}-A",
            "NFoodwebs": len(joined),
            "SpearmanRho": float(result.statistic),
            "PValueRaw": float(result.pvalue),
        })
    return pd.DataFrame(rows)


def add_exploratory_ecosystems(per_foodweb: pd.DataFrame, metadata_path: Path, output_dir: Path) -> None:
    metadata = pd.read_csv(metadata_path, usecols=["Foodweb", "EcosystemType"])
    joined = per_foodweb.merge(metadata, on="Foodweb", how="left", validate="many_to_one")
    require(joined["EcosystemType"].notna().all(), "Some food webs lack posthoc EcosystemType metadata")
    summary = joined.groupby(["EcosystemType", "Variant"])["MeanNormalizedPRAUC"].agg(
        Count="count", Mean="mean", Median="median"
    ).reset_index()
    summary.to_csv(output_dir / "exploratory_ecosystem_summary.csv", index=False)


def analyze(parent: Path, output_dir: Path, ecosystem_csv: Path | None = None) -> None:
    variants = {code: read_variant(parent, code) for code in VARIANTS}
    verify_pairing(variants)
    per_run = build_per_run_aulc(variants)
    per_foodweb = per_run.groupby(["Foodweb", "Variant"], as_index=False)["NormalizedPRAUC"].mean()
    per_foodweb = per_foodweb.rename(columns={"NormalizedPRAUC": "MeanNormalizedPRAUC"})
    require(per_foodweb["Foodweb"].nunique() == 290, "Primary inference requires 290 food webs")
    experiment_counts = per_run.groupby(["Foodweb", "Variant"])["ExperimentID"].nunique()
    require((experiment_counts == 100).all(), "Primary inference requires 100 experiments per food web and variant")

    output_dir.mkdir(parents=True, exist_ok=False)
    per_run.to_csv(output_dir / "per_run_normalized_pr_auc.csv", index=False)
    per_foodweb.to_csv(output_dir / "per_foodweb_normalized_pr_auc.csv", index=False)
    primary_comparisons(per_foodweb).to_csv(output_dir / "primary_comparisons.csv", index=False)
    secondary_pairwise(per_foodweb).to_csv(output_dir / "secondary_pairwise_comparisons.csv", index=False)
    secondary_by_ratio(variants).to_csv(output_dir / "secondary_metrics_by_train_ratio.csv", index=False)
    coverage_associations(variants, per_foodweb).to_csv(output_dir / "mass_coverage_associations.csv", index=False)
    if ecosystem_csv is not None:
        add_exploratory_ecosystems(per_foodweb, ecosystem_csv, output_dir)
    print(f"ANALYSIS_OK output={output_dir.resolve()} foodwebs=290 experiments=100")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--ecosystem-csv", type=Path, default=None)
    args = parser.parse_args()
    analyze(args.parent_root, args.output_dir, args.ecosystem_csv)


if __name__ == "__main__":
    main()
