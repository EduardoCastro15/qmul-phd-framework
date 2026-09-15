#!/usr/bin/env python3
"""Calculate Figure 5 paired Wilcoxon tests within ecosystem types.

The food web is the unit of analysis. For every food web and ecological
metric, the paired difference is:

    MeanPseudoAfterFiltering - EmpiricalValue

The test configuration matches the existing Figure 5 analysis:
two-sided, zero_method='wilcox', method='auto', no continuity correction,
and differences rounded to 12 decimal places. Holm-adjusted p-values are
reported across all 20 ecosystem-by-metric tests (primary), within each
ecosystem across the four metrics (literal extension of the prior protocol),
and within each metric across its five ecosystem types (secondary).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_NAME = "tukey_iqr_1p5_min50pct_threshold0p50_v1"
SCENARIO = "role_only_5_role_constraints_checkconnfalse_adaptivefalse_threshold0p50"
TRAIN_RATIO = 60
THRESHOLD = 0.50
ALPHA = 0.05
DIFFERENCE_DECIMALS = 12

RESULT_ROOT = (
    REPO_ROOT
    / "src/matlab/data"
    / (
        "result_wlnm_dir_neg_sweep_train_ratios_"
        "10-90_pseudo_properties_Apocrita_neg_const"
    )
)
RETENTION_ROOT = RESULT_ROOT / "retention_protocol" / PROTOCOL_NAME
OUTPUT_DIR = RETENTION_ROOT / "statistical_tests_Wilcoxon signed-rank test"

INPUT_PAIRS = OUTPUT_DIR / (
    "figure_5_wilcoxon_train60_roleonly_5constraints_"
    "checkconnfalse_adaptivefalse_threshold0p50_after_tukey_pairs.csv"
)
OUTPUT_STEM = (
    "figure_5_wilcoxon_by_ecosystem_train60_roleonly_5constraints_"
    "checkconnfalse_adaptivefalse_threshold0p50_after_tukey"
)
OUTPUT_RESULTS = OUTPUT_DIR / f"{OUTPUT_STEM}_results.csv"
OUTPUT_PAIRS = OUTPUT_DIR / f"{OUTPUT_STEM}_pairs.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / f"{OUTPUT_STEM}_summary.md"

ECOSYSTEM_ORDER = [
    "lakes",
    "streams",
    "marine",
    "terrestrial aboveground",
    "terrestrial belowground",
]
ECOSYSTEM_LABELS = {
    "lakes": "Lakes",
    "streams": "Streams",
    "marine": "Marine",
    "terrestrial aboveground": "Terrestrial aboveground",
    "terrestrial belowground": "Terrestrial belowground",
}
METRIC_ORDER = [
    "Connectance",
    "MeanTrophicHeight",
    "MeanGenerality",
    "MeanVulnerability",
]
METRIC_LABELS = {
    "Connectance": "Connectance",
    "MeanTrophicHeight": "Mean trophic height",
    "MeanGenerality": "Mean generality",
    "MeanVulnerability": "Mean vulnerability",
}


def adjust_pvalues_holm(p_values: np.ndarray) -> np.ndarray:
    """Return Holm-adjusted p-values in their original order."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(p_values))
    if valid_indices.size == 0:
        return adjusted

    valid_p_values = p_values[valid_indices]
    order = np.argsort(valid_p_values)
    ordered_p_values = valid_p_values[order]
    number_of_tests = len(ordered_p_values)
    multipliers = number_of_tests - np.arange(number_of_tests)
    ordered_adjusted = np.maximum.accumulate(multipliers * ordered_p_values)
    ordered_adjusted = np.minimum(ordered_adjusted, 1.0)
    valid_adjusted = np.empty_like(ordered_adjusted)
    valid_adjusted[order] = ordered_adjusted
    adjusted[valid_indices] = valid_adjusted
    return adjusted


def calculate_signed_rank_effect(differences: np.ndarray):
    """Return positive/negative rank sums and rank-biserial correlation."""
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[differences != 0.0]
    if nonzero.size == 0:
        return 0.0, 0.0, 0.0

    ranks = rankdata(np.abs(nonzero), method="average")
    positive_rank_sum = float(ranks[nonzero > 0.0].sum())
    negative_rank_sum = float(ranks[nonzero < 0.0].sum())
    total_rank_sum = positive_rank_sum + negative_rank_sum
    rank_biserial = (
        (positive_rank_sum - negative_rank_sum) / total_rank_sum
    )
    return positive_rank_sum, negative_rank_sum, float(rank_biserial)


def validate_and_prepare_pairs(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Figure 5 paired input: {path}")

    pairs = pd.read_csv(path)
    required = {
        "Scenario",
        "Version",
        "FoodWeb",
        "FW_KEY",
        "Ecosystem",
        "Metric",
        "EmpiricalValue",
        "MeanPseudoAfterFiltering",
        "ValidRunsAfterFiltering",
        "OutlierRunsRemoved",
    }
    missing = sorted(required.difference(pairs.columns))
    if missing:
        raise ValueError("Input is missing columns: " + ", ".join(missing))

    pairs["Ecosystem"] = (
        pairs["Ecosystem"].astype(str).str.strip().str.casefold()
    )
    pairs["Metric"] = pairs["Metric"].astype(str).str.strip()
    for column in ["EmpiricalValue", "MeanPseudoAfterFiltering"]:
        pairs[column] = pd.to_numeric(pairs[column], errors="coerce")

    if pairs[["EmpiricalValue", "MeanPseudoAfterFiltering"]].isna().any().any():
        raise ValueError("Missing empirical or pseudo values in paired input.")

    unexpected_ecosystems = sorted(set(pairs["Ecosystem"]) - set(ECOSYSTEM_ORDER))
    if unexpected_ecosystems:
        raise ValueError(
            "Unexpected ecosystem types: " + ", ".join(unexpected_ecosystems)
        )
    missing_ecosystems = sorted(set(ECOSYSTEM_ORDER) - set(pairs["Ecosystem"]))
    if missing_ecosystems:
        raise ValueError(
            "Expected ecosystem types are absent: " + ", ".join(missing_ecosystems)
        )

    unexpected_metrics = sorted(set(pairs["Metric"]) - set(METRIC_ORDER))
    if unexpected_metrics:
        raise ValueError("Unexpected metrics: " + ", ".join(unexpected_metrics))
    missing_metrics = sorted(set(METRIC_ORDER) - set(pairs["Metric"]))
    if missing_metrics:
        raise ValueError("Expected metrics are absent: " + ", ".join(missing_metrics))

    duplicates = pairs.duplicated(subset=["FW_KEY", "Metric"], keep=False)
    if duplicates.any():
        duplicate_rows = pairs.loc[duplicates, ["FW_KEY", "Metric"]]
        raise ValueError(
            "Duplicated food-web/metric pairs found:\n"
            + duplicate_rows.to_string(index=False)
        )

    pairs["Difference"] = np.round(
        pairs["MeanPseudoAfterFiltering"] - pairs["EmpiricalValue"],
        decimals=DIFFERENCE_DECIMALS,
    )
    pairs["DifferenceDirection"] = np.select(
        [pairs["Difference"] > 0.0, pairs["Difference"] < 0.0],
        ["Pseudo > empirical", "Pseudo < empirical"],
        default="Equal",
    )
    pairs["EcosystemLabel"] = pairs["Ecosystem"].map(ECOSYSTEM_LABELS)
    pairs["MetricLabel"] = pairs["Metric"].map(METRIC_LABELS)
    return pairs


def run_tests(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ecosystem in ECOSYSTEM_ORDER:
        for metric in METRIC_ORDER:
            group = (
                pairs.loc[
                    pairs["Ecosystem"].eq(ecosystem)
                    & pairs["Metric"].eq(metric)
                ]
                .sort_values("FW_KEY")
                .copy()
            )
            if group.empty:
                raise ValueError(f"No pairs for ecosystem={ecosystem}, metric={metric}")

            differences = group["Difference"].to_numpy(dtype=float)
            nonzero_count = int(np.count_nonzero(differences != 0.0))
            positive_rank_sum, negative_rank_sum, rank_biserial = (
                calculate_signed_rank_effect(differences)
            )

            if nonzero_count == 0:
                statistic = 0.0
                p_value = 1.0
                status = "all_differences_zero"
            else:
                test = wilcoxon(
                    differences,
                    zero_method="wilcox",
                    correction=False,
                    alternative="two-sided",
                    method="auto",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
                status = "ok"

            rows.append(
                {
                    "Analysis": "paired_wilcoxon_signed_rank_by_foodweb_and_ecosystem",
                    "Ecosystem": ecosystem,
                    "EcosystemLabel": ECOSYSTEM_LABELS[ecosystem],
                    "Metric": metric,
                    "MetricLabel": METRIC_LABELS[metric],
                    "NPairs": len(group),
                    "NNonzeroDifferences": nonzero_count,
                    "NPositiveDifferences": int(np.count_nonzero(differences > 0.0)),
                    "NNegativeDifferences": int(np.count_nonzero(differences < 0.0)),
                    "NZeroDifferences": int(np.count_nonzero(differences == 0.0)),
                    "MedianEmpirical": float(group["EmpiricalValue"].median()),
                    "MedianPseudoAfterTukey": float(
                        group["MeanPseudoAfterFiltering"].median()
                    ),
                    "MeanDifference": float(np.mean(differences)),
                    "MedianDifference": float(np.median(differences)),
                    "WilcoxonStatistic": statistic,
                    "PositiveRankSum": positive_rank_sum,
                    "NegativeRankSum": negative_rank_sum,
                    "RankBiserialCorrelation": rank_biserial,
                    "PValueRaw": p_value,
                    "Status": status,
                }
            )

    results = pd.DataFrame(rows)
    results["PValueHolmGlobal20"] = adjust_pvalues_holm(
        results["PValueRaw"].to_numpy(dtype=float)
    )
    results["PValueHolmWithinEcosystem4"] = np.nan
    for ecosystem in ECOSYSTEM_ORDER:
        mask = results["Ecosystem"].eq(ecosystem)
        results.loc[mask, "PValueHolmWithinEcosystem4"] = adjust_pvalues_holm(
            results.loc[mask, "PValueRaw"].to_numpy(dtype=float)
        )
    results["PValueHolmWithinMetric5"] = np.nan
    for metric in METRIC_ORDER:
        mask = results["Metric"].eq(metric)
        results.loc[mask, "PValueHolmWithinMetric5"] = adjust_pvalues_holm(
            results.loc[mask, "PValueRaw"].to_numpy(dtype=float)
        )

    results["RejectH0Raw"] = results["PValueRaw"] < ALPHA
    results["RejectH0HolmGlobal20"] = results["PValueHolmGlobal20"] < ALPHA
    results["RejectH0HolmWithinEcosystem4"] = (
        results["PValueHolmWithinEcosystem4"] < ALPHA
    )
    results["RejectH0HolmWithinMetric5"] = (
        results["PValueHolmWithinMetric5"] < ALPHA
    )
    results["TrainRatio"] = TRAIN_RATIO
    results["Threshold"] = THRESHOLD
    results["DifferenceDefinition"] = (
        "MeanPseudoAfterFiltering - EmpiricalValue"
    )
    results["Alternative"] = "two-sided"
    results["ZeroMethod"] = "wilcox"
    results["MethodRequested"] = "auto"
    results["ContinuityCorrection"] = False
    results["PrimaryPValueAdjustment"] = "Holm across all 20 ecosystem-by-metric tests"
    results["ProtocolMatchedPValueAdjustment"] = (
        "Holm within each ecosystem across the four ecological metrics"
    )
    results["AdditionalPValueAdjustment"] = (
        "Holm within each metric across the five ecosystem types"
    )
    results["Alpha"] = ALPHA
    results["DifferenceDecimals"] = DIFFERENCE_DECIMALS
    results["RetentionProtocol"] = PROTOCOL_NAME
    results["Scenario"] = SCENARIO
    results["SciPyVersion"] = scipy.__version__
    results["SourceFile"] = str(INPUT_PAIRS)
    return results


def format_p(value: float) -> str:
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def write_summary(results: pd.DataFrame, output_path: Path):
    lines = [
        "# Figure 5 Wilcoxon signed-rank tests by ecosystem type",
        "",
        f"- Train ratio: {TRAIN_RATIO}%",
        f"- Threshold: {THRESHOLD:.2f}",
        "- Unit of analysis: food web",
        "- Difference: post-Tukey pseudo mean minus empirical value",
        "- Test: paired, two-sided Wilcoxon signed-rank; `zero_method=wilcox`; `method=auto`",
        "- Primary multiplicity correction: Holm across all 20 ecosystem-by-metric tests",
        "- Protocol-matched correction: Holm within each ecosystem across four ecological metrics",
        "- Additional correction: Holm within each metric across five ecosystem types",
        "",
        "A positive median difference means that pseudo food webs have a higher value; a negative value means they have a lower value.",
        "",
        "| Ecosystem | Metric | n | Median difference | Rank-biserial r | Raw p | Holm p (20) | Reject global H0 |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.EcosystemLabel} | {row.MetricLabel} | {row.NPairs} | "
            f"{row.MedianDifference:.6g} | {row.RankBiserialCorrelation:.3f} | "
            f"{format_p(row.PValueRaw)} | {format_p(row.PValueHolmGlobal20)} | "
            f"{'Yes' if row.RejectH0HolmGlobal20 else 'No'} |"
        )

    significant = results.loc[results["RejectH0HolmGlobal20"]]
    nonsignificant = results.loc[~results["RejectH0HolmGlobal20"]]
    lines.extend(
        [
            "",
            "## Plain-language summary",
            "",
            f"After the global Holm correction, {len(significant)} of {len(results)} ecosystem-by-metric comparisons reject the null hypothesis of a zero median paired difference.",
            "",
        ]
    )
    if not significant.empty:
        lines.append("Globally significant comparisons:")
        lines.append("")
        for row in significant.itertuples(index=False):
            direction = "higher" if row.MedianDifference > 0 else "lower"
            lines.append(
                f"- {row.EcosystemLabel}, {row.MetricLabel}: pseudo values are typically {direction} "
                f"(median difference {row.MedianDifference:.6g}, Holm p={format_p(row.PValueHolmGlobal20)}, "
                f"rank-biserial r={row.RankBiserialCorrelation:.3f})."
            )
    if not nonsignificant.empty:
        lines.extend(
            [
                "",
                "A non-significant result does not demonstrate equivalence or similarity; it only means that this test did not provide sufficient evidence of a non-zero paired difference for that ecosystem and metric.",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    global INPUT_PAIRS, OUTPUT_RESULTS, OUTPUT_PAIRS, OUTPUT_SUMMARY

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    INPUT_PAIRS = args.input.resolve()
    output_dir = args.output_dir.resolve()
    OUTPUT_RESULTS = output_dir / f"{OUTPUT_STEM}_results.csv"
    OUTPUT_PAIRS = output_dir / f"{OUTPUT_STEM}_pairs.csv"
    OUTPUT_SUMMARY = output_dir / f"{OUTPUT_STEM}_summary.md"
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = validate_and_prepare_pairs(INPUT_PAIRS)
    results = run_tests(pairs)

    ecosystem_rank = {value: index for index, value in enumerate(ECOSYSTEM_ORDER)}
    metric_rank = {value: index for index, value in enumerate(METRIC_ORDER)}
    pairs = pairs.assign(
        _ecosystem_order=pairs["Ecosystem"].map(ecosystem_rank),
        _metric_order=pairs["Metric"].map(metric_rank),
    ).sort_values(["_ecosystem_order", "_metric_order", "FoodWeb"])
    pairs = pairs.drop(columns=["_ecosystem_order", "_metric_order"])

    results.to_csv(OUTPUT_RESULTS, index=False)
    pairs.to_csv(OUTPUT_PAIRS, index=False)
    write_summary(results, OUTPUT_SUMMARY)

    display_columns = [
        "EcosystemLabel",
        "MetricLabel",
        "NPairs",
        "MedianDifference",
        "RankBiserialCorrelation",
        "PValueRaw",
        "PValueHolmGlobal20",
        "RejectH0HolmGlobal20",
    ]
    print(results[display_columns].to_string(index=False))
    print(f"\nResults: {OUTPUT_RESULTS}")
    print(f"Pairs:   {OUTPUT_PAIRS}")
    print(f"Summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
