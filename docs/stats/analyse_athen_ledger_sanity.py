#!/usr/bin/env python3
"""Ledger et al. sanity comparison for WLNM_dir_neg reconstructions.

This analysis harmonises definitions before comparing reconstruction error
with the unweighted Mill Stream benchmarks:

* connectance = L / S^2, as reported by Ledger et al.;
* mean generality = L / number of consumers;
* mean vulnerability = L / number of resources;
* mean trophic height = mean of all NetworkX trophic levels in the largest
  weakly connected component, including basal species at level 1.

The existing logs store the trophic mean after excluding basal species. For a
successful solve, the all-species mean is reconstructed exactly from the
stored mean and species counts, so WLNM does not need to be rerun.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from analyse_athen_web_mean_equivalence import (
    confidence_interval,
    parse_float,
    parse_int,
    read_metadata,
    sample_stats,
    web_from_filename,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = (
    ROOT
    / "src/matlab/data/"
    "result_wlnm_dir_neg_sweep_train_ratios_10-90_pseudo_properties_stats_Apocrita"
)


@dataclass(frozen=True)
class LedgerBenchmark:
    metric: str
    display_name: str
    mean: float
    sem: float
    source: str
    definition: str

    @property
    def relative_sem_percent(self) -> float:
        return 100.0 * self.sem / self.mean


BENCHMARKS = (
    LedgerBenchmark(
        "Connectance",
        "Connectance",
        0.09,
        0.01,
        "Ledger et al. 2013 Supplementary Table 3, control webs",
        "L / S^2 (unweighted)",
    ),
    LedgerBenchmark(
        "MeanGenerality",
        "Mean generality",
        11.68,
        1.11,
        "Ledger et al. 2013 Supplementary Table 3, control webs",
        "L / number of consumers (unweighted)",
    ),
    LedgerBenchmark(
        "MeanVulnerability",
        "Mean vulnerability",
        6.63,
        0.63,
        "Ledger et al. 2013 Supplementary Table 3, control webs",
        "L / number of resources (unweighted)",
    ),
    LedgerBenchmark(
        "MeanTrophicHeight",
        "Mean trophic height",
        1.54,
        0.004,
        "Ledger et al. 2013 main article, control webs",
        "Prey-averaged trophic level; basal species included at level 1",
    ),
)
BENCHMARK_BY_METRIC = {benchmark.metric: benchmark for benchmark in BENCHMARKS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=RESULTS_DIR / "prediction_scores_logs",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=ROOT / "src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "statistical_tests/athen/ledger_sanity",
    )
    parser.add_argument("--main-train-ratio", type=int, default=60)
    parser.add_argument("--min-valid-trophic-runs", type=int, default=10)
    return parser.parse_args()


def ledger_connectance(row: Dict[str, str], side: str) -> Optional[float]:
    species = parse_int(row.get(f"{side}NumSpecies"))
    links = parse_int(row.get(f"{side}Links"))
    if species is None or links is None or species <= 0:
        return None
    return links / (species * species)


def logged_metric(row: Dict[str, str], side: str, suffix: str) -> Optional[float]:
    return parse_float(row.get(f"{side}{suffix}"))


def networkx_all_lcc_mean(row: Dict[str, str], side: str) -> Optional[float]:
    """Recover the NetworkX mean including basal species at trophic level 1."""
    status = parse_int(row.get(f"{side}NetworkXTrophicLevelStatusCode"))
    nonbasal_mean = parse_float(row.get(f"{side}NetworkXMeanTrophicLevel"))
    largest_n = parse_int(row.get(f"{side}NetworkXTrophicLevelNumSpeciesLargest"))
    nonbasal_n = parse_int(row.get(f"{side}NetworkXTrophicLevelNumSpeciesWithLevel"))

    if (
        status != 0
        or nonbasal_mean is None
        or largest_n is None
        or nonbasal_n is None
        or largest_n <= 0
        or nonbasal_n < 0
        or nonbasal_n > largest_n
    ):
        return None

    basal_n = largest_n - nonbasal_n
    return (nonbasal_mean * nonbasal_n + basal_n) / largest_n


MetricFunction = Callable[[Dict[str, str], str], Optional[float]]
METRIC_FUNCTIONS: Dict[str, MetricFunction] = {
    "Connectance": ledger_connectance,
    "MeanGenerality": lambda row, side: logged_metric(row, side, "MeanGenerality"),
    "MeanVulnerability": lambda row, side: logged_metric(
        row, side, "MeanVulnerability"
    ),
    "MeanTrophicHeight": networkx_all_lcc_mean,
}


def read_web_level_rows(
    logs_dir: Path,
    metadata: Dict[str, str],
    min_valid_trophic_runs: int,
) -> Tuple[List[Dict[str, object]], int, int]:
    grouped: Dict[Tuple[str, int, str], Dict[str, object]] = {}
    files = sorted(logs_dir.glob("*_results_random_wlnm_dir_neg.csv"))
    raw_rows = 0

    for path in files:
        web = web_from_filename(path)
        ecosystem = metadata.get(web, "unknown")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                raw_rows += 1
                ratio = parse_int(row.get("TrainRatio"))
                if ratio is None:
                    continue

                for metric, metric_function in METRIC_FUNCTIONS.items():
                    key = (web, ratio, metric)
                    group = grouped.setdefault(
                        key,
                        {
                            "FoodWeb": web,
                            "EcosystemType": ecosystem,
                            "TrainRatio": ratio,
                            "Metric": metric,
                            "TotalRuns": 0,
                            "EmpiricalValues": [],
                            "PseudoValues": [],
                        },
                    )
                    group["TotalRuns"] = int(group["TotalRuns"]) + 1
                    empirical = metric_function(row, "Empirical")
                    pseudo = metric_function(row, "Pseudo")
                    if empirical is not None:
                        group["EmpiricalValues"].append(empirical)  # type: ignore[union-attr]
                    if pseudo is not None:
                        group["PseudoValues"].append(pseudo)  # type: ignore[union-attr]

    result: List[Dict[str, object]] = []
    for group in grouped.values():
        empirical_values = list(group["EmpiricalValues"])  # type: ignore[arg-type]
        pseudo_values = list(group["PseudoValues"])  # type: ignore[arg-type]
        total_runs = int(group["TotalRuns"])
        metric = str(group["Metric"])
        required_runs = min_valid_trophic_runs if metric == "MeanTrophicHeight" else total_runs
        empirical = statistics.mean(empirical_values) if empirical_values else math.nan
        pseudo = statistics.mean(pseudo_values) if pseudo_values else math.nan
        valid_runs = len(pseudo_values)
        included = (
            math.isfinite(empirical)
            and math.isfinite(pseudo)
            and empirical != 0
            and valid_runs >= required_runs
        )
        delta = pseudo - empirical if included else math.nan
        relative = delta / empirical if included else math.nan
        result.append(
            {
                "FoodWeb": group["FoodWeb"],
                "EcosystemType": group["EcosystemType"],
                "TrainRatio": group["TrainRatio"],
                "Metric": metric,
                "TotalRuns": total_runs,
                "ValidPseudoRuns": valid_runs,
                "InvalidPseudoRuns": total_runs - valid_runs,
                "EmpiricalValue": empirical,
                "MeanPseudo": pseudo,
                "DeltaPseudoMinusEmpirical": delta,
                "RelativeError": relative,
                "RelativeErrorPercent": 100.0 * relative if included else math.nan,
                "Included": int(included),
                "Complete20": int(valid_runs == 20),
            }
        )

    result.sort(
        key=lambda item: (
            int(item["TrainRatio"]),
            str(item["Metric"]),
            str(item["FoodWeb"]),
        )
    )
    return result, len(files), raw_rows


def summarise_subset(
    rows: Sequence[Dict[str, object]],
    ratio: int,
    metric: str,
    scope: str,
    ecosystem: str,
) -> Optional[Dict[str, object]]:
    selected = [
        row
        for row in rows
        if int(row["TrainRatio"]) == ratio
        and row["Metric"] == metric
        and int(row["Included"]) == 1
        and (scope == "Overall" or row["EcosystemType"] == ecosystem)
    ]
    if not selected:
        return None

    empirical = [float(row["EmpiricalValue"]) for row in selected]
    pseudo = [float(row["MeanPseudo"]) for row in selected]
    delta = [float(row["DeltaPseudoMinusEmpirical"]) for row in selected]
    relative = [float(row["RelativeError"]) for row in selected]
    valid_runs = [int(row["ValidPseudoRuns"]) for row in selected]
    empirical_stats = sample_stats(empirical)
    pseudo_stats = sample_stats(pseudo)
    delta_stats = sample_stats(delta)
    relative_stats = sample_stats(relative)
    delta_ci = confidence_interval(delta)
    relative_ci = confidence_interval(relative)
    benchmark = BENCHMARK_BY_METRIC[metric]
    relative_sem = benchmark.relative_sem_percent
    mean_relative_percent = 100.0 * relative_stats["Mean"]
    relative_lower_percent = 100.0 * relative_ci[0]
    relative_upper_percent = 100.0 * relative_ci[1]
    worst_ci_percent = max(abs(relative_lower_percent), abs(relative_upper_percent))

    return {
        "TrainRatio": ratio,
        "Scope": scope,
        "EcosystemType": ecosystem,
        "Metric": metric,
        "MetricLabel": benchmark.display_name,
        "NumFoodWebs": len(selected),
        "NumComplete20FoodWebs": sum(int(row["Complete20"]) for row in selected),
        "MinValidPseudoRuns": min(valid_runs),
        "MedianValidPseudoRuns": statistics.median(valid_runs),
        "MeanEmpirical": empirical_stats["Mean"],
        "MeanPseudo": pseudo_stats["Mean"],
        "MeanDelta": delta_stats["Mean"],
        "DeltaCI90Lower": delta_ci[0],
        "DeltaCI90Upper": delta_ci[1],
        "MeanRelativeErrorPercent": mean_relative_percent,
        "RelativeCI90LowerPercent": relative_lower_percent,
        "RelativeCI90UpperPercent": relative_upper_percent,
        "RelativeError90CI": (
            f"{mean_relative_percent:+.2f}% "
            f"[{relative_lower_percent:+.2f}%, {relative_upper_percent:+.2f}%]"
        ),
        "LedgerMean": benchmark.mean,
        "LedgerSEM": benchmark.sem,
        "LedgerRelativeSEMPercent": relative_sem,
        "AbsMeanDeltaWithinLedgerSEM": int(abs(delta_stats["Mean"]) <= benchmark.sem),
        "AbsMeanRelativeErrorWithinLedgerRelativeSEM": int(
            abs(mean_relative_percent) <= relative_sem
        ),
        "CI90WithinLedgerRelativeSEM": int(worst_ci_percent <= relative_sem),
        "SanityConclusion": (
            "corresponds_within_ledger_relative_sem"
            if worst_ci_percent <= relative_sem
            else "does_not_correspond_within_ledger_relative_sem"
        ),
        "ComparisonRule": (
            "entire 90% CI of mean web-level relative error must fall within "
            "+/- Ledger relative SEM"
        ),
    }


def build_summaries(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    ratios = sorted({int(row["TrainRatio"]) for row in rows})
    ecosystems = sorted(
        {str(row["EcosystemType"]) for row in rows if row["EcosystemType"] != "unknown"}
    )
    for ratio in ratios:
        for benchmark in BENCHMARKS:
            overall = summarise_subset(
                rows, ratio, benchmark.metric, "Overall", "all"
            )
            if overall is not None:
                summaries.append(overall)
            for ecosystem in ecosystems:
                summary = summarise_subset(
                    rows, ratio, benchmark.metric, "Ecosystem", ecosystem
                )
                if summary is not None:
                    summaries.append(summary)
    return summaries


def benchmark_rows() -> List[Dict[str, object]]:
    return [
        {
            "Metric": item.metric,
            "MetricLabel": item.display_name,
            "Mean": item.mean,
            "SEM": item.sem,
            "RelativeSEMPercent": item.relative_sem_percent,
            "Definition": item.definition,
            "Source": item.source,
        }
        for item in BENCHMARKS
    ]


def excel_rows(main_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    scope_order = {
        "all": 0,
        "lakes": 1,
        "marine": 2,
        "streams": 3,
        "terrestrial aboveground": 4,
        "terrestrial belowground": 5,
    }
    metric_order = {item.metric: index for index, item in enumerate(BENCHMARKS)}
    ordered = sorted(
        main_rows,
        key=lambda row: (
            scope_order[str(row["EcosystemType"])],
            metric_order[str(row["Metric"])],
        ),
    )
    return [
        {
            "Section": row["EcosystemType"],
            "Metric": row["MetricLabel"],
            "N webs": row["NumFoodWebs"],
            "Empirical mean": row["MeanEmpirical"],
            "Mean pseudo": row["MeanPseudo"],
            "Mean delta": row["MeanDelta"],
            "Absolute mean delta": abs(float(row["MeanDelta"])),
            "Mean relative error (90% CI)": row["RelativeError90CI"],
            "Ledger mean": row["LedgerMean"],
            "Ledger SEM": row["LedgerSEM"],
            "Ledger relative SEM (%)": row["LedgerRelativeSEMPercent"],
            "Corresponds (90% CI within relative SEM)": row[
                "CI90WithinLedgerRelativeSEM"
            ],
        }
        for row in ordered
    ]


def write_readme(path: Path, main_rows: Sequence[Dict[str, object]]) -> None:
    overall = [row for row in main_rows if row["Scope"] == "Overall"]
    lines = [
        "# Ledger sanity comparison",
        "",
        "This is a descriptive sanity comparison, not an equivalence test.",
        "One observation is one empirical food web compared with the mean of its valid pseudo reconstructions.",
        "Connectance is recalculated as L/S^2. NetworkX trophic height includes basal species at level 1 and is averaged over the largest weakly connected component.",
        "",
        "The conservative correspondence rule requires the entire 90% confidence interval of the mean web-level relative error to be contained within +/- the relative SEM reported for the four Ledger control webs.",
        "SEM describes precision of the four-web mean; it is not the full range of natural variation.",
        "",
        "## TrainRatio 60 overall",
        "",
        "| Metric | N | Mean empirical | Mean pseudo | Mean delta | Relative error (90% CI) | Ledger relative SEM | Corresponds? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            "| {MetricLabel} | {NumFoodWebs} | {MeanEmpirical:.4f} | "
            "{MeanPseudo:.4f} | {MeanDelta:+.4f} | {RelativeError90CI} | "
            "+/-{LedgerRelativeSEMPercent:.2f}% | {CI90WithinLedgerRelativeSEM} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    metadata = read_metadata(args.metadata_file)
    web_rows, log_files, raw_rows = read_web_level_rows(
        args.logs_dir, metadata, args.min_valid_trophic_runs
    )
    summaries = build_summaries(web_rows)
    main_rows = [
        row for row in summaries if int(row["TrainRatio"]) == args.main_train_ratio
    ]
    overall_main = [row for row in main_rows if row["Scope"] == "Overall"]
    ecosystem_main = [row for row in main_rows if row["Scope"] == "Ecosystem"]

    output = args.output_dir
    write_csv(output / "ledger_benchmarks.csv", benchmark_rows())
    write_csv(output / "ledger_web_level_metrics_all_ratios.csv", web_rows)
    write_csv(output / "ledger_sanity_all_ratios.csv", summaries)
    write_csv(output / "ledger_sanity_train60_overall.csv", overall_main)
    write_csv(output / "ledger_sanity_train60_by_ecosystem.csv", ecosystem_main)
    write_csv(output / "ledger_excel_table_train60.csv", excel_rows(main_rows))
    write_csv(
        output / "ledger_sanity_run_summary.csv",
        [
            {"Key": "Analysis", "Value": "Ledger descriptive sanity comparison"},
            {"Key": "LogFiles", "Value": log_files},
            {"Key": "RawRows", "Value": raw_rows},
            {"Key": "WebLevelRows", "Value": len(web_rows)},
            {"Key": "MainTrainRatio", "Value": args.main_train_ratio},
            {"Key": "MinimumValidTrophicRuns", "Value": args.min_valid_trophic_runs},
            {"Key": "ConnectanceDefinition", "Value": "L/S^2"},
            {
                "Key": "TrophicHeightDefinition",
                "Value": "NetworkX all species in largest weakly connected component",
            },
        ],
    )
    write_readme(output / "README.md", main_rows)

    print(f"[LedgerSanity] Log files: {log_files}")
    print(f"[LedgerSanity] Raw rows: {raw_rows}")
    print(f"[LedgerSanity] Web-level rows: {len(web_rows)}")
    print(f"[LedgerSanity] TrainRatio {args.main_train_ratio} rows: {len(main_rows)}")
    print(f"[LedgerSanity] Output: {output}")


if __name__ == "__main__":
    main()
