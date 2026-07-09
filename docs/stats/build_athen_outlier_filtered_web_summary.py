#!/usr/bin/env python3
"""Build outlier-filtered web-level summaries for Athen's Fig. 5.

This script keeps the final web-level analysis unit:

    one empirical food web vs the mean of its pseudo-web realisations

but computes the mean pseudo value after flagging and removing outlying
pseudo-run metric values within each food web and metric. The original WLNM
logs are not modified.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
class MetricSpec:
    name: str
    label: str
    source_metric: str
    empirical_column: str
    pseudo_column: str
    trophic: bool = False
    requires_networkx_diagnostics: bool = False


DEFAULT_METRICS = (
    MetricSpec(
        "Connectance",
        "Connectance",
        "Connectance",
        "EmpiricalConnectance",
        "PseudoConnectance",
    ),
    MetricSpec(
        "MeanTrophicHeight",
        "Mean trophic height",
        "NetworkXMeanTrophicLevel",
        "EmpiricalNetworkXMeanTrophicLevel",
        "PseudoNetworkXMeanTrophicLevel",
        trophic=True,
        requires_networkx_diagnostics=True,
    ),
    MetricSpec(
        "MeanGenerality",
        "Mean generality",
        "MeanGenerality",
        "EmpiricalMeanGenerality",
        "PseudoMeanGenerality",
    ),
    MetricSpec(
        "MeanVulnerability",
        "Mean vulnerability",
        "MeanVulnerability",
        "EmpiricalMeanVulnerability",
        "PseudoMeanVulnerability",
    ),
)


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
        "--output-base-dir",
        type=Path,
        default=RESULTS_DIR / "statistical_tests/athen/outlier_filtered",
    )
    parser.add_argument(
        "--train-ratios",
        default="60,90",
        help="Comma-separated train ratios to process. Values are rounded to the nearest integer.",
    )
    parser.add_argument(
        "--min-retained-runs",
        type=int,
        default=10,
        help="Minimum retained pseudo-runs required after outlier filtering.",
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="Tukey fence multiplier.",
    )
    parser.add_argument(
        "--trophic-source",
        choices=("auto", "networkx", "legacy"),
        default="auto",
        help=(
            "Trophic-height columns to use. 'auto' uses NetworkX columns when "
            "present, otherwise falls back to legacy MeanTrophicLevel columns."
        ),
    )
    parser.add_argument(
        "--split-by-cvk",
        action="store_true",
        help="Keep k-fold CvK values in separate grouping/output folders.",
    )
    parser.add_argument(
        "--threshold-mode",
        default="",
        help=(
            "Optional ThresholdMode filter, for example 'threshold_sweep'. "
            "Leave empty to keep all rows."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional classification-threshold filter. Use this with "
            "threshold-sweep logs to reconstruct the fixed-threshold figure, "
            "for example --threshold 0.5."
        ),
    )
    parser.add_argument(
        "--threshold-tolerance",
        type=float,
        default=1e-9,
        help="Absolute tolerance used when filtering Threshold.",
    )
    return parser.parse_args()


def first_csv_header(logs_dir: Path) -> Sequence[str]:
    for path in sorted(logs_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            return reader.fieldnames or []
    return []


def resolve_metrics(logs_dir: Path, trophic_source: str) -> Tuple[MetricSpec, ...]:
    header = set(first_csv_header(logs_dir))
    has_networkx = {
        "EmpiricalNetworkXMeanTrophicLevel",
        "PseudoNetworkXMeanTrophicLevel",
        "EmpiricalNetworkXTrophicLevelStatusCode",
        "PseudoNetworkXTrophicLevelStatusCode",
    }.issubset(header)

    if trophic_source == "networkx" and not has_networkx:
        raise SystemExit(
            "NetworkX trophic columns were requested but are absent from the logs."
        )

    use_networkx = trophic_source == "networkx" or (
        trophic_source == "auto" and has_networkx
    )

    if use_networkx:
        return DEFAULT_METRICS

    return (
        DEFAULT_METRICS[0],
        MetricSpec(
            "MeanTrophicHeight",
            "Mean trophic height",
            "MeanTrophicLevel",
            "EmpiricalMeanTrophicLevel",
            "PseudoMeanTrophicLevel",
            trophic=True,
            requires_networkx_diagnostics=False,
        ),
        DEFAULT_METRICS[2],
        DEFAULT_METRICS[3],
    )


def percentile(values: Sequence[float], q: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return finite[int(position)]
    fraction = position - lo
    return finite[lo] * (1.0 - fraction) + finite[hi] * fraction


def tukey_fences(values: Sequence[float], multiplier: float) -> Dict[str, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {
            "Q1": math.nan,
            "Q3": math.nan,
            "IQR": math.nan,
            "LowerFence": math.nan,
            "UpperFence": math.nan,
        }
    q1 = percentile(finite, 0.25)
    q3 = percentile(finite, 0.75)
    iqr = q3 - q1
    if not math.isfinite(iqr) or iqr < 0:
        iqr = math.nan
    if not math.isfinite(iqr) or iqr == 0:
        lower = -math.inf
        upper = math.inf
    else:
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "LowerFence": lower,
        "UpperFence": upper,
    }


def mean_or_nan(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else math.nan


def metric_value(row: Dict[str, str], column: str) -> Optional[float]:
    return parse_float(row.get(column))


def trophic_valid(row: Dict[str, str], side: str) -> Tuple[bool, str]:
    status = str(row.get(f"{side}NetworkXTrophicLevelStatusCode", "")).strip()
    n_with_level = parse_int(row.get(f"{side}NetworkXTrophicLevelNumSpeciesWithLevel"))
    if status != "0":
        return False, f"{side.lower()}_status_{status or 'missing'}"
    if n_with_level is None or n_with_level < 2:
        return False, f"{side.lower()}_fewer_than_2_species_with_level"
    return True, ""


def run_is_valid_for_metric(row: Dict[str, str], spec: MetricSpec) -> Tuple[bool, str]:
    empirical = metric_value(row, spec.empirical_column)
    pseudo = metric_value(row, spec.pseudo_column)
    if empirical is None:
        return False, "missing_empirical_value"
    if pseudo is None:
        return False, "missing_pseudo_value"
    if spec.requires_networkx_diagnostics:
        empirical_ok, empirical_reason = trophic_valid(row, "Empirical")
        if not empirical_ok:
            return False, empirical_reason
        pseudo_ok, pseudo_reason = trophic_valid(row, "Pseudo")
        if not pseudo_ok:
            return False, pseudo_reason
    return True, ""


def read_run_rows(
    logs_dir: Path,
    metadata: Dict[str, str],
    train_ratios: Sequence[int],
    metrics: Sequence[MetricSpec],
    threshold_mode: str,
    threshold: Optional[float],
    threshold_tolerance: float,
) -> List[Dict[str, object]]:
    train_ratio_set = set(train_ratios)
    threshold_mode_filter = threshold_mode.strip().lower()
    rows: List[Dict[str, object]] = []
    for path in sorted(logs_dir.glob("*_results_random_wlnm_dir_neg*.csv")):
        food_web = web_from_filename(path)
        ecosystem = metadata.get(food_web, "unknown")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                ratio_raw = str(raw.get("TrainRatio", "")).strip()
                ratio = parse_int(ratio_raw)
                if ratio not in train_ratio_set:
                    continue
                row_threshold_mode = str(raw.get("ThresholdMode", "")).strip()
                if (
                    threshold_mode_filter
                    and row_threshold_mode.lower() != threshold_mode_filter
                ):
                    continue
                row_threshold = parse_float(raw.get("Threshold"))
                if threshold is not None and (
                    row_threshold is None
                    or not math.isfinite(row_threshold)
                    or abs(row_threshold - threshold) > threshold_tolerance
                ):
                    continue
                for spec in metrics:
                    empirical = metric_value(raw, spec.empirical_column)
                    pseudo = metric_value(raw, spec.pseudo_column)
                    valid, invalid_reason = run_is_valid_for_metric(raw, spec)
                    rows.append(
                        {
                            "FoodWeb": food_web,
                            "EcosystemType": ecosystem,
                            "TrainRatio": ratio,
                            "TrainRatioRaw": ratio_raw,
                            "CvK": parse_int(raw.get("CvK")),
                            "FoldID": parse_int(raw.get("FoldID")),
                            "NumFolds": parse_int(raw.get("NumFolds")),
                            "ThresholdMode": row_threshold_mode,
                            "Threshold": row_threshold if row_threshold is not None else math.nan,
                            "Metric": spec.name,
                            "MetricLabel": spec.label,
                            "SourceMetric": spec.source_metric,
                            "Iteration": parse_int(raw.get("Iteration")),
                            "Seed": parse_int(raw.get("Seed")),
                            "EmpiricalValue": empirical if empirical is not None else math.nan,
                            "PseudoValue": pseudo if pseudo is not None else math.nan,
                            "ValidBeforeFiltering": int(valid),
                            "InvalidReason": invalid_reason,
                            "EmpiricalNetworkXTrophicLevelStatusCode": raw.get(
                                "EmpiricalNetworkXTrophicLevelStatusCode", ""
                            ),
                            "PseudoNetworkXTrophicLevelStatusCode": raw.get(
                                "PseudoNetworkXTrophicLevelStatusCode", ""
                            ),
                            "PseudoNetworkXTrophicLevelNumSpeciesLargest": raw.get(
                                "PseudoNetworkXTrophicLevelNumSpeciesLargest", ""
                            ),
                            "PseudoNetworkXTrophicLevelNumSpeciesWithLevel": raw.get(
                                "PseudoNetworkXTrophicLevelNumSpeciesWithLevel", ""
                            ),
                            "PseudoNetworkXTrophicLevelRange": raw.get(
                                "PseudoNetworkXTrophicLevelRange", ""
                            ),
                        }
                    )
    return rows


def group_key(row: Dict[str, object], split_by_cvk: bool) -> Tuple[object, ...]:
    base: Tuple[object, ...] = (
        str(row["FoodWeb"]),
        int(row["TrainRatio"]),
        str(row["Metric"]),
    )
    if split_by_cvk:
        return base + (int(row["CvK"]) if row["CvK"] is not None else -1,)
    return base


def iter_groups(
    rows: Iterable[Dict[str, object]],
    split_by_cvk: bool,
) -> Iterable[List[Dict[str, object]]]:
    grouped: Dict[Tuple[str, int, str], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(group_key(row, split_by_cvk), []).append(row)
    for key in sorted(grouped):
        yield grouped[key]


def add_outlier_flags(
    run_rows: Sequence[Dict[str, object]],
    iqr_multiplier: float,
    split_by_cvk: bool,
) -> List[Dict[str, object]]:
    flagged: List[Dict[str, object]] = []
    for group in iter_groups(run_rows, split_by_cvk):
        valid_values = [
            float(row["PseudoValue"])
            for row in group
            if int(row["ValidBeforeFiltering"]) == 1
            and math.isfinite(float(row["PseudoValue"]))
        ]
        fences = tukey_fences(valid_values, iqr_multiplier)
        lower = fences["LowerFence"]
        upper = fences["UpperFence"]
        for row in group:
            pseudo = float(row["PseudoValue"])
            valid = int(row["ValidBeforeFiltering"]) == 1 and math.isfinite(pseudo)
            is_outlier = valid and (pseudo < lower or pseudo > upper)
            retained = valid and not is_outlier
            if not valid:
                reason = str(row["InvalidReason"])
            elif is_outlier:
                reason = "tukey_1_5_iqr_outlier"
            else:
                reason = ""
            flagged.append(
                {
                    **row,
                    **fences,
                    "IQRMultiplier": iqr_multiplier,
                    "IsOutlier": int(is_outlier),
                    "RetainedAfterFiltering": int(retained),
                    "FilteringReason": reason,
                }
            )
    return sorted(
        flagged,
        key=lambda row: (
            int(row["TrainRatio"]),
            int(row["CvK"]) if row["CvK"] is not None else -1,
            str(row["Metric"]),
            str(row["FoodWeb"]),
            int(row["Iteration"]) if row["Iteration"] is not None else -1,
        ),
    )


def build_web_rows(
    flagged_rows: Sequence[Dict[str, object]],
    min_retained_runs: int,
    split_by_cvk: bool,
) -> List[Dict[str, object]]:
    web_rows: List[Dict[str, object]] = []
    for group in iter_groups(flagged_rows, split_by_cvk):
        valid_before_rows = [row for row in group if int(row["ValidBeforeFiltering"]) == 1]
        retained_rows = [row for row in group if int(row["RetainedAfterFiltering"]) == 1]
        outlier_rows = [row for row in group if int(row["IsOutlier"]) == 1]

        empirical_values = [
            float(row["EmpiricalValue"])
            for row in valid_before_rows
            if math.isfinite(float(row["EmpiricalValue"]))
        ]
        pseudo_before = [
            float(row["PseudoValue"])
            for row in valid_before_rows
            if math.isfinite(float(row["PseudoValue"]))
        ]
        pseudo_after = [
            float(row["PseudoValue"])
            for row in retained_rows
            if math.isfinite(float(row["PseudoValue"]))
        ]

        first = group[0]
        empirical = mean_or_nan(empirical_values)
        before = mean_or_nan(pseudo_before)
        after = mean_or_nan(pseudo_after)

        delta_before = before - empirical if math.isfinite(before) and math.isfinite(empirical) else math.nan
        delta_after = after - empirical if math.isfinite(after) and math.isfinite(empirical) else math.nan
        rel_before = delta_before / empirical if math.isfinite(delta_before) and empirical != 0 else math.nan
        rel_after = delta_after / empirical if math.isfinite(delta_after) and empirical != 0 else math.nan
        included = math.isfinite(rel_after) and len(pseudo_after) >= min_retained_runs

        if included:
            reason = ""
        elif not math.isfinite(empirical):
            reason = "missing_empirical_value"
        elif not math.isfinite(after):
            reason = "missing_filtered_pseudo_mean"
        else:
            reason = "fewer_than_minimum_retained_runs"

        web_rows.append(
            {
                "FoodWeb": first["FoodWeb"],
                "EcosystemType": first["EcosystemType"],
                "TrainRatio": first["TrainRatio"],
                "TrainRatioRaw": first["TrainRatioRaw"],
                "CvK": first["CvK"],
                "NumFolds": first["NumFolds"],
                "Metric": first["Metric"],
                "MetricLabel": first["MetricLabel"],
                "SourceMetric": first["SourceMetric"],
                "TotalRuns": len(group),
                "ValidRunsBeforeFiltering": len(pseudo_before),
                "OutlierRunsRemoved": len(outlier_rows),
                "ValidRunsAfterFiltering": len(pseudo_after),
                "Complete20BeforeFiltering": int(len(pseudo_before) == 20),
                "Complete20AfterFiltering": int(len(pseudo_after) == 20),
                "EmpiricalValue": empirical,
                "MeanPseudoBeforeFiltering": before,
                "MeanPseudoAfterFiltering": after,
                "DeltaBeforeFiltering": delta_before,
                "DeltaAfterFiltering": delta_after,
                "RelativeErrorBeforeFiltering": rel_before,
                "RelativeErrorPercentBeforeFiltering": 100.0 * rel_before
                if math.isfinite(rel_before)
                else math.nan,
                "RelativeErrorAfterFiltering": rel_after,
                "RelativeErrorPercentAfterFiltering": 100.0 * rel_after
                if math.isfinite(rel_after)
                else math.nan,
                "IncludedAfterFiltering": int(included),
                "ExclusionReason": reason,
                "Q1": first["Q1"],
                "Q3": first["Q3"],
                "IQR": first["IQR"],
                "LowerFence": first["LowerFence"],
                "UpperFence": first["UpperFence"],
                "OutlierValues": ";".join(
                    f"{float(row['PseudoValue']):.6g}" for row in outlier_rows
                ),
                "OutlierIterations": ";".join(
                    str(row["Iteration"]) for row in outlier_rows
                ),
            }
        )
    return sorted(
        web_rows,
        key=lambda row: (
            int(row["TrainRatio"]),
            int(row["CvK"]) if row["CvK"] is not None else -1,
            str(row["Metric"]),
            str(row["FoodWeb"]),
        ),
    )


def summarise_rows(
    web_rows: Sequence[Dict[str, object]],
    scope: str,
    ecosystem: str,
    metrics: Sequence[MetricSpec],
) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    metric_by_name = {spec.name: spec for spec in metrics}
    for spec in metrics:
        selected = [
            row
            for row in web_rows
            if row["Metric"] == spec.name
            and int(row["IncludedAfterFiltering"]) == 1
            and (scope == "Overall" or row["EcosystemType"] == ecosystem)
        ]
        if not selected:
            continue

        empirical = [float(row["EmpiricalValue"]) for row in selected]
        before = [float(row["MeanPseudoBeforeFiltering"]) for row in selected]
        after = [float(row["MeanPseudoAfterFiltering"]) for row in selected]
        delta_after = [float(row["DeltaAfterFiltering"]) for row in selected]
        abs_delta_after = [abs(value) for value in delta_after]
        relative_after = [float(row["RelativeErrorAfterFiltering"]) for row in selected]
        retained = [int(row["ValidRunsAfterFiltering"]) for row in selected]

        empirical_stats = sample_stats(empirical)
        before_stats = sample_stats(before)
        after_stats = sample_stats(after)
        delta_stats = sample_stats(delta_after)
        abs_delta_stats = sample_stats(abs_delta_after)
        relative_stats = sample_stats(relative_after)
        delta_ci = confidence_interval(delta_after)
        relative_ci = confidence_interval(relative_after)

        result.append(
            {
                "Scope": scope,
                "EcosystemType": ecosystem,
                "TrainRatio": selected[0]["TrainRatio"],
                "TrainRatioRaw": selected[0]["TrainRatioRaw"],
                "CvK": selected[0]["CvK"],
                "NumFolds": selected[0]["NumFolds"],
                "Metric": spec.name,
                "MetricLabel": metric_by_name[spec.name].label,
                "SourceMetric": spec.source_metric,
                "NumFoodWebs": len(selected),
                "NumComplete20AfterFiltering": sum(
                    int(row["Complete20AfterFiltering"]) for row in selected
                ),
                "TotalOutlierRunsRemoved": sum(
                    int(row["OutlierRunsRemoved"]) for row in selected
                ),
                "MinRetainedRuns": min(retained),
                "MedianRetainedRuns": statistics.median(retained),
                "MeanEmpirical": empirical_stats["Mean"],
                "MeanPseudoBeforeFiltering": before_stats["Mean"],
                "MeanPseudoAfterFiltering": after_stats["Mean"],
                "MeanDeltaAfterFiltering": delta_stats["Mean"],
                "AbsoluteMeanDeltaAfterFiltering": abs(delta_stats["Mean"]),
                "MeanAbsoluteDeltaAfterFiltering": abs_delta_stats["Mean"],
                "DeltaCI90LowerAfterFiltering": delta_ci[0],
                "DeltaCI90UpperAfterFiltering": delta_ci[1],
                "MeanRelativeErrorAfterFiltering": relative_stats["Mean"],
                "MeanRelativeErrorPercentAfterFiltering": 100.0
                * relative_stats["Mean"],
                "RelativeCI90LowerPercentAfterFiltering": 100.0 * relative_ci[0],
                "RelativeCI90UpperPercentAfterFiltering": 100.0 * relative_ci[1],
                "RelativeError90CIAfterFiltering": (
                    f"{100.0 * relative_stats['Mean']:+.2f}% "
                    f"[{100.0 * relative_ci[0]:+.2f}%, "
                    f"{100.0 * relative_ci[1]:+.2f}%]"
                ),
            }
        )
    return result


def build_summaries(
    web_rows: Sequence[Dict[str, object]],
    metrics: Sequence[MetricSpec],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    ecosystems = sorted(
        {
            str(row["EcosystemType"])
            for row in web_rows
            if str(row["EcosystemType"]) != "unknown"
        }
    )
    overall = summarise_rows(web_rows, "Overall", "all", metrics)
    by_ecosystem: List[Dict[str, object]] = []
    for ecosystem in ecosystems:
        by_ecosystem.extend(summarise_rows(web_rows, "Ecosystem", ecosystem, metrics))
    return overall, by_ecosystem


def write_readme(
    path: Path,
    train_ratio: int,
    overall: Sequence[Dict[str, object]],
    min_retained_runs: int,
    iqr_multiplier: float,
    trophic_source: str,
    cvk: Optional[int],
    threshold_mode: str,
    threshold: Optional[float],
) -> None:
    cvk_line = f"CvK: {cvk}" if cvk is not None else "CvK: not used as grouping variable"
    threshold_mode_line = (
        f"ThresholdMode filter: {threshold_mode}."
        if threshold_mode
        else "ThresholdMode filter: not applied."
    )
    threshold_line = (
        f"Classification-threshold filter: {threshold:g}."
        if threshold is not None
        else "Classification-threshold filter: not applied."
    )
    lines = [
        f"# Outlier-filtered web-level reconstruction summary, train ratio {train_ratio}",
        "",
        "This output is derived from the original WLNM logs; the original logs are not modified.",
        "One analysis unit is one empirical food web compared with the mean of retained pseudo-web realisations.",
        cvk_line,
        threshold_mode_line,
        threshold_line,
        f"Trophic-height source metric: {trophic_source}.",
        f"Outliers are pseudo-run metric values outside Tukey fences: Q1 - {iqr_multiplier} * IQR and Q3 + {iqr_multiplier} * IQR.",
        "Fences are calculated within each food web and metric.",
        f"Food-web metrics are included only when at least {min_retained_runs} pseudo-runs remain after filtering.",
        "",
        "## Overall summary",
        "",
        "| Metric | N webs | Outlier runs removed | Mean empirical | Mean pseudo after filtering | Mean delta | Relative error (90% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            "| {MetricLabel} | {NumFoodWebs} | {TotalOutlierRunsRemoved} | "
            "{MeanEmpirical:.4f} | {MeanPseudoAfterFiltering:.4f} | "
            "{MeanDeltaAfterFiltering:+.4f} | {RelativeError90CIAfterFiltering} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs_for_ratio(
    output_base: Path,
    train_ratio: int,
    flagged_rows: Sequence[Dict[str, object]],
    web_rows: Sequence[Dict[str, object]],
    min_retained_runs: int,
    iqr_multiplier: float,
    metrics: Sequence[MetricSpec],
    trophic_source: str,
    cvk: Optional[int] = None,
    threshold_mode: str = "",
    threshold: Optional[float] = None,
) -> None:
    out_dir = output_base / f"train_ratio_{train_ratio}"
    ratio_flags = [
        row
        for row in flagged_rows
        if int(row["TrainRatio"]) == train_ratio
        and (cvk is None or int(row["CvK"]) == cvk)
    ]
    ratio_web = [
        row
        for row in web_rows
        if int(row["TrainRatio"]) == train_ratio
        and (cvk is None or int(row["CvK"]) == cvk)
    ]
    overall, by_ecosystem = build_summaries(ratio_web, metrics)
    write_csv(
        out_dir / f"athen_run_level_outlier_flags_train{train_ratio}.csv",
        ratio_flags,
    )
    write_csv(
        out_dir / f"athen_web_level_metrics_train{train_ratio}_outlier_filtered.csv",
        ratio_web,
    )
    write_csv(
        out_dir / f"athen_metric_summary_train{train_ratio}_outlier_filtered_overall.csv",
        overall,
    )
    write_csv(
        out_dir / f"athen_metric_summary_train{train_ratio}_outlier_filtered_by_ecosystem.csv",
        by_ecosystem,
    )
    write_readme(
        out_dir / "README.md",
        train_ratio,
        overall,
        min_retained_runs,
        iqr_multiplier,
        trophic_source,
        cvk,
        threshold_mode,
        threshold,
    )


def main() -> None:
    args = parse_args()
    train_ratios = [
        int(value.strip())
        for value in args.train_ratios.split(",")
        if value.strip()
    ]
    metrics = resolve_metrics(args.logs_dir, args.trophic_source)
    trophic_source = next(spec.source_metric for spec in metrics if spec.trophic)
    metadata = read_metadata(args.metadata_file)
    run_rows = read_run_rows(
        args.logs_dir,
        metadata,
        train_ratios,
        metrics,
        args.threshold_mode,
        args.threshold,
        args.threshold_tolerance,
    )
    flagged_rows = add_outlier_flags(
        run_rows,
        args.iqr_multiplier,
        args.split_by_cvk,
    )
    web_rows = build_web_rows(
        flagged_rows,
        args.min_retained_runs,
        args.split_by_cvk,
    )
    if args.split_by_cvk:
        cvks = sorted(
            {
                int(row["CvK"])
                for row in web_rows
                if row["CvK"] is not None and int(row["CvK"]) > 0
            }
        )
        for cvk in cvks:
            cvk_base = args.output_base_dir / f"cvK_{cvk}"
            ratio_subset = sorted(
                {
                    int(row["TrainRatio"])
                    for row in web_rows
                    if row["CvK"] is not None and int(row["CvK"]) == cvk
                }
            )
            for ratio in ratio_subset:
                write_outputs_for_ratio(
                    cvk_base,
                    ratio,
                    flagged_rows,
                    web_rows,
                    args.min_retained_runs,
                    args.iqr_multiplier,
                    metrics,
                    trophic_source,
                    cvk=cvk,
                    threshold_mode=args.threshold_mode,
                    threshold=args.threshold,
                )
    else:
        for ratio in train_ratios:
            write_outputs_for_ratio(
                args.output_base_dir,
                ratio,
                flagged_rows,
                web_rows,
                args.min_retained_runs,
                args.iqr_multiplier,
                metrics,
                trophic_source,
                threshold_mode=args.threshold_mode,
                threshold=args.threshold,
            )

    print(f"[OutlierFilter] Ratios: {train_ratios}")
    print(f"[OutlierFilter] Trophic source: {trophic_source}")
    print(f"[OutlierFilter] Split by CvK: {args.split_by_cvk}")
    print(f"[OutlierFilter] ThresholdMode filter: {args.threshold_mode or 'not applied'}")
    print(f"[OutlierFilter] Threshold filter: {args.threshold if args.threshold is not None else 'not applied'}")
    print(f"[OutlierFilter] Run-level metric rows: {len(run_rows)}")
    print(f"[OutlierFilter] Web-level metric rows: {len(web_rows)}")
    print(f"[OutlierFilter] Output base: {args.output_base_dir}")


if __name__ == "__main__":
    main()
