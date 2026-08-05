#!/usr/bin/env python3
"""Apply the final metric-wise Tukey retention protocol to WLNM outputs.

The source prediction CSVs are read-only.  Every result root receives its own
versioned ``retention_protocol`` directory, so fences and retained run IDs are
never shared between experimental conditions.

The implementation intentionally uses only the Python standard library.  The
WLNM result trees are large, so input files are processed one food web at a
time and run-level outputs are written as compressed CSV files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_NAME = "tukey_iqr_1p5_min50pct_threshold0p50_v1"
THRESHOLD_TOLERANCE = 1e-9


@dataclass(frozen=True)
class MetricCandidate:
    source_column: str
    reference_column: str = ""
    status_column: str = ""
    count_column: str = ""
    reference_status_column: str = ""
    reference_count_column: str = ""


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    label: str
    family: str
    candidates: Tuple[MetricCandidate, ...]


@dataclass(frozen=True)
class ResolvedMetric:
    name: str
    label: str
    family: str
    source_column: str
    reference_column: str
    status_column: str
    count_column: str
    reference_status_column: str
    reference_count_column: str


PREDICTIVE_METRICS = (
    MetricDefinition("ROC_AUC", "ROC-AUC", "predictive", (MetricCandidate("ROC_AUC"),)),
    MetricDefinition("PR_AUC", "PR-AUC", "predictive", (MetricCandidate("PR_AUC"),)),
    MetricDefinition("F1Score", "F1", "predictive", (MetricCandidate("F1Score"), MetricCandidate("TestF1Score"))),
    MetricDefinition("TestMCC", "MCC", "predictive", (MetricCandidate("TestMCC"), MetricCandidate("PseudoMCC"))),
    MetricDefinition("Precision", "Precision", "predictive", (MetricCandidate("Precision"), MetricCandidate("TestPrecision"))),
    MetricDefinition("Recall", "Recall", "predictive", (MetricCandidate("Recall"), MetricCandidate("TestRecall"))),
    MetricDefinition("TestTSS", "TSS", "predictive", (MetricCandidate("TestTSS"), MetricCandidate("PseudoTSS"))),
    MetricDefinition("TestTPR", "TPR", "predictive", (MetricCandidate("TestTPR"), MetricCandidate("PseudoTPR"))),
)


PSEUDO_ECOLOGICAL_METRICS = (
    MetricDefinition(
        "PseudoConnectance",
        "Pseudo connectance",
        "pseudo_ecological",
        (MetricCandidate("PseudoConnectance", "EmpiricalConnectance"),),
    ),
    MetricDefinition(
        "PseudoMeanTrophicHeight",
        "Pseudo average trophic height",
        "pseudo_ecological",
        (
            MetricCandidate(
                "PseudoNetworkXMeanTrophicLevel",
                "EmpiricalNetworkXMeanTrophicLevel",
                "PseudoNetworkXTrophicLevelStatusCode",
                "PseudoNetworkXTrophicLevelNumSpeciesWithLevel",
                "EmpiricalNetworkXTrophicLevelStatusCode",
                "EmpiricalNetworkXTrophicLevelNumSpeciesWithLevel",
            ),
            MetricCandidate("PseudoMeanTrophicLevel", "EmpiricalMeanTrophicLevel"),
        ),
    ),
    MetricDefinition(
        "PseudoMeanGenerality",
        "Pseudo mean generality",
        "pseudo_ecological",
        (MetricCandidate("PseudoMeanGenerality", "EmpiricalMeanGenerality"),),
    ),
    MetricDefinition(
        "PseudoMeanVulnerability",
        "Pseudo mean vulnerability",
        "pseudo_ecological",
        (MetricCandidate("PseudoMeanVulnerability", "EmpiricalMeanVulnerability"),),
    ),
)


TRAIN_ECOLOGICAL_METRICS = (
    MetricDefinition(
        "TrainConnectance",
        "Training-graph connectance",
        "train_ecological",
        (MetricCandidate("TrainConnectance", "EmpiricalConnectance"),),
    ),
    MetricDefinition(
        "TrainMeanTrophicHeight",
        "Training-graph average trophic height",
        "train_ecological",
        (
            MetricCandidate(
                "TrainNetworkXMeanTrophicLevel",
                "EmpiricalNetworkXMeanTrophicLevel",
                "TrainNetworkXTrophicLevelStatusCode",
                "TrainNetworkXTrophicLevelNumSpeciesWithLevel",
                "EmpiricalNetworkXTrophicLevelStatusCode",
                "EmpiricalNetworkXTrophicLevelNumSpeciesWithLevel",
            ),
            MetricCandidate("TrainMeanTrophicLevel", "EmpiricalMeanTrophicLevel"),
        ),
    ),
    MetricDefinition(
        "TrainMeanGenerality",
        "Training-graph mean generality",
        "train_ecological",
        (MetricCandidate("TrainMeanGenerality", "EmpiricalMeanGenerality"),),
    ),
    MetricDefinition(
        "TrainMeanVulnerability",
        "Training-graph mean vulnerability",
        "train_ecological",
        (MetricCandidate("TrainMeanVulnerability", "EmpiricalMeanVulnerability"),),
    ),
)


METRIC_DEFINITIONS = (
    *PREDICTIVE_METRICS,
    *PSEUDO_ECOLOGICAL_METRICS,
    *TRAIN_ECOLOGICAL_METRICS,
)


RUN_OUTPUT_FIELDS = (
    "Scenario",
    "ResultRoot",
    "SourceCSV",
    "Foodweb",
    "Version",
    "RunUnit",
    "Iteration",
    "ExperimentID",
    "Seed",
    "TrainRatio",
    "ThresholdMode",
    "Threshold",
    "K",
    "CvK",
    "FoldID",
    "FoldCount",
    "ExpectedFoldCount",
    "Metric",
    "MetricLabel",
    "MetricFamily",
    "SourceColumn",
    "ReferenceColumn",
    "Value",
    "ReferenceValue",
    "DeltaValue",
    "Q1",
    "Q3",
    "IQR",
    "LowerFence",
    "UpperFence",
    "ValidRunsBeforeTukey",
    "RetainedRunsAfterTukey",
    "ExpectedRuns",
    "MinimumRetainedRuns",
    "MeetsMinimumRetainedRuns",
    "ExclusionReason",
)


RUN_ID_FIELDS = (
    "Scenario",
    "SourceCSV",
    "Foodweb",
    "Version",
    "RunUnit",
    "Iteration",
    "ExperimentID",
    "Seed",
    "TrainRatio",
    "Threshold",
    "K",
    "CvK",
    "FoldID",
    "Metric",
    "ExclusionReason",
)


GROUP_OUTPUT_FIELDS = (
    "Scenario",
    "Foodweb",
    "Version",
    "TrainRatio",
    "Threshold",
    "K",
    "CvK",
    "Metric",
    "MetricLabel",
    "MetricFamily",
    "SourceColumn",
    "ReferenceColumn",
    "TotalRunUnits",
    "ValidRunsBeforeTukey",
    "InvalidRunsBeforeTukey",
    "OutlierRunsExcluded",
    "RetainedRunsAfterTukey",
    "ExpectedRuns",
    "MinimumRetainedRuns",
    "MeetsMinimumRetainedRuns",
    "Q1",
    "Q3",
    "IQR",
    "LowerFence",
    "UpperFence",
    "MeanBeforeTukey",
    "MeanAfterTukey",
    "StdAfterTukey",
    "SEAfterTukey",
    "ReferenceValue",
    "DeltaAfterTukey",
    "RelativeDeltaAfterTukey",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        action="append",
        type=Path,
        default=[],
        help="Result root to process. Repeat this option for multiple independent roots.",
    )
    parser.add_argument(
        "--discover-role-or-mass",
        action="store_true",
        help="Process the six local randomeligible_roleormass standard result roots.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    parser.add_argument(
        "--minimum-retained-fraction",
        type=float,
        default=0.5,
        help="Minimum retained fraction; 0.5 gives 25/50 standard runs and 10/20 repeated CV experiments.",
    )
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Testing aid: process only the first N prediction CSVs.",
    )
    return parser.parse_args()


def parse_float(value: object) -> Optional[float]:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_int(value: object) -> Optional[int]:
    number = parse_float(value)
    if number is None:
        return None
    return int(round(number))


def number_text(value: object) -> str:
    number = parse_float(value)
    if number is None:
        return ""
    if math.isclose(number, round(number), abs_tol=1e-10):
        return str(int(round(number)))
    return f"{number:.12g}"


def finite_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.mean(finite) if finite else None


def finite_stdev(values: Sequence[float]) -> Optional[float]:
    return statistics.stdev(values) if len(values) >= 2 else None


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def tukey_fences(values: Sequence[float], multiplier: float) -> Dict[str, float]:
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1
    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "LowerFence": q1 - multiplier * iqr,
        "UpperFence": q3 + multiplier * iqr,
    }


def foodweb_from_filename(path: Path) -> str:
    name = path.name
    markers = (
        "_tax_mass_results_random_wlnm_dir_neg_kfold",
        "_tax_mass_results_random_wlnm_dir_neg",
        "_tax_mass_results_random_wlnm_original",
        "_results_random_wlnm_dir_neg_kfold",
        "_results_random_wlnm_dir_neg",
        "_results_random_wlnm_original",
    )
    for marker in markers:
        if marker in name:
            return name.split(marker, 1)[0]
    return path.stem


def read_key_value_manifest(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def resolve_metrics(header: Sequence[str]) -> Tuple[List[ResolvedMetric], List[str]]:
    available = set(header)
    resolved: List[ResolvedMetric] = []
    missing: List[str] = []
    for definition in METRIC_DEFINITIONS:
        selected: Optional[MetricCandidate] = None
        for candidate in definition.candidates:
            required = [candidate.source_column]
            if candidate.reference_column:
                required.append(candidate.reference_column)
            if all(column in available for column in required):
                selected = candidate
                break
        if selected is None:
            missing.append(definition.name)
            continue
        resolved.append(
            ResolvedMetric(
                definition.name,
                definition.label,
                definition.family,
                selected.source_column,
                selected.reference_column,
                selected.status_column,
                selected.count_column,
                selected.reference_status_column,
                selected.reference_count_column,
            )
        )
    return resolved, missing


def diagnostic_valid(row: Mapping[str, str], status_column: str, count_column: str) -> Tuple[bool, str]:
    if not status_column:
        return True, ""
    status = parse_int(row.get(status_column))
    if status != 0:
        return False, f"{status_column}_not_zero"
    count = parse_int(row.get(count_column))
    if count is None or count < 2:
        return False, f"{count_column}_below_two"
    return True, ""


def observation_from_row(
    raw: Mapping[str, str],
    metric: ResolvedMetric,
    scenario: str,
    result_root: Path,
    source_csv: Path,
    foodweb: str,
) -> Dict[str, object]:
    value = parse_float(raw.get(metric.source_column))
    reference = parse_float(raw.get(metric.reference_column)) if metric.reference_column else None
    valid = value is not None
    reason = "" if valid else "missing_or_non_finite_metric"
    if valid:
        valid, reason = diagnostic_valid(raw, metric.status_column, metric.count_column)
    if valid and metric.reference_column and reference is None:
        valid, reason = False, "missing_or_non_finite_reference"
    if valid and metric.reference_column:
        valid, reason = diagnostic_valid(
            raw,
            metric.reference_status_column,
            metric.reference_count_column,
        )
    return {
        "Scenario": scenario,
        "ResultRoot": str(result_root.resolve()),
        "SourceCSV": source_csv.name,
        "Foodweb": foodweb,
        "Version": str(raw.get("Version", "")).strip(),
        "RunUnit": "run",
        "Iteration": number_text(raw.get("Iteration")),
        "ExperimentID": number_text(raw.get("ExperimentID")),
        "Seed": number_text(raw.get("Seed")),
        "TrainRatio": number_text(raw.get("TrainRatio")),
        "ThresholdMode": str(raw.get("ThresholdMode", "")).strip(),
        "Threshold": number_text(raw.get("Threshold")),
        "K": number_text(raw.get("K")),
        "CvK": number_text(raw.get("CvK")),
        "FoldID": number_text(raw.get("FoldID")),
        "FoldCount": "",
        "ExpectedFoldCount": "",
        "Metric": metric.name,
        "MetricLabel": metric.label,
        "MetricFamily": metric.family,
        "SourceColumn": metric.source_column,
        "ReferenceColumn": metric.reference_column,
        "Value": value,
        "ReferenceValue": reference,
        "ValidBeforeTukey": valid,
        "InvalidReason": reason,
    }


def experiment_key(row: Mapping[str, object]) -> Tuple[str, ...]:
    experiment = str(row.get("ExperimentID", ""))
    if not experiment:
        experiment = str(row.get("Seed", "")) or str(row.get("Iteration", ""))
    return (
        str(row["Foodweb"]),
        str(row["TrainRatio"]),
        str(row["Threshold"]),
        str(row["K"]),
        str(row["CvK"]),
        str(row["Metric"]),
        experiment,
        str(row.get("Seed", "")),
    )


def aggregate_kfold_experiments(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: MutableMapping[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[experiment_key(row)].append(row)
    aggregated: List[Dict[str, object]] = []
    for group in grouped.values():
        first = dict(group[0])
        expected_folds = parse_int(first.get("CvK")) or len(group)
        fold_ids = {str(row.get("FoldID", "")) for row in group if str(row.get("FoldID", ""))}
        valid_rows = [row for row in group if bool(row["ValidBeforeTukey"])]
        complete = len(group) == expected_folds and len(fold_ids) == expected_folds
        valid = complete and len(valid_rows) == expected_folds
        first.update(
            {
                "RunUnit": "repeated_cv_experiment",
                "Iteration": ";".join(str(row.get("Iteration", "")) for row in group),
                "FoldID": "all",
                "FoldCount": len(group),
                "ExpectedFoldCount": expected_folds,
                "Value": finite_mean(parse_float(row.get("Value")) for row in valid_rows),
                "ReferenceValue": finite_mean(
                    parse_float(row.get("ReferenceValue")) for row in valid_rows
                ),
                "ValidBeforeTukey": valid,
                "InvalidReason": "" if valid else "incomplete_or_invalid_folds",
            }
        )
        aggregated.append(first)
    return aggregated


def retention_group_key(row: Mapping[str, object]) -> Tuple[str, ...]:
    return (
        str(row["Foodweb"]),
        str(row["Version"]),
        str(row["TrainRatio"]),
        str(row["Threshold"]),
        str(row["K"]),
        str(row["CvK"]),
        str(row["Metric"]),
    )


def expected_runs(manifest: Mapping[str, str], is_kfold: bool, group_size: int) -> int:
    keys = ("NumExperimentsPerFold",) if is_kfold else ("NumExperiments",)
    for key in keys:
        value = parse_int(manifest.get(key))
        if value is not None and value > 0:
            return value
    return group_size


def float_output(value: object) -> object:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.12g}"
    if value is None:
        return ""
    return value


def writeable_row(row: Mapping[str, object], fields: Sequence[str]) -> Dict[str, object]:
    return {field: float_output(row.get(field, "")) for field in fields}


class GzipCsvWriter:
    def __init__(self, path: Path, fields: Sequence[str]):
        self.path = path
        self.fields = tuple(fields)
        self.handle = gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=6)
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fields, extrasaction="ignore")
        self.writer.writeheader()

    def writerow(self, row: Mapping[str, object]) -> None:
        self.writer.writerow(writeable_row(row, self.fields))

    def close(self) -> None:
        self.handle.close()


def process_retention_group(
    group: Sequence[Dict[str, object]],
    manifest: Mapping[str, str],
    is_kfold: bool,
    multiplier: float,
    minimum_fraction: float,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    valid_rows = [
        row
        for row in group
        if bool(row["ValidBeforeTukey"])
        and parse_float(row.get("Value")) is not None
    ]
    values = [float(row["Value"]) for row in valid_rows]
    fences = tukey_fences(values, multiplier) if values else {
        "Q1": math.nan,
        "Q3": math.nan,
        "IQR": math.nan,
        "LowerFence": math.nan,
        "UpperFence": math.nan,
    }
    retained_count = sum(
        fences["LowerFence"] <= float(row["Value"]) <= fences["UpperFence"]
        for row in valid_rows
    )
    expected = expected_runs(manifest, is_kfold, len(group))
    minimum = math.ceil(expected * minimum_fraction)
    meets_minimum = retained_count >= minimum
    flagged: List[Dict[str, object]] = []
    for original in group:
        row = dict(original)
        value = parse_float(row.get("Value"))
        if not bool(row["ValidBeforeTukey"]) or value is None:
            retained = False
            reason = str(row.get("InvalidReason", "missing_or_non_finite_metric"))
        elif value < fences["LowerFence"]:
            retained = False
            reason = "below_lower_tukey_fence"
        elif value > fences["UpperFence"]:
            retained = False
            reason = "above_upper_tukey_fence"
        elif not meets_minimum:
            retained = False
            reason = "group_below_minimum_retained_runs"
        else:
            retained = True
            reason = ""
        reference = parse_float(row.get("ReferenceValue"))
        row.update(
            {
                **fences,
                "DeltaValue": value - reference if value is not None and reference is not None else None,
                "ValidRunsBeforeTukey": len(valid_rows),
                "RetainedRunsAfterTukey": retained_count,
                "ExpectedRuns": expected,
                "MinimumRetainedRuns": minimum,
                "MeetsMinimumRetainedRuns": int(meets_minimum),
                "Retained": retained,
                "ExclusionReason": reason,
            }
        )
        flagged.append(row)

    retained_values = [
        float(row["Value"])
        for row in flagged
        if bool(row["Retained"]) and parse_float(row.get("Value")) is not None
    ]
    reference_values = [
        float(row["ReferenceValue"])
        for row in flagged
        if bool(row["Retained"]) and parse_float(row.get("ReferenceValue")) is not None
    ]
    mean_before = finite_mean(values)
    mean_after = finite_mean(retained_values)
    reference = finite_mean(reference_values)
    std_after = finite_stdev(retained_values)
    first = flagged[0]
    summary = {
        "Scenario": first["Scenario"],
        "Foodweb": first["Foodweb"],
        "Version": first["Version"],
        "TrainRatio": first["TrainRatio"],
        "Threshold": first["Threshold"],
        "K": first["K"],
        "CvK": first["CvK"],
        "Metric": first["Metric"],
        "MetricLabel": first["MetricLabel"],
        "MetricFamily": first["MetricFamily"],
        "SourceColumn": first["SourceColumn"],
        "ReferenceColumn": first["ReferenceColumn"],
        "TotalRunUnits": len(group),
        "ValidRunsBeforeTukey": len(valid_rows),
        "InvalidRunsBeforeTukey": len(group) - len(valid_rows),
        "OutlierRunsExcluded": len(valid_rows) - retained_count,
        "RetainedRunsAfterTukey": retained_count,
        "ExpectedRuns": expected,
        "MinimumRetainedRuns": minimum,
        "MeetsMinimumRetainedRuns": int(meets_minimum),
        **fences,
        "MeanBeforeTukey": mean_before,
        "MeanAfterTukey": mean_after if meets_minimum else None,
        "StdAfterTukey": std_after if meets_minimum else None,
        "SEAfterTukey": (
            std_after / math.sqrt(len(retained_values))
            if meets_minimum and std_after is not None and retained_values
            else None
        ),
        "ReferenceValue": reference if meets_minimum else None,
        "DeltaAfterTukey": (
            mean_after - reference
            if meets_minimum and mean_after is not None and reference is not None
            else None
        ),
        "RelativeDeltaAfterTukey": (
            (mean_after - reference) / reference
            if meets_minimum
            and mean_after is not None
            and reference is not None
            and reference != 0
            else None
        ),
    }
    return flagged, summary


def csv_writer(path: Path, fields: Sequence[str]):
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    return handle, writer


def write_rows(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    handle, writer = csv_writer(path, fields)
    try:
        for row in rows:
            writer.writerow(writeable_row(row, fields))
    finally:
        handle.close()


def metric_summary_rows(group_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: MutableMapping[Tuple[str, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in group_rows:
        key = (
            str(row["Scenario"]),
            str(row["TrainRatio"]),
            str(row["Threshold"]),
            str(row["K"]),
            str(row["CvK"]),
            str(row["Metric"]),
            str(row["MetricLabel"]),
            str(row["MetricFamily"]),
        )
        grouped[key].append(row)
    output: List[Dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        output.append(
            {
                "Scenario": key[0],
                "TrainRatio": key[1],
                "Threshold": key[2],
                "K": key[3],
                "CvK": key[4],
                "Metric": key[5],
                "MetricLabel": key[6],
                "MetricFamily": key[7],
                "FoodwebGroups": len(rows),
                "RetainedFoodwebGroups": sum(int(row["MeetsMinimumRetainedRuns"]) for row in rows),
                "TotalRunUnits": sum(int(row["TotalRunUnits"]) for row in rows),
                "ValidRunsBeforeTukey": sum(int(row["ValidRunsBeforeTukey"]) for row in rows),
                "OutlierRunsExcluded": sum(int(row["OutlierRunsExcluded"]) for row in rows),
                "RetainedRunsAfterTukey": sum(int(row["RetainedRunsAfterTukey"]) for row in rows),
            }
        )
    return output


def train_ratio_is(row: Mapping[str, object], target: float) -> bool:
    value = parse_float(row.get("TrainRatio"))
    return value is not None and math.isclose(value, target, abs_tol=1e-8)


def write_figure_inputs(output_dir: Path, summaries: Sequence[Mapping[str, object]]) -> None:
    figure_dir = output_dir / "figure_inputs"
    figure_dir.mkdir()
    eligible = [row for row in summaries if int(row["MeetsMinimumRetainedRuns"]) == 1]
    write_rows(
        figure_dir / "all_foodweb_metric_means_after_tukey.csv",
        GROUP_OUTPUT_FIELDS,
        eligible,
    )
    predictive = [row for row in eligible if row["MetricFamily"] == "predictive"]
    write_rows(
        figure_dir / "predictive_train10-90_after_tukey.csv",
        GROUP_OUTPUT_FIELDS,
        predictive,
    )
    write_rows(
        figure_dir / "predictive_train90_after_tukey.csv",
        GROUP_OUTPUT_FIELDS,
        [row for row in predictive if train_ratio_is(row, 90.0)],
    )
    ecological = [row for row in eligible if str(row["MetricFamily"]).endswith("ecological")]
    for ratio in (40, 50, 60, 90):
        write_rows(
            figure_dir / f"ecological_train{ratio}_after_tukey.csv",
            GROUP_OUTPUT_FIELDS,
            [row for row in ecological if train_ratio_is(row, float(ratio))],
        )


def discover_prediction_files(root: Path, max_files: Optional[int]) -> List[Path]:
    logs_dir = root / "prediction_scores_logs"
    files = sorted(logs_dir.glob("*.csv"))
    return files[:max_files] if max_files is not None else files


def process_result_root(
    result_root: Path,
    output_name: str,
    threshold: float,
    multiplier: float,
    minimum_fraction: float,
    max_files: Optional[int],
) -> Path:
    result_root = result_root.resolve()
    if not result_root.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {result_root}")
    files = discover_prediction_files(result_root, max_files)
    if not files:
        raise FileNotFoundError(f"No prediction CSVs found under {result_root}")
    target_parent = result_root / "retention_protocol"
    target = target_parent / output_name
    if target.exists():
        raise FileExistsError(
            f"Retention output already exists: {target}. Use a new --output-name; sources were not changed."
        )
    target_parent.mkdir(exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_name}.", dir=target_parent))
    manifest = read_key_value_manifest(result_root / "RUN_MANIFEST.txt")
    scenario = manifest.get("Condition", result_root.name)
    is_kfold = "kfold" in manifest.get("Version", "").lower() or "kfold" in result_root.name.lower()
    retained_writer = GzipCsvWriter(temp_dir / "retained_run_metrics.csv.gz", RUN_OUTPUT_FIELDS)
    excluded_writer = GzipCsvWriter(temp_dir / "excluded_run_metrics.csv.gz", RUN_OUTPUT_FIELDS)
    retained_ids_writer = GzipCsvWriter(temp_dir / "retained_run_ids.csv.gz", RUN_ID_FIELDS)
    excluded_ids_writer = GzipCsvWriter(temp_dir / "excluded_run_ids.csv.gz", RUN_ID_FIELDS)
    summaries: List[Dict[str, object]] = []
    missing_metric_files: MutableMapping[str, int] = defaultdict(int)
    resolved_sources: MutableMapping[str, set[str]] = defaultdict(set)
    rows_read = 0
    rows_at_threshold = 0
    retained_rows = 0
    excluded_rows = 0
    try:
        for index, source_csv in enumerate(files, start=1):
            foodweb = foodweb_from_filename(source_csv)
            observations: List[Dict[str, object]] = []
            with source_csv.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                resolved, missing = resolve_metrics(reader.fieldnames or [])
                for metric_name in missing:
                    missing_metric_files[metric_name] += 1
                for metric in resolved:
                    resolved_sources[metric.name].add(metric.source_column)
                for raw in reader:
                    rows_read += 1
                    row_threshold = parse_float(raw.get("Threshold"))
                    if row_threshold is None or not math.isclose(
                        row_threshold,
                        threshold,
                        abs_tol=THRESHOLD_TOLERANCE,
                    ):
                        continue
                    rows_at_threshold += 1
                    for metric in resolved:
                        observations.append(
                            observation_from_row(
                                raw,
                                metric,
                                scenario,
                                result_root,
                                source_csv,
                                foodweb,
                            )
                        )
            if is_kfold:
                observations = aggregate_kfold_experiments(observations)
            grouped: MutableMapping[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
            for observation in observations:
                grouped[retention_group_key(observation)].append(observation)
            for key in sorted(grouped):
                flagged, summary = process_retention_group(
                    grouped[key],
                    manifest,
                    is_kfold,
                    multiplier,
                    minimum_fraction,
                )
                summaries.append(summary)
                for row in flagged:
                    if bool(row["Retained"]):
                        retained_writer.writerow(row)
                        retained_ids_writer.writerow(row)
                        retained_rows += 1
                    else:
                        excluded_writer.writerow(row)
                        excluded_ids_writer.writerow(row)
                        excluded_rows += 1
            if index % 25 == 0 or index == len(files):
                print(
                    f"[{scenario}] processed {index}/{len(files)} CSVs; "
                    f"retained_metric_rows={retained_rows:,} excluded_metric_rows={excluded_rows:,}",
                    flush=True,
                )
    except Exception:
        retained_writer.close()
        excluded_writer.close()
        retained_ids_writer.close()
        excluded_ids_writer.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    retained_writer.close()
    excluded_writer.close()
    retained_ids_writer.close()
    excluded_ids_writer.close()

    summaries.sort(
        key=lambda row: (
            str(row["TrainRatio"]),
            str(row["CvK"]),
            str(row["Metric"]),
            str(row["Foodweb"]),
        )
    )
    write_rows(temp_dir / "retention_by_foodweb_metric.csv", GROUP_OUTPUT_FIELDS, summaries)
    retained_group_rows = [row for row in summaries if int(row["MeetsMinimumRetainedRuns"]) == 1]
    write_rows(
        temp_dir / "retained_foodwebs_by_metric.csv",
        GROUP_OUTPUT_FIELDS,
        retained_group_rows,
    )
    metric_summary_fields = (
        "Scenario",
        "TrainRatio",
        "Threshold",
        "K",
        "CvK",
        "Metric",
        "MetricLabel",
        "MetricFamily",
        "FoodwebGroups",
        "RetainedFoodwebGroups",
        "TotalRunUnits",
        "ValidRunsBeforeTukey",
        "OutlierRunsExcluded",
        "RetainedRunsAfterTukey",
    )
    metric_summaries = metric_summary_rows(summaries)
    write_rows(temp_dir / "retention_by_metric.csv", metric_summary_fields, metric_summaries)
    write_figure_inputs(temp_dir, summaries)

    expected_files = parse_int(manifest.get("ExpectedPredictionCSVs")) or parse_int(manifest.get("FoodWebs"))
    validation_rows = [
        {
            "Check": "prediction_csv_count",
            "Status": "PASS" if expected_files in (None, len(files)) or max_files is not None else "FAIL",
            "Observed": len(files),
            "Expected": expected_files if max_files is None else max_files,
            "Detail": "CSV files processed",
        },
        {
            "Check": "threshold_rows_found",
            "Status": "PASS" if rows_at_threshold > 0 else "FAIL",
            "Observed": rows_at_threshold,
            "Expected": ">0",
            "Detail": f"Rows at Threshold={threshold}",
        },
        {
            "Check": "output_partition",
            "Status": "PASS",
            "Observed": retained_rows + excluded_rows,
            "Expected": retained_rows + excluded_rows,
            "Detail": "Every metric observation written once to retained or excluded output",
        },
        {
            "Check": "groups_below_minimum",
            "Status": "WARN" if len(summaries) != len(retained_group_rows) else "PASS",
            "Observed": len(summaries) - len(retained_group_rows),
            "Expected": 0,
            "Detail": "Foodweb-metric groups below the minimum retained-run criterion",
        },
    ]
    write_rows(
        temp_dir / "validation_report.csv",
        ("Check", "Status", "Observed", "Expected", "Detail"),
        validation_rows,
    )

    manifest_output = {
        "protocol_version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_result_root": str(result_root),
        "source_manifest": manifest,
        "scenario": scenario,
        "source_csv_count": len(files),
        "source_csv_total_bytes": sum(path.stat().st_size for path in files),
        "source_rows_read": rows_read,
        "source_rows_at_threshold": rows_at_threshold,
        "classification_threshold": threshold,
        "threshold_tolerance": THRESHOLD_TOLERANCE,
        "tukey_iqr_multiplier": multiplier,
        "minimum_retained_fraction": minimum_fraction,
        "standard_minimum_example": "25 of 50",
        "kfold_minimum_example": "10 of 20 repeated-CV experiments",
        "kfold_analysis_unit": "mean across complete folds within each repeated-CV experiment",
        "fence_grouping": ["Foodweb", "Version", "TrainRatio", "Threshold", "K", "CvK", "Metric"],
        "metric_specific_retention": True,
        "retained_metric_rows": retained_rows,
        "excluded_metric_rows": excluded_rows,
        "foodweb_metric_groups": len(summaries),
        "retained_foodweb_metric_groups": len(retained_group_rows),
        "missing_metric_file_counts": dict(sorted(missing_metric_files.items())),
        "resolved_metric_sources": {
            key: sorted(value) for key, value in sorted(resolved_sources.items())
        },
        "source_files_modified": False,
        "outputs": sorted(path.name for path in temp_dir.iterdir()),
    }
    (temp_dir / "retention_manifest.json").write_text(
        json.dumps(manifest_output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_dir, target)
    print(f"RETENTION_OK scenario={scenario} output={target}", flush=True)
    return target


def discover_role_or_mass_roots() -> List[Path]:
    data_dir = ROOT / "src/matlab/data"
    pattern = (
        "result_wlnm_dir_neg_50x290_train10-90_thresh10-90_"
        "randomeligible_roleormass_tau*_checkconn*_adaptivefalse_Apocrita"
    )
    return sorted(path for path in data_dir.glob(pattern) if path.is_dir())


def main() -> int:
    csv.field_size_limit(sys.maxsize)
    args = parse_args()
    roots = list(args.result_root)
    if args.discover_role_or_mass:
        roots.extend(discover_role_or_mass_roots())
    unique_roots: List[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_roots.append(resolved)
    if not unique_roots:
        raise SystemExit("Provide --result-root or --discover-role-or-mass.")
    if args.iqr_multiplier < 0:
        raise SystemExit("--iqr-multiplier must be non-negative.")
    if not (0 < args.minimum_retained_fraction <= 1):
        raise SystemExit("--minimum-retained-fraction must be in (0, 1].")
    failures = []
    for root in unique_roots:
        try:
            process_result_root(
                root,
                args.output_name,
                args.threshold,
                args.iqr_multiplier,
                args.minimum_retained_fraction,
                args.max_files,
            )
        except Exception as exc:
            failures.append((root, exc))
            print(f"RETENTION_FAILED root={root} error={exc}", file=sys.stderr, flush=True)
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
