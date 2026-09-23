#!/usr/bin/env python3
"""Derive and Tukey-filter WLNM resource/consumer ratio metrics.

The WLNM prediction logs are treated as immutable inputs.  This script derives
the two reciprocal ratios for every pseudo-web run, applies the established
metric-wise 1.5 x IQR retention rule, and writes a new versioned analysis
directory.  It also combines the train-60 ratio summaries with the existing
four-metric plotting input used by the Week 39 notebook.
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
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from apply_wlnm_tukey_retention import (
    foodweb_from_filename,
    number_text,
    parse_float,
    parse_int,
    read_key_value_manifest,
    tukey_fences,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = (
    REPO_ROOT
    / "src/matlab/data"
    / (
        "result_wlnm_dir_neg_roleonly_trophicv2_"
        "100x290_train10-90_Apocrita_20260910_011848"
    )
)
SOURCE_RETENTION_NAME = "tukey_iqr_1p5_min25runs_threshold0p50_v1"
DEFAULT_OUTPUT_NAME = "ratio_metrics_tukey_iqr_1p5_min25runs_threshold0p50_v1"
SOURCE_PAIRS_NAME = (
    "figure_5_wilcoxon_train60_roleonly_trophicv2_"
    "100runs_threshold0p50_after_tukey_min25runs_pairs.csv"
)
COMBINED_FIGURE_INPUT_NAME = "athen_empirical_vs_inferred_train60_after_tukey.csv"
THRESHOLD_TOLERANCE = 1e-9
RECIPROCAL_TOLERANCE = 1e-12


RATIO_DEFINITIONS = (
    {
        "name": "ResourceToConsumerRatio",
        "label": "Resource-to-consumer ratio",
        "numerator": "PseudoMeanGenerality",
        "denominator": "PseudoMeanVulnerability",
        "reference_numerator": "EmpiricalMeanGenerality",
        "reference_denominator": "EmpiricalMeanVulnerability",
    },
    {
        "name": "ConsumerToResourceRatio",
        "label": "Consumer-to-resource ratio",
        "numerator": "PseudoMeanVulnerability",
        "denominator": "PseudoMeanGenerality",
        "reference_numerator": "EmpiricalMeanVulnerability",
        "reference_denominator": "EmpiricalMeanGenerality",
    },
)

BASE_METRIC_ORDER = (
    "Connectance",
    "MeanTrophicHeight",
    "MeanGenerality",
    "MeanVulnerability",
)
COMBINED_METRIC_ORDER = BASE_METRIC_ORDER + tuple(
    definition["name"] for definition in RATIO_DEFINITIONS
)

RUN_OUTPUT_FIELDS = (
    "Scenario",
    "SourceCSV",
    "Foodweb",
    "Version",
    "Iteration",
    "ExperimentID",
    "Seed",
    "TrainRatio",
    "Threshold",
    "K",
    "Metric",
    "MetricLabel",
    "MetricFamily",
    "Formula",
    "ReferenceFormula",
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

SUMMARY_OUTPUT_FIELDS = (
    "Scenario",
    "Foodweb",
    "Version",
    "TrainRatio",
    "Threshold",
    "K",
    "Metric",
    "MetricLabel",
    "MetricFamily",
    "Formula",
    "ReferenceFormula",
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

COMBINED_OUTPUT_FIELDS = (
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
    "Difference",
    "DifferenceDirection",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    parser.add_argument("--minimum-retained-runs", type=int, default=25)
    parser.add_argument("--train-ratio-for-figures", type=int, default=60)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Testing aid: process only the first N prediction CSVs.",
    )
    return parser.parse_args()


def ratio_value(numerator: object, denominator: object, context: str) -> float:
    numerator_value = parse_float(numerator)
    denominator_value = parse_float(denominator)
    if numerator_value is None or numerator_value <= 0:
        raise ValueError(f"{context}: numerator must be finite and positive")
    if denominator_value is None or denominator_value <= 0:
        raise ValueError(f"{context}: denominator must be finite and positive")
    return numerator_value / denominator_value


def normalize_foodweb_key(value: object) -> str:
    name = Path(str(value).strip()).name
    for extension in (".mat", ".csv"):
        if name.casefold().endswith(extension):
            name = name[: -len(extension)].rstrip()
    for suffix in ("_tax_mass", "_taxmass"):
        if name.casefold().endswith(suffix):
            name = name[: -len(suffix)].rstrip()
    return " ".join(name.replace("_", " ").casefold().split())


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
        self.fields = tuple(fields)
        self.handle = gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=6)
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fields, extrasaction="ignore")
        self.writer.writeheader()

    def writerow(self, row: Mapping[str, object]) -> None:
        self.writer.writerow(writeable_row(row, self.fields))

    def close(self) -> None:
        self.handle.close()


def write_rows(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(writeable_row(row, fields))


def finite_mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else math.nan


def finite_stdev(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else math.nan


def retention_group_key(row: Mapping[str, object]) -> Tuple[str, ...]:
    return (
        str(row["Foodweb"]),
        str(row["Version"]),
        str(row["TrainRatio"]),
        str(row["Threshold"]),
        str(row["K"]),
        str(row["Metric"]),
    )


def derive_observations(
    raw: Mapping[str, str],
    scenario: str,
    source_csv: Path,
    foodweb: str,
) -> Tuple[List[Dict[str, object]], float]:
    observations: List[Dict[str, object]] = []
    derived_values: Dict[str, float] = {}
    for definition in RATIO_DEFINITIONS:
        context = (
            f"{source_csv.name}, iteration={raw.get('Iteration', '')}, "
            f"train_ratio={raw.get('TrainRatio', '')}, metric={definition['name']}"
        )
        value = ratio_value(
            raw.get(definition["numerator"]),
            raw.get(definition["denominator"]),
            context,
        )
        reference = ratio_value(
            raw.get(definition["reference_numerator"]),
            raw.get(definition["reference_denominator"]),
            context + ", empirical reference",
        )
        derived_values[definition["name"]] = value
        observations.append(
            {
                "Scenario": scenario,
                "SourceCSV": source_csv.name,
                "Foodweb": foodweb,
                "Version": str(raw.get("Version", "")).strip(),
                "Iteration": number_text(raw.get("Iteration")),
                "ExperimentID": number_text(raw.get("ExperimentID")),
                "Seed": number_text(raw.get("Seed")),
                "TrainRatio": number_text(raw.get("TrainRatio")),
                "Threshold": number_text(raw.get("Threshold")),
                "K": number_text(raw.get("K")),
                "Metric": definition["name"],
                "MetricLabel": definition["label"],
                "MetricFamily": "pseudo_ratio",
                "Formula": f"{definition['numerator']}/{definition['denominator']}",
                "ReferenceFormula": (
                    f"{definition['reference_numerator']}/"
                    f"{definition['reference_denominator']}"
                ),
                "Value": value,
                "ReferenceValue": reference,
            }
        )
    reciprocal_error = abs(
        derived_values["ResourceToConsumerRatio"]
        * derived_values["ConsumerToResourceRatio"]
        - 1.0
    )
    return observations, reciprocal_error


def process_retention_group(
    group: Sequence[Dict[str, object]],
    multiplier: float,
    expected_runs: int,
    minimum_retained_runs: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    values = [float(row["Value"]) for row in group]
    references = [float(row["ReferenceValue"]) for row in group]
    if len(group) != expected_runs:
        raise ValueError(
            f"{retention_group_key(group[0])}: expected {expected_runs} runs, found {len(group)}"
        )
    if max(references) - min(references) > RECIPROCAL_TOLERANCE:
        raise ValueError(
            f"{retention_group_key(group[0])}: empirical ratio varies across runs"
        )

    fences = tukey_fences(values, multiplier)
    retained_mask = [
        fences["LowerFence"] <= value <= fences["UpperFence"] for value in values
    ]
    retained_values = [value for value, retained in zip(values, retained_mask) if retained]
    retained_count = len(retained_values)
    meets_minimum = retained_count >= minimum_retained_runs
    reference = references[0]

    flagged: List[Dict[str, object]] = []
    for original, retained in zip(group, retained_mask):
        row = dict(original)
        value = float(row["Value"])
        if not retained:
            reason = (
                "below_lower_tukey_fence"
                if value < fences["LowerFence"]
                else "above_upper_tukey_fence"
            )
        elif not meets_minimum:
            retained = False
            reason = "group_below_minimum_retained_runs"
        else:
            reason = ""
        row.update(
            {
                **fences,
                "DeltaValue": value - reference,
                "ValidRunsBeforeTukey": len(values),
                "RetainedRunsAfterTukey": retained_count,
                "ExpectedRuns": expected_runs,
                "MinimumRetainedRuns": minimum_retained_runs,
                "MeetsMinimumRetainedRuns": int(meets_minimum),
                "Retained": retained,
                "ExclusionReason": reason,
            }
        )
        flagged.append(row)

    mean_before = finite_mean(values)
    mean_after = finite_mean(retained_values)
    std_after = finite_stdev(retained_values)
    first = group[0]
    summary = {
        "Scenario": first["Scenario"],
        "Foodweb": first["Foodweb"],
        "Version": first["Version"],
        "TrainRatio": first["TrainRatio"],
        "Threshold": first["Threshold"],
        "K": first["K"],
        "Metric": first["Metric"],
        "MetricLabel": first["MetricLabel"],
        "MetricFamily": first["MetricFamily"],
        "Formula": first["Formula"],
        "ReferenceFormula": first["ReferenceFormula"],
        "TotalRunUnits": len(group),
        "ValidRunsBeforeTukey": len(values),
        "InvalidRunsBeforeTukey": 0,
        "OutlierRunsExcluded": len(values) - retained_count,
        "RetainedRunsAfterTukey": retained_count,
        "ExpectedRuns": expected_runs,
        "MinimumRetainedRuns": minimum_retained_runs,
        "MeetsMinimumRetainedRuns": int(meets_minimum),
        **fences,
        "MeanBeforeTukey": mean_before,
        "MeanAfterTukey": mean_after if meets_minimum else None,
        "StdAfterTukey": std_after if meets_minimum else None,
        "SEAfterTukey": (
            std_after / math.sqrt(retained_count)
            if meets_minimum and math.isfinite(std_after) and retained_count
            else None
        ),
        "ReferenceValue": reference if meets_minimum else None,
        "DeltaAfterTukey": mean_after - reference if meets_minimum else None,
        "RelativeDeltaAfterTukey": (
            (mean_after - reference) / reference if meets_minimum else None
        ),
    }
    return flagged, summary


def difference_direction(value: float) -> str:
    if value > 0:
        return "Pseudo > empirical"
    if value < 0:
        return "Pseudo < empirical"
    return "Equal"


def build_combined_figure_input(
    source_pairs: Path,
    summaries: Sequence[Mapping[str, object]],
    train_ratio: int,
) -> List[Dict[str, object]]:
    if not source_pairs.is_file():
        raise FileNotFoundError(f"Missing existing four-metric pairs file: {source_pairs}")
    with source_pairs.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(COMBINED_OUTPUT_FIELDS).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Existing pairs file is missing columns: {sorted(missing)}")
        base_rows = [dict(row) for row in reader]

    processed_keys = {
        normalize_foodweb_key(summary["Foodweb"])
        for summary in summaries
        if parse_int(summary["TrainRatio"]) == train_ratio
    }
    base_rows = [
        row
        for row in base_rows
        if normalize_foodweb_key(row["FoodWeb"]) in processed_keys
    ]
    expected_foodwebs = len(processed_keys)

    base_counts = Counter(row["Metric"] for row in base_rows)
    if set(base_counts) != set(BASE_METRIC_ORDER) or any(
        base_counts[metric] != expected_foodwebs for metric in BASE_METRIC_ORDER
    ):
        raise ValueError(f"Unexpected existing pairs metric counts: {dict(base_counts)}")

    metadata_by_key: Dict[str, Tuple[str, str, str]] = {}
    for row in base_rows:
        key = normalize_foodweb_key(row["FoodWeb"])
        metadata = (row["FoodWeb"], row["FW_KEY"], row["Ecosystem"])
        if key in metadata_by_key and metadata_by_key[key] != metadata:
            raise ValueError(f"Conflicting ecosystem metadata for {row['FoodWeb']}")
        metadata_by_key[key] = metadata

    ratio_rows: List[Dict[str, object]] = []
    for summary in summaries:
        if parse_int(summary["TrainRatio"]) != train_ratio:
            continue
        if int(summary["MeetsMinimumRetainedRuns"]) != 1:
            continue
        key = normalize_foodweb_key(summary["Foodweb"])
        if key not in metadata_by_key:
            raise ValueError(f"Missing ecosystem metadata for {summary['Foodweb']}")
        foodweb, fw_key, ecosystem = metadata_by_key[key]
        empirical = float(summary["ReferenceValue"])
        pseudo = float(summary["MeanAfterTukey"])
        difference = pseudo - empirical
        ratio_rows.append(
            {
                "Scenario": summary["Scenario"],
                "Version": summary["Version"],
                "FoodWeb": foodweb,
                "FW_KEY": fw_key,
                "Ecosystem": ecosystem,
                "Metric": summary["Metric"],
                "EmpiricalValue": empirical,
                "MeanPseudoAfterFiltering": pseudo,
                "ValidRunsAfterFiltering": summary["RetainedRunsAfterTukey"],
                "OutlierRunsRemoved": summary["OutlierRunsExcluded"],
                "Difference": difference,
                "DifferenceDirection": difference_direction(difference),
            }
        )

    combined = base_rows + ratio_rows
    order = {metric: index for index, metric in enumerate(COMBINED_METRIC_ORDER)}
    combined.sort(
        key=lambda row: (
            order[str(row["Metric"])],
            str(row["Ecosystem"]),
            normalize_foodweb_key(row["FoodWeb"]),
        )
    )
    return combined


def validation_row(
    check: str,
    observed: object,
    expected: object,
    passed: bool,
    detail: str,
) -> Dict[str, object]:
    return {
        "Check": check,
        "Status": "PASS" if passed else "FAIL",
        "Observed": observed,
        "Expected": expected,
        "Detail": detail,
    }


def process_result_root(
    result_root: Path,
    output_name: str,
    threshold: float,
    multiplier: float,
    minimum_retained_runs: int,
    train_ratio_for_figures: int,
    max_files: Optional[int] = None,
) -> Path:
    result_root = result_root.resolve()
    logs_dir = result_root / "prediction_scores_logs"
    files = sorted(logs_dir.glob("*.csv"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No prediction CSVs found under {logs_dir}")

    manifest_path = result_root / "RUN_MANIFEST.txt"
    source_manifest = read_key_value_manifest(manifest_path)
    expected_files = parse_int(source_manifest.get("FoodWebs")) or len(files)
    expected_runs = parse_int(source_manifest.get("NumExperiments")) or 100
    expected_train_ratios = tuple(
        int(value.strip())
        for value in source_manifest.get(
            "TrainRatioRange", "10,20,30,40,50,60,70,80,90"
        ).split(",")
        if value.strip()
    )
    if minimum_retained_runs > expected_runs:
        raise ValueError("minimum retained runs cannot exceed expected runs")

    target_parent = result_root / "retention_protocol"
    target = target_parent / output_name
    if target.exists():
        raise FileExistsError(
            f"Ratio-metric output already exists: {target}. Use a new --output-name."
        )
    target_parent.mkdir(exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_name}.", dir=target_parent))
    figure_dir = temp_dir / "figure_inputs"
    figure_dir.mkdir()

    retained_writer = GzipCsvWriter(
        temp_dir / "retained_run_ratio_metrics.csv.gz", RUN_OUTPUT_FIELDS
    )
    excluded_writer = GzipCsvWriter(
        temp_dir / "excluded_run_ratio_metrics.csv.gz", RUN_OUTPUT_FIELDS
    )

    scenario = source_manifest.get("Condition", result_root.name)
    summaries: List[Dict[str, object]] = []
    source_rows_read = 0
    source_rows_at_threshold = 0
    ratio_observations = 0
    retained_observations = 0
    excluded_observations = 0
    reciprocal_max_error = 0.0

    try:
        for file_index, source_csv in enumerate(files, start=1):
            foodweb = foodweb_from_filename(source_csv)
            observations: List[Dict[str, object]] = []
            run_ids_by_ratio: MutableMapping[str, set[Tuple[str, str, str]]] = defaultdict(set)
            with source_csv.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                required_columns = {
                    "Iteration",
                    "ExperimentID",
                    "Seed",
                    "TrainRatio",
                    "Threshold",
                    "K",
                    "EmpiricalMeanGenerality",
                    "EmpiricalMeanVulnerability",
                    "PseudoMeanGenerality",
                    "PseudoMeanVulnerability",
                }
                missing = required_columns.difference(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"{source_csv.name} is missing columns: {sorted(missing)}")
                for raw in reader:
                    source_rows_read += 1
                    row_threshold = parse_float(raw.get("Threshold"))
                    if row_threshold is None or not math.isclose(
                        row_threshold, threshold, abs_tol=THRESHOLD_TOLERANCE
                    ):
                        continue
                    source_rows_at_threshold += 1
                    train_ratio = number_text(raw.get("TrainRatio"))
                    run_id = (
                        number_text(raw.get("Iteration")),
                        number_text(raw.get("ExperimentID")),
                        number_text(raw.get("Seed")),
                    )
                    if run_id in run_ids_by_ratio[train_ratio]:
                        raise ValueError(
                            f"{source_csv.name}: duplicate run identifier {run_id} "
                            f"at train ratio {train_ratio}"
                        )
                    run_ids_by_ratio[train_ratio].add(run_id)
                    derived, reciprocal_error = derive_observations(
                        raw, scenario, source_csv, foodweb
                    )
                    reciprocal_max_error = max(reciprocal_max_error, reciprocal_error)
                    observations.extend(derived)
                    ratio_observations += len(derived)

            observed_train_ratios = tuple(sorted(int(value) for value in run_ids_by_ratio))
            if observed_train_ratios != tuple(sorted(expected_train_ratios)):
                raise ValueError(
                    f"{source_csv.name}: train ratios {observed_train_ratios}, "
                    f"expected {tuple(sorted(expected_train_ratios))}"
                )
            for ratio, run_ids in run_ids_by_ratio.items():
                if len(run_ids) != expected_runs:
                    raise ValueError(
                        f"{source_csv.name}: train ratio {ratio} has {len(run_ids)} runs; "
                        f"expected {expected_runs}"
                    )

            grouped: MutableMapping[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
            for observation in observations:
                grouped[retention_group_key(observation)].append(observation)
            for key in sorted(grouped):
                flagged, summary = process_retention_group(
                    grouped[key], multiplier, expected_runs, minimum_retained_runs
                )
                summaries.append(summary)
                for row in flagged:
                    if bool(row["Retained"]):
                        retained_writer.writerow(row)
                        retained_observations += 1
                    else:
                        excluded_writer.writerow(row)
                        excluded_observations += 1

            if file_index % 25 == 0 or file_index == len(files):
                print(
                    f"processed {file_index}/{len(files)} files; "
                    f"ratio_observations={ratio_observations:,}",
                    flush=True,
                )
    except Exception:
        retained_writer.close()
        excluded_writer.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    retained_writer.close()
    excluded_writer.close()

    summaries.sort(
        key=lambda row: (
            parse_int(row["TrainRatio"]) or -1,
            str(row["Metric"]),
            str(row["Foodweb"]),
        )
    )
    write_rows(
        temp_dir / "ratio_retention_by_foodweb_metric.csv",
        SUMMARY_OUTPUT_FIELDS,
        summaries,
    )
    eligible_summaries = [
        row for row in summaries if int(row["MeetsMinimumRetainedRuns"]) == 1
    ]
    write_rows(
        figure_dir / "ratio_metric_means_after_tukey.csv",
        SUMMARY_OUTPUT_FIELDS,
        eligible_summaries,
    )

    source_pairs = (
        result_root
        / "retention_protocol"
        / SOURCE_RETENTION_NAME
        / "statistical_tests"
        / SOURCE_PAIRS_NAME
    )
    combined_rows = build_combined_figure_input(
        source_pairs, summaries, train_ratio_for_figures
    )
    combined_path = figure_dir / COMBINED_FIGURE_INPUT_NAME
    write_rows(combined_path, COMBINED_OUTPUT_FIELDS, combined_rows)

    ratio_counts = Counter(
        row["Metric"]
        for row in combined_rows
        if row["Metric"] in {definition["name"] for definition in RATIO_DEFINITIONS}
    )
    combined_counts = Counter(row["Metric"] for row in combined_rows)
    expected_summary_groups = len(files) * len(expected_train_ratios) * len(RATIO_DEFINITIONS)
    expected_source_rows = len(files) * len(expected_train_ratios) * expected_runs
    expected_ratio_observations = expected_source_rows * len(RATIO_DEFINITIONS)
    expected_combined_rows = len(files) * len(COMBINED_METRIC_ORDER)
    validation_rows = [
        validation_row(
            "prediction_csv_count",
            len(files),
            expected_files if max_files is None else len(files),
            len(files) == expected_files or max_files is not None,
            "Prediction CSV files processed",
        ),
        validation_row(
            "source_rows_at_threshold",
            source_rows_at_threshold,
            expected_source_rows,
            source_rows_at_threshold == expected_source_rows,
            f"Rows at Threshold={threshold}",
        ),
        validation_row(
            "ratio_observations",
            ratio_observations,
            expected_ratio_observations,
            ratio_observations == expected_ratio_observations,
            "Two derived ratio observations per source run",
        ),
        validation_row(
            "foodweb_trainratio_metric_groups",
            len(summaries),
            expected_summary_groups,
            len(summaries) == expected_summary_groups,
            "Foodweb x train-ratio x ratio-metric summaries",
        ),
        validation_row(
            "retained_excluded_partition",
            retained_observations + excluded_observations,
            ratio_observations,
            retained_observations + excluded_observations == ratio_observations,
            "Every ratio observation written once",
        ),
        validation_row(
            "groups_meeting_minimum",
            len(eligible_summaries),
            len(summaries),
            len(eligible_summaries) == len(summaries),
            f"Every group retains at least {minimum_retained_runs} runs",
        ),
        validation_row(
            "reciprocal_identity_max_error",
            reciprocal_max_error,
            f"<={RECIPROCAL_TOLERANCE}",
            reciprocal_max_error <= RECIPROCAL_TOLERANCE,
            "Per-run ResourceToConsumerRatio x ConsumerToResourceRatio",
        ),
        validation_row(
            "train60_ratio_rows_per_metric",
            dict(sorted(ratio_counts.items())),
            {definition["name"]: len(files) for definition in RATIO_DEFINITIONS},
            all(ratio_counts[definition["name"]] == len(files) for definition in RATIO_DEFINITIONS),
            "Ratio rows included in the train-60 plotting input",
        ),
        validation_row(
            "combined_train60_rows",
            len(combined_rows),
            expected_combined_rows,
            len(combined_rows) == expected_combined_rows,
            "Food webs x six metrics",
        ),
        validation_row(
            "combined_metric_counts",
            dict(sorted(combined_counts.items())),
            {metric: len(files) for metric in COMBINED_METRIC_ORDER},
            all(combined_counts[metric] == len(files) for metric in COMBINED_METRIC_ORDER),
            "Each plotting metric has one row per food web",
        ),
    ]
    write_rows(
        temp_dir / "validation_report.csv",
        ("Check", "Status", "Observed", "Expected", "Detail"),
        validation_rows,
    )
    failed_checks = [row["Check"] for row in validation_rows if row["Status"] != "PASS"]
    if failed_checks:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Validation failed: {', '.join(failed_checks)}")

    manifest = {
        "protocol_version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_result_root": str(result_root),
        "source_manifest_path": str(manifest_path),
        "source_manifest": source_manifest,
        "source_pairs_file": str(source_pairs),
        "source_files_modified": False,
        "source_csv_count": len(files),
        "source_rows_read": source_rows_read,
        "source_rows_at_threshold": source_rows_at_threshold,
        "ratio_observations": ratio_observations,
        "classification_threshold": threshold,
        "tukey_iqr_multiplier": multiplier,
        "minimum_retained_runs": minimum_retained_runs,
        "expected_runs_per_group": expected_runs,
        "train_ratios": list(expected_train_ratios),
        "figure_train_ratio": train_ratio_for_figures,
        "metric_specific_retention": True,
        "fence_grouping": [
            "Foodweb",
            "Version",
            "TrainRatio",
            "Threshold",
            "K",
            "Metric",
        ],
        "ratio_definitions": list(RATIO_DEFINITIONS),
        "reciprocal_identity_max_error": reciprocal_max_error,
        "retained_ratio_observations": retained_observations,
        "excluded_ratio_observations": excluded_observations,
        "foodweb_trainratio_metric_groups": len(summaries),
        "combined_figure_input_rows": len(combined_rows),
        "outputs": sorted(
            str(path.relative_to(temp_dir))
            for path in temp_dir.rglob("*")
            if path.is_file()
        ),
    }
    (temp_dir / "ratio_metrics_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_dir, target)
    print(f"RATIO_METRICS_OK output={target}", flush=True)
    return target


def main() -> int:
    csv.field_size_limit(2**31 - 1)
    args = parse_args()
    if args.iqr_multiplier < 0:
        raise SystemExit("--iqr-multiplier must be non-negative")
    if args.minimum_retained_runs <= 0:
        raise SystemExit("--minimum-retained-runs must be positive")
    process_result_root(
        args.result_root,
        args.output_name,
        args.threshold,
        args.iqr_multiplier,
        args.minimum_retained_runs,
        args.train_ratio_for_figures,
        args.max_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
