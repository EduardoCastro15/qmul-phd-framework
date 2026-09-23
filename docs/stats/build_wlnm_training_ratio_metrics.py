#!/usr/bin/env python3
"""Derive and Tukey-filter WLNM training-graph ratio metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Tuple

from apply_wlnm_tukey_retention import (
    foodweb_from_filename,
    number_text,
    parse_float,
    parse_int,
    read_key_value_manifest,
)
from build_wlnm_ratio_metrics import (
    DEFAULT_RESULT_ROOT,
    RECIPROCAL_TOLERANCE,
    RUN_OUTPUT_FIELDS,
    SUMMARY_OUTPUT_FIELDS,
    THRESHOLD_TOLERANCE,
    GzipCsvWriter,
    process_retention_group,
    retention_group_key,
    validation_row,
    write_rows,
)


DEFAULT_OUTPUT_NAME = (
    "training_ratio_metrics_tukey_iqr_1p5_"
    "min25runs_threshold0p50_v1"
)

RATIO_DEFINITIONS = (
    {
        "name": "ResourceToConsumerRatio",
        "label": "Resource-to-consumer ratio",
        "numerator": "TrainMeanGenerality",
        "denominator": "TrainMeanVulnerability",
        "reference_numerator": "EmpiricalMeanGenerality",
        "reference_denominator": "EmpiricalMeanVulnerability",
    },
    {
        "name": "ConsumerToResourceRatio",
        "label": "Consumer-to-resource ratio",
        "numerator": "TrainMeanVulnerability",
        "denominator": "TrainMeanGenerality",
        "reference_numerator": "EmpiricalMeanVulnerability",
        "reference_denominator": "EmpiricalMeanGenerality",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    parser.add_argument("--minimum-retained-runs", type=int, default=25)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def positive_ratio(numerator: object, denominator: object, context: str) -> float:
    numerator_value = parse_float(numerator)
    denominator_value = parse_float(denominator)
    if numerator_value is None or numerator_value <= 0:
        raise ValueError(f"{context}: numerator must be finite and positive")
    if denominator_value is None or denominator_value <= 0:
        raise ValueError(f"{context}: denominator must be finite and positive")
    return numerator_value / denominator_value


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
        value = positive_ratio(
            raw.get(definition["numerator"]),
            raw.get(definition["denominator"]),
            context,
        )
        reference = positive_ratio(
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
                "MetricFamily": "training_ratio",
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


def process_result_root(
    result_root: Path,
    output_name: str,
    threshold: float,
    multiplier: float,
    minimum_retained_runs: int,
    max_files: Optional[int] = None,
) -> Path:
    result_root = result_root.resolve()
    files = sorted((result_root / "prediction_scores_logs").glob("*.csv"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError("No prediction CSV files were found")

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

    target_parent = result_root / "retention_protocol"
    target = target_parent / output_name
    if target.exists():
        raise FileExistsError(f"Training-ratio output already exists: {target}")
    target_parent.mkdir(exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_name}.", dir=target_parent))
    figure_dir = temp_dir / "figure_inputs"
    figure_dir.mkdir()

    retained_writer = GzipCsvWriter(
        temp_dir / "retained_run_training_ratio_metrics.csv.gz",
        RUN_OUTPUT_FIELDS,
    )
    excluded_writer = GzipCsvWriter(
        temp_dir / "excluded_run_training_ratio_metrics.csv.gz",
        RUN_OUTPUT_FIELDS,
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
            run_ids_by_ratio: MutableMapping[
                str, set[Tuple[str, str, str]]
            ] = defaultdict(set)
            with source_csv.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                required_columns = {
                    "Iteration", "ExperimentID", "Seed", "TrainRatio",
                    "Threshold", "K", "Version",
                    "EmpiricalMeanGenerality", "EmpiricalMeanVulnerability",
                    "TrainMeanGenerality", "TrainMeanVulnerability",
                }
                missing = required_columns.difference(reader.fieldnames or [])
                if missing:
                    raise ValueError(
                        f"{source_csv.name} is missing columns: {sorted(missing)}"
                    )
                for raw in reader:
                    source_rows_read += 1
                    row_threshold = parse_float(raw.get("Threshold"))
                    if row_threshold is None or not math.isclose(
                        row_threshold,
                        threshold,
                        abs_tol=THRESHOLD_TOLERANCE,
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
                            f"{source_csv.name}: duplicate run {run_id} "
                            f"at train ratio {train_ratio}"
                        )
                    run_ids_by_ratio[train_ratio].add(run_id)
                    derived, reciprocal_error = derive_observations(
                        raw, scenario, source_csv, foodweb
                    )
                    reciprocal_max_error = max(
                        reciprocal_max_error, reciprocal_error
                    )
                    observations.extend(derived)
                    ratio_observations += len(derived)

            observed_train_ratios = tuple(
                sorted(int(value) for value in run_ids_by_ratio)
            )
            if observed_train_ratios != tuple(sorted(expected_train_ratios)):
                raise ValueError(
                    f"{source_csv.name}: train ratios {observed_train_ratios}; "
                    f"expected {tuple(sorted(expected_train_ratios))}"
                )
            for ratio, run_ids in run_ids_by_ratio.items():
                if len(run_ids) != expected_runs:
                    raise ValueError(
                        f"{source_csv.name}: train ratio {ratio} has "
                        f"{len(run_ids)} runs; expected {expected_runs}"
                    )

            grouped: MutableMapping[
                Tuple[str, ...], List[Dict[str, object]]
            ] = defaultdict(list)
            for observation in observations:
                grouped[retention_group_key(observation)].append(observation)
            for key in sorted(grouped):
                flagged, summary = process_retention_group(
                    grouped[key],
                    multiplier,
                    expected_runs,
                    minimum_retained_runs,
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
                    f"training_ratio_observations={ratio_observations:,}",
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
        temp_dir / "training_ratio_retention_by_foodweb_metric.csv",
        SUMMARY_OUTPUT_FIELDS,
        summaries,
    )
    eligible_summaries = [
        row for row in summaries if int(row["MeetsMinimumRetainedRuns"]) == 1
    ]
    write_rows(
        figure_dir / "training_ratio_metric_means_after_tukey.csv",
        SUMMARY_OUTPUT_FIELDS,
        eligible_summaries,
    )

    expected_source_rows = len(files) * len(expected_train_ratios) * expected_runs
    expected_observations = expected_source_rows * len(RATIO_DEFINITIONS)
    expected_groups = len(files) * len(expected_train_ratios) * len(RATIO_DEFINITIONS)
    validation_rows = [
        validation_row(
            "prediction_csv_count", len(files),
            expected_files if max_files is None else len(files),
            len(files) == expected_files or max_files is not None,
            "Prediction CSV files processed",
        ),
        validation_row(
            "source_rows_at_threshold", source_rows_at_threshold,
            expected_source_rows, source_rows_at_threshold == expected_source_rows,
            f"Rows at Threshold={threshold}",
        ),
        validation_row(
            "training_ratio_observations", ratio_observations,
            expected_observations, ratio_observations == expected_observations,
            "Two training-graph ratios per source run",
        ),
        validation_row(
            "foodweb_trainratio_metric_groups", len(summaries),
            expected_groups, len(summaries) == expected_groups,
            "Foodweb x train-ratio x ratio-metric summaries",
        ),
        validation_row(
            "retained_excluded_partition",
            retained_observations + excluded_observations,
            ratio_observations,
            retained_observations + excluded_observations == ratio_observations,
            "Every training-ratio observation written once",
        ),
        validation_row(
            "groups_meeting_minimum", len(eligible_summaries), len(summaries),
            len(eligible_summaries) == len(summaries),
            f"Every group retains at least {minimum_retained_runs} runs",
        ),
        validation_row(
            "reciprocal_identity_max_error", reciprocal_max_error,
            f"<={RECIPROCAL_TOLERANCE}",
            reciprocal_max_error <= RECIPROCAL_TOLERANCE,
            "Per-run resource-to-consumer x consumer-to-resource",
        ),
    ]
    write_rows(
        temp_dir / "validation_report.csv",
        ("Check", "Status", "Observed", "Expected", "Detail"),
        validation_rows,
    )
    failed = [row for row in validation_rows if row["Status"] != "PASS"]
    if failed:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError(f"Training-ratio validation failed: {failed}")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": "v1",
        "source_result_root": str(result_root),
        "source_manifest_path": str(manifest_path),
        "source_manifest": source_manifest,
        "source_csv_count": len(files),
        "source_rows_read": source_rows_read,
        "source_rows_at_threshold": source_rows_at_threshold,
        "classification_threshold": threshold,
        "train_ratios": list(expected_train_ratios),
        "expected_runs_per_group": expected_runs,
        "minimum_retained_runs": minimum_retained_runs,
        "tukey_iqr_multiplier": multiplier,
        "metric_specific_retention": True,
        "metric_family": "training_ratio",
        "ratio_definitions": list(RATIO_DEFINITIONS),
        "training_ratio_observations": ratio_observations,
        "retained_training_ratio_observations": retained_observations,
        "excluded_training_ratio_observations": excluded_observations,
        "foodweb_trainratio_metric_groups": len(summaries),
        "reciprocal_identity_max_error": reciprocal_max_error,
        "source_files_modified": False,
        "outputs": [
            "retained_run_training_ratio_metrics.csv.gz",
            "excluded_run_training_ratio_metrics.csv.gz",
            "training_ratio_retention_by_foodweb_metric.csv",
            "figure_inputs/training_ratio_metric_means_after_tukey.csv",
            "validation_report.csv",
        ],
    }
    (temp_dir / "training_ratio_metrics_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_dir.replace(target)
    print(f"wrote training-ratio protocol: {target}")
    return target


def main() -> None:
    args = parse_args()
    process_result_root(
        result_root=args.result_root,
        output_name=args.output_name,
        threshold=args.threshold,
        multiplier=args.iqr_multiplier,
        minimum_retained_runs=args.minimum_retained_runs,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
