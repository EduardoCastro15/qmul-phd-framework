#!/usr/bin/env python3
"""Derive Athen-normalized WLNM generality/vulnerability and apply Tukey.

The historical prediction CSVs are immutable inputs.  For each pseudo-web
run this script derives the positive-only normalized means from the logged
mean, link count, and species count, applies the established metric-specific
1.5 x IQR rule, and writes a new versioned retention directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from apply_wlnm_tukey_retention import (
    foodweb_from_filename,
    number_text,
    parse_float,
    parse_int,
    read_key_value_manifest,
)
from build_wlnm_ratio_metrics import (
    GzipCsvWriter,
    RUN_OUTPUT_FIELDS,
    SUMMARY_OUTPUT_FIELDS,
    difference_direction,
    normalize_foodweb_key,
    process_retention_group,
    retention_group_key,
    validation_row,
    write_rows,
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
DEFAULT_OUTPUT_NAME = (
    "normalized_gv_athen_tukey_iqr_1p5_"
    "min25runs_threshold0p50_v1"
)
COMBINED_FIGURE_INPUT_NAME = (
    "athen_normalized_empirical_vs_inferred_"
    "all_train_ratios_after_tukey.csv"
)
THRESHOLD_TOLERANCE = 1e-9
REFERENCE_TOLERANCE = 1e-12


NORMALIZED_METRIC_DEFINITIONS = (
    {
        "name": "MeanNormalizedGeneralityConsumersOnly",
        "label": "Mean normalized generality (consumers only)",
        "mean_suffix": "MeanGenerality",
    },
    {
        "name": "MeanNormalizedVulnerabilityResourcesOnly",
        "label": "Mean normalized vulnerability (resources only)",
        "mean_suffix": "MeanVulnerability",
    },
)

BASE_METRIC_MAP = {
    "PseudoConnectance": "Connectance",
    "PseudoMeanTrophicHeight": "MeanTrophicHeight",
}
BASE_METRIC_ORDER = tuple(BASE_METRIC_MAP.values())
NORMALIZED_METRIC_ORDER = tuple(
    definition["name"] for definition in NORMALIZED_METRIC_DEFINITIONS
)
COMBINED_METRIC_ORDER = BASE_METRIC_ORDER + NORMALIZED_METRIC_ORDER

COMBINED_OUTPUT_FIELDS = (
    "Scenario",
    "Version",
    "TrainRatio",
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
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Testing aid: process only the first N prediction CSVs.",
    )
    return parser.parse_args()


def normalized_positive_mean(
    mean_value: object,
    links: object,
    species: object,
    context: str,
) -> float:
    mean_number = parse_float(mean_value)
    link_count = parse_float(links)
    species_count = parse_float(species)
    if mean_number is None or mean_number < 0:
        raise ValueError(f"{context}: mean must be finite and non-negative")
    if link_count is None or link_count <= 0:
        raise ValueError(f"{context}: link count must be finite and positive")
    if species_count is None or species_count <= 0:
        raise ValueError(f"{context}: species count must be finite and positive")
    return mean_number / (link_count / species_count)


def derive_observations(
    raw: Mapping[str, str],
    scenario: str,
    source_csv: Path,
    foodweb: str,
) -> List[Dict[str, object]]:
    observations: List[Dict[str, object]] = []
    for definition in NORMALIZED_METRIC_DEFINITIONS:
        context = (
            f"{source_csv.name}, iteration={raw.get('Iteration', '')}, "
            f"train_ratio={raw.get('TrainRatio', '')}, "
            f"metric={definition['name']}"
        )
        pseudo = normalized_positive_mean(
            raw.get(f"Pseudo{definition['mean_suffix']}"),
            raw.get("PseudoLinks"),
            raw.get("PseudoNumSpecies"),
            context,
        )
        empirical = normalized_positive_mean(
            raw.get(f"Empirical{definition['mean_suffix']}"),
            raw.get("EmpiricalLinks"),
            raw.get("EmpiricalNumSpecies"),
            context + ", empirical reference",
        )
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
                "MetricFamily": "pseudo_normalized_athen",
                "Formula": (
                    f"Pseudo{definition['mean_suffix']}/"
                    "(PseudoLinks/PseudoNumSpecies)"
                ),
                "ReferenceFormula": (
                    f"Empirical{definition['mean_suffix']}/"
                    "(EmpiricalLinks/EmpiricalNumSpecies)"
                ),
                "Value": pseudo,
                "ReferenceValue": empirical,
            }
        )
    return observations


def load_foodweb_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing food-web metadata: {path}")
    metadata: Dict[str, Dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Foodweb", "EcosystemType"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
        for row in reader:
            key = normalize_foodweb_key(row["Foodweb"])
            value = {
                "FoodWeb": str(row["Foodweb"]).strip(),
                "FW_KEY": key,
                "Ecosystem": str(row["EcosystemType"]).strip(),
            }
            if key in metadata and metadata[key] != value:
                raise ValueError(f"Conflicting metadata for {row['Foodweb']}")
            metadata[key] = value
    return metadata


def combined_row(
    summary: Mapping[str, object],
    metric: str,
    metadata: Mapping[str, str],
) -> Dict[str, object]:
    empirical = parse_float(summary.get("ReferenceValue"))
    pseudo = parse_float(summary.get("MeanAfterTukey"))
    retained = parse_int(summary.get("RetainedRunsAfterTukey"))
    removed = parse_int(summary.get("OutlierRunsExcluded"))
    if empirical is None or pseudo is None or retained is None or removed is None:
        raise ValueError(
            f"Incomplete retained summary for {summary.get('Foodweb')} / {metric}"
        )
    difference = pseudo - empirical
    return {
        "Scenario": summary["Scenario"],
        "Version": summary["Version"],
        "TrainRatio": number_text(summary["TrainRatio"]),
        **metadata,
        "Metric": metric,
        "EmpiricalValue": empirical,
        "MeanPseudoAfterFiltering": pseudo,
        "ValidRunsAfterFiltering": retained,
        "OutlierRunsRemoved": removed,
        "Difference": difference,
        "DifferenceDirection": difference_direction(difference),
    }


def build_combined_figure_input(
    source_summary: Path,
    normalized_summaries: Sequence[Mapping[str, object]],
    metadata_path: Path,
    expected_train_ratios: Sequence[int],
) -> List[Dict[str, object]]:
    if not source_summary.is_file():
        raise FileNotFoundError(f"Missing existing Tukey summary: {source_summary}")
    metadata = load_foodweb_metadata(metadata_path)
    rows: List[Dict[str, object]] = []

    with source_summary.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "Scenario",
            "Foodweb",
            "Version",
            "TrainRatio",
            "Metric",
            "ReferenceValue",
            "MeanAfterTukey",
            "RetainedRunsAfterTukey",
            "OutlierRunsExcluded",
            "MeetsMinimumRetainedRuns",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Existing Tukey summary is missing columns: {sorted(missing)}"
            )
        for summary in reader:
            if summary["Metric"] not in BASE_METRIC_MAP:
                continue
            if parse_int(summary["TrainRatio"]) not in expected_train_ratios:
                continue
            if parse_int(summary["MeetsMinimumRetainedRuns"]) != 1:
                continue
            key = normalize_foodweb_key(summary["Foodweb"])
            if key not in metadata:
                raise ValueError(f"Missing ecosystem metadata for {summary['Foodweb']}")
            rows.append(
                combined_row(summary, BASE_METRIC_MAP[summary["Metric"]], metadata[key])
            )

    for summary in normalized_summaries:
        if parse_int(summary["MeetsMinimumRetainedRuns"]) != 1:
            continue
        key = normalize_foodweb_key(summary["Foodweb"])
        if key not in metadata:
            raise ValueError(f"Missing ecosystem metadata for {summary['Foodweb']}")
        rows.append(combined_row(summary, str(summary["Metric"]), metadata[key]))

    order = {metric: index for index, metric in enumerate(COMBINED_METRIC_ORDER)}
    rows.sort(
        key=lambda row: (
            parse_int(row["TrainRatio"]) or -1,
            order[str(row["Metric"])],
            str(row["Ecosystem"]),
            str(row["FW_KEY"]),
        )
    )
    return rows


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
    if minimum_retained_runs > expected_runs:
        raise ValueError("minimum retained runs cannot exceed expected runs")

    target_parent = result_root / "retention_protocol"
    target = target_parent / output_name
    if target.exists():
        raise FileExistsError(
            f"Normalized-metric output already exists: {target}. "
            "Use a new --output-name."
        )
    target_parent.mkdir(exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_name}.", dir=target_parent))
    figure_dir = temp_dir / "figure_inputs"
    figure_dir.mkdir()

    retained_writer = GzipCsvWriter(
        temp_dir / "retained_run_normalized_gv_metrics.csv.gz",
        RUN_OUTPUT_FIELDS,
    )
    excluded_writer = GzipCsvWriter(
        temp_dir / "excluded_run_normalized_gv_metrics.csv.gz",
        RUN_OUTPUT_FIELDS,
    )

    scenario = source_manifest.get("Condition", result_root.name)
    summaries: List[Dict[str, object]] = []
    source_rows_read = 0
    source_rows_at_threshold = 0
    normalized_observations = 0
    retained_observations = 0
    excluded_observations = 0
    rows_by_train_ratio: MutableMapping[int, int] = defaultdict(int)
    maximum_reference_spread = 0.0

    try:
        for file_index, source_csv in enumerate(files, start=1):
            foodweb = foodweb_from_filename(source_csv)
            observations: List[Dict[str, object]] = []
            run_ids_by_ratio: MutableMapping[
                int, set[Tuple[str, str, str]]
            ] = defaultdict(set)
            with source_csv.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                required_columns = {
                    "Iteration",
                    "ExperimentID",
                    "Seed",
                    "TrainRatio",
                    "Threshold",
                    "K",
                    "Version",
                    "EmpiricalNumSpecies",
                    "EmpiricalLinks",
                    "EmpiricalMeanGenerality",
                    "EmpiricalMeanVulnerability",
                    "PseudoNumSpecies",
                    "PseudoLinks",
                    "PseudoMeanGenerality",
                    "PseudoMeanVulnerability",
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
                    if str(raw.get("Version", "")).strip() != "WLNM_dir_neg":
                        raise ValueError(
                            f"{source_csv.name}: expected Version=WLNM_dir_neg"
                        )
                    train_ratio = parse_int(raw.get("TrainRatio"))
                    if train_ratio is None:
                        raise ValueError(f"{source_csv.name}: invalid TrainRatio")
                    source_rows_at_threshold += 1
                    rows_by_train_ratio[train_ratio] += 1
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
                    derived = derive_observations(raw, scenario, source_csv, foodweb)
                    observations.extend(derived)
                    normalized_observations += len(derived)

            observed_ratios = tuple(sorted(run_ids_by_ratio))
            if observed_ratios != tuple(sorted(expected_train_ratios)):
                raise ValueError(
                    f"{source_csv.name}: train ratios {observed_ratios}; "
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
                references = [float(row["ReferenceValue"]) for row in grouped[key]]
                maximum_reference_spread = max(
                    maximum_reference_spread,
                    max(references) - min(references),
                )
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
                    f"normalized_observations={normalized_observations:,}",
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
        temp_dir / "normalized_gv_retention_by_foodweb_metric.csv",
        SUMMARY_OUTPUT_FIELDS,
        summaries,
    )
    eligible_summaries = [
        row for row in summaries if int(row["MeetsMinimumRetainedRuns"]) == 1
    ]
    write_rows(
        figure_dir / "normalized_gv_metric_means_after_tukey.csv",
        SUMMARY_OUTPUT_FIELDS,
        eligible_summaries,
    )

    source_summary = (
        result_root
        / "retention_protocol"
        / SOURCE_RETENTION_NAME
        / "retained_foodwebs_by_metric.csv"
    )
    metadata_path = (
        REPO_ROOT
        / "src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv"
    )
    combined_rows = build_combined_figure_input(
        source_summary,
        summaries,
        metadata_path,
        expected_train_ratios,
    )
    combined_path = figure_dir / COMBINED_FIGURE_INPUT_NAME
    write_rows(combined_path, COMBINED_OUTPUT_FIELDS, combined_rows)

    expected_source_rows = len(files) * len(expected_train_ratios) * expected_runs
    expected_observations = expected_source_rows * len(NORMALIZED_METRIC_DEFINITIONS)
    expected_groups = len(files) * len(expected_train_ratios) * len(
        NORMALIZED_METRIC_DEFINITIONS
    )
    expected_combined_rows = len(files) * len(expected_train_ratios) * len(
        COMBINED_METRIC_ORDER
    )
    expected_rows_per_ratio = len(files) * expected_runs
    observed_rows_per_ratio = dict(sorted(rows_by_train_ratio.items()))
    expected_rows_by_ratio = {
        ratio: expected_rows_per_ratio for ratio in expected_train_ratios
    }
    summary_counts = Counter(str(row["Metric"]) for row in summaries)
    combined_counts = Counter(str(row["Metric"]) for row in combined_rows)
    combined_ratio_counts = Counter(
        parse_int(row["TrainRatio"]) for row in combined_rows
    )
    minimum_retained = {
        f"{ratio}:{metric}": min(
            int(row["RetainedRunsAfterTukey"])
            for row in summaries
            if parse_int(row["TrainRatio"]) == ratio and row["Metric"] == metric
        )
        for ratio in expected_train_ratios
        for metric in NORMALIZED_METRIC_ORDER
    }

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
            "source_rows_by_train_ratio",
            observed_rows_per_ratio,
            expected_rows_by_ratio,
            observed_rows_per_ratio == expected_rows_by_ratio,
            "Each train ratio has one hundred runs for every food web",
        ),
        validation_row(
            "normalized_observations",
            normalized_observations,
            expected_observations,
            normalized_observations == expected_observations,
            "Two normalized observations per source run",
        ),
        validation_row(
            "foodweb_trainratio_metric_groups",
            len(summaries),
            expected_groups,
            len(summaries) == expected_groups,
            "Foodweb x train-ratio x normalized-metric summaries",
        ),
        validation_row(
            "retained_excluded_partition",
            retained_observations + excluded_observations,
            normalized_observations,
            retained_observations + excluded_observations == normalized_observations,
            "Every normalized observation written once",
        ),
        validation_row(
            "groups_meeting_minimum",
            len(eligible_summaries),
            len(summaries),
            len(eligible_summaries) == len(summaries),
            f"Every group retains at least {minimum_retained_runs} runs",
        ),
        validation_row(
            "empirical_reference_max_spread",
            maximum_reference_spread,
            f"<={REFERENCE_TOLERANCE}",
            maximum_reference_spread <= REFERENCE_TOLERANCE,
            "Empirical reference is constant within every Tukey group",
        ),
        validation_row(
            "summary_metric_counts",
            dict(sorted(summary_counts.items())),
            {metric: len(files) * len(expected_train_ratios) for metric in NORMALIZED_METRIC_ORDER},
            all(
                summary_counts[metric] == len(files) * len(expected_train_ratios)
                for metric in NORMALIZED_METRIC_ORDER
            ),
            "Each normalized metric has one summary per food web and train ratio",
        ),
        validation_row(
            "combined_all_train_ratio_rows",
            len(combined_rows),
            expected_combined_rows,
            len(combined_rows) == expected_combined_rows,
            "Food webs x train ratios x four plotting metrics",
        ),
        validation_row(
            "combined_metric_counts",
            dict(sorted(combined_counts.items())),
            {metric: len(files) * len(expected_train_ratios) for metric in COMBINED_METRIC_ORDER},
            all(
                combined_counts[metric] == len(files) * len(expected_train_ratios)
                for metric in COMBINED_METRIC_ORDER
            ),
            "Each plotting metric has one row per food web and train ratio",
        ),
        validation_row(
            "combined_rows_per_train_ratio",
            dict(sorted(combined_ratio_counts.items())),
            {ratio: len(files) * len(COMBINED_METRIC_ORDER) for ratio in expected_train_ratios},
            all(
                combined_ratio_counts[ratio] == len(files) * len(COMBINED_METRIC_ORDER)
                for ratio in expected_train_ratios
            ),
            "Each train ratio contains 290 food webs x four plotting metrics",
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
        "source_existing_tukey_summary": str(source_summary),
        "source_foodweb_metadata": str(metadata_path),
        "source_files_modified": False,
        "source_csv_count": len(files),
        "source_rows_read": source_rows_read,
        "source_rows_at_threshold": source_rows_at_threshold,
        "normalized_observations": normalized_observations,
        "classification_threshold": threshold,
        "tukey_iqr_multiplier": multiplier,
        "minimum_retained_runs": minimum_retained_runs,
        "expected_runs_per_group": expected_runs,
        "train_ratios": list(expected_train_ratios),
        "metric_specific_retention": True,
        "fence_grouping": [
            "Foodweb",
            "Version",
            "TrainRatio",
            "Threshold",
            "K",
            "Metric",
        ],
        "normalized_metric_definitions": list(NORMALIZED_METRIC_DEFINITIONS),
        "maximum_empirical_reference_spread": maximum_reference_spread,
        "minimum_retained_runs_by_train_ratio_metric": minimum_retained,
        "retained_normalized_observations": retained_observations,
        "excluded_normalized_observations": excluded_observations,
        "foodweb_trainratio_metric_groups": len(summaries),
        "combined_figure_input_rows": len(combined_rows),
        "outputs": sorted(
            str(path.relative_to(temp_dir))
            for path in temp_dir.rglob("*")
            if path.is_file()
        ),
    }
    (temp_dir / "normalized_gv_metrics_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_dir, target)
    print(f"NORMALIZED_GV_METRICS_OK output={target}", flush=True)
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
        args.max_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
