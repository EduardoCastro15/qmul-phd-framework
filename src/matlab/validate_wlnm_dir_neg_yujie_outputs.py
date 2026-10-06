#!/usr/bin/env python3
"""Validate a complete Yujie ``observed_zero`` WLNM result root."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from collections import Counter
from pathlib import Path


RESULT_SUFFIX = "_results_random_wlnm_dir_neg.csv"
REQUIRED_MANIFEST = {
    "Version": "WLNM_dir_neg",
    "Eligibility": "observed_zero",
    "NegativePositiveRatio": "2",
    "NegativeSampling": "uniform_without_replacement",
    "NegativeTopupPolicy": "error",
    "ClassificationThreshold": "0.50",
    "Backbone": "false",
    "CheckConnectivity": "false",
    "CrossValidation": "false",
    "MassEligibility": "false",
    "GraphEncodingParallel": "false",
    "ComputeEcologicalMetrics": "false",
    "BaseSeed": "12345",
    "ResampleSplits": "true",
}
REQUIRED_COLUMNS = {
    "Version",
    "ROC_AUC",
    "PR_AUC",
    "TrainRatio",
    "ExperimentID",
    "Seed",
    "Threshold",
    "NegativeEligibilityMode",
    "NegativePositiveRatio",
    "NegativeSamplingStrategy",
    "NegativeTopupPolicy",
    "CandidatePairCount",
    "ObservedZeroPoolSize",
    "EligiblePoolSize",
    "FullNegativePoolSize",
    "RequestedNegativeCount",
    "SelectedNegativeCount",
    "EligibleShortfall",
    "FullPoolShortfall",
    "RandomTopupCount",
    "TrainNegativeCount",
    "TestNegativeCount",
    "DataRegime",
    "SpatialFold",
    "GroupID",
    "Year",
    "InputPositiveCount",
    "InputObservedZeroCount",
    "InputUnresolvedCount",
    "InputCandidateCount",
    "OriginalStatus1Count",
    "OriginalStatus0Count",
    "OriginalStatusNACount",
    "FinalStatus1Count",
    "FinalStatus0Count",
    "FinalStatusNACount",
    "RequiredNegativeEligibilityMode",
    "SourceEvidenceSHA256",
    "PositiveSplitHash",
    "UniquePositiveSplitCount",
    "TrainNegativeBothEndpointsVisible",
    "TrainNegativeOneEndpointVisible",
    "TrainNegativeNeitherEndpointVisible",
    "TestNegativeBothEndpointsVisible",
    "TestNegativeOneEndpointVisible",
    "TestNegativeNeitherEndpointVisible",
    "TestPositiveBothEndpointsVisible",
    "TestPositiveOneEndpointVisible",
    "TestPositiveNeitherEndpointVisible",
    "TrainTestNegativeOverlapCount",
    "PositiveTrainTestOverlapCount",
    "TotalLinks",
    "TrainLinks",
    "TestLinks",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_manifest(path: Path) -> dict[str, str]:
    require(path.is_file(), f"Missing run manifest: {path}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def parse_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_int(row: dict[str, str], key: str) -> int:
    value = float(row[key])
    require(math.isfinite(value) and value.is_integer(), f"Invalid integer {key}={row[key]!r}")
    return int(value)


def as_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    require(math.isfinite(value), f"Non-finite {key}={row[key]!r}")
    return value


def expected_ratios(manifest: dict[str, str]) -> set[float]:
    return {float(value) for value in manifest["TrainRatioRange"].split(",")}


def validate_sacct(path: Path, job_id: str, expected_tasks: int) -> None:
    require(path.is_file(), f"Missing sacct export: {path}")
    task_pattern = re.compile(rf"^{re.escape(job_id)}_(\d+)$")
    task_rows: dict[int, tuple[str, str]] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.strip().split("|")
        if len(parts) < 3:
            continue
        match = task_pattern.match(parts[0])
        if match:
            task_rows[int(match.group(1))] = (parts[1].split()[0], parts[2])
    require(len(task_rows) == expected_tasks,
            f"sacct: expected {expected_tasks} array tasks, found {len(task_rows)}")
    failures = {
        task: values for task, values in task_rows.items()
        if values[0] != "COMPLETED" or values[1] != "0:0"
    }
    require(not failures, f"sacct has failed/incomplete tasks: {failures}")


def validate_result_csv(
    path: Path,
    input_row: dict[str, str],
    regime: str,
    experiments: int,
    ratios: set[float],
) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"Missing header: {path}")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        require(not missing, f"{path.name}: missing columns {sorted(missing)}")
        rows = list(reader)

    expected_rows = experiments * len(ratios)
    require(len(rows) == expected_rows,
            f"{path.name}: expected {expected_rows} rows, found {len(rows)}")
    keys = Counter((as_float(row, "TrainRatio"), as_int(row, "ExperimentID")) for row in rows)
    expected_keys = Counter((ratio, exp) for ratio in ratios for exp in range(1, experiments + 1))
    require(keys == expected_keys, f"{path.name}: incomplete/duplicate ratio and ExperimentID keys")

    hashes_by_ratio: dict[float, set[str]] = {ratio: set() for ratio in ratios}
    for line_number, row in enumerate(rows, 2):
        prefix = f"{path.name}:{line_number}"
        require(row["Version"] == "WLNM_dir_neg", f"{prefix}: wrong Version")
        require(row["NegativeEligibilityMode"] == "observed_zero", f"{prefix}: wrong mode")
        require(row["RequiredNegativeEligibilityMode"] == "observed_zero",
                f"{prefix}: input MAT did not require observed_zero")
        require(row["NegativeSamplingStrategy"] == "uniform_without_replacement",
                f"{prefix}: wrong sampling strategy")
        require(row["NegativeTopupPolicy"] == "error", f"{prefix}: fallback enabled")
        require(abs(as_float(row, "NegativePositiveRatio") - 2.0) < 1e-12,
                f"{prefix}: ratio is not 2:1")
        require(abs(as_float(row, "Threshold") - 0.5) < 1e-12,
                f"{prefix}: threshold is not 0.50")
        require(row["DataRegime"] == regime, f"{prefix}: wrong data regime")
        require(row["GroupID"] == input_row["group_id"], f"{prefix}: wrong group")
        require(as_int(row, "SpatialFold") == int(input_row["fold"]), f"{prefix}: wrong fold")
        require(as_int(row, "Year") == int(input_row["year"]), f"{prefix}: wrong year")
        require(row["SourceEvidenceSHA256"] == input_row["source_evidence_sha256"],
                f"{prefix}: source hash mismatch")

        input_pos = as_int(row, "InputPositiveCount")
        input_zero = as_int(row, "InputObservedZeroCount")
        input_candidate = as_int(row, "InputCandidateCount")
        require(input_pos == int(input_row["n_positive"]), f"{prefix}: positive count mismatch")
        require(input_zero == int(input_row["n_observed_zero"]), f"{prefix}: zero count mismatch")
        require(input_candidate == int(input_row["n_candidate"]),
                f"{prefix}: candidate count mismatch")
        require(as_int(row, "InputUnresolvedCount") == int(input_row["n_unresolved"]),
                f"{prefix}: unresolved count mismatch")
        for csv_field, manifest_field in (
            ("OriginalStatus1Count", "n_original_1"),
            ("OriginalStatus0Count", "n_original_0"),
            ("OriginalStatusNACount", "n_original_na"),
            ("FinalStatus1Count", "n_final_1"),
            ("FinalStatus0Count", "n_final_0"),
            ("FinalStatusNACount", "n_final_na"),
        ):
            require(as_int(row, csv_field) == int(input_row[manifest_field]),
                    f"{prefix}: {csv_field} mismatch")
        require(
            sum(as_int(row, field) for field in (
                "OriginalStatus1Count", "OriginalStatus0Count", "OriginalStatusNACount"
            )) == 831,
            f"{prefix}: original 1/0/NA counts do not sum to 831",
        )
        require(
            sum(as_int(row, field) for field in (
                "FinalStatus1Count", "FinalStatus0Count", "FinalStatusNACount"
            )) == 831,
            f"{prefix}: final 1/0/NA counts do not sum to 831",
        )
        require(as_int(row, "CandidatePairCount") == input_candidate,
                f"{prefix}: candidate diagnostic mismatch")
        require(as_int(row, "ObservedZeroPoolSize") == input_zero,
                f"{prefix}: observed-zero diagnostic mismatch")
        require(as_int(row, "EligiblePoolSize") == input_zero,
                f"{prefix}: eligible pool is not the explicit-zero pool")
        require(as_int(row, "FullNegativePoolSize") == input_zero,
                f"{prefix}: negative pool escaped observed zeroes")

        total_pos = as_int(row, "TotalLinks")
        train_pos = as_int(row, "TrainLinks")
        test_pos = as_int(row, "TestLinks")
        requested = as_int(row, "RequestedNegativeCount")
        selected = as_int(row, "SelectedNegativeCount")
        train_neg = as_int(row, "TrainNegativeCount")
        test_neg = as_int(row, "TestNegativeCount")
        require(total_pos == input_pos == train_pos + test_pos, f"{prefix}: positive split mismatch")
        require(requested == selected == 2 * total_pos, f"{prefix}: total 2:1 balance failed")
        require(train_neg == 2 * train_pos, f"{prefix}: TRAIN 2:1 balance failed")
        require(test_neg == 2 * test_pos, f"{prefix}: TEST 2:1 balance failed")
        require(as_int(row, "EligibleShortfall") == 0, f"{prefix}: eligible shortfall")
        require(as_int(row, "FullPoolShortfall") == 0, f"{prefix}: full-pool shortfall")
        require(as_int(row, "RandomTopupCount") == 0, f"{prefix}: fallback/top-up used")
        require(as_int(row, "TrainTestNegativeOverlapCount") == 0,
                f"{prefix}: negative TRAIN/TEST overlap")
        require(as_int(row, "PositiveTrainTestOverlapCount") == 0,
                f"{prefix}: positive TRAIN/TEST overlap")

        for kind, expected in (("TrainNegative", train_neg), ("TestNegative", test_neg),
                               ("TestPositive", test_pos)):
            visibility_total = sum(
                as_int(row, f"{kind}{suffix}")
                for suffix in (
                    "BothEndpointsVisible",
                    "OneEndpointVisible",
                    "NeitherEndpointVisible",
                )
            )
            require(visibility_total == expected, f"{prefix}: {kind} visibility total mismatch")

        require(math.isfinite(float(row["ROC_AUC"])) and math.isfinite(float(row["PR_AUC"])),
                f"{prefix}: non-finite predictive metric without a justified failure")
        split_hash = row["PositiveSplitHash"]
        require(re.fullmatch(r"[0-9a-f]{64}", split_hash) is not None,
                f"{prefix}: invalid positive split hash")
        hashes_by_ratio[as_float(row, "TrainRatio")].add(split_hash)

    for ratio in ratios:
        ratio_rows = [row for row in rows if as_float(row, "TrainRatio") == ratio]
        reported_unique = {as_int(row, "UniquePositiveSplitCount") for row in ratio_rows}
        actual_unique = len(hashes_by_ratio[ratio])
        require(reported_unique == {actual_unique},
                f"{path.name}: ratio {ratio:g} reported unique split count "
                "does not match hashes")


def validate(root: Path, sacct_file: Path | None, skip_sacct: bool) -> None:
    manifest = parse_manifest(root / "RUN_MANIFEST.txt")
    for key, expected in REQUIRED_MANIFEST.items():
        require(manifest.get(key) == expected,
                f"Manifest {key}: expected {expected}, got {manifest.get(key)}")

    input_copy = root / "INPUT_NETWORKS.csv"
    require(input_copy.is_file(), f"Missing copied input manifest: {input_copy}")
    require(sha256_file(input_copy) == manifest["InputManifestSHA256"],
            "Copied input manifest SHA-256 mismatch")
    with input_copy.open(newline="", encoding="utf-8") as handle:
        input_rows = list(csv.DictReader(handle))
    input_by_name = {row["Foodweb"]: row for row in input_rows}
    require(len(input_by_name) == len(input_rows), "INPUT_NETWORKS.csv has duplicate Foodweb values")

    expected_networks = int(manifest["ExpectedPredictionCSVs"])
    require(len(input_rows) == expected_networks,
            f"Expected {expected_networks} input rows, found {len(input_rows)}")
    experiments = int(manifest["NumExperiments"])
    ratios = expected_ratios(manifest)
    regime = manifest["DataRegime"]

    csvs = sorted((root / "prediction_scores_logs").glob("*.csv"))
    logs = sorted((root / "terminal_logs").glob("*.txt"))
    markers = sorted((root / "completion_markers").glob("*.complete"))
    require(len(csvs) == expected_networks, f"Expected {expected_networks} CSVs, found {len(csvs)}")
    require(len(logs) == int(manifest["ExpectedTerminalLogs"]),
            f"Expected {manifest['ExpectedTerminalLogs']} terminal logs, found {len(logs)}")
    require(len(markers) == int(manifest["ExpectedCompletionMarkers"]),
            f"Expected {manifest['ExpectedCompletionMarkers']} markers, found {len(markers)}")

    observed_names: set[str] = set()
    for path in csvs:
        require(path.name.endswith(RESULT_SUFFIX), f"Unexpected result filename: {path.name}")
        name = path.name[:-len(RESULT_SUFFIX)]
        require(name in input_by_name, f"Unexpected result network: {name}")
        observed_names.add(name)
        validate_result_csv(path, input_by_name[name], regime, experiments, ratios)
    require(observed_names == set(input_by_name), "Result CSV coverage differs from input manifest")
    require({p.stem for p in markers} == set(input_by_name), "Completion marker coverage mismatch")
    for marker in markers:
        values = parse_key_values(marker)
        require(values.get("Foodweb") == marker.stem, f"{marker.name}: Foodweb mismatch")
        require(values.get("InputMatSHA256") == input_by_name[marker.stem]["mat_sha256"],
                f"{marker.name}: input MAT hash mismatch")

    expected_protocol_records = experiments * len(ratios)
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        protocol_records = re.findall(
            r"\[NegativeProtocol\] eligibility=observed_zero ratio=2 "
            r"strategy=uniform_without_replacement topup_policy=error",
            text,
        )
        require(len(protocol_records) == expected_protocol_records,
                f"{path.name}: expected {expected_protocol_records} observed-zero records, "
                f"found {len(protocol_records)}")
        require("random_topup=0" in text, f"{path.name}: missing zero-top-up evidence")
        require("Falling back" not in text and "top-up" not in text,
                f"{path.name}: fallback warning detected")

    if skip_sacct:
        require(sacct_file is None, "Use either --sacct-file or --skip-sacct, not both")
    else:
        require(sacct_file is not None,
                "A sacct export is required for acceptance (or pass --skip-sacct for provisional validation)")
        validate_sacct(sacct_file, manifest["JobID"], expected_networks)

    print(
        "VALIDATION_OK "
        f"root={root} networks={expected_networks} experiments={experiments} "
        f"ratios={sorted(ratios)} sacct={'skipped' if skip_sacct else 'verified'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sacct-file",
        type=Path,
        help="Output of: sacct -j JOBID --format=JobIDRaw,State,ExitCode -P -n",
    )
    group.add_argument(
        "--skip-sacct",
        action="store_true",
        help="Provisional content validation only; not sufficient for campaign acceptance.",
    )
    args = parser.parse_args()
    validate(args.root.resolve(), args.sacct_file, args.skip_sacct)


if __name__ == "__main__":
    main()
