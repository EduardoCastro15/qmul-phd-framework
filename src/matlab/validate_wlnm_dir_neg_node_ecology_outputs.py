#!/usr/bin/env python3
"""Validate a fixed 160-dimensional WLNM_dir_neg_node_ecology run root."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


VERSION = "WLNM_dir_neg_node_ecology"
RESULT_SUFFIX = "_results_random_wlnm_dir_neg_node_ecology.csv"
TEXT_PROTOCOL = {
    "Version": VERSION,
    "ThresholdMode": "fixed",
    "NegativeEligibilityMode": "role_only",
    "NegativeSamplingStrategy": "uniform_without_replacement",
    "NegativeTopupPolicy": "uniform_remaining_nonlinks",
    "MassPreprocessingProtocol": "ln_train_subgraph_median_mean_std_v1",
    "TargetMaskingProtocol": "binary_and_weighted_candidate_mask_v1",
    "NodeAlignmentProtocol": "canonical_node_order_node_major_padding0_v1",
    "NegativeRoleSource": "train_positive_adjacency_v1",
}
REQUIRED_COLUMNS = {
    *TEXT_PROTOCOL,
    "NodeEcologyFeatureDim",
    "ModelInputDim",
    "ValidMassNodeCount",
    "MissingMassNodeCount",
    "TrainingNodeCount",
    "NegativeSplitHash",
    "K",
    "TrainRatio",
    "ExperimentID",
    "NegativePositiveRatio",
    "MassPoolSize",
    "ROC_AUC",
    "PR_AUC",
    "Threshold",
    "Precision",
    "Recall",
    "F1Score",
    "TestTP",
    "TestFP",
    "TestFN",
    "TestTN",
}
FINITE_COLUMNS = {
    "ROC_AUC", "PR_AUC", "Threshold", "Precision", "Recall", "F1Score",
    "TestTP", "TestFP", "TestFN", "TestTN", "ValidMassNodeCount",
    "MissingMassNodeCount", "TrainingNodeCount",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_manifest(path: Path) -> dict[str, str]:
    require(path.is_file(), f"Missing manifest: {path}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def expected_ratios(raw: str) -> set[float]:
    if ":" not in raw:
        return {round(100.0 * float(raw), 10)}
    start, step, end = (float(value) for value in raw.split(":"))
    values: set[float] = set()
    current = start
    while current <= end + 1e-12:
        values.add(round(100.0 * current, 10))
        current += step
    return values


def foodweb_from_path(path: Path) -> str:
    require(path.name.endswith(RESULT_SUFFIX), f"Unexpected result filename: {path.name}")
    return path.name[: -len(RESULT_SUFFIX)]


def validate_csv(path: Path, rows_expected: int, ratios: set[float], experiments: int) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"Missing header: {path}")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        require(not missing, f"{path.name}: missing columns {sorted(missing)}")
        rows = list(reader)

    require(len(rows) == rows_expected, f"{path.name}: expected {rows_expected} rows, found {len(rows)}")
    expected_keys = {(ratio, experiment) for ratio in ratios for experiment in range(1, experiments + 1)}
    observed_keys: set[tuple[float, int]] = set()
    for line, row in enumerate(rows, start=2):
        for column, expected in TEXT_PROTOCOL.items():
            require(row[column] == expected, f"{path.name}:{line}: {column}={row[column]!r}")
        require(int(float(row["NodeEcologyFeatureDim"])) == 70, f"{path.name}:{line}: NodeEcologyFeatureDim is not 70")
        require(int(float(row["ModelInputDim"])) == 160, f"{path.name}:{line}: ModelInputDim is not 160")
        require(int(float(row["K"])) == 10, f"{path.name}:{line}: K is not 10")
        require(abs(float(row["Threshold"]) - 0.50) < 1e-12, f"{path.name}:{line}: threshold is not 0.50")
        require(abs(float(row["NegativePositiveRatio"]) - 2.0) < 1e-12, f"{path.name}:{line}: negative ratio is not 2")
        require(abs(float(row["MassPoolSize"])) < 1e-12, f"{path.name}:{line}: mass eligibility was active")
        require(SHA256_RE.fullmatch(row["NegativeSplitHash"]) is not None, f"{path.name}:{line}: invalid NegativeSplitHash")
        invalid = [column for column in FINITE_COLUMNS if not finite(row[column])]
        require(not invalid, f"{path.name}:{line}: non-finite values {sorted(invalid)}")
        valid_mass = int(float(row["ValidMassNodeCount"]))
        missing_mass = int(float(row["MissingMassNodeCount"]))
        training_nodes = int(float(row["TrainingNodeCount"]))
        require(valid_mass >= 0 and missing_mass >= 0, f"{path.name}:{line}: negative mass counts")
        require(0 < training_nodes <= valid_mass + missing_mass, f"{path.name}:{line}: invalid TrainingNodeCount")
        key = (round(float(row["TrainRatio"]), 10), int(float(row["ExperimentID"])))
        require(key not in observed_keys, f"{path.name}:{line}: duplicate key {key}")
        observed_keys.add(key)
    require(observed_keys == expected_keys, f"{path.name}: incomplete ratio/experiment keys")
    return len(rows)


def validate(root: Path) -> None:
    manifest = parse_manifest(root / "RUN_MANIFEST.txt")
    run_mode = manifest.get("RunMode", "")
    require(run_mode in {"smoke", "full"}, f"Unexpected RunMode={run_mode!r}")
    required_manifest = {
        "Version": VERSION,
        "NodeEcologyFeatureDim": "70",
        "ModelInputDim": "160",
        "TrainingEpochs": "200",
        "MassPreprocessingProtocol": "ln_train_subgraph_median_mean_std_v1",
        "TargetMaskingProtocol": "binary_and_weighted_candidate_mask_v1",
        "NodeAlignmentProtocol": "canonical_node_order_node_major_padding0_v1",
        "NegativeRoleSource": "train_positive_adjacency_v1",
        "SubgraphK": "10",
        "ClassificationThreshold": "0.50",
        "ThresholdMode": "fixed",
        "ThresholdSweep": "false",
        "NegativeEligibilityMode": "role_only",
        "NegativePositiveRatio": "2",
        "NegativeSamplingStrategy": "uniform_without_replacement",
        "NegativeTopupPolicy": "uniform_remaining_nonlinks",
        "NegativeMassEligibilityEnabled": "false",
        "SubmissionState": "HOLD",
    }
    for key, expected in required_manifest.items():
        require(manifest.get(key) == expected, f"Manifest {key}: expected {expected}, got {manifest.get(key)}")

    expected_csvs = int(manifest["ExpectedPredictionCSVs"])
    expected_logs = int(manifest["ExpectedTerminalLogs"])
    expected_markers = int(manifest["ExpectedCompletionMarkers"])
    expected_rows = int(manifest["ExpectedRowsPerCSV"])
    experiments = int(manifest["NumExperiments"])
    ratios = expected_ratios(manifest["TrainRatioRange"])
    expected_models = int(manifest["ExpectedModels"])

    require(expected_csvs == (1 if run_mode == "smoke" else 290), "Unexpected CSV count in manifest")
    require(expected_rows == (1 if run_mode == "smoke" else 900), "Unexpected rows-per-CSV in manifest")
    require(expected_models == (1 if run_mode == "smoke" else 261000), "Unexpected model count in manifest")

    csvs = sorted((root / "prediction_scores_logs").glob("*.csv"))
    logs = sorted((root / "terminal_logs").glob("*.txt"))
    markers = sorted((root / "completion_markers").glob("*.complete"))
    require(len(csvs) == expected_csvs, f"Expected {expected_csvs} CSVs, found {len(csvs)}")
    require(len(logs) == expected_logs, f"Expected {expected_logs} logs, found {len(logs)}")
    require(len(markers) == expected_markers, f"Expected {expected_markers} markers, found {len(markers)}")

    foodwebs = {foodweb_from_path(path) for path in csvs}
    if run_mode == "smoke":
        require(foodwebs == {"SF1M2_tax_mass"}, f"Smoke must contain only SF1M2_tax_mass, got {sorted(foodwebs)}")

    actual_models = sum(validate_csv(path, expected_rows, ratios, experiments) for path in csvs)
    require(actual_models == expected_models, f"Expected {expected_models} models, found {actual_models}")
    print(f"VALIDATION_OK root={root.resolve()} mode={run_mode} models={actual_models} foodwebs={len(csvs)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    validate(args.root)


if __name__ == "__main__":
    main()
