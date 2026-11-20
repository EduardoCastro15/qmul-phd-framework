#!/usr/bin/env python3
"""Validate WLNM_dir_neg_node_ecology_kfold repeated-CV outputs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


VERSION = "WLNM_dir_neg_node_ecology_kfold"
NODE_PROTOCOL_VERSION = "WLNM_dir_neg_node_ecology"
PARTITION_PROTOCOL = "repeated_balanced_positive_edges_v1"
PRAUC_PROTOCOL = "average_precision_noninterpolated_v1"
RESULT_RE = re.compile(
    r"^(?P<foodweb>.+)_results_random_"
    r"wlnm_dir_neg_node_ecology_kfold_cvK(?P<cvk>3|5|10)\.csv$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PREDICTIVE_COLUMNS = {
    "ROC_AUC",
    "PR_AUC",
    "F1Score",
    "TestMCC",
    "Precision",
    "Recall",
    "TestTSS",
}
REQUIRED_COLUMNS = {
    "Version",
    "K",
    "TrainRatio",
    "ExperimentID",
    "Seed",
    "ThresholdMode",
    "Threshold",
    "CvK",
    "FoldID",
    "NumFolds",
    "CvSplitSeed",
    "CvPartitionHash",
    "CvPartitionProtocol",
    "PositiveSplitHash",
    "NegativeSplitHash",
    "NodeEcologyProtocolVersion",
    "PRAUCProtocol",
    "NodeEcologyFeatureDim",
    "ModelInputDim",
    "MassPreprocessingProtocol",
    "NodeAlignmentProtocol",
    "TargetMaskingProtocol",
    "NegativeRoleSource",
    "ValidMassNodeCount",
    "MissingMassNodeCount",
    "TrainingNodeCount",
    "NegativeEligibilityMode",
    "NegativePositiveRatio",
    "NegativeSamplingStrategy",
    "NegativeTopupPolicy",
    "MassPoolSize",
    "TrainTestNegativeOverlapCount",
    "PositiveTrainTestOverlapCount",
    "TotalLinks",
    "TrainLinks",
    "TestLinks",
    *PREDICTIVE_COLUMNS,
}


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


def as_int(value: str) -> int:
    number = float(value)
    require(math.isfinite(number) and number.is_integer(), f"Not an integer: {value!r}")
    return int(number)


def validate_manifest(root: Path) -> dict[str, str]:
    manifest = parse_manifest(root / "RUN_MANIFEST.txt")
    mode = manifest.get("RunMode", "")
    require(mode in {"smoke", "calibration", "full"}, f"Unexpected RunMode={mode!r}")
    expected = {
        "Version": VERSION,
        "NodeEcologyProtocolVersion": NODE_PROTOCOL_VERSION,
        "CvPartitionProtocol": PARTITION_PROTOCOL,
        "RepeatedFoldPartitions": "true",
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
        "PRAUCProtocol": PRAUC_PROTOCOL,
        "NegativeEligibilityMode": "role_only",
        "NegativePositiveRatio": "2",
        "NegativeSamplingStrategy": "uniform_without_replacement",
        "NegativeTopupPolicy": "uniform_remaining_nonlinks",
        "NegativeMassEligibilityEnabled": "false",
        "ComputeEcologicalMetrics": "false",
        "SubmissionState": "HOLD",
    }
    for key, expected_value in expected.items():
        require(
            manifest.get(key) == expected_value,
            f"Manifest {key}: expected {expected_value!r}, got {manifest.get(key)!r}",
        )

    mode_expectations = {
        "smoke": {
            "FoodWebs": 1,
            "NumExperimentsPerCvK": 1,
            "ExpectedModels": 3,
            "ExpectedPredictionCSVs": 1,
            "ExpectedTerminalLogs": 1,
            "ExpectedCompletionMarkers": 1,
            "ExpectedDataRowsCvK3": 3,
            "ExpectedDataRowsCvK5": 0,
            "ExpectedDataRowsCvK10": 0,
        },
        "calibration": {
            "FoodWebs": 1,
            "NumExperimentsPerCvK": 2,
            "ExpectedModels": 36,
            "ExpectedPredictionCSVs": 3,
            "ExpectedTerminalLogs": 3,
            "ExpectedCompletionMarkers": 1,
            "ExpectedDataRowsCvK3": 6,
            "ExpectedDataRowsCvK5": 10,
            "ExpectedDataRowsCvK10": 20,
        },
        "full": {
            "FoodWebs": 290,
            "NumExperimentsPerCvK": 100,
            "ExpectedModels": 522000,
            "ExpectedPredictionCSVs": 870,
            "ExpectedTerminalLogs": 870,
            "ExpectedCompletionMarkers": 290,
            "ExpectedDataRowsCvK3": 300,
            "ExpectedDataRowsCvK5": 500,
            "ExpectedDataRowsCvK10": 1000,
        },
    }[mode]
    for key, expected_value in mode_expectations.items():
        require(as_int(manifest[key]) == expected_value, f"Manifest {key} mismatch")
    return manifest


def validate_result_csv(path: Path, cvk: int, experiments: int) -> tuple[str, int]:
    match = RESULT_RE.fullmatch(path.name)
    require(match is not None, f"Unexpected result filename: {path.name}")
    require(int(match.group("cvk")) == cvk, f"Filename CvK mismatch: {path.name}")
    foodweb = match.group("foodweb")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"Missing header: {path}")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        require(not missing, f"{path.name}: missing columns {sorted(missing)}")
        rows = list(reader)

    require(
        len(rows) == experiments * cvk,
        f"{path.name}: expected {experiments * cvk} rows, found {len(rows)}",
    )
    by_experiment: dict[int, list[dict[str, str]]] = defaultdict(list)
    model_seeds: set[int] = set()
    for line, row in enumerate(rows, start=2):
        prefix = f"{path.name}:{line}"
        require(row["Version"] == VERSION, f"{prefix}: Version={row['Version']!r}")
        require(row["NodeEcologyProtocolVersion"] == NODE_PROTOCOL_VERSION, f"{prefix}: node protocol")
        require(row["PRAUCProtocol"] == PRAUC_PROTOCOL, f"{prefix}: PR-AUC protocol")
        require(row["CvPartitionProtocol"] == PARTITION_PROTOCOL, f"{prefix}: partition protocol")
        require(row["ThresholdMode"] == "fixed", f"{prefix}: threshold mode")
        require(row["NegativeEligibilityMode"] == "role_only", f"{prefix}: eligibility")
        require(row["NegativeSamplingStrategy"] == "uniform_without_replacement", f"{prefix}: sampling")
        require(row["NegativeTopupPolicy"] == "uniform_remaining_nonlinks", f"{prefix}: top-up")
        require(as_int(row["K"]) == 10, f"{prefix}: K")
        require(as_int(row["CvK"]) == cvk, f"{prefix}: CvK")
        require(as_int(row["NumFolds"]) == cvk, f"{prefix}: NumFolds")
        require(abs(float(row["TrainRatio"]) - 100.0 * (cvk - 1) / cvk) < 1e-8, f"{prefix}: TrainRatio")
        require(abs(float(row["Threshold"]) - 0.50) < 1e-12, f"{prefix}: threshold")
        require(abs(float(row["NegativePositiveRatio"]) - 2.0) < 1e-12, f"{prefix}: negative ratio")
        require(abs(float(row["MassPoolSize"])) < 1e-12, f"{prefix}: mass eligibility active")
        require(as_int(row["NodeEcologyFeatureDim"]) == 70, f"{prefix}: feature dim")
        require(as_int(row["ModelInputDim"]) == 160, f"{prefix}: model input dim")
        require(row["MassPreprocessingProtocol"] == "ln_train_subgraph_median_mean_std_v1", f"{prefix}: mass protocol")
        require(row["NodeAlignmentProtocol"] == "canonical_node_order_node_major_padding0_v1", f"{prefix}: alignment")
        require(row["TargetMaskingProtocol"] == "binary_and_weighted_candidate_mask_v1", f"{prefix}: masking")
        require(row["NegativeRoleSource"] == "train_positive_adjacency_v1", f"{prefix}: role source")
        require(SHA256_RE.fullmatch(row["CvPartitionHash"]) is not None, f"{prefix}: partition hash")
        require(SHA256_RE.fullmatch(row["PositiveSplitHash"]) is not None, f"{prefix}: positive hash")
        require(SHA256_RE.fullmatch(row["NegativeSplitHash"]) is not None, f"{prefix}: negative hash")
        require(float(row["TrainTestNegativeOverlapCount"]) == 0, f"{prefix}: negative overlap")
        require(float(row["PositiveTrainTestOverlapCount"]) == 0, f"{prefix}: positive overlap")
        require(as_int(row["TrainLinks"]) + as_int(row["TestLinks"]) == as_int(row["TotalLinks"]), f"{prefix}: link coverage")
        require(0 < as_int(row["TrainingNodeCount"]) <= as_int(row["ValidMassNodeCount"]) + as_int(row["MissingMassNodeCount"]), f"{prefix}: training nodes")
        for metric in PREDICTIVE_COLUMNS:
            require(finite(row[metric]), f"{prefix}: non-finite {metric}")
            value = float(row[metric])
            lower = -1.0 if metric in {"TestMCC", "TestTSS"} else 0.0
            require(lower <= value <= 1.0, f"{prefix}: {metric}={value}")

        experiment = as_int(row["ExperimentID"])
        require(1 <= experiment <= experiments, f"{prefix}: ExperimentID")
        fold = as_int(row["FoldID"])
        require(1 <= fold <= cvk, f"{prefix}: FoldID")
        seed = as_int(row["Seed"])
        require(seed not in model_seeds, f"{prefix}: duplicate model seed")
        model_seeds.add(seed)
        by_experiment[experiment].append(row)

    require(set(by_experiment) == set(range(1, experiments + 1)), f"{path.name}: missing experiments")
    partition_hashes: set[str] = set()
    partition_seeds: set[int] = set()
    for experiment, experiment_rows in by_experiment.items():
        folds = {as_int(row["FoldID"]) for row in experiment_rows}
        require(folds == set(range(1, cvk + 1)), f"{path.name}: repeat {experiment} incomplete folds")
        hashes = {row["CvPartitionHash"] for row in experiment_rows}
        seeds = {as_int(row["CvSplitSeed"]) for row in experiment_rows}
        positive_hashes = {row["PositiveSplitHash"] for row in experiment_rows}
        require(len(hashes) == 1 and len(seeds) == 1, f"{path.name}: repeat {experiment} partition metadata mismatch")
        require(
            len(positive_hashes) == cvk,
            f"{path.name}: repeat {experiment} contains duplicate fold splits",
        )
        partition_hashes.update(hashes)
        partition_seeds.update(seeds)
    require(len(partition_hashes) == experiments, f"{path.name}: repeated partitions are not unique")
    require(len(partition_seeds) == experiments, f"{path.name}: repeated split seeds are not unique")
    return foodweb, len(rows)


def validate(root: Path) -> None:
    manifest = validate_manifest(root)
    mode = manifest["RunMode"]
    experiments = as_int(manifest["NumExperimentsPerCvK"])
    expected_csvs = as_int(manifest["ExpectedPredictionCSVs"])
    expected_logs = as_int(manifest["ExpectedTerminalLogs"])
    expected_markers = as_int(manifest["ExpectedCompletionMarkers"])
    expected_models = as_int(manifest["ExpectedModels"])

    csvs = sorted((root / "prediction_scores_logs").glob("*.csv"))
    logs = sorted((root / "terminal_logs").glob("*.txt"))
    markers = sorted((root / "completion_markers").glob("*.complete"))
    require(len(csvs) == expected_csvs, f"Expected {expected_csvs} CSVs, found {len(csvs)}")
    require(len(logs) == expected_logs, f"Expected {expected_logs} logs, found {len(logs)}")
    require(len(markers) == expected_markers, f"Expected {expected_markers} markers, found {len(markers)}")

    foodweb_cvks: dict[str, set[int]] = defaultdict(set)
    actual_models = 0
    for path in csvs:
        match = RESULT_RE.fullmatch(path.name)
        require(match is not None, f"Unexpected result filename: {path.name}")
        cvk = int(match.group("cvk"))
        expected_rows = as_int(manifest[f"ExpectedDataRowsCvK{cvk}"])
        require(expected_rows == experiments * cvk, f"Manifest rows mismatch for CvK={cvk}")
        foodweb, rows = validate_result_csv(path, cvk, experiments)
        foodweb_cvks[foodweb].add(cvk)
        actual_models += rows

    expected_cvks = {3} if mode == "smoke" else {3, 5, 10}
    require(all(cvks == expected_cvks for cvks in foodweb_cvks.values()), "Incomplete CvK coverage")
    require(len(foodweb_cvks) == as_int(manifest["FoodWebs"]), "Food-web coverage mismatch")
    if mode == "smoke":
        require(set(foodweb_cvks) == {"SF1M2_tax_mass"}, "Unexpected smoke food web")
    if mode == "calibration":
        require(set(foodweb_cvks) == {"Weddell Sea_tax_mass"}, "Unexpected calibration food web")
    require(actual_models == expected_models, f"Expected {expected_models} models, found {actual_models}")
    print(
        f"VALIDATION_OK root={root.resolve()} mode={mode} "
        f"models={actual_models} foodwebs={len(foodweb_cvks)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    validate(args.root)


if __name__ == "__main__":
    main()
