#!/usr/bin/env python3
"""Validate an A-D WLNM focal-mass pilot or full campaign root."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


VARIANTS = {
    "A_structure_only": ("structure_only", 0, 90),
    "B_endpoint_masses": ("endpoint_masses", 4, 94),
    "C_mass_ratio": ("mass_ratio", 2, 92),
    "D_endpoint_masses_ratio": ("endpoint_masses_ratio", 5, 95),
}
TEXT_PROTOCOL = {
    "Version": "WLNM_dir_neg_focal_mass",
    "MassPreprocessingProtocol": "log10_within_web_median_iqr_v1",
    "TargetMaskingProtocol": "historical_weighted_v1",
    "NegativeEligibilityMode": "role_only",
    "NegativeSamplingStrategy": "uniform_without_replacement",
    "NegativeTopupPolicy": "uniform_remaining_nonlinks",
}
REQUIRED_COLUMNS = {
    *TEXT_PROTOCOL,
    "FocalMassFeatureMode",
    "FocalMassFeatureDim",
    "ModelInputDim",
    "ValidMassNodeCount",
    "MissingMassNodeCount",
    "MassLog10Median",
    "MassLog10IQR",
    "TrainPairsBothMassValid",
    "TestPairsBothMassValid",
    "PositiveSplitHash",
    "NegativeSplitHash",
    "NegativePositiveRatio",
    "K",
    "TrainRatio",
    "ExperimentID",
    "ThresholdMode",
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


def foodweb_from_path(path: Path) -> str:
    suffix = "_results_random_wlnm_dir_neg_focal_mass.csv"
    require(path.name.endswith(suffix), f"Unexpected result filename: {path.name}")
    return path.name[: -len(suffix)]


def validate_variant(root: Path, variant: str) -> dict[tuple[str, float, int], tuple[str, str]]:
    mode, feature_dim, model_dim = VARIANTS[variant]
    manifest = parse_manifest(root / "RUN_MANIFEST.txt")
    expected_manifest = {
        "Variant": variant,
        "Version": "WLNM_dir_neg_focal_mass",
        "FocalMassFeatureMode": mode,
        "FocalMassFeatureDim": str(feature_dim),
        "ModelInputDim": str(model_dim),
        "MassPreprocessingProtocol": "log10_within_web_median_iqr_v1",
        "TargetMaskingProtocol": "historical_weighted_v1",
        "EcosystemTypeModelFeature": "false",
        "NegativeEligibilityMode": "role_only",
        "NegativePositiveRatio": "2",
        "NegativeSamplingStrategy": "uniform_without_replacement",
        "NegativeTopupPolicy": "uniform_remaining_nonlinks",
        "NegativeMassEligibilityEnabled": "false",
    }
    for key, expected in expected_manifest.items():
        require(manifest.get(key) == expected, f"{variant} manifest {key}: {manifest.get(key)!r}")

    expected_csvs = int(manifest["ExpectedPredictionCSVs"])
    expected_rows = int(manifest["ExpectedRowsPerCSV"])
    expected_logs = int(manifest["ExpectedTerminalLogs"])
    expected_markers = int(manifest["ExpectedCompletionMarkers"])
    csvs = sorted((root / "prediction_scores_logs").glob("*.csv"))
    logs = sorted((root / "terminal_logs").glob("*.txt"))
    markers = sorted((root / "completion_markers").glob("*.complete"))
    require(len(csvs) == expected_csvs, f"{variant}: expected {expected_csvs} CSVs, found {len(csvs)}")
    require(len(logs) == expected_logs, f"{variant}: expected {expected_logs} logs, found {len(logs)}")
    require(len(markers) == expected_markers, f"{variant}: expected {expected_markers} markers, found {len(markers)}")

    split_hashes: dict[tuple[str, float, int], tuple[str, str]] = {}
    predictive = ("ROC_AUC", "PR_AUC", "Threshold", "Precision", "Recall", "F1Score", "TestTP", "TestFP", "TestFN", "TestTN")
    mass_numeric = ("ValidMassNodeCount", "MissingMassNodeCount", "MassLog10Median", "MassLog10IQR", "TrainPairsBothMassValid", "TestPairsBothMassValid")
    for path in csvs:
        foodweb = foodweb_from_path(path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames is not None, f"Missing header: {path}")
            missing = REQUIRED_COLUMNS - set(reader.fieldnames)
            require(not missing, f"{path.name}: missing {sorted(missing)}")
            rows = list(reader)
        require(len(rows) == expected_rows, f"{path.name}: expected {expected_rows} rows, found {len(rows)}")
        for line, row in enumerate(rows, start=2):
            for column, expected in TEXT_PROTOCOL.items():
                require(row[column] == expected, f"{path.name}:{line}: {column}={row[column]!r}")
            require(row["FocalMassFeatureMode"] == mode, f"{path.name}:{line}: wrong mode")
            require(int(float(row["FocalMassFeatureDim"])) == feature_dim, f"{path.name}:{line}: wrong feature dim")
            require(int(float(row["ModelInputDim"])) == model_dim, f"{path.name}:{line}: wrong model input dim")
            require(abs(float(row["NegativePositiveRatio"]) - 2.0) < 1e-12, f"{path.name}:{line}: wrong negative ratio")
            require(int(float(row["K"])) == 10, f"{path.name}:{line}: K is not 10")
            require(row["ThresholdMode"].lower() == "fixed", f"{path.name}:{line}: threshold mode is not fixed")
            require(abs(float(row["Threshold"]) - 0.50) < 1e-12, f"{path.name}:{line}: threshold is not 0.50")
            require(all(finite(row[column]) for column in predictive + mass_numeric), f"{path.name}:{line}: non-finite metric or mass diagnostic")
            positive_hash = row["PositiveSplitHash"]
            negative_hash = row["NegativeSplitHash"]
            require(len(positive_hash) == 64 and len(negative_hash) == 64, f"{path.name}:{line}: invalid split hash")
            key = (foodweb, round(float(row["TrainRatio"]), 10), int(float(row["ExperimentID"])))
            require(key not in split_hashes, f"{path.name}:{line}: duplicate key {key}")
            split_hashes[key] = (positive_hash, negative_hash)
    return split_hashes


def validate(parent: Path) -> None:
    parent_manifest = parse_manifest(parent / "ABLATION_MANIFEST.txt")
    require(parent_manifest.get("Version") == "WLNM_dir_neg_focal_mass", "Wrong parent Version")
    require(parent_manifest.get("SubmissionState") == "HOLD", "Campaign was not declared HOLD")

    hashes: dict[str, dict[tuple[str, float, int], tuple[str, str]]] = {}
    for variant in VARIANTS:
        hashes[variant] = validate_variant(parent / variant, variant)

    reference = hashes["A_structure_only"]
    for variant, observed in hashes.items():
        require(set(observed) == set(reference), f"{variant}: train-ratio/experiment keys differ from A")
        mismatches = [key for key in reference if observed[key] != reference[key]]
        require(not mismatches, f"{variant}: split hashes differ from A, first={mismatches[:3]}")

    expected_models = int(parent_manifest["ExpectedModels"])
    actual_models = sum(len(value) for value in hashes.values())
    require(actual_models == expected_models, f"Expected {expected_models} models, found {actual_models}")
    print(f"VALIDATION_OK root={parent.resolve()} models={actual_models} paired_keys={len(reference)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_root", type=Path)
    args = parser.parse_args()
    validate(args.parent_root)


if __name__ == "__main__":
    main()
