#!/usr/bin/env python3
"""Validate a frozen WLNM_dir_neg role-only result root."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


REQUIRED_PROTOCOL = {
    "Version": "WLNM_dir_neg",
    "Eligibility": "role_only",
    "MassEligibilityEnabled": "false",
    "NegativeSampling": "uniform_without_replacement",
    "TargetNegativePositiveRatio": "2",
    "NegativeTopupPolicy": "uniform_remaining_nonlinks",
    "ClassificationThreshold": "0.50",
    "ThresholdMode": "fixed",
    "ThresholdSweep": "false",
    "CheckConnectivity": "false",
    "AdaptiveConnectivity": "false",
}

REQUIRED_CSV_COLUMNS = {
    "Version",
    "TrainRatio",
    "ExperimentID",
    "Seed",
    "ThresholdMode",
    "Threshold",
    "NegativeEligibilityMode",
    "NegativePositiveRatio",
    "NegativeSamplingStrategy",
    "NegativeTopupPolicy",
    "RolePoolSize",
    "EligiblePoolSize",
    "FullNegativePoolSize",
    "RequestedNegativeCount",
    "SelectedNegativeCount",
    "EligibleNegativeCount",
    "EligibleShortfall",
    "RandomTopupCount",
    "TopupProportion",
    "TrainNegativeCount",
    "TestNegativeCount",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_manifest(path: Path) -> dict[str, str]:
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


def validate_csv(path: Path, expected_rows: int, expected_ratios: set[float], experiments: int) -> bool:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"Missing header: {path}")
        missing = REQUIRED_CSV_COLUMNS - set(reader.fieldnames)
        require(not missing, f"{path.name}: missing columns {sorted(missing)}")
        rows = list(reader)

    require(len(rows) == expected_rows, f"{path.name}: expected {expected_rows} rows, found {len(rows)}")
    require(all(row["Version"] == "WLNM_dir_neg" for row in rows), f"{path.name}: wrong Version")
    require(
        all(row["NegativeEligibilityMode"] == "role_only" for row in rows),
        f"{path.name}: expected role_only eligibility",
    )
    require(
        all(abs(float(row["NegativePositiveRatio"]) - 2.0) < 1e-12 for row in rows),
        f"{path.name}: expected negative-positive ratio 2",
    )
    require(
        all(row["NegativeSamplingStrategy"] == "uniform_without_replacement" for row in rows),
        f"{path.name}: wrong sampling strategy",
    )
    require(
        all(row["NegativeTopupPolicy"] == "uniform_remaining_nonlinks" for row in rows),
        f"{path.name}: wrong top-up policy",
    )
    require(
        all(row["ThresholdMode"].lower() == "fixed" and abs(float(row["Threshold"]) - 0.5) < 1e-12 for row in rows),
        f"{path.name}: expected only fixed threshold 0.5",
    )

    actual_keys = {
        (round(float(row["TrainRatio"]), 10), int(float(row["ExperimentID"])))
        for row in rows
    }
    expected_keys = {
        (round(ratio, 10), experiment)
        for ratio in expected_ratios
        for experiment in range(1, experiments + 1)
    }
    require(actual_keys == expected_keys, f"{path.name}: incomplete train-ratio/experiment pairs")

    numeric_columns = REQUIRED_CSV_COLUMNS - {
        "Version",
        "ThresholdMode",
        "NegativeEligibilityMode",
        "NegativeSamplingStrategy",
        "NegativeTopupPolicy",
    }
    invalid = [
        (index + 2, column)
        for index, row in enumerate(rows)
        for column in numeric_columns
        if not finite(row[column])
    ]
    require(not invalid, f"{path.name}: invalid sampler values, first={invalid[:5]}")

    for index, row in enumerate(rows, start=2):
        selected = int(float(row["SelectedNegativeCount"]))
        eligible = int(float(row["EligibleNegativeCount"]))
        topup = int(float(row["RandomTopupCount"]))
        train_neg = int(float(row["TrainNegativeCount"]))
        test_neg = int(float(row["TestNegativeCount"]))
        require(selected == eligible + topup, f"{path.name}:{index}: selected != eligible + top-up")
        require(selected == train_neg + test_neg, f"{path.name}:{index}: selected != train + test negatives")
        expected_topup_proportion = topup / max(1, selected)
        require(
            abs(float(row["TopupProportion"]) - expected_topup_proportion) < 1e-10,
            f"{path.name}:{index}: incorrect top-up proportion",
        )

    return any(int(float(row["RandomTopupCount"])) > 0 for row in rows)


def parse_train_ratios(raw: str) -> set[float]:
    if ":" not in raw:
        return {100.0 * float(raw)}
    start, step, end = (float(value) for value in raw.split(":"))
    ratios: set[float] = set()
    value = start
    while value <= end + 1e-12:
        ratios.add(round(100.0 * value, 10))
        value += step
    return ratios


def validate(root: Path) -> None:
    manifest_path = root / "RUN_MANIFEST.txt"
    require(manifest_path.is_file(), f"Missing manifest: {manifest_path}")
    manifest = parse_manifest(manifest_path)

    for key, expected in REQUIRED_PROTOCOL.items():
        require(manifest.get(key) == expected, f"Manifest {key}: expected {expected}, got {manifest.get(key)}")

    expected_csvs = int(manifest["ExpectedPredictionCSVs"])
    expected_logs = int(manifest["ExpectedTerminalLogs"])
    expected_markers = int(manifest["ExpectedCompletionMarkers"])
    expected_rows = int(manifest["ExpectedDataRowsPerCSV"])
    expected_topup_webs = int(manifest["ExpectedTopupFoodWebs"])
    experiments = int(manifest["NumExperimentsPerTrainRatio"])
    expected_ratios = parse_train_ratios(manifest["TrainRatioRange"])

    csvs = sorted((root / "prediction_scores_logs").glob("*.csv"))
    logs = sorted((root / "terminal_logs").glob("*.txt"))
    markers = sorted((root / "completion_markers").glob("*.complete"))
    require(len(csvs) == expected_csvs, f"Expected {expected_csvs} CSVs, found {len(csvs)}")
    require(len(logs) == expected_logs, f"Expected {expected_logs} logs, found {len(logs)}")
    require(len(markers) == expected_markers, f"Expected {expected_markers} markers, found {len(markers)}")

    topup_csvs = sum(validate_csv(path, expected_rows, expected_ratios, experiments) for path in csvs)
    require(topup_csvs == expected_topup_webs, f"Expected {expected_topup_webs} top-up food webs, found {topup_csvs}")

    expected_records_per_log = len(expected_ratios) * experiments
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        role_records = re.findall(r"\[NegPool\].*eligibility=role_only", text)
        mass_records = re.findall(r"\[NegPool\].*mass_filter=0", text)
        protocol_records = re.findall(
            r"\[NegativeProtocol\] eligibility=role_only ratio=2 strategy=uniform_without_replacement "
            r"topup_policy=uniform_remaining_nonlinks",
            text,
        )
        require(len(role_records) == expected_records_per_log, f"{path.name}: wrong role-only record count")
        require(len(mass_records) == expected_records_per_log, f"{path.name}: mass filtering was not disabled")
        require(len(protocol_records) == expected_records_per_log, f"{path.name}: wrong protocol record count")

    print(
        "VALIDATION_OK "
        f"root={root.resolve()} csvs={len(csvs)} rows_per_csv={expected_rows} "
        f"topup_foodwebs={topup_csvs}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Role-only result root")
    args = parser.parse_args()
    validate(args.root)


if __name__ == "__main__":
    main()
