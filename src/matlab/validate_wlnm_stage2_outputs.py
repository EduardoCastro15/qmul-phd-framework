#!/usr/bin/env python3
"""Validate the Stage 2 WLNM_original and WLNM_dir_neg_kfold roots."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


METRICS = (
    "ROC_AUC",
    "PR_AUC",
    "TestF1Score",
    "TestMCC",
    "TestPrecision",
    "TestRecall",
    "TestTSS",
)


def parse_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_common_csv(path: Path, expected_version: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"Missing header: {path}")
        required = {
            "Version",
            "K",
            "TrainRatio",
            "ExperimentID",
            "Seed",
            "ThresholdMode",
            "Threshold",
            *METRICS,
        }
        missing = required - set(reader.fieldnames)
        require(not missing, f"{path.name}: missing columns {sorted(missing)}")
        rows = list(reader)

    require(rows, f"No data rows: {path}")
    require(
        all(row["Version"] == expected_version for row in rows),
        f"{path.name}: unexpected Version",
    )
    require(
        all(abs(float(row["Threshold"]) - 0.5) < 1e-12 for row in rows),
        f"{path.name}: expected only Threshold=0.5",
    )
    require(
        all(int(float(row["K"])) == 10 for row in rows),
        f"{path.name}: expected subgraph K=10",
    )
    require(
        all(row["ThresholdMode"].lower() == "fixed" for row in rows),
        f"{path.name}: expected ThresholdMode=fixed",
    )
    invalid_metrics = [
        (index + 2, metric)
        for index, row in enumerate(rows)
        for metric in METRICS
        if not finite_number(row[metric])
    ]
    require(
        not invalid_metrics,
        f"{path.name}: invalid metrics, first={invalid_metrics[:5]}",
    )
    return rows


def validate_original(root: Path, manifest: dict[str, str]) -> None:
    expected_csvs = int(manifest["ExpectedPredictionCSVs"])
    expected_logs = int(manifest["ExpectedTerminalLogs"])
    expected_rows = int(manifest["ExpectedDataRowsPerCSV"])
    expected_pool_limited = int(manifest["ExpectedNegativePoolLimitedFoodWebs"])
    base_seed = int(manifest["BaseSeed"])

    csvs = sorted((root / "prediction_scores_logs").glob("*.csv"))
    logs = sorted((root / "terminal_logs").glob("*.txt"))
    require(len(csvs) == expected_csvs, f"Expected {expected_csvs} CSVs, found {len(csvs)}")
    require(len(logs) == expected_logs, f"Expected {expected_logs} logs, found {len(logs)}")

    for path in csvs:
        rows = validate_common_csv(path, "WLNM_original")
        suffix = "_results_random_wlnm_original.csv"
        require(path.name.endswith(suffix), f"Unexpected original filename: {path.name}")
        foodweb = path.name[: -len(suffix)]
        require(len(rows) == expected_rows, f"{path.name}: expected {expected_rows} rows, found {len(rows)}")
        require(
            {int(row["ExperimentID"]) for row in rows} == set(range(1, expected_rows + 1)),
            f"{path.name}: incomplete ExperimentID set",
        )
        require(
            all(abs(float(row["TrainRatio"]) - 90.0) < 1e-12 for row in rows),
            f"{path.name}: expected TrainRatio=90",
        )
        name_hash = sum(ord(character) for character in foodweb)
        for row in rows:
            experiment = int(row["ExperimentID"])
            raw_seed = (
                base_seed
                + 1_000_003 * name_hash
                + 1_009 * 10
                + 9_173 * 900
                + experiment
            )
            expected_seed = round(raw_seed) % 2_147_483_646 + 1
            require(
                int(float(row["Seed"])) == expected_seed,
                f"{path.name}: incorrect seed for ExperimentID={experiment}",
            )

    shortfall_logs = 0
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        records = re.findall(r"\[NegPool\] mode=undirected_uniform\b", text)
        require(
            len(records) == expected_rows,
            f"{path.name}: expected {expected_rows} NegPool records, found {len(records)}",
        )
        if re.search(r"full_pool_shortfall=[1-9][0-9]*", text):
            shortfall_logs += 1

    require(
        shortfall_logs == expected_pool_limited,
        (
            "Expected "
            f"{expected_pool_limited} food webs with a capped negative pool, "
            f"found {shortfall_logs}"
        ),
    )
    print(f"Original roots with a capped negative pool: {shortfall_logs}/{expected_logs}")


def validate_kfold(root: Path, manifest: dict[str, str]) -> None:
    expected_csvs = int(manifest["ExpectedPredictionCSVs"])
    expected_logs = int(manifest["ExpectedTerminalLogs"])
    experiments = int(manifest["NumExperimentsPerFold"])
    cv_seed = int(manifest["CvSeed"])
    expected_tau = float(manifest["TauMass"])
    expected_by_k = {
        3: int(manifest["ExpectedDataRowsCvK3"]),
        5: int(manifest["ExpectedDataRowsCvK5"]),
        10: int(manifest["ExpectedDataRowsCvK10"]),
    }

    csvs = sorted((root / "prediction_scores_logs").glob("*.csv"))
    logs = sorted((root / "terminal_logs").glob("*.txt"))
    require(len(csvs) == expected_csvs, f"Expected {expected_csvs} CSVs, found {len(csvs)}")
    require(len(logs) == expected_logs, f"Expected {expected_logs} logs, found {len(logs)}")

    for path in csvs:
        match = re.search(r"_cvK(3|5|10)\.csv$", path.name)
        require(match is not None, f"Cannot resolve CvK from {path.name}")
        cv_k = int(match.group(1))
        expected_rows = expected_by_k[cv_k]
        require(expected_rows > 0, f"Unexpected CvK={cv_k} output: {path.name}")

        rows = validate_common_csv(path, "WLNM_dir_neg_kfold")
        require(len(rows) == expected_rows, f"{path.name}: expected {expected_rows} rows, found {len(rows)}")
        require(all(int(row["CvK"]) == cv_k for row in rows), f"{path.name}: incorrect CvK values")
        expected_train_ratio = 100.0 * (cv_k - 1) / cv_k
        require(
            all(abs(float(row["TrainRatio"]) - expected_train_ratio) < 1e-9 for row in rows),
            f"{path.name}: incorrect TrainRatio for CvK={cv_k}",
        )
        expected_keys = {
            (fold, experiment)
            for fold in range(1, cv_k + 1)
            for experiment in range(1, experiments + 1)
        }
        actual_keys = {
            (int(row["FoldID"]), int(row["ExperimentID"]))
            for row in rows
        }
        require(actual_keys == expected_keys, f"{path.name}: incomplete fold/experiment keys")
        require(
            all(
                int(float(row["Seed"]))
                == cv_seed + 1_000 * int(row["FoldID"]) + int(row["ExperimentID"])
                for row in rows
            ),
            f"{path.name}: incorrect deterministic seeds",
        )

    for path in logs:
        match = re.search(r"_cvK(3|5|10)_terminal_log\.txt$", path.name)
        require(match is not None, f"Cannot resolve CvK from {path.name}")
        cv_k = int(match.group(1))
        expected_records = expected_by_k[cv_k]
        text = path.read_text(encoding="utf-8", errors="replace")
        negpool_records = re.findall(
            r"\[NegPool\].*eligibility=role_or_mass",
            text,
        )
        mass_records = re.findall(
            r"\[NegMassPref\].*priority_sampling=0.*threshold=([0-9.]+)",
            text,
        )
        require(
            len(negpool_records) == expected_records,
            f"{path.name}: expected {expected_records} role-or-mass records, found {len(negpool_records)}",
        )
        require(
            len(mass_records) == expected_records
            and all(abs(float(value) - expected_tau) < 1e-12 for value in mass_records),
            (
                f"{path.name}: expected {expected_records} tau={expected_tau:.2f} "
                f"records, found {len(mass_records)}"
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = root / "RUN_MANIFEST.txt"
    require(manifest_path.is_file(), f"Missing manifest: {manifest_path}")
    manifest = parse_manifest(manifest_path)
    version = manifest.get("Version")

    if version == "WLNM_original":
        validate_original(root, manifest)
    elif version == "WLNM_dir_neg_kfold":
        validate_kfold(root, manifest)
    else:
        raise RuntimeError(f"Unsupported Version={version!r}")

    print(f"VALIDATION_OK version={version} root={root}")


if __name__ == "__main__":
    main()
