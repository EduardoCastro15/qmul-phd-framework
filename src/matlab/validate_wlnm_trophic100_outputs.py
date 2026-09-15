#!/usr/bin/env python3
"""Validate the isolated PlotB/Ythan 10%-90% WLNM_dir_neg sweep output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SMOKE_TARGETS = {
    "Ythan Estuary_tax_mass",
    "Dutch Microfauna food web PlotB_tax_mass",
}
RESULT_SUFFIX = "_results_random_wlnm_dir_neg.csv"
EXPECTED_RATIOS = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_manifest(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()

    manifest = parse_manifest(root / "RUN_MANIFEST.txt")
    run_mode = manifest["RunMode"]
    require(run_mode in {"smoke", "full"}, f"Unexpected RunMode={run_mode}")
    expected_runs = int(manifest["NumExperiments"])
    require(expected_runs in {1, 100}, f"Unexpected NumExperiments={expected_runs}")
    require(manifest["Version"] == "WLNM_dir_neg", "Unexpected version")
    require(manifest["Eligibility"] == "role_only", "Expected role_only")
    require(manifest["MassEligibilityEnabled"].lower() == "false", "Mass filter enabled")
    require(manifest["TrophicLevelProtocol"] == "validated_v2", "Expected validated_v2")
    require(manifest["SweepTrainRatios"].lower() == "true", "Expected train-ratio sweep")
    require({float(value) for value in manifest["TrainRatioRange"].split(",")} == EXPECTED_RATIOS,
            "Expected train ratios 10%-90%")

    expected_foodwebs = 2 if run_mode == "smoke" else 290
    csvs = sorted((root / "prediction_scores_logs").glob("*.csv"))
    logs = sorted((root / "terminal_logs").glob("*.txt"))
    markers = sorted((root / "completion_markers").glob("*.complete"))
    snapshots = sorted((root / "ecological_snapshots").glob("*.mat"))
    require(len(csvs) == expected_foodwebs,
            f"Expected {expected_foodwebs} result CSVs, found {len(csvs)}")
    require(len(logs) == expected_foodwebs,
            f"Expected {expected_foodwebs} terminal logs, found {len(logs)}")
    require(len(markers) == expected_foodwebs,
            f"Expected {expected_foodwebs} completion markers, found {len(markers)}")
    expected_rows = expected_runs * len(EXPECTED_RATIOS)
    expected_snapshots = expected_foodwebs * expected_rows
    require(len(snapshots) == expected_snapshots,
            f"Expected {expected_snapshots} snapshots, found {len(snapshots)}")

    observed = set()
    for path in csvs:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == expected_rows,
                f"{path.name}: expected {expected_rows} rows, found {len(rows)}")
        require(path.name.endswith(RESULT_SUFFIX), f"Unexpected result filename: {path.name}")
        name = path.name[:-len(RESULT_SUFFIX)]
        observed.add(name)
        require({float(row["TrainRatio"]) for row in rows} == EXPECTED_RATIOS,
                f"{path.name}: incomplete train-ratio set")
        for ratio in EXPECTED_RATIOS:
            ratio_rows = [row for row in rows if float(row["TrainRatio"]) == ratio]
            require(len(ratio_rows) == expected_runs,
                    f"{path.name}: ratio {ratio:g} expected {expected_runs} rows, found {len(ratio_rows)}")
            require({int(float(row["ExperimentID"])) for row in ratio_rows}
                    == set(range(1, expected_runs + 1)),
                    f"{path.name}: ratio {ratio:g} has incomplete ExperimentID set")
        for row in rows:
            require(row["Version"] == "WLNM_dir_neg", f"{path.name}: wrong version")
            require(abs(float(row["Threshold"]) - 0.5) < 1e-12, f"{path.name}: wrong threshold")
            require(row["TrophicLevelProtocol"] == "validated_v2", f"{path.name}: wrong trophic protocol")
            require(row["NegativeEligibilityMode"] == "role_only", f"{path.name}: wrong negative mode")
            require(float(row["MassPoolSize"]) == 0, f"{path.name}: mass pool is active")
            require(Path(row["TrophicSnapshotFile"]).is_file(), f"{path.name}: missing snapshot")

    if run_mode == "smoke":
        require(observed == SMOKE_TARGETS, f"Unexpected food webs: {sorted(observed)}")
    else:
        require(len(observed) == 290, f"Expected 290 unique food webs, found {len(observed)}")
    print(f"VALIDATION_OK foodwebs={expected_foodwebs} runs_per_ratio={expected_runs} root={root}")


if __name__ == "__main__":
    main()
