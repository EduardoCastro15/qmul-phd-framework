#!/usr/bin/env python3
"""Validate and merge a complete SEAL-directed GPU campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_foodweb_seal_directed import load_config, read_manifest
from seal_run_artifacts import run_directory, safe_file_stem, validate_run_bundle


PREDICTIVE_METRICS = (
    "ROC_AUC", "PR_AUC", "Precision", "Recall", "F1Score", "TestMCC", "TestTSS"
)


def _pair_set(array):
    return {tuple(int(value) for value in row) for row in np.asarray(array).reshape(-1, 2)}


def validate_artifact_contents(run_dir, row, metrics):
    errors = []
    with np.load(Path(run_dir) / "artifacts.npz", allow_pickle=False) as artifact:
        pairs = {
            name: _pair_set(artifact[name])
            for name in ("train_pos", "train_neg", "test_pos", "test_neg")
        }
        if len(pairs["train_pos"]) != len(artifact["train_pos"]):
            errors.append("duplicate train positives")
        if len(pairs["train_neg"]) != len(artifact["train_neg"]):
            errors.append("duplicate train negatives")
        if len(pairs["test_pos"]) != len(artifact["test_pos"]):
            errors.append("duplicate test positives")
        if len(pairs["test_neg"]) != len(artifact["test_neg"]):
            errors.append("duplicate test negatives")
        if pairs["train_pos"] & pairs["test_pos"]:
            errors.append("train/test positive overlap")
        if pairs["train_neg"] & pairs["test_neg"]:
            errors.append("train/test negative overlap")
        positives = pairs["train_pos"] | pairs["test_pos"]
        negatives = pairs["train_neg"] | pairs["test_neg"]
        if positives & negatives:
            errors.append("positive/negative overlap")
        if len(pairs["train_pos"]) != len(pairs["train_neg"]):
            errors.append("train negative:positive ratio is not 1:1")
        if len(pairs["test_pos"]) != len(pairs["test_neg"]):
            errors.append("test negative:positive ratio is not 1:1")
        expected_test = len(pairs["test_pos"]) + len(pairs["test_neg"])
        if artifact["test_labels"].size != expected_test:
            errors.append("test label count does not match test pairs")

    if metrics.get("Foodweb") != row["Foodweb"]:
        errors.append("Foodweb identifier mismatch")
    if int(metrics.get("ExperimentID", -1)) != int(row["ExperimentID"]):
        errors.append("ExperimentID mismatch")
    if int(metrics.get("Seed", -1)) != int(row["Seed"]):
        errors.append("Seed mismatch")
    if int(metrics.get("FoodwebIndex", -1)) != int(row["FoodwebIndex"]):
        errors.append("FoodwebIndex mismatch")
    for metric in PREDICTIVE_METRICS:
        value = metrics.get(metric)
        if value is None or not math.isfinite(float(value)):
            errors.append("{} is missing/non-finite".format(metric))
            continue
        lower, upper = (-1.0, 1.0) if metric in ("TestMCC", "TestTSS") else (0.0, 1.0)
        if not lower <= float(value) <= upper:
            errors.append("{} outside [{}, {}]".format(metric, lower, upper))
    return errors


def check_slurm(job_id):
    completed = subprocess.run(
        [
            "sacct", "-j", str(job_id), "--parsable2", "--noheader",
            "--format=JobIDRaw,State,ExitCode",
        ],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("sacct failed: {}".format(completed.stderr.strip()))
    task_states = {}
    prefix = "{}_".format(job_id)
    for line in completed.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 3 or not fields[0].startswith(prefix):
            continue
        suffix = fields[0][len(prefix):]
        if suffix.isdigit():
            task_states[int(suffix)] = (fields[1], fields[2])
    bad = {
        task: state for task, state in task_states.items()
        if state[0] != "COMPLETED" or state[1] != "0:0"
    }
    if bad:
        raise ValueError("Non-completed Slurm array tasks: {}".format(bad))
    return task_states


def _atomic_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".{}.".format(path.name), dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_summaries(result_root, rows):
    summary_root = Path(result_root) / "summary"
    preferred = [
        "FoodwebIndex", "Foodweb", "ExperimentID", "Seed", "Version",
        "RequestedTrainRatio", "RealizedTrainRatio", "Threshold", *PREDICTIVE_METRICS,
    ]
    all_fields = sorted({key for row in rows for key in row})
    fields = preferred + [field for field in all_fields if field not in preferred]
    _atomic_csv(summary_root / "raw_runs.csv", rows, fields)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["Foodweb"]].append(row)
    means = []
    for foodweb, foodweb_rows in sorted(grouped.items(), key=lambda item: int(item[1][0]["FoodwebIndex"])):
        output = {
            "FoodwebIndex": int(foodweb_rows[0]["FoodwebIndex"]),
            "Foodweb": foodweb,
            "Runs": len(foodweb_rows),
            "Aggregation": "mean_of_all_raw_runs_no_tukey",
        }
        for metric in PREDICTIVE_METRICS:
            output[metric] = statistics.fmean(float(row[metric]) for row in foodweb_rows)
        means.append(output)
    _atomic_csv(
        summary_root / "foodweb_metric_means_raw.csv",
        means,
        ["FoodwebIndex", "Foodweb", "Runs", "Aggregation", *PREDICTIVE_METRICS],
    )

    compatibility_root = Path(result_root) / "prediction_scores_logs"
    for foodweb, foodweb_rows in grouped.items():
        ordered = sorted(foodweb_rows, key=lambda row: int(row["ExperimentID"]))
        _atomic_csv(
            compatibility_root / "{}_results_SEAL_directed.csv".format(safe_file_stem(foodweb)),
            ordered,
            fields,
        )

    fallback_rows = [row for row in rows if int(row.get("RolePoolFallbackUsed", 0)) == 1]
    counts = Counter(row["Foodweb"] for row in fallback_rows)
    fallback_summary = [
        {"Foodweb": foodweb, "FallbackRuns": count}
        for foodweb, count in sorted(counts.items())
    ]
    _atomic_csv(
        summary_root / "negative_pool_fallback_summary.csv",
        fallback_summary,
        ["Foodweb", "FallbackRuns"],
    )
    return means, fallback_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--job-id", action="append", default=[],
        help="Slurm job ID to verify; repeat for calibration and full-array jobs",
    )
    parser.add_argument("--write-summaries", action="store_true")
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config, config_hash = load_config(args.config)
    manifest = read_manifest(args.manifest)
    expected = int(config["num_foodwebs"]) * int(config["num_experiments"])
    errors = []
    if len(manifest) != expected:
        errors.append("manifest rows: expected {}, found {}".format(expected, len(manifest)))
    seeds = [int(row["Seed"]) for row in manifest]
    if len(seeds) != len(set(seeds)):
        errors.append("manifest seeds are not unique")

    validated_metrics = []
    environment_fingerprints = set()
    missing = []
    for index, row in enumerate(manifest, start=1):
        run_dir = run_directory(
            args.result_root, row["Foodweb"], row["ExperimentID"], row["Seed"]
        )
        if not run_dir.exists():
            missing.append(str(run_dir))
            continue
        try:
            payload = validate_run_bundle(
                run_dir,
                expected_config_hash=config_hash,
                verify_checksums=not args.skip_checksums,
            )
            run_errors = validate_artifact_contents(run_dir, row, payload["metrics"])
            provenance = payload.get("provenance", {})
            frozen_commit = config.get("source_commit", "unspecified")
            if frozen_commit not in ("", "unspecified", provenance.get("source_commit")):
                run_errors.append("source commit does not match RUN_CONFIG")
            expected_deterministic = int(config.get("deterministic_algorithms", True))
            if int(payload["metrics"].get("DeterministicAlgorithms", -1)) != expected_deterministic:
                run_errors.append("determinism mode does not match RUN_CONFIG")
            if args.require_cuda and (
                payload["metrics"].get("Device") != "cuda:0"
                or not provenance.get("cuda_available")
            ):
                run_errors.append("run was not executed on required CUDA device")
            environment_fingerprints.add((
                provenance.get("python_version"),
                provenance.get("torch_version"),
                provenance.get("torch_cuda_version"),
            ))
            if run_errors:
                errors.append("{}: {}".format(run_dir, "; ".join(run_errors)))
            else:
                validated_metrics.append(payload["metrics"])
        except Exception as error:
            errors.append("{}: {}".format(run_dir, error))
        if index % 1000 == 0:
            print("[PROGRESS] validated {}/{} manifest rows".format(index, len(manifest)))

    if missing:
        errors.append("missing run directories: {} (first: {})".format(len(missing), missing[0]))
    if len(environment_fingerprints) > 1:
        errors.append(
            "multiple Python/PyTorch/CUDA environments detected: {}".format(
                sorted(environment_fingerprints)
            )
        )
    slurm_states = {}
    for job_id in args.job_id:
        try:
            for task, state in check_slurm(job_id).items():
                slurm_states["{}_{}".format(job_id, task)] = state
        except Exception as error:
            errors.append("Slurm validation for {}: {}".format(job_id, error))

    if errors:
        print("[INVALID] {} problem(s)".format(len(errors)))
        for error in errors[:50]:
            print("[ERROR] {}".format(error))
        if len(errors) > 50:
            print("[ERROR] ... {} additional problems".format(len(errors) - 50))
        raise SystemExit(1)

    fallback_rows = []
    if args.write_summaries:
        _, fallback_rows = write_summaries(args.result_root, validated_metrics)
    report = {
        "status": "VALIDATION_OK",
        "config_hash": config_hash,
        "manifest_rows": len(manifest),
        "validated_runs": len(validated_metrics),
        "foodwebs": len({row["Foodweb"] for row in validated_metrics}),
        "experiments": len({int(row["ExperimentID"]) for row in validated_metrics}),
        "unique_seeds": len({int(row["Seed"]) for row in validated_metrics}),
        "fallback_runs": sum(
            int(row.get("RolePoolFallbackUsed", 0)) for row in validated_metrics
        ),
        "slurm_tasks_checked": len(slurm_states),
        "summaries_written": bool(args.write_summaries),
    }
    if args.write_summaries:
        report_path = Path(args.result_root) / "summary" / "RUN_VALIDATION.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
