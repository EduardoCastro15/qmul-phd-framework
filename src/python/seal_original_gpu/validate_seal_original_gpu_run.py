#!/usr/bin/env python3
"""Validate and summarize a SEAL original GPU campaign without correcting it."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from generate_seal_original_manifest import experiment_seed
from run_foodweb_seal_original_gpu import load_config, read_manifest
from seal_original_run_artifacts import run_directory, safe_file_stem, validate_run_bundle


PREDICTIVE_METRICS = (
    "ROC_AUC", "PR_AUC", "Precision", "Recall", "F1Score", "TestMCC", "TestTSS"
)


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
    main_state = None
    task_states = {}
    job_id = str(job_id)
    prefix = "{}_".format(job_id)
    for line in completed.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 3:
            continue
        sacct_id = fields[0]
        if sacct_id == job_id:
            main_state = (fields[1], fields[2])
        elif sacct_id.startswith(prefix):
            suffix = sacct_id[len(prefix):]
            if suffix.isdigit():
                task_states[int(suffix)] = (fields[1], fields[2])
    if not task_states:
        if main_state is None:
            raise ValueError("No top-level Slurm records found for job {}".format(job_id))
        task_states["job"] = main_state
    bad = {
        task: state for task, state in task_states.items()
        if state[0] != "COMPLETED" or state[1] != "0:0"
    }
    if bad:
        raise ValueError("Non-completed Slurm tasks: {}".format(bad))
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


def validate_manifest_protocol(manifest, config):
    errors = []
    expected_rows = int(config["num_foodwebs"]) * int(config["num_experiments"])
    if len(manifest) != expected_rows:
        errors.append("manifest rows: expected {}, found {}".format(expected_rows, len(manifest)))
    keys = [(row["Foodweb"], int(row["ExperimentID"])) for row in manifest]
    if len(keys) != len(set(keys)):
        errors.append("duplicate Foodweb/ExperimentID manifest rows")

    foodwebs = {row["Foodweb"] for row in manifest}
    if len(foodwebs) != int(config["num_foodwebs"]):
        errors.append("unexpected number of food webs")
    for experiment_id in range(1, int(config["num_experiments"]) + 1):
        rows = [row for row in manifest if int(row["ExperimentID"]) == experiment_id]
        expected_seed = experiment_seed(experiment_id, config["base_seed"])
        if len(rows) != int(config["num_foodwebs"]):
            errors.append("ExperimentID {} has {} rows".format(experiment_id, len(rows)))
        if {int(row["Seed"]) for row in rows} != {expected_seed}:
            errors.append("ExperimentID {} does not reuse historical seed {}".format(
                experiment_id, expected_seed
            ))
    expected_distinct = int(config["num_experiments"])
    observed_distinct = len({int(row["Seed"]) for row in manifest})
    if observed_distinct != expected_distinct:
        errors.append("expected {} distinct historical seeds, found {}".format(
            expected_distinct, observed_distinct
        ))
    return errors


def validate_result(run_dir, row, config, bundle, require_cuda):
    errors = []
    result = bundle["result"]
    metadata = bundle["metadata"]
    expected = {
        "Version": config["version"],
        "ExperimentID": int(row["ExperimentID"]),
        "Seed": int(row["Seed"]),
        "K": int(config["hop"]),
        "ThresholdMode": "fixed",
    }
    if result.get("Version") != expected["Version"]:
        errors.append("Version mismatch")
    for field in ("ExperimentID", "Seed", "K"):
        try:
            observed = int(float(result.get(field, "nan")))
        except (TypeError, ValueError, OverflowError):
            errors.append("{} is invalid".format(field))
        else:
            if observed != expected[field]:
                errors.append("{} mismatch".format(field))
    if result.get("ThresholdMode") != "fixed":
        errors.append("ThresholdMode mismatch")
    try:
        if abs(float(result["Threshold"]) - float(config["threshold"])) > 1e-12:
            errors.append("Threshold mismatch")
    except (KeyError, TypeError, ValueError):
        errors.append("Threshold is invalid")

    if metadata.get("Foodweb") != row["Foodweb"]:
        errors.append("Foodweb metadata mismatch")
    if int(metadata.get("ExperimentID", -1)) != int(row["ExperimentID"]):
        errors.append("ExperimentID metadata mismatch")
    if int(metadata.get("Seed", -1)) != int(row["Seed"]):
        errors.append("Seed metadata mismatch")
    if metadata.get("mat_sha256") != row["MatSHA256"]:
        errors.append("MAT checksum metadata mismatch")
    if metadata.get("core_source_sha256") != config.get("core_source_sha256"):
        errors.append("frozen core-source hashes mismatch")

    command = metadata.get("command", [])
    forbidden = ("--use-attribute", "--use-embedding", "--all-unknown-as-negative", "--no-parallel")
    for flag in forbidden:
        if flag in command:
            errors.append("historical default changed by {}".format(flag))
    if metadata.get("device_requested") == "cuda" and "--no-cuda" in command:
        errors.append("CUDA run included --no-cuda")

    environment = metadata.get("environment") or {}
    if require_cuda:
        if metadata.get("device_requested") != "cuda":
            errors.append("run did not request CUDA")
        if not environment.get("cuda_available"):
            errors.append("CUDA was unavailable")
        terminal_text = (Path(run_dir) / "terminal.log").read_text(
            encoding="utf-8", errors="replace"
        )
        if "cuda=True" not in terminal_text:
            errors.append("Main.py did not report cuda=True")

    nonfinite = []
    for metric in PREDICTIVE_METRICS:
        try:
            value = float(result[metric])
        except (KeyError, TypeError, ValueError):
            errors.append("{} is missing/invalid".format(metric))
            continue
        if not math.isfinite(value):
            nonfinite.append(metric)
            continue
        lower, upper = (-1.0, 1.0) if metric in ("TestMCC", "TestTSS") else (0.0, 1.0)
        if not lower <= value <= upper:
            errors.append("{} outside [{}, {}]".format(metric, lower, upper))
    return errors, nonfinite


def write_summaries(result_root, validated_rows):
    summary_root = Path(result_root) / "summary"
    preferred = ["FoodwebIndex", "Foodweb", "ExperimentID", "Seed", "Version", *PREDICTIVE_METRICS]
    all_fields = sorted({key for row in validated_rows for key in row})
    fields = preferred + [field for field in all_fields if field not in preferred]
    _atomic_csv(summary_root / "raw_runs.csv", validated_rows, fields)

    grouped = defaultdict(list)
    for row in validated_rows:
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
            values = [float(row[metric]) for row in foodweb_rows if math.isfinite(float(row[metric]))]
            output[metric] = statistics.fmean(values) if values else "NaN"
        means.append(output)
    _atomic_csv(
        summary_root / "foodweb_metric_means_raw.csv",
        means,
        ["FoodwebIndex", "Foodweb", "Runs", "Aggregation", *PREDICTIVE_METRICS],
    )

    compatibility_root = Path(result_root) / "prediction_scores_logs"
    for foodweb, foodweb_rows in grouped.items():
        ordered = sorted(foodweb_rows, key=lambda row: int(row["ExperimentID"]))
        original_fields = [
            field for field in fields if field not in ("FoodwebIndex", "Foodweb")
        ]
        compatibility_rows = []
        for index, row in enumerate(ordered, start=1):
            output = dict(row)
            output["Iteration"] = index
            compatibility_rows.append(output)
        _atomic_csv(
            compatibility_root / "{}_results_SEAL.csv".format(safe_file_stem(foodweb)),
            compatibility_rows,
            original_fields,
        )
    return means


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--write-summaries", action="store_true")
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config, config_hash = load_config(args.config)
    manifest = read_manifest(args.manifest)
    errors = validate_manifest_protocol(manifest, config)
    validated_rows = []
    environment_fingerprints = set()
    nonfinite_metrics = 0
    missing = []

    for index, row in enumerate(manifest, start=1):
        run_dir = run_directory(args.result_root, row["Foodweb"], row["ExperimentID"], row["Seed"])
        if not run_dir.exists():
            missing.append(str(run_dir))
            continue
        try:
            bundle = validate_run_bundle(
                run_dir,
                expected_config_hash=config_hash,
                verify_checksums=not args.skip_checksums,
            )
            run_errors, nonfinite = validate_result(
                run_dir, row, config, bundle, args.require_cuda
            )
            metadata = bundle["metadata"]
            frozen_commit = config.get("source_commit", "unspecified")
            if frozen_commit not in ("", "unspecified", metadata.get("source_commit")):
                run_errors.append("source commit does not match RUN_CONFIG")
            environment = metadata.get("environment") or {}
            environment_fingerprints.add((
                environment.get("python"), environment.get("torch"),
                environment.get("torch_cuda"), environment.get("gensim"),
            ))
            if run_errors:
                errors.append("{}: {}".format(run_dir, "; ".join(run_errors)))
            else:
                output = dict(bundle["result"])
                output["FoodwebIndex"] = int(row["FoodwebIndex"])
                output["Foodweb"] = row["Foodweb"]
                validated_rows.append(output)
                nonfinite_metrics += len(nonfinite)
        except Exception as error:
            errors.append("{}: {}".format(run_dir, error))
        if index % 1000 == 0:
            print("[PROGRESS] validated {}/{} manifest rows".format(index, len(manifest)))

    if missing:
        errors.append("missing run directories: {} (first: {})".format(len(missing), missing[0]))
    if len(environment_fingerprints) > 1:
        errors.append("multiple Python/PyTorch/CUDA environments detected: {}".format(
            sorted(environment_fingerprints)
        ))

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

    if args.write_summaries:
        write_summaries(args.result_root, validated_rows)
    report = {
        "status": "VALIDATION_OK",
        "config_hash": config_hash,
        "manifest_rows": len(manifest),
        "validated_runs": len(validated_rows),
        "foodwebs": len({row["Foodweb"] for row in validated_rows}),
        "experiments": len({int(row["ExperimentID"]) for row in validated_rows}),
        "distinct_numeric_seeds": len({int(row["Seed"]) for row in validated_rows}),
        "seed_reuse_policy": "same seed reused across all food webs within each experiment",
        "nonfinite_predictive_metrics": nonfinite_metrics,
        "slurm_tasks_checked": len(slurm_states),
        "summaries_written": bool(args.write_summaries),
        "core_logic_policy": config["core_logic_policy"],
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
