#!/usr/bin/env python3
"""Run one manifest shard of the reproducible SEAL-directed GPU campaign."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from seal_run_artifacts import (
    configuration_hash,
    file_sha256,
    is_valid_complete_run,
    run_directory,
    safe_file_stem,
)


def read_manifest(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "FoodwebIndex", "Foodweb", "ExperimentID", "Seed", "MatPath",
        "MatSHA256", "ConfigHash",
    }
    if not rows:
        raise ValueError("Manifest is empty: {}".format(path))
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("Manifest is missing columns: {}".format(", ".join(sorted(missing))))
    return rows


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    return config, configuration_hash(config)


def resolve_experiment_id(value):
    if value is not None:
        return int(value)
    slurm_value = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_value:
        return int(slurm_value)
    raise ValueError(
        "Provide --experiment-id or run as a Slurm array with SLURM_ARRAY_TASK_ID"
    )


def detect_source_commit(repo_root):
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def validate_python_environment(python_executable, device, require_cuda):
    code = (
        "import json, numpy, scipy, sklearn, networkx, torch; "
        "print(json.dumps({'python': __import__('sys').version.split()[0], "
        "'torch': torch.__version__, 'torch_cuda': torch.version.cuda, "
        "'cuda_available': torch.cuda.is_available(), "
        "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    completed = subprocess.run(
        [str(python_executable), "-c", code], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("SEAL environment validation failed:\n{}".format(completed.stderr))
    environment = json.loads(completed.stdout.strip().splitlines()[-1])
    if device == "cuda" and require_cuda and not environment["cuda_available"]:
        raise RuntimeError("CUDA is required but unavailable: {}".format(environment))
    print("[ENV] {}".format(json.dumps(environment, sort_keys=True)))
    return environment


def validate_native_library(script_dir):
    library = script_dir.parent / "pytorch_DGCNN" / "lib" / "build" / "dll" / "libgnn.so"
    if not library.is_file():
        raise FileNotFoundError(
            "Missing native DGCNN library {}. Run `make clean && make` in its lib directory.".format(
                library
            )
        )
    with open(library, "rb") as handle:
        magic = handle.read(4)
    if platform.system() == "Linux" and magic != b"\x7fELF":
        raise RuntimeError("{} is not an ELF Linux library".format(library))
    return library


def stage_mat_file(mat_path, scratch_root, expected_hash, experiment_id):
    """Copy one immutable MAT input to task-local storage and verify the copy."""
    task_token = "{}_{}".format(
        os.environ.get("SLURM_JOB_ID", "local"),
        os.environ.get("SLURM_ARRAY_TASK_ID", experiment_id),
    )
    stage_dir = Path(scratch_root) / "seal_directed_{}".format(task_token) / "mats"
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged_path = stage_dir / mat_path.name
    if staged_path.is_file() and file_sha256(staged_path) == expected_hash:
        return staged_path

    temporary_path = staged_path.with_name(
        ".{}.{}.tmp".format(staged_path.name, os.getpid())
    )
    if temporary_path.exists():
        temporary_path.unlink()
    shutil.copy2(str(mat_path), str(temporary_path))
    if file_sha256(temporary_path) != expected_hash:
        temporary_path.unlink()
        raise RuntimeError("Staged MAT checksum mismatch: {}".format(mat_path))
    os.replace(str(temporary_path), str(staged_path))
    return staged_path


def build_command(args, config, row, run_dir, source_commit, script_dir, mat_path):
    command = [
        str(args.python_executable), str(script_dir / "Main_directed.py"),
        "--data-name", row["Foodweb"],
        "--mat-folder", str(mat_path.parent),
        "--run-dir", str(run_dir),
        "--config-hash", row["ConfigHash"],
        "--source-commit", source_commit,
        "--foodweb-index", row["FoodwebIndex"],
        "--experiment-id", row["ExperimentID"],
        "--seed", row["Seed"],
        "--device", args.device,
        "--num-workers", str(args.num_workers),
        "--test-ratio", str(config["test_ratio"]),
        "--requested-train-ratio", str(config["requested_train_ratio"]),
        "--hop", str(config["hop"]),
        "--batch-size", str(config["batch_size"]),
        "--max-train-num", str(config["max_train_num"]),
        "--num-epochs", str(config["num_epochs"]),
        "--threshold", str(config["threshold"]),
        "--quiet", "--log-every", "10",
    ]
    if args.require_cuda:
        command.append("--require-cuda")
    if config.get("deterministic_algorithms", True):
        command.append("--deterministic")
    if config.get("use_attribute"):
        command.append("--use-attribute")
    if config.get("use_embedding"):
        command.append("--use-embedding")
    if not config.get("role_filter", True):
        command.append("--no-role-filter")
    if config.get("all_unknown_as_negative"):
        command.append("--all-unknown-as-negative")
    if args.num_workers == 1:
        command.append("--no-parallel")
    return command


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--experiment-id", type=int, default=None)
    parser.add_argument("--only-foodweb", action="append", default=[])
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-input-hash-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument(
        "--scratch-root", type=Path, default=None,
        help="task-local staging directory, normally Slurm's TMPDIR",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    args.manifest = args.manifest.resolve()
    args.config = args.config.resolve()
    args.result_root = args.result_root.resolve()
    args.python_executable = args.python_executable.resolve()
    if args.scratch_root is not None:
        args.scratch_root = args.scratch_root.resolve()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    experiment_id = resolve_experiment_id(args.experiment_id)

    config, config_hash = load_config(args.config)
    rows = read_manifest(args.manifest)
    manifest_hashes = {row["ConfigHash"] for row in rows}
    if manifest_hashes != {config_hash}:
        raise ValueError(
            "RUN_CONFIG hash does not match manifest: config={}, manifest={}".format(
                config_hash, sorted(manifest_hashes)
            )
        )
    rows = [row for row in rows if int(row["ExperimentID"]) == experiment_id]
    if args.only_foodweb:
        selected = set(args.only_foodweb)
        rows = [row for row in rows if row["Foodweb"] in selected]
        missing = selected.difference(row["Foodweb"] for row in rows)
        if missing:
            raise ValueError("Food webs absent from shard: {}".format(", ".join(sorted(missing))))
    if not rows:
        raise ValueError("No manifest rows selected for ExperimentID={}".format(experiment_id))

    source_commit = args.source_commit or detect_source_commit(repo_root)
    frozen_commit = config.get("source_commit", "unspecified")
    if frozen_commit not in ("", "unspecified", source_commit):
        raise ValueError(
            "Source commit mismatch: RUN_CONFIG={}, checkout={}".format(
                frozen_commit, source_commit
            )
        )
    args.result_root.mkdir(parents=True, exist_ok=True)
    log_root = args.result_root / "task_logs" / "experiment_{:03d}".format(experiment_id)
    log_root.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        validate_python_environment(args.python_executable, args.device, args.require_cuda)
        library = validate_native_library(script_dir)
        print("[NATIVE] {}".format(library))

    failures = []
    skipped = 0
    verified_mat_hashes = {}
    task_start = time.time()
    for row in rows:
        mat_path = Path(row["MatPath"])
        if not mat_path.is_file():
            failures.append((row["Foodweb"], "missing MAT {}".format(mat_path)))
            if args.fail_fast:
                break
            continue
        if not args.skip_input_hash_check:
            observed_hash = verified_mat_hashes.setdefault(str(mat_path), file_sha256(mat_path))
            if observed_hash != row["MatSHA256"]:
                failures.append((row["Foodweb"], "MAT checksum mismatch"))
                if args.fail_fast:
                    break
                continue

        run_dir = run_directory(
            args.result_root, row["Foodweb"], row["ExperimentID"], row["Seed"]
        )
        if run_dir.exists():
            if args.resume and is_valid_complete_run(run_dir, row["ConfigHash"]):
                print("[SKIP] valid completed run {}".format(run_dir))
                skipped += 1
                continue
            failures.append(
                (row["Foodweb"], "existing run is invalid or resume was not requested: {}".format(run_dir))
            )
            if args.fail_fast:
                break
            continue

        command_mat_path = mat_path
        if args.scratch_root is not None:
            try:
                command_mat_path = stage_mat_file(
                    mat_path, args.scratch_root, row["MatSHA256"], experiment_id
                )
            except Exception as error:
                failures.append((row["Foodweb"], "input staging failed: {}".format(error)))
                if args.fail_fast:
                    break
                continue
        command = build_command(
            args, config, row, run_dir, source_commit, script_dir, command_mat_path
        )
        log_path = log_root / "{}_seed_{}.log".format(
            safe_file_stem(row["Foodweb"]), row["Seed"]
        )
        print(
            "[RUN] experiment={} foodweb_index={} seed={} foodweb={!r}".format(
                row["ExperimentID"], row["FoodwebIndex"], row["Seed"], row["Foodweb"]
            )
        )
        if args.dry_run:
            print(subprocess.list2cmdline(command))
            continue
        with open(log_path, "w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command, cwd=str(script_dir), stdout=log_handle,
                stderr=subprocess.STDOUT, check=False,
                env=dict(os.environ, PYTHONHASHSEED=row["Seed"]),
            )
        if completed.returncode != 0 or not is_valid_complete_run(run_dir, row["ConfigHash"]):
            failures.append(
                (row["Foodweb"], "exit={} log={}".format(completed.returncode, log_path))
            )
            if args.fail_fast:
                break

    elapsed = time.time() - task_start
    print(
        "[SUMMARY] selected={} skipped={} failed={} elapsed_seconds={:.1f}".format(
            len(rows), skipped, len(failures), elapsed
        )
    )
    for foodweb, message in failures:
        print("[FAILED] {}: {}".format(foodweb, message), file=sys.stderr)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
