#!/usr/bin/env python3
"""Run one manifest shard through the unchanged SEAL original Main.py."""

from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from generate_seal_original_manifest import CORE_SOURCE_FILES, experiment_seed
from seal_original_run_artifacts import (
    configuration_hash,
    file_sha256,
    finalize_run_bundle,
    is_valid_complete_run,
    preserve_failed_staging,
    read_single_result_row,
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
    raise ValueError("Provide --experiment-id or run as a Slurm array")


def detect_source_commit(repo_root):
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def validate_core_sources(repo_root, config):
    expected = config.get("core_source_sha256", {})
    errors = []
    for relative in CORE_SOURCE_FILES:
        path = Path(repo_root) / relative
        if not path.is_file():
            errors.append("missing {}".format(relative))
            continue
        observed = file_sha256(path)
        if expected.get(relative) != observed:
            errors.append("hash mismatch for {}".format(relative))
    if errors:
        raise ValueError("Frozen SEAL original core differs: {}".format("; ".join(errors)))


def validate_python_environment(python_executable, device, require_cuda):
    code = (
        "import json,sys,gensim,numpy,scipy,sklearn,networkx,torch; "
        "print(json.dumps({'python': sys.version.split()[0], "
        "'torch': torch.__version__, 'torch_cuda': torch.version.cuda, "
        "'cuda_available': torch.cuda.is_available(), "
        "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
        "'gensim': gensim.__version__, 'numpy': numpy.__version__, "
        "'scipy': scipy.__version__, 'sklearn': sklearn.__version__, "
        "'networkx': networkx.__version__}))"
    )
    completed = subprocess.run(
        [str(python_executable), "-c", code], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("SEAL environment validation failed:\n{}".format(completed.stderr))
    environment = json.loads(completed.stdout.strip().splitlines()[-1])
    if device == "cuda" and require_cuda and not environment["cuda_available"]:
        raise RuntimeError("CUDA is required but unavailable: {}".format(environment))
    print("[ENV] {}".format(json.dumps(environment, sort_keys=True)))
    return environment


def validate_native_library(python_root):
    library = Path(python_root) / "pytorch_DGCNN" / "lib" / "build" / "dll" / "libgnn.so"
    if not library.is_file():
        raise FileNotFoundError(
            "Missing native DGCNN library {}. Build it with /usr/bin/g++.".format(library)
        )
    with open(library, "rb") as handle:
        magic = handle.read(4)
    if platform.system() == "Linux" and magic != b"\x7fELF":
        raise RuntimeError("{} is not an ELF Linux library".format(library))
    ctypes.CDLL(str(library))
    return library


def stage_mat_file(mat_path, scratch_root, expected_hash, experiment_id):
    task_token = "{}_{}".format(
        os.environ.get("SLURM_JOB_ID", "local"),
        os.environ.get("SLURM_ARRAY_TASK_ID", experiment_id),
    )
    stage_dir = Path(scratch_root) / "seal_original_{}".format(task_token) / "mats"
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged_path = stage_dir / mat_path.name
    if staged_path.is_file() and file_sha256(staged_path) == expected_hash:
        return staged_path
    temporary = staged_path.with_name(".{}.{}.tmp".format(staged_path.name, os.getpid()))
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(str(mat_path), str(temporary))
    if file_sha256(temporary) != expected_hash:
        temporary.unlink()
        raise RuntimeError("Staged MAT checksum mismatch: {}".format(mat_path))
    os.replace(str(temporary), str(staged_path))
    return staged_path


def build_command(python_executable, python_root, config, row, result_dir, mat_path, device):
    """Build only arguments already supported by historical Main.py."""
    command = [
        str(python_executable), str(Path(python_root) / "Main.py"),
        "--data-name", row["Foodweb"],
        "--mat-folder", str(mat_path.parent),
        "--result-dir", str(result_dir),
        "--hop", str(config["hop"]),
        "--test-ratio", str(config["test_ratio"]),
        "--seed", row["Seed"],
        "--experiment-id", row["ExperimentID"],
        "--batch-size", str(config["batch_size"]),
        "--max-train-num", str(config["max_train_num"]),
        "--num-epochs", str(config["num_epochs"]),
        "--threshold", str(config["threshold"]),
    ]
    if device == "cpu":
        command.append("--no-cuda")
    # Deliberately do not add --no-parallel, --use-attribute, --use-embedding,
    # or --all-unknown-as-negative. Their historical defaults are preserved.
    return command


def _utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _validate_generated_row(result_row, row, config):
    problems = []
    if result_row.get("Version") != config["version"]:
        problems.append("Version")
    if int(float(result_row.get("ExperimentID", -1))) != int(row["ExperimentID"]):
        problems.append("ExperimentID")
    if int(float(result_row.get("Seed", -1))) != int(row["Seed"]):
        problems.append("Seed")
    if int(float(result_row.get("K", -1))) != int(config["hop"]):
        problems.append("K")
    if abs(float(result_row.get("Threshold", "nan")) - float(config["threshold"])) > 1e-12:
        problems.append("Threshold")
    if problems:
        raise ValueError("Generated CSV does not match frozen run: {}".format(", ".join(problems)))


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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-input-hash-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--scratch-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.manifest = args.manifest.resolve()
    args.config = args.config.resolve()
    args.result_root = args.result_root.resolve()
    args.python_executable = args.python_executable.resolve()
    if args.scratch_root is not None:
        args.scratch_root = args.scratch_root.resolve()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    python_root = repo_root / "src" / "python"
    experiment_id = resolve_experiment_id(args.experiment_id)
    config, config_hash = load_config(args.config)
    rows = read_manifest(args.manifest)
    if {row["ConfigHash"] for row in rows} != {config_hash}:
        raise ValueError("RUN_CONFIG hash does not match manifest")
    if config.get("core_logic_policy") != "exact_historical_behavior_no_corrections":
        raise ValueError("RUN_CONFIG is not the frozen historical SEAL original protocol")
    validate_core_sources(repo_root, config)

    rows = [row for row in rows if int(row["ExperimentID"]) == experiment_id]
    if args.only_foodweb:
        selected = set(args.only_foodweb)
        rows = [row for row in rows if row["Foodweb"] in selected]
        missing = selected.difference(row["Foodweb"] for row in rows)
        if missing:
            raise ValueError("Food webs absent from shard: {}".format(", ".join(sorted(missing))))
    if not rows:
        raise ValueError("No rows selected for ExperimentID={}".format(experiment_id))
    for row in rows:
        expected_seed = experiment_seed(row["ExperimentID"], config["base_seed"])
        if int(row["Seed"]) != expected_seed:
            raise ValueError("Historical seed pattern mismatch for {}".format(row["Foodweb"]))

    source_commit = args.source_commit or detect_source_commit(repo_root)
    frozen_commit = config.get("source_commit", "unspecified")
    if frozen_commit not in ("", "unspecified", source_commit):
        raise ValueError("Source commit mismatch: config={}, checkout={}".format(frozen_commit, source_commit))

    environment = None
    if not args.dry_run:
        environment = validate_python_environment(args.python_executable, args.device, args.require_cuda)
        library = validate_native_library(python_root)
        print("[NATIVE] {}".format(library))

    args.result_root.mkdir(parents=True, exist_ok=True)
    failures = []
    skipped = 0
    verified_hashes = {}
    task_start = time.time()

    for row in rows:
        mat_path = Path(row["MatPath"])
        if not mat_path.is_file():
            failures.append((row["Foodweb"], "missing MAT {}".format(mat_path)))
            if args.fail_fast:
                break
            continue
        if not args.skip_input_hash_check:
            observed = verified_hashes.setdefault(str(mat_path), file_sha256(mat_path))
            if observed != row["MatSHA256"]:
                failures.append((row["Foodweb"], "MAT checksum mismatch"))
                if args.fail_fast:
                    break
                continue

        run_dir = run_directory(args.result_root, row["Foodweb"], row["ExperimentID"], row["Seed"])
        if run_dir.exists():
            if args.resume and is_valid_complete_run(run_dir, row["ConfigHash"]):
                print("[SKIP] {}".format(run_dir))
                skipped += 1
                continue
            failures.append((row["Foodweb"], "existing run is invalid or resume was not requested"))
            if args.fail_fast:
                break
            continue

        command_mat = mat_path
        if args.scratch_root is not None:
            try:
                command_mat = stage_mat_file(
                    mat_path, args.scratch_root, row["MatSHA256"], experiment_id
                )
            except Exception as error:
                failures.append((row["Foodweb"], "input staging failed: {}".format(error)))
                if args.fail_fast:
                    break
                continue
        staging_parent = run_dir.parent
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=".{}.tmp.".format(run_dir.name), dir=str(staging_parent)))
        main_output = staging_dir / "main_output"
        main_output.mkdir()
        command = build_command(
            args.python_executable, python_root, config, row, main_output, command_mat, args.device
        )
        print("[RUN] experiment={} foodweb_index={} seed={} foodweb={!r}".format(
            row["ExperimentID"], row["FoodwebIndex"], row["Seed"], row["Foodweb"]
        ))
        if args.dry_run:
            print(subprocess.list2cmdline(command))
            shutil.rmtree(staging_dir)
            continue

        started_at = _utc_now()
        run_start = time.time()
        log_path = staging_dir / "terminal.log"
        try:
            with open(log_path, "w", encoding="utf-8") as log_handle:
                completed = subprocess.run(
                    command,
                    cwd=str(python_root),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError("Main.py exited with {}".format(completed.returncode))
            csv_files = list(main_output.glob("*_results_SEAL.csv"))
            if len(csv_files) != 1:
                raise ValueError("Expected one original result CSV, found {}".format(len(csv_files)))
            result_row, _ = read_single_result_row(csv_files[0])
            _validate_generated_row(result_row, row, config)
            os.replace(str(csv_files[0]), str(staging_dir / "result.csv"))
            main_output.rmdir()

            metadata = {
                "FoodwebIndex": int(row["FoodwebIndex"]),
                "Foodweb": row["Foodweb"],
                "ExperimentID": int(row["ExperimentID"]),
                "Seed": int(row["Seed"]),
                "mat_path": str(mat_path),
                "mat_sha256": row["MatSHA256"],
                "source_commit": source_commit,
                "core_source_sha256": config["core_source_sha256"],
                "device_requested": args.device,
                "require_cuda": bool(args.require_cuda),
                "environment": environment,
                "command": command,
                "working_directory": str(python_root),
                "started_at_utc": started_at,
                "completed_at_utc": _utc_now(),
                "elapsed_seconds": time.time() - run_start,
            }
            finalize_run_bundle(staging_dir, run_dir, metadata, row["ConfigHash"])
        except Exception as error:
            failed_dir = preserve_failed_staging(
                staging_dir, args.result_root, row["Foodweb"], row["ExperimentID"], row["Seed"]
            )
            failures.append((row["Foodweb"], "{}; preserved at {}".format(error, failed_dir)))
            if args.fail_fast:
                break

    elapsed = time.time() - task_start
    print("[SUMMARY] selected={} skipped={} failed={} elapsed_seconds={:.1f}".format(
        len(rows), skipped, len(failures), elapsed
    ))
    for foodweb, message in failures:
        print("[FAILED] {}: {}".format(foodweb, message), file=sys.stderr)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
