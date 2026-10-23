#!/usr/bin/env python3
"""Execute one manifest shard (variant x experiment) with guarded resume."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from models import VARIANTS
from run_artifacts import configuration_hash, file_sha256, is_valid_complete_run, run_directory, safe_stem


def read_manifest(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def resolve_shard(args, config):
    task_id = args.task_id
    if task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if task_id is not None:
        experiments = int(config["num_experiments"])
        maximum = len(VARIANTS) * experiments
        if not 1 <= int(task_id) <= maximum:
            raise ValueError("task id must be in 1..{}".format(maximum))
        variant = VARIANTS[(int(task_id) - 1) // experiments]
        experiment_id = (int(task_id) - 1) % experiments + 1
        return variant, experiment_id
    if args.variant is None or args.experiment_id is None:
        raise ValueError("provide --task-id or both --variant and --experiment-id")
    return args.variant, int(args.experiment_id)


def stage_file(path, scratch, expected_hash):
    path = Path(path)
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise ValueError("source checksum mismatch: {}".format(path))
    if scratch is None:
        return path
    destination = Path(scratch) / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and file_sha256(destination) == expected_hash:
        return destination
    temporary = destination.with_name(".{}.{}.tmp".format(destination.name, os.getpid()))
    shutil.copy2(path, temporary)
    if file_sha256(temporary) != expected_hash:
        temporary.unlink(missing_ok=True)
        raise ValueError("staged checksum mismatch: {}".format(path))
    os.replace(temporary, destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--only-foodweb", action="append", default=[])
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args(argv)
    if args.num_workers is not None and args.num_workers < 1:
        parser.error("--num-workers must be at least 1")

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    config_hash = configuration_hash(config)
    if args.source_commit not in {"unknown", config.get("source_commit")}:
        raise ValueError(
            "source commit mismatch: config={} runner={}".format(
                config.get("source_commit"), args.source_commit
            )
        )
    rows = read_manifest(args.manifest)
    variant, experiment_id = resolve_shard(args, config)
    rows = [
        row for row in rows
        if row["ModelVariant"] == variant and int(row["ExperimentID"]) == experiment_id
    ]
    if args.only_foodweb:
        selected = set(args.only_foodweb)
        rows = [row for row in rows if row["Foodweb"] in selected]
    if not rows:
        raise ValueError("empty manifest shard")
    if {row["ConfigHash"] for row in rows} != {config_hash}:
        raise ValueError("manifest/config hash mismatch")
    failures, skipped = [], 0
    log_root = args.result_root / "task_logs" / variant / "experiment_{:03d}".format(experiment_id)
    log_root.mkdir(parents=True, exist_ok=True)
    task_scratch = None
    if args.scratch_root:
        token = "{}_{}".format(os.environ.get("SLURM_JOB_ID", "local"), os.getpid())
        task_scratch = args.scratch_root / "seal_weighted_{}".format(token)
    start = time.time()
    for row in rows:
        run_dir = run_directory(
            args.result_root, variant, row["Foodweb"], experiment_id, row["Seed"]
        )
        if run_dir.exists():
            if args.resume and is_valid_complete_run(run_dir, config_hash):
                skipped += 1
                continue
            failures.append((row["Foodweb"], "existing invalid/non-resume run"))
            if args.fail_fast:
                break
            continue
        try:
            mat_path = stage_file(row["MatPath"], task_scratch / "mats" if task_scratch else None, row["MatSHA256"])
            split_path = stage_file(
                row["SplitPath"], task_scratch / "splits" if task_scratch else None,
                row["SplitFileSHA256"],
            )
            command = [
                str(args.python_executable), str(Path(__file__).with_name("train_one.py")),
                "--foodweb", row["Foodweb"],
                "--foodweb-index", row["FoodwebIndex"],
                "--experiment-id", row["ExperimentID"],
                "--seed", row["Seed"],
                "--variant", variant,
                "--mat-path", str(mat_path),
                "--split-bundle", str(split_path),
                "--split-file-sha256", row["SplitFileSHA256"],
                "--split-hash", row["SplitHash"],
                "--role-pool-fallback-used", row["RolePoolFallbackUsed"],
                "--run-dir", str(run_dir),
                "--config-hash", config_hash,
                "--source-commit", args.source_commit,
                "--device", args.device,
                "--hop", str(config["hop"]),
                "--batch-size", str(config["batch_size"]),
                "--num-epochs", str(config["num_epochs"]),
                "--learning-rate", str(config["learning_rate"]),
                "--threshold", str(config["threshold"]),
                "--edge-chunk-size", str(config["edge_chunk_size"]),
                "--num-workers", str(args.num_workers or config.get("subgraph_workers", 1)),
            ]
            if args.require_cuda:
                command.append("--require-cuda")
            if config.get("deterministic_algorithms", True):
                command.append("--deterministic")
            print("[RUN] variant={} experiment={} foodweb={!r}".format(variant, experiment_id, row["Foodweb"]))
            if args.dry_run:
                print(subprocess.list2cmdline(command))
                continue
            log_path = log_root / "{}_seed_{}.log".format(safe_stem(row["Foodweb"]), row["Seed"])
            with open(log_path, "w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=False,
                    env=dict(os.environ, PYTHONHASHSEED=row["Seed"]),
                )
            if completed.returncode or not is_valid_complete_run(run_dir, config_hash):
                raise RuntimeError("exit={} log={}".format(completed.returncode, log_path))
        except Exception as error:
            failures.append((row["Foodweb"], str(error)))
            if args.fail_fast:
                break
    print(
        "[SUMMARY] variant={} experiment={} selected={} skipped={} failed={} seconds={:.1f}".format(
            variant, experiment_id, len(rows), skipped, len(failures), time.time() - start
        )
    )
    for name, message in failures:
        print("[FAILED] {}: {}".format(name, message), file=sys.stderr)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
