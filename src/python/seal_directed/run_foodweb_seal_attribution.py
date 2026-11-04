#!/usr/bin/env python3
"""Run one manifest shard of immutable SEAL-D-NodeAttr attribution bundles."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from attribution_artifacts import (
    attribution_directory,
    is_valid_attribution_bundle,
)
from run_foodweb_seal_directed import (
    load_config,
    read_manifest,
    resolve_experiment_id,
    stage_mat_file,
    validate_native_library,
    validate_python_environment,
)
from seal_run_artifacts import configuration_hash, file_sha256, run_directory, safe_file_stem


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--attribution-config", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--training-result-root", type=Path, default=None,
        help="root containing training bundles; defaults to --result-root",
    )
    parser.add_argument("--experiment-id", type=int, default=None)
    parser.add_argument("--only-foodweb", action="append", default=[])
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--scratch-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--randomize-through",
        choices=("output", "mlp", "sortpool_head", "message_passing"),
        default=None,
        help="sanity-check attribution using progressively randomized weights",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.manifest = args.manifest.resolve()
    args.run_config = args.run_config.resolve()
    args.attribution_config = args.attribution_config.resolve()
    args.result_root = args.result_root.resolve()
    args.training_result_root = (
        args.training_result_root.resolve()
        if args.training_result_root is not None else args.result_root
    )
    args.python_executable = args.python_executable.resolve()
    experiment_id = resolve_experiment_id(args.experiment_id)
    run_config, run_config_hash = load_config(args.run_config)
    with open(args.attribution_config, encoding="utf-8") as handle:
        attribution_config = json.load(handle)
    attribution_config_hash = configuration_hash(attribution_config)
    rows = read_manifest(args.manifest)
    if {row["ConfigHash"] for row in rows} != {run_config_hash}:
        raise ValueError("RUN_CONFIG hash does not match manifest")
    rows = [row for row in rows if int(row["ExperimentID"]) == experiment_id]
    if args.only_foodweb:
        selected = set(args.only_foodweb)
        rows = [row for row in rows if row["Foodweb"] in selected]
    if not rows:
        raise ValueError("No manifest rows selected")

    script_dir = Path(__file__).resolve().parent
    if not args.dry_run:
        environment = validate_python_environment(
            args.python_executable, args.device, args.require_cuda
        )
        if environment.get("captum") != attribution_config.get("captum_version"):
            raise RuntimeError("Runtime Captum version does not match ATTRIBUTION_CONFIG")
        validate_native_library(script_dir)
    log_root = args.result_root / "attributions" / "task_logs" / "experiment_{:03d}".format(
        experiment_id
    )
    log_root.mkdir(parents=True, exist_ok=True)
    failures = []
    skipped = 0
    started = time.time()
    for row in rows:
        training_dir = run_directory(
            args.training_result_root, row["Foodweb"], row["ExperimentID"], row["Seed"]
        )
        if not training_dir.is_dir():
            failures.append((row["Foodweb"], "missing training bundle {}".format(training_dir)))
            if args.fail_fast:
                break
            continue
        output_dir = attribution_directory(
            args.result_root, row["Foodweb"], row["ExperimentID"], row["Seed"]
        )
        if output_dir.exists():
            if args.resume and is_valid_attribution_bundle(output_dir, attribution_config_hash):
                print("[SKIP] valid attribution {}".format(output_dir))
                skipped += 1
                continue
            failures.append((row["Foodweb"], "existing attribution bundle is invalid"))
            if args.fail_fast:
                break
            continue
        mat_path = Path(row["MatPath"])
        if not mat_path.is_file() or file_sha256(mat_path) != row["MatSHA256"]:
            failures.append((row["Foodweb"], "MAT input missing or checksum mismatch"))
            if args.fail_fast:
                break
            continue
        if args.scratch_root is not None:
            mat_path = stage_mat_file(
                mat_path, args.scratch_root.resolve(), row["MatSHA256"], experiment_id
            )
        command = [
            str(args.python_executable), str(script_dir / "explain_seal_directed.py"),
            "--training-run-dir", str(training_dir),
            "--mat-path", str(mat_path),
            "--attribution-config", str(args.attribution_config),
            "--output-dir", str(output_dir),
            "--device", args.device,
            "--deterministic",
        ]
        if args.require_cuda:
            command.append("--require-cuda")
        if args.randomize_through:
            command.extend(["--randomize-through", args.randomize_through])
        log_path = log_root / "{}_seed_{}.log".format(
            safe_file_stem(row["Foodweb"]), row["Seed"]
        )
        print("[ATTR] experiment={} foodweb={!r}".format(experiment_id, row["Foodweb"]))
        if args.dry_run:
            print(subprocess.list2cmdline(command))
            continue
        with open(log_path, "w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=str(script_dir),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                env=dict(os.environ, PYTHONHASHSEED=str(int(row["Seed"]) + 1000000)),
            )
        if completed.returncode != 0 or not is_valid_attribution_bundle(
            output_dir, attribution_config_hash
        ):
            failures.append((row["Foodweb"], "exit={} log={}".format(completed.returncode, log_path)))
            if args.fail_fast:
                break
    print(
        "[SUMMARY] selected={} skipped={} failed={} elapsed_seconds={:.1f}".format(
            len(rows), skipped, len(failures), time.time() - started
        )
    )
    for foodweb, message in failures:
        print("[FAILED] {}: {}".format(foodweb, message), file=sys.stderr)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
