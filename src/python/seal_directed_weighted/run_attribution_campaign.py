#!/usr/bin/env python3
"""Run one variant x experiment shard of weighted attribution bundles."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from attribution_artifacts import attribution_directory, is_valid_attribution_bundle
from generate_attribution_config import ATTRIBUTION_VARIANTS
from run_artifacts import configuration_hash, run_directory, safe_stem, validate_run_bundle
from run_campaign import read_manifest, resolve_block_task, select_manifest_row


def resolve_shard(task_id, variant, experiment_id, experiments):
    if task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if task_id is not None:
        maximum = len(ATTRIBUTION_VARIANTS) * int(experiments)
        if not 1 <= int(task_id) <= maximum:
            raise ValueError("task id must be in 1..{}".format(maximum))
        return (ATTRIBUTION_VARIANTS[(int(task_id) - 1) // int(experiments)],
                (int(task_id) - 1) % int(experiments) + 1)
    if variant is None or experiment_id is None:
        raise ValueError("provide --task-id or both --variant and --experiment-id")
    return variant, int(experiment_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--attribution-config", type=Path, required=True)
    parser.add_argument("--training-result-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--historical-result-root", type=Path)
    parser.add_argument("--task-id", type=int)
    parser.add_argument(
        "--manifest-row-id", type=int,
        help="Run one 1-based manifest row; intended for fine-grained Slurm arrays.",
    )
    parser.add_argument(
        "--block-task-id", type=int,
        help="Run one compact variant/experiment/food-web block task.",
    )
    parser.add_argument("--foodweb-block-size", type=int, default=6)
    parser.add_argument("--variant", choices=ATTRIBUTION_VARIANTS)
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--only-foodweb", action="append", default=[])
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    with open(args.run_config, encoding="utf-8") as handle:
        run_config = json.load(handle)
    with open(args.attribution_config, encoding="utf-8") as handle:
        attribution_config = json.load(handle)
    run_hash = configuration_hash(run_config)
    attr_hash = configuration_hash(attribution_config)
    manifest_rows = read_manifest(args.manifest)
    selector_count = sum(value is not None for value in (
        args.manifest_row_id, args.block_task_id, args.task_id,
    ))
    if selector_count > 1:
        raise ValueError(
            "manifest-row, block-task and shard task selectors are exclusive"
        )
    if args.manifest_row_id is not None:
        if args.variant is not None or args.experiment_id is not None:
            raise ValueError("--manifest-row-id is mutually exclusive with shard selectors")
        rows = [select_manifest_row(manifest_rows, args.manifest_row_id)]
        variant = rows[0]["ModelVariant"]
        experiment = int(rows[0]["ExperimentID"])
    elif args.block_task_id is not None:
        if args.variant is not None or args.experiment_id is not None:
            raise ValueError("--block-task-id is mutually exclusive with shard selectors")
        variant, experiment, block_index, _ = resolve_block_task(
            args.block_task_id, run_config, args.foodweb_block_size
        )
        rows = sorted(
            (
                row for row in manifest_rows
                if row["ModelVariant"] == variant
                and int(row["ExperimentID"]) == experiment
            ),
            key=lambda row: (int(row["FoodwebIndex"]), row["Foodweb"]),
        )
        start = block_index * int(args.foodweb_block_size)
        rows = rows[start:start + int(args.foodweb_block_size)]
    else:
        variant, experiment = resolve_shard(
            args.task_id, args.variant, args.experiment_id,
            run_config["num_experiments"],
        )
        rows = [row for row in manifest_rows
                if row["ModelVariant"] == variant and int(row["ExperimentID"]) == experiment]
    if args.only_foodweb:
        selected = set(args.only_foodweb); rows = [row for row in rows if row["Foodweb"] in selected]
    if not rows or {row["ConfigHash"] for row in rows} != {run_hash}:
        raise ValueError("empty or config-mismatched attribution shard")
    log_root = args.result_root / "attributions" / "task_logs" / variant / "experiment_{:03d}".format(experiment)
    log_root.mkdir(parents=True, exist_ok=True)
    failed, skipped, start = [], 0, time.time()
    script = Path(__file__).with_name("explain_weighted_attribution.py")
    for row in rows:
        training_dir = run_directory(args.training_result_root, variant, row["Foodweb"], experiment, row["Seed"])
        output_dir = attribution_directory(args.result_root, variant, row["Foodweb"], experiment, row["Seed"])
        try:
            validate_run_bundle(training_dir, run_hash)
            if output_dir.exists():
                if args.resume and is_valid_attribution_bundle(output_dir, attr_hash):
                    skipped += 1; continue
                raise ValueError("existing attribution bundle is invalid")
            command = [str(args.python_executable), str(script),
                       "--training-run-dir", str(training_dir),
                       "--attribution-config", str(args.attribution_config.resolve()),
                       "--output-dir", str(output_dir), "--device", args.device, "--deterministic"]
            if args.require_cuda: command.append("--require-cuda")
            if args.historical_result_root:
                historical = run_directory(args.historical_result_root, variant, row["Foodweb"], experiment, row["Seed"])
                if historical.is_dir(): command.extend(["--historical-run-dir", str(historical)])
            print("[ATTR] variant={} experiment={} foodweb={!r}".format(variant, experiment, row["Foodweb"]))
            if args.dry_run:
                print(subprocess.list2cmdline(command)); continue
            log_path = log_root / "{}_seed_{}.log".format(safe_stem(row["Foodweb"]), row["Seed"])
            with open(log_path, "w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           check=False, env=dict(os.environ, PYTHONHASHSEED=str(
                                               int(row["Seed"]) + int(attribution_config["attribution_seed_offset"]))))
            if completed.returncode or not is_valid_attribution_bundle(output_dir, attr_hash):
                raise RuntimeError("exit={} log={}".format(completed.returncode, log_path))
        except Exception as error:
            failed.append((row["Foodweb"], str(error)))
            if args.fail_fast: break
    print("[SUMMARY] variant={} experiment={} selected={} skipped={} failed={} seconds={:.1f}".format(
        variant, experiment, len(rows), skipped, len(failed), time.time() - start))
    for name, message in failed: print("[FAILED] {}: {}".format(name, message), file=sys.stderr)
    if failed: raise SystemExit(1)


if __name__ == "__main__": main()
