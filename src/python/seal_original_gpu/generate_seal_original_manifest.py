#!/usr/bin/env python3
"""Freeze a 1:1 execution ledger for the unchanged SEAL original code."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

from seal_original_run_artifacts import (
    canonical_json_bytes,
    configuration_hash,
    file_sha256,
    safe_file_stem,
)


CORE_SOURCE_FILES = (
    "src/python/Main.py",
    "src/python/util_functions.py",
    "src/python/run_foodweb_seal.py",
)

DEFAULT_CONFIG = {
    "schema_version": "seal-original-gpu-config-v1",
    "version": "SEAL_undirected",
    "requested_train_ratio": 90.0,
    "test_ratio": 1.0 - 0.9,
    "hop": 1,
    "batch_size": 50,
    "max_train_num": 100000,
    "num_epochs": 50,
    "threshold": 0.5,
    "use_attribute": False,
    "use_embedding": False,
    "all_unknown_as_negative": False,
    "no_parallel": False,
    "retention": "raw_no_outlier_filter",
    "threshold_operator": "score_gt_threshold",
    "sampler": "unchanged_src_python_util_functions_sample_neg",
    "core_logic_policy": "exact_historical_behavior_no_corrections",
}


def experiment_seed(experiment_id, base_seed=12345):
    if int(experiment_id) < 1:
        raise ValueError("experiment_id is one-based")
    return int(base_seed) + int(experiment_id) - 1


def read_foodwebs(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Foodweb" not in reader.fieldnames:
            raise ValueError("CSV must contain a Foodweb column: {}".format(path))
        names = [row["Foodweb"].strip() for row in reader if row.get("Foodweb", "").strip()]
    if len(names) != len(set(names)):
        raise ValueError("Foodweb names are not unique in {}".format(path))
    return names


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".{}.".format(path.name), dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def core_source_hashes(repo_root):
    repo_root = Path(repo_root).resolve()
    hashes = {}
    for relative in CORE_SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[relative] = file_sha256(path)
    return hashes


def build_manifest(
    foodweb_csv,
    mat_folder,
    output_dir,
    repo_root,
    num_experiments=10,
    base_seed=12345,
    expected_foodwebs=290,
    only_foodwebs=None,
    source_commit="unspecified",
):
    foodweb_csv = Path(foodweb_csv).resolve()
    mat_folder = Path(mat_folder).resolve()
    output_dir = Path(output_dir).resolve()
    repo_root = Path(repo_root).resolve()
    only_foodwebs = set(only_foodwebs or [])

    foodwebs = read_foodwebs(foodweb_csv)
    if only_foodwebs:
        unknown = sorted(only_foodwebs.difference(foodwebs))
        if unknown:
            raise ValueError("Unknown food webs: {}".format(", ".join(unknown)))
        foodwebs = [name for name in foodwebs if name in only_foodwebs]
        expected_foodwebs = len(only_foodwebs)
    if len(foodwebs) != int(expected_foodwebs):
        raise ValueError("Expected {} food webs, found {}".format(expected_foodwebs, len(foodwebs)))
    if int(num_experiments) < 1:
        raise ValueError("num_experiments must be positive")

    config = dict(DEFAULT_CONFIG)
    config.update(
        {
            "num_foodwebs": len(foodwebs),
            "num_experiments": int(num_experiments),
            "base_seed": int(base_seed),
            "seed_formula": "base_seed + experiment_id - 1; reused across food webs",
            "source_commit": str(source_commit),
            "foodweb_csv_sha256": file_sha256(foodweb_csv),
            "core_source_sha256": core_source_hashes(repo_root),
        }
    )
    config_hash = configuration_hash(config)

    rows = []
    seen_stems = set()
    for foodweb_index, foodweb in enumerate(foodwebs, start=1):
        mat_path = mat_folder / "{}.mat".format(foodweb)
        if not mat_path.is_file():
            raise FileNotFoundError(mat_path)
        stem = safe_file_stem(foodweb)
        if stem in seen_stems:
            raise ValueError("Safe filename collision for {}".format(foodweb))
        seen_stems.add(stem)
        mat_hash = file_sha256(mat_path)
        for experiment_id in range(1, int(num_experiments) + 1):
            rows.append(
                {
                    "FoodwebIndex": foodweb_index,
                    "Foodweb": foodweb,
                    "ExperimentID": experiment_id,
                    "Seed": experiment_seed(experiment_id, base_seed),
                    "MatPath": str(mat_path),
                    "MatSHA256": mat_hash,
                    "ConfigHash": config_hash,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "RUN_CONFIG.json"
    manifest_path = output_dir / "manifest.csv"
    if config_path.exists() or manifest_path.exists():
        raise FileExistsError("Manifest output already exists in {}; use a new run root".format(output_dir))
    _atomic_write(config_path, canonical_json_bytes(config) + b"\n")

    fieldnames = (
        "FoodwebIndex", "Foodweb", "ExperimentID", "Seed",
        "MatPath", "MatSHA256", "ConfigHash",
    )
    descriptor, temporary = tempfile.mkstemp(prefix=".manifest.", dir=str(output_dir))
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, manifest_path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return manifest_path, config_path, rows


def parse_args(argv=None):
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foodweb-csv", type=Path, required=True)
    parser.add_argument("--mat-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--num-experiments", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--expected-foodwebs", type=int, default=290)
    parser.add_argument("--only-foodweb", action="append", default=[])
    parser.add_argument("--source-commit", default="unspecified")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest_path, config_path, rows = build_manifest(
        args.foodweb_csv,
        args.mat_folder,
        args.output_dir,
        args.repo_root,
        num_experiments=args.num_experiments,
        base_seed=args.base_seed,
        expected_foodwebs=args.expected_foodwebs,
        only_foodwebs=args.only_foodweb,
        source_commit=args.source_commit,
    )
    print("[OK] Manifest: {}".format(manifest_path))
    print("[OK] Configuration: {}".format(config_path))
    print("[OK] Runs: {}; distinct numeric seeds: {}".format(
        len(rows), len({int(row["Seed"]) for row in rows})
    ))


if __name__ == "__main__":
    main()
