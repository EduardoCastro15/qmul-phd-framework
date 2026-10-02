#!/usr/bin/env python3
"""Freeze the 100 x 290 SEAL-directed experiment ledger."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

from seal_run_artifacts import canonical_json_bytes, configuration_hash, file_sha256


DEFAULT_CONFIG = {
    "schema_version": "seal-directed-gpu-config-v1",
    "version": "SEAL_directed_gpu_v1",
    "requested_train_ratio": 90.0,
    "test_ratio": 0.1,
    "hop": 1,
    "batch_size": 50,
    "max_train_num": 100000,
    "num_epochs": 50,
    "threshold": 0.5,
    "use_attribute": True,
    "use_embedding": False,
    "role_filter": True,
    "all_unknown_as_negative": False,
    "negative_positive_ratio": 1,
    "negative_fallback": "replace_with_all_directed_nonlinks",
    "learning_rate": 0.0001,
    "latent_dimensions": [32, 32, 32, 1],
    "hidden_dimensions": 128,
    "dropout": True,
    "float_precision": "float32",
    "retention": "raw_no_outlier_filter",
    "threshold_operator": "score_gt_threshold",
    "deterministic_algorithms": True,
}


def experiment_seed(foodweb_index, experiment_id, base_seed=12345, experiments=100):
    if foodweb_index < 1 or experiment_id < 1:
        raise ValueError("foodweb_index and experiment_id are one-based")
    return int(base_seed + experiments * (foodweb_index - 1) + experiment_id - 1)


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
    descriptor, temp_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name), dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def build_manifest(
    foodweb_csv,
    mat_folder,
    output_dir,
    num_experiments=100,
    base_seed=12345,
    expected_foodwebs=290,
    only_foodwebs=None,
    source_commit="unspecified",
    deterministic_algorithms=True,
):
    foodweb_csv = Path(foodweb_csv).resolve()
    mat_folder = Path(mat_folder).resolve()
    output_dir = Path(output_dir).resolve()
    only_foodwebs = set(only_foodwebs or [])

    foodwebs = read_foodwebs(foodweb_csv)
    if only_foodwebs:
        unknown = sorted(only_foodwebs.difference(foodwebs))
        if unknown:
            raise ValueError("Unknown food webs: {}".format(", ".join(unknown)))
        foodwebs = [name for name in foodwebs if name in only_foodwebs]
        expected_foodwebs = len(only_foodwebs)

    if len(foodwebs) != int(expected_foodwebs):
        raise ValueError(
            "Expected {} food webs, found {}".format(expected_foodwebs, len(foodwebs))
        )
    if num_experiments < 1:
        raise ValueError("num_experiments must be positive")

    config = dict(DEFAULT_CONFIG)
    config.update(
        {
            "num_foodwebs": len(foodwebs),
            "num_experiments": int(num_experiments),
            "base_seed": int(base_seed),
            "seed_formula": "base_seed + num_experiments*(foodweb_index-1) + experiment_id-1",
            "source_commit": str(source_commit),
            "deterministic_algorithms": bool(deterministic_algorithms),
        }
    )
    config_hash = configuration_hash(config)

    rows = []
    seen_stems = set()
    for foodweb_index, foodweb in enumerate(foodwebs, start=1):
        mat_path = mat_folder / "{}.mat".format(foodweb)
        if not mat_path.is_file():
            raise FileNotFoundError(mat_path)
        safe_stem = "".join(
            character if character.isalnum() or character in ("-", "_", ".") else "_"
            for character in foodweb
        )
        if safe_stem in seen_stems:
            raise ValueError("Safe filename collision for {}".format(foodweb))
        seen_stems.add(safe_stem)
        mat_sha256 = file_sha256(mat_path)
        for experiment_id in range(1, int(num_experiments) + 1):
            rows.append(
                {
                    "FoodwebIndex": foodweb_index,
                    "Foodweb": foodweb,
                    "ExperimentID": experiment_id,
                    "Seed": experiment_seed(
                        foodweb_index,
                        experiment_id,
                        base_seed=base_seed,
                        experiments=num_experiments,
                    ),
                    "MatPath": str(mat_path),
                    "MatSHA256": mat_sha256,
                    "ConfigHash": config_hash,
                }
            )

    if len({row["Seed"] for row in rows}) != len(rows):
        raise ValueError("The manifest contains seed collisions")

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "RUN_CONFIG.json"
    manifest_path = output_dir / "manifest.csv"
    if config_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Manifest output already exists in {}; use a new run root".format(output_dir)
        )

    _atomic_write(config_path, canonical_json_bytes(config) + b"\n")
    fieldnames = [
        "FoodwebIndex",
        "Foodweb",
        "ExperimentID",
        "Seed",
        "MatPath",
        "MatSHA256",
        "ConfigHash",
    ]
    descriptor, temp_name = tempfile.mkstemp(prefix=".manifest.", dir=str(output_dir))
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, manifest_path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

    return manifest_path, config_path, rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foodweb-csv", type=Path, required=True)
    parser.add_argument("--mat-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-experiments", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--expected-foodwebs", type=int, default=290)
    parser.add_argument("--only-foodweb", action="append", default=[])
    parser.add_argument("--source-commit", default="unspecified")
    parser.add_argument(
        "--non-deterministic", action="store_true",
        help="record an explicitly approved CUDA determinism exception",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest_path, config_path, rows = build_manifest(
        args.foodweb_csv,
        args.mat_folder,
        args.output_dir,
        num_experiments=args.num_experiments,
        base_seed=args.base_seed,
        expected_foodwebs=args.expected_foodwebs,
        only_foodwebs=args.only_foodweb,
        source_commit=args.source_commit,
        deterministic_algorithms=not args.non_deterministic,
    )
    print("[OK] Manifest: {}".format(manifest_path))
    print("[OK] Configuration: {}".format(config_path))
    print("[OK] Experiments: {}".format(len(rows)))


if __name__ == "__main__":
    main()
