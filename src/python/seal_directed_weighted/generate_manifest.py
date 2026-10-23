#!/usr/bin/env python3
"""Freeze shared splits and a four-variant SEAL-directed-weighted manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

from edge_schema import SCHEMA_SHA256
from run_artifacts import canonical_json_bytes, configuration_hash, file_sha256, safe_stem
from split_bundles import create_split_from_mat


VARIANTS = ("legacy_weak_node", "directed_node", "directed_scalar_gate", "directed_ecc")
DEFAULT_CONFIG = {
    "schema_version": "seal-directed-weighted-config-v1",
    "version": "SEAL_directed_weighted_v1",
    "task": "binary_resource_to_consumer_link_prediction",
    "edge_semantics": "contextual_covariates_not_interaction_strength",
    "edge_schema": "full_v1",
    "edge_schema_sha256": SCHEMA_SHA256,
    "model_variants": list(VARIANTS),
    "requested_train_ratio": 90.0,
    "test_ratio": 0.1,
    "hop": 1,
    "batch_size": 50,
    "max_train_num": 100000,
    "num_epochs": 50,
    "threshold": 0.5,
    "role_filter": True,
    "negative_positive_ratio": 1,
    "negative_fallback": "replace_with_all_directed_nonlinks",
    "node_attributes": 44,
    "use_embedding": False,
    "learning_rate": 0.0001,
    "latent_dimensions": [32, 32, 32, 1],
    "hidden_dimensions": 128,
    "dropout": True,
    "message_passing": "separate_self_incoming_outgoing_mean_aggregation",
    "scalar_gate_architecture": [17, 32, 1],
    "scalar_gate_activation": "sigmoid",
    "ecc_filter_architecture": [17, 32, "input_dim_times_output_dim"],
    "ecc_filter_scope": "independent_per_layer_and_direction",
    "sort_pooling_fraction": 0.6,
    "checkpoint_selection": "minimum_validation_loss",
    "edge_chunk_size": 8192,
    "subgraph_workers": 4,
    "edge_imputation": "training_context_mean_then_standardized_zero",
    "target_edge_policy": "remove_candidate_edge_and_its_covariates",
    "retention": "raw_no_outlier_filter",
    "primary_metric": "PR_AUC",
    "threshold_operator": "score_gt_threshold",
    "deterministic_algorithms": True,
}


def source_commit(repo_root):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def read_foodwebs(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "Foodweb" not in (reader.fieldnames or []):
            raise ValueError("Foodweb column missing from {}".format(path))
        result = [row["Foodweb"].strip() for row in reader if row.get("Foodweb", "").strip()]
    if len(result) != len(set(result)):
        raise ValueError("duplicate food-web names")
    return result


def seed_for(foodweb_index, experiment_id, base_seed=22345, experiments=10):
    return int(base_seed + experiments * (int(foodweb_index) - 1) + int(experiment_id) - 1)


def build_manifest(foodweb_csv, mat_folder, output_dir, num_experiments=10, base_seed=22345,
                   expected_foodwebs=290, only_foodwebs=None, frozen_source_commit=None):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("manifest directory is not empty: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    names = read_foodwebs(foodweb_csv)
    only = set(only_foodwebs or [])
    if only:
        names = [name for name in names if name in only]
        expected_foodwebs = len(only)
    if len(names) != int(expected_foodwebs):
        raise ValueError("expected {} food webs, found {}".format(expected_foodwebs, len(names)))
    if int(num_experiments) < 1:
        raise ValueError("num_experiments must be positive")

    repo_root = Path(__file__).resolve().parents[3]
    config = dict(DEFAULT_CONFIG)
    config.update(
        num_foodwebs=len(names),
        num_experiments=int(num_experiments),
        base_seed=int(base_seed),
        seed_formula="base_seed + num_experiments*(foodweb_index-1) + experiment_id-1",
        source_commit=frozen_source_commit or source_commit(repo_root),
        expected_unique_splits=len(names) * int(num_experiments),
        expected_runs=len(names) * int(num_experiments) * len(VARIANTS),
    )
    config_hash = configuration_hash(config)
    split_root = output_dir / "splits"
    rows = []
    split_hashes = set()
    for foodweb_index, name in enumerate(names, 1):
        mat_path = Path(mat_folder).resolve() / "{}.mat".format(name)
        if not mat_path.is_file():
            raise FileNotFoundError(mat_path)
        mat_hash = file_sha256(mat_path)
        for experiment_id in range(1, int(num_experiments) + 1):
            seed = seed_for(foodweb_index, experiment_id, base_seed, num_experiments)
            split_path = split_root / safe_stem(name) / "experiment_{:03d}_seed_{}.npz".format(
                experiment_id, seed
            )
            metadata = create_split_from_mat(
                mat_path,
                seed,
                split_path,
                test_ratio=config["test_ratio"],
                max_train_num=config["max_train_num"],
                role_filter=config["role_filter"],
            )
            split_hashes.add(metadata["logical_split_hash"])
            for variant in VARIANTS:
                rows.append({
                    "FoodwebIndex": foodweb_index,
                    "Foodweb": name,
                    "ExperimentID": experiment_id,
                    "Seed": seed,
                    "ModelVariant": variant,
                    "MatPath": str(mat_path),
                    "MatSHA256": mat_hash,
                    "SplitPath": str(split_path),
                    "SplitFileSHA256": metadata["split_file_sha256"],
                    "SplitHash": metadata["logical_split_hash"],
                    "RolePoolFallbackUsed": int(metadata["role_pool_fallback_used"]),
                    "ConfigHash": config_hash,
                })
    expected_splits = len(names) * int(num_experiments)
    if len(split_hashes) != expected_splits:
        raise ValueError("split hash collision detected")
    seeds = {int(row["Seed"]) for row in rows}
    if len(seeds) != expected_splits:
        raise ValueError("seed collision detected")

    with open(output_dir / "RUN_CONFIG.json", "wb") as handle:
        handle.write(canonical_json_bytes(config) + b"\n")
    fieldnames = list(rows[0])
    with open(output_dir / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("[OK] Food webs: {}".format(len(names)))
    print("[OK] Shared splits: {}".format(expected_splits))
    print("[OK] Manifest runs: {}".format(len(rows)))
    return output_dir / "manifest.csv", output_dir / "RUN_CONFIG.json", rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foodweb-csv", type=Path, required=True)
    parser.add_argument("--mat-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-experiments", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=22345)
    parser.add_argument("--expected-foodwebs", type=int, default=290)
    parser.add_argument("--only-foodweb", action="append", default=[])
    parser.add_argument("--source-commit")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    build_manifest(
        args.foodweb_csv, args.mat_folder, args.output_dir,
        num_experiments=args.num_experiments,
        base_seed=args.base_seed,
        expected_foodwebs=args.expected_foodwebs,
        only_foodwebs=args.only_foodweb,
        frozen_source_commit=args.source_commit,
    )


if __name__ == "__main__":
    main()
