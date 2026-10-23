#!/usr/bin/env python3
"""Fail-closed validation and raw summaries for a weighted SEAL campaign."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from models import VARIANTS
from run_artifacts import configuration_hash, logical_split_hash, run_directory, validate_run_bundle


METRICS = ("ROC_AUC", "PR_AUC", "F1Score", "TestMCC", "Precision", "Recall", "TestTSS")


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_slurm(job_ids):
    checked = 0
    for job_id in job_ids:
        completed = subprocess.run(
            ["sacct", "-X", "-n", "-P", "-j", str(job_id), "--format=State,ExitCode"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if completed.returncode:
            raise RuntimeError("sacct failed for {}: {}".format(job_id, completed.stderr.strip()))
        records = [line.split("|")[:2] for line in completed.stdout.splitlines() if line.strip()]
        terminal = [(state, code) for state, code in records if state not in {"PENDING", "RUNNING"}]
        if not terminal or not any(state.startswith("COMPLETED") and code == "0:0" for state, code in terminal):
            raise ValueError("no successful terminal Slurm record for {}: {}".format(job_id, records))
        checked += 1
    return checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument(
        "--task-id", action="append", type=int, default=[],
        help="validate only selected variant x experiment shards (for smoke/calibration)",
    )
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--write-summaries", action="store_true")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    config_hash = configuration_hash(config)
    manifest = read_rows(args.manifest)
    if not manifest:
        raise ValueError("empty manifest")
    if {row["ConfigHash"] for row in manifest} != {config_hash}:
        raise ValueError("manifest/config hash mismatch")
    if args.task_id:
        experiments = int(config["num_experiments"])
        selected = set()
        for task_id in args.task_id:
            if not 1 <= task_id <= len(VARIANTS) * experiments:
                raise ValueError("invalid task id {}".format(task_id))
            selected.add((VARIANTS[(task_id - 1) // experiments], (task_id - 1) % experiments + 1))
        manifest = [
            row for row in manifest
            if (row["ModelVariant"], int(row["ExperimentID"])) in selected
        ]
    expected_keys = {
        (row["Foodweb"], int(row["ExperimentID"]), row["ModelVariant"])
        for row in manifest
    }
    if len(expected_keys) != len(manifest):
        raise ValueError("duplicate foodweb/experiment/variant manifest rows")

    raw_rows = []
    split_variants = defaultdict(set)
    split_artifact_hashes = defaultdict(set)
    fallback_runs = 0
    for index, row in enumerate(manifest, 1):
        run_dir = run_directory(
            args.result_root, row["ModelVariant"], row["Foodweb"],
            row["ExperimentID"], row["Seed"],
        )
        payload = validate_run_bundle(run_dir, config_hash)
        provenance = payload["provenance"]
        values = payload["metrics"]
        if values.get("ModelVariant") != row["ModelVariant"]:
            raise ValueError("model variant mismatch in {}".format(run_dir))
        if provenance.get("mat_sha256") != row["MatSHA256"]:
            raise ValueError("MAT checksum provenance mismatch in {}".format(run_dir))
        if provenance.get("split_file_sha256") != row["SplitFileSHA256"]:
            raise ValueError("split-file checksum provenance mismatch in {}".format(run_dir))
        if provenance.get("split_hash") != row["SplitHash"] or values.get("SplitHash") != row["SplitHash"]:
            raise ValueError("split hash mismatch in {}".format(run_dir))
        if provenance.get("edge_schema_sha256") != config["edge_schema_sha256"]:
            raise ValueError("edge schema mismatch in {}".format(run_dir))
        if int(values.get("RolePoolFallbackUsed", -1)) != int(row["RolePoolFallbackUsed"]):
            raise ValueError("fallback status mismatch in {}".format(run_dir))
        if args.require_cuda and not (
            provenance.get("cuda_available") and str(provenance.get("device", "")).startswith("cuda")
        ):
            raise ValueError("run did not use CUDA: {}".format(run_dir))
        numeric = [values.get(metric) for metric in METRICS]
        if any(value is None or not np.isfinite(float(value)) for value in numeric):
            raise ValueError("non-finite predictive metric in {}".format(run_dir))
        with np.load(run_dir / "artifacts.npz", allow_pickle=False) as artifact:
            arrays = {name: np.asarray(artifact[name], dtype=np.int64).reshape(-1, 2) for name in (
                "train_pos", "train_neg", "test_pos", "test_neg"
            )}
        observed_hash = logical_split_hash(**arrays)
        if observed_hash != row["SplitHash"]:
            raise ValueError("artifact split mismatch in {}".format(run_dir))
        sets = {name: set(map(tuple, array.tolist())) for name, array in arrays.items()}
        if sets["train_pos"] & sets["test_pos"]:
            raise ValueError("positive train/test overlap in {}".format(run_dir))
        if sets["train_neg"] & sets["test_neg"]:
            raise ValueError("negative train/test overlap in {}".format(run_dir))
        if (sets["train_pos"] | sets["test_pos"]) & (sets["train_neg"] | sets["test_neg"]):
            raise ValueError("positive/negative overlap in {}".format(run_dir))
        if any(source == target for pairs in sets.values() for source, target in pairs):
            raise ValueError("self-loop in split {}".format(run_dir))

        split_key = (row["Foodweb"], int(row["ExperimentID"]), row["SplitHash"])
        split_variants[split_key].add(row["ModelVariant"])
        split_artifact_hashes[split_key].add(observed_hash)
        fallback_runs += int(row["RolePoolFallbackUsed"])
        raw_rows.append(dict(values, ConfigHash=config_hash, RunDirectory=str(run_dir)))
        if index % 1000 == 0:
            print("[PROGRESS] validated {}/{} runs".format(index, len(manifest)))

    expected_variant_set = {row["ModelVariant"] for row in manifest}
    for key, variants in split_variants.items():
        if variants != expected_variant_set:
            raise ValueError("split not shared by all variants: {} -> {}".format(key, variants))
        if len(split_artifact_hashes[key]) != 1:
            raise ValueError("variant split artifacts differ: {}".format(key))
    expected_split_count = len({(row["Foodweb"], int(row["ExperimentID"])) for row in manifest})
    if len(split_variants) != expected_split_count:
        raise ValueError("expected {} shared splits, found {}".format(expected_split_count, len(split_variants)))

    slurm_checked = validate_slurm(args.job_id) if args.job_id else 0
    if args.write_summaries:
        summary = args.result_root / "summary"
        write_csv(summary / "raw_runs.csv", raw_rows)
        grouped = defaultdict(list)
        for row in raw_rows:
            grouped[(row["Foodweb"], row["ModelVariant"])].append(row)
        means = []
        for (foodweb, variant), group in sorted(grouped.items()):
            output = {"Foodweb": foodweb, "ModelVariant": variant, "Runs": len(group)}
            for metric in METRICS:
                output[metric] = float(np.mean([float(row[metric]) for row in group]))
            means.append(output)
        write_csv(summary / "foodweb_metric_means_raw.csv", means)

        by_web = defaultdict(dict)
        for row in means:
            by_web[row["Foodweb"]][row["ModelVariant"]] = row
        differences = []
        for foodweb, variants in sorted(by_web.items()):
            if set(variants) != expected_variant_set:
                raise ValueError("food web missing a variant in means: {}".format(foodweb))
            for comparator in ("legacy_weak_node", "directed_node", "directed_scalar_gate"):
                output = {"Foodweb": foodweb, "Contrast": "directed_ecc-minus-{}".format(comparator)}
                for metric in METRICS:
                    output[metric] = float(variants["directed_ecc"][metric]) - float(variants[comparator][metric])
                differences.append(output)
        write_csv(summary / "paired_metric_differences_raw.csv", differences)
        fallback_summary = [
            {"RolePoolFallbackUsed": value, "Runs": count}
            for value, count in sorted(Counter(int(row["RolePoolFallbackUsed"]) for row in raw_rows).items())
        ]
        write_csv(summary / "negative_pool_fallback_summary.csv", fallback_summary)

    validation = {
        "status": "VALIDATION_OK",
        "config_hash": config_hash,
        "version": config["version"],
        "foodwebs": len({row["Foodweb"] for row in manifest}),
        "experiments": len({int(row["ExperimentID"]) for row in manifest}),
        "variants": sorted({row["ModelVariant"] for row in manifest}),
        "manifest_rows": len(manifest),
        "validated_runs": len(raw_rows),
        "shared_split_hashes": len(split_variants),
        "fallback_variant_runs": fallback_runs,
        "nonfinite_predictive_metrics": 0,
        "slurm_records_checked": slurm_checked,
        "retention": "raw_no_outlier_filter",
    }
    if args.write_summaries:
        with open(args.result_root / "summary" / "RUN_VALIDATION.json", "w", encoding="utf-8") as handle:
            json.dump(validation, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
