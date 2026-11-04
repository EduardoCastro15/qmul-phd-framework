#!/usr/bin/env python3
"""Fail-closed validation of all training and attribution bundles in a campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from attribution_artifacts import attribution_directory, validate_attribution_bundle
from run_foodweb_seal_directed import load_config, read_manifest
from seal_run_artifacts import configuration_hash, file_sha256, run_directory, validate_run_bundle
from validate_seal_directed_gpu_run import check_slurm


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--attribution-config", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--training-job-id", action="append", default=[])
    parser.add_argument("--attribution-job-id", action="append", default=[])
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--expected-foodwebs", type=int, default=290)
    parser.add_argument("--expected-experiments", type=int, default=10)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_config, run_config_hash = load_config(args.run_config)
    with open(args.attribution_config, encoding="utf-8") as handle:
        attribution_config = json.load(handle)
    attribution_config_hash = configuration_hash(attribution_config)
    manifest = read_manifest(args.manifest)
    expected = int(run_config["num_foodwebs"]) * int(run_config["num_experiments"])
    errors = []
    if int(run_config.get("num_foodwebs", -1)) != args.expected_foodwebs:
        errors.append("RUN_CONFIG does not contain expected food-web count")
    if int(run_config.get("num_experiments", -1)) != args.expected_experiments:
        errors.append("RUN_CONFIG does not contain expected experiment count")
    if len(manifest) != expected:
        errors.append("manifest has {} rows; expected {}".format(len(manifest), expected))
    training_count = 0
    attribution_count = 0
    candidate_count = 0
    eg_completeness = []
    drnl_completeness = []
    environment_fingerprints = set()
    for index, row in enumerate(manifest, 1):
        training_dir = run_directory(
            args.result_root, row["Foodweb"], row["ExperimentID"], row["Seed"]
        )
        attribution_dir = attribution_directory(
            args.result_root, row["Foodweb"], row["ExperimentID"], row["Seed"]
        )
        try:
            training = validate_run_bundle(training_dir, expected_config_hash=run_config_hash)
            training_count += 1
            checkpoint_hash = file_sha256(training_dir / "checkpoint.pt")
            training_success_hash = file_sha256(training_dir / "_SUCCESS")
            attribution = validate_attribution_bundle(
                attribution_dir,
                expected_config_hash=attribution_config_hash,
                expected_training_success_hash=training_success_hash,
                expected_checkpoint_hash=checkpoint_hash,
            )
            attribution_count += 1
            if attribution.get("Foodweb") != row["Foodweb"]:
                raise ValueError("Foodweb mismatch")
            if int(attribution.get("ExperimentID", -1)) != int(row["ExperimentID"]):
                raise ValueError("ExperimentID mismatch")
            if int(attribution.get("training_seed", -1)) != int(row["Seed"]):
                raise ValueError("Training seed mismatch")
            if int(attribution.get("attribution_seed", -1)) != int(row["Seed"]) + 1000000:
                raise ValueError("Attribution seed mismatch")
            if attribution.get("source_commit") != training["provenance"].get("source_commit"):
                raise ValueError("Attribution source commit differs from training bundle")
            if args.require_cuda and attribution.get("device") != "cuda:0":
                raise ValueError("Attribution did not run on cuda:0")
            environment_fingerprints.add((
                attribution.get("python_version"),
                attribution.get("torch_version"),
                attribution.get("captum_version"),
            ))
            with np.load(training_dir / "artifacts.npz", allow_pickle=False) as train_arrays, np.load(
                attribution_dir / "attributions.npz", allow_pickle=False
            ) as attr_arrays:
                expected_pairs = np.vstack([train_arrays["test_pos"], train_arrays["test_neg"]])
                if not np.array_equal(attr_arrays["candidate_pairs"], expected_pairs):
                    raise ValueError("Candidate order differs from training test split")
                if not np.array_equal(attr_arrays["candidate_labels"], train_arrays["test_labels"]):
                    raise ValueError("Attribution labels differ from training test labels")
                if not np.allclose(
                    attr_arrays["candidate_scores"], train_arrays["test_scores"],
                    rtol=0.0,
                    atol=float(attribution_config["score_parity_absolute_tolerance"]),
                ):
                    raise ValueError("Attribution scores differ from frozen training scores")
                if not np.array_equal(attr_arrays["node_offsets"], train_arrays["test_node_offsets"]):
                    raise ValueError("Attribution node offsets differ from training graphs")
                if not np.array_equal(attr_arrays["node_ids"], train_arrays["test_node_ids"]):
                    raise ValueError("Attribution node IDs differ from training graphs")
                candidate_count += len(attr_arrays["candidate_pairs"])
                eg_completeness.extend(attr_arrays["eg_completeness_relative"].tolist())
                drnl_completeness.extend(attr_arrays["drnl_completeness_relative"].tolist())
        except Exception as error:
            errors.append("{}: {}".format(row["Foodweb"], error))
        if index % 250 == 0:
            print("[PROGRESS] {}/{}".format(index, len(manifest)))

    for job_id in args.training_job_id + args.attribution_job_id:
        try:
            check_slurm(job_id)
        except Exception as error:
            errors.append("Slurm job {}: {}".format(job_id, error))
    if len(environment_fingerprints) > 1:
        errors.append("multiple attribution software environments detected")
    convergence = {}
    for name, values in (
        ("expected_gradients", eg_completeness),
        ("drnl_integrated_gradients", drnl_completeness),
    ):
        if values:
            convergence[name] = {
                "relative_completeness_median": float(np.median(values)),
                "relative_completeness_p95": float(np.percentile(values, 95)),
            }
            if (
                convergence[name]["relative_completeness_median"]
                > float(attribution_config["calibration_completeness_relative_median_max"])
                or convergence[name]["relative_completeness_p95"]
                > float(attribution_config["calibration_completeness_relative_p95_max"])
            ):
                errors.append("{} completeness gates failed".format(name))
    if errors:
        print("[INVALID] {} problem(s)".format(len(errors)))
        for error in errors[:50]:
            print("[ERROR] {}".format(error))
        raise SystemExit(1)
    report = {
        "status": "VALIDATION_OK",
        "expected_training_bundles": expected,
        "validated_training_bundles": training_count,
        "expected_attribution_bundles": expected,
        "validated_attribution_bundles": attribution_count,
        "test_candidates_explained": candidate_count,
        "all_test_candidates_explained_exactly_once": True,
        "run_config_hash": run_config_hash,
        "attribution_config_hash": attribution_config_hash,
        "software_environment_fingerprints": [list(value) for value in environment_fingerprints],
        "completeness": convergence,
    }
    if args.write_report:
        output = args.result_root / "summary" / "ATTRIBUTION_VALIDATION.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError("Validation report already exists: {}".format(output))
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
