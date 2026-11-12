#!/usr/bin/env python3
"""Fail-closed validation of all 5,800 training and attribution v2 bundles."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from attribution_artifacts import attribution_directory, validate_attribution_bundle
from attribution_manifest_lock import validate_manifest_lock
from run_artifacts import configuration_hash, file_sha256, run_directory, validate_run_bundle
from run_campaign import read_manifest
from validate_run import validate_slurm


def main(argv=None):
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
    args = parser.parse_args(argv)
    validate_manifest_lock(args.manifest, args.run_config)
    with open(args.run_config, encoding="utf-8") as handle: run_config = json.load(handle)
    with open(args.attribution_config, encoding="utf-8") as handle: attr_config = json.load(handle)
    run_hash, attr_hash = configuration_hash(run_config), configuration_hash(attr_config)
    rows, errors = read_manifest(args.manifest), []
    expected = args.expected_foodwebs * args.expected_experiments * 2
    if len(rows) != expected: errors.append("manifest has {} rows; expected {}".format(len(rows), expected))
    identities = {
        (row["ModelVariant"], row["Foodweb"], int(row["ExperimentID"]), int(row["Seed"]))
        for row in rows
    }
    if len(identities) != len(rows):
        errors.append("manifest contains duplicate model/foodweb/experiment/seed rows")
    if set(run_config.get("model_variants", [])) != {"directed_node", "directed_ecc"}:
        errors.append("RUN_CONFIG variants are not directed_node + directed_ecc")
    if run_config.get("expected_runs") != expected: errors.append("RUN_CONFIG expected_runs mismatch")
    counts, candidate_count, environments, completeness = Counter(), 0, set(), {
        "node": [], "drnl": [], "edge": []}
    historical = Counter()
    faithfulness = {"node_top": [], "node_bottom": [], "edge_top": [], "edge_bottom": []}
    for index, row in enumerate(rows, 1):
        variant = row["ModelVariant"]
        training_dir = run_directory(args.result_root, variant, row["Foodweb"], row["ExperimentID"], row["Seed"])
        attr_dir = attribution_directory(args.result_root, variant, row["Foodweb"], row["ExperimentID"], row["Seed"])
        try:
            training = validate_run_bundle(training_dir, run_hash)
            metadata = validate_attribution_bundle(
                attr_dir, attr_hash, file_sha256(training_dir / "_SUCCESS"),
                file_sha256(training_dir / "checkpoint.pt"))
            if metadata["ModelVariant"] != variant or metadata["Foodweb"] != row["Foodweb"]:
                raise ValueError("attribution identity mismatch")
            if int(metadata["ExperimentID"]) != int(row["ExperimentID"]):
                raise ValueError("ExperimentID mismatch")
            if int(metadata["training_seed"]) != int(row["Seed"]):
                raise ValueError("training seed mismatch")
            if training["metrics"].get("SplitHash") != row["SplitHash"]:
                raise ValueError("training split hash differs from manifest")
            if args.require_cuda and metadata.get("device") != "cuda:0":
                raise ValueError("attribution was not run on CUDA")
            environments.add((metadata.get("python_version"), metadata.get("torch_version"),
                              metadata.get("captum_version")))
            comparison = metadata.get("historical_comparison", {})
            if not comparison.get("available", False):
                historical["unavailable"] += 1
                raise ValueError("historical score comparison is unavailable")
            historical["exact" if comparison.get("exact_replay") else "replica_v2"] += 1
            with np.load(training_dir / "artifacts.npz", allow_pickle=False) as train, np.load(
                    attr_dir / "attributions.npz", allow_pickle=False) as attr:
                expected_pairs = np.vstack([train["test_pos"], train["test_neg"]])
                if not np.array_equal(attr["candidate_pairs"], expected_pairs):
                    raise ValueError("candidate order differs from training bundle")
                if not np.array_equal(attr["candidate_labels"], train["test_labels"]):
                    raise ValueError("candidate labels differ from training bundle")
                if not np.allclose(attr["candidate_scores"], train["test_scores"], rtol=0,
                                   atol=float(attr_config["score_parity_absolute_tolerance"])):
                    raise ValueError("attribution score parity failed")
                if not np.array_equal(attr["node_offsets"], train["test_node_offsets"]):
                    raise ValueError("node offsets differ from training bundle")
                if not np.array_equal(attr["node_ids"], train["test_node_ids"]):
                    raise ValueError("node IDs differ from training bundle")
                if variant == "directed_ecc":
                    if not np.array_equal(attr["edge_offsets"], train["test_edge_offsets"]):
                        raise ValueError("ECC edge offsets differ from training bundle")
                    if not np.array_equal(attr["edge_index"], train["test_edge_index"]):
                        raise ValueError("ECC edge order differs from training bundle")
                elif (
                    not np.all(attr["edge_offsets"] == 0)
                    or np.asarray(attr["edge_index"]).size
                    or np.asarray(attr["edge_expected_gradients"]).size
                ):
                    raise ValueError("directed_node contains edge attributions")
                candidate_count += len(attr["candidate_pairs"])
                completeness["node"].extend(attr["node_completeness_relative"].tolist())
                completeness["drnl"].extend(attr["drnl_completeness_relative"].tolist())
                if variant == "directed_ecc":
                    completeness["edge"].extend(attr["edge_completeness_relative"].tolist())
                if int(attr_config.get("faithfulness_background_samples", 0)) > 0:
                    for channel in (("node", "edge") if variant == "directed_ecc" else ("node",)):
                        top = "faithfulness_{}_top_abs_log_odds_change".format(channel)
                        bottom = "faithfulness_{}_bottom_abs_log_odds_change".format(channel)
                        if top not in attr.files or bottom not in attr.files:
                            raise ValueError("missing {} faithfulness arrays".format(channel))
                        faithfulness[channel + "_top"].extend(attr[top].tolist())
                        faithfulness[channel + "_bottom"].extend(attr[bottom].tolist())
            counts[variant] += 1
        except Exception as error:
            errors.append("{} {} exp {}: {}".format(variant, row["Foodweb"], row["ExperimentID"], error))
        if index % 250 == 0: print("[PROGRESS] {}/{}".format(index, len(rows)))
    if len(environments) > 1: errors.append("multiple software environments detected")
    expected_per_variant = args.expected_foodwebs * args.expected_experiments
    for variant in ("directed_node", "directed_ecc"):
        if counts[variant] != expected_per_variant:
            errors.append("{} bundles for {}; expected {}".format(counts[variant], variant, expected_per_variant))
    convergence = {}
    for channel, values in completeness.items():
        if not values: continue
        convergence[channel] = {"relative_completeness_median": float(np.median(values)),
                                "relative_completeness_p95": float(np.percentile(values, 95))}
        if (convergence[channel]["relative_completeness_median"]
                > attr_config["calibration_completeness_relative_median_max"]
                or convergence[channel]["relative_completeness_p95"]
                > attr_config["calibration_completeness_relative_p95_max"]):
            errors.append("{} completeness gates failed".format(channel))
    faithfulness_report = {}
    if int(attr_config.get("faithfulness_background_samples", 0)) > 0:
        for channel in ("node", "edge"):
            top, bottom = faithfulness[channel + "_top"], faithfulness[channel + "_bottom"]
            if not top or not bottom:
                errors.append("{} faithfulness evidence is empty".format(channel))
                continue
            faithfulness_report[channel] = {
                "top_median_abs_log_odds_change": float(np.median(top)),
                "bottom_median_abs_log_odds_change": float(np.median(bottom)),
                "top_exceeds_bottom": bool(np.median(top) > np.median(bottom)),
            }
            if not faithfulness_report[channel]["top_exceeds_bottom"]:
                errors.append("{} top-vs-bottom faithfulness gate failed".format(channel))
    for job_id in args.training_job_id + args.attribution_job_id:
        try: validate_slurm([job_id])
        except Exception as error: errors.append("Slurm job {}: {}".format(job_id, error))
    if errors:
        print("[INVALID] {} problem(s)".format(len(errors)))
        for error in errors[:100]: print("[ERROR] {}".format(error))
        raise SystemExit(1)
    report = {"status": "VALIDATION_OK", "expected_training_bundles": expected,
              "validated_training_bundles": sum(counts.values()),
              "expected_attribution_bundles": expected,
              "validated_attribution_bundles": sum(counts.values()),
              "bundles_by_variant": dict(counts), "test_candidates_explained": candidate_count,
              "all_test_candidates_explained_exactly_once": True,
              "historical_score_replay": dict(historical), "run_config_hash": run_hash,
              "attribution_config_hash": attr_hash, "software_environments": [list(v) for v in environments],
              "completeness": convergence, "faithfulness": faithfulness_report,
              "nonfinite_attributions": 0}
    if args.write_report:
        path = args.result_root / "summary" / "ATTRIBUTION_VALIDATION.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): raise FileExistsError(path)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__": main()
