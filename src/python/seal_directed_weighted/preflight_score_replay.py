#!/usr/bin/env python3
"""Validate batched predictive replay and bounded single-graph attribution drift."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import socket
import tempfile
from pathlib import Path

import numpy as np
import torch

from explain_weighted_attribution import load_checkpoint, validate_forward_parity
from attribution_manifest_lock import validate_manifest_lock
from graph_artifacts_v2 import unpack_graphs
from models import build_classifier
from run_artifacts import canonical_json_bytes, configuration_hash, run_directory, validate_run_bundle
from run_campaign import read_manifest
from weighted_attribution import _prediction, one_hot_drnl, validate_candidate_score_replay


def atomic_preflight_directory(output_dir, rows, report):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("preflight output is immutable: {}".format(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".{}.tmp.".format(output_dir.name), dir=output_dir.parent))
    try:
        with open(temporary / "SCORE_REPLAY_PREFLIGHT.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["status"])
            writer.writeheader()
            if rows:
                writer.writerows(rows)
        (temporary / "SCORE_REPLAY_PREFLIGHT.json").write_bytes(
            canonical_json_bytes(report) + b"\n"
        )
        if report["status"] == "SCORE_REPLAY_PREFLIGHT_OK":
            (temporary / "SCORE_REPLAY_PREFLIGHT_OK").write_bytes(
                canonical_json_bytes({
                    "status": report["status"],
                    "bundles": report["validated_bundles"],
                    "candidates": report["validated_candidates"],
                    "attribution_source_commit": report["attribution_source_commit"],
                }) + b"\n"
            )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--attribution-config", type=Path, required=True)
    parser.add_argument("--training-result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", type=int, default=1)
    parser.add_argument("--expected-foodwebs", type=int, default=290)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--attribution-source-commit", required=True)
    args = parser.parse_args(argv)

    if args.require_cuda and args.device != "cuda":
        raise ValueError("--require-cuda requires --device cuda")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")

    with open(args.run_config, encoding="utf-8") as handle:
        run_config = json.load(handle)
    with open(args.attribution_config, encoding="utf-8") as handle:
        attribution_config = json.load(handle)
    if attribution_config.get("schema_version") != "seal-directed-node-ecc-attribution-config-v3":
        raise ValueError("preflight requires an attribution-v3 config")
    run_hash = configuration_hash(run_config)
    validate_manifest_lock(args.manifest, args.run_config)
    rows = [
        row for row in read_manifest(args.manifest)
        if int(row["ExperimentID"]) == int(args.experiment_id)
    ]
    expected = int(args.expected_foodwebs) * 2
    if len(rows) != expected:
        raise ValueError("expected {} preflight rows, found {}".format(expected, len(rows)))
    if {row.get("ConfigHash") for row in rows} != {run_hash}:
        raise ValueError("preflight manifest rows do not match RUN_CONFIG")

    predictive_tolerance = float(attribution_config["score_parity_absolute_tolerance"])
    attribution_tolerance = float(
        attribution_config["attribution_forward_absolute_tolerance"]
    )
    require_agreement = bool(attribution_config["require_prediction_agreement"])
    output_rows, all_drifts, errors = [], [], []
    candidate_total = 0

    for row in rows:
        identity = "{} {} exp {}".format(
            row["ModelVariant"], row["Foodweb"], row["ExperimentID"]
        )
        try:
            training_dir = run_directory(
                args.training_result_root,
                row["ModelVariant"],
                row["Foodweb"],
                row["ExperimentID"],
                row["Seed"],
            )
            training = validate_run_bundle(training_dir, run_hash)
            with open(training_dir / "model_config.json", encoding="utf-8") as handle:
                model_config = json.load(handle)
            model = build_classifier(
                row["ModelVariant"],
                model_config["node_attribute_dim"],
                model_config["label_dim"],
                model_config["sortpooling_k"],
                edge_chunk_size=model_config["edge_chunk_size"],
                latent_dims=tuple(model_config["latent_dimensions"]),
                hidden=model_config["hidden_dimension"],
                dropout=model_config["dropout"],
            ).to(device)
            load_checkpoint(model, training_dir / "checkpoint.pt", row["ModelVariant"])
            with np.load(training_dir / "artifacts.npz", allow_pickle=False) as artifact:
                test_graphs = unpack_graphs(artifact, "test")
                predictive_scores = np.asarray(artifact["test_scores"], dtype=np.float64)
            replay_batch_size = int(model_config["batch_size"])
            batched_difference = validate_forward_parity(
                model,
                test_graphs,
                predictive_scores,
                model_config["label_dim"],
                replay_batch_size,
                device,
                predictive_tolerance,
            )
            drifts, disagreements = [], 0
            threshold = float(training["metrics"]["Threshold"])
            for candidate_index, (graph, predictive_score) in enumerate(
                zip(test_graphs, predictive_scores)
            ):
                drnl = one_hot_drnl(graph, model_config["label_dim"], device)
                node_attributes = torch.as_tensor(
                    np.asarray(graph.node_features, dtype=np.float32), device=device
                )
                edge_attributes = torch.as_tensor(
                    np.asarray(graph.edge_attr, dtype=np.float32), device=device
                )
                attribution_score, _ = _prediction(
                    model, graph, drnl, node_attributes, edge_attributes
                )
                drift, agreement = validate_candidate_score_replay(
                    predictive_score,
                    attribution_score,
                    threshold,
                    attribution_tolerance,
                    require_agreement,
                    candidate_index,
                )
                drifts.append(drift)
                disagreements += int(not agreement)
            output_rows.append({
                "ModelVariant": row["ModelVariant"],
                "Foodweb": row["Foodweb"],
                "ExperimentID": int(row["ExperimentID"]),
                "Seed": int(row["Seed"]),
                "Candidates": len(drifts),
                "PredictiveReplayMaxAbsDiff": batched_difference,
                "AttributionForwardMaxAbsDrift": float(np.max(drifts)),
                "AttributionForwardMedianAbsDrift": float(np.median(drifts)),
                "PredictionDisagreements": disagreements,
                "Status": "OK",
            })
            all_drifts.extend(drifts)
            candidate_total += len(drifts)
        except Exception as error:
            errors.append("{}: {}".format(identity, error))
            output_rows.append({
                "ModelVariant": row["ModelVariant"],
                "Foodweb": row["Foodweb"],
                "ExperimentID": int(row["ExperimentID"]),
                "Seed": int(row["Seed"]),
                "Candidates": 0,
                "PredictiveReplayMaxAbsDiff": "",
                "AttributionForwardMaxAbsDrift": "",
                "AttributionForwardMedianAbsDrift": "",
                "PredictionDisagreements": "",
                "Status": "FAILED: {}".format(error),
            })

    status = "SCORE_REPLAY_PREFLIGHT_OK" if not errors else "SCORE_REPLAY_PREFLIGHT_FAILED"
    drift_summary = None
    if all_drifts:
        drift_summary = {
            "maximum": float(np.max(all_drifts)),
            "median": float(np.median(all_drifts)),
            "p95": float(np.percentile(all_drifts, 95)),
            "p99": float(np.percentile(all_drifts, 99)),
            "tolerance": attribution_tolerance,
        }
    report = {
        "schema_version": "seal-directed-node-ecc-score-replay-preflight-v3",
        "status": status,
        "expected_bundles": expected,
        "validated_bundles": expected - len(errors),
        "validated_candidates": candidate_total,
        "experiment_id": int(args.experiment_id),
        "predictive_replay_absolute_tolerance": predictive_tolerance,
        "attribution_forward_score_drift": drift_summary,
        "require_prediction_agreement": require_agreement,
        "errors": errors,
        "attribution_source_commit": args.attribution_source_commit,
        "run_config_hash": run_hash,
        "attribution_config_hash": configuration_hash(attribution_config),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "hostname": socket.gethostname(),
    }
    atomic_preflight_directory(args.output_dir, output_rows, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
