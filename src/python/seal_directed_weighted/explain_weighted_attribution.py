#!/usr/bin/env python3
"""Explain one attribution-ready directed-node or ECC training bundle."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from pathlib import Path

import numpy as np
import torch

from attribution_artifacts import write_attribution_bundle
from edge_schema import FEATURE_NAMES as EDGE_FEATURE_NAMES
from generate_attribution_config import ATTRIBUTION_VARIANTS
from graph_artifacts_v2 import unpack_graphs
from models import build_classifier, collate_directed
from run_artifacts import configuration_hash, file_sha256, validate_run_bundle
from weighted_attribution import attribute_test_graphs, one_hot_drnl


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def node_family(name):
    if name == "log10_mass_z": return "mass"
    if name == "mass_missing": return "mass_missing"
    if name.startswith("taxonomy_level="): return "taxonomy"
    if name.startswith("metabolic_type="): return "metabolism"
    if name.startswith("movement_type="): return "movement"
    if name == "lifestage_missing" or name.startswith("lifestage="): return "life_stage"
    raise ValueError("unmapped node feature {}".format(name))


def edge_family(index):
    if index == 0: return "mass_ratio"
    if index <= 2: return "mass_missingness"
    if index <= 10: return "interaction_type"
    if index <= 13: return "interaction_dimensionality"
    return "interaction_classification"


def load_checkpoint(model, path, variant):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema_version") != "seal-directed-weighted-checkpoint-v2":
        raise ValueError("unexpected checkpoint schema")
    if payload.get("ModelVariant") != variant:
        raise ValueError("checkpoint variant mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return payload


def prepared_scores(model, graphs, label_dim, batch_size, device, permute_edges=False):
    scores = []
    with torch.no_grad():
        for start in range(0, len(graphs), int(batch_size)):
            batch = graphs[start:start + int(batch_size)]
            node_input, edge_index, edge_attr, sizes, _ = collate_directed(batch, label_dim, device)
            drnl, attributes = node_input[:, :label_dim], node_input[:, label_dim:]
            if permute_edges and len(edge_attr):
                edge_attr = torch.roll(edge_attr, shifts=1, dims=0)
            output = model.forward_prepared(drnl, attributes, edge_index, edge_attr, sizes)
            scores.append(torch.exp(output[:, 1]).cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float64)


def validate_forward_parity(model, graphs, expected_scores, label_dim, batch_size, device, tolerance):
    legacy, explicit = [], []
    with torch.no_grad():
        for start in range(0, len(graphs), int(batch_size)):
            batch = graphs[start:start + int(batch_size)]
            legacy.append(torch.exp(model(batch)[:, 1]).cpu().numpy())
    legacy = np.concatenate(legacy)
    explicit = prepared_scores(model, graphs, label_dim, batch_size, device)
    maximum = float(max(np.max(np.abs(legacy - explicit)),
                        np.max(np.abs(explicit - np.asarray(expected_scores)))))
    if maximum > float(tolerance):
        raise ValueError("forward/checkpoint score parity failed: {:.12g}".format(maximum))
    return maximum


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run-dir", type=Path, required=True)
    parser.add_argument("--attribution-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-run-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    args.training_run_dir = args.training_run_dir.resolve()
    args.attribution_config = args.attribution_config.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.require_cuda and args.device != "cuda":
        raise ValueError("--require-cuda requires --device cuda")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    training = validate_run_bundle(args.training_run_dir)
    if training.get("schema_version") != "seal-directed-weighted-attribution-run-v2":
        raise ValueError("attribution requires an attribution-v2 training bundle")
    model_config = load_json(args.training_run_dir / "model_config.json")
    config = load_json(args.attribution_config)
    config_hash = configuration_hash(config)
    variant = model_config["ModelVariant"]
    if variant not in ATTRIBUTION_VARIANTS or variant not in config.get("model_variants", []):
        raise ValueError("training/config variant mismatch")
    try:
        import captum
    except ImportError as error:
        raise RuntimeError("captum=={} is required".format(config["captum_version"])) from error
    if captum.__version__ != str(config["captum_version"]):
        raise ValueError("Captum version does not match attribution config")
    model = build_classifier(
        variant, model_config["node_attribute_dim"], model_config["label_dim"],
        model_config["sortpooling_k"], edge_chunk_size=model_config["edge_chunk_size"],
        latent_dims=tuple(model_config["latent_dimensions"]),
        hidden=model_config["hidden_dimension"], dropout=model_config["dropout"],
    ).to(device)
    load_checkpoint(model, args.training_run_dir / "checkpoint.pt", variant)
    with np.load(args.training_run_dir / "artifacts.npz", allow_pickle=False) as artifact:
        train_graphs = unpack_graphs(artifact, "train")
        test_graphs = unpack_graphs(artifact, "test")
        expected_scores = np.asarray(artifact["test_scores"], dtype=np.float64)
    tolerance = float(config["score_parity_absolute_tolerance"])
    parity = validate_forward_parity(
        model, test_graphs, expected_scores, model_config["label_dim"],
        args.batch_size, device, tolerance,
    )
    edge_invariance = None
    if variant == "directed_node":
        permuted = prepared_scores(
            model, test_graphs, model_config["label_dim"], args.batch_size,
            device, permute_edges=True,
        )
        edge_invariance = float(np.max(np.abs(permuted - expected_scores)))
        if edge_invariance > tolerance:
            raise ValueError("directed_node changed after edge-attribute permutation")
    historical = {"available": False, "exact_replay": False, "max_abs_score_difference": None}
    if args.historical_run_dir is not None:
        historical_dir = args.historical_run_dir.resolve()
        validate_run_bundle(historical_dir)
        with np.load(historical_dir / "artifacts.npz", allow_pickle=False) as old:
            old_scores = np.asarray(old["test_scores"], dtype=np.float64)
            old_labels = np.asarray(old["test_labels"])
            old_pairs = np.vstack([old["test_pos"], old["test_neg"]])
        v2_pairs = np.stack([np.asarray(graph.candidate, dtype=np.int64) for graph in test_graphs])
        if old_scores.shape != expected_scores.shape or not np.array_equal(
            old_labels, np.asarray([graph.graph_label for graph in test_graphs])
        ) or not np.array_equal(old_pairs, v2_pairs):
            raise ValueError("historical candidates/labels do not align with v2")
        difference = float(np.max(np.abs(old_scores - expected_scores)))
        historical = {"available": True, "exact_replay": difference <= tolerance,
                      "max_abs_score_difference": difference}
    node_names = list(model_config["node_feature_names"])
    edge_names = list(model_config["edge_feature_names"])
    arrays = attribute_test_graphs(
        model, variant, train_graphs, test_graphs, expected_scores,
        model_config["label_dim"],
        int(training["metrics"]["Seed"]) + int(config["attribution_seed_offset"]),
        int(config["expected_gradients_samples"]), int(config["drnl_steps"]),
        edge_samples=(int(config["expected_gradients_samples"])
                      if variant == "directed_ecc" else None),
        threshold=float(training["metrics"]["Threshold"]),
        internal_batch_size=int(config["internal_batch_size"]),
        score_tolerance=tolerance,
        node_feature_families=[node_family(name) for name in node_names],
        edge_feature_families=[edge_family(index) for index in range(len(edge_names))],
        faithfulness_background_samples=int(config.get("faithfulness_background_samples", 0)),
    )
    checkpoint_hash = file_sha256(args.training_run_dir / "checkpoint.pt")
    metadata = {
        "Version": config["version"], "ModelID": model_config["ModelID"],
        "ModelVariant": variant, "VariantDisplayName": model_config["VariantDisplayName"],
        "Foodweb": training["metrics"]["Foodweb"],
        "ExperimentID": int(training["metrics"]["ExperimentID"]),
        "training_seed": int(training["metrics"]["Seed"]),
        "attribution_seed": int(training["metrics"]["Seed"]) + int(config["attribution_seed_offset"]),
        "training_bundle_success_sha256": file_sha256(args.training_run_dir / "_SUCCESS"),
        "checkpoint_sha256": checkpoint_hash, "graph_input_sha256": model_config["graph_input_sha256"],
        "source_commit": model_config["source_commit"], "target": config["target"],
        "node_feature_names": node_names,
        "node_feature_families": [node_family(name) for name in node_names],
        "edge_feature_names": edge_names,
        "edge_feature_families": [edge_family(index) for index in range(len(edge_names))],
        "background_training_graph_count": len(train_graphs),
        "expected_gradients_samples": int(config["expected_gradients_samples"]),
        "drnl_steps": int(config["drnl_steps"]),
        "faithfulness_background_samples": int(
            config.get("faithfulness_background_samples", 0)
        ),
        "score_parity_max_abs_diff": parity,
        "directed_node_edge_invariance_max_abs_diff": edge_invariance,
        "historical_comparison": historical, "device": str(device),
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "captum_version": captum.__version__, "hostname": socket.gethostname(),
        "elapsed_seconds": time.time() - started,
        "interpretation": "model dependence, not causal ecological importance",
    }
    write_attribution_bundle(args.output_dir, metadata, arrays, config_hash)
    print(json.dumps({"status": "ATTRIBUTION_OK", "variant": variant,
                      "candidates": len(arrays["candidate_pairs"]),
                      "elapsed_seconds": time.time() - started}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
