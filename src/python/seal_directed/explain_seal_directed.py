#!/usr/bin/env python3
"""Explain one immutable SEAL-D-NodeAttr training bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR.parent
if str(PYTHON_DIR / "pytorch_DGCNN") not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR / "pytorch_DGCNN"))

from main import Classifier
from util import GNNGraph, cmd_args

from attribution import CONFUSION_CODES, NODE_ROLE_CODES, attribute_test_graphs
from attribution_artifacts import write_attribution_bundle
from graph_artifacts import unpack_graphs
from nodeattr_schema import MODEL_ID, feature_names_from_mat, validate_feature_schema
from seal_run_artifacts import (
    configuration_hash,
    file_sha256,
    node_order_hash,
    validate_run_bundle,
)
from util_functions_directed import binarize_adjacency


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def configure_model(model_config, device):
    for name in (
        "gm", "feat_dim", "attr_dim", "edge_feat_dim", "sortpooling_k",
        "latent_dim", "hidden", "out_dim", "dropout", "num_class",
        "conv1d_activation",
    ):
        if name not in model_config:
            raise ValueError("model_config.json is missing {}".format(name))
        setattr(cmd_args, name, model_config[name])
    cmd_args.mode = "gpu" if device.type == "cuda" else "cpu"
    cmd_args.device = str(device)
    classifier = Classifier().to(device)
    return classifier


def load_checkpoint(classifier, checkpoint_path):
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if payload.get("schema_version") != "seal-directed-checkpoint-v1":
        raise ValueError("Unexpected checkpoint schema")
    classifier.load_state_dict(payload["state_dict"], strict=True)
    classifier.eval()
    return payload


RANDOMIZATION_STAGES = ("output", "mlp", "sortpool_head", "message_passing")


def progressive_randomize(classifier, through_stage, seed):
    if through_stage not in RANDOMIZATION_STAGES:
        raise ValueError("Unknown randomization stage: {}".format(through_stage))
    torch.manual_seed(int(seed))
    if next(classifier.parameters()).is_cuda:
        torch.cuda.manual_seed_all(int(seed))
    groups = {
        "output": [classifier.mlp.h2_weights],
        "mlp": [classifier.mlp.h1_weights],
        "sortpool_head": [
            classifier.gnn.conv1d_params2,
            classifier.gnn.conv1d_params1,
        ] + ([classifier.gnn.out_params] if hasattr(classifier.gnn, "out_params") else []),
        "message_passing": list(classifier.gnn.conv_params),
    }
    for stage in RANDOMIZATION_STAGES:
        for module in groups[stage]:
            module.reset_parameters()
        if stage == through_stage:
            break
    buffer = io.BytesIO()
    torch.save(
        {key: value.detach().cpu() for key, value in classifier.state_dict().items()},
        buffer,
    )
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def validate_explicit_forward_parity(classifier, graphs, expected_scores, feat_dim, batch_size, tolerance):
    explicit_scores = []
    legacy_scores = []
    device = next(classifier.parameters()).device
    with torch.no_grad():
        for start in range(0, len(graphs), int(batch_size)):
            batch = graphs[start:start + int(batch_size)]
            tags = []
            attributes = []
            for graph in batch:
                labels = torch.as_tensor(graph.node_tags, dtype=torch.long, device=device)
                one_hot = torch.zeros((graph.num_nodes, int(feat_dim)), device=device)
                one_hot.scatter_(1, labels.view(-1, 1), 1.0)
                tags.append(one_hot)
                attributes.append(torch.as_tensor(graph.node_features, dtype=torch.float32, device=device))
            explicit = classifier.forward_prepared(
                batch,
                torch.cat(tags, dim=0),
                node_attributes=torch.cat(attributes, dim=0),
                labels=None,
            )
            legacy = classifier(batch)[0]
            if not torch.equal(explicit, legacy):
                difference = float(torch.max(torch.abs(explicit - legacy)).cpu())
                if difference > float(tolerance):
                    raise ValueError("Legacy/explicit forward parity failed: {}".format(difference))
            explicit_scores.append(torch.exp(explicit[:, 1]).cpu().numpy())
            legacy_scores.append(torch.exp(legacy[:, 1]).cpu().numpy())
    explicit_scores = np.concatenate(explicit_scores)
    legacy_scores = np.concatenate(legacy_scores)
    expected_scores = np.asarray(expected_scores, dtype=np.float64)
    if not np.allclose(explicit_scores, expected_scores, rtol=0.0, atol=float(tolerance)):
        raise ValueError(
            "Frozen score parity failed; maximum absolute difference={}".format(
                float(np.max(np.abs(explicit_scores - expected_scores)))
            )
        )
    if not np.array_equal(explicit_scores, legacy_scores):
        raise ValueError("Activating the explicit attribution forward changed prediction scores")
    return float(np.max(np.abs(explicit_scores - expected_scores)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run-dir", type=Path, required=True)
    parser.add_argument("--mat-path", type=Path, required=True)
    parser.add_argument("--attribution-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--randomize-through", choices=RANDOMIZATION_STAGES, default=None,
        help="sanity check: progressively reinitialize parameters through this stage",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    args.training_run_dir = args.training_run_dir.resolve()
    args.mat_path = args.mat_path.resolve()
    args.attribution_config = args.attribution_config.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.require_cuda and args.device != "cuda":
        raise RuntimeError("--require-cuda requires --device cuda")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    training_payload = validate_run_bundle(args.training_run_dir)
    if training_payload.get("schema_version") != "seal-directed-run-v2":
        raise ValueError("Attribution requires a version-2 training bundle with checkpoint")
    metrics = training_payload["metrics"]
    provenance = training_payload["provenance"]
    model_config = load_json(args.training_run_dir / "model_config.json")
    attribution_config = load_json(args.attribution_config)
    attribution_config_hash = configuration_hash(attribution_config)
    permitted_training_ids = {MODEL_ID}
    if bool(metrics.get("LabelPermutationControl", 0)):
        permitted_training_ids.add("SEAL_directed_nodeattr_permuted_labels_control_v1")
    if (
        model_config.get("ModelID") not in permitted_training_ids
        or attribution_config.get("ModelID") != MODEL_ID
    ):
        raise ValueError("Training and attribution configs do not target the node-attribute family")
    try:
        import captum
    except ImportError as error:
        raise RuntimeError("captum==0.9.0 is required") from error
    if captum.__version__ != str(attribution_config.get("captum_version")):
        raise ValueError(
            "Captum version mismatch: config={}, runtime={}".format(
                attribution_config.get("captum_version"), captum.__version__
            )
        )

    data = sio.loadmat(args.mat_path)
    attributes = data["group"]
    if hasattr(attributes, "toarray"):
        attributes = attributes.toarray()
    attributes = np.asarray(attributes, dtype=np.float32)
    feature_schema = validate_feature_schema(feature_names_from_mat(data))
    if feature_schema["feature_schema_hash"] != model_config["feature_schema_hash"]:
        raise ValueError("MAT feature schema does not match checkpoint")
    if attributes.shape[1] != int(model_config["attr_dim"]):
        raise ValueError("MAT attribute width does not match checkpoint")
    net = binarize_adjacency(data["net"])
    if node_order_hash(data.get("taxonomy"), net) != provenance.get("node_order_hash"):
        raise ValueError("MAT node order does not match training bundle")

    checkpoint_path = args.training_run_dir / "checkpoint.pt"
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != model_config["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum does not match model_config.json")
    classifier = configure_model(model_config, device)
    checkpoint = load_checkpoint(classifier, checkpoint_path)
    if checkpoint.get("config_hash") != training_payload.get("config_hash"):
        raise ValueError("Checkpoint RUN_CONFIG hash mismatch")

    with np.load(args.training_run_dir / "artifacts.npz", allow_pickle=False) as artifact:
        train_graphs = unpack_graphs(artifact, "train", attributes, GNNGraph)
        test_graphs = unpack_graphs(artifact, "test", attributes, GNNGraph)
        expected_scores = np.asarray(artifact["test_scores"], dtype=np.float64)
        test_pairs = np.vstack([artifact["test_pos"], artifact["test_neg"]])
        if not np.array_equal(
            test_pairs,
            np.stack([graph.candidate_pair for graph in test_graphs]),
        ):
            raise ValueError("Test graph order does not match candidate pair order")

    score_parity_max_difference = None
    if not args.randomize_through:
        score_parity_max_difference = validate_explicit_forward_parity(
            classifier,
            test_graphs,
            expected_scores,
            int(model_config["feat_dim"]),
            int(model_config.get("batch_size", 50)),
            float(attribution_config["score_parity_absolute_tolerance"]),
        )

    training_seed = int(metrics["Seed"])
    attribution_seed = training_seed + int(attribution_config["attribution_seed_offset"])
    used_checkpoint_hash = checkpoint_hash
    if args.randomize_through:
        used_checkpoint_hash = progressive_randomize(
            classifier,
            args.randomize_through,
            seed=training_seed + 3000000,
        )
    torch.manual_seed(attribution_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(attribution_seed)
    arrays = attribute_test_graphs(
        classifier,
        train_graphs,
        test_graphs,
        None if args.randomize_through else expected_scores,
        feat_dim=int(model_config["feat_dim"]),
        attribution_seed=attribution_seed,
        expected_gradients_samples=int(attribution_config["expected_gradients_samples"]),
        drnl_steps=int(attribution_config["drnl_steps"]),
        threshold=float(metrics["Threshold"]),
        internal_batch_size=int(attribution_config["internal_batch_size"]),
        score_tolerance=float(attribution_config["score_parity_absolute_tolerance"]),
        feature_families=feature_schema["feature_families"],
        faithfulness_background_samples=int(
            attribution_config.get("faithfulness_background_samples", 0)
        ),
        scores_prevalidated=not bool(args.randomize_through),
    )
    if not np.array_equal(arrays["candidate_pairs"], test_pairs):
        raise ValueError("Attribution output reordered test candidates")

    elapsed = time.time() - started
    metadata = {
        "ModelID": model_config["ModelID"],
        "DisplayName": model_config["DisplayName"],
        "LegacyFamily": model_config["LegacyFamily"],
        "Foodweb": metrics["Foodweb"],
        "FoodwebIndex": int(metrics["FoodwebIndex"]),
        "ExperimentID": int(metrics["ExperimentID"]),
        "training_seed": training_seed,
        "attribution_seed": attribution_seed,
        "training_config_hash": training_payload["config_hash"],
        "source_commit": model_config["source_commit"],
        "training_bundle_success_sha256": file_sha256(args.training_run_dir / "_SUCCESS"),
        "checkpoint_sha256": used_checkpoint_hash,
        "base_checkpoint_sha256": checkpoint_hash,
        "sanity_randomization_through": args.randomize_through,
        "label_permutation_control": bool(metrics.get("LabelPermutationControl", 0)),
        "feature_schema_hash": feature_schema["feature_schema_hash"],
        "feature_names": feature_schema["feature_names"],
        "feature_families": feature_schema["feature_families"],
        "candidate_count": int(len(test_graphs)),
        "total_nodes": int(arrays["node_ids"].size),
        "background_source": "training_subgraphs_only",
        "background_training_graph_count": int(len(train_graphs)),
        "expected_gradients_samples": int(attribution_config["expected_gradients_samples"]),
        "drnl_steps": int(attribution_config["drnl_steps"]),
        "target": "log_p_positive_minus_log_p_negative",
        "confusion_codes": CONFUSION_CODES,
        "node_role_codes": NODE_ROLE_CODES,
        "eg_completeness_relative_median": float(np.median(arrays["eg_completeness_relative"])),
        "eg_completeness_relative_p95": float(np.percentile(arrays["eg_completeness_relative"], 95)),
        "drnl_completeness_relative_median": float(np.median(arrays["drnl_completeness_relative"])),
        "drnl_completeness_relative_p95": float(np.percentile(arrays["drnl_completeness_relative"], 95)),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "captum_version": captum.__version__,
        "device": str(device),
        "hostname": socket.gethostname(),
        "elapsed_seconds": elapsed,
        "language_guardrail": "model reliance on input attributes; not causal effects",
        "score_parity_max_absolute_difference": score_parity_max_difference,
    }
    write_attribution_bundle(
        args.output_dir,
        metadata,
        arrays,
        attribution_config_hash,
    )
    print("[OK] Attribution bundle: {}".format(args.output_dir))
    print("[OK] Candidates explained exactly once: {}".format(len(test_graphs)))
    print("[OK] Elapsed seconds: {:.1f}".format(elapsed))


if __name__ == "__main__":
    main()
