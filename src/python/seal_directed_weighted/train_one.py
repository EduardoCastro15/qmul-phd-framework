#!/usr/bin/env python3
"""Train one frozen SEAL-directed-weighted run and write an atomic bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import platform
import random
import socket
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
from sklearn import metrics as sklearn_metrics

from edge_schema import FEATURE_NAMES, SCHEMA_SHA256, preprocess_training_edge_attributes
from models import (
    DISPLAY_NAME,
    LEGACY_FAMILY,
    MODEL_ID,
    VARIANTS,
    VARIANT_DISPLAY_NAMES,
    build_classifier,
    choose_sortpooling_k,
)
from graph_artifacts_v2 import pack_graphs, packed_graph_hash
from run_artifacts import canonical_json_bytes, file_sha256, write_run_bundle
from split_bundles import load_split_bundle
from subgraphs import extract_dataset


def matlab_string(value):
    current = value
    while isinstance(current, np.ndarray):
        if current.size == 0:
            return ""
        current = current.reshape(-1)[0]
    return str(current).strip()


def node_feature_schema(data):
    if "group_feature_names" not in data:
        raise ValueError("attribution_v2 requires group_feature_names in the weighted MAT")
    names = [
        matlab_string(value)
        for value in np.asarray(data["group_feature_names"]).reshape(-1)
    ]
    if len(names) != 44 or any(not name for name in names) or len(set(names)) != 44:
        raise ValueError("group_feature_names must contain 44 unique non-empty names")
    digest = hashlib.sha256(canonical_json_bytes(names)).hexdigest()
    return names, digest


def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def prediction_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    roc_auc = sklearn_metrics.roc_auc_score(labels, scores)
    precision_curve, recall_curve, _ = sklearn_metrics.precision_recall_curve(labels, scores)
    pr_auc = sklearn_metrics.auc(recall_curve[::-1], precision_curve[::-1])
    predictions = scores > float(threshold)
    positive = labels == 1
    negative = ~positive
    tp = int(np.sum(predictions & positive))
    fp = int(np.sum(predictions & negative))
    fn = int(np.sum(~predictions & positive))
    tn = int(np.sum(~predictions & negative))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    mcc_denominator = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_divide(tp * tn - fn * fp, mcc_denominator)
    tnr = safe_divide(tn, tn + fp)
    return {
        "ROC_AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "Precision": precision,
        "Recall": recall,
        "F1Score": f1,
        "TestMCC": mcc,
        "TestTSS": recall + tnr - 1.0,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Threshold": float(threshold),
    }


def _batch_loss(model, graphs, device, optimizer=None):
    log_probabilities = model(graphs)
    labels = torch.tensor([graph.graph_label for graph in graphs], dtype=torch.long, device=device)
    loss = F.nll_loss(log_probabilities, labels)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    correct = int((log_probabilities.argmax(dim=1) == labels).sum().item())
    return float(loss.detach().cpu()), correct, len(graphs), log_probabilities.detach()


def evaluate(model, graphs, batch_size, device, return_scores=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    labels, scores = [], []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = graphs[start: start + batch_size]
            loss, correct, count, output = _batch_loss(model, batch, device)
            total_loss += loss * count
            total_correct += correct
            total += count
            if return_scores:
                labels.extend(graph.graph_label for graph in batch)
                scores.extend(output[:, 1].exp().cpu().numpy().tolist())
    result = (total_loss / total, total_correct / float(total))
    if return_scores:
        return result + (np.asarray(labels, dtype=np.int8), np.asarray(scores, dtype=float))
    return result


def fit_model(model, train_graphs, validation_graphs, epochs, batch_size, learning_rate,
              seed, device, log_every=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    best_loss, best_epoch, best_state = None, None, None
    for epoch in range(int(epochs)):
        model.train()
        order = np.random.RandomState(int(seed) + epoch).permutation(len(train_graphs))
        for start in range(0, len(order), int(batch_size)):
            batch = [train_graphs[index] for index in order[start: start + int(batch_size)]]
            _batch_loss(model, batch, device, optimizer=optimizer)
        validation_loss, validation_accuracy = evaluate(
            model, validation_graphs, int(batch_size), device
        )
        if best_loss is None or validation_loss <= best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 0 or (epoch + 1) % int(log_every) == 0 or epoch + 1 == int(epochs):
            print(
                "[EPOCH] {} validation_loss={:.6f} validation_accuracy={:.6f}".format(
                    epoch + 1, validation_loss, validation_accuracy
                )
            )
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return best_epoch, float(best_loss)


def load_mat(path):
    data = sio.loadmat(path, spmatrix=True)
    required = {"net", "group", "edge_index", "edge_attr_raw", "edge_attr_schema_sha256"}
    missing = required.difference(data)
    if missing:
        raise ValueError("weighted MAT missing fields: {}".format(sorted(missing)))
    if matlab_string(data["edge_attr_schema_sha256"]) != SCHEMA_SHA256:
        raise ValueError("edge schema hash mismatch in {}".format(path))
    net = ssp.csr_matrix(data["net"])
    net.setdiag(0)
    net.eliminate_zeros()
    net.data = np.ones_like(net.data)
    group = data["group"].toarray() if ssp.issparse(data["group"]) else np.asarray(data["group"])
    group = np.asarray(group, dtype=np.float32)
    if group.shape != (net.shape[0], 44):
        raise ValueError("expected 44 node attributes, found {}".format(group.shape))
    edges = np.asarray(data["edge_index"], dtype=np.int64).reshape(-1, 2)
    edge_attr = np.asarray(data["edge_attr_raw"], dtype=np.float32).reshape(-1, 17)
    return data, net, group, edges, edge_attr


def run(args):
    start_time = time.time()
    artifact_mode = getattr(args, "artifact_mode", "v1")
    if artifact_mode not in {"v1", "attribution_v2"}:
        raise ValueError("Unknown artifact mode: {}".format(artifact_mode))
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if args.variant not in VARIANTS:
        raise ValueError(args.variant)
    if artifact_mode == "attribution_v2" and args.variant not in {
        "directed_node", "directed_ecc"
    }:
        raise ValueError("attribution_v2 supports only directed_node and directed_ecc")
    if args.require_cuda and args.device != "cuda":
        raise ValueError("--require-cuda requires --device cuda")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    data, net, node_features, raw_edges, raw_edge_attr = load_mat(args.mat_path)
    feature_names = feature_schema_sha256 = None
    if artifact_mode == "attribution_v2":
        feature_names, feature_schema_sha256 = node_feature_schema(data)
    splits, observed_split_hash = load_split_bundle(
        args.split_bundle,
        expected_file_hash=args.split_file_sha256,
        expected_logical_hash=args.split_hash,
    )
    train_adjacency = net.copy().tolil()
    if len(splits["test_pos"]):
        train_adjacency[splits["test_pos"][:, 0], splits["test_pos"][:, 1]] = 0
    train_adjacency = train_adjacency.tocsr()
    train_adjacency.eliminate_zeros()
    context_edges, context_attr, preprocessing = preprocess_training_edge_attributes(
        raw_edges, raw_edge_attr, train_adjacency
    )
    expected_context = int(net.nnz - len(splits["test_pos"]))
    if context_edges.shape[0] != expected_context:
        raise ValueError(
            "training context edge count mismatch: expected {}, found {}".format(
                expected_context, context_edges.shape[0]
            )
        )

    extraction_start = time.time()
    train_graphs = extract_dataset(
        train_adjacency, node_features, context_edges, context_attr,
        splits["train_pos"], splits["train_neg"], hop=args.hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        num_workers=args.num_workers,
    )
    test_graphs = extract_dataset(
        train_adjacency, node_features, context_edges, context_attr,
        splits["test_pos"], splits["test_neg"], hop=args.hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        num_workers=args.num_workers,
    )
    extraction_seconds = time.time() - extraction_start
    all_graphs = train_graphs + test_graphs
    max_label = max(int(graph.node_labels.max()) for graph in all_graphs)
    label_dim = max_label + 1
    sortpooling_k = choose_sortpooling_k(all_graphs, 0.6)

    validation_count = max(1, int(0.1 * len(train_graphs)))
    validation_order = np.random.RandomState(args.seed + 1000003).permutation(len(train_graphs))
    validation_indices = set(validation_order[:validation_count].tolist())
    validation_graphs = [g for index, g in enumerate(train_graphs) if index in validation_indices]
    fit_graphs = [g for index, g in enumerate(train_graphs) if index not in validation_indices]
    if not fit_graphs:
        raise ValueError("training set is empty after validation split")

    model = build_classifier(
        args.variant,
        node_attribute_dim=node_features.shape[1],
        label_dim=label_dim,
        sortpooling_k=sortpooling_k,
        edge_chunk_size=args.edge_chunk_size,
    ).to(device)
    training_start = time.time()
    best_epoch, best_validation_loss = fit_model(
        model, fit_graphs, validation_graphs, args.num_epochs, args.batch_size,
        args.learning_rate, args.seed, device, args.log_every,
    )
    test_loss, test_accuracy, test_labels, test_scores = evaluate(
        model, test_graphs, args.batch_size, device, return_scores=True
    )
    checkpoint_bytes = model_config = None
    checkpoint_score_max_abs_diff = None
    if artifact_mode == "attribution_v2":
        checkpoint_buffer = io.BytesIO()
        portable_state_dict = {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        }
        torch.save(
            {
                "schema_version": "seal-directed-weighted-checkpoint-v2",
                "state_dict": portable_state_dict,
                "best_epoch": int(best_epoch),
                "config_hash": args.config_hash,
                "ModelID": MODEL_ID,
                "ModelVariant": args.variant,
            },
            checkpoint_buffer,
        )
        checkpoint_bytes = checkpoint_buffer.getvalue()
        restored_payload = torch.load(
            io.BytesIO(checkpoint_bytes), map_location=device, weights_only=False
        )
        restored_model = build_classifier(
            args.variant,
            node_attribute_dim=node_features.shape[1],
            label_dim=label_dim,
            sortpooling_k=sortpooling_k,
            edge_chunk_size=args.edge_chunk_size,
        ).to(device)
        restored_model.load_state_dict(restored_payload["state_dict"])
        _, _, restored_labels, restored_scores = evaluate(
            restored_model, test_graphs, args.batch_size, device, return_scores=True
        )
        if not np.array_equal(test_labels, restored_labels):
            raise ValueError("checkpoint reload changed test label order")
        checkpoint_score_max_abs_diff = float(
            np.max(np.abs(test_scores - restored_scores)) if test_scores.size else 0.0
        )
        if checkpoint_score_max_abs_diff > 1e-7:
            raise ValueError(
                "checkpoint score parity failed: max abs diff {:.12g}".format(
                    checkpoint_score_max_abs_diff
                )
            )
    training_seconds = time.time() - training_start
    metric_values = prediction_metrics(test_labels, test_scores, args.threshold)
    gate_min = gate_max = gate_mean = None
    if args.variant == "directed_scalar_gate":
        model.eval()
        with torch.no_grad():
            edge_tensor = torch.as_tensor(context_attr, dtype=torch.float32, device=device)
            gates = model.encoder.scalar_gate(edge_tensor).cpu().numpy().reshape(-1)
        gate_min, gate_max, gate_mean = map(float, (gates.min(), gates.max(), gates.mean()))
        if gate_min < 0 or gate_max > 1 or not np.all(np.isfinite(gates)):
            raise ValueError("invalid scalar gates")

    total_seconds = time.time() - start_time
    metrics_row = {
        "Version": (
            "SEAL_directed_node_ecc_attribution_v2"
            if artifact_mode == "attribution_v2" else "SEAL_directed_weighted_v1"
        ),
        "ModelID": MODEL_ID,
        "DisplayName": DISPLAY_NAME,
        "LegacyFamily": LEGACY_FAMILY,
        "ModelVariant": args.variant,
        "VariantDisplayName": VARIANT_DISPLAY_NAMES[args.variant],
        "Foodweb": args.foodweb,
        "FoodwebIndex": int(args.foodweb_index),
        "ExperimentID": int(args.experiment_id),
        "Seed": int(args.seed),
        "RequestedTrainRatio": 90.0,
        "RealizedTrainRatio": 100.0 * len(splits["train_pos"]) / (
            len(splits["train_pos"]) + len(splits["test_pos"])
        ),
        "TrainLinks": len(splits["train_pos"]),
        "TestLinks": len(splits["test_pos"]),
        "ThresholdMode": "fixed",
        "BestEpoch": int(best_epoch),
        "BestValidationLoss": best_validation_loss,
        "TestLoss": float(test_loss),
        "TestAccuracy": float(test_accuracy),
        "SortPoolingK": int(sortpooling_k),
        "MaxNodeLabel": int(max_label),
        "NodeAttributeDim": int(node_features.shape[1]),
        "EdgeAttributeDim": len(FEATURE_NAMES),
        "SubgraphWorkers": int(args.num_workers),
        "SplitHash": observed_split_hash,
        "RolePoolFallbackUsed": int(args.role_pool_fallback_used),
        "GateMin": gate_min,
        "GateMax": gate_max,
        "GateMean": gate_mean,
        "SubgraphExtractionSeconds": extraction_seconds,
        "TrainingInferenceSeconds": training_seconds,
        "TimeElapsedSeconds": total_seconds,
    }
    if artifact_mode == "attribution_v2":
        metrics_row.update({
            "ArtifactMode": artifact_mode,
            "CheckpointScoreMaxAbsDiff": checkpoint_score_max_abs_diff,
            "AttributionTarget": "log_p_positive_minus_log_p_negative",
        })
    metrics_row.update(metric_values)
    provenance = {
        "schema_version": (
            "seal-directed-weighted-provenance-v2"
            if artifact_mode == "attribution_v2"
            else "seal-directed-weighted-provenance-v1"
        ),
        "source_commit": args.source_commit,
        "config_hash": args.config_hash,
        "mat_sha256": file_sha256(args.mat_path),
        "split_file_sha256": file_sha256(args.split_bundle),
        "split_hash": observed_split_hash,
        "edge_schema_sha256": SCHEMA_SHA256,
        "edge_semantics": "contextual_covariates_not_interaction_strength",
        "target_edge_policy": "candidate_edge_and_covariates_removed",
        "preprocessing": preprocessing,
        "validation_indices_sha256": __import__("hashlib").sha256(
            np.asarray(sorted(validation_indices), dtype=np.int64).tobytes()
        ).hexdigest(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_total_memory": (
            int(torch.cuda.get_device_properties(0).total_memory)
            if device.type == "cuda" else None
        ),
        "hostname": socket.gethostname(),
    }
    arrays = dict(splits)
    arrays.update(
        test_labels=test_labels,
        test_scores=test_scores,
        validation_indices=np.asarray(sorted(validation_indices), dtype=np.int64),
        context_edge_index=context_edges,
    )
    extra_files = None
    if artifact_mode == "attribution_v2":
        arrays["context_edge_attr"] = context_attr
        arrays.update(pack_graphs(train_graphs, "train"))
        arrays.update(pack_graphs(test_graphs, "test"))
        graph_input_sha256 = packed_graph_hash(arrays)
        provenance.update({
            "graph_input_sha256": graph_input_sha256,
            "node_feature_schema_sha256": feature_schema_sha256,
            "node_alignment": "focal_resource_focal_consumer_sorted_context",
            "directed_edge_order": "frozen_exact_local_csr_order",
            "checkpoint_score_absolute_tolerance": 1e-7,
        })
        checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        model_config = {
            "schema_version": "seal-directed-weighted-model-config-v2",
            "ModelID": MODEL_ID,
            "DisplayName": DISPLAY_NAME,
            "LegacyFamily": LEGACY_FAMILY,
            "ModelVariant": args.variant,
            "VariantDisplayName": VARIANT_DISPLAY_NAMES[args.variant],
            "config_hash": args.config_hash,
            "source_commit": args.source_commit,
            "checkpoint_sha256": checkpoint_sha256,
            "best_epoch": int(best_epoch),
            "label_dim": int(label_dim),
            "node_attribute_dim": int(node_features.shape[1]),
            "edge_attribute_dim": int(len(FEATURE_NAMES)),
            "sortpooling_k": int(sortpooling_k),
            "latent_dimensions": [32, 32, 32, 1],
            "hidden_dimension": 128,
            "dropout": True,
            "edge_chunk_size": int(args.edge_chunk_size),
            "batch_size": int(args.batch_size),
            "hop": int(args.hop),
            "max_nodes_per_hop": args.max_nodes_per_hop,
            "node_feature_names": list(feature_names),
            "node_feature_schema_sha256": feature_schema_sha256,
            "edge_feature_names": list(FEATURE_NAMES),
            "edge_feature_schema_sha256": SCHEMA_SHA256,
            "preprocessing": preprocessing,
            "graph_input_sha256": graph_input_sha256,
            "test_score_semantics": "positive_class_probability",
            "attribution_target": "log_p_positive_minus_log_p_negative",
        }
        extra_files = {
            "checkpoint.pt": checkpoint_bytes,
            "model_config.json": canonical_json_bytes(model_config) + b"\n",
        }
    write_run_bundle(
        args.run_dir, metrics_row, arrays, provenance, args.config_hash,
        extra_files=extra_files,
    )
    print(
        "[OK] variant={} foodweb={!r} experiment={} split={} seconds={:.1f}".format(
            args.variant, args.foodweb, args.experiment_id, observed_split_hash[:12], total_seconds
        )
    )
    return metrics_row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foodweb", required=True)
    parser.add_argument("--foodweb-index", type=int, required=True)
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--mat-path", type=Path, required=True)
    parser.add_argument("--split-bundle", type=Path, required=True)
    parser.add_argument("--split-file-sha256", required=True)
    parser.add_argument("--split-hash", required=True)
    parser.add_argument("--role-pool-fallback-used", type=int, choices=(0, 1), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument(
        "--artifact-mode", choices=("v1", "attribution_v2"), default="v1",
        help="Opt-in attribution-ready bundle; v1 remains the historical default.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--hop", type=int, default=1)
    parser.add_argument("--max-nodes-per-hop", type=int)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--edge-chunk-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
