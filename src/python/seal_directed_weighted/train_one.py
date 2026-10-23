#!/usr/bin/env python3
"""Train one frozen SEAL-directed-weighted run and write an atomic bundle."""

from __future__ import annotations

import argparse
import copy
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
from models import VARIANTS, build_classifier, choose_sortpooling_k
from run_artifacts import file_sha256, write_run_bundle
from split_bundles import load_split_bundle
from subgraphs import extract_dataset


def matlab_string(value):
    current = value
    while isinstance(current, np.ndarray):
        if current.size == 0:
            return ""
        current = current.reshape(-1)[0]
    return str(current).strip()


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
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if args.variant not in VARIANTS:
        raise ValueError(args.variant)
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
        "Version": "SEAL_directed_weighted_v1",
        "ModelVariant": args.variant,
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
    metrics_row.update(metric_values)
    provenance = {
        "schema_version": "seal-directed-weighted-provenance-v1",
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
    write_run_bundle(args.run_dir, metrics_row, arrays, provenance, args.config_hash)
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
