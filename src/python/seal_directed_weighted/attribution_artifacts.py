"""Atomic, variant-aware artifacts for SEAL directed node/ECC attribution v2."""

from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
from pathlib import Path

import numpy as np

from run_artifacts import canonical_json_bytes, file_sha256, safe_stem


ATTRIBUTION_SCHEMA = "seal-directed-node-ecc-attribution-v2"
VARIANTS = ("directed_node", "directed_ecc")


def attribution_directory(root, variant, foodweb, experiment_id, seed):
    if variant not in VARIANTS:
        raise ValueError("unsupported attribution variant: {}".format(variant))
    return (Path(root) / "attributions" / "runs" / variant / safe_stem(foodweb)
            / "experiment_{:03d}_seed_{}".format(int(experiment_id), int(seed)))


def write_attribution_bundle(bundle_dir, metadata, arrays, attribution_config_hash):
    bundle_dir = Path(bundle_dir)
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    if bundle_dir.exists():
        raise FileExistsError("attribution bundle already exists: {}".format(bundle_dir))
    temporary = Path(tempfile.mkdtemp(prefix=".{}.tmp.".format(bundle_dir.name),
                                      dir=bundle_dir.parent))
    try:
        payload = dict(metadata)
        payload.update({"schema_version": ATTRIBUTION_SCHEMA,
                        "attribution_config_hash": str(attribution_config_hash)})
        metadata_path = temporary / "metadata.json"
        metadata_path.write_bytes(canonical_json_bytes(payload) + b"\n")
        arrays_path = temporary / "attributions.npz"
        np.savez_compressed(arrays_path, **{key: np.asarray(value) for key, value in arrays.items()})
        success = {
            "schema_version": ATTRIBUTION_SCHEMA,
            "attribution_config_hash": str(attribution_config_hash),
            "hostname": socket.gethostname(),
            "training_bundle_success_sha256": payload["training_bundle_success_sha256"],
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "files_sha256": {
                "metadata.json": file_sha256(metadata_path),
                "attributions.npz": file_sha256(arrays_path),
            },
        }
        (temporary / "_SUCCESS").write_bytes(canonical_json_bytes(success) + b"\n")
        os.replace(temporary, bundle_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return bundle_dir


def _one_per_candidate(arrays, names, count):
    for name in names:
        if np.asarray(arrays[name]).reshape(-1).size != count:
            raise ValueError("{} must contain one value per candidate".format(name))


def validate_attribution_bundle(bundle_dir, expected_config_hash=None,
                                expected_training_success_hash=None,
                                expected_checkpoint_hash=None, verify_checksums=True):
    bundle_dir = Path(bundle_dir)
    paths = {"metadata": bundle_dir / "metadata.json",
             "arrays": bundle_dir / "attributions.npz",
             "success": bundle_dir / "_SUCCESS"}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError("incomplete attribution bundle: missing {}".format(", ".join(missing)))
    with open(paths["metadata"], encoding="utf-8") as handle:
        metadata = json.load(handle)
    with open(paths["success"], encoding="utf-8") as handle:
        success = json.load(handle)
    if metadata.get("schema_version") != ATTRIBUTION_SCHEMA or success.get("schema_version") != ATTRIBUTION_SCHEMA:
        raise ValueError("attribution schema mismatch")
    variant = metadata.get("ModelVariant")
    if variant not in VARIANTS:
        raise ValueError("invalid ModelVariant")
    config_hash = metadata.get("attribution_config_hash")
    if config_hash != success.get("attribution_config_hash"):
        raise ValueError("attribution config hashes disagree")
    if expected_config_hash and config_hash != expected_config_hash:
        raise ValueError("unexpected attribution config hash")
    for key, expected in (("training_bundle_success_sha256", expected_training_success_hash),
                          ("checkpoint_sha256", expected_checkpoint_hash)):
        if metadata.get(key) != success.get(key):
            raise ValueError("{} references disagree".format(key))
        if expected and metadata.get(key) != expected:
            raise ValueError("unexpected {}".format(key))
    if set(success.get("files_sha256", {})) != {"metadata.json", "attributions.npz"}:
        raise ValueError("success marker does not checksum every bundle file")
    if verify_checksums:
        for filename, digest in success["files_sha256"].items():
            if file_sha256(bundle_dir / filename) != digest:
                raise ValueError("{} checksum mismatch".format(filename))
    required = {
        "candidate_pairs", "candidate_labels", "candidate_scores", "candidate_log_odds",
        "candidate_confusion", "node_offsets", "node_ids", "node_roles",
        "node_feature_values", "node_expected_gradients",
        "node_expected_gradients_abs_proportion", "node_background_train_indices",
        "node_completeness_absolute", "node_completeness_relative",
        "node_output_difference", "node_attribution_sum", "drnl_labels",
        "drnl_integrated_gradients", "drnl_integrated_gradients_abs_proportion",
        "drnl_active_attribution",
        "drnl_completeness_absolute", "drnl_completeness_relative",
        "drnl_output_difference", "drnl_attribution_sum", "edge_offsets", "edge_index",
        "edge_role_pairs", "edge_feature_values", "edge_expected_gradients",
        "edge_expected_gradients_abs_proportion", "edge_baseline_fallback_tiers",
        "edge_baseline_seed", "edge_pool_sha256", "edge_completeness_absolute",
        "edge_completeness_relative", "edge_output_difference", "edge_attribution_sum",
    }
    with np.load(paths["arrays"], allow_pickle=False) as arrays:
        absent = sorted(required.difference(arrays.files))
        if absent:
            raise ValueError("missing attribution arrays: {}".format(", ".join(absent)))
        candidates = np.asarray(arrays["candidate_pairs"], dtype=np.int64).reshape(-1, 2)
        count = len(candidates)
        if len({tuple(pair) for pair in candidates.tolist()}) != count:
            raise ValueError("candidate pairs are not unique")
        _one_per_candidate(arrays, (
            "candidate_labels", "candidate_scores", "candidate_log_odds", "candidate_confusion",
            "node_completeness_absolute", "node_completeness_relative",
            "node_output_difference", "node_attribution_sum", "drnl_completeness_absolute",
            "drnl_completeness_relative", "drnl_output_difference", "drnl_attribution_sum",
            "edge_baseline_seed", "edge_completeness_absolute", "edge_completeness_relative",
            "edge_output_difference", "edge_attribution_sum",
        ), count)
        node_offsets = np.asarray(arrays["node_offsets"], dtype=np.int64)
        if node_offsets.shape != (count + 1,) or node_offsets[0] != 0 or np.any(np.diff(node_offsets) < 2):
            raise ValueError("invalid node_offsets")
        nodes = int(node_offsets[-1])
        for name in ("node_ids", "node_roles", "drnl_labels", "drnl_active_attribution"):
            if np.asarray(arrays[name]).reshape(-1).size != nodes:
                raise ValueError("{} does not cover every node".format(name))
        node_values = np.asarray(arrays["node_feature_values"])
        node_attr = np.asarray(arrays["node_expected_gradients"])
        node_norm = np.asarray(arrays["node_expected_gradients_abs_proportion"])
        if node_values.shape != (nodes, 44) or node_attr.shape != (nodes, 44) or node_norm.shape != node_attr.shape:
            raise ValueError("node arrays must be total_nodes x 44")
        drnl = np.asarray(arrays["drnl_integrated_gradients"])
        drnl_norm = np.asarray(arrays["drnl_integrated_gradients_abs_proportion"])
        if drnl.ndim != 2 or drnl.shape[0] != nodes or drnl_norm.shape != drnl.shape:
            raise ValueError("invalid DRNL attribution shape")
        backgrounds = np.asarray(arrays["node_background_train_indices"])
        if backgrounds.ndim != 2 or backgrounds.shape[0] != count or np.any(backgrounds < 0):
            raise ValueError("invalid node training-background indices")
        training_count = int(metadata.get("background_training_graph_count", -1))
        if training_count < 1 or np.any(backgrounds >= training_count):
            raise ValueError("node baseline references a non-training graph")
        edge_offsets = np.asarray(arrays["edge_offsets"], dtype=np.int64)
        if edge_offsets.shape != (count + 1,) or edge_offsets[0] != 0 or np.any(np.diff(edge_offsets) < 0):
            raise ValueError("invalid edge_offsets")
        edges = int(edge_offsets[-1])
        if np.asarray(arrays["edge_index"]).shape != (edges, 2):
            raise ValueError("invalid edge_index shape")
        if np.asarray(arrays["edge_role_pairs"]).shape != (edges, 2):
            raise ValueError("invalid edge_role_pairs shape")
        edge_values = np.asarray(arrays["edge_feature_values"])
        edge_attr = np.asarray(arrays["edge_expected_gradients"])
        edge_norm = np.asarray(arrays["edge_expected_gradients_abs_proportion"])
        if edge_values.shape != (edges, 17) or edge_attr.shape != (edges, 17) or edge_norm.shape != edge_attr.shape:
            raise ValueError("edge arrays must be total_edges x 17")
        fallback = np.asarray(arrays["edge_baseline_fallback_tiers"])
        if fallback.reshape(-1).size != edges or (edges and np.any((fallback < 0) | (fallback > 2))):
            raise ValueError("invalid edge baseline fallback tiers")
        pools = np.asarray(arrays["edge_pool_sha256"])
        if pools.reshape(-1).size not in (count, edges):
            raise ValueError("edge_pool_sha256 must be candidate- or edge-aligned")
        if variant == "directed_node" and edges != 0:
            raise ValueError("directed_node must not contain edge attributions")
        if variant == "directed_ecc" and edges == 0 and np.any(np.diff(edge_offsets) != 0):
            raise ValueError("ECC edge offsets disagree")
        excluded = {"candidate_pairs", "candidate_confusion", "node_offsets", "node_ids",
                    "node_roles", "drnl_labels", "node_background_train_indices", "edge_offsets",
                    "edge_index", "edge_role_pairs", "edge_baseline_fallback_tiers",
                    "edge_baseline_seed", "edge_pool_sha256"}
        for name in required.difference(excluded):
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError("non-finite values in {}".format(name))
        if np.any(node_norm < 0) or np.any(drnl_norm < 0) or np.any(edge_norm < 0):
            raise ValueError("normalized absolute attributions cannot be negative")
        faithfulness_samples = int(metadata.get("faithfulness_background_samples", 0))
        for channel in ("node", "edge"):
            names = {
                "faithfulness_{}_family_order".format(channel),
                "faithfulness_{}_top_family".format(channel),
                "faithfulness_{}_bottom_family".format(channel),
                "faithfulness_{}_top_abs_log_odds_change".format(channel),
                "faithfulness_{}_bottom_abs_log_odds_change".format(channel),
            }
            present = names.intersection(arrays.files)
            if faithfulness_samples > 0 and present != names:
                raise ValueError("missing {} faithfulness arrays".format(channel))
            if present and present != names:
                raise ValueError("incomplete {} faithfulness arrays".format(channel))
            if present:
                family_count = len(arrays["faithfulness_{}_family_order".format(channel)])
                edge_counts = np.diff(edge_offsets)
                for position in ("top", "bottom"):
                    values = np.asarray(arrays[
                        "faithfulness_{}_{}_family".format(channel, position)]).reshape(-1)
                    if values.size != count:
                        raise ValueError("invalid {} faithfulness family index".format(channel))
                    if channel == "edge" and variant == "directed_node":
                        if family_count != 0 or np.any(values != -1):
                            raise ValueError("directed_node has edge faithfulness indices")
                    elif channel == "edge" and variant == "directed_ecc":
                        empty = edge_counts == 0
                        if (
                            family_count < 1
                            or np.any(values[empty] != -1)
                            or np.any(values[~empty] < 0)
                            or np.any(values[~empty] >= family_count)
                        ):
                            raise ValueError("invalid edge faithfulness family index")
                    elif (
                        family_count < 1
                        or np.any(values < 0)
                        or np.any(values >= family_count)
                    ):
                        raise ValueError("invalid {} faithfulness family index".format(channel))
                    changes = np.asarray(arrays[
                        "faithfulness_{}_{}_abs_log_odds_change".format(channel, position)]).reshape(-1)
                    if (
                        changes.size != count
                        or not np.all(np.isfinite(changes))
                        or np.any(changes < 0)
                    ):
                        raise ValueError("invalid {} faithfulness changes".format(channel))
                    if (
                        channel == "edge"
                        and variant == "directed_ecc"
                        and np.any(changes[edge_counts == 0] != 0)
                    ):
                        raise ValueError(
                            "edge-free ECC candidates must have zero faithfulness change"
                        )
    return metadata


def is_valid_attribution_bundle(bundle_dir, expected_config_hash=None):
    try:
        validate_attribution_bundle(bundle_dir, expected_config_hash)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True
