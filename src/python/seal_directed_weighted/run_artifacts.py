"""Hashing and atomic output helpers for SEAL-directed-weighted."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import tempfile
from pathlib import Path

import numpy as np


RUN_SCHEMA_V1 = "seal-directed-weighted-run-v1"
RUN_SCHEMA_V2 = "seal-directed-weighted-attribution-run-v2"
RUN_SCHEMA = RUN_SCHEMA_V1
SUPPORTED_RUN_SCHEMAS = {RUN_SCHEMA_V1, RUN_SCHEMA_V2}


def canonical_json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def configuration_hash(config):
    return hashlib.sha256(canonical_json_bytes(config)).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pairs_array(value):
    array = np.asarray(value, dtype=np.int64)
    return array.reshape(-1, 2)


def logical_split_hash(train_pos, train_neg, test_pos, test_neg):
    digest = hashlib.sha256()
    for name, pairs in (
        ("train_pos", train_pos),
        ("train_neg", train_neg),
        ("test_pos", test_pos),
        ("test_neg", test_neg),
    ):
        array = np.ascontiguousarray(pairs_array(pairs), dtype=np.int64)
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def safe_stem(value):
    return "".join(
        char if char.isalnum() or char in ("-", "_", ".") else "_"
        for char in str(value)
    )


def run_directory(root, variant, foodweb, experiment_id, seed):
    return (
        Path(root)
        / "runs"
        / str(variant)
        / safe_stem(foodweb)
        / "experiment_{:03d}_seed_{}".format(int(experiment_id), int(seed))
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_run_bundle(
    run_dir, metrics, arrays, provenance, config_hash, extra_files=None,
):
    extra_files = dict(extra_files or {})
    schema_version = RUN_SCHEMA_V2 if extra_files else RUN_SCHEMA_V1
    if extra_files and set(extra_files) != {"checkpoint.pt", "model_config.json"}:
        raise ValueError(
            "Attribution-v2 bundles require checkpoint.pt and model_config.json"
        )
    for name, content in extra_files.items():
        if Path(name).name != name or not isinstance(content, (bytes, bytearray)):
            raise ValueError("Extra bundle files must be byte payloads with plain filenames")
    run_dir = Path(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    temp_dir = Path(tempfile.mkdtemp(prefix=".{}.tmp.".format(run_dir.name), dir=run_dir.parent))
    try:
        payload = {
            "schema_version": schema_version,
            "config_hash": str(config_hash),
            "metrics": _json_safe(metrics),
            "provenance": _json_safe(provenance),
        }
        metrics_path = temp_dir / "metrics.json"
        with open(metrics_path, "wb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
        artifact_path = temp_dir / "artifacts.npz"
        np.savez_compressed(artifact_path, **{key: np.asarray(value) for key, value in arrays.items()})
        for name, content in extra_files.items():
            with open(temp_dir / name, "wb") as handle:
                handle.write(content)
        file_hashes = {
            path.name: file_sha256(path)
            for path in sorted(temp_dir.iterdir()) if path.is_file()
        }
        success = {
            "schema_version": schema_version,
            "config_hash": str(config_hash),
            "hostname": socket.gethostname(),
            "metrics_sha256": file_sha256(metrics_path),
            "artifacts_sha256": file_sha256(artifact_path),
        }
        if extra_files:
            success["files_sha256"] = file_hashes
        with open(temp_dir / "_SUCCESS", "wb") as handle:
            handle.write(canonical_json_bytes(success) + b"\n")
        os.replace(temp_dir, run_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return run_dir


def validate_run_bundle(run_dir, expected_config_hash=None):
    run_dir = Path(run_dir)
    with open(run_dir / "metrics.json", encoding="utf-8") as handle:
        payload = json.load(handle)
    with open(run_dir / "_SUCCESS", encoding="utf-8") as handle:
        success = json.load(handle)
    schema_version = payload.get("schema_version")
    if schema_version not in SUPPORTED_RUN_SCHEMAS or success.get("schema_version") != schema_version:
        raise ValueError("run schema mismatch: {}".format(run_dir))
    if payload.get("config_hash") != success.get("config_hash"):
        raise ValueError("config hashes disagree: {}".format(run_dir))
    if expected_config_hash and payload.get("config_hash") != expected_config_hash:
        raise ValueError("unexpected config hash: {}".format(run_dir))
    if file_sha256(run_dir / "metrics.json") != success.get("metrics_sha256"):
        raise ValueError("metrics checksum mismatch: {}".format(run_dir))
    if file_sha256(run_dir / "artifacts.npz") != success.get("artifacts_sha256"):
        raise ValueError("artifacts checksum mismatch: {}".format(run_dir))
    if schema_version == RUN_SCHEMA_V2:
        expected_files = {
            "metrics.json", "artifacts.npz", "checkpoint.pt", "model_config.json"
        }
        recorded = success.get("files_sha256", {})
        if set(recorded) != expected_files:
            raise ValueError("v2 success marker does not cover every bundle file")
        for name, digest in recorded.items():
            path = run_dir / name
            if not path.is_file() or file_sha256(path) != digest:
                raise ValueError("{} checksum mismatch: {}".format(name, run_dir))
        with open(run_dir / "model_config.json", encoding="utf-8") as handle:
            model_config = json.load(handle)
        required_model_fields = {
            "schema_version", "ModelID", "ModelVariant", "config_hash",
            "checkpoint_sha256", "label_dim", "node_attribute_dim",
            "edge_attribute_dim", "sortpooling_k", "latent_dimensions",
            "hidden_dimension", "dropout", "edge_chunk_size",
            "node_feature_names", "node_feature_schema_sha256",
            "edge_feature_names", "edge_feature_schema_sha256",
            "attribution_target", "graph_input_sha256",
        }
        missing_model_fields = sorted(required_model_fields.difference(model_config))
        if missing_model_fields:
            raise ValueError(
                "model_config.json is missing {}".format(", ".join(missing_model_fields))
            )
        if model_config["config_hash"] != payload.get("config_hash"):
            raise ValueError("model config hash disagrees with metrics")
        if model_config["checkpoint_sha256"] != file_sha256(run_dir / "checkpoint.pt"):
            raise ValueError("checkpoint hash disagrees with model config")
        if int(model_config["node_attribute_dim"]) != len(model_config["node_feature_names"]):
            raise ValueError("node feature names disagree with node_attribute_dim")
        if int(model_config["edge_attribute_dim"]) != len(model_config["edge_feature_names"]):
            raise ValueError("edge feature names disagree with edge_attribute_dim")
        observed_node_schema = hashlib.sha256(
            canonical_json_bytes(model_config["node_feature_names"])
        ).hexdigest()
        if observed_node_schema != model_config["node_feature_schema_sha256"]:
            raise ValueError("node feature schema hash disagrees with model config")
        from edge_schema import FEATURE_NAMES, SCHEMA_SHA256
        if (
            model_config["edge_feature_names"] != list(FEATURE_NAMES)
            or model_config["edge_feature_schema_sha256"] != SCHEMA_SHA256
        ):
            raise ValueError("edge feature schema disagrees with the frozen full_v1 schema")
        metrics = payload.get("metrics", {})
        if metrics.get("ModelVariant") != model_config.get("ModelVariant"):
            raise ValueError("model variant disagrees between metrics and model config")
    with np.load(run_dir / "artifacts.npz", allow_pickle=False) as arrays:
        required = {
            "train_pos", "train_neg", "test_pos", "test_neg", "test_labels", "test_scores"
        }
        missing = required.difference(arrays.files)
        if missing:
            raise ValueError("missing arrays {}: {}".format(sorted(missing), run_dir))
        if arrays["test_labels"].shape != arrays["test_scores"].shape:
            raise ValueError("label/score shape mismatch: {}".format(run_dir))
        if not np.all(np.isfinite(arrays["test_scores"])):
            raise ValueError("non-finite scores: {}".format(run_dir))
        if schema_version == RUN_SCHEMA_V2:
            from graph_artifacts_v2 import packed_graph_hash, validate_packed_graphs

            required_v2 = {
                "validation_indices", "context_edge_index", "context_edge_attr",
            }
            missing_v2 = sorted(required_v2.difference(arrays.files))
            if missing_v2:
                raise ValueError(
                    "v2 artifacts are missing {}".format(", ".join(missing_v2))
                )

            train_pairs = np.vstack([arrays["train_pos"], arrays["train_neg"]])
            train_labels = np.concatenate([
                np.ones(len(arrays["train_pos"]), dtype=np.int8),
                np.zeros(len(arrays["train_neg"]), dtype=np.int8),
            ])
            test_pairs = np.vstack([arrays["test_pos"], arrays["test_neg"]])
            validate_packed_graphs(
                arrays, "train", expected_pairs=train_pairs, expected_labels=train_labels
            )
            validate_packed_graphs(
                arrays, "test", expected_pairs=test_pairs,
                expected_labels=arrays["test_labels"],
            )
            context_edges = np.asarray(arrays["context_edge_index"]).reshape(-1, 2)
            context_attr = np.asarray(arrays["context_edge_attr"])
            if context_attr.shape != (context_edges.shape[0], 17):
                raise ValueError("context edges and attributes are not aligned")
            validation_indices = np.asarray(
                arrays["validation_indices"], dtype=np.int64
            ).reshape(-1)
            if (
                validation_indices.size < 1
                or len(np.unique(validation_indices)) != validation_indices.size
                or np.any(validation_indices < 0)
                or np.any(validation_indices >= len(train_pairs))
            ):
                raise ValueError("invalid validation_indices in v2 artifacts")
            observed_graph_hash = packed_graph_hash(arrays)
            expected_graph_hash = payload.get("provenance", {}).get("graph_input_sha256")
            model_graph_hash = model_config.get("graph_input_sha256")
            if (
                observed_graph_hash != expected_graph_hash
                or observed_graph_hash != model_graph_hash
            ):
                raise ValueError("packed graph hash mismatch")
    return payload


def is_valid_complete_run(run_dir, expected_config_hash=None):
    try:
        validate_run_bundle(run_dir, expected_config_hash)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True

