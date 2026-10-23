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


RUN_SCHEMA = "seal-directed-weighted-run-v1"


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


def write_run_bundle(run_dir, metrics, arrays, provenance, config_hash):
    run_dir = Path(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    temp_dir = Path(tempfile.mkdtemp(prefix=".{}.tmp.".format(run_dir.name), dir=run_dir.parent))
    try:
        payload = {
            "schema_version": RUN_SCHEMA,
            "config_hash": str(config_hash),
            "metrics": _json_safe(metrics),
            "provenance": _json_safe(provenance),
        }
        metrics_path = temp_dir / "metrics.json"
        with open(metrics_path, "wb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
        artifact_path = temp_dir / "artifacts.npz"
        np.savez_compressed(artifact_path, **{key: np.asarray(value) for key, value in arrays.items()})
        success = {
            "schema_version": RUN_SCHEMA,
            "config_hash": str(config_hash),
            "hostname": socket.gethostname(),
            "metrics_sha256": file_sha256(metrics_path),
            "artifacts_sha256": file_sha256(artifact_path),
        }
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
    if payload.get("schema_version") != RUN_SCHEMA or success.get("schema_version") != RUN_SCHEMA:
        raise ValueError("run schema mismatch: {}".format(run_dir))
    if payload.get("config_hash") != success.get("config_hash"):
        raise ValueError("config hashes disagree: {}".format(run_dir))
    if expected_config_hash and payload.get("config_hash") != expected_config_hash:
        raise ValueError("unexpected config hash: {}".format(run_dir))
    if file_sha256(run_dir / "metrics.json") != success.get("metrics_sha256"):
        raise ValueError("metrics checksum mismatch: {}".format(run_dir))
    if file_sha256(run_dir / "artifacts.npz") != success.get("artifacts_sha256"):
        raise ValueError("artifacts checksum mismatch: {}".format(run_dir))
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
    return payload


def is_valid_complete_run(run_dir, expected_config_hash=None):
    try:
        validate_run_bundle(run_dir, expected_config_hash)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True

