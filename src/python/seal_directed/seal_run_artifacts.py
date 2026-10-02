"""Atomic, self-validating artifacts for one SEAL-directed experiment."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as ssp


SCHEMA_VERSION = "seal-directed-run-v1"


def safe_file_stem(value):
    value = os.path.splitext(os.path.basename(str(value or "seal")))[0]
    return "".join(
        character if character.isalnum() or character in ("-", "_", ".") else "_"
        for character in value
    )


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


def pairs_to_array(pairs):
    if pairs is None:
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack(
        [np.asarray(pairs[0], dtype=np.int64), np.asarray(pairs[1], dtype=np.int64)]
    )


def sparse_to_arrays(matrix, prefix="pseudo"):
    matrix = ssp.csr_matrix(matrix)
    return {
        "{}_data".format(prefix): matrix.data,
        "{}_indices".format(prefix): matrix.indices,
        "{}_indptr".format(prefix): matrix.indptr,
        "{}_shape".format(prefix): np.asarray(matrix.shape, dtype=np.int64),
    }


def arrays_to_sparse(artifact, prefix="pseudo"):
    shape = tuple(int(value) for value in artifact["{}_shape".format(prefix)])
    return ssp.csr_matrix(
        (
            artifact["{}_data".format(prefix)],
            artifact["{}_indices".format(prefix)],
            artifact["{}_indptr".format(prefix)],
        ),
        shape=shape,
    )


def split_hash(train_pos, train_neg, test_pos, test_neg):
    digest = hashlib.sha256()
    for name, pairs in (
        ("train_pos", train_pos),
        ("train_neg", train_neg),
        ("test_pos", test_pos),
        ("test_neg", test_neg),
    ):
        array = np.ascontiguousarray(pairs_to_array(pairs), dtype=np.int64)
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def node_order_hash(taxonomy, net):
    digest = hashlib.sha256()
    if taxonomy is not None:
        flattened = np.asarray(taxonomy, dtype=object).reshape(-1)
        for value in flattened:
            while isinstance(value, np.ndarray) and value.size == 1:
                value = value.reshape(-1)[0]
            digest.update(str(value).strip().encode("utf-8"))
            digest.update(b"\0")
    else:
        matrix = ssp.csr_matrix(net)
        digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
        digest.update(matrix.indptr.astype(np.int64, copy=False).tobytes())
        digest.update(matrix.indices.astype(np.int64, copy=False).tobytes())
    return digest.hexdigest()


def run_directory(result_root, foodweb, experiment_id, seed):
    return (
        Path(result_root)
        / "runs"
        / safe_file_stem(foodweb)
        / "experiment_{:03d}_seed_{}".format(int(experiment_id), int(seed))
    )


def _json_compatible(value):
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_run_bundle(run_dir, metrics, artifact_arrays, provenance, config_hash):
    """Write one complete run using a same-filesystem atomic directory rename."""
    run_dir = Path(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise FileExistsError(
            "Run directory already exists; validate/resume it or move it aside: {}".format(
                run_dir
            )
        )

    temp_dir = Path(
        tempfile.mkdtemp(prefix=".{}.tmp.".format(run_dir.name), dir=str(run_dir.parent))
    )
    try:
        metrics_payload = {
            "schema_version": SCHEMA_VERSION,
            "config_hash": str(config_hash),
            "metrics": _json_compatible(metrics),
            "provenance": _json_compatible(provenance),
        }
        metrics_path = temp_dir / "metrics.json"
        with open(metrics_path, "wb") as handle:
            handle.write(canonical_json_bytes(metrics_payload))
            handle.write(b"\n")

        artifact_path = temp_dir / "artifacts.npz"
        np.savez_compressed(
            artifact_path,
            **{key: np.asarray(value) for key, value in artifact_arrays.items()},
        )

        success_payload = {
            "schema_version": SCHEMA_VERSION,
            "config_hash": str(config_hash),
            "hostname": socket.gethostname(),
            "metrics_sha256": file_sha256(metrics_path),
            "artifacts_sha256": file_sha256(artifact_path),
        }
        success_path = temp_dir / "_SUCCESS"
        with open(success_path, "wb") as handle:
            handle.write(canonical_json_bytes(success_payload))
            handle.write(b"\n")

        os.replace(str(temp_dir), str(run_dir))
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return run_dir


def validate_run_bundle(run_dir, expected_config_hash=None, verify_checksums=True):
    run_dir = Path(run_dir)
    required = {
        "metrics": run_dir / "metrics.json",
        "artifacts": run_dir / "artifacts.npz",
        "success": run_dir / "_SUCCESS",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ValueError("Incomplete run {}: missing {}".format(run_dir, ", ".join(missing)))

    with open(required["metrics"], encoding="utf-8") as handle:
        payload = json.load(handle)
    with open(required["success"], encoding="utf-8") as handle:
        success = json.load(handle)

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected metrics schema in {}".format(run_dir))
    if success.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected success-marker schema in {}".format(run_dir))
    if payload.get("config_hash") != success.get("config_hash"):
        raise ValueError("Configuration hashes disagree in {}".format(run_dir))
    if expected_config_hash and payload.get("config_hash") != expected_config_hash:
        raise ValueError(
            "Configuration hash mismatch in {}: expected {}, found {}".format(
                run_dir, expected_config_hash, payload.get("config_hash")
            )
        )

    if verify_checksums:
        observed_metrics = file_sha256(required["metrics"])
        observed_artifacts = file_sha256(required["artifacts"])
        if observed_metrics != success.get("metrics_sha256"):
            raise ValueError("metrics.json checksum mismatch in {}".format(run_dir))
        if observed_artifacts != success.get("artifacts_sha256"):
            raise ValueError("artifacts.npz checksum mismatch in {}".format(run_dir))

    with np.load(required["artifacts"], allow_pickle=False) as artifact:
        required_arrays = {
            "train_pos",
            "train_neg",
            "test_pos",
            "test_neg",
            "test_labels",
            "test_scores",
            "pseudo_data",
            "pseudo_indices",
            "pseudo_indptr",
            "pseudo_shape",
        }
        absent = sorted(required_arrays.difference(artifact.files))
        if absent:
            raise ValueError(
                "Missing artifact arrays in {}: {}".format(run_dir, ", ".join(absent))
            )
        if artifact["test_labels"].shape != artifact["test_scores"].shape:
            raise ValueError("Test labels/scores shape mismatch in {}".format(run_dir))
        if not np.all(np.isfinite(artifact["test_scores"])):
            raise ValueError("Non-finite test scores in {}".format(run_dir))

    return payload


def is_valid_complete_run(run_dir, expected_config_hash=None):
    try:
        validate_run_bundle(run_dir, expected_config_hash=expected_config_hash)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True
