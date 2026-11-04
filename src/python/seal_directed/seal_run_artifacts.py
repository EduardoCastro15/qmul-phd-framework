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


SCHEMA_VERSION_V1 = "seal-directed-run-v1"
SCHEMA_VERSION_V2 = "seal-directed-run-v2"
SCHEMA_VERSION = SCHEMA_VERSION_V2
SUPPORTED_SCHEMA_VERSIONS = {SCHEMA_VERSION_V1, SCHEMA_VERSION_V2}


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


def write_run_bundle(
    run_dir,
    metrics,
    artifact_arrays,
    provenance,
    config_hash,
    extra_files=None,
):
    """Write one complete run using a same-filesystem atomic directory rename."""
    extra_files = dict(extra_files or {})
    schema_version = SCHEMA_VERSION_V2 if extra_files else SCHEMA_VERSION_V1
    if extra_files and set(extra_files) != {"checkpoint.pt", "model_config.json"}:
        raise ValueError(
            "Version-2 bundles require exactly checkpoint.pt and model_config.json"
        )
    for name, content in extra_files.items():
        if Path(name).name != name or not isinstance(content, (bytes, bytearray)):
            raise ValueError("Extra bundle files must be byte payloads with plain filenames")
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
            "schema_version": schema_version,
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

        for name, content in extra_files.items():
            with open(temp_dir / name, "wb") as handle:
                handle.write(content)

        file_hashes = {
            path.name: file_sha256(path)
            for path in sorted(temp_dir.iterdir())
            if path.is_file()
        }

        success_payload = {
            "schema_version": schema_version,
            "config_hash": str(config_hash),
            "hostname": socket.gethostname(),
            "metrics_sha256": file_sha256(metrics_path),
            "artifacts_sha256": file_sha256(artifact_path),
            "files_sha256": file_hashes,
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

    schema_version = payload.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("Unexpected metrics schema in {}".format(run_dir))
    if success.get("schema_version") != schema_version:
        raise ValueError("Unexpected success-marker schema in {}".format(run_dir))
    if payload.get("config_hash") != success.get("config_hash"):
        raise ValueError("Configuration hashes disagree in {}".format(run_dir))
    if expected_config_hash and payload.get("config_hash") != expected_config_hash:
        raise ValueError(
            "Configuration hash mismatch in {}: expected {}, found {}".format(
                run_dir, expected_config_hash, payload.get("config_hash")
            )
        )

    recorded_files = success.get("files_sha256", {})
    if schema_version == SCHEMA_VERSION_V2 and set(recorded_files) != {
        "metrics.json", "artifacts.npz", "checkpoint.pt", "model_config.json"
    }:
        raise ValueError("Version-2 _SUCCESS does not checksum every bundle file")

    if verify_checksums:
        observed_metrics = file_sha256(required["metrics"])
        observed_artifacts = file_sha256(required["artifacts"])
        if observed_metrics != success.get("metrics_sha256"):
            raise ValueError("metrics.json checksum mismatch in {}".format(run_dir))
        if observed_artifacts != success.get("artifacts_sha256"):
            raise ValueError("artifacts.npz checksum mismatch in {}".format(run_dir))
        for filename, expected_hash in recorded_files.items():
            file_path = run_dir / filename
            if not file_path.is_file() or file_sha256(file_path) != expected_hash:
                raise ValueError("{} checksum mismatch in {}".format(filename, run_dir))

    if schema_version == SCHEMA_VERSION_V2:
        checkpoint_path = run_dir / "checkpoint.pt"
        model_config_path = run_dir / "model_config.json"
        if not checkpoint_path.is_file() or not model_config_path.is_file():
            raise ValueError("Version-2 run is missing checkpoint.pt/model_config.json")
        with open(model_config_path, encoding="utf-8") as handle:
            model_config = json.load(handle)
        required_model_fields = {
            "schema_version",
            "ModelID",
            "DisplayName",
            "LegacyFamily",
            "feature_names",
            "feature_schema_hash",
            "feat_dim",
            "attr_dim",
            "sortpooling_k",
            "checkpoint_sha256",
        }
        absent_model_fields = sorted(required_model_fields.difference(model_config))
        if absent_model_fields:
            raise ValueError(
                "model_config.json is missing {} in {}".format(
                    ", ".join(absent_model_fields), run_dir
                )
            )
        if model_config["checkpoint_sha256"] != file_sha256(checkpoint_path):
            raise ValueError("Checkpoint hash disagrees with model_config.json in {}".format(run_dir))
        from nodeattr_schema import feature_schema_hash

        if int(model_config["attr_dim"]) != len(model_config["feature_names"]):
            raise ValueError("Attribute dimension disagrees with feature names in {}".format(run_dir))
        if int(model_config["attr_dim"]) > 0 and model_config[
            "feature_schema_hash"
        ] != feature_schema_hash(model_config["feature_names"]):
            raise ValueError("Feature schema hash disagrees with model_config.json in {}".format(run_dir))
        for key in ("ModelID", "DisplayName", "LegacyFamily"):
            if payload.get("metrics", {}).get(key) != model_config.get(key):
                raise ValueError("{} disagrees between metrics/model config in {}".format(key, run_dir))

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

        if schema_version == SCHEMA_VERSION_V2:
            from graph_artifacts import validate_packed_graphs

            train_pairs = np.vstack([artifact["train_pos"], artifact["train_neg"]])
            train_labels = np.concatenate([
                np.ones(len(artifact["train_pos"]), dtype=np.int8),
                np.zeros(len(artifact["train_neg"]), dtype=np.int8),
            ])
            test_pairs = np.vstack([artifact["test_pos"], artifact["test_neg"]])
            permutation_control = bool(
                payload.get("metrics", {}).get("LabelPermutationControl", 0)
            )
            validate_packed_graphs(
                artifact,
                "train",
                expected_pairs=train_pairs,
                expected_labels=None if permutation_control else train_labels,
            )
            if permutation_control:
                observed_train_labels = np.asarray(artifact["train_graph_labels"])
                if not np.array_equal(
                    np.sort(observed_train_labels), np.sort(train_labels)
                ):
                    raise ValueError("Permuted training labels do not preserve class counts")
            validate_packed_graphs(
                artifact,
                "test",
                expected_pairs=test_pairs,
                expected_labels=artifact["test_labels"],
            )

    return payload


def is_valid_complete_run(run_dir, expected_config_hash=None):
    try:
        validate_run_bundle(run_dir, expected_config_hash=expected_config_hash)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True
