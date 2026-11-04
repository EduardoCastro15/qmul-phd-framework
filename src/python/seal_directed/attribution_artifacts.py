"""Atomic and fail-closed artifacts for one SEAL-D-NodeAttr explanation run."""

from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
from pathlib import Path

import numpy as np

from seal_run_artifacts import canonical_json_bytes, file_sha256, safe_file_stem


ATTRIBUTION_SCHEMA_VERSION = "seal-directed-attribution-v1"


def attribution_directory(result_root, foodweb, experiment_id, seed):
    return (
        Path(result_root)
        / "attributions"
        / "runs"
        / safe_file_stem(foodweb)
        / "experiment_{:03d}_seed_{}".format(int(experiment_id), int(seed))
    )


def write_attribution_bundle(bundle_dir, metadata, arrays, attribution_config_hash):
    bundle_dir = Path(bundle_dir)
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    if bundle_dir.exists():
        raise FileExistsError(
            "Attribution bundle already exists; validate it or use a new root: {}".format(
                bundle_dir
            )
        )
    temporary = Path(
        tempfile.mkdtemp(prefix=".{}.tmp.".format(bundle_dir.name), dir=str(bundle_dir.parent))
    )
    try:
        metadata_payload = dict(metadata)
        metadata_payload.update({
            "schema_version": ATTRIBUTION_SCHEMA_VERSION,
            "attribution_config_hash": str(attribution_config_hash),
        })
        metadata_path = temporary / "metadata.json"
        metadata_path.write_bytes(canonical_json_bytes(metadata_payload) + b"\n")
        arrays_path = temporary / "attributions.npz"
        np.savez_compressed(
            arrays_path,
            **{key: np.asarray(value) for key, value in arrays.items()},
        )
        success = {
            "schema_version": ATTRIBUTION_SCHEMA_VERSION,
            "attribution_config_hash": str(attribution_config_hash),
            "hostname": socket.gethostname(),
            "training_bundle_success_sha256": metadata_payload[
                "training_bundle_success_sha256"
            ],
            "checkpoint_sha256": metadata_payload["checkpoint_sha256"],
            "files_sha256": {
                "metadata.json": file_sha256(metadata_path),
                "attributions.npz": file_sha256(arrays_path),
            },
        }
        (temporary / "_SUCCESS").write_bytes(canonical_json_bytes(success) + b"\n")
        os.replace(str(temporary), str(bundle_dir))
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return bundle_dir


def validate_attribution_bundle(
    bundle_dir,
    expected_config_hash=None,
    expected_training_success_hash=None,
    expected_checkpoint_hash=None,
    verify_checksums=True,
):
    bundle_dir = Path(bundle_dir)
    paths = {
        "metadata": bundle_dir / "metadata.json",
        "arrays": bundle_dir / "attributions.npz",
        "success": bundle_dir / "_SUCCESS",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError("Incomplete attribution bundle: missing {}".format(", ".join(missing)))
    with open(paths["metadata"], encoding="utf-8") as handle:
        metadata = json.load(handle)
    with open(paths["success"], encoding="utf-8") as handle:
        success = json.load(handle)
    if metadata.get("schema_version") != ATTRIBUTION_SCHEMA_VERSION:
        raise ValueError("Unexpected attribution metadata schema")
    if success.get("schema_version") != ATTRIBUTION_SCHEMA_VERSION:
        raise ValueError("Unexpected attribution success-marker schema")
    config_hash = metadata.get("attribution_config_hash")
    if config_hash != success.get("attribution_config_hash"):
        raise ValueError("Attribution config hashes disagree")
    if expected_config_hash and config_hash != expected_config_hash:
        raise ValueError("Attribution config hash mismatch")
    training_hash = metadata.get("training_bundle_success_sha256")
    checkpoint_hash = metadata.get("checkpoint_sha256")
    if training_hash != success.get("training_bundle_success_sha256"):
        raise ValueError("Training bundle references disagree")
    if checkpoint_hash != success.get("checkpoint_sha256"):
        raise ValueError("Checkpoint references disagree")
    if expected_training_success_hash and training_hash != expected_training_success_hash:
        raise ValueError("Wrong training bundle referenced by attribution")
    if expected_checkpoint_hash and checkpoint_hash != expected_checkpoint_hash:
        raise ValueError("Wrong checkpoint referenced by attribution")
    if set(success.get("files_sha256", {})) != {"metadata.json", "attributions.npz"}:
        raise ValueError("Attribution _SUCCESS does not checksum every bundle file")
    if verify_checksums:
        for filename, expected_hash in success.get("files_sha256", {}).items():
            path = bundle_dir / filename
            if not path.is_file() or file_sha256(path) != expected_hash:
                raise ValueError("{} checksum mismatch".format(filename))

    with np.load(paths["arrays"], allow_pickle=False) as arrays:
        required = {
            "candidate_pairs", "candidate_labels", "candidate_scores",
            "candidate_log_odds", "candidate_confusion", "node_offsets",
            "node_ids", "node_roles", "expected_gradients",
            "expected_gradients_abs_proportion", "drnl_labels",
            "drnl_integrated_gradients", "drnl_active_attribution",
            "eg_completeness_absolute", "eg_completeness_relative",
            "eg_output_difference", "eg_attribution_sum",
            "drnl_completeness_absolute", "drnl_completeness_relative",
            "drnl_output_difference", "drnl_attribution_sum",
            "background_train_indices",
        }
        absent = sorted(required.difference(arrays.files))
        if absent:
            raise ValueError("Missing attribution arrays: {}".format(", ".join(absent)))
        candidates = np.asarray(arrays["candidate_pairs"]).reshape(-1, 2)
        count = candidates.shape[0]
        for name in (
            "candidate_labels", "candidate_scores", "candidate_log_odds",
            "candidate_confusion", "eg_completeness_absolute",
            "eg_completeness_relative", "eg_output_difference",
            "eg_attribution_sum", "drnl_completeness_absolute",
            "drnl_completeness_relative", "drnl_output_difference",
            "drnl_attribution_sum",
        ):
            if np.asarray(arrays[name]).reshape(-1).size != count:
                raise ValueError("{} does not have one value per candidate".format(name))
        if len({tuple(row) for row in candidates.tolist()}) != count:
            raise ValueError("Candidates are not unique within attribution bundle")
        offsets = np.asarray(arrays["node_offsets"], dtype=np.int64)
        if offsets.shape != (count + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 2):
            raise ValueError("Invalid node_offsets")
        total_nodes = int(offsets[-1])
        if np.asarray(arrays["node_ids"]).size != total_nodes:
            raise ValueError("node_offsets do not cover node_ids")
        if np.asarray(arrays["node_roles"]).size != total_nodes:
            raise ValueError("node_offsets do not cover node_roles")
        eg = np.asarray(arrays["expected_gradients"])
        eg_norm = np.asarray(arrays["expected_gradients_abs_proportion"])
        if eg.shape != (total_nodes, 44) or eg_norm.shape != eg.shape:
            raise ValueError("Expected Gradients arrays must be total_nodes x 44")
        drnl = np.asarray(arrays["drnl_integrated_gradients"])
        if drnl.ndim != 2 or drnl.shape[0] != total_nodes:
            raise ValueError("Invalid DRNL Integrated Gradients shape")
        if np.asarray(arrays["drnl_labels"]).size != total_nodes:
            raise ValueError("DRNL labels do not cover all nodes")
        if np.asarray(arrays["drnl_active_attribution"]).size != total_nodes:
            raise ValueError("DRNL active attributions do not cover all nodes")
        finite_names = required.difference({"candidate_pairs", "candidate_confusion", "node_offsets", "node_ids", "node_roles", "drnl_labels", "background_train_indices"})
        for name in finite_names:
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError("Non-finite values in {}".format(name))
        if np.any(eg_norm < 0):
            raise ValueError("Normalized absolute attributions cannot be negative")
        for index in range(count):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            total = float(np.sum(eg_norm[start:stop]))
            raw_total = float(np.sum(np.abs(eg[start:stop])))
            if raw_total > 0 and not np.isclose(total, 1.0, atol=1e-5):
                raise ValueError("Candidate {} normalized attribution does not sum to one".format(index))
        backgrounds = np.asarray(arrays["background_train_indices"])
        if backgrounds.ndim != 2 or backgrounds.shape[0] != count:
            raise ValueError("background_train_indices must be candidate x sample")
        if np.any(backgrounds < 0):
            raise ValueError("Negative training background index")
        training_graph_count = int(metadata.get("background_training_graph_count", -1))
        if training_graph_count < 1 or np.any(backgrounds >= training_graph_count):
            raise ValueError("Background index is not a training-subgraph index")
        faithfulness_names = {
            "faithfulness_family_order",
            "faithfulness_top_family",
            "faithfulness_bottom_family",
            "faithfulness_top_log_odds_reduction",
            "faithfulness_bottom_log_odds_reduction",
        }
        present_faithfulness = faithfulness_names.intersection(arrays.files)
        if present_faithfulness and present_faithfulness != faithfulness_names:
            raise ValueError("Faithfulness arrays are incomplete")
        if present_faithfulness:
            family_count = len(arrays["faithfulness_family_order"])
            for name in ("faithfulness_top_family", "faithfulness_bottom_family"):
                values = np.asarray(arrays[name]).reshape(-1)
                if values.size != count or np.any(values < 0) or np.any(values >= family_count):
                    raise ValueError("Invalid {}".format(name))
            for name in (
                "faithfulness_top_log_odds_reduction",
                "faithfulness_bottom_log_odds_reduction",
            ):
                values = np.asarray(arrays[name]).reshape(-1)
                if values.size != count or not np.all(np.isfinite(values)):
                    raise ValueError("Invalid {}".format(name))
    return metadata


def is_valid_attribution_bundle(bundle_dir, expected_config_hash=None):
    try:
        validate_attribution_bundle(bundle_dir, expected_config_hash=expected_config_hash)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True
