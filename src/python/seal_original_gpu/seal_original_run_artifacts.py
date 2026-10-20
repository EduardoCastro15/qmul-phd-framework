"""Atomic run bundles for the unchanged SEAL original implementation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import socket
from pathlib import Path


SCHEMA_VERSION = "seal-original-gpu-run-v1"
REQUIRED_RUN_FILES = ("result.csv", "terminal.log", "run_metadata.json")


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


def run_directory(result_root, foodweb, experiment_id, seed):
    return (
        Path(result_root)
        / "runs"
        / safe_file_stem(foodweb)
        / "experiment_{:03d}_seed_{}".format(int(experiment_id), int(seed))
    )


def read_single_result_row(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if len(rows) != 1:
        raise ValueError("Expected exactly one SEAL result row in {}, found {}".format(path, len(rows)))
    return rows[0], fieldnames


def finalize_run_bundle(staging_dir, run_dir, metadata, config_hash):
    """Complete a staged run and atomically rename it into its final location."""
    staging_dir = Path(staging_dir)
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    for filename in ("result.csv", "terminal.log"):
        if not (staging_dir / filename).is_file():
            raise ValueError("Staged run is missing {}".format(filename))

    metadata_payload = dict(metadata)
    metadata_payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "config_hash": str(config_hash),
        }
    )
    metadata_path = staging_dir / "run_metadata.json"
    with open(metadata_path, "wb") as handle:
        handle.write(canonical_json_bytes(metadata_payload))
        handle.write(b"\n")

    success_payload = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": str(config_hash),
        "hostname": socket.gethostname(),
        "checksums": {
            filename: file_sha256(staging_dir / filename)
            for filename in REQUIRED_RUN_FILES
        },
    }
    with open(staging_dir / "_SUCCESS", "wb") as handle:
        handle.write(canonical_json_bytes(success_payload))
        handle.write(b"\n")

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(staging_dir), str(run_dir))
    return run_dir


def preserve_failed_staging(staging_dir, result_root, foodweb, experiment_id, seed):
    staging_dir = Path(staging_dir)
    if not staging_dir.exists():
        return None
    destination = (
        Path(result_root)
        / "failed_runs"
        / safe_file_stem(foodweb)
        / "experiment_{:03d}_seed_{}".format(int(experiment_id), int(seed))
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(str(staging_dir), str(destination))
    return destination


def validate_run_bundle(run_dir, expected_config_hash=None, verify_checksums=True):
    run_dir = Path(run_dir)
    required = [run_dir / name for name in (*REQUIRED_RUN_FILES, "_SUCCESS")]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError("Incomplete run {}: missing {}".format(run_dir, ", ".join(missing)))

    with open(run_dir / "run_metadata.json", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with open(run_dir / "_SUCCESS", encoding="utf-8") as handle:
        success = json.load(handle)

    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected metadata schema in {}".format(run_dir))
    if success.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected success-marker schema in {}".format(run_dir))
    if metadata.get("config_hash") != success.get("config_hash"):
        raise ValueError("Configuration hashes disagree in {}".format(run_dir))
    if expected_config_hash and metadata.get("config_hash") != expected_config_hash:
        raise ValueError("Configuration hash mismatch in {}".format(run_dir))

    if verify_checksums:
        for filename in REQUIRED_RUN_FILES:
            expected = success.get("checksums", {}).get(filename)
            observed = file_sha256(run_dir / filename)
            if expected != observed:
                raise ValueError("{} checksum mismatch in {}".format(filename, run_dir))

    result_row, fieldnames = read_single_result_row(run_dir / "result.csv")
    return {"metadata": metadata, "result": result_row, "fieldnames": fieldnames}


def is_valid_complete_run(run_dir, expected_config_hash=None):
    try:
        validate_run_bundle(run_dir, expected_config_hash=expected_config_hash)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True
