#!/usr/bin/env python3
"""Write and validate an immutable order lock for attribution-v2 manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from run_artifacts import canonical_json_bytes, configuration_hash, file_sha256


LOCK_SCHEMA = "seal-directed-node-ecc-attribution-manifest-lock-v2"
LOCK_FILENAME = "MANIFEST_LOCK.json"
IDENTITY_FIELDS = (
    "ModelVariant", "ExperimentID", "FoodwebIndex", "Foodweb", "Seed", "SplitHash",
)


def read_manifest_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def row_order_sha256(rows):
    ordered = [
        {field: str(row[field]) for field in IDENTITY_FIELDS}
        for row in rows
    ]
    return hashlib.sha256(canonical_json_bytes(ordered)).hexdigest()


def build_lock_payload(manifest_path, config_path):
    manifest_path = Path(manifest_path).resolve()
    config_path = Path(config_path).resolve()
    rows = read_manifest_rows(manifest_path)
    if not rows:
        raise ValueError("cannot lock an empty attribution manifest")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    identities = [tuple(str(row[field]) for field in IDENTITY_FIELDS) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("manifest lock refuses duplicate ordered identities")
    return {
        "schema_version": LOCK_SCHEMA,
        "manifest_sha256": file_sha256(manifest_path),
        "run_config_sha256": file_sha256(config_path),
        "run_config_hash": configuration_hash(config),
        "row_order_sha256": row_order_sha256(rows),
        "row_count": len(rows),
        "identity_fields": list(IDENTITY_FIELDS),
        "first_identity": dict(zip(IDENTITY_FIELDS, identities[0])),
        "last_identity": dict(zip(IDENTITY_FIELDS, identities[-1])),
    }


def write_manifest_lock(manifest_path, config_path, lock_path=None):
    manifest_path = Path(manifest_path).resolve()
    config_path = Path(config_path).resolve()
    lock_path = Path(lock_path or manifest_path.with_name(LOCK_FILENAME)).resolve()
    if lock_path.exists():
        raise FileExistsError(lock_path)
    payload = build_lock_payload(manifest_path, config_path)
    lock_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def validate_manifest_lock(manifest_path, config_path, lock_path=None):
    manifest_path = Path(manifest_path).resolve()
    config_path = Path(config_path).resolve()
    lock_path = Path(lock_path or manifest_path.with_name(LOCK_FILENAME)).resolve()
    with open(lock_path, encoding="utf-8") as handle:
        locked = json.load(handle)
    if locked.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("attribution manifest lock schema mismatch")
    observed = build_lock_payload(manifest_path, config_path)
    for field in (
        "manifest_sha256", "run_config_sha256", "run_config_hash",
        "row_order_sha256", "row_count", "identity_fields",
        "first_identity", "last_identity",
    ):
        if locked.get(field) != observed[field]:
            raise ValueError("attribution manifest lock mismatch: {}".format(field))
    return locked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    payload = (
        write_manifest_lock(args.manifest, args.config, args.lock)
        if args.write else
        validate_manifest_lock(args.manifest, args.config, args.lock)
    )
    print(json.dumps({
        "status": "MANIFEST_LOCK_OK",
        "rows": payload["row_count"],
        "row_order_sha256": payload["row_order_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
