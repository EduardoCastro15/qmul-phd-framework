#!/usr/bin/env python3
"""Filter the frozen four-variant campaign into an attribution-ready v2 manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from attribution_manifest_lock import write_manifest_lock
from generate_attribution_config import ATTRIBUTION_VARIANTS
from run_artifacts import canonical_json_bytes, configuration_hash, file_sha256


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def build_manifest(source_manifest, source_config, output_dir, expected_foodwebs=290,
                   expected_experiments=10, only_foodwebs=None, source_commit=None):
    source_manifest = Path(source_manifest).resolve()
    source_config = Path(source_config).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("manifest directory is not empty: {}".format(output_dir))
    with open(source_config, encoding="utf-8") as handle:
        original_config = json.load(handle)
    rows = [row for row in read_csv(source_manifest)
            if row["ModelVariant"] in ATTRIBUTION_VARIANTS]
    selected = set(only_foodwebs or [])
    if selected:
        rows = [row for row in rows if row["Foodweb"] in selected]
        expected_foodwebs = len(selected)
    if not rows:
        raise ValueError("source manifest contains no selected rows")
    foodwebs = sorted({row["Foodweb"] for row in rows})
    experiments = sorted({int(row["ExperimentID"]) for row in rows})
    if len(foodwebs) != int(expected_foodwebs):
        raise ValueError("expected {} food webs, found {}".format(expected_foodwebs, len(foodwebs)))
    if experiments != list(range(1, int(expected_experiments) + 1)):
        raise ValueError("experiment IDs are not 1..{}".format(expected_experiments))
    expected_pairs = {
        (foodweb, experiment)
        for foodweb in foodwebs for experiment in experiments
    }
    expected_identities = {
        (foodweb, experiment, variant)
        for foodweb, experiment in expected_pairs
        for variant in ATTRIBUTION_VARIANTS
    }
    observed_identities = [
        (row["Foodweb"], int(row["ExperimentID"]), row["ModelVariant"])
        for row in rows
    ]
    if len(set(observed_identities)) != len(observed_identities):
        raise ValueError("source manifest contains duplicate attribution identities")
    missing_identities = sorted(expected_identities.difference(observed_identities))
    unexpected_identities = sorted(set(observed_identities).difference(expected_identities))
    if missing_identities or unexpected_identities:
        raise ValueError(
            "source manifest attribution cross-product is incomplete: missing={} unexpected={}".format(
                len(missing_identities), len(unexpected_identities)
            )
        )
    expected = len(expected_identities)
    if len(rows) != expected:
        raise ValueError("expected {} rows, found {}".format(expected, len(rows)))
    grouped = {}
    for row in rows:
        grouped.setdefault((row["Foodweb"], int(row["ExperimentID"])), []).append(row)
    if set(grouped) != expected_pairs:
        raise ValueError("source manifest does not contain every foodweb/experiment pair")
    for key, group in grouped.items():
        if len(group) != len(ATTRIBUTION_VARIANTS) or {
            row["ModelVariant"] for row in group
        } != set(ATTRIBUTION_VARIANTS):
            raise ValueError("missing attribution variant for {}".format(key))
        for field in ("Seed", "SplitPath", "SplitFileSHA256", "SplitHash",
                      "MatPath", "MatSHA256"):
            if len({row[field] for row in group}) != 1:
                raise ValueError("{} differs between variants for {}".format(field, key))
    config = dict(original_config)
    historical_source_commit = original_config.get("source_commit", "unknown")
    config.update({
        "schema_version": "seal-directed-weighted-attribution-training-config-v2",
        "version": "SEAL_directed_node_ecc_attribution_v2",
        "model_variants": list(ATTRIBUTION_VARIANTS),
        "num_foodwebs": len(foodwebs),
        "num_experiments": len(experiments),
        "expected_unique_splits": len(grouped),
        "expected_runs": expected,
        "attribution_ready_artifacts": True,
        "historical_source_manifest_sha256": file_sha256(source_manifest),
        "historical_source_config_sha256": file_sha256(source_config),
        "historical_source_config_hash": configuration_hash(original_config),
        "historical_source_commit": historical_source_commit,
        "base_training_commit": historical_source_commit,
        "source_commit": str(source_commit or "unknown"),
    })
    config_hash = configuration_hash(config)
    historical_hashes = {row.get("ConfigHash", "") for row in rows}
    if len(historical_hashes) != 1:
        raise ValueError("source manifest contains multiple historical ConfigHash values")
    historical_hash = next(iter(historical_hashes))
    expected_historical_hash = configuration_hash(original_config)
    if historical_hash != expected_historical_hash:
        raise ValueError(
            "source manifest ConfigHash does not match source RUN_CONFIG"
        )
    for row in rows:
        row["HistoricalConfigHash"] = historical_hash
        row["ConfigHash"] = config_hash
    rows.sort(key=lambda row: (
        ATTRIBUTION_VARIANTS.index(row["ModelVariant"]),
        int(row["ExperimentID"]), int(row["FoodwebIndex"]), row["Foodweb"],
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "RUN_CONFIG.json"
    manifest_path = output_dir / "manifest.csv"
    config_path.write_bytes(canonical_json_bytes(config) + b"\n")
    fieldnames = list(rows[0])
    if "HistoricalConfigHash" not in fieldnames:
        fieldnames.append("HistoricalConfigHash")
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_manifest_lock(manifest_path, config_path)
    return rows, config, config_hash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-foodwebs", type=int, default=290)
    parser.add_argument("--expected-experiments", type=int, default=10)
    parser.add_argument("--only-foodweb", action="append", default=[])
    parser.add_argument("--source-commit", default="unknown")
    args = parser.parse_args(argv)
    rows, config, digest = build_manifest(
        args.source_manifest, args.source_config, args.output_dir,
        args.expected_foodwebs, args.expected_experiments, args.only_foodweb,
        args.source_commit,
    )
    print(json.dumps({"status": "MANIFEST_OK", "rows": len(rows),
                      "foodwebs": config["num_foodwebs"],
                      "experiments": config["num_experiments"],
                      "config_hash": digest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
