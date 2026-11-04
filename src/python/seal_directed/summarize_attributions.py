#!/usr/bin/env python3
"""Hierarchical, food-web-weighted summaries of SEAL-D-NodeAttr attribution bundles."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from attribution import CONFUSION_CODES, NODE_ROLE_CODES
from attribution_artifacts import attribution_directory, validate_attribution_bundle
from nodeattr_schema import FEATURE_FAMILY_ORDER
from run_foodweb_seal_directed import read_manifest
from seal_run_artifacts import run_directory, validate_run_bundle


CONFUSION_NAMES = {value: key for key, value in CONFUSION_CODES.items()}
ROLE_NAMES = {value: key for key, value in NODE_ROLE_CODES.items()}


def read_ecosystems(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "Foodweb" not in rows[0] or "EcosystemType" not in rows[0]:
        raise ValueError("Ecosystem CSV must contain Foodweb and EcosystemType")
    return {row["Foodweb"]: row["EcosystemType"] for row in rows}


def bootstrap_equal_webs(matrix, replicates, rng):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError("Bootstrap requires at least one food web")
    observed = np.mean(matrix, axis=0)
    if matrix.shape[0] == 1:
        return observed, observed.copy(), observed.copy()
    boot = np.empty((int(replicates), matrix.shape[1]), dtype=np.float64)
    for start in range(0, int(replicates), 250):
        stop = min(start + 250, int(replicates))
        indices = rng.integers(0, matrix.shape[0], size=(stop - start, matrix.shape[0]))
        boot[start:stop] = np.mean(matrix[indices], axis=1)
    return observed, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def mean_or_none(matrix, mask=None):
    matrix = np.asarray(matrix)
    if mask is not None:
        matrix = matrix[np.asarray(mask, dtype=bool)]
    return np.mean(matrix, axis=0) if matrix.shape[0] else None


def load_run_summary(training_dir, attribution_dir):
    training = validate_run_bundle(training_dir)
    metadata = validate_attribution_bundle(attribution_dir)
    with np.load(attribution_dir / "attributions.npz", allow_pickle=False) as arrays:
        offsets = np.asarray(arrays["node_offsets"], dtype=np.int64)
        raw = np.asarray(arrays["expected_gradients"], dtype=np.float64)
        normalized = np.asarray(
            arrays["expected_gradients_abs_proportion"], dtype=np.float64
        )
        roles = np.asarray(arrays["node_roles"], dtype=np.int8)
        confusion = np.asarray(arrays["candidate_confusion"], dtype=np.int8)
        drnl = np.asarray(arrays["drnl_integrated_gradients"], dtype=np.float64)
        channel_abs = []
        channel_signed = []
        role_abs = []
        role_signed = []
        drnl_abs = []
        drnl_signed = []
        for index in range(len(offsets) - 1):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            channel_abs.append(np.sum(normalized[start:stop], axis=0))
            channel_signed.append(np.sum(raw[start:stop], axis=0))
            role_abs.append([
                float(np.sum(normalized[start:stop][roles[start:stop] == role]))
                for role in range(3)
            ])
            role_signed.append([
                float(np.sum(raw[start:stop][roles[start:stop] == role]))
                for role in range(3)
            ])
            drnl_abs.append(np.sum(np.abs(drnl[start:stop]), axis=0))
            drnl_signed.append(np.sum(drnl[start:stop], axis=0))
    channel_abs = np.asarray(channel_abs)
    channel_signed = np.asarray(channel_signed)
    families = metadata["feature_families"]
    family_abs = np.stack([
        np.sum(channel_abs[:, np.asarray([value == family for value in families])], axis=1)
        for family in FEATURE_FAMILY_ORDER
    ], axis=1)
    family_signed = np.stack([
        np.sum(channel_signed[:, np.asarray([value == family for value in families])], axis=1)
        for family in FEATURE_FAMILY_ORDER
    ], axis=1)
    metrics = training["metrics"]
    return {
        "foodweb": metrics["Foodweb"],
        "experiment_id": int(metrics["ExperimentID"]),
        "fallback": int(metrics.get("RolePoolFallbackUsed", 0)),
        "feature_names": metadata["feature_names"],
        "channel_abs": channel_abs,
        "channel_signed": channel_signed,
        "family_abs": family_abs,
        "family_signed": family_signed,
        "role_abs": np.asarray(role_abs),
        "role_signed": np.asarray(role_signed),
        "drnl_abs": np.asarray(drnl_abs),
        "drnl_signed": np.asarray(drnl_signed),
        "confusion": confusion,
    }


def web_matrices(runs, abs_key, signed_key, candidate_mask=None):
    by_web_abs = defaultdict(list)
    by_web_signed = defaultdict(list)
    for run in runs:
        mask = candidate_mask(run) if candidate_mask is not None else None
        absolute = mean_or_none(run[abs_key], mask)
        signed = mean_or_none(run[signed_key], mask)
        if absolute is not None:
            by_web_abs[run["foodweb"]].append(absolute)
            by_web_signed[run["foodweb"]].append(signed)
    webs = sorted(by_web_abs)
    return (
        webs,
        np.stack([np.mean(by_web_abs[web], axis=0) for web in webs]),
        np.stack([np.mean(by_web_signed[web], axis=0) for web in webs]),
    )


def summary_rows(runs, abs_key, signed_key, labels, group_type, group_value, replicates, rng):
    webs, absolute, signed = web_matrices(runs, abs_key, signed_key)
    abs_mean, abs_low, abs_high = bootstrap_equal_webs(absolute, replicates, rng)
    signed_mean, signed_low, signed_high = bootstrap_equal_webs(signed, replicates, rng)
    return [{
        "GroupType": group_type,
        "GroupValue": group_value,
        "Input": label,
        "Foodwebs": len(webs),
        "Aggregation": "candidate_to_run_to_foodweb_to_cohort_equal_foodweb_weight",
        "MeanAbsoluteNormalized": float(abs_mean[index]),
        "MeanAbsoluteNormalizedCI95Low": float(abs_low[index]),
        "MeanAbsoluteNormalizedCI95High": float(abs_high[index]),
        "MeanSigned": float(signed_mean[index]),
        "MeanSignedCI95Low": float(signed_low[index]),
        "MeanSignedCI95High": float(signed_high[index]),
        "BootstrapReplicates": int(replicates),
    } for index, label in enumerate(labels)]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Summary output already exists: {}".format(path))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--ecosystem-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=918273)
    args = parser.parse_args(argv)
    manifest = read_manifest(args.manifest)
    ecosystems = read_ecosystems(args.ecosystem_csv)
    runs = []
    for index, row in enumerate(manifest, 1):
        runs.append(load_run_summary(
            run_directory(args.result_root, row["Foodweb"], row["ExperimentID"], row["Seed"]),
            attribution_directory(
                args.result_root, row["Foodweb"], row["ExperimentID"], row["Seed"]
            ),
        ))
        if index % 250 == 0:
            print("[PROGRESS] loaded {}/{} bundles".format(index, len(manifest)))
    feature_names = runs[0]["feature_names"]
    if any(run["feature_names"] != feature_names for run in runs):
        raise ValueError("Feature schemas differ across attribution bundles")
    rng = np.random.default_rng(args.bootstrap_seed)
    common = dict(replicates=args.bootstrap_replicates, rng=rng)
    write_csv(
        args.output_dir / "importance_44_channels.csv",
        summary_rows(runs, "channel_abs", "channel_signed", feature_names, "cohort", "all", **common),
    )
    write_csv(
        args.output_dir / "importance_6_families.csv",
        summary_rows(runs, "family_abs", "family_signed", FEATURE_FAMILY_ORDER, "cohort", "all", **common),
    )
    write_csv(
        args.output_dir / "importance_node_roles.csv",
        summary_rows(
            runs, "role_abs", "role_signed",
            [ROLE_NAMES[index] for index in range(3)], "cohort", "all", **common
        ),
    )
    drnl_width = runs[0]["drnl_abs"].shape[1]
    write_csv(
        args.output_dir / "drnl_importance_separate.csv",
        summary_rows(
            runs, "drnl_abs", "drnl_signed",
            ["DRNL_{}".format(index) for index in range(drnl_width)],
            "cohort", "all", **common
        ),
    )

    confusion_rows = []
    for code, name in sorted(CONFUSION_NAMES.items()):
        subset_runs = []
        for run in runs:
            mask = run["confusion"] == code
            if np.any(mask):
                copied = dict(run)
                copied["family_abs"] = run["family_abs"][mask]
                copied["family_signed"] = run["family_signed"][mask]
                subset_runs.append(copied)
        if subset_runs:
            confusion_rows.extend(summary_rows(
                subset_runs, "family_abs", "family_signed", FEATURE_FAMILY_ORDER,
                "confusion", name, **common
            ))
    write_csv(args.output_dir / "importance_families_by_confusion.csv", confusion_rows)

    ecosystem_rows = []
    for ecosystem in sorted(set(ecosystems.values())):
        subset = [run for run in runs if ecosystems.get(run["foodweb"]) == ecosystem]
        if subset:
            ecosystem_rows.extend(summary_rows(
                subset, "family_abs", "family_signed", FEATURE_FAMILY_ORDER,
                "ecosystem", ecosystem, **common
            ))
    write_csv(args.output_dir / "importance_families_by_ecosystem.csv", ecosystem_rows)

    fallback_rows = []
    for state in (0, 1):
        subset = [run for run in runs if run["fallback"] == state]
        if subset:
            fallback_rows.extend(summary_rows(
                subset, "family_abs", "family_signed", FEATURE_FAMILY_ORDER,
                "negative_pool_fallback", str(state), **common
            ))
    write_csv(args.output_dir / "importance_families_by_fallback.csv", fallback_rows)
    report = {
        "schema_version": "seal-directed-attribution-summary-v1",
        "runs": len(runs),
        "foodwebs": len({run["foodweb"] for run in runs}),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_unit": "foodweb",
        "primary": "mean absolute attribution normalized within candidate",
        "secondary": "signed attribution",
        "drnl_reported_separately": True,
        "p_values": False,
        "interpretation": "model reliance on input attributes; not causal effects",
    }
    report_path = args.output_dir / "SUMMARY_METADATA.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
