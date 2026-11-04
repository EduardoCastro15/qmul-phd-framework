#!/usr/bin/env python3
"""Select the least-cost calibrated EG/IG settings and freeze ATTRIBUTION_CONFIG."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from attribution_artifacts import attribution_directory, validate_attribution_bundle
from generate_attribution_config import DEFAULT_ATTRIBUTION_CONFIG, write_attribution_config
from run_foodweb_seal_directed import read_manifest
from seal_run_artifacts import canonical_json_bytes


SETTINGS = (32, 64, 128, 256)


def parse_calibration_roots(values):
    roots = {}
    for value in values:
        setting, separator, path = value.partition("=")
        if not separator or int(setting) not in SETTINGS:
            raise ValueError("Use --calibration-root N=/path for N in 32,64,128,256")
        roots[int(setting)] = Path(path).resolve()
    if set(roots) != set(SETTINGS):
        raise ValueError("Exactly four calibration roots (32, 64, 128, 256) are required")
    return roots


def candidate_vectors(arrays, method):
    offsets = np.asarray(arrays["node_offsets"], dtype=np.int64)
    matrix_name = "expected_gradients" if method == "expected_gradients" else "drnl_integrated_gradients"
    matrix = np.asarray(arrays[matrix_name], dtype=np.float64)
    return np.stack([
        np.sum(np.abs(matrix[int(offsets[index]):int(offsets[index + 1])]), axis=0)
        for index in range(len(offsets) - 1)
    ])


def rank_metrics(candidate, reference):
    correlation = spearmanr(candidate, reference).statistic
    if not np.isfinite(correlation):
        correlation = 1.0 if np.array_equal(candidate, reference) else 0.0
    k = min(10, candidate.size)
    candidate_top = set(np.argsort(candidate)[-k:])
    reference_top = set(np.argsort(reference)[-k:])
    return float(correlation), float(len(candidate_top & reference_top) / k)


def summarize_setting(records, method):
    correlations = np.asarray([record["spearman"] for record in records], dtype=float)
    overlaps = np.asarray([record["top10_overlap"] for record in records], dtype=float)
    completeness = np.asarray([record["completeness"] for record in records], dtype=float)
    return {
        "method": method,
        "candidate_count": int(len(records)),
        "median_spearman": float(np.median(correlations)),
        "median_top10_overlap": float(np.median(overlaps)),
        "completeness_relative_median": float(np.median(completeness)),
        "completeness_relative_p95": float(np.percentile(completeness, 95)),
    }


def passes(summary, config):
    return (
        summary["median_spearman"] >= config["calibration_median_spearman_min"]
        and summary["median_top10_overlap"] >= config["calibration_top10_overlap_min"]
        and summary["completeness_relative_median"]
        <= config["calibration_completeness_relative_median_max"]
        and summary["completeness_relative_p95"]
        <= config["calibration_completeness_relative_p95_max"]
    )


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Calibration output is immutable: {}".format(path))
    descriptor, temporary = tempfile.mkstemp(prefix=".{}.".format(path.name), dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration-root", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-foodwebs", type=int, default=290)
    args = parser.parse_args(argv)
    roots = parse_calibration_roots(args.calibration_root)
    rows = [row for row in read_manifest(args.manifest) if int(row["ExperimentID"]) == 1]
    if len(rows) != args.expected_foodwebs:
        raise ValueError(
            "Expected {} ExperimentID=1 rows, found {}".format(args.expected_foodwebs, len(rows))
        )
    config = dict(DEFAULT_ATTRIBUTION_CONFIG)
    records = {
        method: {setting: [] for setting in SETTINGS}
        for method in ("expected_gradients", "drnl_integrated_gradients")
    }
    for row in rows:
        loaded = {}
        for setting, root in roots.items():
            bundle = attribution_directory(
                root, row["Foodweb"], row["ExperimentID"], row["Seed"]
            )
            metadata = validate_attribution_bundle(bundle)
            if int(metadata["expected_gradients_samples"]) != setting or int(metadata["drnl_steps"]) != setting:
                raise ValueError("Calibration bundle setting mismatch: {}".format(bundle))
            loaded[setting] = np.load(bundle / "attributions.npz", allow_pickle=False)
        try:
            reference_eg = candidate_vectors(loaded[256], "expected_gradients")
            reference_drnl = candidate_vectors(loaded[256], "drnl_integrated_gradients")
            for setting in SETTINGS:
                for method, reference, completeness_name in (
                    ("expected_gradients", reference_eg, "eg_completeness_relative"),
                    ("drnl_integrated_gradients", reference_drnl, "drnl_completeness_relative"),
                ):
                    vectors = candidate_vectors(loaded[setting], method)
                    completeness = np.asarray(loaded[setting][completeness_name], dtype=float)
                    if vectors.shape != reference.shape or vectors.shape[0] != completeness.size:
                        raise ValueError("Calibration candidates do not align across settings")
                    for vector, reference_vector, error in zip(vectors, reference, completeness):
                        correlation, overlap = rank_metrics(vector, reference_vector)
                        records[method][setting].append({
                            "spearman": correlation,
                            "top10_overlap": overlap,
                            "completeness": float(error),
                        })
        finally:
            for arrays in loaded.values():
                arrays.close()

    summaries = {
        method: {
            str(setting): summarize_setting(method_records[setting], method)
            for setting in SETTINGS
        }
        for method, method_records in records.items()
    }
    selected = {}
    for method in summaries:
        qualifying = [
            setting for setting in SETTINGS if passes(summaries[method][str(setting)], config)
        ]
        if not qualifying:
            report = {
                "schema_version": "seal-directed-attribution-calibration-v1",
                "status": "FAILED_NO_CONFIGURATION_MEETS_GATES",
                "foodwebs": len(rows),
                "reference": 256,
                "summaries": summaries,
                "thresholds": {
                    key: value for key, value in config.items() if key.startswith("calibration_")
                },
            }
            args.output_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output_dir / "CALIBRATION_REPORT.json", report)
            raise SystemExit(
                "Calibration failed for {}; ExperimentID 2-10 must not be released".format(method)
            )
        selected[method] = min(qualifying)

    report = {
        "schema_version": "seal-directed-attribution-calibration-v1",
        "status": "CALIBRATION_OK",
        "foodwebs": len(rows),
        "reference": 256,
        "selected_expected_gradients_samples": selected["expected_gradients"],
        "selected_drnl_steps": selected["drnl_integrated_gradients"],
        "summaries": summaries,
        "thresholds": {
            key: value for key, value in config.items() if key.startswith("calibration_")
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "CALIBRATION_REPORT.json"
    atomic_json(report_path, report)
    final_config, final_hash = write_attribution_config(
        args.output_dir / "ATTRIBUTION_CONFIG.json",
        expected_gradients_samples=selected["expected_gradients"],
        drnl_steps=selected["drnl_integrated_gradients"],
        calibration_evidence=report_path,
    )
    print(json.dumps({
        "status": "CALIBRATION_OK",
        "expected_gradients_samples": final_config["expected_gradients_samples"],
        "drnl_steps": final_config["drnl_steps"],
        "attribution_config_hash": final_hash,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
