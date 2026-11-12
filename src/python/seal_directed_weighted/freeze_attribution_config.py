#!/usr/bin/env python3
"""Select one common calibrated setting for node, DRNL and ECC-edge attribution."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from attribution_artifacts import attribution_directory, validate_attribution_bundle
from generate_attribution_config import CALIBRATION_SETTINGS, DEFAULT_ATTRIBUTION_CONFIG, write_attribution_config
from run_artifacts import canonical_json_bytes, file_sha256
from run_campaign import read_manifest


def parse_roots(values):
    result = {}
    for value in values:
        setting, separator, path = value.partition("=")
        if not separator or int(setting) not in CALIBRATION_SETTINGS:
            raise ValueError("use --calibration-root N=/path for 32,64,128,256")
        result[int(setting)] = Path(path).resolve()
    if set(result) != set(CALIBRATION_SETTINGS):
        raise ValueError("exactly four calibration roots are required")
    return result


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".{}.".format(path.name), dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle: handle.write(canonical_json_bytes(payload) + b"\n")
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except OSError: pass
        raise


def candidate_vectors(arrays, channel):
    prefix = "edge" if channel == "edge" else "node"
    offsets = np.asarray(arrays["{}_offsets".format(prefix)], dtype=np.int64)
    name = {"node": "node_expected_gradients", "drnl": "drnl_integrated_gradients",
            "edge": "edge_expected_gradients"}[channel]
    matrix = np.asarray(arrays[name], dtype=np.float64)
    return np.stack([np.sum(np.abs(matrix[offsets[i]:offsets[i + 1]]), axis=0)
                     for i in range(len(offsets) - 1)])


def rank_metrics(candidate, reference, k):
    correlation = spearmanr(candidate, reference).statistic
    if not np.isfinite(correlation): correlation = 1.0 if np.array_equal(candidate, reference) else 0.0
    k = min(int(k), candidate.size)
    overlap = len(set(np.argsort(candidate)[-k:]) & set(np.argsort(reference)[-k:])) / float(k)
    return float(correlation), float(overlap)


def summarize(records):
    if not records:
        raise ValueError("calibration channel contains no attributable candidates")
    return {"candidate_count": len(records),
            "median_spearman": float(np.median([item[0] for item in records])),
            "median_topk_overlap": float(np.median([item[1] for item in records])),
            "completeness_relative_median": float(np.median([item[2] for item in records])),
            "completeness_relative_p95": float(np.percentile([item[2] for item in records], 95))}


def passes(summary, edge=False, require_faithfulness=False):
    config = DEFAULT_ATTRIBUTION_CONFIG
    converged = (summary["median_spearman"] >= config["calibration_median_spearman_min"]
            and summary["median_topk_overlap"] >= config[
                "calibration_edge_top5_overlap_min" if edge else "calibration_top10_overlap_min"]
            and summary["completeness_relative_median"] <= config["calibration_completeness_relative_median_max"]
            and summary["completeness_relative_p95"] <= config["calibration_completeness_relative_p95_max"])
    if require_faithfulness:
        converged = (converged and summary.get("faithfulness_top_median", 0.0)
                     > summary.get("faithfulness_bottom_median", 0.0))
    return converged


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration-root", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-foodwebs", type=int, default=290)
    args = parser.parse_args(argv)
    roots = parse_roots(args.calibration_root)
    rows = [row for row in read_manifest(args.manifest) if int(row["ExperimentID"]) == 1]
    if len(rows) != args.expected_foodwebs * 2:
        raise ValueError("expected {} ExperimentID=1 rows, found {}".format(args.expected_foodwebs * 2, len(rows)))
    records = {(variant, channel, setting): [] for variant in ("directed_node", "directed_ecc")
               for channel in (("node", "drnl") if variant == "directed_node" else ("node", "drnl", "edge"))
               for setting in CALIBRATION_SETTINGS}
    faithfulness = defaultdict(list)
    for row in rows:
        loaded = {}
        variant = row["ModelVariant"]
        for setting, root in roots.items():
            bundle = attribution_directory(root, variant, row["Foodweb"], 1, row["Seed"])
            metadata = validate_attribution_bundle(bundle)
            if metadata["expected_gradients_samples"] != setting or metadata["drnl_steps"] != setting:
                raise ValueError("calibration setting mismatch in {}".format(bundle))
            loaded[setting] = np.load(bundle / "attributions.npz", allow_pickle=False)
        try:
            channels = ("node", "drnl") if variant == "directed_node" else ("node", "drnl", "edge")
            references = {channel: candidate_vectors(loaded[256], channel) for channel in channels}
            for setting in CALIBRATION_SETTINGS:
                for channel in channels:
                    vectors = candidate_vectors(loaded[setting], channel)
                    completeness = np.asarray(loaded[setting]["{}_completeness_relative".format(channel)])
                    eligible = (
                        np.diff(np.asarray(loaded[setting]["edge_offsets"], dtype=np.int64)) > 0
                        if channel == "edge" else np.ones(len(vectors), dtype=bool)
                    )
                    for vector, reference, error in zip(
                        vectors[eligible], references[channel][eligible], completeness[eligible]
                    ):
                        correlation, overlap = rank_metrics(vector, reference, 5 if channel == "edge" else 10)
                        records[(variant, channel, setting)].append((correlation, overlap, float(error)))
                    if channel in ("node", "edge"):
                        top_name = "faithfulness_{}_top_abs_log_odds_change".format(channel)
                        bottom_name = "faithfulness_{}_bottom_abs_log_odds_change".format(channel)
                        if top_name not in loaded[setting].files or bottom_name not in loaded[setting].files:
                            raise ValueError("missing {} faithfulness calibration arrays".format(channel))
                        top = np.asarray(loaded[setting][top_name], dtype=float)
                        bottom = np.asarray(loaded[setting][bottom_name], dtype=float)
                        if channel == "edge":
                            top, bottom = top[eligible], bottom[eligible]
                        faithfulness[(variant, channel, setting)].extend(zip(top, bottom))
        finally:
            for arrays in loaded.values(): arrays.close()
    summaries = {variant: {channel: {str(setting): summarize(records[(variant, channel, setting)])
                                     for setting in CALIBRATION_SETTINGS}
                           for channel in (("node", "drnl") if variant == "directed_node" else ("node", "drnl", "edge"))}
                 for variant in ("directed_node", "directed_ecc")}
    for variant, channel, setting in faithfulness:
        values = np.asarray(faithfulness[(variant, channel, setting)], dtype=float)
        summaries[variant][channel][str(setting)].update({
            "faithfulness_top_median": float(np.median(values[:, 0])),
            "faithfulness_bottom_median": float(np.median(values[:, 1])),
            "faithfulness_top_exceeds_bottom": bool(np.median(values[:, 0]) > np.median(values[:, 1])),
        })
    qualifying = []
    for setting in CALIBRATION_SETTINGS:
        if all(passes(summaries[variant][channel][str(setting)], channel == "edge",
                      channel in ("node", "edge"))
               for variant in summaries for channel in summaries[variant]):
            qualifying.append(setting)
    status = "CALIBRATION_OK" if qualifying else "CALIBRATION_FAILED"
    report = {"schema_version": "seal-directed-node-ecc-attribution-calibration-v2",
              "status": status, "foodwebs": args.expected_foodwebs, "reference": 256,
              "selected_common_setting": min(qualifying) if qualifying else None,
              "summaries": summaries,
              "thresholds": {key: value for key, value in DEFAULT_ATTRIBUTION_CONFIG.items()
                             if key.startswith("calibration_")}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "CALIBRATION_REPORT.json"; atomic_json(report_path, report)
    if not qualifying:
        print(json.dumps(report, indent=2, sort_keys=True)); raise SystemExit(1)
    config, digest = write_attribution_config(
        args.output_dir / "ATTRIBUTION_CONFIG.json", min(qualifying), report_path,
        faithfulness_background_samples=0,
    )
    atomic_json(args.output_dir / "ATTRIBUTION_CONFIG_LOCK.json", {
        "schema_version": "seal-directed-node-ecc-attribution-config-lock-v2",
        "selected_common_setting": min(qualifying),
        "attribution_config_hash": digest,
        "attribution_config_sha256": file_sha256(
            args.output_dir / "ATTRIBUTION_CONFIG.json"
        ),
        "calibration_report_sha256": file_sha256(report_path),
    })
    report["attribution_config_hash"] = digest
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__": main()
