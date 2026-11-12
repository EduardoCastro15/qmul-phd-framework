#!/usr/bin/env python3
"""Create immutable configs for node/ECC Expected- and Integrated-Gradients."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from run_artifacts import canonical_json_bytes, configuration_hash, file_sha256


CALIBRATION_SETTINGS = (32, 64, 128, 256)
ATTRIBUTION_VARIANTS = ("directed_node", "directed_ecc")

DEFAULT_ATTRIBUTION_CONFIG = {
    "schema_version": "seal-directed-node-ecc-attribution-config-v2",
    "version": "SEAL_directed_node_ecc_attribution_v2",
    "model_variants": list(ATTRIBUTION_VARIANTS),
    "target": "log_p_positive_minus_log_p_negative",
    "coverage": "all_test_candidates_exactly_once",
    "node_method": "Expected Gradients",
    "node_channels": 44,
    "node_background_source": "training_subgraphs_only",
    "node_background_roles": ["focal_resource", "focal_consumer", "context"],
    "expected_gradients_samples": 256,
    "drnl_method": "Integrated Gradients",
    "drnl_baseline": "all_zero_onehot",
    "drnl_steps": 256,
    "edge_method": "Expected Gradients",
    "edge_channels": 17,
    "edge_variants": ["directed_ecc"],
    "edge_background_source": "training_subgraph_edges_only",
    "edge_background_unit": "complete_17_channel_vector",
    "edge_background_roles": ["focal_resource", "focal_consumer", "context"],
    "edge_background_fallback": [
        "exact_ordered_role_pair", "one_endpoint_role", "global_training_pool"
    ],
    "edge_background_sampling": "with_replacement_deterministic",
    "calibration_candidates": list(CALIBRATION_SETTINGS),
    "calibration_reference": 256,
    "calibration_median_spearman_min": 0.95,
    "calibration_top10_overlap_min": 0.80,
    "calibration_edge_top5_overlap_min": 0.80,
    "calibration_completeness_relative_median_max": 0.01,
    "calibration_completeness_relative_p95_max": 0.05,
    "internal_batch_size": 32,
    "attribution_seed_offset": 1000000,
    "score_parity_absolute_tolerance": 1e-7,
    "captum_version": "0.9.0",
    "signed_attribution": True,
    "normalized_absolute_attribution": "within_candidate_and_channel",
    "causal_language_allowed": False,
    "faithfulness_background_samples": 32,
}


def write_attribution_config(output, setting=256, calibration_evidence=None,
                             faithfulness_background_samples=32):
    output = Path(output)
    setting = int(setting)
    if setting not in CALIBRATION_SETTINGS:
        raise ValueError("setting must be one of 32, 64, 128, 256")
    if output.exists():
        raise FileExistsError("Attribution config is immutable: {}".format(output))
    config = dict(DEFAULT_ATTRIBUTION_CONFIG)
    config["expected_gradients_samples"] = setting
    config["drnl_steps"] = setting
    config["faithfulness_background_samples"] = int(faithfulness_background_samples)
    if config["faithfulness_background_samples"] < 0:
        raise ValueError("faithfulness_background_samples cannot be negative")
    if calibration_evidence is not None:
        evidence = Path(calibration_evidence).resolve()
        if not evidence.is_file():
            raise FileNotFoundError(evidence)
        config["calibration_evidence_path"] = str(evidence)
        config["calibration_evidence_sha256"] = file_sha256(evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".{}.".format(output.name), dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(config) + b"\n")
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return config, configuration_hash(config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", type=int, choices=CALIBRATION_SETTINGS, default=256)
    parser.add_argument("--calibration-evidence", type=Path)
    parser.add_argument("--faithfulness-background-samples", type=int, default=32)
    args = parser.parse_args(argv)
    config, digest = write_attribution_config(
        args.output, args.setting, args.calibration_evidence,
        args.faithfulness_background_samples,
    )
    print(json.dumps({"status": "CONFIG_OK", "setting": config["expected_gradients_samples"],
                      "attribution_config_hash": digest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
