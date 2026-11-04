#!/usr/bin/env python3
"""Write one immutable, post-calibration SEAL-D-NodeAttr attribution config."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from nodeattr_schema import DISPLAY_NAME, MODEL_ID
from seal_run_artifacts import canonical_json_bytes, configuration_hash, file_sha256


DEFAULT_ATTRIBUTION_CONFIG = {
    "schema_version": "seal-directed-attribution-config-v1",
    "ModelID": MODEL_ID,
    "DisplayName": DISPLAY_NAME,
    "attribute_method": "Expected Gradients",
    "attribute_target": "log_p_positive_minus_log_p_negative",
    "attribute_channels": 44,
    "attribute_background_source": "training_subgraphs_only",
    "attribute_background_roles": ["source_resource", "target_consumer", "context"],
    "attribute_context_completion": "within_reference_context_sampling_with_replacement",
    "expected_gradients_samples": 256,
    "expected_gradients_calibration_candidates": [32, 64, 128, 256],
    "drnl_method": "Integrated Gradients",
    "drnl_baseline": "all_zero_onehot",
    "drnl_steps": 256,
    "drnl_calibration_candidates": [32, 64, 128, 256],
    "internal_batch_size": 32,
    "attribution_seed_offset": 1000000,
    "coverage": "all_test_candidates_exactly_once",
    "captum_version": "0.9.0",
    "score_parity_absolute_tolerance": 1e-7,
    "calibration_reference": 256,
    "calibration_median_spearman_min": 0.95,
    "calibration_top10_overlap_min": 0.80,
    "calibration_completeness_relative_median_max": 0.01,
    "calibration_completeness_relative_p95_max": 0.05,
    "signed_attribution": True,
    "normalized_absolute_attribution": "within_candidate",
    "causal_language_allowed": False,
    "faithfulness_background_samples": 0,
}


def write_attribution_config(
    output,
    expected_gradients_samples=256,
    drnl_steps=256,
    calibration_evidence=None,
    faithfulness_background_samples=0,
):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Attribution configuration is immutable: {}".format(output))
    config = dict(DEFAULT_ATTRIBUTION_CONFIG)
    config["expected_gradients_samples"] = int(expected_gradients_samples)
    config["drnl_steps"] = int(drnl_steps)
    config["faithfulness_background_samples"] = int(faithfulness_background_samples)
    if config["faithfulness_background_samples"] < 0:
        raise ValueError("faithfulness_background_samples cannot be negative")
    allowed = set(config["expected_gradients_calibration_candidates"])
    if config["expected_gradients_samples"] not in allowed or config["drnl_steps"] not in allowed:
        raise ValueError("Selected values must be one of 32, 64, 128, 256")
    if calibration_evidence is not None:
        evidence = Path(calibration_evidence).resolve()
        if not evidence.is_file():
            raise FileNotFoundError(evidence)
        config["calibration_evidence_path"] = str(evidence)
        config["calibration_evidence_sha256"] = file_sha256(evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".{}.".format(output.name), dir=str(output.parent))
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
    parser.add_argument("--expected-gradients-samples", type=int, default=256)
    parser.add_argument("--drnl-steps", type=int, default=256)
    parser.add_argument("--calibration-evidence", type=Path, default=None)
    parser.add_argument("--faithfulness-background-samples", type=int, default=0)
    args = parser.parse_args(argv)
    _, digest = write_attribution_config(
        args.output,
        args.expected_gradients_samples,
        args.drnl_steps,
        args.calibration_evidence,
        args.faithfulness_background_samples,
    )
    print("[OK] Attribution config: {}".format(args.output.resolve()))
    print("[OK] Attribution config hash: {}".format(digest))


if __name__ == "__main__":
    main()
