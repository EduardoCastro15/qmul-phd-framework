#!/usr/bin/env python3
"""Validate randomization, permuted-label and top-vs-bottom faithfulness controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from attribution_artifacts import attribution_directory, validate_attribution_bundle
from run_foodweb_seal_directed import read_manifest


STAGES = ("output", "mlp", "sortpool_head", "message_passing")


def parse_roots(values):
    result = {}
    for value in values:
        stage, separator, path = value.partition("=")
        if not separator or stage not in STAGES:
            raise ValueError("Use --randomized-root STAGE=/path")
        result[stage] = Path(path).resolve()
    if set(result) != set(STAGES):
        raise ValueError("All four progressive randomization roots are required")
    return result


def vectors(arrays):
    offsets = np.asarray(arrays["node_offsets"], dtype=np.int64)
    values = np.asarray(arrays["expected_gradients"], dtype=float)
    return np.stack([
        np.sum(np.abs(values[int(offsets[index]):int(offsets[index + 1])]), axis=0)
        for index in range(len(offsets) - 1)
    ])


def correlation(left, right):
    value = spearmanr(left, right).statistic
    if not np.isfinite(value):
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--valid-root", type=Path, required=True)
    parser.add_argument("--randomized-root", action="append", required=True)
    parser.add_argument("--permuted-label-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-foodwebs", type=int, default=3)
    parser.add_argument("--max-median-absolute-correlation", type=float, default=0.5)
    args = parser.parse_args(argv)
    random_roots = parse_roots(args.randomized_root)
    rows = [row for row in read_manifest(args.manifest) if int(row["ExperimentID"]) == 1]
    if len(rows) != args.expected_foodwebs:
        raise ValueError("Sanity manifest must contain {} ExperimentID=1 webs".format(args.expected_foodwebs))
    correlations = {stage: [] for stage in STAGES}
    correlations["permuted_labels"] = []
    top_reductions = []
    bottom_reductions = []
    for row in rows:
        valid_dir = attribution_directory(
            args.valid_root, row["Foodweb"], row["ExperimentID"], row["Seed"]
        )
        validate_attribution_bundle(valid_dir)
        with np.load(valid_dir / "attributions.npz", allow_pickle=False) as valid_arrays:
            valid_vectors = vectors(valid_arrays)
            if "faithfulness_top_log_odds_reduction" not in valid_arrays.files:
                raise ValueError("Valid sanity bundle lacks faithfulness substitution results")
            top_reductions.extend(valid_arrays["faithfulness_top_log_odds_reduction"].tolist())
            bottom_reductions.extend(valid_arrays["faithfulness_bottom_log_odds_reduction"].tolist())
            valid_pairs = valid_arrays["candidate_pairs"].copy()
        for stage, root in random_roots.items():
            control_dir = attribution_directory(
                root, row["Foodweb"], row["ExperimentID"], row["Seed"]
            )
            metadata = validate_attribution_bundle(control_dir)
            if metadata.get("sanity_randomization_through") != stage:
                raise ValueError("Randomization stage metadata mismatch")
            with np.load(control_dir / "attributions.npz", allow_pickle=False) as control:
                if not np.array_equal(control["candidate_pairs"], valid_pairs):
                    raise ValueError("Randomized control candidates do not align")
                for left, right in zip(valid_vectors, vectors(control)):
                    correlations[stage].append(abs(correlation(left, right)))
        permuted_dir = attribution_directory(
            args.permuted_label_root, row["Foodweb"], row["ExperimentID"], row["Seed"]
        )
        permuted_metadata = validate_attribution_bundle(permuted_dir)
        if not permuted_metadata.get("label_permutation_control"):
            raise ValueError("Permuted-label bundle is not marked as a control")
        with np.load(permuted_dir / "attributions.npz", allow_pickle=False) as control:
            if not np.array_equal(control["candidate_pairs"], valid_pairs):
                raise ValueError("Permuted-label candidates do not align")
            for left, right in zip(valid_vectors, vectors(control)):
                correlations["permuted_labels"].append(abs(correlation(left, right)))

    medians = {name: float(np.median(values)) for name, values in correlations.items()}
    faithfulness = {
        "top_ranked_median_log_odds_reduction": float(np.median(top_reductions)),
        "bottom_ranked_median_log_odds_reduction": float(np.median(bottom_reductions)),
    }
    failures = []
    if medians["message_passing"] >= args.max_median_absolute_correlation:
        failures.append("fully randomized attribution correlation is not below threshold")
    if medians["permuted_labels"] >= args.max_median_absolute_correlation:
        failures.append("permuted-label attribution correlation is not below threshold")
    if faithfulness["top_ranked_median_log_odds_reduction"] <= faithfulness[
        "bottom_ranked_median_log_odds_reduction"
    ]:
        failures.append("top-ranked feature families are not more faithful than bottom-ranked families")
    report = {
        "schema_version": "seal-directed-attribution-sanity-v1",
        "status": "SANITY_OK" if not failures else "SANITY_FAILED",
        "foodwebs": len(rows),
        "median_absolute_spearman": medians,
        "required_max_median_absolute_correlation": args.max_median_absolute_correlation,
        "faithfulness": faithfulness,
        "failures": failures,
        "interpretation": "model reliance sanity checks; not causal effects",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Sanity report is immutable: {}".format(args.output))
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
