#!/usr/bin/env python3
"""Descriptive paired summaries and ecosystem-stratified bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("PR_AUC", "ROC_AUC", "F1Score", "TestMCC", "Precision", "Recall", "TestTSS")
VARIANTS = ("legacy_weak_node", "directed_node", "directed_scalar_gate", "directed_ecc")


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def bootstrap_stratified(values, ecosystems, replicates=10000, seed=73201):
    values = np.asarray(values, dtype=float)
    ecosystems = np.asarray(ecosystems, dtype=object)
    rng = np.random.RandomState(int(seed))
    strata = [np.where(ecosystems == ecosystem)[0] for ecosystem in sorted(set(ecosystems))]
    estimates = np.empty(int(replicates), dtype=float)
    for replicate in range(int(replicates)):
        sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in strata])
        estimates[replicate] = float(np.mean(values[sampled]))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foodweb-means", type=Path, required=True)
    parser.add_argument("--raw-runs", type=Path, required=True)
    parser.add_argument("--ecosystem-csv", type=Path, required=True)
    parser.add_argument("--weighted-mat-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=73201)
    args = parser.parse_args(argv)

    rows = read_csv(args.foodweb_means)
    raw_runs = read_csv(args.raw_runs)
    ecosystem_rows = read_csv(args.ecosystem_csv)
    audit_rows = read_csv(args.weighted_mat_audit)
    ecosystem = {row["Foodweb"]: row.get("EcosystemType", row.get("Ecosystem", "unknown")) for row in ecosystem_rows}
    coverage = {
        row["foodweb"]: float(row["finite_mass_ratios"]) / float(row["links"])
        for row in audit_rows if float(row["links"]) > 0
    }
    by_web = defaultdict(dict)
    for row in rows:
        by_web[row["Foodweb"]][row["ModelVariant"]] = row
    if any(set(variants) != set(VARIANTS) for variants in by_web.values()):
        raise ValueError("every food web must contain all four variants")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    distribution_rows = []
    for variant in VARIANTS:
        for metric in METRICS:
            values = np.asarray([float(by_web[web][variant][metric]) for web in sorted(by_web)])
            distribution_rows.append({
                "ModelVariant": variant,
                "Metric": metric,
                "N": len(values),
                "Median": float(np.median(values)),
                "Q1": float(np.quantile(values, 0.25)),
                "Q3": float(np.quantile(values, 0.75)),
                "Mean": float(np.mean(values)),
            })
    with open(args.output_dir / "descriptive_distributions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distribution_rows[0]))
        writer.writeheader(); writer.writerows(distribution_rows)

    contrast_rows = []
    web_names = sorted(by_web)
    web_ecosystems = [ecosystem.get(web, "unknown") for web in web_names]
    for comparator in ("legacy_weak_node", "directed_node", "directed_scalar_gate"):
        for metric in METRICS:
            differences = np.asarray([
                float(by_web[web]["directed_ecc"][metric]) - float(by_web[web][comparator][metric])
                for web in web_names
            ])
            lower, upper = bootstrap_stratified(
                differences, web_ecosystems, args.bootstrap_replicates, args.seed
            )
            contrast_rows.append({
                "Contrast": "directed_ecc-minus-{}".format(comparator),
                "Metric": metric,
                "N": len(differences),
                "MeanDifference": float(np.mean(differences)),
                "MedianDifference": float(np.median(differences)),
                "ImprovedFraction": float(np.mean(differences > 0)),
                "Bootstrap95Lower": lower,
                "Bootstrap95Upper": upper,
                "BootstrapReplicates": int(args.bootstrap_replicates),
                "Inference": "descriptive_CI_no_p_values",
            })
    with open(args.output_dir / "paired_bootstrap_descriptive.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrast_rows[0]))
        writer.writeheader(); writer.writerows(contrast_rows)

    # Run-level fallback contrasts use the paired split identity, never
    # independent samples. Missingness strata use median mass-ratio coverage.
    raw_by_key = defaultdict(dict)
    for row in raw_runs:
        key = (row["Foodweb"], int(float(row["ExperimentID"])), row["SplitHash"])
        raw_by_key[key][row["ModelVariant"]] = row
    coverage_values = np.asarray([coverage.get(web, np.nan) for web in web_names], dtype=float)
    coverage_median = float(np.nanmedian(coverage_values))
    strata_rows = []
    for key, variants in sorted(raw_by_key.items()):
        if set(variants) != set(VARIANTS):
            raise ValueError("raw split is missing variants: {}".format(key))
        foodweb = key[0]
        fallback = int(float(variants["directed_node"]["RolePoolFallbackUsed"]))
        for variant in variants.values():
            if int(float(variant["RolePoolFallbackUsed"])) != fallback:
                raise ValueError("fallback status differs across variants for {}".format(key))
        mass_coverage = coverage.get(foodweb, np.nan)
        for metric in METRICS:
            difference = float(variants["directed_ecc"][metric]) - float(variants["directed_node"][metric])
            strata_rows.append({
                "Foodweb": foodweb,
                "ExperimentID": key[1],
                "SplitHash": key[2],
                "Metric": metric,
                "DifferenceECCMinusDirectedNode": difference,
                "Ecosystem": ecosystem.get(foodweb, "unknown"),
                "FallbackStratum": "fallback" if fallback else "no_fallback",
                "MassRatioCoverage": mass_coverage,
                "MissingnessStratum": (
                    "low_missingness" if np.isfinite(mass_coverage) and mass_coverage >= coverage_median
                    else "high_missingness"
                ),
            })
    with open(args.output_dir / "ecc_directed_node_stratified_run_differences.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(strata_rows[0]))
        writer.writeheader(); writer.writerows(strata_rows)

    stratum_summary = []
    for field in ("FallbackStratum", "MissingnessStratum", "Ecosystem"):
        values_by_stratum = defaultdict(list)
        for row in strata_rows:
            if row["Metric"] == "PR_AUC":
                values_by_stratum[row[field]].append(float(row["DifferenceECCMinusDirectedNode"]))
        for stratum, values in sorted(values_by_stratum.items()):
            array = np.asarray(values, dtype=float)
            stratum_summary.append({
                "Stratification": field,
                "Stratum": stratum,
                "Metric": "PR_AUC",
                "NPairedRuns": len(array),
                "MeanDifference": float(np.mean(array)),
                "MedianDifference": float(np.median(array)),
                "ImprovedFraction": float(np.mean(array > 0)),
            })
    with open(args.output_dir / "ecc_directed_node_strata_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stratum_summary[0]))
        writer.writeheader(); writer.writerows(stratum_summary)

    primary = next(
        row for row in contrast_rows
        if row["Contrast"] == "directed_ecc-minus-directed_node" and row["Metric"] == "PR_AUC"
    )
    secondary = {
        row["Metric"]: row for row in contrast_rows
        if row["Contrast"] == "directed_ecc-minus-directed_node"
    }
    ecosystem_means = {
        row["Stratum"]: float(row["MeanDifference"])
        for row in stratum_summary if row["Stratification"] == "Ecosystem"
    }
    robustness = {
        row["Stratum"]: float(row["MeanDifference"])
        for row in stratum_summary
        if row["Stratification"] in {"FallbackStratum", "MissingnessStratum"}
    }
    checks = {
        "pr_auc_bootstrap_lower_positive": float(primary["Bootstrap95Lower"]) > 0,
        "positive_in_at_least_four_ecosystems": sum(value > 0 for value in ecosystem_means.values()) >= 4,
        "improves_at_least_60_percent_foodwebs": float(primary["ImprovedFraction"]) >= 0.60,
        "roc_auc_not_worse_than_minus_0p01": float(secondary["ROC_AUC"]["MeanDifference"]) >= -0.01,
        "mcc_not_worse_than_minus_0p01": float(secondary["TestMCC"]["MeanDifference"]) >= -0.01,
        "robustness_strata_do_not_reverse": all(value >= 0 for value in robustness.values()),
    }
    decision = {
        "decision": "PROMOTE_ECC" if all(checks.values()) else "DO_NOT_PROMOTE_ECC",
        "checks": checks,
        "ecosystem_mean_pr_auc_differences": ecosystem_means,
        "robustness_mean_pr_auc_differences": robustness,
        "mass_ratio_coverage_median": coverage_median,
        "interpretation": "descriptive_predeclared_rule_no_p_values",
    }
    with open(args.output_dir / "ECC_GO_NO_GO.json", "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("[OK] Wrote descriptive summaries without p-values to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
