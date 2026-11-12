#!/usr/bin/env python3
"""Equal-food-web summaries and deterministic beeswarm samples for attribution v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from attribution_artifacts import attribution_directory, validate_attribution_bundle
from edge_schema import FEATURE_NAMES as EDGE_FEATURE_NAMES
from generate_attribution_config import ATTRIBUTION_VARIANTS
from run_campaign import read_manifest


ROLE_NAMES = {0: "focal_resource", 1: "focal_consumer", 2: "context"}
CONFUSION_NAMES = {0: "TN", 1: "FP", 2: "FN", 3: "TP"}
NODE_FAMILY_ORDER = (
    "mass", "mass_missing", "taxonomy", "metabolism", "movement", "life_stage",
)
EDGE_FAMILY_ORDER = (
    "mass_ratio", "mass_missingness", "interaction_type",
    "interaction_dimensionality", "interaction_classification",
)


def node_family(name):
    if name == "log10_mass_z": return "mass"
    if name == "mass_missing": return "mass_missing"
    if name.startswith("taxonomy_level="): return "taxonomy"
    if name.startswith("metabolic_type="): return "metabolism"
    if name.startswith("movement_type="): return "movement"
    if name == "lifestage_missing" or name.startswith("lifestage="): return "life_stage"
    return "unmapped"


def edge_family(index):
    if index == 0: return "mass_ratio"
    if index <= 2: return "mass_missingness"
    if index <= 10: return "interaction_type"
    if index <= 13: return "interaction_dimensionality"
    return "interaction_classification"


def bootstrap_equal_webs(matrix, replicates, rng):
    matrix = np.asarray(matrix, dtype=np.float64)
    observed = np.mean(matrix, axis=0)
    if len(matrix) == 1:
        return observed, observed.copy(), observed.copy()
    values = np.empty((int(replicates), matrix.shape[1]), dtype=np.float64)
    for start in range(0, int(replicates), 250):
        stop = min(start + 250, int(replicates))
        indices = rng.integers(0, len(matrix), size=(stop - start, len(matrix)))
        values[start:stop] = np.mean(matrix[indices], axis=1)
    return observed, np.percentile(values, 2.5, axis=0), np.percentile(values, 97.5, axis=0)


class DeterministicReservoir:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.groups = defaultdict(list)

    def add(self, group, identity, row):
        priority = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
        item = (-priority, identity, row)
        heap = self.groups[group]
        if len(heap) < self.capacity:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    def rows(self):
        result = []
        for group in sorted(self.groups):
            result.extend(item[2] for item in sorted(self.groups[group], key=lambda value: value[1]))
        return result


def candidate_sums(matrix, offsets, absolute=False):
    # Attribution bundles store float32 values. Keep that compact dtype while
    # retaining one candidate-level row; cohort_rows promotes means to float64.
    matrix = np.asarray(matrix, dtype=np.float32)
    return np.stack([np.sum(
        np.abs(matrix[int(offsets[i]):int(offsets[i + 1])]) if absolute
        else matrix[int(offsets[i]):int(offsets[i + 1])], axis=0)
        for i in range(len(offsets) - 1)]).astype(np.float32, copy=False)


def collapse_feature_families(matrix, feature_families, family_order):
    """Sum feature contributions into a fixed, auditable family order."""
    matrix = np.asarray(matrix)
    feature_families = tuple(str(value) for value in feature_families)
    family_order = tuple(str(value) for value in family_order)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_families):
        raise ValueError("feature matrix and family labels are not aligned")
    unknown = sorted(set(feature_families).difference(family_order))
    missing = sorted(set(family_order).difference(feature_families))
    if unknown or missing:
        raise ValueError(
            "family schema mismatch; unknown={} missing={}".format(unknown, missing)
        )
    return np.stack([
        np.sum(matrix[:, np.asarray(feature_families) == family], axis=1)
        for family in family_order
    ], axis=1)


def family_cohort_rows(runs, abs_key, signed_key, feature_families, family_order,
                       group_type, group_value, replicates, rng, selector=None):
    collapsed = [dict(
        run,
        family_abs=collapse_feature_families(
            run[abs_key], feature_families, family_order
        ),
        family_signed=collapse_feature_families(
            run[signed_key], feature_families, family_order
        ),
    ) for run in runs]
    rows = cohort_rows(
        collapsed, "family_abs", "family_signed", family_order, family_order,
        group_type, group_value, replicates, rng, selector,
    )
    result = []
    for row in rows:
        converted = {
            "ModelVariant": row["ModelVariant"],
            "GroupType": row["GroupType"],
            "GroupValue": row["GroupValue"],
            "Family": row["Feature"],
        }
        converted.update({
            key: value for key, value in row.items()
            if key not in {"ModelVariant", "GroupType", "GroupValue", "Feature", "FeatureFamily"}
        })
        result.append(converted)
    return result


def _reservoir_nodes(row, variant, arrays, names, reservoir):
    offsets = arrays["node_offsets"]
    roles, values = arrays["node_roles"], arrays["node_feature_values"]
    attr = arrays["node_expected_gradients"]
    labels, confusion = arrays["candidate_labels"], arrays["candidate_confusion"]
    for candidate in range(len(labels)):
        start, stop = offsets[candidate:candidate + 2]
        for local in range(int(start), int(stop)):
            role = ROLE_NAMES[int(roles[local])]
            for feature, name in enumerate(names):
                identity = "N|{}|{}|{}|{}|{}|{}".format(
                    variant, row["Foodweb"], row["ExperimentID"], candidate, local, feature)
                reservoir.add((variant, name, row["Foodweb"], role), identity, {
                    "ModelVariant": variant, "Feature": name, "FeatureFamily": node_family(name),
                    "Value": float(values[local, feature]), "Attribution": float(attr[local, feature]),
                    "Foodweb": row["Foodweb"], "ExperimentID": int(row["ExperimentID"]),
                    "Role": role, "CandidateLabel": int(labels[candidate]),
                    "Confusion": CONFUSION_NAMES[int(confusion[candidate])],
                })


def _reservoir_drnl(row, variant, arrays, reservoir):
    offsets, roles = arrays["node_offsets"], arrays["node_roles"]
    values, attr = arrays["drnl_labels"], arrays["drnl_integrated_gradients"]
    labels, confusion = arrays["candidate_labels"], arrays["candidate_confusion"]
    for candidate in range(len(labels)):
        start, stop = offsets[candidate:candidate + 2]
        for local in range(int(start), int(stop)):
            role = ROLE_NAMES[int(roles[local])]
            for feature in range(attr.shape[1]):
                name = "DRNL_{}".format(feature)
                identity = "D|{}|{}|{}|{}|{}|{}".format(
                    variant, row["Foodweb"], row["ExperimentID"], candidate, local, feature)
                reservoir.add((variant, name, row["Foodweb"], role), identity, {
                    "ModelVariant": variant, "Feature": name, "FeatureFamily": "DRNL",
                    "Value": int(values[local] == feature), "Attribution": float(attr[local, feature]),
                    "Foodweb": row["Foodweb"], "ExperimentID": int(row["ExperimentID"]),
                    "Role": role, "CandidateLabel": int(labels[candidate]),
                    "Confusion": CONFUSION_NAMES[int(confusion[candidate])],
                })


def _reservoir_edges(row, variant, arrays, names, reservoir):
    offsets, roles = arrays["edge_offsets"], arrays["edge_role_pairs"]
    values, attr = arrays["edge_feature_values"], arrays["edge_expected_gradients"]
    labels, confusion = arrays["candidate_labels"], arrays["candidate_confusion"]
    for candidate in range(len(labels)):
        start, stop = offsets[candidate:candidate + 2]
        for local in range(int(start), int(stop)):
            role = "{}->{}".format(ROLE_NAMES[int(roles[local, 0])], ROLE_NAMES[int(roles[local, 1])])
            for feature, name in enumerate(names):
                identity = "E|{}|{}|{}|{}|{}|{}".format(
                    variant, row["Foodweb"], row["ExperimentID"], candidate, local, feature)
                reservoir.add((variant, name, row["Foodweb"], role), identity, {
                    "ModelVariant": variant, "Feature": name, "FeatureFamily": edge_family(feature),
                    "Value": float(values[local, feature]), "Attribution": float(attr[local, feature]),
                    "Foodweb": row["Foodweb"], "ExperimentID": int(row["ExperimentID"]),
                    "Role": role, "CandidateLabel": int(labels[candidate]),
                    "Confusion": CONFUSION_NAMES[int(confusion[candidate])],
                })


def load_run(row, root, node_sample, edge_sample, drnl_sample):
    variant = row["ModelVariant"]
    bundle = attribution_directory(root, variant, row["Foodweb"], row["ExperimentID"], row["Seed"])
    metadata = validate_attribution_bundle(bundle)
    with np.load(bundle / "attributions.npz", allow_pickle=False) as handle:
        a = {name: np.asarray(handle[name]) for name in handle.files}
    node_names = list(metadata["node_feature_names"])
    edge_names = list(metadata.get("edge_feature_names", EDGE_FEATURE_NAMES))
    _reservoir_nodes(row, variant, a, node_names, node_sample)
    _reservoir_drnl(row, variant, a, drnl_sample)
    _reservoir_edges(row, variant, a, edge_names, edge_sample)
    node_role_abs, node_role_signed = {}, {}
    drnl_role_abs, drnl_role_signed = {}, {}
    for code, name in ROLE_NAMES.items():
        absolute, signed, drnl_absolute, drnl_signed = [], [], [], []
        for index in range(len(a["candidate_labels"])):
            start, stop = a["node_offsets"][index:index + 2]
            mask = a["node_roles"][start:stop] == code
            absolute.append(np.sum(a["node_expected_gradients_abs_proportion"][start:stop][mask], axis=0))
            signed.append(np.sum(a["node_expected_gradients"][start:stop][mask], axis=0))
            drnl_absolute.append(np.sum(
                a["drnl_integrated_gradients_abs_proportion"][start:stop][mask], axis=0
            ))
            drnl_signed.append(np.sum(
                a["drnl_integrated_gradients"][start:stop][mask], axis=0
            ))
        node_role_abs[name], node_role_signed[name] = (
            np.asarray(absolute, dtype=np.float32),
            np.asarray(signed, dtype=np.float32),
        )
        drnl_role_abs[name], drnl_role_signed[name] = (
            np.asarray(drnl_absolute, dtype=np.float32),
            np.asarray(drnl_signed, dtype=np.float32),
        )
    edge_role_abs, edge_role_signed = {}, {}
    for left in ROLE_NAMES:
        for right in ROLE_NAMES:
            name, absolute, signed = "{}->{}".format(ROLE_NAMES[left], ROLE_NAMES[right]), [], []
            for index in range(len(a["candidate_labels"])):
                start, stop = a["edge_offsets"][index:index + 2]
                mask = ((a["edge_role_pairs"][start:stop, 0] == left)
                        & (a["edge_role_pairs"][start:stop, 1] == right))
                absolute.append(np.sum(a["edge_expected_gradients_abs_proportion"][start:stop][mask], axis=0))
                signed.append(np.sum(a["edge_expected_gradients"][start:stop][mask], axis=0))
            edge_role_abs[name], edge_role_signed[name] = (
                np.asarray(absolute, dtype=np.float32),
                np.asarray(signed, dtype=np.float32),
            )
    candidates = len(a["candidate_labels"])
    return {
        "variant": variant, "foodweb": row["Foodweb"], "experiment": int(row["ExperimentID"]),
        "confusion": a["candidate_confusion"], "node_names": node_names, "edge_names": edge_names,
        "node_abs": candidate_sums(a["node_expected_gradients_abs_proportion"], a["node_offsets"]),
        "node_signed": candidate_sums(a["node_expected_gradients"], a["node_offsets"]),
        "node_role_abs": node_role_abs, "node_role_signed": node_role_signed,
        "drnl_abs": candidate_sums(a["drnl_integrated_gradients_abs_proportion"], a["node_offsets"]),
        "drnl_signed": candidate_sums(a["drnl_integrated_gradients"], a["node_offsets"]),
        "drnl_role_abs": drnl_role_abs, "drnl_role_signed": drnl_role_signed,
        "edge_abs": (candidate_sums(a["edge_expected_gradients_abs_proportion"], a["edge_offsets"])
                     if len(a["edge_expected_gradients"]) else np.zeros((candidates, 17), dtype=np.float32)),
        "edge_signed": (candidate_sums(a["edge_expected_gradients"], a["edge_offsets"])
                        if len(a["edge_expected_gradients"]) else np.zeros((candidates, 17), dtype=np.float32)),
        "edge_role_abs": edge_role_abs, "edge_role_signed": edge_role_signed,
    }


def cohort_rows(runs, abs_key, signed_key, features, families, group_type, group_value,
                replicates, rng, selector=None):
    by_abs, by_signed, run_count = defaultdict(list), defaultdict(list), 0
    for run in runs:
        absolute, signed = run[abs_key], run[signed_key]
        mask = selector(run) if selector else np.ones(len(absolute), dtype=bool)
        if np.any(mask):
            by_abs[run["foodweb"]].append(
                np.mean(absolute[mask], axis=0, dtype=np.float64)
            )
            by_signed[run["foodweb"]].append(
                np.mean(signed[mask], axis=0, dtype=np.float64)
            ); run_count += 1
    webs = sorted(by_abs)
    if not webs: return []
    absolute = np.stack([np.mean(by_abs[web], axis=0) for web in webs])
    signed = np.stack([np.mean(by_signed[web], axis=0) for web in webs])
    am, al, ah = bootstrap_equal_webs(absolute, replicates, rng)
    sm, sl, sh = bootstrap_equal_webs(signed, replicates, rng)
    return [{
        "ModelVariant": runs[0]["variant"], "GroupType": group_type, "GroupValue": group_value,
        "Feature": feature, "FeatureFamily": families[index], "Foodwebs": len(webs),
        "Runs": run_count,
        "Aggregation": "candidate_to_experiment_to_foodweb_to_cohort_equal_foodweb_weight",
        "MeanAbsoluteNormalized": float(am[index]), "MeanAbsoluteNormalizedCI95Low": float(al[index]),
        "MeanAbsoluteNormalizedCI95High": float(ah[index]), "MeanSignedLogOdds": float(sm[index]),
        "MeanSignedLogOddsCI95Low": float(sl[index]), "MeanSignedLogOddsCI95High": float(sh[index]),
        "BootstrapReplicates": int(replicates),
    } for index, feature in enumerate(features)]


def align_drnl_widths(runs):
    """Pad run-specific DRNL widths without changing any observed channel."""
    if not runs:
        return 0
    width = max(run["drnl_abs"].shape[1] for run in runs)
    for run in runs:
        missing = width - run["drnl_abs"].shape[1]
        if not missing:
            continue
        run["drnl_abs"] = np.pad(run["drnl_abs"], ((0, 0), (0, missing)))
        run["drnl_signed"] = np.pad(run["drnl_signed"], ((0, 0), (0, missing)))
        for role in ROLE_NAMES.values():
            run["drnl_role_abs"][role] = np.pad(
                run["drnl_role_abs"][role], ((0, 0), (0, missing))
            )
            run["drnl_role_signed"][role] = np.pad(
                run["drnl_role_signed"][role], ((0, 0), (0, missing))
            )
    return width


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    if not rows: raise ValueError("cannot write empty {}".format(path))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=918273)
    parser.add_argument("--beeswarm-per-web-role-feature", type=int, default=5)
    args = parser.parse_args(argv)
    manifest = [row for row in read_manifest(args.manifest) if row["ModelVariant"] in ATTRIBUTION_VARIANTS]
    node_sample, edge_sample, drnl_sample = [
        DeterministicReservoir(args.beeswarm_per_web_role_feature) for _ in range(3)]
    rng = np.random.default_rng(args.bootstrap_seed)
    node_rows, node_family_rows = [], []
    edge_rows, edge_family_rows, drnl_rows = [], [], []
    loaded_count, observed_foodwebs, observed_variants = 0, set(), set()
    for variant in ATTRIBUTION_VARIANTS:
        selected = []
        for row in manifest:
            if row["ModelVariant"] != variant:
                continue
            selected.append(load_run(
                row, args.result_root, node_sample, edge_sample, drnl_sample
            ))
            loaded_count += 1
            observed_foodwebs.add(row["Foodweb"])
            observed_variants.add(variant)
            if loaded_count % 250 == 0:
                print("[PROGRESS] loaded {}/{}".format(loaded_count, len(manifest)))
        if not selected: continue
        align_drnl_widths(selected)
        node_names = selected[0]["node_names"]; node_families = [node_family(name) for name in node_names]
        node_rows += cohort_rows(selected, "node_abs", "node_signed", node_names, node_families,
                                 "cohort", "all", args.bootstrap_replicates, rng)
        node_family_rows += family_cohort_rows(
            selected, "node_abs", "node_signed", node_families,
            NODE_FAMILY_ORDER, "cohort", "all", args.bootstrap_replicates, rng,
        )
        for role in ROLE_NAMES.values():
            copied = [dict(run, role_abs=run["node_role_abs"][role],
                           role_signed=run["node_role_signed"][role]) for run in selected]
            node_rows += cohort_rows(copied, "role_abs", "role_signed", node_names, node_families,
                                     "node_role", role, args.bootstrap_replicates, rng)
            node_family_rows += family_cohort_rows(
                copied, "role_abs", "role_signed", node_families,
                NODE_FAMILY_ORDER, "node_role", role,
                args.bootstrap_replicates, rng,
            )
        for code, name in CONFUSION_NAMES.items():
            node_rows += cohort_rows(selected, "node_abs", "node_signed", node_names, node_families,
                                     "confusion", name, args.bootstrap_replicates, rng,
                                     lambda run, code=code: run["confusion"] == code)
            node_family_rows += family_cohort_rows(
                selected, "node_abs", "node_signed", node_families,
                NODE_FAMILY_ORDER, "confusion", name,
                args.bootstrap_replicates, rng,
                selector=lambda run, code=code: run["confusion"] == code,
            )
        drnl_names = ["DRNL_{}".format(i) for i in range(selected[0]["drnl_abs"].shape[1])]
        drnl_rows += cohort_rows(selected, "drnl_abs", "drnl_signed", drnl_names,
                                 ["DRNL"] * len(drnl_names), "cohort", "all",
                                 args.bootstrap_replicates, rng)
        for role in ROLE_NAMES.values():
            copied = [dict(
                run,
                role_abs=run["drnl_role_abs"][role],
                role_signed=run["drnl_role_signed"][role],
            ) for run in selected]
            drnl_rows += cohort_rows(
                copied, "role_abs", "role_signed", drnl_names,
                ["DRNL"] * len(drnl_names), "node_role", role,
                args.bootstrap_replicates, rng,
            )
        for code, name in CONFUSION_NAMES.items():
            drnl_rows += cohort_rows(
                selected, "drnl_abs", "drnl_signed", drnl_names,
                ["DRNL"] * len(drnl_names), "confusion", name,
                args.bootstrap_replicates, rng,
                selector=lambda run, code=code: run["confusion"] == code,
            )
        if variant == "directed_ecc":
            edge_names = selected[0]["edge_names"]; families = [edge_family(i) for i in range(len(edge_names))]
            edge_rows += cohort_rows(selected, "edge_abs", "edge_signed", edge_names, families,
                                     "cohort", "all", args.bootstrap_replicates, rng)
            edge_family_rows += family_cohort_rows(
                selected, "edge_abs", "edge_signed", families,
                EDGE_FAMILY_ORDER, "cohort", "all", args.bootstrap_replicates, rng,
            )
            for code, name in CONFUSION_NAMES.items():
                edge_rows += cohort_rows(
                    selected, "edge_abs", "edge_signed", edge_names, families,
                    "confusion", name, args.bootstrap_replicates, rng,
                    selector=lambda run, code=code: run["confusion"] == code,
                )
                edge_family_rows += family_cohort_rows(
                    selected, "edge_abs", "edge_signed", families,
                    EDGE_FAMILY_ORDER, "confusion", name,
                    args.bootstrap_replicates, rng,
                    selector=lambda run, code=code: run["confusion"] == code,
                )
            for role in sorted(selected[0]["edge_role_abs"]):
                copied = [dict(run, role_abs=run["edge_role_abs"][role],
                               role_signed=run["edge_role_signed"][role]) for run in selected]
                edge_rows += cohort_rows(copied, "role_abs", "role_signed", edge_names, families,
                                         "edge_role_pair", role, args.bootstrap_replicates, rng)
                edge_family_rows += family_cohort_rows(
                    copied, "role_abs", "role_signed", families,
                    EDGE_FAMILY_ORDER, "edge_role_pair", role,
                    args.bootstrap_replicates, rng,
                )
    write_csv(args.output_dir / "node_feature_summary.csv", node_rows)
    write_csv(args.output_dir / "node_family_summary.csv", node_family_rows)
    write_csv(args.output_dir / "edge_feature_summary.csv", edge_rows)
    write_csv(args.output_dir / "edge_family_summary.csv", edge_family_rows)
    write_csv(args.output_dir / "drnl_summary.csv", drnl_rows)
    write_csv(args.output_dir / "node_beeswarm_sample.csv", node_sample.rows())
    write_csv(args.output_dir / "edge_beeswarm_sample.csv", edge_sample.rows())
    write_csv(args.output_dir / "drnl_beeswarm_sample.csv", drnl_sample.rows())
    metadata = {"schema_version": "seal-directed-node-ecc-attribution-summary-v2",
                "runs": loaded_count, "foodwebs": len(observed_foodwebs),
                "variants": sorted(observed_variants),
                "aggregation": "candidate_to_experiment_to_foodweb_to_cohort_equal_foodweb_weight",
                "bootstrap_unit": "foodweb", "bootstrap_replicates": args.bootstrap_replicates,
                "beeswarm_sampling": "deterministic_equal_cap_per_model_feature_foodweb_role",
                "summary_tables": [
                    "node_feature_summary.csv", "node_family_summary.csv",
                    "edge_feature_summary.csv", "edge_family_summary.csv",
                    "drnl_summary.csv",
                ],
                "peak_candidate_matrix_scope": "one_model_variant",
                "tukey_filter": False, "outlier_removal": False, "channels_combined": False,
                "interpretation": "model dependence, not causal ecological importance"}
    path = args.output_dir / "SUMMARY_METADATA.json"
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
