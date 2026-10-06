#!/usr/bin/env python3
"""Create immutable, validation-backed WLNM inputs from Yujie's release.

The source release is read-only.  This program writes two independent label
regimes under ``src/matlab/data/yujie_wlnm_preprocessed_v1``:

* ``original_observed`` uses ``edge_status_original_grouped``.
* ``final_filled_sensitivity`` uses the final/imputed ``edge_status``.

In both regimes, explicit zeroes are the *only* negative candidates.  Missing
states remain unresolved and never enter ``observed_negative_mask``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import savemat


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO_ROOT / (
    "data/DAPSTOM – An Integrated Database & Portal for Fish Stomach Records/"
    "Yujie Data/grouped_filled_data_foodwebs_by_fold_group_year_2026-10-04"
)
DEFAULT_OUTPUT = REPO_ROOT / "src/matlab/data/yujie_wlnm_preprocessed_v1"
EXPECTED_NETWORKS = 855
EXPECTED_PAIRS = 831
EXPECTED_HASHES = 1768


@dataclass(frozen=True)
class Regime:
    name: str
    label_column: str
    description: str


REGIMES = (
    Regime(
        "original_observed",
        "edge_status_original_grouped",
        "Original grouped observations only; no imputed states.",
    ),
    Regime(
        "final_filled_sensitivity",
        "edge_status",
        "Final filled states; exploratory imputation sensitivity.",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_release_hashes(source: Path) -> list[dict[str, str]]:
    checksum_path = source / "SHA256SUMS.txt"
    entries: list[dict[str, str]] = []
    for line_number, raw in enumerate(checksum_path.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            expected, relative = raw.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Invalid checksum line {line_number}: {raw!r}") from exc
        relative = relative.lstrip("*")
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(f"Checksum target does not exist: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {relative}: expected {expected}, got {actual}"
            )
        entries.append({"path": relative, "sha256": actual})
    if len(entries) != EXPECTED_HASHES:
        raise ValueError(
            f"Expected {EXPECTED_HASHES} release hashes, found {len(entries)}"
        )
    return entries


def derive_roles(net: sparse.spmatrix) -> np.ndarray:
    """Return the exact GATEWAy/WLNM role rule for A(prey,predator)=1."""
    in_degree = np.asarray(net.sum(axis=0)).reshape(-1)
    out_degree = np.asarray(net.sum(axis=1)).reshape(-1)
    role = np.full(net.shape[0], "isolate", dtype=object)
    role[(in_degree == 0) & (out_degree > 0)] = "resource"
    role[(in_degree > 0) & (out_degree == 0)] = "consumer"
    role[(in_degree > 0) & (out_degree > 0)] = "consumer-resource"
    return role


def matlab_cellstr(values: Iterable[str], *, column: bool = True) -> np.ndarray:
    values = list(values)
    shape = (len(values), 1) if column else (1, len(values))
    return np.asarray(values, dtype=object).reshape(shape)


def scalar_text(value: object) -> np.ndarray:
    return np.asarray([[str(value)]], dtype=object)


def network_id(fold: int, group_id: str, year: int) -> str:
    return f"F{int(fold):02d}_{group_id}_{int(year):04d}"


def validate_evidence_table(
    frame: pd.DataFrame, fold: int, group_id: str, year: int, path: Path
) -> None:
    required = {
        "fold",
        "group_id",
        "year",
        "predator",
        "prey",
        "edge_status",
        "edge_status_original_grouped",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if len(frame) != EXPECTED_PAIRS:
        raise ValueError(f"{path}: expected {EXPECTED_PAIRS} rows, got {len(frame)}")
    if frame[["prey", "predator"]].duplicated().any():
        raise ValueError(f"{path}: prey->predator candidates are not unique")
    if (frame["prey"] == frame["predator"]).any():
        raise ValueError(f"{path}: self-loops are not allowed")
    for column, expected in (("fold", fold), ("group_id", group_id), ("year", year)):
        values = frame[column].drop_duplicates().tolist()
        if values != [expected]:
            raise ValueError(f"{path}: {column}={values!r}, expected {expected!r}")
    for column in ("edge_status", "edge_status_original_grouped"):
        states = set(frame[column].dropna().astype(int).tolist())
        if not states.issubset({0, 1}):
            raise ValueError(f"{path}: {column} contains states outside 0/1/NA")


def exclusive_status(n_pos: int, n_zero: int, n_resolved: int) -> str:
    if n_resolved == 0:
        return "no_evidence"
    if n_pos == 0:
        return "no_positives"
    if n_zero == 0:
        return "no_observed_zeros"
    if n_zero < 2 * n_pos:
        return "insufficient_2to1_pool"
    if n_pos < 2:
        return "insufficient_for_nonempty_train_test"
    if n_pos < 10:
        return "eligible_60_only_lt10_positives"
    return "eligible_60_and_sweep"


def build_network(
    frame: pd.DataFrame,
    label_column: str,
    nodes_order: list[str],
    node_roles: dict[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    labels = frame[label_column]
    resolved = labels.notna()
    endpoints = set(frame.loc[resolved, "prey"]) | set(frame.loc[resolved, "predator"])
    taxonomy = [node for node in nodes_order if node in endpoints]
    extra = endpoints.difference(taxonomy)
    if extra:
        raise ValueError(f"Evidence includes nodes absent from nodes.csv: {sorted(extra)}")
    index = {name: position for position, name in enumerate(taxonomy)}
    n = len(taxonomy)

    candidate_rows: list[int] = []
    candidate_cols: list[int] = []
    positive_rows: list[int] = []
    positive_cols: list[int] = []
    zero_rows: list[int] = []
    zero_cols: list[int] = []
    unresolved_rows: list[int] = []
    unresolved_cols: list[int] = []

    for item in frame.itertuples(index=False):
        if item.prey not in index or item.predator not in index:
            continue
        row = index[item.prey]
        col = index[item.predator]
        candidate_rows.append(row)
        candidate_cols.append(col)
        state = getattr(item, label_column)
        if pd.isna(state):
            unresolved_rows.append(row)
            unresolved_cols.append(col)
        elif int(state) == 1:
            positive_rows.append(row)
            positive_cols.append(col)
        else:
            zero_rows.append(row)
            zero_cols.append(col)

    def binary_matrix(rows: list[int], cols: list[int]) -> sparse.csc_matrix:
        values = np.ones(len(rows), dtype=np.uint8)
        return sparse.csc_matrix((values, (rows, cols)), shape=(n, n))

    net = binary_matrix(positive_rows, positive_cols)
    observed_negative_mask = binary_matrix(zero_rows, zero_cols)
    candidate_mask = binary_matrix(candidate_rows, candidate_cols)
    unresolved_mask = binary_matrix(unresolved_rows, unresolved_cols)

    if net.multiply(observed_negative_mask).nnz:
        raise AssertionError("Positive and observed-zero masks overlap")
    if net.multiply(unresolved_mask).nnz or observed_negative_mask.multiply(unresolved_mask).nnz:
        raise AssertionError("Resolved and unresolved masks overlap")
    if (net + observed_negative_mask + unresolved_mask != candidate_mask).nnz:
        raise AssertionError("Candidate mask is not the disjoint union of 1/0/NA")

    role = derive_roles(net)
    yujie_role = np.asarray([node_roles[name] for name in taxonomy], dtype=object)
    n_pos = int(net.nnz)
    n_zero = int(observed_negative_mask.nnz)
    n_na = int(unresolved_mask.nnz)
    n_candidate = int(candidate_mask.nnz)
    n_resolved = n_pos + n_zero
    # DivideNet uses ceil((1-ratio)*L) for TEST.  At 60%, L=1 therefore
    # leaves an empty positive TRAIN graph and cannot be a valid WLNM run.
    eligible_60 = n_pos >= 2 and n_zero >= 2 * n_pos
    eligible_sweep = eligible_60 and n_pos >= 10

    mat = {
        "net": net.astype(np.float64),
        "observed_negative_mask": observed_negative_mask.astype(np.float64),
        "candidate_mask": candidate_mask.astype(np.float64),
        "unresolved_mask": unresolved_mask.astype(np.float64),
        "taxonomy": matlab_cellstr(taxonomy),
        "role": matlab_cellstr(role.tolist()),
        "yujie_network_role": matlab_cellstr(yujie_role.tolist()),
        "mass": np.full((1, n), np.nan, dtype=np.float64),
    }
    manifest = {
        "n_nodes": n,
        "n_positive": n_pos,
        "n_observed_zero": n_zero,
        "n_unresolved": n_na,
        "n_candidate": n_candidate,
        "n_resolved": n_resolved,
        "n_resource": int(np.count_nonzero(role == "resource")),
        "n_consumer": int(np.count_nonzero(role == "consumer")),
        "n_consumer_resource": int(np.count_nonzero(role == "consumer-resource")),
        "n_isolate": int(np.count_nonzero(role == "isolate")),
        "negative_required_2to1": 2 * n_pos,
        "negative_shortfall_2to1": max(0, 2 * n_pos - n_zero),
        "eligible_train60": eligible_60,
        "eligible_sweep_10_90": eligible_sweep,
        "exclusion_or_eligibility": exclusive_status(n_pos, n_zero, n_resolved),
    }
    return mat, manifest


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_smoke_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    eligible = sorted(
        (row for row in rows if row["eligible_train60"]),
        key=lambda row: (int(row["n_positive"]), str(row["network_id"])),
    )
    if len(eligible) < 3:
        raise ValueError("At least three eligible networks are required for smoke selection")
    positions = (0, len(eligible) // 2, len(eligible) - 1)
    output: list[dict[str, object]] = []
    for size_class, position in zip(("small", "medium", "large"), positions):
        row = dict(eligible[position])
        row["smoke_size_class"] = size_class
        output.append(row)
    return output


def coverage_summary(
    rows: list[dict[str, object]], group_fields: tuple[str, ...]
) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    statuses = sorted(frame["exclusion_or_eligibility"].unique())
    output: list[dict[str, object]] = []
    grouper: object = list(group_fields) if len(group_fields) > 1 else group_fields[0]
    for key, group in frame.groupby(grouper, sort=True, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        record: dict[str, object] = dict(zip(group_fields, values))
        record.update(
            {
                "n_networks": int(len(group)),
                "n_with_evidence": int((group["n_resolved"] > 0).sum()),
                "n_with_positives": int((group["n_positive"] > 0).sum()),
                "n_eligible_train60": int(group["eligible_train60"].sum()),
                "n_excluded_train60": int((~group["eligible_train60"]).sum()),
                "n_eligible_sweep_10_90": int(group["eligible_sweep_10_90"].sum()),
            }
        )
        counts = group["exclusion_or_eligibility"].value_counts()
        for status in statuses:
            record[f"status_{status}"] = int(counts.get(status, 0))
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing derived output directory; never touches the source.",
    )
    parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help="For tests only. Production preprocessing must verify the release hashes.",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError(
            "Output must be separate from the source release and must not be its ancestor"
        )
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}; use --overwrite")
        shutil.rmtree(output)

    output.mkdir(parents=True)
    (output / "manifests").mkdir()
    (output / "validation").mkdir()
    for regime in REGIMES:
        (output / regime.name).mkdir()

    hash_rows: list[dict[str, str]] = []
    if not args.skip_hash_verification:
        hash_rows = verify_release_hashes(source)
        write_csv(output / "validation/source_hashes_verified.csv", hash_rows)

    release = pd.read_csv(source / "release_manifest.csv")
    if len(release) != EXPECTED_NETWORKS:
        raise ValueError(f"Expected {EXPECTED_NETWORKS} release rows, found {len(release)}")
    key_columns = ["fold", "group_id", "year"]
    if release[key_columns].duplicated().any():
        raise ValueError("release_manifest.csv has duplicate fold/group/year keys")

    nodes = pd.read_csv(source / "nodes.csv", keep_default_na=False)
    if nodes["node"].duplicated().any():
        raise ValueError("nodes.csv contains duplicate node names")
    allowed_source_roles = {"prey_only", "predator_only"}
    if not set(nodes["network_role"]).issubset(allowed_source_roles):
        raise ValueError("nodes.csv contains unexpected network_role values")
    nodes_order = nodes["node"].tolist()
    node_roles = dict(zip(nodes["node"], nodes["network_role"]))

    manifests: dict[str, list[dict[str, object]]] = {r.name: [] for r in REGIMES}
    derived_hashes: list[dict[str, object]] = []

    for release_row in release.sort_values(key_columns).itertuples(index=False):
        fold = int(release_row.fold)
        group_id = str(release_row.group_id)
        year = int(release_row.year)
        identifier = network_id(fold, group_id, year)
        evidence_relative = Path(str(release_row.evidence_path))
        evidence_path = source / evidence_relative
        evidence_hash = sha256_file(evidence_path)
        if evidence_hash != str(release_row.evidence_sha256):
            raise ValueError(f"Release manifest hash mismatch: {evidence_relative}")
        frame = pd.read_csv(evidence_path)
        validate_evidence_table(frame, fold, group_id, year, evidence_path)

        source_counts: dict[str, int] = {}
        for prefix, column in (
            ("original", "edge_status_original_grouped"),
            ("final", "edge_status"),
        ):
            source_counts[f"n_{prefix}_1"] = int((frame[column] == 1).sum())
            source_counts[f"n_{prefix}_0"] = int((frame[column] == 0).sum())
            source_counts[f"n_{prefix}_na"] = int(frame[column].isna().sum())
            if sum(source_counts[f"n_{prefix}_{state}"] for state in ("1", "0", "na")) != EXPECTED_PAIRS:
                raise AssertionError(f"{evidence_path}: invalid {prefix} 1/0/NA partition")

        unknown_nodes = (set(frame["prey"]) | set(frame["predator"])) - set(nodes_order)
        if unknown_nodes:
            raise ValueError(f"{evidence_path}: unknown nodes {sorted(unknown_nodes)}")

        for regime in REGIMES:
            mat, counts = build_network(frame, regime.label_column, nodes_order, node_roles)
            mat.update(
                {
                    "network_id": scalar_text(identifier),
                    "data_regime": scalar_text(regime.name),
                    "label_column": scalar_text(regime.label_column),
                    "spatial_fold": np.asarray([[fold]], dtype=np.float64),
                    "group_id": scalar_text(group_id),
                    "year": np.asarray([[year]], dtype=np.float64),
                    "source_evidence_path": scalar_text(evidence_relative.as_posix()),
                    "source_evidence_sha256": scalar_text(evidence_hash),
                    "input_positive_count": np.asarray([[counts["n_positive"]]], dtype=np.float64),
                    "input_observed_zero_count": np.asarray(
                        [[counts["n_observed_zero"]]], dtype=np.float64
                    ),
                    "input_unresolved_count": np.asarray(
                        [[counts["n_unresolved"]]], dtype=np.float64
                    ),
                    "input_candidate_count": np.asarray(
                        [[counts["n_candidate"]]], dtype=np.float64
                    ),
                    "original_status_1_count": np.asarray(
                        [[source_counts["n_original_1"]]], dtype=np.float64
                    ),
                    "original_status_0_count": np.asarray(
                        [[source_counts["n_original_0"]]], dtype=np.float64
                    ),
                    "original_status_na_count": np.asarray(
                        [[source_counts["n_original_na"]]], dtype=np.float64
                    ),
                    "final_status_1_count": np.asarray(
                        [[source_counts["n_final_1"]]], dtype=np.float64
                    ),
                    "final_status_0_count": np.asarray(
                        [[source_counts["n_final_0"]]], dtype=np.float64
                    ),
                    "final_status_na_count": np.asarray(
                        [[source_counts["n_final_na"]]], dtype=np.float64
                    ),
                    "negative_eligibility_mode": scalar_text("observed_zero"),
                }
            )
            mat_path = output / regime.name / f"{identifier}.mat"
            savemat(mat_path, mat, do_compression=True, oned_as="column")
            mat_hash = sha256_file(mat_path)
            row: dict[str, object] = {
                "Foodweb": identifier,
                "network_id": identifier,
                "fold": fold,
                "group_id": group_id,
                "year": year,
                "data_regime": regime.name,
                "label_column": regime.label_column,
                **source_counts,
                **counts,
                "source_evidence_path": evidence_relative.as_posix(),
                "source_evidence_sha256": evidence_hash,
                "mat_path": f"{regime.name}/{identifier}.mat",
                "mat_sha256": mat_hash,
            }
            manifests[regime.name].append(row)
            derived_hashes.append(
                {"data_regime": regime.name, "network_id": identifier, "sha256": mat_hash}
            )

    summary: dict[str, object] = {
        "source": str(source),
        "output": str(output),
        "source_hash_verification_enabled": not args.skip_hash_verification,
        "source_hash_count": len(hash_rows),
        "expected_networks": EXPECTED_NETWORKS,
        "expected_pairs_per_network": EXPECTED_PAIRS,
        "orientation": "A(prey,predator)=1; source=prey; target=predator",
        "role_rule": {
            "resource": "in_degree=0 and out_degree>0",
            "consumer": "in_degree>0 and out_degree=0",
            "consumer-resource": "in_degree>0 and out_degree>0",
            "isolate": "in_degree=0 and out_degree=0",
        },
        "negative_mode": "observed_zero",
        "negative_positive_ratio": 2,
        "regimes": {},
    }
    for regime in REGIMES:
        rows = manifests[regime.name]
        manifest_path = output / "manifests" / f"{regime.name}_manifest.csv"
        write_csv(manifest_path, rows)
        train60 = [row for row in rows if row["eligible_train60"]]
        sweep = [row for row in rows if row["eligible_sweep_10_90"]]
        write_csv(output / "manifests" / f"{regime.name}_train60_eligible.csv", train60)
        write_csv(output / "manifests" / f"{regime.name}_sweep_eligible.csv", sweep)
        if regime.name == "original_observed":
            write_csv(output / "manifests/original_observed_smoke.csv", select_smoke_rows(rows))
        for group_fields, suffix in (
            (("fold",), "fold"),
            (("year",), "year"),
            (("fold", "year"), "fold_year"),
        ):
            write_csv(
                output / "validation" / f"{regime.name}_coverage_by_{suffix}.csv",
                coverage_summary(rows, group_fields),
            )
        status_counts = pd.Series([row["exclusion_or_eligibility"] for row in rows]).value_counts()
        summary["regimes"][regime.name] = {
            "description": regime.description,
            "network_count": len(rows),
            "positive_network_count": sum(int(row["n_positive"]) > 0 for row in rows),
            "eligible_train60_count": len(train60),
            "eligible_sweep_count": len(sweep),
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        }

    write_csv(output / "validation/derived_mat_hashes.csv", derived_hashes)
    (output / "validation/preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output / "README.md").write_text(
        "# Yujie inputs for WLNM_dir_neg\n\n"
        "This directory is generated by `src/matlab/software/preprocess_yujie_wlnm.py`. "
        "The original Yujie release is read-only and is never modified.\n\n"
        "- `original_observed/`: labels from `edge_status_original_grouped`.\n"
        "- `final_filled_sensitivity/`: labels from final/imputed `edge_status`.\n"
        "- `manifests/`: all networks plus exact 60%, sweep, and smoke selections.\n"
        "- `validation/`: verified source hashes, derived MAT hashes, and counts.\n\n"
        "Each MAT contains `net`, `observed_negative_mask`, `candidate_mask`, "
        "`unresolved_mask`, canonical WLNM `role`, original `yujie_network_role`, "
        "`taxonomy`, all-NaN `mass`, and provenance metadata. The orientation is "
        "`A(prey,predator)=1`. Provenance includes both original and final "
        "`1/0/NA` counts. Missing states are never negative examples.\n\n"
        "Regenerate from the repository root with:\n\n"
        "```bash\n"
        "/Users/acw792/miniconda3/envs/Foodweb/bin/python "
        "src/matlab/software/preprocess_yujie_wlnm.py --overwrite\n"
        "```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["regimes"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
