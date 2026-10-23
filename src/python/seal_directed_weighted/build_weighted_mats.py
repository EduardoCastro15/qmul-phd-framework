#!/usr/bin/env python3
"""Build node-attributed MAT files with aligned contextual edge covariates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as ssp

from edge_schema import (
    FEATURE_NAMES,
    SCHEMA_JSON,
    SCHEMA_SHA256,
    aggregate_raw_web,
    build_edge_vector,
)


RAW_DEFAULT = Path("data/raw/283_3_Dataset/283_2_FoodWebDataBase_2018_12_10.csv")
MAT_DEFAULT = Path("src/python/seal_directed/data/foodwebs_mat_seal_attrs")
OUT_DEFAULT = Path("src/python/seal_directed_weighted/data/foodwebs_mat_seal_weighted")
RAW_COLUMNS = (
    "foodweb.name",
    "res.taxonomy",
    "con.taxonomy",
    "res.mass.mean.g.",
    "con.mass.mean.g.",
    "interaction.type",
    "interaction.dimensionality",
    "interaction.classification",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matlab_string(value):
    current = value
    while isinstance(current, np.ndarray):
        if current.size == 0:
            return ""
        current = current.reshape(-1)[0]
    return str(current).strip()


def matlab_strings(value):
    return [matlab_string(item) for item in np.asarray(value, dtype=object).reshape(-1)]


def raw_foodweb_name(mat_stem):
    suffix = "_tax_mass"
    return mat_stem[: -len(suffix)].strip() if mat_stem.endswith(suffix) else mat_stem.strip()


def read_raw_by_web(path):
    grouped = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(RAW_COLUMNS).difference(reader.fieldnames or []))
        if missing:
            raise ValueError("Raw CSV missing columns: {}".format(", ".join(missing)))
        for raw_row in reader:
            web = str(raw_row.get("foodweb.name", "")).strip()
            if web:
                grouped.setdefault(web, []).append({key: raw_row.get(key, "") for key in RAW_COLUMNS})
    return grouped


def build_one(mat_path, raw_rows, output_path, raw_hash, overwrite=False):
    if output_path.exists() and not overwrite:
        return {"foodweb": mat_path.stem, "status": "skipped_exists"}
    data = sio.loadmat(mat_path, spmatrix=True)
    payload = {key: value for key, value in data.items() if not key.startswith("__")}
    if "net" not in payload or "group" not in payload or "taxonomy" not in payload:
        raise ValueError("{} lacks net, group, or taxonomy".format(mat_path))

    net = ssp.csr_matrix(payload["net"])
    net = net.copy()
    net.setdiag(0)
    net.eliminate_zeros()
    net.data = np.ones_like(net.data)
    row, col = net.nonzero()
    edge_index = np.column_stack([row, col]).astype(np.int64, copy=False)
    order = np.lexsort((edge_index[:, 1], edge_index[:, 0]))
    edge_index = edge_index[order]

    taxonomy = matlab_strings(payload["taxonomy"])
    if len(taxonomy) != net.shape[0]:
        raise ValueError("taxonomy/net size mismatch in {}".format(mat_path))
    masses, link_rows = aggregate_raw_web(raw_rows)
    edge_attr = []
    unmatched = 0
    for source, target in edge_index:
        resource = taxonomy[int(source)]
        consumer = taxonomy[int(target)]
        records = link_rows.get((resource, consumer), [])
        if not records:
            unmatched += 1
        edge_attr.append(
            build_edge_vector(masses.get(resource, np.nan), masses.get(consumer, np.nan), records)
        )
    edge_attr_raw = np.asarray(edge_attr, dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
    if edge_index.shape[0] != net.nnz or edge_attr_raw.shape != (net.nnz, 17):
        raise AssertionError("edge schema cardinality failure for {}".format(mat_path))
    if np.any(edge_index[:, 0] == edge_index[:, 1]):
        raise AssertionError("self-loop found in {}".format(mat_path))

    payload["net"] = net
    payload["edge_index"] = edge_index
    payload["edge_attr_raw"] = edge_attr_raw
    payload["edge_attr_feature_names"] = np.asarray(FEATURE_NAMES, dtype=object)
    payload["edge_attr_schema_json"] = np.asarray([SCHEMA_JSON], dtype=object)
    payload["edge_attr_schema_sha256"] = np.asarray([SCHEMA_SHA256], dtype=object)
    payload["edge_attr_source_sha256"] = np.asarray([raw_hash], dtype=object)
    payload["edge_attr_source"] = np.asarray(
        ["GATEWAy FoodWebDataBase 2018-12-10; contextual covariates, not link strength"],
        dtype=object,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(output_path, payload, do_compression=True)
    return {
        "foodweb": mat_path.stem,
        "status": "written",
        "nodes": int(net.shape[0]),
        "links": int(net.nnz),
        "features": int(edge_attr_raw.shape[1]),
        "unmatched_raw_links": int(unmatched),
        "finite_mass_ratios": int(np.isfinite(edge_attr_raw[:, 0]).sum()),
    }


def validate_weighted_mat(path):
    data = sio.loadmat(path, spmatrix=True)
    net = ssp.csr_matrix(data["net"])
    net.setdiag(0)
    net.eliminate_zeros()
    expected = np.column_stack(net.nonzero()).astype(np.int64)
    expected = expected[np.lexsort((expected[:, 1], expected[:, 0]))]
    observed = np.asarray(data["edge_index"], dtype=np.int64).reshape(-1, 2)
    attrs = np.asarray(data["edge_attr_raw"], dtype=np.float32)
    if not np.array_equal(expected, observed):
        raise ValueError("edge_index does not exactly match net in {}".format(path))
    if attrs.shape != (observed.shape[0], len(FEATURE_NAMES)):
        raise ValueError("edge_attr_raw shape mismatch in {}".format(path))
    if matlab_string(data.get("edge_attr_schema_sha256", "")) != SCHEMA_SHA256:
        raise ValueError("edge schema hash mismatch in {}".format(path))
    return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--mat-folder", type=Path, default=MAT_DEFAULT)
    parser.add_argument("--output-folder", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--only-foodweb", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    raw_path = args.raw_csv.resolve()
    mat_folder = args.mat_folder.resolve()
    output_folder = args.output_folder.resolve()
    if args.validate_only:
        paths = sorted(output_folder.glob("*.mat"))
        for path in paths:
            validate_weighted_mat(path)
        print("[OK] Validated {} weighted MAT files".format(len(paths)))
        return

    raw_by_web = read_raw_by_web(raw_path)
    raw_hash = file_sha256(raw_path)
    mat_paths = sorted(mat_folder.glob("*.mat"))
    if args.only_foodweb:
        requested = set(args.only_foodweb)
        mat_paths = [p for p in mat_paths if p.stem in requested or p.name in requested]
    if args.limit is not None:
        mat_paths = mat_paths[: args.limit]
    if not mat_paths:
        raise ValueError("No attributed MAT files selected")

    audit = []
    for mat_path in mat_paths:
        name = raw_foodweb_name(mat_path.stem)
        audit.append(
            build_one(
                mat_path,
                raw_by_web.get(name, []),
                output_folder / mat_path.name,
                raw_hash,
                overwrite=args.overwrite,
            )
        )
        validate_weighted_mat(output_folder / mat_path.name)
    audit_path = output_folder / "weighted_mat_build_audit.csv"
    fieldnames = sorted({key for row in audit for key in row})
    with open(audit_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit)
    metadata = {
        "schema": json.loads(SCHEMA_JSON),
        "schema_sha256": SCHEMA_SHA256,
        "source_csv": str(raw_path),
        "source_sha256": raw_hash,
        "mat_files": len(mat_paths),
    }
    with open(output_folder / "EDGE_SCHEMA.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("[OK] Built and validated {} weighted MAT files".format(len(mat_paths)))
    print("[OK] Audit: {}".format(audit_path))


if __name__ == "__main__":
    main()
