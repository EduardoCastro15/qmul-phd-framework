"""Creation and validation of frozen, variant-independent directed splits."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as ssp

from run_artifacts import canonical_json_bytes, file_sha256, logical_split_hash


SPLIT_SCHEMA = "seal-directed-weighted-split-v1"


def matlab_string(value):
    current = value
    while isinstance(current, np.ndarray):
        if current.size == 0:
            return ""
        current = current.reshape(-1)[0]
    return str(current).strip()


def role_codes(role, nodes):
    result = np.zeros(nodes, dtype=np.int8)
    for index, item in enumerate(np.asarray(role, dtype=object).reshape(-1)[:nodes]):
        text = matlab_string(item).lower()
        if text == "consumer":
            result[index] = 1
        elif text == "resource":
            result[index] = 2
        elif text in {"consumer-resource", "resource-consumer", "both"}:
            result[index] = 3
    return result


def _negative_pool(net, codes, role_filter):
    occupied = ssp.csr_matrix(net).toarray().astype(bool)
    candidates = ~occupied
    np.fill_diagonal(candidates, False)
    if role_filter and np.any(codes):
        source_ok = (codes == 2) | (codes == 3)
        target_ok = (codes == 1) | (codes == 3)
        candidates &= source_ok[:, None] & target_ok[None, :]
    return np.column_stack(np.where(candidates)).astype(np.int64)


def create_split(net, role, seed, test_ratio=0.1, max_train_num=100000, role_filter=True):
    matrix = ssp.csr_matrix(net).copy()
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    matrix.data = np.ones_like(matrix.data)
    row, col = matrix.nonzero()
    positives = np.column_stack([row, col]).astype(np.int64)
    python_rng = random.Random(int(seed))
    numpy_rng = np.random.RandomState(int(seed))
    order = python_rng.sample(range(len(positives)), len(positives))
    positives = positives[order]
    split_at = int(math.ceil(len(positives) * (1.0 - float(test_ratio))))
    if len(positives) > 1 and test_ratio > 0:
        split_at = min(max(split_at, 1), len(positives) - 1)
    train_pos = positives[:split_at]
    test_pos = positives[split_at:]
    if max_train_num is not None:
        permutation = numpy_rng.permutation(len(train_pos))[: int(max_train_num)]
        train_pos = train_pos[permutation]

    codes = role_codes(role, matrix.shape[0])
    constrained = _negative_pool(matrix, codes, role_filter=role_filter)
    required = len(train_pos) + len(test_pos)
    fallback = bool(role_filter and len(constrained) < required)
    selected_pool = _negative_pool(matrix, codes, role_filter=False) if fallback else constrained
    if len(selected_pool) < required:
        raise ValueError("not enough directed non-links: need {}, have {}".format(required, len(selected_pool)))
    neg_order = numpy_rng.permutation(len(selected_pool))[:required]
    negatives = selected_pool[neg_order]
    train_neg = negatives[: len(train_pos)]
    test_neg = negatives[len(train_pos):]
    split_digest = logical_split_hash(train_pos, train_neg, test_pos, test_neg)
    metadata = {
        "schema_version": SPLIT_SCHEMA,
        "seed": int(seed),
        "test_ratio": float(test_ratio),
        "negative_sampling": "uniform_without_replacement",
        "negative_positive_ratio": 1,
        "role_filter_requested": bool(role_filter),
        "role_constrained_pool_size": int(len(constrained)),
        "selected_pool_size": int(len(selected_pool)),
        "required_negative_count": int(required),
        "role_pool_fallback_used": fallback,
        "fallback_policy": "replace_with_all_directed_nonlinks",
        "logical_split_hash": split_digest,
    }
    return {
        "train_pos": train_pos,
        "train_neg": train_neg,
        "test_pos": test_pos,
        "test_neg": test_neg,
    }, metadata


def write_split_bundle(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(path)
    descriptor, temp_name = tempfile.mkstemp(prefix=".{}.".format(path.name), dir=path.parent)
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        with open(temp_path, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temp_path, path)
        metadata = dict(metadata, split_file_sha256=file_sha256(path))
        with open(path.with_suffix(".json"), "wb") as handle:
            handle.write(canonical_json_bytes(metadata) + b"\n")
    except Exception:
        temp_path.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)
        raise
    return metadata


def load_split_bundle(path, expected_file_hash=None, expected_logical_hash=None):
    path = Path(path)
    if expected_file_hash and file_sha256(path) != expected_file_hash:
        raise ValueError("split file checksum mismatch: {}".format(path))
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key], dtype=np.int64).reshape(-1, 2) for key in loaded.files}
    required = {"train_pos", "train_neg", "test_pos", "test_neg"}
    if set(arrays) != required:
        raise ValueError("split arrays mismatch: {}".format(path))
    digest = logical_split_hash(**arrays)
    if expected_logical_hash and digest != expected_logical_hash:
        raise ValueError("logical split hash mismatch: {}".format(path))
    return arrays, digest


def create_split_from_mat(mat_path, seed, output_path, **kwargs):
    data = sio.loadmat(mat_path, spmatrix=True)
    arrays, metadata = create_split(data["net"], data.get("role", []), seed, **kwargs)
    return write_split_bundle(output_path, arrays, metadata)

