"""Definition and deterministic construction of the ``full_v1`` edge schema."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict

import numpy as np


INTERACTION_TYPES = (
    "predacious",
    "herbivorous",
    "detritivorous",
    "fungivorous",
    "parasitoid",
    "parasitic",
    "bacterivorous",
    "unknown",
)
DIMENSIONALITY = ("2d", "3d", "unknown")
CLASSIFICATION = ("ibi", "nibi", "unknown")

FEATURE_NAMES = (
    "log10_consumer_resource_mass_ratio",
    "resource_mass_missing",
    "consumer_mass_missing",
    *("interaction_type={}".format(value) for value in INTERACTION_TYPES),
    *("interaction_dimensionality={}".format(value) for value in DIMENSIONALITY),
    *("interaction_classification={}".format(value) for value in CLASSIFICATION),
)

SCHEMA = {
    "schema": "full_v1",
    "semantics": "contextual_edge_covariates_not_interaction_strength",
    "orientation": "resource_to_consumer",
    "feature_names": list(FEATURE_NAMES),
    "mass_aggregation": "median_positive_mass_per_taxon_within_foodweb",
    "categorical_aggregation": "deterministic_multi_hot_across_duplicate_records",
    "missing_category_policy": "unknown_channel",
    "mass_ratio_missing_policy": "nan_in_mat_then_training_context_mean_imputation",
}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


SCHEMA_JSON = canonical_json(SCHEMA)
SCHEMA_SHA256 = hashlib.sha256(SCHEMA_JSON.encode("utf-8")).hexdigest()


def clean_text(value):
    if value is None:
        return ""
    try:
        if bool(np.isnan(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    return "" if text in {"", "nan", "none", "-999", "na", "n/a"} else text


def positive_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(number) or number <= 0 or number == -999:
        return np.nan
    return number


def median_positive(values):
    clean = np.asarray([positive_float(value) for value in values], dtype=float)
    clean = clean[np.isfinite(clean)]
    return float(np.median(clean)) if clean.size else np.nan


def _multi_hot(values, vocabulary):
    known = {clean_text(value) for value in values}
    known.discard("")
    vector = np.zeros(len(vocabulary), dtype=np.float32)
    recognized = [value for value in known if value in vocabulary[:-1]]
    if recognized:
        for value in recognized:
            vector[vocabulary.index(value)] = 1.0
    else:
        vector[-1] = 1.0
    return vector


def build_edge_vector(resource_mass, consumer_mass, rows):
    """Build one 17-channel raw vector from all records for an observed link."""
    resource_missing = not np.isfinite(resource_mass) or resource_mass <= 0
    consumer_missing = not np.isfinite(consumer_mass) or consumer_mass <= 0
    ratio = np.nan
    if not resource_missing and not consumer_missing:
        ratio = math.log10(float(consumer_mass) / float(resource_mass))

    rows = list(rows)
    types = [row.get("interaction.type", "") for row in rows]
    dims = [row.get("interaction.dimensionality", "") for row in rows]
    classes = [row.get("interaction.classification", "") for row in rows]
    return np.concatenate(
        [
            np.asarray(
                [ratio, float(resource_missing), float(consumer_missing)],
                dtype=np.float32,
            ),
            _multi_hot(types, INTERACTION_TYPES),
            _multi_hot(dims, DIMENSIONALITY),
            _multi_hot(classes, CLASSIFICATION),
        ]
    ).astype(np.float32, copy=False)


def aggregate_raw_web(rows):
    """Return deterministic taxon masses and duplicate-aware link record groups."""
    mass_values = defaultdict(list)
    links = defaultdict(list)
    for row in rows:
        resource = str(row.get("res.taxonomy", "")).strip()
        consumer = str(row.get("con.taxonomy", "")).strip()
        if not resource or not consumer:
            continue
        mass_values[resource].append(row.get("res.mass.mean.g."))
        mass_values[consumer].append(row.get("con.mass.mean.g."))
        links[(resource, consumer)].append(row)
    masses = {taxon: median_positive(values) for taxon, values in mass_values.items()}
    return masses, links


def preprocess_training_edge_attributes(edge_index, edge_attr_raw, training_adjacency):
    """Mask non-context edges and fit ratio scaling strictly on training context."""
    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(-1, 2)
    raw = np.asarray(edge_attr_raw, dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
    if edge_index.shape[0] != raw.shape[0]:
        raise ValueError("edge_index and edge_attr_raw row counts differ")

    context_mask = np.asarray(
        [bool(training_adjacency[int(source), int(target)]) for source, target in edge_index],
        dtype=bool,
    )
    context_index = edge_index[context_mask]
    context_attr = raw[context_mask].copy()
    ratios = context_attr[:, 0]
    finite = np.isfinite(ratios)
    if finite.any():
        mean = float(np.mean(ratios[finite]))
        std = float(np.std(ratios[finite]))
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
    else:
        mean, std = 0.0, 1.0
    context_attr[finite, 0] = (ratios[finite] - mean) / std
    context_attr[~finite, 0] = 0.0
    if not np.all(np.isfinite(context_attr)):
        raise ValueError("non-finite edge attributes remain after preprocessing")
    return context_index, context_attr, {
        "ratio_mean": mean,
        "ratio_std": std,
        "ratio_finite_training_edges": int(finite.sum()),
        "training_context_edges": int(context_mask.sum()),
        "masked_edges": int((~context_mask).sum()),
    }

