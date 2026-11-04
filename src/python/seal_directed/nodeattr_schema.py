"""Frozen names and feature-schema helpers for the SEAL node-attribute families."""

from __future__ import annotations

import hashlib
import json

import numpy as np


MODEL_ID = "SEAL_directed_nodeattr_v1"
DISPLAY_NAME = "SEAL-D-NodeAttr"
LEGACY_FAMILY = "SEAL_directed"

NODEEDGE_MODEL_ID = "SEAL_directed_nodeedgeattr_v1"
NODEEDGE_DISPLAY_NAME = "SEAL-D-NodeEdgeAttr"
NODEEDGE_LEGACY_FAMILY = "SEAL_directed_weighted_v1"

WEIGHTED_VARIANT_DISPLAY_NAMES = {
    "legacy_weak_node": "SEAL-D-NodeAttr-WeakMP",
    "directed_node": "SEAL-D-NodeAttr-DirectedMP",
    "directed_scalar_gate": "SEAL-D-NodeEdgeAttr-Gate",
    "directed_ecc": "SEAL-D-NodeEdgeAttr-ECC",
}

EXPECTED_NODE_ATTRIBUTE_COUNT = 44
FEATURE_FAMILY_ORDER = (
    "mass",
    "mass_missing",
    "taxonomy",
    "metabolism",
    "movement",
    "life_stage",
)


def _matlab_string(value):
    current = value
    while isinstance(current, np.ndarray):
        if current.size == 0:
            return ""
        if current.dtype.kind in ("U", "S") and current.size > 1:
            return "".join(str(item) for item in current.reshape(-1)).strip()
        current = current.reshape(-1)[0]
    if isinstance(current, bytes):
        return current.decode("utf-8").strip()
    return str(current).strip()


def feature_names_from_mat(data):
    if "group_feature_names" not in data:
        raise ValueError("MAT input is missing group_feature_names")
    names = [_matlab_string(value) for value in np.asarray(data["group_feature_names"]).reshape(-1)]
    if any(not name for name in names):
        raise ValueError("group_feature_names contains an empty feature name")
    if len(names) != EXPECTED_NODE_ATTRIBUTE_COUNT:
        raise ValueError(
            "Expected {} node attributes, found {}".format(
                EXPECTED_NODE_ATTRIBUTE_COUNT, len(names)
            )
        )
    if len(names) != len(set(names)):
        raise ValueError("group_feature_names contains duplicates")
    return names


def feature_schema_hash(feature_names):
    payload = json.dumps(
        list(feature_names), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def feature_family(name):
    if name == "log10_mass_z":
        return "mass"
    if name == "mass_missing":
        return "mass_missing"
    if name.startswith("taxonomy_level="):
        return "taxonomy"
    if name.startswith("metabolic_type="):
        return "metabolism"
    if name.startswith("movement_type="):
        return "movement"
    if name == "lifestage_missing" or name.startswith("lifestage="):
        return "life_stage"
    raise ValueError("Unmapped node attribute: {}".format(name))


def validate_feature_schema(feature_names):
    names = list(feature_names)
    if len(names) != EXPECTED_NODE_ATTRIBUTE_COUNT:
        raise ValueError(
            "Expected {} node attributes, found {}".format(
                EXPECTED_NODE_ATTRIBUTE_COUNT, len(names)
            )
        )
    families = [feature_family(name) for name in names]
    absent = sorted(set(FEATURE_FAMILY_ORDER).difference(families))
    if absent:
        raise ValueError("Feature schema is missing families: {}".format(", ".join(absent)))
    return {
        "feature_names": names,
        "feature_families": families,
        "feature_schema_hash": feature_schema_hash(names),
    }
