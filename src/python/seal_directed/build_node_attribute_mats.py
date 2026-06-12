import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as ssp


RAW_DEFAULT = Path("data/raw/283_3_Dataset/283_2_FoodWebDataBase_2018_12_10.csv")
MAT_DEFAULT = Path("src/matlab/data/foodwebs_mat")
OUT_DEFAULT = Path("src/python/seal_directed/data/foodwebs_mat_seal_attrs")


CATEGORY_COLUMNS = [
    "taxonomy_level",
    "metabolic_type",
    "movement_type",
    "lifestage",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build SEAL_directed .mat files with node attribute matrix `group`."
    )
    parser.add_argument("--raw-csv", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--mat-folder", type=Path, default=MAT_DEFAULT)
    parser.add_argument("--output-folder", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-foodweb", action="append", default=[])
    parser.add_argument("--include-ecosystem", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "-999"}:
        return ""
    return text


def normalize_category(value):
    text = clean_text(value).lower()
    return text if text else "unknown"


def normalize_taxonomy(value):
    return clean_text(value)


def valid_positive(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(number) or number <= 0 or number == -999:
        return np.nan
    return number


def matlab_cell_to_string(value):
    current = value
    while isinstance(current, np.ndarray):
        if current.size == 0:
            return ""
        current = current.reshape(-1)[0]
    return clean_text(current)


def matlab_cellstr_vector(value):
    arr = np.asarray(value).reshape(-1)
    return [matlab_cell_to_string(item) for item in arr]


def foodweb_stem_to_raw_name(stem):
    suffix = "_tax_mass"
    if stem.endswith(suffix):
        return stem[: -len(suffix)].strip()
    return stem.strip()


def most_common_nonempty(values):
    values = [normalize_category(value) for value in values]
    values = [value for value in values if value != "unknown"]
    if not values:
        return "unknown"
    return Counter(values).most_common(1)[0][0]


def load_raw_long(raw_csv):
    usecols = [
        "foodweb.name",
        "ecosystem.type",
        "con.taxonomy",
        "con.taxonomy.level",
        "con.lifestage",
        "con.metabolic.type",
        "con.movement.type",
        "con.mass.mean.g.",
        "res.taxonomy",
        "res.taxonomy.level",
        "res.lifestage",
        "res.metabolic.type",
        "res.movement.type",
        "res.mass.mean.g.",
    ]
    raw = pd.read_csv(raw_csv, usecols=usecols, low_memory=False)
    endpoint_frames = []
    for prefix in ["con", "res"]:
        endpoint = pd.DataFrame(
            {
                "foodweb": raw["foodweb.name"].map(clean_text),
                "ecosystem_type": raw["ecosystem.type"].map(normalize_category),
                "taxonomy": raw[f"{prefix}.taxonomy"].map(normalize_taxonomy),
                "taxonomy_level": raw[f"{prefix}.taxonomy.level"].map(normalize_category),
                "lifestage": raw[f"{prefix}.lifestage"].map(normalize_category),
                "metabolic_type": raw[f"{prefix}.metabolic.type"].map(normalize_category),
                "movement_type": raw[f"{prefix}.movement.type"].map(normalize_category),
                "mass_mean_g": raw[f"{prefix}.mass.mean.g."].map(valid_positive),
            }
        )
        endpoint = endpoint[(endpoint["foodweb"] != "") & (endpoint["taxonomy"] != "")]
        endpoint_frames.append(endpoint)
    return pd.concat(endpoint_frames, ignore_index=True)


def build_category_vocab(raw_long, include_ecosystem):
    vocab = {}
    for column in CATEGORY_COLUMNS:
        values = sorted(set(raw_long[column].dropna().map(normalize_category)))
        if "unknown" not in values:
            values.append("unknown")
        vocab[column] = values
    if include_ecosystem:
        values = sorted(set(raw_long["ecosystem_type"].dropna().map(normalize_category)))
        if "unknown" not in values:
            values.append("unknown")
        vocab["ecosystem_type"] = values
    return vocab


def build_feature_names(vocab, include_ecosystem):
    names = ["log10_mass_z", "mass_missing"]
    for column in ["taxonomy_level", "metabolic_type", "movement_type"]:
        names.extend([f"{column}={value}" for value in vocab[column]])
    names.append("lifestage_missing")
    names.extend([f"lifestage={value}" for value in vocab["lifestage"]])
    if include_ecosystem:
        names.extend([f"ecosystem_type={value}" for value in vocab["ecosystem_type"]])
    return names


def aggregate_web_attributes(raw_long):
    web_lookup = {}
    grouped = raw_long.groupby(["foodweb", "taxonomy"], sort=False)
    for (foodweb, taxonomy), rows in grouped:
        masses = rows["mass_mean_g"].dropna().to_numpy(dtype=float)
        web_lookup.setdefault(foodweb, {})[taxonomy] = {
            "taxonomy_level": most_common_nonempty(rows["taxonomy_level"]),
            "lifestage": most_common_nonempty(rows["lifestage"]),
            "metabolic_type": most_common_nonempty(rows["metabolic_type"]),
            "movement_type": most_common_nonempty(rows["movement_type"]),
            "mass_mean_g": float(np.median(masses)) if masses.size else np.nan,
        }
    ecosystem_lookup = (
        raw_long.groupby("foodweb")["ecosystem_type"]
        .agg(most_common_nonempty)
        .to_dict()
    )
    return web_lookup, ecosystem_lookup


def compute_mass_scaler(mat_files, web_lookup):
    values = []
    for mat_file in mat_files:
        data = sio.loadmat(mat_file)
        taxonomy = matlab_cellstr_vector(data.get("taxonomy", []))
        mass = np.asarray(data.get("mass", []), dtype=float).reshape(-1)
        raw_name = foodweb_stem_to_raw_name(mat_file.stem)
        raw_attrs = web_lookup.get(raw_name, {})
        for idx, taxon in enumerate(taxonomy):
            value = mass[idx] if idx < mass.size else np.nan
            value = valid_positive(value)
            if not np.isfinite(value):
                value = raw_attrs.get(taxon, {}).get("mass_mean_g", np.nan)
            if np.isfinite(value) and value > 0:
                values.append(math.log10(value))
    if not values:
        raise ValueError("No positive masses found for node attribute scaling.")
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std == 0:
        std = 1.0
    median = float(np.median(values))
    return {"mean": mean, "std": std, "median": median}


def add_one_hot(features, names, vocab, column, value):
    value = normalize_category(value)
    if value not in vocab[column]:
        value = "unknown"
    offset = len(features)
    features.extend([0.0] * len(vocab[column]))
    features[offset + vocab[column].index(value)] = 1.0
    names.extend([f"{column}={category}" for category in vocab[column]])


def node_features_for_taxon(
    taxon,
    raw_attrs,
    mat_mass,
    mass_scaler,
    vocab,
    include_ecosystem,
    ecosystem_type,
):
    attrs = raw_attrs.get(taxon, {})
    mass_value = valid_positive(mat_mass)
    if not np.isfinite(mass_value):
        mass_value = attrs.get("mass_mean_g", np.nan)
    mass_missing = 0.0
    if not np.isfinite(mass_value):
        mass_value = 10 ** mass_scaler["median"]
        mass_missing = 1.0
    log_mass_z = (math.log10(mass_value) - mass_scaler["mean"]) / mass_scaler["std"]

    features = [float(log_mass_z), mass_missing]
    feature_names = ["log10_mass_z", "mass_missing"]

    for column in ["taxonomy_level", "metabolic_type", "movement_type"]:
        add_one_hot(features, feature_names, vocab, column, attrs.get(column, "unknown"))

    lifestage = normalize_category(attrs.get("lifestage", "unknown"))
    features.append(1.0 if lifestage == "unknown" else 0.0)
    feature_names.append("lifestage_missing")
    add_one_hot(features, feature_names, vocab, "lifestage", lifestage)

    if include_ecosystem:
        add_one_hot(features, feature_names, vocab, "ecosystem_type", ecosystem_type)

    return features, feature_names


def copy_mat_with_group(
    mat_file,
    output_folder,
    web_lookup,
    ecosystem_lookup,
    mass_scaler,
    vocab,
    feature_names,
    include_ecosystem,
    overwrite,
):
    out_file = output_folder / mat_file.name
    if out_file.exists() and not overwrite:
        return {"foodweb": mat_file.stem, "status": "skipped_exists"}

    data = sio.loadmat(mat_file)
    payload = {key: value for key, value in data.items() if not key.startswith("__")}

    taxonomy = matlab_cellstr_vector(payload.get("taxonomy", []))
    mass = np.asarray(payload.get("mass", []), dtype=float).reshape(-1)
    raw_name = foodweb_stem_to_raw_name(mat_file.stem)
    raw_attrs = web_lookup.get(raw_name, {})
    ecosystem_type = ecosystem_lookup.get(raw_name, "unknown")

    group_rows = []
    missing_attr_count = 0
    mass_missing_count = 0
    inferred_feature_names = None

    for idx, taxon in enumerate(taxonomy):
        mat_mass = mass[idx] if idx < mass.size else np.nan
        row, row_names = node_features_for_taxon(
            taxon,
            raw_attrs,
            mat_mass,
            mass_scaler,
            vocab,
            include_ecosystem,
            ecosystem_type,
        )
        if inferred_feature_names is None:
            inferred_feature_names = row_names
        if taxon not in raw_attrs:
            missing_attr_count += 1
        if row[1] == 1.0:
            mass_missing_count += 1
        group_rows.append(row)

    group = np.asarray(group_rows, dtype=np.float32)
    if inferred_feature_names != feature_names:
        raise ValueError(f"Feature names mismatch while processing {mat_file}")

    payload["group"] = ssp.csr_matrix(group)
    payload["group_feature_names"] = np.asarray(feature_names, dtype=object)
    payload["group_feature_source"] = np.asarray(
        ["GATEWAy raw CSV + existing MAT mass; node-aligned for SEAL"], dtype=object
    )
    payload["group_mass_scaler_json"] = np.asarray([json.dumps(mass_scaler)], dtype=object)

    output_folder.mkdir(parents=True, exist_ok=True)
    sio.savemat(out_file, payload)

    return {
        "foodweb": mat_file.stem,
        "status": "written",
        "nodes": len(taxonomy),
        "features": group.shape[1],
        "missing_raw_node_attrs": missing_attr_count,
        "mass_imputed": mass_missing_count,
    }


def select_mat_files(mat_folder, only_foodweb, limit):
    mat_files = sorted(mat_folder.glob("*.mat"))
    if only_foodweb:
        requested = set(only_foodweb)
        selected = []
        for mat_file in mat_files:
            if mat_file.stem in requested or mat_file.name in requested:
                selected.append(mat_file)
        missing = requested.difference({path.stem for path in selected}).difference(
            {path.name for path in selected}
        )
        if missing:
            raise ValueError(f"Requested food webs not found: {', '.join(sorted(missing))}")
        mat_files = selected
    if limit is not None:
        mat_files = mat_files[:limit]
    return mat_files


def write_feature_names(output_folder, feature_names):
    output_folder.mkdir(parents=True, exist_ok=True)
    path = output_folder / "seal_node_attribute_feature_names.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_index", "feature_name"])
        for idx, name in enumerate(feature_names):
            writer.writerow([idx, name])
    return path


def main():
    args = parse_args()
    raw_csv = args.raw_csv.resolve()
    mat_folder = args.mat_folder.resolve()
    output_folder = args.output_folder.resolve()

    if not raw_csv.is_file():
        raise FileNotFoundError(raw_csv)
    if not mat_folder.is_dir():
        raise FileNotFoundError(mat_folder)

    all_mat_files = sorted(mat_folder.glob("*.mat"))
    mat_files = select_mat_files(mat_folder, args.only_foodweb, args.limit)
    if not mat_files:
        raise ValueError("No .mat files selected.")

    raw_long = load_raw_long(raw_csv)
    vocab = build_category_vocab(raw_long, args.include_ecosystem)
    feature_names = build_feature_names(vocab, args.include_ecosystem)
    web_lookup, ecosystem_lookup = aggregate_web_attributes(raw_long)
    mass_scaler = compute_mass_scaler(all_mat_files, web_lookup)

    rows = []
    for mat_file in mat_files:
        rows.append(
            copy_mat_with_group(
                mat_file,
                output_folder,
                web_lookup,
                ecosystem_lookup,
                mass_scaler,
                vocab,
                feature_names,
                args.include_ecosystem,
                args.overwrite,
            )
        )

    feature_file = write_feature_names(output_folder, feature_names)
    summary_file = output_folder / "seal_node_attribute_build_summary.csv"
    pd.DataFrame(rows).to_csv(summary_file, index=False)

    print(f"Selected MAT files: {len(mat_files)}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Output folder: {output_folder}")
    print(f"Feature names: {feature_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
