#!/usr/bin/env python3
"""
Extract web-level predictive metric and confusion-count tables for
WLNM_original vs WLNM_dir_neg.

This script intentionally uses only the Python standard library so it can run
outside the notebook environment.
"""

import csv
import math
import os
from collections import defaultdict
from pathlib import Path


TRAIN_RATIO_TARGET = 90
NODES_THRESHOLD = 0
K_TARGET = None

RESULTS_DIR_ORIGINAL = (
    "src/matlab/data/"
    "result_wlnm_original_sweep_train_ratios_10-90_pseudo_properties_Apocrita/"
    "prediction_scores_logs"
)

RESULTS_DIR_DIR_NEG = (
    "src/matlab/data/"
    "result_wlnm_dir_neg_sweep_train_ratios_10-90_pseudo_properties_stats_Apocrita/"
    "prediction_scores_logs"
)

FOODWEB_CSV = "src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv"

OUT_DIR = (
    "src/matlab/data/"
    "result_wlnm_dir_neg_sweep_train_ratios_10-90_pseudo_properties_stats_Apocrita/"
    "statistical_tests/athen"
)

GLOB_PATTERN_ORIGINAL = "*_tax_mass_results_random_wlnm_original.csv"
GLOB_PATTERN_DIR_NEG = "*_tax_mass_results_random_wlnm_dir_neg.csv"

VERSIONS = [
    ("original", "Original", "Original", RESULTS_DIR_ORIGINAL, GLOB_PATTERN_ORIGINAL),
    ("dir_neg", "Directed-negative", "DirNeg", RESULTS_DIR_DIR_NEG, GLOB_PATTERN_DIR_NEG),
]

METRIC_COLS = [
    "ROC_AUC",
    "PR_AUC",
    "F1Score",
    "TestMCC",
    "Precision",
    "Recall",
    "TestTSS",
]

# Test-set confusion matrix counts. If you need pseudo-network counts instead,
# replace this list with: PseudoTP, PseudoFP, PseudoFN, PseudoTN.
CONFUSION_COLS = ["TestTP", "TestFP", "TestFN", "TestTN"]

VALUE_COLS = CONFUSION_COLS + METRIC_COLS
ID_COLS = [
    "Foodweb",
    "FW_KEY",
    "Version",
    "VersionLabel",
    "Run",
    "ValidRuns",
    "file",
]


def find_repo_root(start=Path.cwd()):
    start = Path(start).resolve()
    for p in [start, *start.parents]:
        if (p / "src/matlab").exists() and (p / "docs").exists():
            return p
    raise RuntimeError(f"Could not locate repo root from {start}")


def normalize_fw_label(name):
    name = str(name).strip()
    name, _ = os.path.splitext(name)
    name = name.replace(" ", "_")
    for token in [
        "_results_",
        "_result_",
        "_results",
        "_random",
        "_tax_mass",
        "_tax",
        "_mass",
    ]:
        name = name.replace(token, "_")
    return "_".join(part for part in name.split("_") if part)


def parse_foodweb_from_name(fname):
    base = Path(fname).name
    if "_results_" in base:
        return base.split("_results_")[0]
    return Path(fname).stem


def to_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def is_finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def mean(values):
    vals = [v for v in values if is_finite(v)]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def median(values):
    vals = sorted(v for v in values if is_finite(v))
    n = len(vals)
    if n == 0:
        return math.nan
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def variance(values):
    vals = [v for v in values if is_finite(v)]
    n = len(vals)
    if n < 2:
        return math.nan
    mu = sum(vals) / n
    return sum((v - mu) ** 2 for v in vals) / (n - 1)


def std(values):
    var = variance(values)
    if not is_finite(var):
        return math.nan
    return math.sqrt(var)


def round_or_blank(value, digits=6):
    if not is_finite(value):
        return ""
    return round(value, digits)


def read_allowed_foodweb_keys(repo_root):
    path = repo_root / FOODWEB_CSV
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        field_map = {name.strip().lower(): name for name in reader.fieldnames or []}
        fw_col = field_map.get("foodweb")
        nodes_col = field_map.get("nodes")
        if fw_col is None or nodes_col is None:
            raise ValueError(f"Could not find Foodweb and Nodes columns in {path}")

        allowed = set()
        for row in reader:
            nodes = to_float(row.get(nodes_col))
            if is_finite(nodes) and nodes >= NODES_THRESHOLD:
                allowed.add(normalize_fw_label(row.get(fw_col, "")))
    return allowed


def keep_row(row):
    train_ratio = to_float(row.get("TrainRatio"))
    if not is_finite(train_ratio) or abs(train_ratio - TRAIN_RATIO_TARGET) > 1e-9:
        return False

    if K_TARGET is not None:
        k = to_float(row.get("K"))
        if not is_finite(k) or abs(k - K_TARGET) > 1e-9:
            return False

    return True


def load_one_csv(csv_path, version, version_label):
    foodweb = parse_foodweb_from_name(csv_path.name)
    fw_key = normalize_fw_label(foodweb)

    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in VALUE_COLS if c not in (reader.fieldnames or [])]
        if missing:
            print(f"[WARN] {csv_path.name} missing columns: {', '.join(missing)}")

        for raw in reader:
            if not keep_row(raw):
                continue

            out = {
                "Foodweb": foodweb,
                "FW_KEY": fw_key,
                "Version": version,
                "VersionLabel": version_label,
                "file": csv_path.name,
            }

            run = raw.get("ExperimentID") or raw.get("Iteration") or ""
            out["Run"] = int(to_float(run)) if is_finite(to_float(run)) else run

            valid_any = False
            for col in VALUE_COLS:
                val = to_float(raw.get(col))
                out[col] = val
                valid_any = valid_any or is_finite(val)

            if valid_any:
                rows.append(out)

    return rows


def load_version_rows(repo_root, version, version_label, folder, pattern):
    folder_path = repo_root / folder
    files = sorted(folder_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {folder_path} matching {pattern}")

    rows = []
    for csv_path in files:
        rows.extend(load_one_csv(csv_path, version, version_label))
    return rows


def aggregate_web_level(run_rows):
    grouped = defaultdict(list)
    for row in run_rows:
        key = (row["Foodweb"], row["FW_KEY"], row["Version"], row["VersionLabel"])
        grouped[key].append(row)

    out = []
    for (foodweb, fw_key, version, version_label), rows in sorted(grouped.items()):
        record = {
            "Foodweb": foodweb,
            "FW_KEY": fw_key,
            "Version": version,
            "VersionLabel": version_label,
            "ValidRuns": len(rows),
            "file": rows[0].get("file", ""),
        }
        for col in VALUE_COLS:
            record[col] = mean(row[col] for row in rows)
        out.append(record)
    return out


def build_summary(web_rows):
    rows = []
    by_version = defaultdict(list)
    for row in web_rows:
        by_version[row["Version"]].append(row)

    for version, version_label, _, _, _ in VERSIONS:
        version_rows = by_version.get(version, [])
        for col in VALUE_COLS:
            values = [row[col] for row in version_rows]
            finite_values = [v for v in values if is_finite(v)]
            rows.append(
                {
                    "Version": version,
                    "VersionLabel": version_label,
                    "Quantity": col,
                    "QuantityType": "Confusion count"
                    if col in CONFUSION_COLS
                    else "Predictive metric",
                    "N food webs": len(finite_values),
                    "Mean": mean(finite_values),
                    "Median": median(finite_values),
                    "SD": std(finite_values),
                    "Variance": variance(finite_values),
                    "SumOfWebMeans": sum(finite_values) if finite_values else math.nan,
                }
            )
    return rows


def build_wide_web_level(web_rows, common_only=False):
    grouped = defaultdict(dict)
    foodweb_names = {}
    for row in web_rows:
        key = row["FW_KEY"]
        foodweb_names[key] = row["Foodweb"]
        grouped[key][row["Version"]] = row

    out = []
    for fw_key in sorted(grouped):
        if common_only and any(version not in grouped[fw_key] for version, _, _, _, _ in VERSIONS):
            continue

        record = {
            "Foodweb": foodweb_names.get(fw_key, fw_key),
            "FW_KEY": fw_key,
        }
        for version, _, prefix, _, _ in VERSIONS:
            row = grouped[fw_key].get(version)
            if row is None:
                record[f"{prefix}_ValidRuns"] = ""
                for col in VALUE_COLS:
                    record[f"{prefix}_{col}"] = ""
            else:
                record[f"{prefix}_ValidRuns"] = row["ValidRuns"]
                for col in VALUE_COLS:
                    record[f"{prefix}_{col}"] = row[col]
        out.append(record)
    return out


def filter_common_web_rows(web_rows):
    versions_by_fw = defaultdict(set)
    for row in web_rows:
        versions_by_fw[row["FW_KEY"]].add(row["Version"])

    required = {version for version, _, _, _, _ in VERSIONS}
    common_keys = {
        fw_key for fw_key, versions in versions_by_fw.items() if required.issubset(versions)
    }
    return [row for row in web_rows if row["FW_KEY"] in common_keys]


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for col in fieldnames:
                val = row.get(col, "")
                clean[col] = round_or_blank(val) if isinstance(val, float) else val
            writer.writerow(clean)


def main():
    repo_root = find_repo_root()
    out_dir = repo_root / OUT_DIR

    allowed_fw_keys = read_allowed_foodweb_keys(repo_root)
    run_rows = []
    for version, version_label, _, folder, pattern in VERSIONS:
        version_rows = load_version_rows(repo_root, version, version_label, folder, pattern)
        run_rows.extend(row for row in version_rows if row["FW_KEY"] in allowed_fw_keys)
        print(f"[INFO] Loaded {len(version_rows):,} rows for {version_label}")

    if not run_rows:
        raise RuntimeError("No rows found after filtering.")

    web_rows = aggregate_web_level(run_rows)
    summary_rows = build_summary(web_rows)
    common_web_rows = filter_common_web_rows(web_rows)
    common_summary_rows = build_summary(common_web_rows)
    wide_rows = build_wide_web_level(web_rows)
    common_wide_rows = build_wide_web_level(web_rows, common_only=True)

    suffix = f"train{TRAIN_RATIO_TARGET}"
    run_path = out_dir / f"predictive_confusion_run_level_{suffix}.csv"
    web_path = out_dir / f"predictive_confusion_web_level_{suffix}.csv"
    wide_path = out_dir / f"predictive_confusion_web_level_wide_{suffix}.csv"
    common_wide_path = out_dir / f"predictive_confusion_web_level_wide_common_{suffix}.csv"
    summary_path = out_dir / f"predictive_confusion_summary_{suffix}.csv"
    common_summary_path = out_dir / f"predictive_confusion_summary_common_{suffix}.csv"

    run_fields = ["Foodweb", "FW_KEY", "Version", "VersionLabel", "Run", "file"] + VALUE_COLS
    web_fields = ["Foodweb", "FW_KEY", "Version", "VersionLabel", "ValidRuns", "file"] + VALUE_COLS
    wide_fields = ["Foodweb", "FW_KEY"]
    for _, _, prefix, _, _ in VERSIONS:
        wide_fields.append(f"{prefix}_ValidRuns")
        wide_fields.extend(f"{prefix}_{col}" for col in VALUE_COLS)
    summary_fields = [
        "Version",
        "VersionLabel",
        "Quantity",
        "QuantityType",
        "N food webs",
        "Mean",
        "Median",
        "SD",
        "Variance",
        "SumOfWebMeans",
    ]

    write_csv(run_path, run_rows, run_fields)
    write_csv(web_path, web_rows, web_fields)
    write_csv(wide_path, wide_rows, wide_fields)
    write_csv(common_wide_path, common_wide_rows, wide_fields)
    write_csv(summary_path, summary_rows, summary_fields)
    write_csv(common_summary_path, common_summary_rows, summary_fields)

    print(f"[INFO] Run-level rows: {len(run_rows):,}")
    print(f"[INFO] Web-level rows: {len(web_rows):,}")
    print(f"[INFO] Common web-level rows: {len(common_web_rows):,}")
    print(f"[INFO] Wrote: {run_path}")
    print(f"[INFO] Wrote: {web_path}")
    print(f"[INFO] Wrote: {wide_path}")
    print(f"[INFO] Wrote: {common_wide_path}")
    print(f"[INFO] Wrote: {summary_path}")
    print(f"[INFO] Wrote: {common_summary_path}")


if __name__ == "__main__":
    main()
