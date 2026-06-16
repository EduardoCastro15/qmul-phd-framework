#!/usr/bin/env python3
"""Prepare Eoin-style paired statistics for WLNM_dir_neg Apocrita results.

This script uses only the Python standard library so it can run in the project
environment without installing scipy/statsmodels. It prepares:

- a long real/pseudo table suitable for R mixed-effects models;
- run-level paired inputs, where each pseudo web replicate is paired with its
  original empirical web;
- paired t-tests over web x run pairs, overall and by ecosystem.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RESULTS_DIR = (
    ROOT
    / "src/matlab/data/result_wlnm_dir_neg_sweep_train_ratios_10-90_pseudo_properties_Apocrita"
)
DEFAULT_LOGS_DIR = DEFAULT_RESULTS_DIR / "prediction_scores_logs"
DEFAULT_METADATA_FILE = ROOT / "src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "statistical_tests/eoin"

TARGET_TRAIN_RATIO = 90.0
ALPHA = 0.05

METRICS: Sequence[Tuple[str, str, str]] = (
    ("Connectance", "EmpiricalConnectance", "PseudoConnectance"),
    ("AverageTrophicHeight", "EmpiricalMeanTrophicLevel", "PseudoMeanTrophicLevel"),
    ("MeanGenerality", "EmpiricalMeanGenerality", "PseudoMeanGenerality"),
    ("MeanVulnerability", "EmpiricalMeanVulnerability", "PseudoMeanVulnerability"),
)

EOIN_METRIC_COLUMNS = {
    "Connectance": "connectance",
    "AverageTrophicHeight": "average_trophic_height",
    "MeanGenerality": "mean_generality",
    "MeanVulnerability": "mean_vulnerability",
}

ECOSYSTEM_ORDER = (
    "marine",
    "streams",
    "lakes",
    "terrestrial aboveground",
    "terrestrial belowground",
    "unknown",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare paired t-test and mixed-model inputs for WLNM_dir_neg."
    )
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=TARGET_TRAIN_RATIO)
    return parser.parse_args()


def train_ratio_suffix(train_ratio: float) -> str:
    percent = train_ratio * 100.0 if train_ratio <= 1.0 else train_ratio
    rounded = round(percent)
    if abs(percent - rounded) <= 1e-9:
        return f"train{int(rounded)}"
    text = f"{percent:g}".replace(".", "p")
    return f"train{text}"


def read_metadata(path: Path) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Foodweb", "EcosystemType"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            foodweb = (row.get("Foodweb") or "").strip()
            ecosystem = (row.get("EcosystemType") or "").strip() or "unknown"
            if foodweb:
                metadata[foodweb] = ecosystem
    return metadata


def foodweb_from_filename(path: Path) -> str:
    marker = "_results_"
    if marker in path.name:
        return path.name.split(marker, 1)[0]
    return path.stem


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def train_ratio_matches(value: Optional[float], target: float) -> bool:
    if value is None:
        return False
    candidates = (target, target / 100.0 if target > 1.0 else target * 100.0)
    return any(abs(value - candidate) <= 1e-9 for candidate in candidates)


def ordered_ecosystems(values: Iterable[str]) -> List[str]:
    seen = {value for value in values if value}
    ordered = [value for value in ECOSYSTEM_ORDER if value in seen]
    ordered.extend(sorted(seen.difference(ordered)))
    return ordered


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[MutableMapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sample_std(values: Sequence[float]) -> Optional[float]:
    n = len(values)
    if n < 2:
        return None
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (n - 1))


def betacf(a: float, b: float, x: float) -> float:
    max_iterations = 200
    eps = 3e-14
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h

    return h


def regularized_betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        value = bt * betacf(a, b, x) / a
    else:
        value = 1.0 - bt * betacf(b, a, 1.0 - x) / b
    return max(0.0, min(1.0, value))


def student_t_two_tailed_p(t_statistic: float, df: int) -> Optional[float]:
    if df <= 0 or not math.isfinite(t_statistic):
        return None
    x = df / (df + t_statistic * t_statistic)
    return regularized_betainc(df / 2.0, 0.5, x)


def paired_ttest(real_values: Sequence[float], pseudo_values: Sequence[float]) -> Dict[str, object]:
    pairs = [
        (real, pseudo)
        for real, pseudo in zip(real_values, pseudo_values)
        if math.isfinite(real) and math.isfinite(pseudo)
    ]
    n = len(pairs)
    if n == 0:
        return {
            "NumFoodWebs": 0,
            "MeanReal": "",
            "MeanPseudo": "",
            "MeanDeltaPseudoMinusReal": "",
            "StdDelta": "",
            "TStatistic": "",
            "DF": "",
            "PValue": "",
            "Alpha": ALPHA,
            "RejectH0": "",
            "Direction": "not_available",
        }

    real = [pair[0] for pair in pairs]
    pseudo = [pair[1] for pair in pairs]
    deltas = [pair[1] - pair[0] for pair in pairs]
    mean_delta = mean(deltas)
    std_delta = sample_std(deltas)

    if n < 2 or std_delta is None or std_delta == 0.0:
        t_statistic: object = ""
        p_value: object = 0.0 if n >= 2 and mean_delta != 0.0 else ""
        reject: object = p_value != "" and p_value < ALPHA
    else:
        t_float = mean_delta / (std_delta / math.sqrt(n))
        p_float = student_t_two_tailed_p(t_float, n - 1)
        t_statistic = t_float
        p_value = p_float if p_float is not None else ""
        reject = p_float is not None and p_float < ALPHA

    if mean_delta > 0:
        direction = "pseudo_higher"
    elif mean_delta < 0:
        direction = "pseudo_lower"
    else:
        direction = "no_mean_difference"

    return {
        "NumFoodWebs": n,
        "MeanReal": mean(real),
        "MeanPseudo": mean(pseudo),
        "MeanDeltaPseudoMinusReal": mean_delta,
        "StdDelta": std_delta if std_delta is not None else "",
        "TStatistic": t_statistic,
        "DF": n - 1 if n > 1 else "",
        "PValue": p_value,
        "Alpha": ALPHA,
        "RejectH0": reject,
        "Direction": direction,
    }


def collect_rows(
    logs_dir: Path, metadata: Dict[str, str], target_train_ratio: float
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    long_rows: List[Dict[str, object]] = []
    files = sorted(logs_dir.glob("*_results_random_wlnm_dir_neg.csv"))
    missing_metadata: List[str] = []
    target_rows = 0
    total_rows = 0
    skipped_missing_metric = 0

    for path in files:
        foodweb = foodweb_from_filename(path)
        ecosystem = metadata.get(foodweb, "unknown")
        if ecosystem == "unknown" and foodweb not in metadata:
            missing_metadata.append(foodweb)

        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                total_rows += 1
                train_ratio = parse_float(row.get("TrainRatio"))
                if not train_ratio_matches(train_ratio, target_train_ratio):
                    continue
                target_rows += 1
                experiment_id = row.get("ExperimentID", "")
                k_value = row.get("K", "")
                threshold_mode = row.get("ThresholdMode", "")
                evaluate_on_all_unseen = row.get("EvaluateOnAllUnseen", "")

                for metric_name, empirical_column, pseudo_column in METRICS:
                    empirical_value = parse_float(row.get(empirical_column))
                    pseudo_value = parse_float(row.get(pseudo_column))
                    if empirical_value is None:
                        skipped_missing_metric += 1
                    else:
                        long_rows.append(
                            {
                                "Version": "wlnm_dir_neg",
                                "Foodweb": foodweb,
                                "EcosystemType": ecosystem,
                                "ExperimentID": experiment_id,
                                "K": k_value,
                                "TrainRatio": train_ratio,
                                "ThresholdMode": threshold_mode,
                                "EvaluateOnAllUnseen": evaluate_on_all_unseen,
                                "Metric": metric_name,
                                "MetricColumn": empirical_column,
                                "WebType": "real",
                                "Value": empirical_value,
                            }
                        )
                    if pseudo_value is None:
                        skipped_missing_metric += 1
                    else:
                        long_rows.append(
                            {
                                "Version": "wlnm_dir_neg",
                                "Foodweb": foodweb,
                                "EcosystemType": ecosystem,
                                "ExperimentID": experiment_id,
                                "K": k_value,
                                "TrainRatio": train_ratio,
                                "ThresholdMode": threshold_mode,
                                "EvaluateOnAllUnseen": evaluate_on_all_unseen,
                                "Metric": metric_name,
                                "MetricColumn": pseudo_column,
                                "WebType": "pseudo",
                                "Value": pseudo_value,
                            }
                        )

    summary = {
        "RequestedTrainRatio": target_train_ratio,
        "SourceLogFiles": len(files),
        "TotalRowsRead": total_rows,
        "TargetTrainRatioRows": target_rows,
        "LongRowsWritten": len(long_rows),
        "MissingMetricValuesSkipped": skipped_missing_metric,
        "FoodwebsMissingMetadata": len(set(missing_metadata)),
        "MissingMetadataFoodwebs": "; ".join(sorted(set(missing_metadata))),
    }
    return long_rows, summary


def build_run_level_pairs(long_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    values: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    for row in long_rows:
        key = (str(row["Foodweb"]), str(row["ExperimentID"]), str(row["Metric"]))
        if key not in grouped:
            grouped[key] = {
                "web": row["Foodweb"],
                "run_id": row["ExperimentID"],
                "ecosystem": row["EcosystemType"],
                "metric": EOIN_METRIC_COLUMNS[str(row["Metric"])],
                "train_ratio": row["TrainRatio"],
                "k": row["K"],
                "threshold_mode": row["ThresholdMode"],
                "evaluate_on_all_unseen": row["EvaluateOnAllUnseen"],
            }
        value = parse_float(row["Value"])
        web_type = str(row["WebType"])
        if value is not None and web_type in ("real", "pseudo"):
            values[(key[0], key[1], key[2], web_type)] = value

    output_rows: List[Dict[str, object]] = []
    for key in sorted(grouped):
        real = values.get((key[0], key[1], key[2], "real"))
        pseudo = values.get((key[0], key[1], key[2], "pseudo"))
        if real is None or pseudo is None:
            continue
        row = dict(grouped[key])
        row.update(
            {
                "real": real,
                "pseudo": pseudo,
                "delta_pseudo_minus_real": pseudo - real,
            }
        )
        output_rows.append(row)

    return output_rows


def build_run_level_connectance_pairs(
    pair_rows: Sequence[Dict[str, object]]
) -> List[Dict[str, object]]:
    return [row for row in pair_rows if row["metric"] == "connectance"]


def build_run_level_mixed_model_long(
    pair_rows: Sequence[Dict[str, object]]
) -> List[Dict[str, object]]:
    output_rows: List[Dict[str, object]] = []
    for row in pair_rows:
        for web_type, value_column in (("real", "real"), ("pseudo", "pseudo")):
            output_rows.append(
                {
                    "web": row["web"],
                    "run_id": row["run_id"],
                    "web_type": web_type,
                    "ecosystem": row["ecosystem"],
                    "metric": row["metric"],
                    "value": row[value_column],
                    "train_ratio": row["train_ratio"],
                    "k": row["k"],
                }
            )
    return output_rows


def build_run_level_mixed_model_wide(
    pair_rows: Sequence[Dict[str, object]]
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    metric_columns = [EOIN_METRIC_COLUMNS[metric_name] for metric_name, _, _ in METRICS]

    for pair in pair_rows:
        for web_type, value_column in (("real", "real"), ("pseudo", "pseudo")):
            key = (
                str(pair["web"]),
                str(pair["run_id"]),
                web_type,
                str(pair["ecosystem"]),
            )
            if key not in grouped:
                grouped[key] = {
                    "web": pair["web"],
                    "run_id": pair["run_id"],
                    "web_type": web_type,
                    "ecosystem": pair["ecosystem"],
                    "train_ratio": pair["train_ratio"],
                    "k": pair["k"],
                }
                for metric_column in metric_columns:
                    grouped[key][metric_column] = ""
            grouped[key][str(pair["metric"])] = pair[value_column]

    return [grouped[key] for key in sorted(grouped)]


def paired_ttest_result_row(
    metric: str,
    scope: str,
    ecosystem: str,
    pair_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    real = [float(row["real"]) for row in pair_rows]
    pseudo = [float(row["pseudo"]) for row in pair_rows]
    test = paired_ttest(real, pseudo)
    return {
        "metric": metric,
        "scope": scope,
        "ecosystem": ecosystem,
        "n_pairs": test["NumFoodWebs"],
        "n_webs": len({str(row["web"]) for row in pair_rows}),
        "mean_real": test["MeanReal"],
        "mean_pseudo": test["MeanPseudo"],
        "delta_pseudo_minus_real": test["MeanDeltaPseudoMinusReal"],
        "std_delta": test["StdDelta"],
        "t_statistic": test["TStatistic"],
        "df": test["DF"],
        "p_value": test["PValue"],
        "alpha": test["Alpha"],
        "reject_h0": test["RejectH0"],
        "direction": test["Direction"],
    }


def build_run_level_general_ttests(
    pair_rows: Sequence[Dict[str, object]]
) -> List[Dict[str, object]]:
    output_rows: List[Dict[str, object]] = []
    for metric_name, _, _ in METRICS:
        metric = EOIN_METRIC_COLUMNS[metric_name]
        metric_rows = [row for row in pair_rows if row["metric"] == metric]
        output_rows.append(
            paired_ttest_result_row(metric, "overall", "all", metric_rows)
        )
    return output_rows


def build_run_level_ecosystem_ttests(
    pair_rows: Sequence[Dict[str, object]]
) -> List[Dict[str, object]]:
    output_rows: List[Dict[str, object]] = []
    ecosystems = ordered_ecosystems(str(row["ecosystem"]) for row in pair_rows)
    for metric_name, _, _ in METRICS:
        metric = EOIN_METRIC_COLUMNS[metric_name]
        metric_rows = [row for row in pair_rows if row["metric"] == metric]
        for ecosystem in ecosystems:
            ecosystem_rows = [
                row for row in metric_rows if str(row["ecosystem"]) == ecosystem
            ]
            if ecosystem_rows:
                output_rows.append(
                    paired_ttest_result_row(metric, "by_ecosystem", ecosystem, ecosystem_rows)
                )
    return output_rows


def build_run_level_summary_rows(
    summary: Dict[str, object], pair_rows: Sequence[Dict[str, object]]
) -> List[Dict[str, object]]:
    foodwebs = {str(row["web"]) for row in pair_rows}
    ecosystems = sorted({str(row["ecosystem"]) for row in pair_rows})
    metrics = sorted({str(row["metric"]) for row in pair_rows})
    connectance_rows = [row for row in pair_rows if row["metric"] == "connectance"]

    runs_by_foodweb: Dict[str, set] = defaultdict(set)
    for row in connectance_rows:
        runs_by_foodweb[str(row["web"])].add(str(row["run_id"]))

    run_counts = [len(values) for values in runs_by_foodweb.values() if values]
    summary = dict(summary)
    summary.update(
        {
            "AnalysisUnit": "web x run",
            "RunLevelPairedRows": len(pair_rows),
            "ConnectancePairedRows": len(connectance_rows),
            "FoodwebsWithRunLevelPairs": len(foodwebs),
            "MetricsPrepared": len(metrics),
            "MetricNames": "; ".join(metrics),
            "EcosystemTypes": "; ".join(ecosystems),
            "MinRunsPerFoodweb": min(run_counts) if run_counts else "",
            "MaxRunsPerFoodweb": max(run_counts) if run_counts else "",
        }
    )
    return [{"Field": key, "Value": value} for key, value in summary.items()]


def main() -> int:
    args = parse_args()
    logs_dir = args.logs_dir.resolve()
    metadata_file = args.metadata_file.resolve()
    output_dir = args.output_dir.resolve()

    if not logs_dir.is_dir():
        print(f"Logs directory not found: {logs_dir}", file=sys.stderr)
        return 2
    if not metadata_file.is_file():
        print(f"Metadata file not found: {metadata_file}", file=sys.stderr)
        return 2

    metadata = read_metadata(metadata_file)
    suffix = train_ratio_suffix(args.train_ratio)
    long_rows, collection_summary = collect_rows(logs_dir, metadata, args.train_ratio)
    eoin_paired_input_rows = build_run_level_pairs(long_rows)
    eoin_connectance_paired_rows = build_run_level_connectance_pairs(eoin_paired_input_rows)
    eoin_mixed_model_long_rows = build_run_level_mixed_model_long(eoin_paired_input_rows)
    eoin_mixed_model_wide_rows = build_run_level_mixed_model_wide(eoin_paired_input_rows)
    eoin_general_ttests = build_run_level_general_ttests(eoin_paired_input_rows)
    eoin_ecosystem_ttests = build_run_level_ecosystem_ttests(eoin_paired_input_rows)
    summary_rows = build_run_level_summary_rows(collection_summary, eoin_paired_input_rows)

    write_csv(
        output_dir / f"eoin_paired_input_{suffix}.csv",
        [
            "web",
            "run_id",
            "ecosystem",
            "metric",
            "real",
            "pseudo",
            "delta_pseudo_minus_real",
            "train_ratio",
            "k",
        ],
        eoin_paired_input_rows,
    )
    write_csv(
        output_dir / f"eoin_connectance_paired_input_{suffix}.csv",
        [
            "web",
            "run_id",
            "ecosystem",
            "real",
            "pseudo",
            "delta_pseudo_minus_real",
            "train_ratio",
            "k",
        ],
        eoin_connectance_paired_rows,
    )
    write_csv(
        output_dir / f"eoin_mixed_model_long_{suffix}.csv",
        [
            "web",
            "run_id",
            "web_type",
            "ecosystem",
            "metric",
            "value",
            "train_ratio",
            "k",
        ],
        eoin_mixed_model_long_rows,
    )
    write_csv(
        output_dir / f"eoin_mixed_model_wide_{suffix}.csv",
        [
            "web",
            "run_id",
            "web_type",
            "ecosystem",
            "connectance",
            "average_trophic_height",
            "mean_generality",
            "mean_vulnerability",
            "train_ratio",
            "k",
        ],
        eoin_mixed_model_wide_rows,
    )
    write_csv(
        output_dir / f"eoin_paired_ttest_general_results_{suffix}.csv",
        [
            "metric",
            "scope",
            "ecosystem",
            "n_pairs",
            "n_webs",
            "mean_real",
            "mean_pseudo",
            "delta_pseudo_minus_real",
            "std_delta",
            "t_statistic",
            "df",
            "p_value",
            "alpha",
            "reject_h0",
            "direction",
        ],
        eoin_general_ttests,
    )
    write_csv(
        output_dir / f"eoin_paired_ttest_by_ecosystem_results_{suffix}.csv",
        [
            "metric",
            "scope",
            "ecosystem",
            "n_pairs",
            "n_webs",
            "mean_real",
            "mean_pseudo",
            "delta_pseudo_minus_real",
            "std_delta",
            "t_statistic",
            "df",
            "p_value",
            "alpha",
            "reject_h0",
            "direction",
        ],
        eoin_ecosystem_ttests,
    )
    write_csv(output_dir / f"eoin_run_summary_{suffix}.csv", ["Field", "Value"], summary_rows)

    print(f"Wrote Eoin-style outputs to {output_dir}")
    print(f"Food webs with run-level pairs: {len({row['web'] for row in eoin_paired_input_rows})}")
    print(f"Long rows: {len(long_rows)}")
    print(f"Run-level paired rows: {len(eoin_paired_input_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
