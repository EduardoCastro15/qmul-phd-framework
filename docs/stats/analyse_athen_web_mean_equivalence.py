#!/usr/bin/env python3
"""Athen-requested web-level equivalence analysis for WLNM_dir_neg.

The primary analysis unit is one empirical food web at each train ratio:

    mean_pseudo = mean(valid pseudo reconstructions)
    delta = mean_pseudo - empirical
    relative_error = delta / empirical

TOST equivalence is evaluated on food-web-level relative errors using
relative margins of 10%, 20%, and 30%. The main paper ratio is 60%.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = (
    ROOT
    / "src/matlab/data/"
    "result_wlnm_dir_neg_sweep_train_ratios_10-90_pseudo_properties_stats_Apocrita"
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    source_metric: str
    empirical_column: str
    pseudo_column: str
    trophic: bool = False


METRICS = (
    MetricSpec("Connectance", "Connectance", "EmpiricalConnectance", "PseudoConnectance"),
    MetricSpec(
        "MeanGenerality",
        "MeanGenerality",
        "EmpiricalMeanGenerality",
        "PseudoMeanGenerality",
    ),
    MetricSpec(
        "MeanVulnerability",
        "MeanVulnerability",
        "EmpiricalMeanVulnerability",
        "PseudoMeanVulnerability",
    ),
    MetricSpec(
        "MeanTrophicHeight",
        "NetworkXMeanTrophicLevel",
        "EmpiricalNetworkXMeanTrophicLevel",
        "PseudoNetworkXMeanTrophicLevel",
        trophic=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=RESULTS_DIR / "prediction_scores_logs",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=ROOT / "src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv",
    )
    parser.add_argument(
        "--lachlan-mean-file",
        type=Path,
        default=(
            ROOT
            / "data/processed/Average_Trophic_Size_Height_Gateway_Dataset/mean_tl.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "statistical_tests/athen",
    )
    parser.add_argument("--main-train-ratio", type=int, default=60)
    parser.add_argument("--min-valid-trophic-runs", type=int, default=10)
    parser.add_argument("--margins", default="10,15,20,30")
    return parser.parse_args()


def parse_float(value: object) -> Optional[float]:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_int(value: object) -> Optional[int]:
    number = parse_float(value)
    if number is None:
        return None
    return int(round(number))


def mean_or_nan(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else math.nan


def sample_stats(values: Sequence[float]) -> Dict[str, float]:
    n = len(values)
    mean = mean_or_nan(values)
    if n < 2:
        return {"N": n, "Mean": mean, "SD": math.nan, "SE": math.nan}
    sd = statistics.stdev(values)
    return {"N": n, "Mean": mean, "SD": sd, "SE": sd / math.sqrt(n)}


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


def student_t_cdf(value: float, df: int) -> float:
    if df <= 0 or math.isnan(value):
        return math.nan
    if math.isinf(value):
        return 1.0 if value > 0 else 0.0
    x = df / (df + value * value)
    ib = regularized_betainc(df / 2.0, 0.5, x)
    result = 1.0 - 0.5 * ib if value >= 0 else 0.5 * ib
    return max(0.0, min(1.0, result))


def student_t_inv_cdf(probability: float, df: int) -> float:
    if not 0.0 < probability < 1.0 or df <= 0:
        return math.nan
    if probability < 0.5:
        return -student_t_inv_cdf(1.0 - probability, df)
    lo = 0.0
    hi = 1.0
    while student_t_cdf(hi, df) < probability:
        hi *= 2.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        if student_t_cdf(mid, df) < probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def confidence_interval(values: Sequence[float], level: float = 0.90) -> Tuple[float, float]:
    stats = sample_stats(values)
    if stats["N"] < 2:
        return math.nan, math.nan
    alpha = 1.0 - level
    critical = student_t_inv_cdf(1.0 - alpha / 2.0, int(stats["N"] - 1))
    distance = critical * stats["SE"]
    return stats["Mean"] - distance, stats["Mean"] + distance


def tost(values: Sequence[float], margin: float, alpha: float = 0.05) -> Dict[str, object]:
    stats = sample_stats(values)
    n = int(stats["N"])
    mean = stats["Mean"]
    sd = stats["SD"]
    se = stats["SE"]
    lower_margin = -margin
    upper_margin = margin
    ci_lower, ci_upper = confidence_interval(values, 1.0 - 2.0 * alpha)

    if n < 2:
        return {
            "TLower": math.nan,
            "PLower": math.nan,
            "TUpper": math.nan,
            "PUpper": math.nan,
            "TOSTPValue": math.nan,
            "Equivalent": False,
            "CILower": ci_lower,
            "CIUpper": ci_upper,
        }

    if sd == 0.0:
        p_lower = 0.0 if mean > lower_margin else 1.0
        p_upper = 0.0 if mean < upper_margin else 1.0
        t_lower = math.inf if p_lower == 0.0 else -math.inf
        t_upper = -math.inf if p_upper == 0.0 else math.inf
    else:
        t_lower = (mean - lower_margin) / se
        p_lower = 1.0 - student_t_cdf(t_lower, n - 1)
        t_upper = (mean - upper_margin) / se
        p_upper = student_t_cdf(t_upper, n - 1)

    equivalent = p_lower < alpha and p_upper < alpha
    return {
        "TLower": t_lower,
        "PLower": p_lower,
        "TUpper": t_upper,
        "PUpper": p_upper,
        "TOSTPValue": max(p_lower, p_upper),
        "Equivalent": equivalent,
        "CILower": ci_lower,
        "CIUpper": ci_upper,
    }


def read_metadata(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            web = (row.get("Foodweb") or "").strip()
            ecosystem = (row.get("EcosystemType") or "unknown").strip() or "unknown"
            if web:
                result[web] = ecosystem
    return result


def web_from_filename(path: Path) -> str:
    return path.name.split("_results_", 1)[0]


def new_group(web: str, ecosystem: str, ratio: int, spec: MetricSpec) -> Dict[str, object]:
    return {
        "web": web,
        "ecosystem": ecosystem,
        "ratio": ratio,
        "metric": spec.name,
        "source_metric": spec.source_metric,
        "trophic": spec.trophic,
        "total_runs": 0,
        "empirical_values": [],
        "valid_empirical_values": [],
        "valid_pseudo_values": [],
        "empirical_status": Counter(),
        "pseudo_status": Counter(),
        "pseudo_species_with_level": [],
        "pseudo_metric_valid_runs": 0,
    }


def read_logs(
    logs_dir: Path,
    metadata: Dict[str, str],
) -> Tuple[Dict[Tuple[str, int, str], Dict[str, object]], Dict[str, Dict[str, object]], int, int]:
    groups: Dict[Tuple[str, int, str], Dict[str, object]] = {}
    empirical_diagnostics: Dict[str, Dict[str, object]] = {}
    raw_rows = 0
    files = sorted(logs_dir.glob("*_results_random_wlnm_dir_neg.csv"))

    for path in files:
        web = web_from_filename(path)
        ecosystem = metadata.get(web, "unknown")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_rows += 1
                ratio = parse_int(row.get("TrainRatio"))
                if ratio is None:
                    continue

                if web not in empirical_diagnostics:
                    empirical_diagnostics[web] = {
                        "FoodWeb": web,
                        "BaseFoodWeb": web.removesuffix("_tax_mass").strip(),
                        "EcosystemType": ecosystem,
                        "LogEmpiricalTrophicHeight": parse_float(
                            row.get("EmpiricalNetworkXMeanTrophicLevel")
                        ),
                        "LogNumSpeciesFull": parse_int(
                            row.get("EmpiricalNetworkXTrophicLevelNumSpeciesFull")
                        ),
                        "LogNumSpeciesLargest": parse_int(
                            row.get("EmpiricalNetworkXTrophicLevelNumSpeciesLargest")
                        ),
                        "LogNumSpeciesWithLevel": parse_int(
                            row.get("EmpiricalNetworkXTrophicLevelNumSpeciesWithLevel")
                        ),
                        "LogStatusCode": parse_int(
                            row.get("EmpiricalNetworkXTrophicLevelStatusCode")
                        ),
                    }

                for spec in METRICS:
                    key = (web, ratio, spec.name)
                    group = groups.setdefault(key, new_group(web, ecosystem, ratio, spec))
                    group["total_runs"] = int(group["total_runs"]) + 1

                    empirical = parse_float(row.get(spec.empirical_column))
                    pseudo = parse_float(row.get(spec.pseudo_column))
                    if empirical is not None:
                        group["empirical_values"].append(empirical)  # type: ignore[union-attr]

                    if spec.trophic:
                        empirical_status = str(row.get("EmpiricalNetworkXTrophicLevelStatusCode", ""))
                        pseudo_status = str(row.get("PseudoNetworkXTrophicLevelStatusCode", ""))
                        pseudo_n = parse_int(row.get("PseudoNetworkXTrophicLevelNumSpeciesWithLevel"))
                        group["empirical_status"][empirical_status] += 1  # type: ignore[index]
                        group["pseudo_status"][pseudo_status] += 1  # type: ignore[index]
                        if pseudo_n is not None:
                            group["pseudo_species_with_level"].append(pseudo_n)  # type: ignore[union-attr]
                        valid = (
                            empirical_status == "0"
                            and pseudo_status == "0"
                            and pseudo_n is not None
                            and pseudo_n >= 2
                            and empirical is not None
                            and pseudo is not None
                        )
                        pseudo_metric_valid = (
                            pseudo_status == "0"
                            and pseudo_n is not None
                            and pseudo_n >= 2
                            and pseudo is not None
                        )
                        if pseudo_metric_valid:
                            group["pseudo_metric_valid_runs"] = (
                                int(group["pseudo_metric_valid_runs"]) + 1
                            )
                    else:
                        valid = empirical is not None and pseudo is not None
                        if pseudo is not None:
                            group["pseudo_metric_valid_runs"] = (
                                int(group["pseudo_metric_valid_runs"]) + 1
                            )

                    if valid:
                        group["valid_empirical_values"].append(empirical)  # type: ignore[union-attr]
                        group["valid_pseudo_values"].append(pseudo)  # type: ignore[union-attr]

    return groups, empirical_diagnostics, len(files), raw_rows


def build_web_rows(
    groups: Dict[Tuple[str, int, str], Dict[str, object]],
    min_valid_trophic_runs: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for group in groups.values():
        empirical_values = list(group["empirical_values"])  # type: ignore[arg-type]
        valid_empirical = list(group["valid_empirical_values"])  # type: ignore[arg-type]
        valid_pseudo = list(group["valid_pseudo_values"])  # type: ignore[arg-type]
        total_runs = int(group["total_runs"])
        valid_runs = len(valid_pseudo)
        trophic = bool(group["trophic"])
        empirical = mean_or_nan(valid_empirical or empirical_values)
        pseudo = mean_or_nan(valid_pseudo)
        delta = pseudo - empirical if math.isfinite(empirical) and math.isfinite(pseudo) else math.nan
        relative = delta / empirical if math.isfinite(delta) and empirical != 0 else math.nan

        minimum = min_valid_trophic_runs if trophic else total_runs
        included = math.isfinite(relative) and valid_runs >= minimum
        if included:
            reason = ""
        elif trophic and not valid_empirical:
            reason = "empirical_trophic_height_undefined"
        elif trophic:
            reason = "fewer_than_minimum_valid_pseudo_runs"
        else:
            reason = "missing_or_nonfinite_metric"

        empirical_status = group["empirical_status"]  # type: ignore[assignment]
        pseudo_status = group["pseudo_status"]  # type: ignore[assignment]
        rows.append(
            {
                "FoodWeb": group["web"],
                "EcosystemType": group["ecosystem"],
                "TrainRatio": group["ratio"],
                "Metric": group["metric"],
                "SourceMetric": group["source_metric"],
                "TotalRuns": total_runs,
                "ValidPairedRuns": valid_runs,
                "InvalidPairedRuns": total_runs - valid_runs,
                "PseudoMetricValidRuns": int(group["pseudo_metric_valid_runs"]),
                "PseudoMetricInvalidRuns": total_runs
                - int(group["pseudo_metric_valid_runs"]),
                "Complete20": int(valid_runs == 20),
                "EmpiricalValue": empirical,
                "MeanPseudo": pseudo,
                "DeltaPseudoMinusEmpirical": delta,
                "RelativeError": relative,
                "RelativeErrorPercent": 100.0 * relative if math.isfinite(relative) else math.nan,
                "IncludedPrimary": int(included),
                "ExclusionReason": reason,
                "EmpiricalStatus0Runs": empirical_status.get("0", 0),
                "EmpiricalStatus2Runs": empirical_status.get("2", 0),
                "PseudoStatus0Runs": pseudo_status.get("0", 0),
                "PseudoStatus2Runs": pseudo_status.get("2", 0),
                "PseudoStatus3Runs": pseudo_status.get("3", 0),
            }
        )
    return sorted(rows, key=lambda row: (int(row["TrainRatio"]), str(row["Metric"]), str(row["FoodWeb"])))


def describe_population(
    rows: Sequence[Dict[str, object]],
    ratio: int,
    metric: str,
    population: str,
) -> Optional[Dict[str, object]]:
    candidates = [row for row in rows if row["TrainRatio"] == ratio and row["Metric"] == metric]
    if population == "Primary":
        selected = [row for row in candidates if row["IncludedPrimary"] == 1]
        population_label = (
            "ConditionalAtLeast10ValidRuns"
            if metric == "MeanTrophicHeight"
            else "PrimaryMean20"
        )
    elif population == "Complete20Sensitivity":
        selected = [
            row
            for row in candidates
            if row["Complete20"] == 1 and math.isfinite(float(row["RelativeError"]))
        ]
        population_label = population
    else:
        raise ValueError(population)
    if not selected:
        return None

    empirical = [float(row["EmpiricalValue"]) for row in selected]
    pseudo = [float(row["MeanPseudo"]) for row in selected]
    delta = [float(row["DeltaPseudoMinusEmpirical"]) for row in selected]
    relative = [float(row["RelativeError"]) for row in selected]
    valid_runs = [int(row["ValidPairedRuns"]) for row in selected]
    empirical_stats = sample_stats(empirical)
    pseudo_stats = sample_stats(pseudo)
    delta_stats = sample_stats(delta)
    relative_stats = sample_stats(relative)
    delta_ci = confidence_interval(delta)
    relative_ci = confidence_interval(relative)
    source_metric = selected[0]["SourceMetric"]

    return {
        "TrainRatio": ratio,
        "Metric": metric,
        "SourceMetric": source_metric,
        "AnalysisPopulation": population_label,
        "NumFoodWebs": len(selected),
        "NumComplete20FoodWebs": sum(int(row["Complete20"]) for row in selected),
        "MinValidRuns": min(valid_runs),
        "MedianValidRuns": statistics.median(valid_runs),
        "MaxValidRuns": max(valid_runs),
        "MeanEmpirical": empirical_stats["Mean"],
        "SDEmpirical": empirical_stats["SD"],
        "SEMEmpirical": empirical_stats["SE"],
        "MeanPseudo": pseudo_stats["Mean"],
        "SDPseudo": pseudo_stats["SD"],
        "MeanDelta": delta_stats["Mean"],
        "SDDelta": delta_stats["SD"],
        "SEDelta": delta_stats["SE"],
        "DeltaCI90Lower": delta_ci[0],
        "DeltaCI90Upper": delta_ci[1],
        "MeanRelativeError": relative_stats["Mean"],
        "SDRelativeError": relative_stats["SD"],
        "SERelativeError": relative_stats["SE"],
        "RelativeCI90Lower": relative_ci[0],
        "RelativeCI90Upper": relative_ci[1],
        "MeanRelativeErrorPercent": 100.0 * relative_stats["Mean"],
        "RelativeCI90LowerPercent": 100.0 * relative_ci[0],
        "RelativeCI90UpperPercent": 100.0 * relative_ci[1],
    }


def add_equivalence(summary: Dict[str, object], margin_percent: float) -> Dict[str, object]:
    # Reconstruct the population values in the caller; this placeholder is
    # replaced there with the TOST result while retaining a flat output row.
    result = dict(summary)
    result["MarginPercent"] = margin_percent
    result["LowerMarginRelative"] = -margin_percent / 100.0
    result["UpperMarginRelative"] = margin_percent / 100.0
    return result


def build_summaries(
    web_rows: Sequence[Dict[str, object]],
    margins: Sequence[float],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    descriptive: List[Dict[str, object]] = []
    equivalence: List[Dict[str, object]] = []
    ratios = sorted({int(row["TrainRatio"]) for row in web_rows})

    for ratio in ratios:
        for spec in METRICS:
            populations = ["Primary"]
            if spec.trophic:
                populations.append("Complete20Sensitivity")
            for population in populations:
                summary = describe_population(web_rows, ratio, spec.name, population)
                if summary is None:
                    continue
                descriptive.append(summary)

                if population == "Primary":
                    selected = [
                        row
                        for row in web_rows
                        if row["TrainRatio"] == ratio
                        and row["Metric"] == spec.name
                        and row["IncludedPrimary"] == 1
                    ]
                else:
                    selected = [
                        row
                        for row in web_rows
                        if row["TrainRatio"] == ratio
                        and row["Metric"] == spec.name
                        and row["Complete20"] == 1
                        and math.isfinite(float(row["RelativeError"]))
                    ]
                values = [float(row["RelativeError"]) for row in selected]

                for margin_percent in margins:
                    result = add_equivalence(summary, margin_percent)
                    test = tost(values, margin_percent / 100.0)
                    result.update(
                        {
                            "TLower": test["TLower"],
                            "PLower": test["PLower"],
                            "TUpper": test["TUpper"],
                            "PUpper": test["PUpper"],
                            "TOSTPValue": test["TOSTPValue"],
                            "Equivalent": int(bool(test["Equivalent"])),
                            "Conclusion": equivalence_conclusion(
                                float(summary["MeanRelativeError"]),
                                float(summary["RelativeCI90Lower"]),
                                float(summary["RelativeCI90Upper"]),
                                margin_percent / 100.0,
                                bool(test["Equivalent"]),
                            ),
                        }
                    )
                    equivalence.append(result)

    return descriptive, equivalence


def build_minimum_margin_rows(
    descriptive: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary in descriptive:
        lower = float(summary["RelativeCI90Lower"])
        upper = float(summary["RelativeCI90Upper"])
        required_fraction = max(abs(lower), abs(upper))
        required_percent = 100.0 * required_fraction
        rows.append(
            {
                "TrainRatio": summary["TrainRatio"],
                "Metric": summary["Metric"],
                "SourceMetric": summary["SourceMetric"],
                "AnalysisPopulation": summary["AnalysisPopulation"],
                "NumFoodWebs": summary["NumFoodWebs"],
                "MeanRelativeErrorPercent": summary["MeanRelativeErrorPercent"],
                "RelativeCI90LowerPercent": summary["RelativeCI90LowerPercent"],
                "RelativeCI90UpperPercent": summary["RelativeCI90UpperPercent"],
                "MinimumMarginPercentExclusive": required_percent,
                "MinimumWholePercentMargin": math.floor(required_percent) + 1,
                "Interpretation": (
                    "Descriptive threshold only; do not select the ecological "
                    "margin post hoc from this value."
                ),
            }
        )
    return rows


def equivalence_conclusion(
    mean: float,
    ci_lower: float,
    ci_upper: float,
    margin: float,
    equivalent: bool,
) -> str:
    if equivalent:
        return "equivalent"
    if mean < -margin:
        return "not_equivalent_pseudo_lower"
    if mean > margin:
        return "not_equivalent_pseudo_higher"
    if ci_lower <= -margin or ci_upper >= margin:
        return "not_equivalent_ci_crosses_margin"
    return "not_equivalent"


def build_trophic_validity(web_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    ratios = sorted({int(row["TrainRatio"]) for row in web_rows})
    for ratio in ratios:
        rows = [
            row
            for row in web_rows
            if row["TrainRatio"] == ratio and row["Metric"] == "MeanTrophicHeight"
        ]
        result.append(
            {
                "TrainRatio": ratio,
                "TotalFoodWebs": len(rows),
                "EmpiricalValidFoodWebs": sum(int(row["EmpiricalStatus0Runs"]) > 0 for row in rows),
                "EmpiricalInvalidFoodWebs": sum(int(row["EmpiricalStatus0Runs"]) == 0 for row in rows),
                "PrimaryIncludedFoodWebs": sum(int(row["IncludedPrimary"]) for row in rows),
                "Complete20FoodWebs": sum(int(row["Complete20"]) for row in rows),
                "Partial10To19FoodWebs": sum(
                    10 <= int(row["ValidPairedRuns"]) < 20
                    and int(row["EmpiricalStatus0Runs"]) > 0
                    for row in rows
                ),
                "Below10ValidFoodWebs": sum(
                    int(row["ValidPairedRuns"]) < 10
                    and int(row["EmpiricalStatus0Runs"]) > 0
                    for row in rows
                ),
                "ValidPairedRuns": sum(int(row["ValidPairedRuns"]) for row in rows),
                "InvalidPairedRuns": sum(int(row["InvalidPairedRuns"]) for row in rows),
                "PseudoMetricValidRuns": sum(
                    int(row["PseudoMetricValidRuns"]) for row in rows
                ),
                "PseudoMetricInvalidRuns": sum(
                    int(row["PseudoMetricInvalidRuns"]) for row in rows
                ),
                "PseudoStatus2Runs": sum(int(row["PseudoStatus2Runs"]) for row in rows),
                "PseudoStatus3Runs": sum(int(row["PseudoStatus3Runs"]) for row in rows),
            }
        )
    return result


def read_lachlan_means(path: Path) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            web = (row.get("Ecosystem") or "").strip()
            value = parse_float(row.get("Mean_Trophic_Level"))
            if web and value is not None:
                result[web] = {
                    "LachlanTrophicHeight": value,
                    "LachlanNumSpeciesFull": parse_int(row.get("Num_Species_full")),
                    "LachlanNumSpeciesLargest": parse_int(row.get("Num_Species_largest")),
                }
    return result


def build_lachlan_audit(
    diagnostics: Dict[str, Dict[str, object]],
    lachlan: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for diagnostic in diagnostics.values():
        base = str(diagnostic["BaseFoodWeb"])
        reference = lachlan.get(base)
        if reference is None:
            continue
        log_value = diagnostic["LogEmpiricalTrophicHeight"]
        lachlan_value = reference["LachlanTrophicHeight"]
        if log_value is None:
            difference = math.nan
        else:
            difference = float(log_value) - float(lachlan_value)
        rows.append(
            {
                **diagnostic,
                **reference,
                "DifferenceLogMinusLachlan": difference,
                "AbsoluteDifference": abs(difference) if math.isfinite(difference) else math.nan,
                "MatchWithin0.001": int(math.isfinite(difference) and abs(difference) <= 0.001),
            }
        )
    return sorted(rows, key=lambda row: str(row["FoodWeb"]))


def build_reference_summary(
    web_rows: Sequence[Dict[str, object]],
    lachlan: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    for spec in METRICS:
        values_by_web: Dict[str, float] = {}
        for row in web_rows:
            if row["Metric"] != spec.name:
                continue
            value = float(row["EmpiricalValue"])
            if math.isfinite(value):
                values_by_web.setdefault(str(row["FoodWeb"]), value)
        stats = sample_stats(list(values_by_web.values()))
        result.append(
            {
                "Reference": "WLNMLogs",
                "Metric": spec.name,
                "SourceMetric": spec.source_metric,
                "NumFoodWebs": stats["N"],
                "Mean": stats["Mean"],
                "SD": stats["SD"],
                "SEM": stats["SE"],
                "SDPercentOfMean": 100.0 * stats["SD"] / stats["Mean"],
                "SEMPercentOfMean": 100.0 * stats["SE"] / stats["Mean"],
            }
        )

    lachlan_values = [float(row["LachlanTrophicHeight"]) for row in lachlan.values()]
    stats = sample_stats(lachlan_values)
    result.append(
        {
            "Reference": "LachlanMeanTLCSV",
            "Metric": "MeanTrophicHeight",
            "SourceMetric": "NetworkX trophic_levels plus Lachlan postprocessing",
            "NumFoodWebs": stats["N"],
            "Mean": stats["Mean"],
            "SD": stats["SD"],
            "SEM": stats["SE"],
            "SDPercentOfMean": 100.0 * stats["SD"] / stats["Mean"],
            "SEMPercentOfMean": 100.0 * stats["SE"] / stats["Mean"],
        }
    )
    return result


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_readme(
    path: Path,
    main_rows: Sequence[Dict[str, object]],
    run_summary: Dict[str, object],
) -> None:
    lines = [
        "# Athen web-level equivalence analysis",
        "",
        "Primary unit: one empirical food web compared with the mean of its pseudo reconstructions.",
        "Relative error: (mean pseudo - empirical) / empirical.",
        "TOST alpha: 0.05; equivalence uses the 90% CI and relative margins 10%, 15%, 20%, 30%.",
        "",
        "Mean trophic height has two explicitly labelled populations:",
        "ConditionalAtLeast10ValidRuns and Complete20Sensitivity.",
        "The difference between them quantifies sensitivity to undefined trophic heights.",
        "Trophic-height results remain provisional until the discrepancies with Lachlan's CSV are resolved.",
        "",
        f"Log files: {run_summary['LogFiles']}",
        f"Raw rows: {run_summary['RawRows']}",
        f"Main train ratio: {run_summary['MainTrainRatio']}",
        "",
        "## Main train-ratio results",
        "",
        "| Metric | Population | Margin | Mean relative error | 90% CI | Equivalent |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            "| {Metric} | {AnalysisPopulation} | {MarginPercent:.0f}% | "
            "{MeanRelativeErrorPercent:.2f}% | [{RelativeCI90LowerPercent:.2f}%, "
            "{RelativeCI90UpperPercent:.2f}%] | {Equivalent} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    margins = [float(value.strip()) for value in args.margins.split(",") if value.strip()]
    metadata = read_metadata(args.metadata_file)
    groups, diagnostics, log_files, raw_rows = read_logs(args.logs_dir, metadata)
    web_rows = build_web_rows(groups, args.min_valid_trophic_runs)
    descriptive, equivalence = build_summaries(web_rows, margins)
    minimum_margins = build_minimum_margin_rows(descriptive)
    trophic_validity = build_trophic_validity(web_rows)
    lachlan = read_lachlan_means(args.lachlan_mean_file)
    lachlan_audit = build_lachlan_audit(diagnostics, lachlan)
    reference_summary = build_reference_summary(web_rows, lachlan)

    main_descriptive = [
        row for row in descriptive if row["TrainRatio"] == args.main_train_ratio
    ]
    main_equivalence = [
        row for row in equivalence if row["TrainRatio"] == args.main_train_ratio
    ]
    main_minimum_margins = [
        row for row in minimum_margins if row["TrainRatio"] == args.main_train_ratio
    ]

    audit_finite = [
        float(row["AbsoluteDifference"])
        for row in lachlan_audit
        if math.isfinite(float(row["AbsoluteDifference"]))
    ]
    run_summary = {
        "AnalysisDesign": "repeated_holdout_20_pseudo_runs_per_foodweb_ratio",
        "PrimaryAnalysisUnit": "foodweb_mean_pseudo_minus_empirical",
        "MainTrainRatio": args.main_train_ratio,
        "RelativeMarginsPercent": ";".join(f"{value:g}" for value in margins),
        "MinValidTrophicRuns": args.min_valid_trophic_runs,
        "LogFiles": log_files,
        "RawRows": raw_rows,
        "WebMetricRows": len(web_rows),
        "ExpectedRawRows": 290 * 9 * 20,
        "LachlanAuditRows": len(lachlan_audit),
        "LachlanMatchesWithin0.001": sum(
            int(row["MatchWithin0.001"]) for row in lachlan_audit
        ),
        "LachlanMismatchesOver0.001": sum(
            not int(row["MatchWithin0.001"]) for row in lachlan_audit
        ),
        "MaxAbsoluteLachlanDifference": max(audit_finite) if audit_finite else math.nan,
    }

    output = args.output_dir
    write_csv(output / "athen_web_level_metrics_all_ratios.csv", web_rows)
    write_csv(output / "athen_metric_summary_all_ratios.csv", descriptive)
    write_csv(output / "athen_metric_summary_train60.csv", main_descriptive)
    write_csv(output / "athen_equivalence_sensitivity_all_ratios.csv", equivalence)
    write_csv(output / "athen_equivalence_sensitivity_train60.csv", main_equivalence)
    write_csv(output / "athen_minimum_margin_required_all_ratios.csv", minimum_margins)
    write_csv(output / "athen_minimum_margin_required_train60.csv", main_minimum_margins)
    write_csv(output / "athen_trophic_height_validity.csv", trophic_validity)
    write_csv(output / "athen_empirical_reference_summary.csv", reference_summary)
    write_csv(output / "athen_trophic_height_lachlan_audit.csv", lachlan_audit)
    write_csv(
        output / "athen_analysis_run_summary.csv",
        [{"Key": key, "Value": value} for key, value in run_summary.items()],
    )
    write_readme(output / "README.md", main_equivalence, run_summary)

    print(f"[AthenAnalysis] Log files: {log_files}")
    print(f"[AthenAnalysis] Raw rows: {raw_rows}")
    print(f"[AthenAnalysis] Web-level rows: {len(web_rows)}")
    print(f"[AthenAnalysis] Main equivalence rows: {len(main_equivalence)}")
    print(f"[AthenAnalysis] Output: {output}")


if __name__ == "__main__":
    main()
