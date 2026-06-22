#!/usr/bin/env python3
"""Mixed-effects estimation plus equivalence margins for WLNM_dir_neg.

This script intentionally uses only the Python standard library. The execution
environment used for this project does not currently provide pandas, scipy,
statsmodels, or Rscript.

The model fitted for each metric is a Gaussian random-intercept model:

    delta ~ fixed_effects + (1 | web)

where delta = pseudo_metric - empirical_metric. Variance components are
estimated by profiling the REML objective over the random-intercept/residual
variance ratio. Fixed-effect confidence intervals use an approximate t
reference with food-web-level degrees of freedom.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = (
    ROOT
    / "src/matlab/data/result_wlnm_dir_neg_sweep_train_ratios_10-90_pseudo_properties_Apocrita"
)
DEFAULT_LOGS_DIR = RESULTS_DIR / "prediction_scores_logs"
DEFAULT_METADATA_FILE = ROOT / "src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv"
DEFAULT_MARGINS_FILE = RESULTS_DIR / "statistical_tests/lachlan/wlnm_dir_neg_delta_equivalence_margins.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "statistical_tests/jorge"

TRAIN_RATIO = 60.0

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
        description="Estimate mixed-effects deltas and compare with equivalence margins."
    )
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA_FILE)
    parser.add_argument("--margins-file", type=Path, default=DEFAULT_MARGINS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Only fit the eight core metrics used in the email summary.",
    )
    return parser.parse_args()


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


def read_margins(path: Path, core_only: bool) -> List[Dict[str, object]]:
    core_metrics = {
        "Connectance",
        "MeanTrophicLevel",
        "MeanDegree",
        "MeanGenerality",
        "MeanVulnerability",
        "PropBasal",
        "PropIntermediate",
        "PropTop",
    }
    margins: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric = (row.get("Metric") or "").strip()
            if not metric:
                continue
            if core_only and metric not in core_metrics:
                continue
            lower = parse_float(row.get("LowerMargin"))
            upper = parse_float(row.get("UpperMargin"))
            if lower is None or upper is None:
                continue
            margins.append(
                {
                    "Metric": metric,
                    "LowerMargin": lower,
                    "UpperMargin": upper,
                    "MarginMode": row.get("MarginMode", "absolute"),
                    "Justification": row.get("Justification", ""),
                }
            )
    return margins


def foodweb_from_filename(path: Path) -> str:
    marker = "_results_"
    if marker in path.name:
        return path.name.split(marker, 1)[0]
    return path.stem


def ordered_ecosystems(values: Iterable[str]) -> List[str]:
    seen = {value for value in values if value}
    ordered = [value for value in ECOSYSTEM_ORDER if value in seen]
    ordered.extend(sorted(seen.difference(ordered)))
    return ordered


def read_delta_records(
    logs_dir: Path,
    metadata: Dict[str, str],
    margins: Sequence[Dict[str, object]],
    train_ratio: float,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    margin_metrics = [str(row["Metric"]) for row in margins]
    records: List[Dict[str, object]] = []
    raw_rows = 0
    matched_rows = 0
    files = sorted(logs_dir.glob("*.csv"))

    for path in files:
        web = foodweb_from_filename(path)
        ecosystem = metadata.get(web, "unknown")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_rows += 1
                row_train_ratio = parse_float(row.get("TrainRatio"))
                if not train_ratio_matches(row_train_ratio, train_ratio):
                    continue
                matched_rows += 1
                run_id = row.get("ExperimentID") or row.get("Iteration") or row.get("run_id") or ""
                k_value = row.get("K") or ""

                for metric in margin_metrics:
                    delta = parse_float(row.get(f"Delta{metric}"))
                    empirical = parse_float(row.get(f"Empirical{metric}"))
                    pseudo = parse_float(row.get(f"Pseudo{metric}"))
                    if delta is None:
                        continue
                    records.append(
                        {
                            "web": web,
                            "run_id": run_id,
                            "ecosystem": ecosystem,
                            "metric": metric,
                            "delta": delta,
                            "empirical": empirical,
                            "pseudo": pseudo,
                            "train_ratio": train_ratio,
                            "k": k_value,
                        }
                    )

    summary = {
        "LogFiles": len(files),
        "RawRows": raw_rows,
        "MatchedTrainRatioRows": matched_rows,
        "DeltaRecords": len(records),
        "TrainRatio": train_ratio,
    }
    return records, summary


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


def student_t_cdf(t_statistic: float, df: float) -> float:
    if df <= 0 or math.isnan(t_statistic):
        return math.nan
    if math.isinf(t_statistic):
        return 1.0 if t_statistic > 0 else 0.0
    x = df / (df + t_statistic * t_statistic)
    ib = regularized_betainc(df / 2.0, 0.5, x)
    if t_statistic >= 0:
        value = 1.0 - 0.5 * ib
    else:
        value = 0.5 * ib
    return max(0.0, min(1.0, value))


def student_t_inv_cdf(probability: float, df: float) -> float:
    if not 0.0 < probability < 1.0 or df <= 0:
        return math.nan
    if abs(probability - 0.5) < 1e-15:
        return 0.0
    if probability < 0.5:
        return -student_t_inv_cdf(1.0 - probability, df)

    lo = 0.0
    hi = 1.0
    while student_t_cdf(hi, df) < probability:
        hi *= 2.0
        if hi > 1e6:
            return math.nan

    for _ in range(120):
        mid = (lo + hi) / 2.0
        if student_t_cdf(mid, df) < probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def zero_matrix(rows: int, cols: int) -> List[List[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def solve_linear_system(a: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    n = len(b)
    aug = [list(a[i]) + [float(b[i])] for i in range(n)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix in mixed-effects fit")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_value = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_value

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def inverse_matrix(a: Sequence[Sequence[float]]) -> List[List[float]]:
    n = len(a)
    inv = zero_matrix(n, n)
    for col in range(n):
        unit = [0.0] * n
        unit[col] = 1.0
        solution = solve_linear_system(a, unit)
        for row in range(n):
            inv[row][col] = solution[row]
    return inv


def log_determinant(a: Sequence[Sequence[float]]) -> float:
    n = len(a)
    mat = [list(row) for row in a]
    logdet = 0.0
    sign = 1.0

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(mat[row][col]))
        if abs(mat[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix determinant in mixed-effects fit")
        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]
            sign *= -1.0

        pivot_value = mat[col][col]
        if pivot_value < 0:
            sign *= -1.0
            pivot_value = -pivot_value
        logdet += math.log(pivot_value)

        for row in range(col + 1, n):
            factor = mat[row][col] / mat[col][col]
            for j in range(col + 1, n):
                mat[row][j] -= factor * mat[col][j]

    if sign <= 0:
        raise ValueError("Non-positive determinant in mixed-effects fit")
    return logdet


def fixed_vector(record: Dict[str, object], columns: Sequence[str]) -> List[float]:
    if columns == ["Intercept"]:
        return [1.0]
    ecosystem = str(record["ecosystem"])
    return [1.0 if ecosystem == column else 0.0 for column in columns]


def grouped_sufficient_statistics(
    records: Sequence[Dict[str, object]],
    columns: Sequence[str],
) -> List[Dict[str, object]]:
    groups: Dict[str, List[Tuple[float, List[float]]]] = defaultdict(list)
    for record in records:
        groups[str(record["web"])].append((float(record["delta"]), fixed_vector(record, columns)))

    stats: List[Dict[str, object]] = []
    p = len(columns)
    for web, rows in groups.items():
        n = len(rows)
        sum_x = [0.0] * p
        sum_xy = [0.0] * p
        sum_xx = zero_matrix(p, p)
        sum_y = 0.0
        sum_y2 = 0.0

        for y, x in rows:
            sum_y += y
            sum_y2 += y * y
            for i in range(p):
                sum_x[i] += x[i]
                sum_xy[i] += x[i] * y
                for j in range(p):
                    sum_xx[i][j] += x[i] * x[j]

        stats.append(
            {
                "web": web,
                "n": n,
                "sum_x": sum_x,
                "sum_xy": sum_xy,
                "sum_xx": sum_xx,
                "sum_y": sum_y,
                "sum_y2": sum_y2,
            }
        )
    return stats


def normal_equations_for_lambda(
    group_stats: Sequence[Dict[str, object]],
    p: int,
    lambda_ratio: float,
) -> Tuple[List[List[float]], List[float], float]:
    a = zero_matrix(p, p)
    b = [0.0] * p
    logdet_r = 0.0

    for group in group_stats:
        n = int(group["n"])
        c = lambda_ratio / (1.0 + n * lambda_ratio)
        logdet_r += math.log1p(n * lambda_ratio)

        sum_x = group["sum_x"]  # type: ignore[assignment]
        sum_xy = group["sum_xy"]  # type: ignore[assignment]
        sum_xx = group["sum_xx"]  # type: ignore[assignment]
        sum_y = float(group["sum_y"])

        for i in range(p):
            b[i] += float(sum_xy[i]) - c * float(sum_x[i]) * sum_y
            for j in range(p):
                a[i][j] += float(sum_xx[i][j]) - c * float(sum_x[i]) * float(sum_x[j])

    return a, b, logdet_r


def residual_quadratic_form(
    group_stats: Sequence[Dict[str, object]],
    beta: Sequence[float],
    lambda_ratio: float,
) -> float:
    q = 0.0
    for group in group_stats:
        n = int(group["n"])
        c = lambda_ratio / (1.0 + n * lambda_ratio)
        sum_x = group["sum_x"]  # type: ignore[assignment]
        sum_xy = group["sum_xy"]  # type: ignore[assignment]
        sum_xx = group["sum_xx"]  # type: ignore[assignment]
        sum_y = float(group["sum_y"])
        sum_y2 = float(group["sum_y2"])

        xb_sum = sum(float(sum_x[i]) * beta[i] for i in range(len(beta)))
        y_minus_xb_sum = sum_y - xb_sum

        residual_ss = sum_y2
        residual_ss -= 2.0 * sum(float(sum_xy[i]) * beta[i] for i in range(len(beta)))
        for i in range(len(beta)):
            for j in range(len(beta)):
                residual_ss += beta[i] * float(sum_xx[i][j]) * beta[j]

        q += residual_ss - c * y_minus_xb_sum * y_minus_xb_sum
    return max(q, 0.0)


def fit_for_lambda(
    group_stats: Sequence[Dict[str, object]],
    n_obs: int,
    p: int,
    lambda_ratio: float,
) -> Dict[str, object]:
    a, b, logdet_r = normal_equations_for_lambda(group_stats, p, lambda_ratio)
    beta = solve_linear_system(a, b)
    q = residual_quadratic_form(group_stats, beta, lambda_ratio)
    df_reml = max(n_obs - p, 1)
    sigma2 = q / df_reml
    logdet_a = log_determinant(a)
    objective = df_reml * math.log(max(sigma2, 1e-300)) + logdet_r + logdet_a
    return {
        "beta": beta,
        "a": a,
        "q": q,
        "sigma2": sigma2,
        "objective": objective,
        "logdet_r": logdet_r,
        "logdet_a": logdet_a,
    }


def optimize_lambda(group_stats: Sequence[Dict[str, object]], n_obs: int, p: int) -> Tuple[float, Dict[str, object]]:
    # First perform a broad grid search over log(lambda), then refine using
    # golden-section search around the best grid point.
    grid = [-30.0 + i * 0.5 for i in range(101)]
    evaluated: List[Tuple[float, float]] = []
    best_theta = grid[0]
    best_objective = math.inf

    for theta in grid:
        lambda_ratio = math.exp(theta)
        try:
            fit = fit_for_lambda(group_stats, n_obs, p, lambda_ratio)
        except (ValueError, OverflowError):
            continue
        objective = float(fit["objective"])
        evaluated.append((theta, objective))
        if objective < best_objective:
            best_theta = theta
            best_objective = objective

    if not evaluated:
        raise ValueError("Unable to evaluate mixed-effects profile objective")

    lo = max(-30.0, best_theta - 1.0)
    hi = min(20.0, best_theta + 1.0)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    invphi = 1.0 / phi
    invphi2 = invphi * invphi

    h = hi - lo
    c = lo + invphi2 * h
    d = lo + invphi * h

    def objective_at(theta: float) -> float:
        return float(fit_for_lambda(group_stats, n_obs, p, math.exp(theta))["objective"])

    try:
        yc = objective_at(c)
        yd = objective_at(d)
        for _ in range(80):
            if yc < yd:
                hi = d
                d = c
                yd = yc
                h = invphi * h
                c = lo + invphi2 * h
                yc = objective_at(c)
            else:
                lo = c
                c = d
                yc = yd
                h = invphi * h
                d = lo + invphi * h
                yd = objective_at(d)
            if abs(hi - lo) < 1e-8:
                break
        best_theta = (lo + hi) / 2.0
    except (ValueError, OverflowError):
        # Fall back to the best grid value.
        pass

    lambda_ratio = math.exp(best_theta)
    fit = fit_for_lambda(group_stats, n_obs, p, lambda_ratio)
    return lambda_ratio, fit


def fit_random_intercept_model(records: Sequence[Dict[str, object]], columns: Sequence[str]) -> Dict[str, object]:
    p = len(columns)
    n_obs = len(records)
    group_stats = grouped_sufficient_statistics(records, columns)
    if n_obs <= p:
        raise ValueError("Not enough observations for mixed-effects fit")

    lambda_ratio, fit = optimize_lambda(group_stats, n_obs, p)
    beta = fit["beta"]  # type: ignore[assignment]
    a = fit["a"]  # type: ignore[assignment]
    sigma2 = float(fit["sigma2"])
    covariance = inverse_matrix(a)
    covariance = [[sigma2 * value for value in row] for row in covariance]
    se = [math.sqrt(max(covariance[i][i], 0.0)) for i in range(p)]

    random_variance = lambda_ratio * sigma2
    residual_variance = sigma2
    icc = random_variance / (random_variance + residual_variance) if random_variance + residual_variance > 0 else math.nan

    return {
        "columns": list(columns),
        "beta": list(beta),
        "se": se,
        "covariance": covariance,
        "lambda_ratio": lambda_ratio,
        "random_intercept_variance": random_variance,
        "residual_variance": residual_variance,
        "icc": icc,
        "n_obs": n_obs,
        "n_groups": len(group_stats),
        "approx_df": max(len(group_stats) - p, 1),
        "profile_objective": float(fit["objective"]),
    }


def direction(estimate: float, lower_margin: float, upper_margin: float, equivalent: bool) -> str:
    if equivalent:
        return "equivalent"
    if estimate < lower_margin:
        return "not_equivalent_pseudo_lower"
    if estimate > upper_margin:
        return "not_equivalent_pseudo_higher"
    return "not_equivalent_ci_crosses_margin"


def result_row(
    *,
    metric: str,
    scope: str,
    term: str,
    estimate: float,
    se: float,
    fit: Dict[str, object],
    lower_margin: float,
    upper_margin: float,
    margin_mode: str,
    margin_justification: str,
    num_foodwebs: int,
    num_runs: int,
    num_rows: int,
    train_ratio: float,
    ecosystem: str = "all",
) -> Dict[str, object]:
    approx_df = float(fit["approx_df"])
    t90 = student_t_inv_cdf(0.95, approx_df)
    t95 = student_t_inv_cdf(0.975, approx_df)
    ci90_lower = estimate - t90 * se
    ci90_upper = estimate + t90 * se
    ci95_lower = estimate - t95 * se
    ci95_upper = estimate + t95 * se
    equivalent90 = ci90_lower > lower_margin and ci90_upper < upper_margin
    equivalent95 = ci95_lower > lower_margin and ci95_upper < upper_margin

    t_lower = (estimate - lower_margin) / se if se > 0 else math.inf
    p_lower = 1.0 - student_t_cdf(t_lower, approx_df)
    t_upper = (estimate - upper_margin) / se if se > 0 else -math.inf
    p_upper = student_t_cdf(t_upper, approx_df)
    tost_p = max(p_lower, p_upper)

    return {
        "Metric": metric,
        "Scope": scope,
        "Term": term,
        "EcosystemType": ecosystem,
        "TrainRatio": train_ratio,
        "Model": "delta ~ fixed_effects + (1 | web)",
        "NumRows": num_rows,
        "NumFoodWebs": num_foodwebs,
        "NumRuns": num_runs,
        "EstimatedDelta": estimate,
        "SE": se,
        "ApproxDF": approx_df,
        "CI90Lower": ci90_lower,
        "CI90Upper": ci90_upper,
        "CI95Lower": ci95_lower,
        "CI95Upper": ci95_upper,
        "LowerMargin": lower_margin,
        "UpperMargin": upper_margin,
        "MarginMode": margin_mode,
        "MarginJustification": margin_justification,
        "TLower": t_lower,
        "PLower": p_lower,
        "TUpper": t_upper,
        "PUpper": p_upper,
        "TOSTPValue": tost_p,
        "Equivalent90": int(equivalent90),
        "Equivalent95": int(equivalent95),
        "Direction": direction(estimate, lower_margin, upper_margin, equivalent90),
        "RandomInterceptVariance": fit["random_intercept_variance"],
        "ResidualVariance": fit["residual_variance"],
        "ICC": fit["icc"],
        "LambdaRatio": fit["lambda_ratio"],
        "ProfileObjective": fit["profile_objective"],
    }


def unique_count(records: Sequence[Dict[str, object]], key: str) -> int:
    return len({str(record.get(key, "")) for record in records})


def fit_metric(
    metric: str,
    records: Sequence[Dict[str, object]],
    margin: Dict[str, object],
    train_ratio: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    lower_margin = float(margin["LowerMargin"])
    upper_margin = float(margin["UpperMargin"])
    margin_mode = str(margin.get("MarginMode", "absolute"))
    margin_justification = str(margin.get("Justification", ""))

    global_fit = fit_random_intercept_model(records, ["Intercept"])
    global_row = result_row(
        metric=metric,
        scope="global",
        term="Intercept",
        ecosystem="all",
        estimate=float(global_fit["beta"][0]),  # type: ignore[index]
        se=float(global_fit["se"][0]),  # type: ignore[index]
        fit=global_fit,
        lower_margin=lower_margin,
        upper_margin=upper_margin,
        margin_mode=margin_mode,
        margin_justification=margin_justification,
        num_foodwebs=unique_count(records, "web"),
        num_runs=unique_count(records, "run_id"),
        num_rows=len(records),
        train_ratio=train_ratio,
    )

    ecosystems = ordered_ecosystems(str(record["ecosystem"]) for record in records)
    ecosystem_fit = fit_random_intercept_model(records, ecosystems)
    ecosystem_rows: List[Dict[str, object]] = []
    for idx, ecosystem in enumerate(ecosystems):
        ecosystem_records = [record for record in records if record["ecosystem"] == ecosystem]
        ecosystem_rows.append(
            result_row(
                metric=metric,
                scope="by_ecosystem",
                term=ecosystem,
                ecosystem=ecosystem,
                estimate=float(ecosystem_fit["beta"][idx]),  # type: ignore[index]
                se=float(ecosystem_fit["se"][idx]),  # type: ignore[index]
                fit=ecosystem_fit,
                lower_margin=lower_margin,
                upper_margin=upper_margin,
                margin_mode=margin_mode,
                margin_justification=margin_justification,
                num_foodwebs=unique_count(ecosystem_records, "web"),
                num_runs=unique_count(ecosystem_records, "run_id"),
                num_rows=len(ecosystem_records),
                train_ratio=train_ratio,
            )
        )

    return global_row, ecosystem_rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[MutableMapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def train_ratio_label(train_ratio: float) -> str:
    if abs(train_ratio - round(train_ratio)) <= 1e-9:
        return f"train_ratio_{int(round(train_ratio))}"
    text = f"{train_ratio:.6g}".replace(".", "p")
    return f"train_ratio_{text}"


def main() -> None:
    args = parse_args()
    metadata = read_metadata(args.metadata_file)
    margins = read_margins(args.margins_file, args.core_only)
    records, run_summary = read_delta_records(args.logs_dir, metadata, margins, args.train_ratio)

    records_by_metric: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        records_by_metric[str(record["metric"])].append(record)

    margin_by_metric = {str(row["Metric"]): row for row in margins}
    global_rows: List[Dict[str, object]] = []
    ecosystem_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []

    for margin in margins:
        metric = str(margin["Metric"])
        metric_records = records_by_metric.get(metric, [])
        if not metric_records:
            skipped_rows.append({"Metric": metric, "Reason": "no finite delta records"})
            continue

        try:
            global_row, metric_ecosystem_rows = fit_metric(
                metric, metric_records, margin_by_metric[metric], args.train_ratio
            )
        except Exception as exc:  # keep the script productive across metrics
            skipped_rows.append({"Metric": metric, "Reason": repr(exc)})
            continue

        global_rows.append(global_row)
        ecosystem_rows.extend(metric_ecosystem_rows)

    ratio_label = train_ratio_label(args.train_ratio)
    output_dir = args.output_dir / ratio_label
    file_prefix = f"mixed_effects_delta_{ratio_label}"
    global_file = output_dir / f"{file_prefix}_global.csv"
    ecosystem_file = output_dir / f"{file_prefix}_by_ecosystem.csv"
    margins_file = output_dir / f"mixed_effects_equivalence_margins_{ratio_label}.csv"
    summary_file = output_dir / f"{file_prefix}_run_summary.csv"
    skipped_file = output_dir / f"{file_prefix}_skipped_metrics.csv"

    result_fields = [
        "Metric",
        "Scope",
        "Term",
        "EcosystemType",
        "TrainRatio",
        "Model",
        "NumRows",
        "NumFoodWebs",
        "NumRuns",
        "EstimatedDelta",
        "SE",
        "ApproxDF",
        "CI90Lower",
        "CI90Upper",
        "CI95Lower",
        "CI95Upper",
        "LowerMargin",
        "UpperMargin",
        "MarginMode",
        "MarginJustification",
        "TLower",
        "PLower",
        "TUpper",
        "PUpper",
        "TOSTPValue",
        "Equivalent90",
        "Equivalent95",
        "Direction",
        "RandomInterceptVariance",
        "ResidualVariance",
        "ICC",
        "LambdaRatio",
        "ProfileObjective",
    ]

    margin_fields = ["Metric", "LowerMargin", "UpperMargin", "MarginMode", "Justification"]
    summary_rows = [
        {
            **run_summary,
            "TrainRatioLabel": ratio_label,
            "MetricsConfigured": len(margins),
            "MetricsFitted": len(global_rows),
            "MetricsSkipped": len(skipped_rows),
            "OutputDir": str(output_dir.relative_to(ROOT)),
            "OutputGlobal": str(global_file.relative_to(ROOT)),
            "OutputByEcosystem": str(ecosystem_file.relative_to(ROOT)),
        }
    ]

    write_csv(global_file, result_fields, global_rows)
    write_csv(ecosystem_file, result_fields, ecosystem_rows)
    write_csv(margins_file, margin_fields, margins)
    write_csv(summary_file, list(summary_rows[0].keys()), summary_rows)
    write_csv(skipped_file, ["Metric", "Reason"], skipped_rows)

    print(f"[JorgeMixedEffects] Wrote {len(global_rows)} global rows to {global_file}")
    print(f"[JorgeMixedEffects] Wrote {len(ecosystem_rows)} ecosystem rows to {ecosystem_file}")
    print(f"[JorgeMixedEffects] Wrote {len(margins)} margins to {margins_file}")
    print(f"[JorgeMixedEffects] Wrote run summary to {summary_file}")
    if skipped_rows:
        print(f"[JorgeMixedEffects] Skipped {len(skipped_rows)} metrics; see {skipped_file}")


if __name__ == "__main__":
    main()
