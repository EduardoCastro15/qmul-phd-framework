#!/usr/bin/env python3
"""Summarize run timings and Slurm GPU monitor traces without inference."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from models import VARIANTS


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def numeric(value):
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else np.nan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = args.result_root / "summary"
    rows = read_csv(summary / "raw_runs.csv")
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row["ModelVariant"]].append(row)
    timing_rows = []
    for variant, group in sorted(by_variant.items()):
        elapsed = np.asarray([float(row["TimeElapsedSeconds"]) for row in group])
        extraction = np.asarray([float(row["SubgraphExtractionSeconds"]) for row in group])
        training = np.asarray([float(row["TrainingInferenceSeconds"]) for row in group])
        timing_rows.append({
            "ModelVariant": variant,
            "Runs": len(group),
            "MeanTotalSeconds": float(np.mean(elapsed)),
            "MedianTotalSeconds": float(np.median(elapsed)),
            "MeanSubgraphExtractionSeconds": float(np.mean(extraction)),
            "MeanTrainingInferenceSeconds": float(np.mean(training)),
        })
    with open(summary / "runtime_by_variant.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timing_rows[0]))
        writer.writeheader(); writer.writerows(timing_rows)

    with open(args.result_root / "manifest" / "RUN_CONFIG.json", encoding="utf-8") as handle:
        config = json.load(handle)
    experiments = int(config["num_experiments"])
    monitor_rows = []
    for path in sorted((args.result_root / "gpu_monitor").glob("*.csv")):
        trace = read_csv(path)
        if not trace:
            continue
        task_match = re.search(r"_(\d+)\.csv$", path.name)
        task = int(task_match.group(1)) if task_match else None
        variant = VARIANTS[(task - 1) // experiments] if task and task <= experiments * 4 else "unknown"
        utilization_key = next((key for key in trace[0] if "utilization.gpu" in key), None)
        used_key = next((key for key in trace[0] if "memory.used" in key), None)
        total_key = next((key for key in trace[0] if "memory.total" in key), None)
        utilization = np.asarray([numeric(row.get(utilization_key, "")) for row in trace])
        used = np.asarray([numeric(row.get(used_key, "")) for row in trace])
        total = np.asarray([numeric(row.get(total_key, "")) for row in trace])
        monitor_rows.append({
            "MonitorFile": path.name,
            "ArrayTaskID": task,
            "ModelVariant": variant,
            "Samples": len(trace),
            "MeanGPUUtilizationPercent": float(np.nanmean(utilization)),
            "MaxGPUUtilizationPercent": float(np.nanmax(utilization)),
            "MaxGPUMemoryUsedMiB": float(np.nanmax(used)),
            "GPUMemoryTotalMiB": float(np.nanmax(total)),
        })
    if monitor_rows:
        with open(summary / "gpu_monitor_summary.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(monitor_rows[0]))
            writer.writeheader(); writer.writerows(monitor_rows)
    print("[OK] Runtime variants: {}; GPU monitor files: {}".format(len(timing_rows), len(monitor_rows)))


if __name__ == "__main__":
    main()
