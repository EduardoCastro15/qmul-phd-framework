#!/usr/bin/env python3
"""Compare v1/v2 on the SAME saved WLNM_dir_neg reconstructions; no training.

The existing Tukey fence implementation is reused. Pilot-only output must not
be substituted directly into the historical Wilcoxon input.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from apply_wlnm_tukey_retention import tukey_fences

TARGETS = {"Dutch Microfauna food web PlotB_tax_mass", "Ythan Estuary_tax_mass"}


def number(row, key):
    try:
        return float(row[key])
    except (ValueError, TypeError):
        return math.nan


def valid(row, prefix, legacy=False):
    if legacy:
        mean = number(row, prefix + "TrophicV2LegacyMean")
        status = number(row, prefix + "TrophicV2LegacyStatusCode")
    else:
        mean = number(row, prefix + "NetworkXMeanTrophicLevel")
        status = number(row, prefix + "NetworkXTrophicLevelStatusCode")
    # Component selection is identical for the two solvers.
    count = number(row, prefix + "NetworkXTrophicLevelNumSpeciesLargest")
    return math.isfinite(mean) and status == 0 and count >= 2


def summarize(rows, expected_runs, minimum_retained=25):
    """Summarize one food web / K / train ratio / threshold condition."""
    summaries = []
    for legacy, protocol in [(True, "legacy_v1"), (False, "validated_v2")]:
        field = "PseudoTrophicV2LegacyMean" if legacy else "PseudoNetworkXMeanTrophicLevel"
        values = [number(r, field) for r in rows if valid(r, "Pseudo", legacy) and valid(r, "Empirical", legacy)]
        fences = tukey_fences(values, 1.5) if values else {"LowerFence": math.nan, "UpperFence": math.nan}
        retained = [v for v in values if fences["LowerFence"] <= v <= fences["UpperFence"]]
        complete = {int(float(r["ExperimentID"])) for r in rows} == set(range(1, expected_runs + 1))
        accepted = complete and len(retained) >= minimum_retained
        summaries.append({
            "Foodweb": rows[0]["Foodweb"], "TrainRatio": rows[0]["TrainRatio"],
            "Protocol": protocol, "ObservedRuns": len(rows), "ExpectedRuns": expected_runs,
            "CompletePilot": complete, "ValidBeforeTukey": len(values),
            "RetainedAfterTukey": len(retained), "MinimumRetained": minimum_retained,
            "EligibleForPairedAnalysis": accepted,
            "MeanAfterTukey": sum(retained) / len(retained) if accepted else None,
            **fences,
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_root.resolve()
    manifest = json.loads((root / "pilot_manifest.json").read_text())
    if manifest["protocol"] != "validated_v2" or manifest["status"] != "completed":
        raise ValueError("Expected a completed isolated v2 pilot.")
    expected_runs = int(manifest["config"]["numExperiments"])
    output = root / "trophic_protocol_comparison"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    summaries, transitions, observed = [], [], set()
    for path in sorted((root / "prediction_scores_logs").glob("*.csv")):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Empty pilot CSV: {path}")
        name = rows[0]["Foodweb"]
        if name not in TARGETS or name in observed:
            raise ValueError(f"Unexpected or duplicate food web: {name}")
        observed.add(name)
        ids = [number(r, "ExperimentID") for r in rows]
        if len(ids) != len(set(ids)) or any(i not in range(1, expected_runs + 1) for i in ids):
            raise ValueError("Duplicate or invalid experiment IDs")
        for row in rows:
            if (row["Foodweb"] != name or row["Version"] != "WLNM_dir_neg"
                    or row["TrophicLevelProtocol"] != "validated_v2"
                    or number(row, "TrainRatio") != 60 or number(row, "K") != 10
                    or number(row, "Threshold") != 0.5):
                raise ValueError("Mixed pilot conditions")
            if not Path(row["TrophicSnapshotFile"]).is_file():
                raise ValueError("Missing reconstruction snapshot")
        summaries.extend(summarize(rows, expected_runs, minimum_retained=25))
        counts = Counter((r["PseudoTrophicV2LegacyStatusCode"],
                          r["PseudoNetworkXTrophicLevelStatusCode"],
                          r["PseudoTrophicV2FailureReason"]) for r in rows)
        transitions.extend({"Foodweb": name, "LegacyStatus": old, "V2Status": new,
                            "V2Reason": reason, "Runs": count}
                           for (old, new, reason), count in sorted(counts.items()))
    if observed != TARGETS:
        raise ValueError("Both authorized food webs must be present")
    output.mkdir()
    for filename, records in [("retention_comparison.csv", summaries), ("status_transitions.csv", transitions)]:
        fields = list(dict.fromkeys(key for record in records for key in record))
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(summaries, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
