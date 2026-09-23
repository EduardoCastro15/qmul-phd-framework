#!/usr/bin/env python3

import importlib.util
import csv
import sys
import tempfile
import unittest
from pathlib import Path


STATS_DIR = Path(__file__).resolve().parent
if str(STATS_DIR) not in sys.path:
    sys.path.insert(0, str(STATS_DIR))
SCRIPT = STATS_DIR / "build_wlnm_ratio_metrics.py"
SPEC = importlib.util.spec_from_file_location("wlnm_ratio_metrics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RatioMetricTests(unittest.TestCase):
    def test_ratio_value(self):
        self.assertAlmostEqual(MODULE.ratio_value("42", "33", "test"), 42 / 33)

    def test_ratio_value_rejects_nonpositive_denominator(self):
        with self.assertRaises(ValueError):
            MODULE.ratio_value(42, 0, "test")

    def test_derived_metrics_are_reciprocal(self):
        raw = {
            "Version": "WLNM_dir_neg",
            "Iteration": "1",
            "ExperimentID": "1",
            "Seed": "123",
            "TrainRatio": "60",
            "Threshold": "0.5",
            "K": "10",
            "EmpiricalMeanGenerality": "42.0348525469",
            "EmpiricalMeanVulnerability": "33.0780590717",
            "PseudoMeanGenerality": "33.2615803815",
            "PseudoMeanVulnerability": "28.5878220141",
        }
        rows, reciprocal_error = MODULE.derive_observations(
            raw, "scenario", Path("web_results_random_wlnm_dir_neg.csv"), "web"
        )
        self.assertEqual(len(rows), 2)
        self.assertLessEqual(reciprocal_error, MODULE.RECIPROCAL_TOLERANCE)
        self.assertAlmostEqual(rows[0]["Value"] * rows[1]["Value"], 1.0)

    def test_tukey_filtering_is_independent_per_ratio_metric(self):
        group = []
        for index, value in enumerate([1.0] * 99 + [10.0], start=1):
            group.append(
                {
                    "Scenario": "scenario",
                    "Foodweb": "web",
                    "Version": "WLNM_dir_neg",
                    "TrainRatio": "60",
                    "Threshold": "0.5",
                    "K": "10",
                    "Metric": "ResourceToConsumerRatio",
                    "MetricLabel": "Resource-to-consumer ratio",
                    "MetricFamily": "pseudo_ratio",
                    "Formula": "PseudoMeanGenerality/PseudoMeanVulnerability",
                    "ReferenceFormula": (
                        "EmpiricalMeanGenerality/EmpiricalMeanVulnerability"
                    ),
                    "Value": value,
                    "ReferenceValue": 1.25,
                    "Iteration": str(index),
                }
            )
        flagged, summary = MODULE.process_retention_group(group, 1.5, 100, 25)
        self.assertEqual(summary["RetainedRunsAfterTukey"], 99)
        self.assertEqual(summary["OutlierRunsExcluded"], 1)
        self.assertTrue(summary["MeetsMinimumRetainedRuns"])
        self.assertEqual(sum(bool(row["Retained"]) for row in flagged), 99)

    def test_foodweb_key_normalization(self):
        self.assertEqual(
            MODULE.normalize_foodweb_key("Chesapeake Bay_tax_mass.mat"),
            "chesapeake bay",
        )

    def test_combined_input_can_use_a_subset_of_foodwebs(self):
        with tempfile.TemporaryDirectory() as directory:
            pairs_path = Path(directory) / "pairs.csv"
            with pairs_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=MODULE.COMBINED_OUTPUT_FIELDS)
                writer.writeheader()
                for metric in MODULE.BASE_METRIC_ORDER:
                    for foodweb in ("Web one", "Web two"):
                        writer.writerow(
                            {
                                "Scenario": "scenario",
                                "Version": "WLNM_dir_neg",
                                "FoodWeb": foodweb,
                                "FW_KEY": foodweb.casefold(),
                                "Ecosystem": "lakes",
                                "Metric": metric,
                                "EmpiricalValue": 1.0,
                                "MeanPseudoAfterFiltering": 1.0,
                                "ValidRunsAfterFiltering": 100,
                                "OutlierRunsRemoved": 0,
                                "Difference": 0.0,
                                "DifferenceDirection": "Equal",
                            }
                        )
            summaries = []
            for definition in MODULE.RATIO_DEFINITIONS:
                summaries.append(
                    {
                        "Scenario": "scenario",
                        "Foodweb": "Web one",
                        "Version": "WLNM_dir_neg",
                        "TrainRatio": "60",
                        "Metric": definition["name"],
                        "ReferenceValue": 1.0,
                        "MeanAfterTukey": 1.0,
                        "RetainedRunsAfterTukey": 100,
                        "OutlierRunsExcluded": 0,
                        "MeetsMinimumRetainedRuns": 1,
                    }
                )
            combined = MODULE.build_combined_figure_input(pairs_path, summaries, 60)
            self.assertEqual(len(combined), 6)
            self.assertEqual({row["FoodWeb"] for row in combined}, {"Web one"})


if __name__ == "__main__":
    unittest.main()
