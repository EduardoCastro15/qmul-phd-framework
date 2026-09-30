#!/usr/bin/env python3

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


STATS_DIR = Path(__file__).resolve().parent
if str(STATS_DIR) not in sys.path:
    sys.path.insert(0, str(STATS_DIR))
SCRIPT = STATS_DIR / "build_wlnm_normalized_gv_metrics.py"
SPEC = importlib.util.spec_from_file_location("wlnm_normalized_gv_metrics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NormalizedGVMetricsTests(unittest.TestCase):
    def test_normalized_positive_mean(self):
        self.assertAlmostEqual(
            MODULE.normalized_positive_mean(6, 12, 8, "test"),
            4.0,
        )

    def test_normalized_positive_mean_rejects_zero_links_or_species(self):
        with self.assertRaises(ValueError):
            MODULE.normalized_positive_mean(2, 0, 8, "zero links")
        with self.assertRaises(ValueError):
            MODULE.normalized_positive_mean(2, 8, 0, "zero species")

    def test_derive_observations_calculates_both_athen_metrics(self):
        raw = {
            "Version": "WLNM_dir_neg",
            "Iteration": "1",
            "ExperimentID": "1",
            "Seed": "123",
            "TrainRatio": "60",
            "Threshold": "0.5",
            "K": "10",
            "EmpiricalNumSpecies": "4",
            "EmpiricalLinks": "6",
            "EmpiricalMeanGenerality": "3",
            "EmpiricalMeanVulnerability": "2",
            "PseudoNumSpecies": "4",
            "PseudoLinks": "8",
            "PseudoMeanGenerality": "4",
            "PseudoMeanVulnerability": "3",
        }
        rows = MODULE.derive_observations(
            raw,
            "scenario",
            Path("web_results_random_wlnm_dir_neg.csv"),
            "web",
        )
        by_metric = {row["Metric"]: row for row in rows}
        self.assertEqual(set(by_metric), set(MODULE.NORMALIZED_METRIC_ORDER))
        self.assertAlmostEqual(
            by_metric["MeanNormalizedGeneralityConsumersOnly"]["Value"],
            2.0,
        )
        self.assertAlmostEqual(
            by_metric["MeanNormalizedGeneralityConsumersOnly"]["ReferenceValue"],
            2.0,
        )
        self.assertAlmostEqual(
            by_metric["MeanNormalizedVulnerabilityResourcesOnly"]["Value"],
            1.5,
        )
        self.assertAlmostEqual(
            by_metric["MeanNormalizedVulnerabilityResourcesOnly"]["ReferenceValue"],
            4 / 3,
        )

    def test_tukey_filtering_is_metric_specific(self):
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
                    "Metric": "MeanNormalizedGeneralityConsumersOnly",
                    "MetricLabel": "Mean normalized generality (consumers only)",
                    "MetricFamily": "pseudo_normalized_athen",
                    "Formula": "formula",
                    "ReferenceFormula": "reference formula",
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

    def test_combined_input_contains_all_ratios_and_four_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_summary = directory_path / "retained_foodwebs_by_metric.csv"
            metadata_path = directory_path / "foodweb_metrics_ecosystem.csv"

            source_fields = (
                "Scenario",
                "Foodweb",
                "Version",
                "TrainRatio",
                "Metric",
                "ReferenceValue",
                "MeanAfterTukey",
                "RetainedRunsAfterTukey",
                "OutlierRunsExcluded",
                "MeetsMinimumRetainedRuns",
            )
            with source_summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=source_fields)
                writer.writeheader()
                for train_ratio in (10, 20):
                    for foodweb in ("Web one", "Web two"):
                        for metric in MODULE.BASE_METRIC_MAP:
                            writer.writerow(
                                {
                                    "Scenario": "scenario",
                                    "Foodweb": foodweb,
                                    "Version": "WLNM_dir_neg",
                                    "TrainRatio": train_ratio,
                                    "Metric": metric,
                                    "ReferenceValue": 1.0,
                                    "MeanAfterTukey": 1.1,
                                    "RetainedRunsAfterTukey": 98,
                                    "OutlierRunsExcluded": 2,
                                    "MeetsMinimumRetainedRuns": 1,
                                }
                            )

            with metadata_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("Foodweb", "EcosystemType"),
                )
                writer.writeheader()
                writer.writerow({"Foodweb": "Web one", "EcosystemType": "lakes"})
                writer.writerow({"Foodweb": "Web two", "EcosystemType": "streams"})

            normalized_summaries = []
            for train_ratio in (10, 20):
                for foodweb in ("Web one", "Web two"):
                    for metric in MODULE.NORMALIZED_METRIC_ORDER:
                        normalized_summaries.append(
                            {
                                "Scenario": "scenario",
                                "Foodweb": foodweb,
                                "Version": "WLNM_dir_neg",
                                "TrainRatio": train_ratio,
                                "Metric": metric,
                                "ReferenceValue": 1.0,
                                "MeanAfterTukey": 1.1,
                                "RetainedRunsAfterTukey": 97,
                                "OutlierRunsExcluded": 3,
                                "MeetsMinimumRetainedRuns": 1,
                            }
                        )

            rows = MODULE.build_combined_figure_input(
                source_summary,
                normalized_summaries,
                metadata_path,
                (10, 20),
            )
            self.assertEqual(len(rows), 16)
            self.assertEqual({row["TrainRatio"] for row in rows}, {"10", "20"})
            self.assertEqual({row["Metric"] for row in rows}, set(MODULE.COMBINED_METRIC_ORDER))
            counts = {}
            for row in rows:
                key = (row["TrainRatio"], row["Metric"])
                counts[key] = counts.get(key, 0) + 1
            self.assertTrue(all(count == 2 for count in counts.values()))


if __name__ == "__main__":
    unittest.main()
