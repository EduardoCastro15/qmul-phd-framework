#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


STATS_DIR = Path(__file__).resolve().parent
if str(STATS_DIR) not in sys.path:
    sys.path.insert(0, str(STATS_DIR))
SCRIPT = STATS_DIR / "build_wlnm_training_normalized_gv_metrics.py"
SPEC = importlib.util.spec_from_file_location(
    "wlnm_training_normalized_gv_metrics", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TrainingNormalizedGVMetricsTests(unittest.TestCase):
    def test_derive_observations_calculates_training_athen_metrics(self):
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
            "TrainNumSpecies": "4",
            "TrainLinks": "8",
            "TrainMeanGenerality": "4",
            "TrainMeanVulnerability": "3",
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
            by_metric["MeanNormalizedGeneralityConsumersOnly"]["Value"], 2.0
        )
        self.assertAlmostEqual(
            by_metric["MeanNormalizedVulnerabilityResourcesOnly"]["Value"], 1.5
        )
        self.assertAlmostEqual(
            by_metric["MeanNormalizedGeneralityConsumersOnly"]["ReferenceValue"],
            2.0,
        )
        self.assertAlmostEqual(
            by_metric["MeanNormalizedVulnerabilityResourcesOnly"]["ReferenceValue"],
            4 / 3,
        )
        self.assertTrue(
            all(row["MetricFamily"] == "training_normalized_athen" for row in rows)
        )

    def test_tukey_filtering_is_independent_for_each_metric(self):
        groups = {}
        for metric in MODULE.NORMALIZED_METRIC_ORDER:
            values = [1.0] * 99 + ([10.0] if "Generality" in metric else [1.0])
            group = []
            for index, value in enumerate(values, start=1):
                group.append(
                    {
                        "Scenario": "scenario",
                        "Foodweb": "web",
                        "Version": "WLNM_dir_neg",
                        "TrainRatio": "60",
                        "Threshold": "0.5",
                        "K": "10",
                        "Metric": metric,
                        "MetricLabel": metric,
                        "MetricFamily": "training_normalized_athen",
                        "Formula": "formula",
                        "ReferenceFormula": "reference formula",
                        "Value": value,
                        "ReferenceValue": 1.25,
                        "Iteration": str(index),
                    }
                )
            _, summary = MODULE.process_retention_group(group, 1.5, 100, 25)
            groups[metric] = summary

        self.assertEqual(
            groups["MeanNormalizedGeneralityConsumersOnly"][
                "RetainedRunsAfterTukey"
            ],
            99,
        )
        self.assertEqual(
            groups["MeanNormalizedVulnerabilityResourcesOnly"][
                "RetainedRunsAfterTukey"
            ],
            100,
        )


if __name__ == "__main__":
    unittest.main()
