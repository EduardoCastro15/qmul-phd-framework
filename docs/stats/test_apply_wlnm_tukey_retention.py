#!/usr/bin/env python3

import importlib.util
import math
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("apply_wlnm_tukey_retention.py")
SPEC = importlib.util.spec_from_file_location("wlnm_tukey", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TukeyRetentionTests(unittest.TestCase):
    def test_linear_percentiles_and_fences(self):
        fences = MODULE.tukey_fences([1.0, 2.0, 3.0, 4.0], 1.5)
        self.assertAlmostEqual(fences["Q1"], 1.75)
        self.assertAlmostEqual(fences["Q3"], 3.25)
        self.assertAlmostEqual(fences["LowerFence"], -0.5)
        self.assertAlmostEqual(fences["UpperFence"], 5.5)

    def test_zero_iqr_does_not_disable_filtering(self):
        fences = MODULE.tukey_fences([1.0] * 7 + [100.0], 1.5)
        self.assertEqual(fences["IQR"], 0.0)
        self.assertEqual(fences["LowerFence"], 1.0)
        self.assertEqual(fences["UpperFence"], 1.0)

    def test_standard_minimum_is_25_of_50(self):
        group = []
        for index in range(50):
            group.append(
                {
                    "Scenario": "test",
                    "Foodweb": "web",
                    "Version": "WLNM_dir_neg",
                    "TrainRatio": "60",
                    "Threshold": "0.5",
                    "K": "10",
                    "CvK": "0",
                    "Metric": "ROC_AUC",
                    "MetricLabel": "ROC-AUC",
                    "MetricFamily": "predictive",
                    "SourceColumn": "ROC_AUC",
                    "ReferenceColumn": "",
                    "Value": 0.5 + index / 1000,
                    "ReferenceValue": None,
                    "ValidBeforeTukey": True,
                    "InvalidReason": "",
                }
            )
        flagged, summary = MODULE.process_retention_group(
            group,
            {"NumExperiments": "50"},
            False,
            1.5,
            0.5,
        )
        self.assertEqual(summary["MinimumRetainedRuns"], 25)
        self.assertTrue(summary["MeetsMinimumRetainedRuns"])
        self.assertEqual(len(flagged), 50)

    def test_kfold_experiment_is_mean_of_complete_folds(self):
        rows = []
        for fold, value in enumerate((0.6, 0.7, 0.8), start=1):
            rows.append(
                {
                    "Foodweb": "web",
                    "TrainRatio": "66.6666666667",
                    "Threshold": "0.5",
                    "K": "10",
                    "CvK": "3",
                    "Metric": "ROC_AUC",
                    "ExperimentID": "1",
                    "Seed": "123",
                    "Iteration": str(fold),
                    "FoldID": str(fold),
                    "Value": value,
                    "ReferenceValue": None,
                    "ValidBeforeTukey": True,
                    "InvalidReason": "",
                }
            )
        aggregated = MODULE.aggregate_kfold_experiments(rows)
        self.assertEqual(len(aggregated), 1)
        self.assertTrue(aggregated[0]["ValidBeforeTukey"])
        self.assertAlmostEqual(aggregated[0]["Value"], 0.7)
        self.assertEqual(aggregated[0]["FoldCount"], 3)


if __name__ == "__main__":
    unittest.main()
