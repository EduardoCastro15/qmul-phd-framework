import unittest

from compare_trophic_level_protocols import summarize


def rows(n=100):
    return [{"Foodweb": "Ythan Estuary_tax_mass", "TrainRatio": "60", "ExperimentID": str(i),
             "PseudoTrophicV2LegacyMean": "2", "PseudoTrophicV2LegacyStatusCode": "0",
             "PseudoNetworkXMeanTrophicLevel": "2", "PseudoNetworkXTrophicLevelStatusCode": "0",
             "PseudoNetworkXTrophicLevelNumSpeciesLargest": "10",
             "EmpiricalTrophicV2LegacyMean": "2", "EmpiricalTrophicV2LegacyStatusCode": "0",
             "EmpiricalNetworkXMeanTrophicLevel": "2", "EmpiricalNetworkXTrophicLevelStatusCode": "0",
             "EmpiricalNetworkXTrophicLevelNumSpeciesLargest": "10"}
            for i in range(1, n+1)]


class ComparisonTests(unittest.TestCase):
    def test_partial_pilot_cannot_qualify(self):
        for summary in summarize(rows(30), expected_runs=100):
            self.assertFalse(summary["EligibleForPairedAnalysis"])
            self.assertIsNone(summary["MeanAfterTukey"])

    def test_invalid_runs_and_tukey_are_separate(self):
        data = rows()
        for row in data[:74]:
            row["PseudoTrophicV2LegacyMean"] = "nan"
            row["PseudoTrophicV2LegacyStatusCode"] = "2"
        data[74]["PseudoTrophicV2LegacyMean"] = "200"
        before, after = summarize(data, expected_runs=100)
        self.assertEqual(before["ValidBeforeTukey"], 26)
        self.assertEqual(before["RetainedAfterTukey"], 25)
        self.assertTrue(before["EligibleForPairedAnalysis"])
        self.assertEqual(after["RetainedAfterTukey"], 100)
        data[75]["PseudoTrophicV2LegacyMean"] = "200"
        self.assertFalse(summarize(data, expected_runs=100)[0]["EligibleForPairedAnalysis"])

    def test_invalid_reference_excludes_pair(self):
        data = rows()
        data[0]["EmpiricalNetworkXTrophicLevelStatusCode"] = "7"
        before, after = summarize(data, expected_runs=100)
        self.assertEqual(before["ValidBeforeTukey"], 100)
        self.assertEqual(after["ValidBeforeTukey"], 99)

    def test_historical_50_run_manifest_remains_supported(self):
        for summary in summarize(rows(50), expected_runs=50):
            self.assertTrue(summary["EligibleForPairedAnalysis"])


if __name__ == "__main__":
    unittest.main()
