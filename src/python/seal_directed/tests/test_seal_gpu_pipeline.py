import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as ssp


SEAL_DIR = Path(__file__).resolve().parents[1]
if str(SEAL_DIR) not in sys.path:
    sys.path.insert(0, str(SEAL_DIR))

from generate_seal_manifest import build_manifest, experiment_seed
from Main_directed import compute_link_prediction_metrics
from run_foodweb_seal_directed import stage_mat_file
from seal_run_artifacts import (
    configuration_hash,
    file_sha256,
    is_valid_complete_run,
    sparse_to_arrays,
    validate_run_bundle,
    write_run_bundle,
)
from util_functions_directed import sample_neg


class SeedAndManifestTests(unittest.TestCase):
    def test_all_full_campaign_seeds_are_unique(self):
        seeds = {
            experiment_seed(foodweb_index, experiment_id)
            for foodweb_index in range(1, 291)
            for experiment_id in range(1, 101)
        }
        self.assertEqual(len(seeds), 29000)
        self.assertEqual(experiment_seed(1, 1), 12345)
        self.assertEqual(experiment_seed(290, 100), 41344)

    def test_manifest_has_expected_cross_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mat_root = root / "mats"
            mat_root.mkdir()
            foodweb_csv = root / "foodwebs.csv"
            with open(foodweb_csv, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Foodweb"])
                writer.writeheader()
                writer.writerows([{"Foodweb": "A"}, {"Foodweb": "B"}])
            (mat_root / "A.mat").write_bytes(b"A")
            (mat_root / "B.mat").write_bytes(b"B")
            manifest_path, config_path, rows = build_manifest(
                foodweb_csv,
                mat_root,
                root / "manifest",
                num_experiments=3,
                expected_foodwebs=2,
            )
            self.assertEqual(len(rows), 6)
            self.assertTrue(manifest_path.is_file())
            with open(config_path, encoding="utf-8") as handle:
                config = json.load(handle)
            self.assertEqual(config["retention"], "raw_no_outlier_filter")
            self.assertEqual({row["ConfigHash"] for row in rows}, {configuration_hash(config)})


class SamplingTests(unittest.TestCase):
    def test_role_pool_fallback_preserves_one_to_one_counts(self):
        net = ssp.csr_matrix(
            (np.ones(2), ([0, 1], [1, 2])),
            shape=(4, 4),
        )
        role_code = np.ones(4, dtype=np.int8)  # consumer-only: constrained source pool is empty
        np.random.seed(17)
        import random
        random.seed(17)
        train_pos, train_neg, test_pos, test_neg, metadata = sample_neg(
            net,
            test_ratio=0.5,
            role_code=role_code,
            use_role_filter=True,
            return_metadata=True,
        )
        self.assertTrue(metadata["role_pool_fallback_used"])
        self.assertEqual(len(train_pos[0]), len(train_neg[0]))
        self.assertEqual(len(test_pos[0]), len(test_neg[0]))
        self.assertFalse(
            set(zip(train_neg[0], train_neg[1])) & set(zip(test_neg[0], test_neg[1]))
        )

    def test_threshold_metrics_use_strict_greater_than(self):
        metrics = compute_link_prediction_metrics(
            labels=np.asarray([1, 0, 1, 0]),
            scores=np.asarray([0.9, 0.8, 0.5, 0.1]),
            threshold=0.5,
        )
        self.assertEqual(metrics["TestTP"], 1)
        self.assertEqual(metrics["TestFP"], 1)
        self.assertEqual(metrics["TestFN"], 1)
        self.assertEqual(metrics["TestTN"], 1)
        self.assertEqual(metrics["F1Score"], 0.5)
        self.assertEqual(metrics["TestMCC"], 0.0)
        self.assertEqual(metrics["TestTSS"], 0.0)


class AtomicArtifactTests(unittest.TestCase):
    def test_input_staging_preserves_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "example.mat"
            source.parent.mkdir()
            source.write_bytes(b"immutable-mat-input")
            expected_hash = file_sha256(source)
            staged = stage_mat_file(source, root / "scratch", expected_hash, 1)
            self.assertNotEqual(staged, source)
            self.assertEqual(file_sha256(staged), expected_hash)
            self.assertEqual(
                stage_mat_file(source, root / "scratch", expected_hash, 1),
                staged,
            )

    def test_atomic_bundle_validates_and_rejects_wrong_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            config_hash = "a" * 64
            arrays = {
                "train_pos": np.asarray([[0, 1]], dtype=np.int64),
                "train_neg": np.asarray([[2, 1]], dtype=np.int64),
                "test_pos": np.asarray([[1, 2]], dtype=np.int64),
                "test_neg": np.asarray([[3, 2]], dtype=np.int64),
                "test_labels": np.asarray([1, 0], dtype=np.int8),
                "test_scores": np.asarray([0.9, 0.1], dtype=np.float64),
                "predicted_links": np.asarray([[1, 2]], dtype=np.int64),
            }
            arrays.update(sparse_to_arrays(ssp.eye(4, format="csr"), prefix="pseudo"))
            write_run_bundle(
                run_dir,
                metrics={"Foodweb": "example", "ExperimentID": 1, "Seed": 12345},
                artifact_arrays=arrays,
                provenance={"split_hash": "example"},
                config_hash=config_hash,
            )
            payload = validate_run_bundle(run_dir, expected_config_hash=config_hash)
            self.assertEqual(payload["metrics"]["Foodweb"], "example")
            self.assertTrue(is_valid_complete_run(run_dir, config_hash))
            self.assertFalse(is_valid_complete_run(run_dir, "b" * 64))


if __name__ == "__main__":
    unittest.main()
