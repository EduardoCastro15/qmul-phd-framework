import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import scipy.sparse as ssp


SEAL_GPU_DIR = Path(__file__).resolve().parents[1]
PYTHON_ROOT = SEAL_GPU_DIR.parent
APOCRITA_DIR = SEAL_GPU_DIR / "apocrita"
for path in (SEAL_GPU_DIR, PYTHON_ROOT, APOCRITA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_seal_original_manifest import build_manifest, experiment_seed
from build_seal_original_apocrita_bundle import collect_files
from run_foodweb_seal import build_command as build_historical_command
from run_foodweb_seal_original_gpu import build_command as build_gpu_wrapper_command
from seal_original_run_artifacts import (
    configuration_hash,
    finalize_run_bundle,
    is_valid_complete_run,
    validate_run_bundle,
)
from util_functions import sample_neg
from validate_seal_original_gpu_run import check_slurm, validate_manifest_protocol


class HistoricalSeedAndManifestTests(unittest.TestCase):
    def test_ten_historical_seeds_are_reused_across_foodwebs(self):
        seeds = [experiment_seed(experiment_id) for experiment_id in range(1, 11)]
        self.assertEqual(seeds, list(range(12345, 12355)))

    def test_manifest_preserves_historical_seed_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo_root = root / "repo"
            for relative in (
                "src/python/Main.py",
                "src/python/util_functions.py",
                "src/python/run_foodweb_seal.py",
            ):
                path = repo_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
            mat_root = root / "mats"
            mat_root.mkdir()
            foodweb_csv = root / "foodwebs.csv"
            with open(foodweb_csv, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Foodweb"])
                writer.writeheader()
                writer.writerows([{"Foodweb": "A"}, {"Foodweb": "B"}])
            (mat_root / "A.mat").write_bytes(b"A")
            (mat_root / "B.mat").write_bytes(b"B")
            _, config_path, rows = build_manifest(
                foodweb_csv,
                mat_root,
                root / "manifest",
                repo_root,
                num_experiments=3,
                expected_foodwebs=2,
            )
            self.assertEqual(len(rows), 6)
            self.assertEqual([int(row["Seed"]) for row in rows], [12345, 12346, 12347] * 2)
            with open(config_path, encoding="utf-8") as handle:
                config = json.load(handle)
            self.assertEqual(config["core_logic_policy"], "exact_historical_behavior_no_corrections")
            self.assertEqual(config["test_ratio"], 1.0 - 0.9)
            self.assertEqual(validate_manifest_protocol(rows, config), [])
            self.assertEqual({row["ConfigHash"] for row in rows}, {configuration_hash(config)})


class CommandEquivalenceTests(unittest.TestCase):
    def _historical_args(self, root, cuda):
        return SimpleNamespace(
            mat_folder=root / "mats",
            result_dir=root / "results",
            hop=1,
            batch_size=50,
            max_train_num=100000,
            num_epochs=50,
            threshold=0.5,
            cuda=cuda,
            no_parallel=False,
            all_unknown_as_negative=False,
            max_nodes_per_hop=None,
            use_embedding=False,
            use_attribute=False,
        )

    def test_cuda_wrapper_command_matches_historical_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "hop": 1,
                "test_ratio": 1.0 - 0.9,
                "batch_size": 50,
                "max_train_num": 100000,
                "num_epochs": 50,
                "threshold": 0.5,
            }
            row = {"Foodweb": "Example_tax_mass", "Seed": "12345", "ExperimentID": "1"}
            historical = build_historical_command(
                self._historical_args(root, cuda=True),
                PYTHON_ROOT,
                row["Foodweb"],
                12345,
                1,
                1.0 - 0.9,
            )
            wrapper = build_gpu_wrapper_command(
                Path(sys.executable),
                PYTHON_ROOT,
                config,
                row,
                root / "results",
                root / "mats" / "Example_tax_mass.mat",
                "cuda",
            )
            self.assertEqual(historical, wrapper)
            for forbidden in (
                "--no-cuda", "--no-parallel", "--use-attribute",
                "--use-embedding", "--all-unknown-as-negative",
            ):
                self.assertNotIn(forbidden, wrapper)

    def test_cpu_and_cuda_commands_only_differ_by_no_cuda(self):
        config = {
            "hop": 1, "test_ratio": 1.0 - 0.9, "batch_size": 50,
            "max_train_num": 100000, "num_epochs": 50, "threshold": 0.5,
        }
        row = {"Foodweb": "A", "Seed": "12345", "ExperimentID": "1"}
        mat_path = Path("/tmp/mats/A.mat")
        cuda = build_gpu_wrapper_command(
            Path(sys.executable), PYTHON_ROOT, config, row, Path("/tmp/out"), mat_path, "cuda"
        )
        cpu = build_gpu_wrapper_command(
            Path(sys.executable), PYTHON_ROOT, config, row, Path("/tmp/out"), mat_path, "cpu"
        )
        self.assertEqual(cpu[:-1], cuda)
        self.assertEqual(cpu[-1], "--no-cuda")


class HistoricalSamplerGoldenTests(unittest.TestCase):
    def test_duplicates_and_reverse_edge_negatives_are_preserved(self):
        adjacency = np.zeros((5, 5), dtype=np.int8)
        for source, target in ((0, 1), (0, 2), (1, 2), (3, 0), (4, 1)):
            adjacency[source, target] = 1
        net = ssp.csr_matrix(adjacency)
        import random
        random.seed(1)
        np.random.seed(1)
        train_pos, train_neg, test_pos, test_neg = sample_neg(
            net, test_ratio=0.34, max_train_num=100000
        )
        self.assertEqual(list(zip(train_pos[0], train_pos[1])), [(0, 1), (1, 2)])
        self.assertEqual(list(zip(test_pos[0], test_pos[1])), [(0, 2)])
        negatives = list(zip(train_neg[0] + test_neg[0], train_neg[1] + test_neg[1]))
        self.assertEqual(negatives, [(0, 3), (0, 3), (0, 3)])
        self.assertEqual(adjacency[3, 0], 1)  # inverse directed edge deliberately exists


class AtomicRunBundleTests(unittest.TestCase):
    def test_bundle_validates_and_rejects_wrong_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            staging.mkdir()
            (staging / "terminal.log").write_text("Namespace(cuda=True)\n", encoding="utf-8")
            with open(staging / "result.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Version", "Seed"])
                writer.writeheader()
                writer.writerow({"Version": "SEAL_undirected", "Seed": 12345})
            run_dir = root / "run"
            config_hash = "a" * 64
            finalize_run_bundle(
                staging,
                run_dir,
                {"Foodweb": "A", "ExperimentID": 1, "Seed": 12345},
                config_hash,
            )
            bundle = validate_run_bundle(run_dir, expected_config_hash=config_hash)
            self.assertEqual(bundle["result"]["Version"], "SEAL_undirected")
            self.assertTrue(is_valid_complete_run(run_dir, config_hash))
            self.assertFalse(is_valid_complete_run(run_dir, "b" * 64))


class SlurmAndDeploymentTests(unittest.TestCase):
    def test_bundle_contains_original_inputs_not_attributed_inputs(self):
        repo_root = PYTHON_ROOT.parents[1]
        files, mat_files = collect_files(repo_root)
        self.assertEqual(len(mat_files), 290)
        self.assertTrue(all(path.parent == repo_root / "src/matlab/data/foodwebs_mat" for path in mat_files))
        self.assertIn(repo_root / "src/python/Main.py", files)
        self.assertNotIn(
            repo_root / "src/python/seal_directed/data/foodwebs_mat_seal_attrs",
            {path.parent for path in mat_files},
        )

    def test_non_array_smoke_job_is_checked(self):
        completed = subprocess.CompletedProcess(
            args=["sacct"], returncode=0,
            stdout="31318196|COMPLETED|0:0\n31318196.batch|COMPLETED|0:0\n",
            stderr="",
        )
        with patch("validate_seal_original_gpu_run.subprocess.run", return_value=completed):
            self.assertEqual(check_slurm("31318196"), {"job": ("COMPLETED", "0:0")})

    def test_submission_is_held_and_uses_validated_gpu_resources(self):
        script = (SEAL_GPU_DIR / "apocrita" / "submit_seal_original_gpu.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('SEAL_GPU_ACCOUNT="${SEAL_GPU_ACCOUNT:-pilot_sae_gpu}"', script)
        self.assertIn('SEAL_GPU_PARTITION="${SEAL_GPU_PARTITION:-sae}"', script)
        self.assertIn('SEAL_GPU_CONSTRAINT="${SEAL_GPU_CONSTRAINT:-ampere}"', script)
        self.assertIn('SEAL_NUM_EXPERIMENTS="${SEAL_NUM_EXPERIMENTS:-10}"', script)
        self.assertIn('SUBMIT_ARGS+=(--hold)', script)
        self.assertIn('ARRAY_ARGS=(--array="1-${CONFIG_EXPERIMENTS}%${SEAL_MAX_CONCURRENT}")', script)
        self.assertIn('--mat-folder "${REPO_ROOT}/src/matlab/data/foodwebs_mat"', script)

    def test_native_library_uses_system_compiler(self):
        script = (SEAL_GPU_DIR / "apocrita" / "create_seal_original_gpu_env.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('make -C "${DGCNN_LIB}" CXX=/usr/bin/g++', script)

    def test_smoke_requests_ampere_gpu(self):
        script = (SEAL_GPU_DIR / "apocrita" / "seal_original_gpushort.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --partition=gpushort", script)
        self.assertIn("#SBATCH --constraint=ampere", script)
        self.assertIn("#SBATCH --gres=gpu:1", script)


if __name__ == "__main__":
    unittest.main()
