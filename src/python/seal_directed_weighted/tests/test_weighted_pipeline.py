from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
import torch


PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from build_weighted_mats import build_one, validate_weighted_mat
from edge_schema import FEATURE_NAMES, SCHEMA_SHA256, build_edge_vector, preprocess_training_edge_attributes
from generate_manifest import VARIANTS, build_manifest, seed_for
from models import DirectedClassifier, DirectedLayer, EdgeGate
from run_artifacts import configuration_hash, is_valid_complete_run, write_run_bundle
from split_bundles import create_split, load_split_bundle
from subgraphs import extract_one, reverse_graph


class WeightedPipelineTests(unittest.TestCase):
    def test_full_v1_vector_and_duplicate_multihot(self):
        vector = build_edge_vector(2.0, 20.0, [
            {"interaction.type": "predacious", "interaction.dimensionality": "2d", "interaction.classification": "ibi"},
            {"interaction.type": "parasitic", "interaction.dimensionality": "3d", "interaction.classification": "nibi"},
        ])
        self.assertEqual(vector.shape, (17,))
        self.assertAlmostEqual(float(vector[0]), 1.0)
        self.assertEqual(float(vector[3]), 1.0)  # predacious
        self.assertEqual(float(vector[8]), 1.0)  # parasitic
        self.assertEqual(float(vector[11]), 1.0)  # 2d
        self.assertEqual(float(vector[12]), 1.0)  # 3d
        self.assertEqual(float(vector[14]), 1.0)  # ibi
        self.assertEqual(float(vector[15]), 1.0)  # nibi

    def test_missing_values_use_masks_and_unknown_channels(self):
        vector = build_edge_vector(np.nan, np.nan, [])
        self.assertTrue(np.isnan(vector[0]))
        self.assertEqual(vector[1:3].tolist(), [1.0, 1.0])
        self.assertEqual(float(vector[10]), 1.0)
        self.assertEqual(float(vector[13]), 1.0)
        self.assertEqual(float(vector[16]), 1.0)

    def test_training_only_scaler_and_masking(self):
        edges = np.asarray([[0, 1], [1, 2], [2, 0]])
        attrs = np.zeros((3, 17), dtype=np.float32)
        attrs[:, 0] = [0.0, 2.0, 1000.0]
        adjacency = ssp.csr_matrix((np.ones(2), ([0, 1], [1, 2])), shape=(3, 3))
        context_edges, transformed, metadata = preprocess_training_edge_attributes(edges, attrs, adjacency)
        np.testing.assert_array_equal(context_edges, edges[:2])
        np.testing.assert_allclose(transformed[:, 0], [-1.0, 1.0])
        self.assertEqual(metadata["masked_edges"], 1)
        self.assertAlmostEqual(metadata["ratio_mean"], 1.0)

    def test_target_edge_and_covariate_are_removed_reverse_is_retained(self):
        adjacency = ssp.csr_matrix((np.ones(3), ([0, 1, 2], [1, 0, 1])), shape=(3, 3))
        edges = np.asarray([[0, 1], [1, 0], [2, 1]])
        attrs = np.arange(51, dtype=np.float32).reshape(3, 17)
        graph = extract_one((0, 1), 1, adjacency, np.zeros((3, 44), np.float32), edges, attrs)
        local = set(map(tuple, graph.edge_index.tolist()))
        self.assertNotIn((0, 1), local)
        self.assertIn((1, 0), local)
        self.assertEqual(graph.edge_attr.shape[0], 2)

    def test_legacy_weak_fixture_matches_existing_directed_extractor(self):
        directed_root = PACKAGE.parent / "seal_directed"
        if str(directed_root) not in sys.path:
            sys.path.insert(0, str(directed_root))
        from util_functions_directed import subgraph_extraction_labeling

        adjacency = ssp.csr_matrix(
            (np.ones(5), ([0, 1, 1, 2, 3], [1, 0, 2, 3, 1])), shape=(4, 4)
        )
        features = np.arange(4 * 44, dtype=np.float32).reshape(4, 44)
        edges = np.column_stack(adjacency.nonzero())
        attrs = np.zeros((len(edges), 17), dtype=np.float32)
        graph = extract_one((0, 1), 1, adjacency, features, edges, attrs)
        legacy_graph, legacy_labels, legacy_features = subgraph_extraction_labeling(
            (0, 1), adjacency, 1, None, features
        )
        self.assertEqual(graph.node_labels.tolist(), legacy_labels)
        np.testing.assert_array_equal(graph.node_features, legacy_features)
        weak_edges = {tuple(sorted(edge)) for edge in graph.edge_index.tolist()}
        self.assertEqual(weak_edges, {tuple(sorted(edge)) for edge in legacy_graph.edges()})

    def test_reverse_changes_directed_layer_result(self):
        layer = DirectedLayer(2, 2, "node")
        with torch.no_grad():
            layer.self_linear.weight.zero_(); layer.self_linear.bias.zero_()
            layer.in_linear.weight.copy_(torch.eye(2))
            layer.out_linear.weight.copy_(2.0 * torch.eye(2))
        states = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        attrs = torch.zeros((1, 17))
        forward = layer(states, torch.tensor([[0], [1]]), attrs)
        reverse = layer(states, torch.tensor([[1], [0]]), attrs)
        self.assertFalse(torch.allclose(forward, reverse))

    def test_scalar_gate_range_and_gradients(self):
        gate = EdgeGate(17)
        values = gate(torch.randn(9, 17, requires_grad=True))
        self.assertTrue(torch.isfinite(values).all())
        self.assertGreaterEqual(float(values.min().detach()), 0.0)
        self.assertLessEqual(float(values.max().detach()), 1.0)
        values.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in gate.parameters()))

    def test_ecc_chunking_is_numerically_identical_and_differentiable(self):
        first = DirectedLayer(3, 2, "ecc", edge_chunk_size=2)
        second = DirectedLayer(3, 2, "ecc", edge_chunk_size=100)
        second.load_state_dict(first.state_dict())
        states = torch.randn(5, 3, requires_grad=True)
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        attrs = torch.randn(4, 17, requires_grad=True)
        left = first(states, edges, attrs)
        right = second(states, edges, attrs)
        self.assertTrue(torch.allclose(left, right, atol=1e-6, rtol=1e-6))
        left.sum().backward()
        self.assertIsNotNone(states.grad)
        self.assertIsNotNone(attrs.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available locally")
    def test_directed_cpu_cuda_forward_tolerance(self):
        adjacency = ssp.csr_matrix((np.ones(2), ([0, 1], [1, 2])), shape=(3, 3))
        edges = np.column_stack(adjacency.nonzero())
        attrs = np.zeros((2, 17), dtype=np.float32)
        graph = extract_one(
            (0, 1), 1, adjacency, np.zeros((3, 44), np.float32), edges, attrs
        )
        cpu = DirectedClassifier(49, 5, "directed_node", 10)
        gpu = DirectedClassifier(49, 5, "directed_node", 10).cuda()
        gpu.load_state_dict(cpu.state_dict())
        cpu.eval(); gpu.eval()
        with torch.no_grad():
            cpu_result = cpu([graph])
            gpu_result = gpu([graph]).cpu()
        self.assertTrue(torch.allclose(cpu_result, gpu_result, atol=1e-5, rtol=1e-5))

    def test_weighted_mat_exactly_aligns_with_net(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "Web_tax_mass.mat"
            output = root / "weighted.mat"
            net = ssp.csr_matrix((np.ones(2), ([0, 1], [1, 2])), shape=(3, 3))
            sio.savemat(source, {
                "net": net,
                "taxonomy": np.asarray(["A", "B", "C"], dtype=object),
                "group": ssp.csr_matrix(np.zeros((3, 44), dtype=np.float32)),
                "mass": np.asarray([1.0, 2.0, 3.0]),
                "role": np.asarray(["resource", "consumer-resource", "consumer"], dtype=object),
            })
            rows = [
                {"res.taxonomy": "A", "con.taxonomy": "B", "res.mass.mean.g.": "1", "con.mass.mean.g.": "2", "interaction.type": "predacious", "interaction.dimensionality": "2d", "interaction.classification": "ibi"},
                {"res.taxonomy": "B", "con.taxonomy": "C", "res.mass.mean.g.": "2", "con.mass.mean.g.": "3", "interaction.type": "herbivorous", "interaction.dimensionality": "3d", "interaction.classification": "nibi"},
            ]
            build_one(source, rows, output, "a" * 64, overwrite=True)
            self.assertTrue(validate_weighted_mat(output))
            data = sio.loadmat(output, spmatrix=True)
            np.testing.assert_array_equal(data["edge_index"], [[0, 1], [1, 2]])
            self.assertEqual(data["edge_attr_raw"].shape, (2, 17))

    def test_manifest_reuses_each_split_byte_for_byte_across_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mat_root = root / "mats"; mat_root.mkdir()
            mat_path = mat_root / "Fixture_tax_mass.mat"
            net = ssp.csr_matrix((np.ones(4), ([0, 0, 1, 2], [1, 2, 2, 3])), shape=(4, 4))
            sio.savemat(mat_path, {
                "net": net,
                "group": ssp.csr_matrix(np.zeros((4, 44), dtype=np.float32)),
                "taxonomy": np.asarray(["A", "B", "C", "D"], dtype=object),
                "role": np.asarray(["resource", "consumer-resource", "consumer-resource", "consumer"], dtype=object),
                "edge_index": np.column_stack(net.nonzero()),
                "edge_attr_raw": np.zeros((4, 17), dtype=np.float32),
                "edge_attr_schema_sha256": np.asarray([SCHEMA_SHA256], dtype=object),
            })
            foodweb_csv = root / "foodwebs.csv"
            with open(foodweb_csv, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Foodweb"])
                writer.writeheader(); writer.writerow({"Foodweb": "Fixture_tax_mass"})
            manifest_path, _, rows = build_manifest(
                foodweb_csv, mat_root, root / "manifest", num_experiments=2,
                expected_foodwebs=1, frozen_source_commit="fixture",
            )
            self.assertEqual(len(rows), 8)
            for experiment in (1, 2):
                subset = [row for row in rows if row["ExperimentID"] == experiment]
                self.assertEqual({row["ModelVariant"] for row in subset}, set(VARIANTS))
                self.assertEqual(len({row["SplitPath"] for row in subset}), 1)
                self.assertEqual(len({row["SplitHash"] for row in subset}), 1)
                self.assertEqual(len({row["SplitFileSHA256"] for row in subset}), 1)
            self.assertEqual(seed_for(290, 10), 25244)
            self.assertTrue(manifest_path.is_file())

    def test_atomic_bundle_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            config_hash = configuration_hash({"fixture": True})
            arrays = {
                "train_pos": np.asarray([[0, 1]]), "train_neg": np.asarray([[1, 0]]),
                "test_pos": np.asarray([[1, 2]]), "test_neg": np.asarray([[2, 1]]),
                "test_labels": np.asarray([1, 0]), "test_scores": np.asarray([0.8, 0.2]),
            }
            write_run_bundle(run_dir, {"PR_AUC": 1.0}, arrays, {}, config_hash)
            self.assertTrue(is_valid_complete_run(run_dir, config_hash))


if __name__ == "__main__":
    unittest.main()

