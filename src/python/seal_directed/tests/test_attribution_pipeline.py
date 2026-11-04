import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
import torch
import torch.nn as nn


SEAL_DIR = Path(__file__).resolve().parents[1]
PYTHON_DIR = SEAL_DIR.parent
for path in (SEAL_DIR, PYTHON_DIR / "pytorch_DGCNN"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from attribution import (
    build_training_backgrounds,
    expected_gradients,
    integrated_gradients_drnl,
)
from attribution_artifacts import (
    validate_attribution_bundle,
    write_attribution_bundle,
)
from graph_artifacts import pack_graphs
from main import Classifier
from nodeattr_schema import feature_names_from_mat, validate_feature_schema
from seal_run_artifacts import (
    canonical_json_bytes,
    file_sha256,
    sparse_to_arrays,
    validate_run_bundle,
    write_run_bundle,
)
from util import GNNGraph, cmd_args


def configure_tiny_classifier(device="cpu"):
    cmd_args.gm = "DGCNN"
    cmd_args.feat_dim = 4
    cmd_args.attr_dim = 44
    cmd_args.edge_feat_dim = 0
    cmd_args.sortpooling_k = 10
    cmd_args.latent_dim = [4, 4, 4, 1]
    cmd_args.hidden = 8
    cmd_args.out_dim = 0
    cmd_args.dropout = False
    cmd_args.num_class = 2
    cmd_args.conv1d_activation = "ReLU"
    graph_nx = nx.Graph()
    graph_nx.add_nodes_from(range(3))
    graph_nx.add_edges_from([(0, 2), (1, 2)])
    attributes = np.random.default_rng(7).normal(size=(3, 44)).astype(np.float32)
    graph = GNNGraph(graph_nx, 1, [1, 2, 3], attributes)
    graph.node_ids = np.asarray([10, 20, 30], dtype=np.int64)
    graph.candidate_pair = np.asarray([10, 20], dtype=np.int64)
    return Classifier().to(torch.device(device)).eval(), graph


class ExplicitForwardTests(unittest.TestCase):
    def test_legacy_and_explicit_forward_are_identical_and_differentiable(self):
        classifier, graph = configure_tiny_classifier()
        legacy = classifier([graph])[0]
        tags = torch.zeros((3, 4), dtype=torch.float32)
        tags.scatter_(1, torch.tensor([1, 2, 3]).view(-1, 1), 1.0)
        attributes = torch.tensor(graph.node_features, requires_grad=True)
        explicit = classifier.forward_prepared([graph], tags, attributes)
        self.assertTrue(torch.equal(legacy, explicit))
        log_odds = classifier.forward_log_odds([graph], tags, attributes)
        log_odds.sum().backward()
        self.assertIsNotNone(attributes.grad)
        self.assertTrue(torch.all(torch.isfinite(attributes.grad)))

        tags = tags.detach().requires_grad_(True)
        classifier.forward_log_odds(
            [graph], tags, torch.tensor(graph.node_features)
        ).sum().backward()
        self.assertIsNotNone(tags.grad)
        self.assertTrue(torch.all(torch.isfinite(tags.grad)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_explicit_gradients_are_finite_on_cuda(self):
        classifier, graph = configure_tiny_classifier("cuda")
        tags = torch.zeros((3, 4), device="cuda")
        tags.scatter_(1, torch.tensor([1, 2, 3], device="cuda").view(-1, 1), 1.0)
        attributes = torch.tensor(graph.node_features, device="cuda", requires_grad=True)
        classifier.forward_log_odds([graph], tags, attributes).sum().backward()
        self.assertTrue(torch.all(torch.isfinite(attributes.grad)))


class ExpectedGradientsTests(unittest.TestCase):
    def test_backgrounds_use_only_supplied_training_graphs_and_role_positions(self):
        train_graphs = []
        for index, nodes in enumerate((3, 4)):
            graph = type("TrainingGraph", (), {})()
            graph.num_nodes = nodes
            graph.node_features = np.full((nodes, 44), 10.0 + index, dtype=np.float32)
            graph.node_features[0] = 100.0 + index
            graph.node_features[1] = 200.0 + index
            train_graphs.append(graph)
        baselines, indices = build_training_backgrounds(
            train_graphs, query_nodes=6, samples=32, rng=np.random.default_rng(11)
        )
        self.assertEqual(baselines.shape, (32, 6, 44))
        self.assertTrue(np.all((indices >= 0) & (indices < len(train_graphs))))
        for baseline, index in zip(baselines, indices):
            self.assertTrue(np.all(baseline[0] == 100.0 + index))
            self.assertTrue(np.all(baseline[1] == 200.0 + index))
            self.assertTrue(np.all(baseline[2:] == 10.0 + index))

    def test_synthetic_known_channel_is_ranked_first(self):
        class KnownChannel(nn.Module):
            def forward(self, values):
                return 3.0 * values[:, :, 17].sum(dim=1)

        inputs = torch.zeros((4, 44), dtype=torch.float32)
        inputs[:, 17] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        baselines = np.zeros((128, 4, 44), dtype=np.float32)
        attribution, diagnostics = expected_gradients(
            KnownChannel(),
            inputs,
            baselines,
            np.random.default_rng(19),
            internal_batch_size=16,
        )
        channel_importance = attribution.abs().sum(dim=0).detach().numpy()
        self.assertEqual(int(np.argmax(channel_importance)), 17)
        self.assertIn(17, np.argsort(channel_importance)[-3:])
        self.assertLess(diagnostics["completeness_relative"], 1e-6)

    @unittest.skipUnless(importlib.util.find_spec("captum"), "Captum is not installed locally")
    def test_drnl_integrated_gradients_is_complete_for_linear_target(self):
        class LinearDrnl(nn.Module):
            def forward(self, values):
                return values.sum(dim=(1, 2))

        values = torch.eye(4, dtype=torch.float32)[:3]
        attribution, diagnostics = integrated_gradients_drnl(
            LinearDrnl(), values, steps=32, internal_batch_size=8
        )
        self.assertTrue(torch.all(torch.isfinite(attribution)))
        self.assertLess(diagnostics["completeness_relative"], 1e-6)


class SchemaAndArtifactTests(unittest.TestCase):
    def test_real_mat_has_frozen_44_feature_schema(self):
        mat_path = next((SEAL_DIR / "data" / "foodwebs_mat_seal_attrs").glob("*.mat"))
        schema = validate_feature_schema(feature_names_from_mat(sio.loadmat(mat_path)))
        self.assertEqual(len(schema["feature_names"]), 44)
        self.assertEqual(len(schema["feature_schema_hash"]), 64)
        self.assertEqual(
            set(schema["feature_families"]),
            {"mass", "mass_missing", "taxonomy", "metabolism", "movement", "life_stage"},
        )

    def test_version_two_training_bundle_fails_closed_after_checkpoint_corruption(self):
        _, graph = configure_tiny_classifier()
        checkpoint = b"portable-checkpoint-test"
        feature_schema = validate_feature_schema(
            feature_names_from_mat(
                sio.loadmat(next((SEAL_DIR / "data" / "foodwebs_mat_seal_attrs").glob("*.mat")))
            )
        )
        feature_names = feature_schema["feature_names"]
        model_config = {
            "schema_version": "seal-directed-model-config-v1",
            "ModelID": "SEAL_directed_nodeattr_v1",
            "DisplayName": "SEAL-D-NodeAttr",
            "LegacyFamily": "SEAL_directed",
            "feature_names": feature_names,
            "feature_schema_hash": feature_schema["feature_schema_hash"],
            "feat_dim": 4,
            "attr_dim": 44,
            "sortpooling_k": 10,
            "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            arrays = {
                "train_pos": np.asarray([[10, 20]], dtype=np.int64),
                "train_neg": np.asarray([[10, 20]], dtype=np.int64),
                "test_pos": np.asarray([[10, 20]], dtype=np.int64),
                "test_neg": np.asarray([[10, 20]], dtype=np.int64),
                "test_labels": np.asarray([1, 0], dtype=np.int8),
                "test_scores": np.asarray([0.8, 0.2], dtype=np.float64),
            }
            train_positive = graph
            train_negative = configure_tiny_classifier()[1]
            train_negative.label = 0
            test_positive = configure_tiny_classifier()[1]
            test_negative = configure_tiny_classifier()[1]
            test_negative.label = 0
            arrays.update(pack_graphs([train_positive, train_negative], "train"))
            arrays.update(pack_graphs([test_positive, test_negative], "test"))
            arrays.update(sparse_to_arrays(ssp.eye(31, format="csr"), "pseudo"))
            write_run_bundle(
                run_dir,
                {
                    "Foodweb": "synthetic",
                    "ModelID": model_config["ModelID"],
                    "DisplayName": model_config["DisplayName"],
                    "LegacyFamily": model_config["LegacyFamily"],
                },
                arrays,
                {"node_order_hash": "synthetic"},
                "a" * 64,
                extra_files={
                    "checkpoint.pt": checkpoint,
                    "model_config.json": canonical_json_bytes(model_config) + b"\n",
                },
            )
            validate_run_bundle(run_dir)
            (run_dir / "checkpoint.pt").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                validate_run_bundle(run_dir)

    def test_attribution_bundle_validates_candidate_offsets_and_training_backgrounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "attribution"
            eg = np.zeros((2, 44), dtype=np.float32)
            eg[0, 3] = 2.0
            eg_norm = np.abs(eg) / np.sum(np.abs(eg))
            arrays = {
                "candidate_pairs": np.asarray([[4, 9]], dtype=np.int64),
                "candidate_labels": np.asarray([1], dtype=np.int8),
                "candidate_scores": np.asarray([0.9]),
                "candidate_log_odds": np.asarray([2.0]),
                "candidate_confusion": np.asarray([3], dtype=np.int8),
                "node_offsets": np.asarray([0, 2], dtype=np.int64),
                "node_ids": np.asarray([4, 9], dtype=np.int64),
                "node_roles": np.asarray([0, 1], dtype=np.int8),
                "expected_gradients": eg,
                "expected_gradients_abs_proportion": eg_norm,
                "drnl_labels": np.asarray([1, 2], dtype=np.int64),
                "drnl_integrated_gradients": np.zeros((2, 4), dtype=np.float32),
                "drnl_active_attribution": np.zeros(2, dtype=np.float32),
                "background_train_indices": np.zeros((1, 32), dtype=np.int64),
                "eg_completeness_absolute": np.zeros(1),
                "eg_completeness_relative": np.zeros(1),
                "eg_output_difference": np.asarray([2.0]),
                "eg_attribution_sum": np.asarray([2.0]),
                "drnl_completeness_absolute": np.zeros(1),
                "drnl_completeness_relative": np.zeros(1),
                "drnl_output_difference": np.zeros(1),
                "drnl_attribution_sum": np.zeros(1),
            }
            metadata = {
                "training_bundle_success_sha256": "a" * 64,
                "checkpoint_sha256": "b" * 64,
                "background_training_graph_count": 1,
            }
            write_attribution_bundle(bundle, metadata, arrays, "c" * 64)
            validate_attribution_bundle(bundle, expected_config_hash="c" * 64)
            arrays["background_train_indices"][0, 0] = 1
            invalid = Path(temporary) / "invalid"
            write_attribution_bundle(invalid, metadata, arrays, "c" * 64)
            with self.assertRaises(ValueError):
                validate_attribution_bundle(invalid)


if __name__ == "__main__":
    unittest.main()
