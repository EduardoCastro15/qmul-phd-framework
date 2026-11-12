from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as ssp
import torch


PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from models import DirectedClassifier, collate_directed
from attribution_artifacts import validate_attribution_bundle, write_attribution_bundle
from subgraphs import extract_one
from weighted_attribution import (
    attribute_test_graphs,
    build_edge_background_pool,
    sample_edge_backgrounds,
    validate_directed_node_edge_invariance,
    validate_explicit_forward_parity,
)


def fixture_graph(candidate=(0, 1), graph_label=1, edge_shift=0.0):
    adjacency = ssp.csr_matrix(
        (np.ones(5), ([0, 1, 1, 2, 3], [2, 0, 3, 1, 2])), shape=(4, 4)
    )
    edges = np.column_stack(adjacency.nonzero())
    attrs = np.arange(len(edges) * 17, dtype=np.float32).reshape(len(edges), 17)
    attrs = attrs + float(edge_shift)
    features = np.arange(4 * 44, dtype=np.float32).reshape(4, 44) / 100.0
    return extract_one(candidate, graph_label, adjacency, features, edges, attrs)


class WeightedAttributionTests(unittest.TestCase):
    def test_forward_prepared_and_log_odds_are_parity_safe(self):
        graph = fixture_graph()
        model = DirectedClassifier(50, 6, "directed_ecc", 10, dropout=False)
        model.eval()
        self.assertEqual(validate_explicit_forward_parity(model, [graph]), 0.0)
        device = next(model.parameters()).device
        packed, edges, edge_attr, sizes, _ = collate_directed([graph], 6, device)
        with torch.no_grad():
            log_prob = model.forward_prepared(
                packed[:, :6], packed[:, 6:], edges, edge_attr, sizes
            )
            log_odds = model.forward_log_odds(
                packed[:, :6], packed[:, 6:], edges, edge_attr, sizes
            )
        torch.testing.assert_close(log_odds, log_prob[:, 1] - log_prob[:, 0])

    def test_ecc_has_node_and_edge_gradients_but_node_variant_is_edge_invariant(self):
        graph = fixture_graph()
        ecc = DirectedClassifier(50, 6, "directed_ecc", 10, dropout=False)
        ecc.eval()
        device = next(ecc.parameters()).device
        packed, edges, edge_attr, sizes, _ = collate_directed([graph], 6, device)
        node_attributes = packed[:, 6:].detach().requires_grad_(True)
        edge_attributes = edge_attr.detach().requires_grad_(True)
        ecc.forward_log_odds(
            packed[:, :6], node_attributes, edges, edge_attributes, sizes
        ).sum().backward()
        self.assertIsNotNone(node_attributes.grad)
        self.assertIsNotNone(edge_attributes.grad)
        self.assertTrue(torch.isfinite(node_attributes.grad).all())
        self.assertTrue(torch.isfinite(edge_attributes.grad).all())

        node = DirectedClassifier(50, 6, "directed_node", 10, dropout=False)
        node.eval()
        self.assertEqual(validate_directed_node_edge_invariance(node, [graph]), 0.0)

    def test_edge_backgrounds_preserve_complete_vectors_and_report_fallback(self):
        train = fixture_graph(edge_shift=1000.0)
        pool = build_edge_background_pool([train])
        # The focal candidate role pair (resource -> consumer) is absent from
        # training subgraphs by construction, so the partial-role tier is used.
        query_edges = np.asarray([[0, 1]], dtype=np.int64)
        backgrounds, tiers = sample_edge_backgrounds(
            pool, query_edges, train.num_nodes, 12, np.random.default_rng(17)
        )
        self.assertEqual(backgrounds.shape, (12, 1, 17))
        self.assertEqual(int(tiers[0]), 1)
        for vector in backgrounds[:, 0]:
            self.assertTrue(any(np.array_equal(vector, row) for row in pool.vectors))

    def test_small_end_to_end_attribution_has_separate_channels(self):
        train_graphs = [fixture_graph((0, 1), 1), fixture_graph((1, 2), 0, 20.0)]
        test_graphs = [fixture_graph((0, 1), 1, 40.0)]
        model = DirectedClassifier(50, 6, "directed_ecc", 10, dropout=False)
        model.eval()
        with torch.no_grad():
            expected = torch.exp(model(test_graphs)[:, 1]).cpu().numpy()
        arrays = attribute_test_graphs(
            model,
            "directed_ecc",
            train_graphs,
            test_graphs,
            expected,
            label_dim=6,
            attribution_seed=123,
            node_samples=2,
            drnl_steps=2,
            edge_samples=2,
            internal_batch_size=2,
            node_feature_families=["mass"] * 2 + ["taxonomy"] * 42,
            edge_feature_families=["mass_ratio"] + ["edge_category"] * 16,
            faithfulness_background_samples=1,
        )
        self.assertEqual(arrays["candidate_scores"].shape, (1,))
        self.assertEqual(arrays["node_expected_gradients"].shape[1], 44)
        self.assertEqual(arrays["drnl_integrated_gradients"].shape[1], 6)
        self.assertEqual(arrays["edge_expected_gradients"].shape[1], 17)
        self.assertEqual(arrays["edge_pool_sha256"].shape, (1,))
        self.assertEqual(arrays["faithfulness_node_top_family"].shape, (1,))
        self.assertEqual(arrays["faithfulness_edge_top_family"].shape, (1,))
        for name in (
            "node_expected_gradients",
            "drnl_integrated_gradients",
            "edge_expected_gradients",
        ):
            self.assertTrue(np.all(np.isfinite(arrays[name])))
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "attribution"
            metadata = {
                "ModelVariant": "directed_ecc",
                "training_bundle_success_sha256": "a" * 64,
                "checkpoint_sha256": "b" * 64,
                "background_training_graph_count": len(train_graphs),
                "node_feature_names": ["feature_{}".format(i) for i in range(44)],
                "edge_feature_names": ["edge_{}".format(i) for i in range(17)],
            }
            write_attribution_bundle(bundle, metadata, arrays, "c" * 64)
            observed = validate_attribution_bundle(bundle, "c" * 64)
            self.assertEqual(observed["ModelVariant"], "directed_ecc")

    def test_directed_node_emits_empty_but_candidate_aligned_edge_channel(self):
        train_graphs = [fixture_graph((0, 1), 1), fixture_graph((1, 2), 0, 20.0)]
        test_graphs = [fixture_graph((0, 1), 1, 40.0)]
        model = DirectedClassifier(50, 6, "directed_node", 10, dropout=False)
        model.eval()
        with torch.no_grad():
            expected = torch.exp(model(test_graphs)[:, 1]).cpu().numpy()
        arrays = attribute_test_graphs(
            model,
            "directed_node",
            train_graphs,
            test_graphs,
            expected,
            label_dim=6,
            attribution_seed=123,
            node_samples=2,
            drnl_steps=2,
            internal_batch_size=2,
        )
        self.assertEqual(arrays["edge_offsets"].tolist(), [0, 0])
        self.assertEqual(arrays["edge_expected_gradients"].shape, (0, 17))
        self.assertEqual(arrays["edge_baseline_seed"].tolist(), [0])
        self.assertEqual(arrays["edge_completeness_absolute"].tolist(), [0.0])


if __name__ == "__main__":
    unittest.main()

