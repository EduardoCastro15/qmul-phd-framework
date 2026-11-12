from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from attribution_artifacts import validate_attribution_bundle, write_attribution_bundle
from attribution_manifest_lock import validate_manifest_lock
from generate_attribution_manifest import build_manifest
from run_artifacts import configuration_hash
from run_attribution_campaign import resolve_shard
from run_campaign import resolve_block_task, select_manifest_row
from summarize_attributions import (
    EDGE_FAMILY_ORDER,
    NODE_FAMILY_ORDER,
    ROLE_NAMES,
    align_drnl_widths,
    collapse_feature_families,
)


def attribution_arrays(variant="directed_node", faithfulness=True, edges_override=None):
    edges = 0 if variant == "directed_node" else 1
    if edges_override is not None:
        edges = int(edges_override)
    arrays = {
        "candidate_pairs": np.asarray([[0, 1]], dtype=np.int64),
        "candidate_labels": np.asarray([1], dtype=np.int8),
        "candidate_scores": np.asarray([0.8]),
        "candidate_log_odds": np.asarray([1.0]),
        "candidate_confusion": np.asarray([3], dtype=np.int8),
        "node_offsets": np.asarray([0, 2]),
        "node_ids": np.asarray([0, 1]),
        "node_roles": np.asarray([0, 1], dtype=np.int8),
        "node_feature_values": np.zeros((2, 44), dtype=np.float32),
        "node_expected_gradients": np.zeros((2, 44), dtype=np.float32),
        "node_expected_gradients_abs_proportion": np.zeros((2, 44), dtype=np.float32),
        "node_background_train_indices": np.asarray([[0, 0]], dtype=np.int64),
        "node_completeness_absolute": np.zeros(1),
        "node_completeness_relative": np.zeros(1),
        "node_output_difference": np.zeros(1),
        "node_attribution_sum": np.zeros(1),
        "drnl_labels": np.asarray([1, 2]),
        "drnl_integrated_gradients": np.zeros((2, 4), dtype=np.float32),
        "drnl_integrated_gradients_abs_proportion": np.zeros((2, 4), dtype=np.float32),
        "drnl_active_attribution": np.zeros(2, dtype=np.float32),
        "drnl_completeness_absolute": np.zeros(1),
        "drnl_completeness_relative": np.zeros(1),
        "drnl_output_difference": np.zeros(1),
        "drnl_attribution_sum": np.zeros(1),
        "edge_offsets": np.asarray([0, edges]),
        "edge_index": np.asarray([[1, 0]], dtype=np.int64)[:edges],
        "edge_role_pairs": np.asarray([[1, 0]], dtype=np.int8)[:edges],
        "edge_feature_values": np.zeros((edges, 17), dtype=np.float32),
        "edge_expected_gradients": np.zeros((edges, 17), dtype=np.float32),
        "edge_expected_gradients_abs_proportion": np.zeros((edges, 17), dtype=np.float32),
        "edge_baseline_fallback_tiers": np.zeros(edges, dtype=np.int8),
        "edge_baseline_seed": np.asarray([0], dtype=np.int64),
        "edge_pool_sha256": np.asarray(["a" * 64], dtype="U64"),
        "edge_completeness_absolute": np.zeros(1),
        "edge_completeness_relative": np.zeros(1),
        "edge_output_difference": np.zeros(1),
        "edge_attribution_sum": np.zeros(1),
    }
    if faithfulness:
        arrays.update({
            "faithfulness_node_family_order": np.asarray(["mass"], dtype="U64"),
            "faithfulness_node_top_family": np.asarray([0], dtype=np.int16),
            "faithfulness_node_bottom_family": np.asarray([0], dtype=np.int16),
            "faithfulness_node_top_abs_log_odds_change": np.asarray([0.2]),
            "faithfulness_node_bottom_abs_log_odds_change": np.asarray([0.1]),
            "faithfulness_edge_family_order": np.asarray(
                [] if variant == "directed_node" else ["mass_ratio"], dtype="U64"
            ),
            "faithfulness_edge_top_family": np.asarray(
                [-1 if variant == "directed_node" or edges == 0 else 0], dtype=np.int16
            ),
            "faithfulness_edge_bottom_family": np.asarray(
                [-1 if variant == "directed_node" or edges == 0 else 0], dtype=np.int16
            ),
            "faithfulness_edge_top_abs_log_odds_change": np.asarray([0.0]),
            "faithfulness_edge_bottom_abs_log_odds_change": np.asarray([0.0]),
        })
    return arrays


class AttributionOrchestrationTests(unittest.TestCase):
    def test_manifest_filters_two_variants_and_freezes_new_source_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_config = root / "source.json"
            source_manifest = root / "source.csv"
            original = {
                "source_commit": "historical-commit",
                "num_experiments": 1,
                "model_variants": [
                    "legacy_weak_node", "directed_node",
                    "directed_scalar_gate", "directed_ecc",
                ],
            }
            source_config.write_text(json.dumps(original), encoding="utf-8")
            rows = []
            for variant in original["model_variants"]:
                rows.append({
                    "FoodwebIndex": "1", "Foodweb": "Fixture_tax_mass",
                    "ExperimentID": "1", "Seed": "7", "ModelVariant": variant,
                    "MatPath": "/tmp/fixture.mat", "MatSHA256": "a" * 64,
                    "SplitPath": "/tmp/fixture.npz", "SplitFileSHA256": "b" * 64,
                    "SplitHash": "c" * 64, "RolePoolFallbackUsed": "0",
                    "ConfigHash": configuration_hash(original),
                })
            with open(source_manifest, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            filtered, config, digest = build_manifest(
                source_manifest, source_config, root / "out",
                expected_foodwebs=1, expected_experiments=1,
                source_commit="attribution-commit",
            )
            self.assertEqual(len(filtered), 2)
            self.assertEqual(
                {row["ModelVariant"] for row in filtered},
                {"directed_node", "directed_ecc"},
            )
            self.assertEqual(config["source_commit"], "attribution-commit")
            self.assertEqual(config["historical_source_commit"], "historical-commit")
            self.assertEqual({row["ConfigHash"] for row in filtered}, {digest})
            lock = validate_manifest_lock(
                root / "out/manifest.csv", root / "out/RUN_CONFIG.json"
            )
            self.assertEqual(lock["row_count"], 2)

    def test_manifest_rejects_duplicate_cross_product_and_historical_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_config = root / "source.json"
            source_manifest = root / "source.csv"
            original = {
                "source_commit": "historical-commit",
                "num_experiments": 1,
                "model_variants": [
                    "legacy_weak_node", "directed_node",
                    "directed_scalar_gate", "directed_ecc",
                ],
            }
            source_config.write_text(json.dumps(original), encoding="utf-8")
            base = {
                "FoodwebIndex": "1", "Foodweb": "Fixture_tax_mass",
                "ExperimentID": "1", "Seed": "7",
                "MatPath": "/tmp/fixture.mat", "MatSHA256": "a" * 64,
                "SplitPath": "/tmp/fixture.npz", "SplitFileSHA256": "b" * 64,
                "SplitHash": "c" * 64, "RolePoolFallbackUsed": "0",
                "ConfigHash": configuration_hash(original),
            }
            rows = [
                dict(base, ModelVariant="directed_node"),
                dict(base, ModelVariant="directed_node"),
            ]
            with open(source_manifest, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "duplicate attribution identities"):
                build_manifest(
                    source_manifest, source_config, root / "duplicate",
                    expected_foodwebs=1, expected_experiments=1,
                )

            rows[1] = dict(base, ModelVariant="directed_ecc", ConfigHash="d" * 64)
            with open(source_manifest, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "multiple historical ConfigHash"):
                build_manifest(
                    source_manifest, source_config, root / "hash-mismatch",
                    expected_foodwebs=1, expected_experiments=1,
                )
            rows = [dict(row, ConfigHash="d" * 64) for row in rows]
            with open(source_manifest, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "does not match source RUN_CONFIG"):
                build_manifest(
                    source_manifest, source_config, root / "wrong-shared-hash",
                    expected_foodwebs=1, expected_experiments=1,
                )

    def test_manifest_lock_detects_reordering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"fixture": True}
            config_path = root / "RUN_CONFIG.json"
            manifest_path = root / "manifest.csv"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            rows = []
            for variant in ("directed_node", "directed_ecc"):
                rows.append({
                    "ModelVariant": variant, "ExperimentID": "1",
                    "FoodwebIndex": "1", "Foodweb": "Fixture_tax_mass",
                    "Seed": "7", "SplitHash": "a" * 64,
                })
            with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            from attribution_manifest_lock import write_manifest_lock
            write_manifest_lock(manifest_path, config_path)
            validate_manifest_lock(manifest_path, config_path)
            with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(reversed(rows))
            with self.assertRaisesRegex(ValueError, "manifest lock mismatch"):
                validate_manifest_lock(manifest_path, config_path)

    def test_fine_grained_manifest_row_selection_is_one_based(self):
        rows = [{"id": "first"}, {"id": "second"}]
        self.assertEqual(select_manifest_row(rows, 1)["id"], "first")
        self.assertEqual(select_manifest_row(rows, 2)["id"], "second")
        with self.assertRaises(ValueError):
            select_manifest_row(rows, 0)
        with self.assertRaises(ValueError):
            select_manifest_row(rows, 3)

    def test_compact_block_mapping_stays_below_common_slurm_limit(self):
        config = {
            "model_variants": ["directed_node", "directed_ecc"],
            "num_experiments": 10,
            "num_foodwebs": 290,
        }
        self.assertEqual(
            resolve_block_task(1, config, 6),
            ("directed_node", 1, 0, 49),
        )
        self.assertEqual(
            resolve_block_task(50, config, 6),
            ("directed_node", 2, 0, 49),
        )
        self.assertEqual(
            resolve_block_task(491, config, 6),
            ("directed_ecc", 1, 0, 49),
        )
        self.assertEqual(
            resolve_block_task(980, config, 6),
            ("directed_ecc", 10, 48, 49),
        )
        with self.assertRaises(ValueError):
            resolve_block_task(981, config, 6)

        observed = {
            resolve_block_task(task_id, config, 6)[:3]
            for task_id in range(1, 981)
        }
        expected = {
            (variant, experiment, block)
            for variant in config["model_variants"]
            for experiment in range(1, 11)
            for block in range(49)
        }
        self.assertEqual(observed, expected)

    def test_variant_experiment_array_mapping_has_twenty_tasks(self):
        self.assertEqual(resolve_shard(1, None, None, 10), ("directed_node", 1))
        self.assertEqual(resolve_shard(10, None, None, 10), ("directed_node", 10))
        self.assertEqual(resolve_shard(11, None, None, 10), ("directed_ecc", 1))
        self.assertEqual(resolve_shard(20, None, None, 10), ("directed_ecc", 10))
        with self.assertRaises(ValueError):
            resolve_shard(21, None, None, 10)

    def test_faithfulness_schema_accepts_node_control_and_ecc(self):
        for variant in ("directed_node", "directed_ecc"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                bundle = Path(temporary) / "bundle"
                metadata = {
                    "ModelVariant": variant,
                    "training_bundle_success_sha256": "a" * 64,
                    "checkpoint_sha256": "b" * 64,
                    "background_training_graph_count": 1,
                    "faithfulness_background_samples": 2,
                }
                write_attribution_bundle(
                    bundle, metadata, attribution_arrays(variant), "c" * 64
                )
                validate_attribution_bundle(bundle, "c" * 64)

    def test_faithfulness_schema_accepts_edge_free_ecc_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            metadata = {
                "ModelVariant": "directed_ecc",
                "training_bundle_success_sha256": "a" * 64,
                "checkpoint_sha256": "b" * 64,
                "background_training_graph_count": 1,
                "faithfulness_background_samples": 2,
            }
            write_attribution_bundle(
                bundle, metadata,
                attribution_arrays("directed_ecc", edges_override=0),
                "c" * 64,
            )
            validate_attribution_bundle(bundle, "c" * 64)

    def test_drnl_width_alignment_pads_all_role_views(self):
        def run(width):
            return {
                "drnl_abs": np.ones((2, width)),
                "drnl_signed": np.ones((2, width)),
                "drnl_role_abs": {
                    role: np.ones((2, width)) for role in ROLE_NAMES.values()
                },
                "drnl_role_signed": {
                    role: np.ones((2, width)) for role in ROLE_NAMES.values()
                },
            }
        runs = [run(3), run(5)]
        self.assertEqual(align_drnl_widths(runs), 5)
        for item in runs:
            self.assertEqual(item["drnl_abs"].shape, (2, 5))
            for role in ROLE_NAMES.values():
                self.assertEqual(item["drnl_role_abs"][role].shape, (2, 5))

    def test_feature_families_sum_in_frozen_order(self):
        node_labels = [
            "mass", "mass_missing", "taxonomy", "taxonomy", "metabolism",
            "movement", "life_stage",
        ]
        node = np.asarray([[1, 2, 3, 4, 5, 6, 7]], dtype=np.float32)
        collapsed = collapse_feature_families(
            node, node_labels, NODE_FAMILY_ORDER
        )
        np.testing.assert_array_equal(
            collapsed, np.asarray([[1, 2, 7, 5, 6, 7]], dtype=np.float32)
        )

        edge_labels = [
            "mass_ratio", "mass_missingness", "mass_missingness",
            "interaction_type", "interaction_dimensionality",
            "interaction_classification",
        ]
        edge = np.asarray([[1, 2, 3, 4, 5, 6]], dtype=np.float32)
        np.testing.assert_array_equal(
            collapse_feature_families(edge, edge_labels, EDGE_FAMILY_ORDER),
            np.asarray([[1, 5, 4, 5, 6]], dtype=np.float32),
        )

    def test_feature_family_collapse_rejects_incomplete_schema(self):
        with self.assertRaises(ValueError):
            collapse_feature_families(
                np.ones((1, 1)), ["mass"], NODE_FAMILY_ORDER
            )


if __name__ == "__main__":
    unittest.main()
