import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


MODULE_PATH = Path(__file__).resolve().parents[1] / "preprocess_yujie_wlnm.py"
SPEC = importlib.util.spec_from_file_location("preprocess_yujie_wlnm", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class YujiePreprocessingTests(unittest.TestCase):
    def test_gateway_role_rule_uses_positive_complete_network(self):
        # A(prey,predator)=1: r -> cr -> c.  z participates only in an
        # explicit zero and must therefore remain an isolate.
        net = sparse.csc_matrix(
            (
                np.ones(2),
                ([0, 2], [2, 1]),
            ),
            shape=(4, 4),
        )
        self.assertEqual(
            MODULE.derive_roles(net).tolist(),
            ["resource", "consumer", "consumer-resource", "isolate"],
        )

    def test_zero_and_na_masks_remain_disjoint(self):
        frame = pd.DataFrame(
            {
                "prey": ["r", "cr", "z", "r", "unresolved_only"],
                "predator": ["cr", "c", "c", "c", "c"],
                "edge_status_original_grouped": [1.0, 1.0, 0.0, np.nan, np.nan],
            }
        )
        node_order = ["r", "c", "cr", "z", "unresolved_only"]
        source_roles = {
            "r": "prey_only",
            "c": "predator_only",
            "cr": "predator_only",
            "z": "prey_only",
            "unresolved_only": "prey_only",
        }
        mat, row = MODULE.build_network(
            frame,
            "edge_status_original_grouped",
            node_order,
            source_roles,
        )

        # unresolved_only is absent because it occurs only in NA pairs and
        # the local universe is defined by endpoints of resolved 0/1 states.
        self.assertEqual(mat["net"].shape, (4, 4))
        self.assertEqual(mat["net"].nnz, 2)
        self.assertEqual(mat["observed_negative_mask"].nnz, 1)
        self.assertEqual(mat["unresolved_mask"].nnz, 1)
        self.assertEqual(mat["candidate_mask"].nnz, 4)
        self.assertEqual(
            mat["role"].reshape(-1).tolist(),
            ["resource", "consumer", "consumer-resource", "isolate"],
        )
        self.assertEqual(row["n_observed_zero"], 1)
        self.assertFalse(row["eligible_train60"])
        self.assertEqual(row["exclusion_or_eligibility"], "insufficient_2to1_pool")

    def test_one_positive_is_not_a_valid_sixty_percent_split(self):
        self.assertEqual(
            MODULE.exclusive_status(1, 2, 3),
            "insufficient_for_nonempty_train_test",
        )


if __name__ == "__main__":
    unittest.main()
