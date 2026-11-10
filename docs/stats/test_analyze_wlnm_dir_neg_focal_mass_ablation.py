from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("analyze_wlnm_dir_neg_focal_mass_ablation.py")
SPEC = importlib.util.spec_from_file_location("focal_mass_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normalized_auc_constant_and_linear() -> None:
    ratios = np.arange(10.0, 91.0, 10.0)
    assert np.isclose(MODULE.normalized_auc(ratios, np.full(9, 0.4)), 0.4)
    assert np.isclose(MODULE.normalized_auc(ratios, ratios / 100.0), 0.5)


def test_holm_is_monotone_in_sorted_pvalue_order() -> None:
    raw = [0.04, 0.001, 0.02]
    adjusted = MODULE.holm_adjust(raw)
    assert np.allclose(adjusted, [0.04, 0.003, 0.04])


def test_rank_biserial_direction() -> None:
    assert MODULE.rank_biserial(np.array([1.0, 2.0, 3.0])) == 1.0
    assert MODULE.rank_biserial(np.array([-1.0, -2.0, -3.0])) == -1.0
    assert MODULE.rank_biserial(np.zeros(3)) == 0.0


def test_bootstrap_median_is_reproducible() -> None:
    differences = np.array([0.1, 0.2, 0.3, 0.4])
    first = MODULE.bootstrap_median_ci(differences, seed=123, draws=500)
    second = MODULE.bootstrap_median_ci(differences, seed=123, draws=500)
    assert first == second
