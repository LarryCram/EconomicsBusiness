"""Tests for the pure MNCS functions in mncs_comparison.py."""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mncs_comparison import (citation_counts, classify, expected_by_group,
                             normalised_scores, unit_means)


def test_expected_is_group_mean():
    c = np.array([1., 3., 10., 0.])
    g = np.array([0, 0, 1, 1])
    assert expected_by_group(c, g, 2).tolist() == [2., 2., 5., 5.]


def test_normalised_scores_average_one_per_group():
    c = np.array([1., 3., 10., 0., 6.])
    g = np.array([0, 0, 1, 1, 1])
    ncs = normalised_scores(c, g, 2)
    assert np.isclose(ncs[g == 0].mean(), 1) and np.isclose(ncs[g == 1].mean(), 1)


def test_uncited_group_scores_zero():
    ncs = normalised_scores(np.array([0., 0.]), np.array([0, 0]), 1)
    assert ncs.tolist() == [0., 0.]


def test_unit_means_fractional_weights():
    ncs = np.array([2., 0., 1.])
    W = sp.csr_matrix(np.array([[1., 1., 0.],      # journal: plain mean of works 0, 1
                                [0.5, 0., 0.25]]))  # institution: fractional weights
    out = unit_means(ncs, W)
    assert np.isclose(out[0], 1.0) and np.isclose(out[1], (1.0 + 0.25) / 0.75)


def test_citation_counts_use_citer_multiplicity():
    cited = np.array([0, 0, 1])
    mult = np.array([2., 0., 1.])        # citer drawn twice, not drawn, once
    assert citation_counts(cited, mult, 3).tolist() == [2., 1., 0.]


def test_classify():
    lo = np.array([-0.1, 0.05, 0.30, -0.5, np.nan])
    hi = np.array([0.1, 0.15, 0.40, -0.45, 0.1])
    assert classify(lo, hi).tolist() == ['in', 'near', 'far', 'far', 'n/a']
