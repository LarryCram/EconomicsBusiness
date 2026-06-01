"""Tests for fig_boot_envelope._fractional_ranks and _envelope."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from spectral_results_analysis.fig_boot_envelope import _fractional_ranks, _envelope


# ── _fractional_ranks ─────────────────────────────────────────────────────────

def test_fractional_ranks_basic():
    # 1 replicate, 4 units: v=[3,1,4,2]
    # unit2 (v=4) → rank1 → frac=0.25; unit1 (v=1) → rank4 → frac=1.0
    v = np.array([[3., 1., 4., 2.]], dtype=np.float32)
    frac = _fractional_ranks(v)
    assert frac.shape == (1, 4)
    np.testing.assert_allclose(frac[0], [0.50, 1.00, 0.25, 0.75], atol=1e-6)


def test_fractional_ranks_best_unit_near_zero():
    # Best unit in every replicate should get fractional rank 1/n (smallest possible)
    n = 10
    B = 5
    rng = np.random.default_rng(0)
    v = rng.random((B, n)).astype(np.float32)
    # Force unit 0 to be best in every replicate
    v[:, 0] = 2.0
    frac = _fractional_ranks(v)
    expected_best = 1.0 / n
    np.testing.assert_allclose(frac[:, 0], expected_best, atol=1e-6)


def test_fractional_ranks_worst_unit_is_one():
    # Worst unit in every replicate should get fractional rank 1.0
    n = 10
    B = 5
    rng = np.random.default_rng(1)
    v = rng.random((B, n)).astype(np.float32)
    v[:, 3] = -1.0   # force unit 3 to be worst
    frac = _fractional_ranks(v)
    np.testing.assert_allclose(frac[:, 3], 1.0, atol=1e-6)


def test_fractional_ranks_nan_stays_nan():
    # NaN units (outside SCC) must remain NaN — not assigned a rank
    v = np.array([[3., np.nan, 1., 2.]], dtype=np.float32)
    frac = _fractional_ranks(v)
    assert np.isnan(frac[0, 1])
    # Valid units still get proper fractional ranks
    assert not np.isnan(frac[0, 0])
    assert not np.isnan(frac[0, 2])


def test_fractional_ranks_all_nan():
    # All-NaN replicate: all units stay NaN
    v = np.array([[np.nan, np.nan, np.nan]], dtype=np.float32)
    frac = _fractional_ranks(v)
    assert np.all(np.isnan(frac[0]))


def test_fractional_ranks_values_in_01():
    # All fractional ranks must lie in (0, 1]
    rng = np.random.default_rng(2)
    v = rng.random((20, 50)).astype(np.float32)
    frac = _fractional_ranks(v)
    assert (frac > 0).all()
    assert (frac <= 1).all()


def test_fractional_ranks_no_ties():
    # Each replicate's valid fractional ranks must be a permutation of k/n
    n = 6
    v = np.array([[5., 3., 1., 4., 2., 6.]], dtype=np.float32)
    frac = _fractional_ranks(v)
    expected = np.sort(np.arange(1, n+1) / n)
    np.testing.assert_allclose(np.sort(frac[0]), expected, atol=1e-6)


def test_fractional_ranks_stable_ranking():
    # If the same unit is always rank-1, its median fractional rank → 1/n
    n = 20
    B = 200
    rng = np.random.default_rng(3)
    v = rng.random((B, n)).astype(np.float32)
    v[:, 7] = 10.0   # unit 7 dominates every replicate
    frac = _fractional_ranks(v)
    assert np.median(frac[:, 7]) == pytest.approx(1.0 / n, abs=1e-6)


# ── _envelope ─────────────────────────────────────────────────────────────────

def test_envelope_diagonal_for_stable_ranking():
    # If each unit keeps its rank exactly in every replicate, the envelope
    # median should equal the baseline fractional rank (y = x diagonal).
    n = 10
    B = 50
    # Construct v_boot such that unit ordering is identical every replicate
    v_base = np.arange(n, 0, -1, dtype=np.float32)   # unit 0 best (v=n)
    v_boot = np.tile(v_base, (B, 1))
    frac = _fractional_ranks(v_boot)
    baseline_order = np.argsort(-v_base)
    x, y_lo, y_med, y_hi = _envelope(frac, baseline_order)

    expected_x = (np.arange(1, n+1)) / n
    np.testing.assert_allclose(x,     expected_x, atol=1e-6)
    np.testing.assert_allclose(y_med, expected_x, atol=1e-6)
    np.testing.assert_allclose(y_lo,  expected_x, atol=1e-6)
    np.testing.assert_allclose(y_hi,  expected_x, atol=1e-6)


def test_envelope_x_monotone_unit_interval():
    n = 15
    B = 10
    rng = np.random.default_rng(4)
    v_boot = rng.random((B, n)).astype(np.float32)
    frac = _fractional_ranks(v_boot)
    v_base = rng.random(n).astype(np.float32)
    baseline_order = np.argsort(-v_base)
    x, y_lo, y_med, y_hi = _envelope(frac, baseline_order)

    assert len(x) == n
    assert np.all(np.diff(x) > 0), "x must be strictly increasing"
    assert x[0]  > 0 and x[-1] == pytest.approx(1.0)
    assert np.all(y_lo  >= 0) and np.all(y_lo  <= 1)
    assert np.all(y_hi  >= 0) and np.all(y_hi  <= 1)
    assert np.all(y_lo  <= y_med)
    assert np.all(y_med <= y_hi)


def test_envelope_top_unit_stays_near_top():
    # Unit pinned to rank 1 in all replicates: its envelope should be tight at y≈1/n
    n = 20
    B = 100
    rng = np.random.default_rng(5)
    v_boot = rng.random((B, n)).astype(np.float32)
    v_boot[:, 0] = 100.0   # unit 0 always rank 1
    frac = _fractional_ranks(v_boot)
    v_base = np.zeros(n, dtype=np.float32)
    v_base[0] = 100.0      # unit 0 also baseline rank 1
    baseline_order = np.argsort(-v_base)
    x, y_lo, y_med, y_hi = _envelope(frac, baseline_order)

    # First position in envelope corresponds to the pinned unit
    assert y_med[0] == pytest.approx(1.0 / n, abs=1e-5)
    assert y_lo[0]  == pytest.approx(1.0 / n, abs=1e-5)
    assert y_hi[0]  == pytest.approx(1.0 / n, abs=1e-5)
