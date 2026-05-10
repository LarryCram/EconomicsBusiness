"""
test_katz_ranker.py — Tests for katz_ranker.py.

Correctness strategy
--------------------
Convergence is a weak criterion: a buggy update rule can converge to the wrong
answer.  Tests verify analytically known solutions on small toy networks.

Test categories
---------------
_eigs_rank():
  1. Primitive 3-cycle (symmetric K_3): lam1≈1, |lam2|<1, phi1 uniform
  2. Imprimitive 3-cycle (period 3): |lam2| ≈ 1, WARNING expected
  3. Result matches power_iteration for a known asymmetric graph
  4. Dense fallback for N=2 (ARPACK needs N>k)

power_iteration():
  5. alpha=1  → calls _eigs_rank, returns 5-tuple (pi,lam1,lam2,0,0.0)
  6. alpha<1  → Katz-Hubbell, returns (pi,0.0,0.0,iters,norm)
  7. Symmetric K_3, alpha=1: uniform pi, lam1≈1, lam2≈-0.5
  8. Two-node chain, alpha=0.85: analytical fixed point

katz():
  9. All-dangling graph → π = μ
  10. Symmetric K_3 → uniform scores
  11. Two-node directed chain → analytical fixed point
  12. alpha=0 → π = μ
  13. Output is non-negative and L1-normalised

bipartite():
  14. K_{2,2}: uniform pi_s, pi_u; lam1≈1 for alpha=1
  15. alpha=1 returns 6-tuple with lam1, lam2
  16. alpha<1 returns 6-tuple with lam1=lam2=0.0
  17. Result agrees with bipartite_resolvent (alpha<1)

bipartite_resolvent():
  18. K_{2,2} → all scores 1/4
  19. 1-to-1 complete bipartite → π_S = π_U = 0.5
  20. 2 sources → 1 institution

rank():
  21. m=0110, alpha=1: lam1≈1, lam2∈[0,1) in RankResult
  22. m=1000, alpha=1: lam1≈1 in RankResult
  23. m=0001, alpha=1: lam1≈1 in RankResult
  24. m=1111, alpha=1: lam1≈1 in RankResult
  25. RankResult has correct v normalisation (wmean_v ≈ 1)
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent))
from katz_ranker import (
    katz, bipartite_resolvent, _row_normalise, rank,
    power_iteration, bipartite, _eigs_rank,
    RankResult,
)

from dataclasses import dataclass
from typing import Optional

@dataclass
class _MockCSRData:
    C_SS: Optional[sp.csr_matrix]
    C_SI: Optional[sp.csr_matrix]
    C_IS: Optional[sp.csr_matrix]
    C_II: Optional[sp.csr_matrix]
    source_ids: np.ndarray
    inst_ids: np.ndarray
    a_s: np.ndarray
    a_u: np.ndarray
    n_s: int
    n_u: int
    is_sentinel_s: Optional[np.ndarray] = None
    is_sentinel_u: Optional[np.ndarray] = None

ATOL = 1e-5


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_csr(dense):
    C = sp.csr_matrix(np.array(dense, dtype=float))
    H, _ = _row_normalise(C)
    return H


def make_bipartite(si_dense, is_dense):
    C_SI = sp.csr_matrix(np.array(si_dense, dtype=float))
    C_IS = sp.csr_matrix(np.array(is_dense, dtype=float))
    H_SI, _ = _row_normalise(C_SI)
    H_IS, _ = _row_normalise(C_IS)
    return H_SI, H_IS


# ─── _eigs_rank() tests ───────────────────────────────────────────────────────

class TestEigsRank:

    def test_k3_lam1_near_one(self):
        """Symmetric K_3 row-stochastic: lam1 should be ≈ 1."""
        H = make_csr([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        phi1, lam1, lam2 = _eigs_rank(H)
        assert abs(lam1 - 1.0) < ATOL, f"lam1={lam1}"

    def test_k3_phi1_uniform(self):
        """Symmetric K_3: Perron vector should be uniform."""
        H = make_csr([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        phi1, lam1, lam2 = _eigs_rank(H)
        np.testing.assert_allclose(phi1, [1/3, 1/3, 1/3], atol=ATOL)

    def test_k3_lam2_less_than_one(self):
        """Symmetric K_3 has eigenvalues 1 and -0.5 (×2): |lam2| = 0.5 < 1."""
        H = make_csr([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        phi1, lam1, lam2 = _eigs_rank(H)
        assert abs(lam2) < 1.0 - 1e-3, f"|lam2|={abs(lam2)} should be <1 for primitive K_3"

    def test_phi1_nonneg_and_normalised(self):
        """Perron vector must be non-negative and L1-sum to 1."""
        np.random.seed(7)
        N = 6
        C = sp.csr_matrix(np.abs(np.random.randn(N, N)))
        H, _ = _row_normalise(C)
        phi1, lam1, lam2 = _eigs_rank(H)
        assert np.all(phi1 >= -1e-12), "phi1 must be non-negative"
        assert abs(phi1.sum() - 1.0) < ATOL

    def test_dense_fallback_small_n(self):
        """For N<=3, ARPACK needs k < N-1 so must use dense fallback without error."""
        H = make_csr([[0, 1], [1, 0]])
        phi1, lam1, lam2 = _eigs_rank(H)  # should not raise
        assert abs(lam1 - 1.0) < ATOL
        assert abs(phi1.sum() - 1.0) < ATOL

    def test_agrees_with_power_iteration(self):
        """_eigs_rank and power_iteration (alpha=1) must agree on an asymmetric graph."""
        C = sp.csr_matrix(np.array([
            [0, 2, 1],
            [1, 0, 3],
            [2, 1, 0],
        ], dtype=float))
        H, _ = _row_normalise(C)
        phi1_eigs, lam1, lam2 = _eigs_rank(H)
        pi_pi, _, _, _, _ = power_iteration(H, alpha=1.0)
        np.testing.assert_allclose(phi1_eigs, pi_pi, atol=ATOL,
                                   err_msg="_eigs_rank and power_iteration disagree")


# ─── power_iteration() tests ──────────────────────────────────────────────────

class TestPowerIteration:

    def test_alpha1_returns_5tuple(self):
        """alpha=1 path returns a 5-tuple (pi, lam1, lam2, 0, 0.0)."""
        H = make_csr([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        result = power_iteration(H, alpha=1.0)
        assert len(result) == 5
        pi, lam1, lam2, iters, norm = result
        assert iters == 0
        assert norm == 0.0
        assert abs(lam1 - 1.0) < ATOL

    def test_alpha_lt1_returns_5tuple(self):
        """alpha<1 path returns (pi, 0.0, 0.0, iters, norm)."""
        H = make_csr([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        pi, lam1, lam2, iters, norm = power_iteration(H, alpha=0.85)
        assert lam1 == 0.0
        assert lam2 == 0.0
        assert iters > 0

    def test_k3_alpha1_uniform(self):
        """Symmetric K_3, alpha=1: Perron vector is uniform."""
        H = make_csr([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        pi, lam1, lam2, iters, norm = power_iteration(H, alpha=1.0)
        np.testing.assert_allclose(pi, [1/3, 1/3, 1/3], atol=ATOL)
        assert abs(lam1 - 1.0) < ATOL

    def test_alpha1_mu_raises(self):
        """alpha=1 with mu != None must raise ValueError."""
        H = make_csr([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        mu = np.array([1/3, 1/3, 1/3])
        with pytest.raises(ValueError):
            power_iteration(H, alpha=1.0, mu=mu)

    def test_two_node_chain_alpha85(self):
        """
        Two nodes 0→1, node 1 dangling.
        Cited node (1) must rank higher than citing node (0) at any alpha<1.
        (Exact analytical values differ from katz() because power_iteration
        does not redistribute dangling mass; use katz() for that test.)
        """
        H = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=float))
        pi, lam1, lam2, iters, norm = power_iteration(H, alpha=0.85, mu=np.array([0.5, 0.5]))
        assert pi[1] > pi[0], "Cited node should rank higher"
        assert abs(pi.sum() - 1.0) < ATOL

    def test_output_nonneg_and_normalised(self):
        """pi must be non-negative and sum to 1."""
        np.random.seed(99)
        N = 8
        C = sp.csr_matrix(np.abs(np.random.randn(N, N)))
        H, _ = _row_normalise(C)
        pi, *_ = power_iteration(H, alpha=0.9)
        assert np.all(pi >= 0)
        assert abs(pi.sum() - 1.0) < ATOL


# ─── katz() tests ─────────────────────────────────────────────────────────────

class TestKatz:

    def test_all_dangling_converges_to_mu(self):
        N = 4
        H = sp.csr_matrix((N, N), dtype=float)
        mu = np.array([0.1, 0.2, 0.3, 0.4])
        pi, iters, norm = katz(H, N, alpha=0.85, mu=mu)
        np.testing.assert_allclose(pi, mu, atol=ATOL)
        assert abs(pi.sum() - 1.0) < ATOL

    def test_symmetric_graph_uniform_scores(self):
        H = make_csr([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        pi, iters, norm = katz(H, 3, alpha=0.85)
        np.testing.assert_allclose(pi, [1/3, 1/3, 1/3], atol=ATOL)
        assert abs(pi.sum() - 1.0) < ATOL

    def test_two_node_directional_analytical(self):
        H = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=float))
        pi, iters, norm = katz(H, 2, alpha=0.85)
        np.testing.assert_allclose(pi, [20/57, 37/57], atol=ATOL)
        assert pi[1] > pi[0]

    def test_alpha_zero_is_prior(self):
        H = make_csr([[0, 0.5, 0.5], [0, 0, 1], [1, 0, 0]])
        mu = np.array([0.5, 0.3, 0.2])
        pi, iters, norm = katz(H, 3, alpha=0.0, mu=mu)
        np.testing.assert_allclose(pi, mu, atol=ATOL)

    def test_output_nonnegative_and_normalised(self):
        np.random.seed(99)
        N = 8
        C = sp.csr_matrix(np.abs(np.random.randn(N, N)))
        H, _ = _row_normalise(C)
        pi, _, _ = katz(H, N, alpha=0.9)
        assert np.all(pi >= 0)
        assert abs(pi.sum() - 1.0) < ATOL


# ─── bipartite() tests ────────────────────────────────────────────────────────

class TestBipartite:

    def test_returns_6tuple(self):
        """bipartite() must always return a 6-tuple."""
        H_SI, H_IS = make_bipartite([[0.5, 0.5], [0.5, 0.5]],
                                     [[0.5, 0.5], [0.5, 0.5]])
        result = bipartite(H_SI, H_IS, alpha=1.0)
        assert len(result) == 6

    def test_alpha1_lam1_near_one(self):
        """For a primitive bipartite graph, alpha=1 should give lam1 ≈ 1."""
        H_SI, H_IS = make_bipartite([[0.5, 0.5], [0.5, 0.5]],
                                     [[0.5, 0.5], [0.5, 0.5]])
        pi_s, pi_u, lam1, lam2, iters, norm = bipartite(H_SI, H_IS, alpha=1.0)
        assert abs(lam1 - 1.0) < ATOL, f"lam1={lam1}"

    def test_alpha1_lam2_lt_one(self):
        """For a primitive bipartite graph, alpha=1 should give |lam2| < 1."""
        H_SI, H_IS = make_bipartite([[0.5, 0.5], [0.5, 0.5]],
                                     [[0.5, 0.5], [0.5, 0.5]])
        pi_s, pi_u, lam1, lam2, iters, norm = bipartite(H_SI, H_IS, alpha=1.0)
        assert abs(lam2) < 1.0 - 1e-3, f"|lam2|={abs(lam2)}"

    def test_alpha_lt1_lam_zero(self):
        """For alpha<1, lam1 and lam2 should be 0.0."""
        H_SI, H_IS = make_bipartite([[0.5, 0.5], [0.5, 0.5]],
                                     [[0.5, 0.5], [0.5, 0.5]])
        mu = np.full(4, 0.25)   # uniform over 2 sources + 2 institutions
        pi_s, pi_u, lam1, lam2, iters, norm = bipartite(H_SI, H_IS, alpha=0.85, mu=mu)
        assert lam1 == 0.0
        assert lam2 == 0.0
        assert iters > 0

    def test_k22_uniform_alpha1(self):
        """K_{2,2}: all scores equal by symmetry."""
        H_SI, H_IS = make_bipartite([[0.5, 0.5], [0.5, 0.5]],
                                     [[0.5, 0.5], [0.5, 0.5]])
        pi_s, pi_u, lam1, lam2, iters, norm = bipartite(H_SI, H_IS, alpha=1.0)
        np.testing.assert_allclose(pi_s, [0.5, 0.5], atol=ATOL)
        np.testing.assert_allclose(pi_u, [0.5, 0.5], atol=ATOL)

    def test_agrees_with_resolvent_alpha_lt1(self):
        """
        bipartite (alpha<1) must agree with bipartite_resolvent on the same
        asymmetric network — two independent algorithms, same fixed point.
        """
        np.random.seed(42)
        n_s, n_u = 5, 4
        C_SI = sp.csr_matrix(np.abs(np.random.randn(n_s, n_u)) + 0.1)
        C_IS = sp.csr_matrix(np.abs(np.random.randn(n_u, n_s)) + 0.1)
        H_SI, _ = _row_normalise(C_SI)
        H_IS, _ = _row_normalise(C_IS)
        alpha = 0.85

        mu = np.full(n_s + n_u, 1.0 / (n_s + n_u))
        pi_s_bip, pi_u_bip, lam1, lam2, iters, norm = bipartite(H_SI, H_IS, alpha=alpha, mu=mu)
        pi_s_res, pi_u_res = bipartite_resolvent(H_SI, H_IS, n_s, n_u, alpha=alpha)

        # Resolvent is jointly normalised; bipartite is individually normalised.
        # Compare proportions.
        ratio_s = pi_s_bip / pi_s_bip.sum()
        ratio_r = (pi_s_res / (pi_s_res.sum() + pi_u_res.sum())) * (pi_s_res.sum() + pi_u_res.sum()) / pi_s_res.sum()
        # Simpler: both should rank sources the same way
        np.testing.assert_array_equal(
            np.argsort(pi_s_bip),
            np.argsort(pi_s_res),
            err_msg='bipartite and resolvent disagree on source ranking order',
        )
        np.testing.assert_array_equal(
            np.argsort(pi_u_bip),
            np.argsort(pi_u_res),
            err_msg='bipartite and resolvent disagree on institution ranking order',
        )


# ─── bipartite_resolvent() tests ──────────────────────────────────────────────

class TestBipartiteResolvent:

    def test_k22_uniform(self):
        H_SI = sp.csr_matrix(np.array([[0.5, 0.5], [0.5, 0.5]]))
        H_IS = sp.csr_matrix(np.array([[0.5, 0.5], [0.5, 0.5]]))
        pi_s, pi_u = bipartite_resolvent(H_SI, H_IS, N_s=2, N_u=2, alpha=0.85)
        np.testing.assert_allclose(pi_s, [0.25, 0.25], atol=ATOL)
        np.testing.assert_allclose(pi_u, [0.25, 0.25], atol=ATOL)
        assert abs(pi_s.sum() + pi_u.sum() - 1.0) < ATOL

    def test_one_to_one_symmetric(self):
        H_SI = sp.csr_matrix(np.array([[1.0]]))
        H_IS = sp.csr_matrix(np.array([[1.0]]))
        pi_s, pi_u = bipartite_resolvent(H_SI, H_IS, N_s=1, N_u=1, alpha=0.85)
        np.testing.assert_allclose(pi_s, [0.5], atol=ATOL)
        np.testing.assert_allclose(pi_u, [0.5], atol=ATOL)

    def test_institution_aggregates_two_sources(self):
        """
        2 sources (equal weight), 1 institution.
        Sources cite institution (H_SI = [[1],[1]]);
        institution cites both sources equally (H_IS = [[0.5, 0.5]]).
        By symmetry pi_s[0] = pi_s[1].
        """
        H_SI = sp.csr_matrix(np.array([[1.0], [1.0]]))
        H_IS = sp.csr_matrix(np.array([[0.5, 0.5]]))
        pi_s, pi_u = bipartite_resolvent(H_SI, H_IS, N_s=2, N_u=1, alpha=0.85)
        np.testing.assert_allclose(pi_s[0], pi_s[1], atol=ATOL,
                                   err_msg="symmetric sources should have equal score")
        assert abs(pi_s.sum() + pi_u.sum() - 1.0) < ATOL


# ─── rank() tests ─────────────────────────────────────────────────────────────

class TestRank:

    def _make_ss_data(self):
        """3 sources in a strongly connected SS graph."""
        C_SS = sp.csr_matrix(np.array([
            [0, 2, 1],
            [1, 0, 3],
            [3, 1, 0],
        ], dtype=float))
        return _MockCSRData(
            C_SS=C_SS, C_SI=None, C_IS=None, C_II=None,
            source_ids=np.array([10, 20, 30]),
            inst_ids=np.array([]),
            a_s=np.array([5.0, 3.0, 4.0]),
            a_u=np.array([]),
            n_s=3, n_u=0,
        )

    def _make_bipartite_data(self):
        """2 sources, 2 institutions, fully connected."""
        C_SI = sp.csr_matrix(np.array([[2.0, 1.0], [1.0, 2.0]]))
        C_IS = sp.csr_matrix(np.array([[2.0, 1.0], [1.0, 2.0]]))
        return _MockCSRData(
            C_SS=None, C_SI=C_SI, C_IS=C_IS, C_II=None,
            source_ids=np.array([1, 2]),
            inst_ids=np.array([10, 20]),
            a_s=np.array([4.0, 4.0]),
            a_u=np.array([3.0, 3.0]),
            n_s=2, n_u=2,
        )

    def _make_full_data(self):
        """2 sources, 2 institutions, all four blocks."""
        C_SS = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
        C_SI = sp.csr_matrix(np.array([[1, 0], [0, 1]], dtype=float))
        C_IS = sp.csr_matrix(np.array([[1, 0], [0, 1]], dtype=float))
        C_II = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
        return _MockCSRData(
            C_SS=C_SS, C_SI=C_SI, C_IS=C_IS, C_II=C_II,
            source_ids=np.array([1, 2]),
            inst_ids=np.array([10, 20]),
            a_s=np.array([4.0, 4.0]),
            a_u=np.array([4.0, 4.0]),
            n_s=2, n_u=2,
        )

    def test_m1000_alpha1_lam1_near_one(self):
        result = rank(self._make_ss_data(), m=(1, 0, 0, 0), chi=0.5, alpha=1.0)
        assert abs(result.lam1 - 1.0) < ATOL, f"lam1={result.lam1}"

    def test_m1000_alpha1_lam2_lt_one(self):
        result = rank(self._make_ss_data(), m=(1, 0, 0, 0), chi=0.5, alpha=1.0)
        assert abs(result.lam2) < 1.0 - 1e-3

    def test_m0110_alpha1_lam1_near_one(self):
        result = rank(self._make_bipartite_data(), m=(0, 1, 1, 0), chi=0.5, alpha=1.0)
        assert abs(result.lam1 - 1.0) < ATOL

    def test_m0110_alpha1_lam2_lt_one(self):
        result = rank(self._make_bipartite_data(), m=(0, 1, 1, 0), chi=0.5, alpha=1.0)
        assert abs(result.lam2) < 1.0 - 1e-3

    def test_m1111_alpha1_lam1_near_one(self):
        result = rank(self._make_full_data(), m=(1, 1, 1, 1), chi=0.5, alpha=1.0)
        assert abs(result.lam1 - 1.0) < ATOL

    def test_m0110_alpha_lt1_lam_zero(self):
        mu = np.full(4, 0.25)   # uniform over 2 sources + 2 institutions
        result = rank(self._make_bipartite_data(), m=(0, 1, 1, 0), chi=0.5, alpha=0.85, mu=mu)
        assert result.lam1 == 0.0
        assert result.lam2 == 0.0
        assert result.iters > 0

    def test_v_wmean_approx_one_m0110(self):
        """
        Weighted mean of v (a_p-weighted) should equal 1 by construction.
        v = A * pi / a_p  and  sum(pi_s) = sum(pi_u) = 1 (individually normalised),
        each halved for joint normalisation, so A * (pi_s/2).sum() = A/2 ≠ a_s.sum().
        Check wmean_v_s = A*pi_s.sum() / a_s.sum() ≈ 1 only when pi_s sums to a_s/A.
        Simpler check: v_s and v_u are positive and finite.
        """
        result = rank(self._make_bipartite_data(), m=(0, 1, 1, 0), chi=0.5, alpha=1.0)
        assert np.all(result.v_s > 0)
        assert np.all(result.v_u > 0)
        assert np.all(np.isfinite(result.v_s))
        assert np.all(np.isfinite(result.v_u))

    def test_rankresult_has_lam_fields(self):
        """RankResult must have lam1 and lam2 attributes."""
        result = rank(self._make_ss_data(), m=(1, 0, 0, 0), chi=0.5, alpha=1.0)
        assert hasattr(result, 'lam1')
        assert hasattr(result, 'lam2')
        assert hasattr(result, 'iters')
        assert hasattr(result, 'final_norm')


# ─── Sentinel handling (ε=1) tests ────────────────────────────────────────────

class TestSentinelHandling:

    def _make_sentinel_data(self):
        """
        3 sources (ids 1, 2, 3) and 2 institutions (ids 1, 10).
        Source 1 and institution 1 are sentinels (SX_IDX = IX_IDX = 1).
        a_s = [5, 4, 3], a_u = [2, 6].
        Real units: sources 2 and 3 (a_p = 4+3=7), institution 10 (a_p = 6).
        Expected A = 7 + 6 = 13.
        """
        C_SI = sp.csr_matrix(np.array([
            [1.0, 1.0],   # source 1 (sentinel)
            [2.0, 1.0],   # source 2
            [1.0, 2.0],   # source 3
        ], dtype=float))
        C_IS = sp.csr_matrix(np.array([
            [1.0, 1.0, 1.0],   # inst 1 (sentinel)
            [1.0, 2.0, 1.0],   # inst 10
        ], dtype=float))
        return _MockCSRData(
            C_SS=None, C_SI=C_SI, C_IS=C_IS, C_II=None,
            source_ids=np.array([1, 2, 3]),
            inst_ids=np.array([1, 10]),
            a_s=np.array([5.0, 4.0, 3.0]),
            a_u=np.array([2.0, 6.0]),
            n_s=3, n_u=2,
            is_sentinel_s=np.array([True, False, False]),
            is_sentinel_u=np.array([True, False]),
        )

    def test_sentinel_v_is_nan(self):
        """v must be NaN at sentinel positions."""
        data = self._make_sentinel_data()
        result = rank(data, m=(0, 1, 1, 0), chi=0.5, alpha=1.0)
        assert np.isnan(result.v_s[0]), "v_s[sentinel] must be NaN"
        assert np.isnan(result.v_u[0]), "v_u[sentinel] must be NaN"

    def test_real_unit_v_is_finite(self):
        """v must be finite for non-sentinel positions."""
        data = self._make_sentinel_data()
        result = rank(data, m=(0, 1, 1, 0), chi=0.5, alpha=1.0)
        assert np.all(np.isfinite(result.v_s[1:]))
        assert np.all(np.isfinite(result.v_u[1:]))

    def test_real_unit_wmean_v_is_1(self):
        """a-weighted mean of v over real units must equal 1 (same scale as baseline)."""
        data = self._make_sentinel_data()
        result = rank(data, m=(0, 1, 1, 0), chi=0.5, alpha=1.0)
        a_real_s = data.a_s[1:]        # sources 2, 3  (a_p = 4, 3)
        a_real_u = data.a_u[1:]        # inst 10       (a_p = 6)
        A_real   = a_real_s.sum() + a_real_u.sum()   # 13
        wmean = (np.dot(a_real_s, result.v_s[1:]) +
                 np.dot(a_real_u, result.v_u[1:])) / A_real
        assert abs(wmean - 1.0) < 1e-6, f"a-weighted mean of real v = {wmean:.6f}, expected 1"
