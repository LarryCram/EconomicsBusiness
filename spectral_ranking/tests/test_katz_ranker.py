"""
test_katz_ranker.py — TDD tests for katz() and bipartite_resolvent().

Correctness strategy
--------------------
Convergence of power iteration is a weak criterion: a buggy update rule can
still converge, just to the wrong answer.  These tests instead verify that
the algorithms return analytically known solutions on small toy networks.

Test categories
---------------
katz():
  1. All-dangling graph → π = μ (prior entirely determines scores)
  2. Symmetric K_3 → uniform scores (1/3 each)
  3. Two-node directed chain → analytical fixed point (exact fractions)
  4. alpha=0 → π = μ regardless of H
  5. Output is non-negative and L1-normalised

bipartite_resolvent():
  6.  K_{2,2} → all scores 1/4 (fully symmetric bipartite)
  7.  1-to-1 complete bipartite → π_S = π_U = 0.5 each
  8.  2 sources → 1 institution: institution aggregates, gets highest score
      (analytical solution derived in docstring)
  9.  Dangling source receives only prior score; cited source ranks higher
  10. Random network: output is non-negative and jointly normalised

Consistency (key correctness test):
  11. bipartite_resolvent must agree with katz on the assembled block matrix.
      Two independent algorithms, same result → catches implementation errors
      in either method.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent))
from katz_ranker import katz, bipartite_resolvent, _row_normalise, rank

# Minimal CSRData stand-in for rank() tests
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

ATOL = 1e-6   # agreement tolerance for analytical comparisons
ATOL_CROSS = 1e-6   # agreement between katz and bipartite_resolvent


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_csr(dense):
    """Dense array → row-normalised CSR (as H, ready for katz)."""
    C = sp.csr_matrix(np.array(dense, dtype=float))
    H, _ = _row_normalise(C)
    return H


# ─── katz() tests ─────────────────────────────────────────────────────────────

class TestKatz:

    def test_all_dangling_converges_to_mu(self):
        """H = 0 (all dangling): π must equal the prior μ exactly."""
        N = 4
        H = sp.csr_matrix((N, N), dtype=float)
        mu = np.array([0.1, 0.2, 0.3, 0.4])
        pi, iters, norm = katz(H, N, alpha=0.85, mu=mu)
        np.testing.assert_allclose(pi, mu, atol=ATOL,
            err_msg="All-dangling graph should converge to prior μ")
        assert abs(pi.sum() - 1.0) < ATOL

    def test_symmetric_graph_uniform_scores(self):
        """
        Complete undirected K_3 with equal weights.
        By symmetry every node must receive score 1/3.
        """
        A = [[0, 1, 1],
             [1, 0, 1],
             [1, 1, 0]]
        H = make_csr(A)
        pi, iters, norm = katz(H, 3, alpha=0.85)
        np.testing.assert_allclose(pi, [1/3, 1/3, 1/3], atol=ATOL,
            err_msg="Symmetric K_3 should give equal scores")
        assert abs(pi.sum() - 1.0) < ATOL

    def test_two_node_directional_analytical(self):
        """
        Two nodes: 0 → 1 (H[0,1]=1), node 1 is dangling.
        μ = [0.5, 0.5], α = 0.85.

        Fixed-point derivation
        ----------------------
        Let π = [a, b], dangling_idx = {1}.
        a = α·0 + (α·b + (1−α))·0.5  →  a = 0.425b + 0.075
        b = α·a + (α·b + (1−α))·0.5  →  b = 0.85a + 0.425b + 0.075
        Solving: b·(1−0.78625) = 0.13875  →  b = 37/57, a = 20/57.
        """
        H = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=float))
        alpha = 0.85
        pi, iters, norm = katz(H, 2, alpha=alpha)

        b_exact = 37.0 / 57.0   # cited node (index 1)
        a_exact = 20.0 / 57.0   # citing node (index 0)
        np.testing.assert_allclose(pi, [a_exact, b_exact], atol=ATOL,
            err_msg="Two-node chain: analytical fixed point not matched")
        assert pi[1] > pi[0], "Cited node should rank higher than citing node"
        assert abs(pi.sum() - 1.0) < ATOL

    def test_alpha_zero_is_prior(self):
        """With α=0, H has no influence; π must equal μ after one step."""
        A = [[0, 0.5, 0.5],
             [0, 0,   1  ],
             [1, 0,   0  ]]
        H = make_csr(A)
        mu = np.array([0.5, 0.3, 0.2])
        pi, iters, norm = katz(H, 3, alpha=0.0, mu=mu)
        np.testing.assert_allclose(pi, mu, atol=ATOL,
            err_msg="alpha=0 should return prior μ regardless of H")

    def test_output_nonnegative_and_normalised(self):
        """π must be non-negative and sum to 1 for arbitrary random inputs."""
        np.random.seed(99)
        N = 8
        A = np.abs(np.random.randn(N, N))
        np.fill_diagonal(A, 0)
        H = make_csr(A)
        pi, _, _ = katz(H, N, alpha=0.9)
        assert np.all(pi >= 0), "π must be non-negative"
        assert abs(pi.sum() - 1.0) < ATOL

    def test_cited_hub_ranks_higher(self):
        """
        Chain 0→1, 0→2, 0→3 all cite node 4 (star out from 0, all to 4).
        Node 4 is the sink and should rank highest.
        """
        N = 5
        # Nodes 0–3 all cite node 4; node 4 is dangling
        rows = [0, 1, 2, 3]
        cols = [4, 4, 4, 4]
        data = [1.0, 1.0, 1.0, 1.0]
        C = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        H, _ = _row_normalise(C)
        pi, _, _ = katz(H, N, alpha=0.85)
        assert pi[4] == pi.max(), "Sink node (most cited) should have highest score"


# ─── bipartite_resolvent() tests ──────────────────────────────────────────────

class TestBipartiteResolvent:

    def test_k22_uniform(self):
        """
        K_{2,2}: 2 sources, 2 institutions, fully connected with equal weights.
        By symmetry all four nodes get score 1/4.

        Analytical verification: see module docstring; c=0.25 for any α.
        """
        H_SI = sp.csr_matrix(np.array([[0.5, 0.5], [0.5, 0.5]]))
        H_IS = sp.csr_matrix(np.array([[0.5, 0.5], [0.5, 0.5]]))
        pi_s, pi_u = bipartite_resolvent(H_SI, H_IS, N_s=2, N_u=2, alpha=0.85)
        np.testing.assert_allclose(pi_s, [0.25, 0.25], atol=ATOL)
        np.testing.assert_allclose(pi_u, [0.25, 0.25], atol=ATOL)
        assert abs(pi_s.sum() + pi_u.sum() - 1.0) < ATOL

    def test_one_to_one_symmetric(self):
        """
        1 source, 1 institution, fully connected (H_SI=[[1]], H_IS=[[1]]).
        Perfect symmetry → π_S = π_U = 0.5.

        Verification: (1−α²)·π_S = (1−α)(1+α)·μ_S = (1−α²)·(1/2)
        → π_S = 1/2.
        """
        H_SI = sp.csr_matrix(np.array([[1.0]]))
        H_IS = sp.csr_matrix(np.array([[1.0]]))
        pi_s, pi_u = bipartite_resolvent(H_SI, H_IS, N_s=1, N_u=1, alpha=0.85)
        np.testing.assert_allclose(pi_s, [0.5], atol=ATOL)
        np.testing.assert_allclose(pi_u, [0.5], atol=ATOL)

    def test_institution_aggregates_two_sources(self):
        """
        2 sources, 1 institution.
        H_SI = [[1],[1]] (both sources fully cite the one institution).
        H_IS = [[0.5, 0.5]] (institution cites both sources equally).

        Analytical solution (α arbitrary)
        ----------------------------------
        By symmetry π_S = [c, c].  N = 3, μ_p = 1/3.
        M_S = H_SI @ H_IS = [[0.5,0.5],[0.5,0.5]]; M_S^T same.
        (I − α² M_S^T)[c,c] = (1−α²)[c,c]
        RHS = (1−α)(μ_S + α H_IS^T μ_I)
            = (1−α)([1/3,1/3] + α·[1/2,1/2]·(1/3))
            = (1−α)(2+α)/6 · [1,1]
        → c = (2+α)/(6(1+α))
        π_U = 1 − 2c = (6(1+α) − 2(2+α))/(6(1+α)) = (2+4α)/(6(1+α))
        """
        H_SI = sp.csr_matrix(np.array([[1.0], [1.0]]))
        H_IS = sp.csr_matrix(np.array([[0.5, 0.5]]))
        alpha = 0.85
        pi_s, pi_u = bipartite_resolvent(H_SI, H_IS, N_s=2, N_u=1, alpha=alpha)

        c_exact = (2 + alpha) / (6 * (1 + alpha))
        pi_u_exact = 1.0 - 2 * c_exact
        np.testing.assert_allclose(pi_s, [c_exact, c_exact], atol=ATOL)
        np.testing.assert_allclose(pi_u, [pi_u_exact], atol=ATOL)
        assert pi_u[0] > pi_s[0], "Institution aggregating two sources should rank highest"

    def test_dangling_source_ranks_lower(self):
        """
        Source 0 is in a 1-to-1 cycle with institution 0.
        Source 1 is dangling (no outgoing SI edges).
        Source 0 should rank higher than source 1.
        """
        # Source 0 → institution 0 (H_SI row 0 = [1])
        # Source 1 → nowhere     (H_SI row 1 = [0])
        H_SI = sp.csr_matrix(np.array([[1.0], [0.0]]))
        # Institution 0 → source 0 only
        H_IS = sp.csr_matrix(np.array([[1.0, 0.0]]))
        pi_s, pi_u = bipartite_resolvent(H_SI, H_IS, N_s=2, N_u=1, alpha=0.85)
        assert abs(pi_s.sum() + pi_u.sum() - 1.0) < ATOL
        assert np.all(pi_s >= 0) and np.all(pi_u >= 0)
        assert pi_s[0] > pi_s[1], "Active source should outrank dangling source"

    def test_jointly_normalised_to_one(self):
        """π_S and π_U must sum to exactly 1 jointly, for random inputs."""
        np.random.seed(17)
        N_s, N_u = 5, 3
        A_SI = np.abs(np.random.randn(N_s, N_u)) + 0.1
        A_SI /= A_SI.sum(axis=1, keepdims=True)
        A_IS = np.abs(np.random.randn(N_u, N_s)) + 0.1
        A_IS /= A_IS.sum(axis=1, keepdims=True)
        H_SI = sp.csr_matrix(A_SI)
        H_IS = sp.csr_matrix(A_IS)
        pi_s, pi_u = bipartite_resolvent(H_SI, H_IS, N_s, N_u, alpha=0.85)
        assert abs(pi_s.sum() + pi_u.sum() - 1.0) < ATOL
        assert np.all(pi_s >= 0) and np.all(pi_u >= 0)

    def test_agrees_with_katz_on_block_matrix(self):
        """
        Key consistency test: bipartite_resolvent and katz must agree when
        applied to the same network.

        bipartite_resolvent solves the fixed-point system directly; katz
        iterates on the assembled N×N block matrix.  These are independent
        code paths — agreement rules out implementation errors in either.

        Network: random (N_s=4, N_u=3) bipartite, all entries positive
        (no dangling nodes, so katz converges without complication).
        """
        np.random.seed(42)
        N_s, N_u = 4, 3
        N = N_s + N_u

        C_SI_raw = np.abs(np.random.randn(N_s, N_u)) + 0.2
        C_IS_raw = np.abs(np.random.randn(N_u, N_s)) + 0.2
        H_SI, _ = _row_normalise(sp.csr_matrix(C_SI_raw))
        H_IS, _ = _row_normalise(sp.csr_matrix(C_IS_raw))

        alpha = 0.85

        # Method 1: bipartite_resolvent (direct solve)
        pi_s_res, pi_u_res = bipartite_resolvent(H_SI, H_IS, N_s, N_u, alpha)

        # Method 2: katz on the assembled block matrix
        Z_ss = sp.csr_matrix((N_s, N_s))
        Z_uu = sp.csr_matrix((N_u, N_u))
        H_block = sp.bmat([[Z_ss, H_SI], [H_IS, Z_uu]], format='csr')
        mu = np.full(N, 1.0 / N)
        pi_katz, iters, norm = katz(H_block, N, alpha, mu=mu, tol=1e-12)

        pi_s_katz = pi_katz[:N_s]
        pi_u_katz = pi_katz[N_s:]

        np.testing.assert_allclose(
            pi_s_res, pi_s_katz, atol=ATOL_CROSS,
            err_msg="bipartite_resolvent π_S disagrees with katz on block matrix"
        )
        np.testing.assert_allclose(
            pi_u_res, pi_u_katz, atol=ATOL_CROSS,
            err_msg="bipartite_resolvent π_U disagrees with katz on block matrix"
        )

    def test_v_weighted_mean_equals_one(self):
        """
        v_p = A × π_p / a_p must have a_p-weighted mean = 1.
        This is the Pinski–Narin normalisation: v=1 ↔ average influence.

        Verified via rank() for the bipartite case with a toy network.
        """
        np.random.seed(3)
        N_s, N_u = 3, 4
        # Random raw count matrices (not yet normalised)
        C_SI = sp.csr_matrix(np.abs(np.random.randn(N_s, N_u)) + 0.1)
        C_IS = sp.csr_matrix(np.abs(np.random.randn(N_u, N_s)) + 0.1)
        a_s = np.array([10.0, 5.0, 20.0])       # integer work counts
        a_u = np.array([3.0, 1.5, 8.0, 4.5])    # fractional work counts
        data = _MockCSRData(
            C_SS=None, C_SI=C_SI, C_IS=C_IS, C_II=None,
            source_ids=np.arange(N_s), inst_ids=np.arange(N_u),
            a_s=a_s, a_u=a_u, n_s=N_s, n_u=N_u,
        )
        result = rank(data, m=(0, 1, 1, 0), chi=0.5, alpha=0.85)

        A = a_s.sum() + a_u.sum()
        weighted_mean = (
            (result.v_s * a_s).sum() + (result.v_u * a_u).sum()
        ) / A
        assert abs(weighted_mean - 1.0) < ATOL, (
            f"a_p-weighted mean of v should be 1.0 (Pinski–Narin), got {weighted_mean:.6f}"
        )

    def test_agrees_with_katz_different_alpha(self):
        """Consistency check repeated with α=0.5 (lower damping)."""
        np.random.seed(7)
        N_s, N_u = 3, 5
        C_SI_raw = np.abs(np.random.randn(N_s, N_u)) + 0.1
        C_IS_raw = np.abs(np.random.randn(N_u, N_s)) + 0.1
        H_SI, _ = _row_normalise(sp.csr_matrix(C_SI_raw))
        H_IS, _ = _row_normalise(sp.csr_matrix(C_IS_raw))

        alpha = 0.50
        N = N_s + N_u
        pi_s_res, pi_u_res = bipartite_resolvent(H_SI, H_IS, N_s, N_u, alpha)

        Z_ss = sp.csr_matrix((N_s, N_s))
        Z_uu = sp.csr_matrix((N_u, N_u))
        H_block = sp.bmat([[Z_ss, H_SI], [H_IS, Z_uu]], format='csr')
        mu = np.full(N, 1.0 / N)
        pi_katz, _, _ = katz(H_block, N, alpha, mu=mu, tol=1e-12)

        np.testing.assert_allclose(pi_s_res, pi_katz[:N_s], atol=ATOL_CROSS)
        np.testing.assert_allclose(pi_u_res, pi_katz[N_s:], atol=ATOL_CROSS)
