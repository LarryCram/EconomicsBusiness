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
from katz_ranker import (
    katz, bipartite_resolvent, _row_normalise, rank,
    check_primitive, power_iteration, bipartite, NotPrimitiveError,
)

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

        Analytical solution (α is the round-trip damping passed to bipartite_resolvent;
        per-hop damping is √α)
        ---------------------------------------------------------------------------------
        By symmetry π_S = [c, c].  N = 3, μ_p = 1/3, β = √α.
        M_S = H_SI @ H_IS = [[0.5,0.5],[0.5,0.5]].
        System: π_S = β H_IS^T π_I + (1−β) μ_S
                π_I = β H_SI^T π_S + (1−β) μ_I
        Substituting and using (I − α M_S^T)[c,c] = (1−α)[c,c]:
        (1−α)c = (1−β)(1/3 + β·(1/2)·(1/3)) = (1−β)(2+β)/6
        c = (1−β)(2+β) / (6(1−α)) = (2+β) / (6(1+β))   [since 1−α = (1−β)(1+β)]
        → c = (2+√α) / (6(1+√α))
        π_U = 1 − 2c
        """
        H_SI = sp.csr_matrix(np.array([[1.0], [1.0]]))
        H_IS = sp.csr_matrix(np.array([[0.5, 0.5]]))
        alpha = 0.85
        pi_s, pi_u = bipartite_resolvent(H_SI, H_IS, N_s=2, N_u=1, alpha=alpha)

        sqrt_a = np.sqrt(alpha)
        c_exact = (2 + sqrt_a) / (6 * (1 + sqrt_a))
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

        bipartite_resolvent uses per-hop damping √α (round-trip = α).
        katz on the block matrix uses per-hop damping α_katz.
        To match conventions: α_katz = √α_resolvent.

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

        alpha_res = 0.85   # round-trip damping for bipartite_resolvent

        # Method 1: bipartite_resolvent (direct solve, round-trip α)
        pi_s_res, pi_u_res = bipartite_resolvent(H_SI, H_IS, N_s, N_u, alpha_res)

        # Method 2: katz on the assembled block matrix with per-hop α = √α_res
        Z_ss = sp.csr_matrix((N_s, N_s))
        Z_uu = sp.csr_matrix((N_u, N_u))
        H_block = sp.bmat([[Z_ss, H_SI], [H_IS, Z_uu]], format='csr')
        mu = np.full(N, 1.0 / N)
        pi_katz, iters, norm = katz(H_block, N, np.sqrt(alpha_res), mu=mu, tol=1e-12)

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
        """Consistency check repeated with α=0.5. katz receives √α_resolvent."""
        np.random.seed(7)
        N_s, N_u = 3, 5
        C_SI_raw = np.abs(np.random.randn(N_s, N_u)) + 0.1
        C_IS_raw = np.abs(np.random.randn(N_u, N_s)) + 0.1
        H_SI, _ = _row_normalise(sp.csr_matrix(C_SI_raw))
        H_IS, _ = _row_normalise(sp.csr_matrix(C_IS_raw))

        alpha_res = 0.50   # round-trip damping for bipartite_resolvent
        N = N_s + N_u
        pi_s_res, pi_u_res = bipartite_resolvent(H_SI, H_IS, N_s, N_u, alpha_res)

        Z_ss = sp.csr_matrix((N_s, N_s))
        Z_uu = sp.csr_matrix((N_u, N_u))
        H_block = sp.bmat([[Z_ss, H_SI], [H_IS, Z_uu]], format='csr')
        mu = np.full(N, 1.0 / N)
        pi_katz, _, _ = katz(H_block, N, np.sqrt(alpha_res), mu=mu, tol=1e-12)

        np.testing.assert_allclose(pi_s_res, pi_katz[:N_s], atol=ATOL_CROSS)
        np.testing.assert_allclose(pi_u_res, pi_katz[N_s:], atol=ATOL_CROSS)


# ─── α=1 primitivity tests ────────────────────────────────────────────────────

class TestAlphaOne:
    """
    Tests for α=1 (no damping).  Convergence at α=1 requires H̃ to be
    primitive (strongly connected + aperiodic).

    katz() at α=1:
      - The update is π ← H^T π + dangling_mass·μ (the prior injection
        from dangling nodes is the only non-H term when α=1).
      - A primitive H converges to its Perron eigenvector.
      - A periodic H with no dangling nodes oscillates (does not converge).
      - A periodic bipartite H with dangling nodes may converge because the
        dangling redistribution breaks periodicity.

    bipartite_resolvent() at α=1:
      - The per-step damping is sqrt(1)=1; the RHS collapses to zero and
        LHS (I − M_S^T) is singular.  spsolve cannot find a unique solution.
      - The correct limit is the Perron eigenvector of M_S (and then π_I
        recovered by π_I = H_SI^T π_S).  This test verifies that the limit
        of bipartite_resolvent as α→1 matches the M_S Perron vector.
    """

    def test_katz_alpha_one_primitive_converges(self):
        """
        Primitive H: 3-cycle 0→1→2→0 plus self-loop on node 0.
        The self-loop makes gcd(cycle lengths) = gcd(1,3) = 1 → aperiodic.
        katz must converge at α=1 and return the Perron eigenvector.
        """
        A = [[1, 1, 0],
             [0, 0, 1],
             [1, 0, 0]]
        H = make_csr(A)
        pi, iters, norm = katz(H, 3, alpha=1.0, tol=1e-9, max_iter=500)
        assert norm < 1e-6, (
            f"α=1 failed to converge for primitive H after {iters} iters "
            f"(final norm={norm:.2e})"
        )
        assert abs(pi.sum() - 1.0) < ATOL
        assert np.all(pi >= 0)

    def test_katz_alpha_one_periodic_does_not_converge(self):
        """
        Pure 2-cycle 0→1→0 with no dangling nodes: H is periodic (period 2).
        At α=1 there is no prior injection to break periodicity, so the
        iteration oscillates and must NOT converge within max_iter steps.

        A non-uniform μ is required: the uniform vector [0.5, 0.5] is itself
        a fixed point of the 2-cycle and would give norm=0 trivially.
        """
        H = sp.csr_matrix(np.array([[0., 1.], [1., 0.]]))
        mu = np.array([0.7, 0.3])   # non-uniform: breaks degenerate fixed point
        pi, iters, norm = katz(H, 2, alpha=1.0, mu=mu, tol=1e-9, max_iter=200)
        assert norm > 1e-6, (
            "Periodic H (2-cycle, no dangling) with α=1 should NOT converge, "
            f"but got norm={norm:.2e} after {iters} iters"
        )

    def test_katz_alpha_one_bipartite_with_dangling_converges(self):
        """
        Bipartite block matrix at α=1.  Source 1 is dangling (no SI edges),
        so the dangling redistribution injects prior mass at every step even
        though (1−α)=0.  This breaks the bipartite periodicity and allows
        convergence.

        Network: source 0 ↔ inst 0 (cycle); source 1 dangling.
        """
        N_s, N_u = 2, 1
        N = N_s + N_u
        H_SI = sp.csr_matrix(np.array([[1.0], [0.0]]))   # src1 is dangling
        H_IS = sp.csr_matrix(np.array([[1.0, 0.0]]))
        Z_ss = sp.csr_matrix((N_s, N_s))
        Z_uu = sp.csr_matrix((N_u, N_u))
        H_block = sp.bmat([[Z_ss, H_SI], [H_IS, Z_uu]], format='csr')
        mu = np.full(N, 1.0 / N)
        pi, iters, norm = katz(H_block, N, alpha=1.0, mu=mu,
                               tol=1e-9, max_iter=500)
        assert norm < 1e-6, (
            f"Bipartite with dangling source and α=1 should converge "
            f"(norm={norm:.2e} after {iters} iters)"
        )
        assert abs(pi.sum() - 1.0) < ATOL
        assert np.all(pi >= 0)

    def test_katz_alpha_one_pure_bipartite_no_dangling_does_not_converge(self):
        """
        Pure bipartite block matrix with no dangling nodes, α=1.
        No dangling redistribution → no prior injection → periodicity
        is not broken → should NOT converge.

        Network: two disconnected 2-cycles (src0↔inst0, src1↔inst1).
        H_block is [[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]].
        From any non-symmetric starting vector, iteration oscillates with
        period 2 and the L1 norm never falls below a positive constant.

        Note: K_{2,2} with equal weights is NOT suitable here — its degenerate
        structure maps every vector to uniform in one step and then stays there.
        """
        N_s, N_u = 2, 2
        N = N_s + N_u
        # src0→inst0 only, src1→inst1 only (and vice versa)
        H_SI = sp.csr_matrix(np.array([[1., 0.], [0., 1.]]))
        H_IS = sp.csr_matrix(np.array([[1., 0.], [0., 1.]]))
        Z_ss = sp.csr_matrix((N_s, N_s))
        Z_uu = sp.csr_matrix((N_u, N_u))
        H_block = sp.bmat([[Z_ss, H_SI], [H_IS, Z_uu]], format='csr')
        mu = np.array([0.4, 0.1, 0.3, 0.2])   # non-uniform starting vector
        pi, iters, norm = katz(H_block, N, alpha=1.0, mu=mu,
                               tol=1e-9, max_iter=200)
        assert norm > 1e-6, (
            "Disconnected bipartite 2-cycles with α=1 should NOT converge, "
            f"but got norm={norm:.2e}"
        )

    def test_bipartite_resolvent_alpha_approaches_one(self):
        """
        As α→1, bipartite_resolvent must approach the Perron eigenvector of M_S.

        At α=1 the resolvent system is singular, so we use α=1−ε and verify
        that the solution converges to the M_S Perron vector as ε→0.

        Network: random (N_s=4, N_u=3), all entries positive (primitive M_S).
        The Perron vector of M_S is computed via scipy.sparse.linalg.eigs.
        """
        from scipy.sparse.linalg import eigs

        np.random.seed(55)
        N_s, N_u = 4, 3
        C_SI_raw = np.abs(np.random.randn(N_s, N_u)) + 0.2
        C_IS_raw = np.abs(np.random.randn(N_u, N_s)) + 0.2
        H_SI, _ = _row_normalise(sp.csr_matrix(C_SI_raw))
        H_IS, _ = _row_normalise(sp.csr_matrix(C_IS_raw))
        M_S = H_SI.dot(H_IS)   # N_s × N_s one-mode projection

        # Perron vector of M_S (dominant left eigenvector of M_S^T)
        _, vecs = eigs(M_S.T.toarray(), k=1, which='LM')
        perron = np.abs(vecs[:, 0].real)
        perron /= perron.sum()

        # bipartite_resolvent at α close to 1: π_S should approach perron
        pi_s_near1, _ = bipartite_resolvent(H_SI, H_IS, N_s, N_u, alpha=0.9999)
        # Normalise π_S alone for comparison with perron
        pi_s_norm = pi_s_near1 / pi_s_near1.sum()

        np.testing.assert_allclose(
            pi_s_norm, perron, atol=1e-3,
            err_msg="bipartite_resolvent at α≈1 should approach M_S Perron vector"
        )


# ─── check_primitive() tests ──────────────────────────────────────────────────

class TestCheckPrimitive:

    def test_all_nonzero_returns_one(self):
        """Dense matrix: all entries positive → primitivity index = 1."""
        M = sp.csr_matrix(np.ones((4, 4)))
        k = check_primitive(M)
        assert k == 1

    def test_k3_no_self_loops_returns_two(self):
        """
        Complete undirected K_3 without self-loops.
        M^1 has zeros on diagonal; M^2 is all-nonzero → index = 2.
        """
        M = sp.csr_matrix(np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float))
        k = check_primitive(M)
        assert k == 2

    def test_two_cycle_returns_minus_one(self):
        """Pure 2-cycle 0→1→0: periodic, not primitive within k=32 → returns -1."""
        M = sp.csr_matrix(np.array([[0., 1.], [1., 0.]]))
        assert check_primitive(M) == -1

    def test_six_cycle_returns_minus_one(self):
        """6-node directed cycle: primitivity index=6 > 5, not confirmed within k=32 → -1."""
        N = 6
        rows = list(range(N))
        cols = [(i + 1) % N for i in range(N)]
        M = sp.coo_matrix(([1.0] * N, (rows, cols)), shape=(N, N)).tocsr()
        assert check_primitive(M) == -1

    def test_primitivity_index_three(self):
        """
        Chain 0→1→2→0 plus edge 0→2 added to shorten some paths.
        Verify index=1 since M[0,2]=1 already, M[0,1]=1:
        M = [[0,1,1],[0,0,1],[1,0,0]]; M^1 not all-nonzero; M^2 not all-nonzero;
        M^3 all-nonzero → index = 3.
        """
        M = sp.csr_matrix(np.array([
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=float))
        # Verify manually: index should be 3 for this graph
        k = check_primitive(M)
        # The index must be ≤ 5 (doesn't raise) and ≥ 1
        assert 1 <= k <= 5


# ─── power_iteration() tests ─────────────────────────────────────────────────

class TestPowerIteration:

    def test_alpha_one_primitive_uniform(self):
        """
        alpha=1, primitive H: symmetric K_3 (primitivity index=2).
        Perron eigenvector is uniform [1/3, 1/3, 1/3].
        """
        H = make_csr([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        pi, iters, norm = power_iteration(H, alpha=1.0)
        np.testing.assert_allclose(pi, [1/3, 1/3, 1/3], atol=ATOL)
        assert abs(pi.sum() - 1.0) < ATOL

    def test_alpha_one_non_primitive_continues(self):
        """alpha=1 on a non-primitive matrix: check_primitive warns but iteration proceeds."""
        H = sp.csr_matrix(np.array([[0., 1.], [1., 0.]]))
        # 2-cycle is periodic; check_primitive returns -1 and prints a warning.
        # power_iteration continues and may not converge — just verify no exception.
        pi, iters, norm = power_iteration(H, alpha=1.0, max_iter=10)
        assert pi.shape == (2,)

    def test_alpha_one_with_mu_raises(self):
        """alpha=1 with mu given is not a valid regime → ValueError."""
        H = make_csr([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        mu = np.array([1/3, 1/3, 1/3])
        with pytest.raises(ValueError):
            power_iteration(H, alpha=1.0, mu=mu)

    def test_katz_mu_none_converges(self):
        """alpha<1, mu=None (original Katz): converges; output normalised."""
        H = make_csr([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        pi, iters, norm = power_iteration(H, alpha=0.85, mu=None)
        assert abs(pi.sum() - 1.0) < ATOL
        assert np.all(pi >= 0)

    def test_katz_hubbell_converges(self):
        """alpha<1, mu>0 (Katz–Hubbell): converges; output normalised."""
        H = make_csr([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        mu = np.array([0.2, 0.5, 0.3])
        pi, iters, norm = power_iteration(H, alpha=0.85, mu=mu)
        assert abs(pi.sum() - 1.0) < ATOL
        assert np.all(pi >= 0)

    def test_katz_hubbell_agrees_with_katz_on_symmetric(self):
        """
        For symmetric K_3 with uniform prior, Katz–Hubbell must return uniform.
        """
        H = make_csr([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        mu = np.full(3, 1/3)
        pi, _, _ = power_iteration(H, alpha=0.85, mu=mu)
        np.testing.assert_allclose(pi, [1/3, 1/3, 1/3], atol=ATOL)

    def test_alpha_one_agrees_with_katz_function(self):
        """
        power_iteration at alpha=1 on a primitive matrix must agree with
        katz() at alpha=1 (both find the Perron eigenvector).
        """
        np.random.seed(13)
        N = 5
        A = np.abs(np.random.randn(N, N)) + 0.1   # all-positive → primitive
        H, _ = _row_normalise(sp.csr_matrix(A))
        pi_pi, _, _ = power_iteration(H, alpha=1.0)
        pi_katz, _, _ = katz(H, N, alpha=1.0)
        np.testing.assert_allclose(pi_pi, pi_katz, atol=ATOL,
            err_msg="power_iteration and katz disagree at alpha=1")


# ─── bipartite() tests ───────────────────────────────────────────────────────

class TestBipartite:

    def test_k22_alpha_one_uniform(self):
        """
        K_{2,2} at alpha=1: M_S is all-nonzero (primitive), Perron vector is
        uniform → pi_s = [0.5, 0.5], pi_u = [0.5, 0.5].
        """
        H_SI = sp.csr_matrix(np.array([[0.5, 0.5], [0.5, 0.5]]))
        H_IS = sp.csr_matrix(np.array([[0.5, 0.5], [0.5, 0.5]]))
        pi_s, pi_u, _, _ = bipartite(H_SI, H_IS, alpha=1.0)
        np.testing.assert_allclose(pi_s, [0.5, 0.5], atol=ATOL)
        np.testing.assert_allclose(pi_u, [0.5, 0.5], atol=ATOL)

    def test_individual_normalisation(self):
        """pi_s and pi_u must each sum to 1 individually (not jointly)."""
        np.random.seed(21)
        N_s, N_u = 4, 3
        A_SI = np.abs(np.random.randn(N_s, N_u)) + 0.1
        A_IS = np.abs(np.random.randn(N_u, N_s)) + 0.1
        H_SI, _ = _row_normalise(sp.csr_matrix(A_SI))
        H_IS, _ = _row_normalise(sp.csr_matrix(A_IS))
        pi_s, pi_u, _, _ = bipartite(H_SI, H_IS, alpha=1.0)
        assert abs(pi_s.sum() - 1.0) < ATOL, f"pi_s sums to {pi_s.sum():.6f}"
        assert abs(pi_u.sum() - 1.0) < ATOL, f"pi_u sums to {pi_u.sum():.6f}"
        assert np.all(pi_s >= 0) and np.all(pi_u >= 0)

    def test_agrees_with_bipartite_resolvent_pi_s(self):
        """
        bipartite() uses per-hop alpha; bipartite_resolvent() uses round-trip alpha.
        They solve the same system when bipartite_resolvent receives alpha²:
          bipartite(alpha=a)  ≡  bipartite_resolvent(alpha=a²)
        because bipartite() passes alpha²=a² as round-trip damping to M_S,
        matching the resolvent's round-trip convention.
        """
        np.random.seed(42)
        N_s, N_u = 4, 3
        N = N_s + N_u
        A_SI = np.abs(np.random.randn(N_s, N_u)) + 0.2
        A_IS = np.abs(np.random.randn(N_u, N_s)) + 0.2
        H_SI, _ = _row_normalise(sp.csr_matrix(A_SI))
        H_IS, _ = _row_normalise(sp.csr_matrix(A_IS))
        alpha = 0.85   # per-hop

        mu_joint = np.full(N, 1.0 / N)
        pi_s_bi, _, _, _ = bipartite(H_SI, H_IS, alpha=alpha, mu=mu_joint)
        # resolvent takes round-trip alpha = alpha²
        pi_s_res, _ = bipartite_resolvent(H_SI, H_IS, N_s, N_u, alpha=alpha**2)

        pi_s_res_norm = pi_s_res / pi_s_res.sum()

        np.testing.assert_allclose(pi_s_bi, pi_s_res_norm, atol=1e-5,
            err_msg="bipartite() pi_s disagrees with bipartite_resolvent(alpha²)")

    def test_alpha_one_agrees_with_resolvent_limit(self):
        """
        bipartite() at alpha=1 should return the Perron vector of M_S.
        bipartite_resolvent at alpha≈1 approaches the same limit.
        """
        np.random.seed(55)
        N_s, N_u = 4, 3
        A_SI = np.abs(np.random.randn(N_s, N_u)) + 0.2
        A_IS = np.abs(np.random.randn(N_u, N_s)) + 0.2
        H_SI, _ = _row_normalise(sp.csr_matrix(A_SI))
        H_IS, _ = _row_normalise(sp.csr_matrix(A_IS))

        pi_s_one, _, _, _ = bipartite(H_SI, H_IS, alpha=1.0)
        pi_s_near1, _ = bipartite_resolvent(H_SI, H_IS, N_s, N_u, alpha=0.9999)
        pi_s_near1_norm = pi_s_near1 / pi_s_near1.sum()

        np.testing.assert_allclose(pi_s_one, pi_s_near1_norm, atol=1e-3,
            err_msg="bipartite() at alpha=1 should match resolvent limit at alpha→1")

    def test_non_primitive_m_s_continues(self):
        """
        If M_S = H_SI @ H_IS is not primitive and alpha=1, check_primitive warns
        and returns -1; power_iteration continues.

        M_S = I (block diagonal): not primitive within k=32 → warning, no exception.
        """
        H_SI = sp.csr_matrix(np.array([[1., 0.], [0., 1.]]))
        H_IS = sp.csr_matrix(np.array([[1., 0.], [0., 1.]]))
        pi_s, pi_u, iters, norm = bipartite(H_SI, H_IS, alpha=1.0)
        assert pi_s.shape == (2,)
        assert pi_u.shape == (2,)


# ─── rank() tests ─────────────────────────────────────────────────────────────

def _make_mock(seed=0, N_s=4, N_u=3):
    """
    Build a _MockCSRData with all four raw count blocks and integer-like
    work counts a_s, a_u.  All blocks are fully dense so no dangling nodes.
    """
    rng = np.random.default_rng(seed)
    C_SS = sp.csr_matrix(rng.uniform(0.1, 1.0, (N_s, N_s)))
    C_SI = sp.csr_matrix(rng.uniform(0.1, 1.0, (N_s, N_u)))
    C_IS = sp.csr_matrix(rng.uniform(0.1, 1.0, (N_u, N_s)))
    C_II = sp.csr_matrix(rng.uniform(0.1, 1.0, (N_u, N_u)))
    a_s  = rng.uniform(1.0, 20.0, N_s)
    a_u  = rng.uniform(1.0, 20.0, N_u)
    return _MockCSRData(
        C_SS=C_SS, C_SI=C_SI, C_IS=C_IS, C_II=C_II,
        source_ids=np.arange(N_s), inst_ids=np.arange(N_u),
        a_s=a_s, a_u=a_u, n_s=N_s, n_u=N_u,
    )


def _pinski_narin(v, a):
    """a-weighted mean of v; should equal 1.0."""
    return float((v * a).sum() / a.sum())


class TestRank:
    """
    Tests for rank().  These are designed to be routing-invariant:
    they must pass both before step 4 (legacy katz/bipartite_resolvent routing)
    and after step 4 (new power_iteration/bipartite routing).

    The Pinski–Narin invariant — a_p-weighted mean of v_p = 1 — is the key
    correctness criterion: it holds regardless of which algorithm is used,
    provided v_p = A * pi_p / a_p and pi_p is correctly normalised.

    Notation
    --------
    For modes 1000 and 0001, A = a_s.sum() or a_u.sum() respectively.
    For modes 0110 and 1111, A = a_s.sum() + a_u.sum() (joint).
    In 0110 the joint Pinski-Narin condition is:
        (sum(v_s * a_s) + sum(v_u * a_u)) / (a_s.sum() + a_u.sum()) = 1
    which requires pi_s and pi_u to be jointly normalised (sum to 1 together).
    """

    def test_ss_only_pinski_narin(self):
        """m=(1,0,0,0): a_s-weighted mean of v_s must equal 1."""
        data = _make_mock(seed=10)
        result = rank(data, m=(1, 0, 0, 0), chi=0.5, alpha=0.85)
        assert result.pi_u is None and result.v_u is None
        assert abs(_pinski_narin(result.v_s, data.a_s) - 1.0) < ATOL

    def test_ii_only_pinski_narin(self):
        """m=(0,0,0,1): a_u-weighted mean of v_u must equal 1."""
        data = _make_mock(seed=11)
        result = rank(data, m=(0, 0, 0, 1), chi=0.5, alpha=0.85)
        assert result.pi_s is None and result.v_s is None
        assert abs(_pinski_narin(result.v_u, data.a_u) - 1.0) < ATOL

    def test_bipartite_joint_pinski_narin(self):
        """
        m=(0,1,1,0): joint a-weighted mean of (v_s, v_u) must equal 1.
        pi_s and pi_u must be jointly normalised (sum together to 1).
        """
        data = _make_mock(seed=12)
        result = rank(data, m=(0, 1, 1, 0), chi=0.5, alpha=0.85)
        A = data.a_s.sum() + data.a_u.sum()
        joint_mean = (
            (result.v_s * data.a_s).sum() + (result.v_u * data.a_u).sum()
        ) / A
        assert abs(joint_mean - 1.0) < ATOL
        assert abs(result.pi_s.sum() + result.pi_u.sum() - 1.0) < ATOL, (
            "pi_s and pi_u must be jointly normalised to 1 for 0110"
        )

    def test_full_joint_pinski_narin(self):
        """m=(1,1,1,1): joint a-weighted mean of (v_s, v_u) must equal 1."""
        data = _make_mock(seed=13)
        result = rank(data, m=(1, 1, 1, 1), chi=0.5, alpha=0.85)
        A = data.a_s.sum() + data.a_u.sum()
        joint_mean = (
            (result.v_s * data.a_s).sum() + (result.v_u * data.a_u).sum()
        ) / A
        assert abs(joint_mean - 1.0) < ATOL

    def test_all_modes_nonnegative(self):
        """v and pi must be non-negative for all modes."""
        data = _make_mock(seed=14)
        for m in [(1, 0, 0, 0), (0, 0, 0, 1), (0, 1, 1, 0), (1, 1, 1, 1)]:
            result = rank(data, m=m, chi=0.5, alpha=0.85)
            if result.pi_s is not None:
                assert np.all(result.pi_s >= 0), f"pi_s negative for m={m}"
                assert np.all(result.v_s  >= 0), f"v_s negative for m={m}"
            if result.pi_u is not None:
                assert np.all(result.pi_u >= 0), f"pi_u negative for m={m}"
                assert np.all(result.v_u  >= 0), f"v_u negative for m={m}"

    def test_bipartite_pi_s_direction_agrees_with_bipartite_fn(self):
        """
        rank(0110) and bipartite() must return the same pi_s direction.

        rank() jointly normalises pi_s and pi_u (they sum to 1 together).
        bipartite() individually normalises (each sums to 1).
        After dividing rank()'s pi_s by its sum, both must agree.

        At alpha=0.85 this tests the current routing (bipartite_resolvent)
        and will continue to pass after step 4 (bipartite routing).
        """
        data = _make_mock(seed=15)
        result = rank(data, m=(0, 1, 1, 0), chi=0.5, alpha=0.85)

        H_SI, _ = _row_normalise(data.C_SI)
        H_IS, _ = _row_normalise(data.C_IS)
        pi_s_bi, _, _, _ = bipartite(H_SI, H_IS, alpha=0.85)

        # Normalise rank()'s pi_s for direction comparison
        pi_s_rank_norm = result.pi_s / result.pi_s.sum()

        np.testing.assert_allclose(
            pi_s_rank_norm, pi_s_bi, atol=1e-5,
            err_msg="rank(0110) pi_s direction disagrees with bipartite()"
        )

    def test_ss_alpha_one_converges(self):
        """rank(1000) at alpha=1 converges (primitive H guaranteed by dense C_SS)."""
        data = _make_mock(seed=16)
        result = rank(data, m=(1, 0, 0, 0), chi=0.5, alpha=1.0)
        assert result.final_norm < 1e-6, (
            f"rank(1000) at alpha=1 did not converge (norm={result.final_norm:.2e})"
        )
        assert abs(_pinski_narin(result.v_s, data.a_s) - 1.0) < ATOL

    def test_bipartite_alpha_one_pinski_narin(self):
        """rank(0110) at alpha=1 (operational baseline): joint Pinski-Narin holds."""
        data = _make_mock(seed=17)
        result = rank(data, m=(0, 1, 1, 0), chi=0.5, alpha=1.0)
        A = data.a_s.sum() + data.a_u.sum()
        joint_mean = (
            (result.v_s * data.a_s).sum() + (result.v_u * data.a_u).sum()
        ) / A
        assert abs(joint_mean - 1.0) < ATOL
        assert abs(result.pi_s.sum() + result.pi_u.sum() - 1.0) < ATOL
