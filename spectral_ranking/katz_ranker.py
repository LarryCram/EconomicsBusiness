"""
katz_ranker.py — Spectral ranking algorithms.

Pure functions; no I/O. Called by run_rankings.py.

Public API
----------
katz(H, N, alpha, mu=None, tol=1e-9, max_iter=500)
    -> (pi, iters, norm)
    Standard Katz–Hubbell iteration for one-mode matrices (SS, II, full joint).
    H must be row-stochastic with dangling rows left as zero rows.

bipartite_resolvent(H_SI, H_IS, N_s, N_u, alpha)
    -> (pi_s, pi_u)
    Direct Breiger-projection resolve for the bipartite SI/IS case.
    H_SI and H_IS must be row-stochastic (dangling rows as zero rows).

rank(csr_data, m, chi, alpha)
    -> RankResult
    Top-level entry point: row-normalises raw C blocks, assembles H per m,
    routes to katz() or bipartite_resolvent(), returns prestige scores and v.

_row_normalise(C)
    -> (H, dangling_mask)
    Internal helper; exposed so tests and build_csr.py can use it.
"""

import sys

import numpy as np
from dataclasses import dataclass
from typing import Optional

from scipy.sparse import csr_matrix, bmat, diags, eye
from scipy.sparse.linalg import spsolve


# ─── Return types ─────────────────────────────────────────────────────────────

@dataclass
class RankResult:
    pi_s: Optional[np.ndarray]   # prestige scores for sources   (None for II-only)
    pi_u: Optional[np.ndarray]   # prestige scores for institutions (None for SS-only)
    v_s:  Optional[np.ndarray]   # prestige per work for sources  (None for II-only)
    v_u:  Optional[np.ndarray]   # prestige per work for institutions (None for SS-only)
    iters: int                   # iterations to convergence (0 for bipartite_resolvent)
    final_norm: float            # L1 residual at convergence (0.0 for bipartite_resolvent)


# ─── Internal helper ──────────────────────────────────────────────────────────

def _row_normalise(C: csr_matrix) -> tuple:
    """
    Row-normalise a CSR matrix to obtain a row-stochastic H.
    Dangling rows (row sum = 0) are left as zero rows; their indices are
    returned in dangling_mask so the caller can apply prior redistribution.

    Returns
    -------
    H : csr_matrix, row-stochastic (dangling rows are zero)
    dangling_mask : np.ndarray of bool, shape (N,)
    """
    row_sums = np.array(C.sum(axis=1)).ravel()
    dangling_mask = row_sums == 0
    # Avoid divide-by-zero: substitute 1 for zero rows (row stays zero after multiply)
    safe_sums = np.where(dangling_mask, 1.0, row_sums)
    H = diags(1.0 / safe_sums) @ C
    return H.tocsr(), dangling_mask


# ─── Core algorithms ──────────────────────────────────────────────────────────

def katz(
    H: csr_matrix,
    N: int,
    alpha: float,
    mu: Optional[np.ndarray] = None,
    tol: float = 1e-9,
    max_iter: int = 500,
) -> tuple:
    """
    Katz–Hubbell power iteration.

    Solves π = H̃^T π where H̃ = αH + (1−α)μ1^T, with dangling rows handled
    by redistributing their mass through the prior at every step.

    Parameters
    ----------
    H : csr_matrix, shape (N, N).  Row-stochastic; dangling rows are zero rows.
    N : int
    alpha : float in [0, 1]
    mu : prior, shape (N,).  Default: uniform 1/N.
    tol : L1 convergence tolerance (default 1e-9)
    max_iter : maximum iterations (default 500)

    Returns
    -------
    pi : np.ndarray, shape (N,), L1-normalised, non-negative
    iters : int, iterations taken
    final_norm : float, L1 residual on exit
    """
    if mu is None:
        mu = np.full(N, 1.0 / N)

    dangling_mask = np.array(H.sum(axis=1)).ravel() == 0

    pi = mu.copy()
    final_norm = 0.0
    for i in range(1, max_iter + 1):
        dangling_mass = pi[dangling_mask].sum()
        pi_new = alpha * H.T.dot(pi) + (alpha * dangling_mass + (1.0 - alpha)) * mu
        pi_new = np.maximum(pi_new, 0.0)          # guard FP underflow
        pi_new /= pi_new.sum()                     # mandatory renormalisation
        final_norm = float(np.abs(pi_new - pi).sum())
        pi = pi_new
        if final_norm < tol:
            return pi, i, final_norm

    return pi, max_iter, final_norm


# ─── New spectral utilities ──────────────────────────────────────────────────

class NotPrimitiveError(RuntimeError):
    """Raised when M is not primitive within the search depth k=5."""


def check_primitive(M: csr_matrix) -> int:
    """
    Test primitivity of M via boolean power iteration on the sparsity pattern,
    restricted to the non-dangling subgraph.

    Dangling nodes (zero rows) converge to π=0 under renormalised power
    iteration — the renormalisation at each step redistributes their absorbed
    mass back into the non-dangling subgraph, which is the standard PageRank
    treatment.  Only the non-dangling subgraph needs to be primitive.

    Returns the primitivity index k of the non-dangling subgraph.
    Raises NotPrimitiveError if k > 5.

    Parameters
    ----------
    M : csr_matrix, shape (N, N)

    Returns
    -------
    k : int, primitivity index in {1, ..., 5}
    """
    non_dangling = np.asarray(M.getnnz(axis=1) > 0).ravel()
    if not non_dangling.any():
        print("  check_primitive: M has no non-dangling nodes — skipping check.",
              flush=True)
        return -1

    M_sub = M[non_dangling][:, non_dangling]
    N = M_sub.shape[0]

    B = M_sub.astype(bool)
    Bk = B.copy()
    for k in range(1, 33):
        if Bk.nnz == N * N:
            return k
        if k < 32:
            Bk = (Bk @ B).astype(bool)
    print(
        f"  check_primitive: non-dangling subgraph (N={N}) not confirmed "
        f"primitive within k=32 (nnz={Bk.nnz}/{N * N}). "
        f"Continuing — convergence will be verified via final_norm.",
        flush=True,
    )
    return -1


def power_iteration(
    M: csr_matrix,
    alpha: float,
    mu: Optional[np.ndarray] = None,
    tol: float = 1e-9,
    max_iter: int = 500,
    skip_primitive_check: bool = False,
) -> tuple:
    """
    Power iteration for the spectral ranking fixed point π = M̃^T π.

    Three valid regimes
    -------------------
    (alpha=1, mu=None)   Pure Perron iteration.  Requires M to be primitive;
                         check_primitive() is called unless skip_primitive_check=True.
    (alpha<1, mu=None)   Original Katz damping.  π_new ← alpha·M^T·π, renormalised.
                         Convergence to the dominant eigenvector of M; no prior.
    (alpha<1, mu>0)      Katz–Hubbell.  π_new ← alpha·M^T·π + (1−alpha)·mu,
                         renormalised.

    Parameters
    ----------
    M : csr_matrix, shape (N, N).  Row-stochastic; dangling rows as zero rows.
    alpha : float in [0, 1].  Damping factor.
    mu : prior, shape (N,), or None.  Must be None when alpha=1.
    tol : L1 convergence tolerance (default 1e-9).
    max_iter : maximum iterations (default 500).
    skip_primitive_check : bool.  If True, bypass check_primitive() when alpha=1.
        Use for bootstrap replicates where the full matrix is known primitive
        but sub-samples may not confirm within k=32.

    Returns
    -------
    pi : np.ndarray, shape (N,), L1-normalised, non-negative
    iters : int, iterations taken
    final_norm : float, L1 residual on exit
    """
    N = M.shape[0]

    if alpha == 1.0:
        if mu is not None:
            raise ValueError(
                "mu must be None when alpha=1 (pure Perron iteration). "
                "Use alpha<1 for Katz–Hubbell."
            )
        if not skip_primitive_check:
            check_primitive(M)
        pi = np.full(N, 1.0 / N)
        final_norm = 0.0
        for i in range(1, max_iter + 1):
            pi_new = M.T.dot(pi)
            pi_new = np.maximum(pi_new, 0.0)
            pi_new /= pi_new.sum()
            final_norm = float(np.abs(pi_new - pi).sum())
            pi = pi_new
            if final_norm < tol:
                return pi, i, final_norm
        return pi, max_iter, final_norm

    # alpha < 1
    if mu is None:
        # Original Katz: no prior injection
        pi = np.full(N, 1.0 / N)
        final_norm = 0.0
        for i in range(1, max_iter + 1):
            pi_new = alpha * M.T.dot(pi)
            pi_new = np.maximum(pi_new, 0.0)
            s = pi_new.sum()
            if s > 0:
                pi_new /= s
            final_norm = float(np.abs(pi_new - pi).sum())
            pi = pi_new
            if final_norm < tol:
                return pi, i, final_norm
        return pi, max_iter, final_norm

    else:
        # Katz–Hubbell: prior injection at each step
        pi = mu.copy()
        final_norm = 0.0
        for i in range(1, max_iter + 1):
            pi_new = alpha * M.T.dot(pi) + (1.0 - alpha) * mu
            pi_new = np.maximum(pi_new, 0.0)
            pi_new /= pi_new.sum()
            final_norm = float(np.abs(pi_new - pi).sum())
            pi = pi_new
            if final_norm < tol:
                return pi, i, final_norm
        return pi, max_iter, final_norm


def bipartite(
    H_SI: csr_matrix,
    H_IS: csr_matrix,
    alpha: float = 1.0,
    mu: Optional[np.ndarray] = None,
    tol: float = 1e-9,
    max_iter: int = 500,
    skip_primitive_check: bool = False,
) -> tuple:
    """
    Bipartite spectral ranking via one-mode projection and power iteration.

    Builds M_S = H_SI @ H_IS (N_s × N_s) and solves for the source prestige
    vector π_S via power_iteration(), then recovers π_I.

    Mode summary
    ------------
    alpha has the same per-hop meaning as in SS (1000) and II (0001) modes:
    each traversal of one edge is attenuated by alpha.  A round trip S→I→S
    therefore attenuates by alpha², and power_iteration is called on M_S with
    alpha² as its damping argument.  This ensures v_S is directly comparable
    across modes at the same alpha.

    alpha=1, mu=None:
        π_S = Perron eigenvector of M_S (requires M_S to be primitive).
        π_I = H_SI^T @ π_S, individually normalised.

    alpha<1, mu=None:
        Original Katz on M_S; no prior injection.
        π_I = alpha · H_SI^T @ π_S, individually normalised.

    alpha<1, mu given (joint prior, length N_s + N_u):
        mu_eff = normalise(mu_S + alpha · H_IS^T @ mu_I)
        π_S from Katz–Hubbell power iteration on M_S with mu_eff.
        π_I = alpha · H_SI^T @ π_S + (1−alpha) · mu_I.
        Both individually normalised.

    Parameters
    ----------
    H_SI : csr_matrix, shape (N_s, N_u).  Row-stochastic; dangling rows as zeros.
    H_IS : csr_matrix, shape (N_u, N_s).  Row-stochastic; dangling rows as zeros.
    alpha : float in (0, 1].  Per-hop damping, same convention as SS/II modes.
    mu : joint prior, shape (N_s + N_u,), or None.  Default: None (α=1 regime).
    tol : L1 convergence tolerance (default 1e-9).
    max_iter : maximum iterations (default 500).

    Returns
    -------
    pi_s : np.ndarray, shape (N_s,), individually L1-normalised.
    pi_u : np.ndarray, shape (N_u,), individually L1-normalised.
    iters : int
    final_norm : float, L1 residual on exit
    Note: joint renormalisation (divide each by 2) is left to the caller.
    """
    N_s = H_SI.shape[0]
    N_u = H_SI.shape[1]

    # One-mode projection: M_S = H_SI @ H_IS  (N_s × N_s)
    M_S = H_SI.dot(H_IS)

    # alpha² is the round-trip damping for M_S; alpha is per-hop
    alpha_rt = alpha * alpha

    # Construct effective prior for power_iteration on M_S
    if mu is None:
        mu_eff = None
        mu_u = None
    else:
        mu_s = mu[:N_s]
        mu_u = mu[N_s:]
        mu_eff = mu_s + alpha * H_IS.T.dot(mu_u)
        s = mu_eff.sum()
        if s > 0:
            mu_eff = mu_eff / s

    # Solve for pi_S via power iteration (round-trip damping alpha²)
    pi_s, iters, final_norm = power_iteration(
        M_S, alpha_rt, mu=mu_eff, tol=tol, max_iter=max_iter,
        skip_primitive_check=skip_primitive_check,
    )

    # Recover π_I (one hop from pi_S, attenuated by alpha)
    pi_u = alpha * H_SI.T.dot(pi_s)
    if mu_u is not None:
        pi_u = pi_u + (1.0 - alpha) * mu_u

    # Individually normalise each side
    pi_s = np.maximum(pi_s, 0.0)
    pi_u = np.maximum(pi_u, 0.0)
    s_s = pi_s.sum()
    s_u = pi_u.sum()
    if s_s > 0:
        pi_s /= s_s
    if s_u > 0:
        pi_u /= s_u

    return pi_s, pi_u, iters, final_norm


def bipartite_resolvent(
    H_SI: csr_matrix,
    H_IS: csr_matrix,
    N_s: int,
    N_u: int,
    alpha: float,
) -> tuple:
    """
    Direct resolvent solve for the bipartite SI/IS case.

    Uses the sqrt(alpha) per-step convention so that round-trip attenuation
    equals alpha, matching the single-step damping of the SS and II modes.

    The system π = sqrt(α) H^T π + (1−sqrt(α)) μ partitions as:
        π_S = sqrt(α) H_IS^T π_I + (1−sqrt(α)) μ_S          (A)
        π_I = sqrt(α) H_SI^T π_S + (1−sqrt(α)) μ_I          (B)
    Substituting (B) into (A) and using the Breiger one-mode projection
    M_S = H_SI @ H_IS yields a small (N_s × N_s) linear system:
        (I − α M_S^T) π_S = (1−sqrt(α))(μ_S + sqrt(α) H_IS^T μ_I)
    solved directly via spsolve.  Institutions are then recovered from (B).

    Prior: μ_p = 1/N_P for p in P, with N_P = N_s + N_u.

    Parameters
    ----------
    H_SI : csr_matrix, shape (N_s, N_u). Row-stochastic; dangling rows as zeros.
    H_IS : csr_matrix, shape (N_u, N_s). Row-stochastic; dangling rows as zeros.
    N_s, N_u : dimensions
    alpha : float in (0, 1). Round-trip damping; per-step damping is sqrt(alpha).

    Returns
    -------
    pi_s : np.ndarray, shape (N_s,)
    pi_u : np.ndarray, shape (N_u,)
    Jointly normalised to sum to 1, non-negative.
    """
    N = N_s + N_u
    mu_s = np.full(N_s, 1.0 / N)
    mu_u = np.full(N_u, 1.0 / N)

    sqrt_alpha = np.sqrt(alpha)

    # One-mode projection M_S = H_SI @ H_IS  (N_s × N_s)
    M_S = H_SI.dot(H_IS)

    # LHS: (I − α M_S^T) — effective round-trip damping α
    A = eye(N_s, format='csc') - alpha * M_S.T.tocsc()

    # RHS: (1−sqrt(α))(μ_S + sqrt(α) H_IS^T μ_I)
    rhs = (1.0 - sqrt_alpha) * (mu_s + sqrt_alpha * H_IS.T.dot(mu_u))

    pi_s = spsolve(A, rhs)

    # Recover institutions: π_I = sqrt(α) H_SI^T π_S + (1−sqrt(α)) μ_I
    pi_u = sqrt_alpha * H_SI.T.dot(pi_s) + (1.0 - sqrt_alpha) * mu_u

    # Joint normalisation
    pi_s = np.maximum(pi_s, 0.0)
    pi_u = np.maximum(pi_u, 0.0)
    total = pi_s.sum() + pi_u.sum()
    pi_s /= total
    pi_u /= total

    return pi_s, pi_u


# ─── Top-level entry point ────────────────────────────────────────────────────

def rank(csr_data, m: tuple, chi: float, alpha: float) -> RankResult:
    """
    Row-normalise raw C blocks, assemble H per m and χ, route to algorithm.

    Parameters
    ----------
    csr_data : CSRData (from build_csr.py).  Accessed via duck typing; must
               have attributes C_SS, C_SI, C_IS, C_II, n_s, n_u, a_s, a_u.
               Blocks not needed for m may be None.
    m : (m_SS, m_SI, m_IS, m_II) block mask ∈ {0,1}^4
    chi : source–institution mixing weight ∈ [0,1]
        Only material for m=(1,1,1,1); absorbed by normalisation otherwise.
    alpha : damping factor ∈ (0,1)

    Returns
    -------
    RankResult with pi_s, pi_u, v_s, v_u, iters, final_norm.
    """
    if m == (1, 0, 0, 0):
        # Source-only: power iteration on (N_s × N_s) H_SS
        H, _ = _row_normalise(csr_data.C_SS)
        pi, iters, norm = power_iteration(H, alpha)
        A = csr_data.a_s.sum()
        v_s = A * pi / csr_data.a_s
        return RankResult(pi_s=pi, pi_u=None, v_s=v_s, v_u=None,
                          iters=iters, final_norm=norm)

    elif m == (0, 0, 0, 1):
        # Institution-only: power iteration on (N_u × N_u) H_II
        H, _ = _row_normalise(csr_data.C_II)
        pi, iters, norm = power_iteration(H, alpha)
        A = csr_data.a_u.sum()
        v_u = A * pi / csr_data.a_u
        return RankResult(pi_s=None, pi_u=pi, v_s=None, v_u=v_u,
                          iters=iters, final_norm=norm)

    elif m == (0, 1, 1, 0):
        # Bipartite SI/IS: power iteration on M_S = H_SI @ H_IS
        H_SI, _ = _row_normalise(csr_data.C_SI)
        H_IS, _ = _row_normalise(csr_data.C_IS)
        pi_s_ind, pi_u_ind, iters, norm = bipartite(H_SI, H_IS, alpha)
        # Joint-normalise (each individually sums to 1; divide by 2 to sum jointly to 1)
        pi_s = pi_s_ind / 2.0
        pi_u = pi_u_ind / 2.0
        # A = total fractional works; v=1 means average influence across both sides
        A = csr_data.a_s.sum() + csr_data.a_u.sum()
        v_s = A * pi_s / csr_data.a_s
        v_u = A * pi_u / csr_data.a_u
        return RankResult(pi_s=pi_s, pi_u=pi_u, v_s=v_s, v_u=v_u,
                          iters=iters, final_norm=norm)

    elif m == (1, 1, 1, 1):
        # Full joint: assemble χ-scaled N×N matrix, then power iteration
        C = bmat(
            [[(1 - chi) ** 2 * csr_data.C_SS,  chi * (1 - chi) * csr_data.C_SI],
             [chi * (1 - chi) * csr_data.C_IS,  chi ** 2        * csr_data.C_II]],
            format='csr',
        )
        H, _ = _row_normalise(C)
        pi, iters, norm = power_iteration(H, alpha)
        pi_s = pi[:csr_data.n_s]
        pi_u = pi[csr_data.n_s:]
        A = csr_data.a_s.sum() + csr_data.a_u.sum()
        v_s = A * pi_s / csr_data.a_s
        v_u = A * pi_u / csr_data.a_u
        return RankResult(pi_s=pi_s, pi_u=pi_u, v_s=v_s, v_u=v_u,
                          iters=iters, final_norm=norm)

    else:
        raise ValueError(f"Unsupported block mask m={m}. "
                         "Expected one of (1,0,0,0), (0,0,0,1), (0,1,1,0), (1,1,1,1).")
