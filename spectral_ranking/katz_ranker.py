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


def bipartite_resolvent(
    H_SI: csr_matrix,
    H_IS: csr_matrix,
    N_s: int,
    N_u: int,
    alpha: float,
) -> tuple:
    """
    Direct resolvent solve for the bipartite SI/IS case.

    The block-off-diagonal system π = αH^T π + (1−α)μ partitions as:
        π_S = α H_IS^T π_I + (1−α) μ_S          (A)
        π_I = α H_SI^T π_S + (1−α) μ_I          (B)
    Substituting (B) into (A) and using the Breiger one-mode projection
    M_S = H_SI @ H_IS yields a small (N_s × N_s) linear system:
        (I − α² M_S^T) π_S = (1−α)(μ_S + α H_IS^T μ_I)
    solved directly via spsolve.  Institutions are then recovered from (B).

    Prior: μ_p = 1/N for all p, with N = N_s + N_u (eq. 5 of paper).

    Parameters
    ----------
    H_SI : csr_matrix, shape (N_s, N_u). Row-stochastic; dangling rows as zeros.
    H_IS : csr_matrix, shape (N_u, N_s). Row-stochastic; dangling rows as zeros.
    N_s, N_u : dimensions
    alpha : float in (0, 1)

    Returns
    -------
    pi_s : np.ndarray, shape (N_s,)
    pi_u : np.ndarray, shape (N_u,)
    Jointly normalised to sum to 1, non-negative.
    """
    N = N_s + N_u
    mu_s = np.full(N_s, 1.0 / N)
    mu_u = np.full(N_u, 1.0 / N)

    # One-mode projection M_S = H_SI @ H_IS  (N_s × N_s)
    M_S = H_SI.dot(H_IS)

    # LHS: (I − α² M_S^T)  in CSC for direct solver
    A = eye(N_s, format='csc') - (alpha ** 2) * M_S.T.tocsc()

    # RHS: (1−α)(μ_S + α H_IS^T μ_I)
    rhs = (1.0 - alpha) * (mu_s + alpha * H_IS.T.dot(mu_u))

    pi_s = spsolve(A, rhs)

    # Recover institutions: π_I = α H_SI^T π_S + (1−α) μ_I
    pi_u = alpha * H_SI.T.dot(pi_s) + (1.0 - alpha) * mu_u

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
        # Source-only: SS Katz on (N_s × N_s) matrix
        H, _ = _row_normalise(csr_data.C_SS)
        N_s = csr_data.n_s
        mu = np.full(N_s, 1.0 / N_s)
        pi, iters, norm = katz(H, N_s, alpha, mu=mu)
        A = csr_data.a_s.sum()
        v_s = A * pi / csr_data.a_s
        return RankResult(pi_s=pi, pi_u=None, v_s=v_s, v_u=None,
                          iters=iters, final_norm=norm)

    elif m == (0, 0, 0, 1):
        # Institution-only: II Katz on (N_u × N_u) matrix
        # Prior is uniform over institutions only; no ω involvement.
        H, _ = _row_normalise(csr_data.C_II)
        N_u = csr_data.n_u
        mu = np.full(N_u, 1.0 / N_u)
        pi, iters, norm = katz(H, N_u, alpha, mu=mu)
        A = csr_data.a_u.sum()
        v_u = A * pi / csr_data.a_u
        return RankResult(pi_s=None, pi_u=pi, v_s=None, v_u=v_u,
                          iters=iters, final_norm=norm)

    elif m == (0, 1, 1, 0):
        # Bipartite SI/IS: direct resolvent, no iteration
        H_SI, _ = _row_normalise(csr_data.C_SI)
        H_IS, _ = _row_normalise(csr_data.C_IS)
        pi_s, pi_u = bipartite_resolvent(H_SI, H_IS, csr_data.n_s, csr_data.n_u, alpha)
        # A = total fractional works across all units in the joint ranking
        A = csr_data.a_s.sum() + csr_data.a_u.sum()
        v_s = A * pi_s / csr_data.a_s
        v_u = A * pi_u / csr_data.a_u
        return RankResult(pi_s=pi_s, pi_u=pi_u, v_s=v_s, v_u=v_u,
                          iters=0, final_norm=0.0)

    elif m == (1, 1, 1, 1):
        # Full joint: assemble χ-scaled N×N matrix, then Katz
        C = bmat(
            [[(1 - chi) ** 2 * csr_data.C_SS,  chi * (1 - chi) * csr_data.C_SI],
             [chi * (1 - chi) * csr_data.C_IS,  chi ** 2        * csr_data.C_II]],
            format='csr',
        )
        H, _ = _row_normalise(C)
        N = csr_data.n_s + csr_data.n_u
        mu = np.full(N, 1.0 / N)
        pi, iters, norm = katz(H, N, alpha, mu=mu)
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
