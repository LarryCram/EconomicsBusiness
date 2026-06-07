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
    routes to the appropriate algorithm, returns prestige scores and v.

_row_normalise(C)
    -> (H, dangling_mask)
    Internal helper; exposed so tests and build_csr.py can use it.

_eigs_rank(M, tol=0)
    -> (phi1, lam1, lam2)
    ARPACK-based Perron eigenvector + first two eigenvalues for M^T.
    Used when alpha=1 (pure Perron iteration). Replaces check_primitive +
    power iteration: lam2 is the spectral gap diagnostic.

power_iteration(M, alpha, mu=None, tol=1e-9, max_iter=500)
    -> (pi, lam1, lam2, iters, final_norm)
    alpha=1  : calls _eigs_rank; returns (phi1, lam1, lam2, 0, 0.0).
    alpha<1  : Katz–Hubbell power iteration; returns (pi, 0.0, 0.0, iters, norm).

bipartite(H_SI, H_IS, alpha=1.0, mu=None, tol=1e-9, max_iter=500)
    -> (pi_s, pi_u, lam1, lam2, iters, final_norm)
    Bipartite Perron/Katz ranking on M_S = H_SI @ H_IS.
    alpha=1  : _eigs_rank on M_S; (lam1, lam2) from ARPACK, iters=0, norm=0.
    alpha<1  : power iteration on M_S; lam1=lam2=0.
"""

import numpy as np
from dataclasses import dataclass, field
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
    lam1: float                  # dominant eigenvalue  (0.0 for alpha<1 runs)
    lam2: float                  # second eigenvalue    (0.0 for alpha<1 runs)
    iters: int                   # power-iteration count (0 for alpha=1 eigs path)
    final_norm: float            # L1 residual at convergence (0.0 for alpha=1 eigs path)


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


# ─── ARPACK eigensolver ───────────────────────────────────────────────────────

def _eigs_rank(M: csr_matrix, tol: float = 0) -> tuple:
    """
    Compute (phi1, lam1, lam2) for M^T via ARPACK eigs.

    Parameters
    ----------
    M   : row-stochastic CSR matrix (dangling rows as zero rows).
    tol : ARPACK relative tolerance. Default 0 = machine precision.

    Returns
    -------
    phi1 : Perron vector, L1-normalised, non-negative, shape (N,).
    lam1 : dominant eigenvalue (real part), should be ≈ 1.0.
    lam2 : second eigenvalue (real part). |lam2| < 1 confirms primitivity.

    Notes
    -----
    Logs a WARNING if |lam2| > 0.999 (near-imprimitive or reducible).
    Falls back to numpy.linalg.eig for N < 3 (ARPACK needs N > k=2).
    Falls back to single eigenvector if ARPACK does not converge on k=2.
    """
    from scipy.sparse.linalg import eigs as sp_eigs, ArpackNoConvergence

    N = M.shape[0]

    if N < 4:
        # Dense fallback: ARPACK eigs requires k < N-1, so N >= 4 for k=2
        A = M.T.toarray()
        vals, vecs = np.linalg.eig(A)
        order = np.argsort(-vals.real)
        lam1  = float(vals[order[0]].real)
        phi1  = vecs[:, order[0]].real
        lam2  = float(vals[order[1]].real) if N >= 2 else 0.0
    else:
        try:
            vals, vecs = sp_eigs(M.T, k=2, which='LM', tol=tol)
            order = np.argsort(-vals.real)
            lam1  = float(vals[order[0]].real)
            phi1  = vecs[:, order[0]].real
            lam2  = float(vals[order[1]].real)
        except ArpackNoConvergence as exc:
            # Use whatever converged eigenpairs are available
            cv = exc.eigenvalues
            cv_vecs = exc.eigenvectors
            if len(cv) >= 1:
                order = np.argsort(-cv.real)
                lam1  = float(cv[order[0]].real)
                phi1  = cv_vecs[:, order[0]].real
                lam2  = float(cv[order[1]].real) if len(cv) >= 2 else 0.0
                print(f'  WARNING: ARPACK did not fully converge (N={N}); '
                      f'using {len(cv)} converged eigenpairs.', flush=True)
            else:
                # Total failure: fall back to dense
                print(f'  WARNING: ARPACK failed entirely (N={N}); '
                      f'falling back to dense eigensolver.', flush=True)
                A = M.T.toarray()
                vals_d, vecs_d = np.linalg.eig(A)
                order = np.argsort(-vals_d.real)
                lam1  = float(vals_d[order[0]].real)
                phi1  = vecs_d[:, order[0]].real
                lam2  = float(vals_d[order[1]].real) if N >= 2 else 0.0

    # Sign convention: component of largest absolute value is positive
    peak = np.argmax(np.abs(phi1))
    if phi1[peak] < 0:
        phi1 = -phi1
    phi1 = np.maximum(phi1, 0.0)
    s = phi1.sum()
    if s > 0:
        phi1 /= s

    if abs(lam2) > 0.999:
        print(f'  WARNING: |λ₂| = {abs(lam2):.6f}  '
              f'spectral_gap = {1.0 - abs(lam2):.2e}  N={N}', flush=True)

    return phi1, lam1, lam2


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


def power_iteration(
    M: csr_matrix,
    alpha: float,
    mu: Optional[np.ndarray] = None,
    tol: float = 1e-9,
    max_iter: int = 500,
) -> tuple:
    """
    Power iteration / Perron eigenvector computation.

    Three valid regimes
    -------------------
    (alpha=1, mu=None)   Pure Perron iteration via _eigs_rank (ARPACK).
                         Returns (phi1, lam1, lam2, 0, 0.0).
    (alpha<1, mu=None)   Original Katz damping.  No prior injection.
                         Returns (pi, 0.0, 0.0, iters, final_norm).
    (alpha<1, mu>0)      Katz–Hubbell.  Prior injection at each step.
                         Returns (pi, 0.0, 0.0, iters, final_norm).

    Parameters
    ----------
    M : csr_matrix, shape (N, N).  Row-stochastic; dangling rows as zero rows.
    alpha : float in [0, 1].  Damping factor.
    mu : prior, shape (N,), or None.
    tol : convergence tolerance (default 1e-9).
    max_iter : maximum iterations for alpha<1 path (default 500).

    Returns
    -------
    pi         : np.ndarray, shape (N,), L1-normalised, non-negative
    lam1       : float — dominant eigenvalue from ARPACK (0.0 for alpha<1)
    lam2       : float — second eigenvalue from ARPACK  (0.0 for alpha<1)
    iters      : int   — power-iteration count (0 for alpha=1 eigs path)
    final_norm : float — L1 residual (0.0 for alpha=1 eigs path)
    """
    N = M.shape[0]

    if alpha == 1.0:
        if mu is not None:
            raise ValueError(
                "mu must be None when alpha=1 (pure Perron iteration). "
                "Use alpha<1 for Katz–Hubbell."
            )
        phi1, lam1, lam2 = _eigs_rank(M, tol=tol)
        return phi1, lam1, lam2, 0, 0.0

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
                return pi, 0.0, 0.0, i, final_norm
        return pi, 0.0, 0.0, max_iter, final_norm

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
                return pi, 0.0, 0.0, i, final_norm
        return pi, 0.0, 0.0, max_iter, final_norm


def bipartite(
    H_SI: csr_matrix,
    H_IS: csr_matrix,
    alpha: float = 1.0,
    mu: Optional[np.ndarray] = None,
    tol: float = 1e-9,
    max_iter: int = 500,
) -> tuple:
    """
    Bipartite spectral ranking via one-mode projection.

    Builds M_S = H_SI @ H_IS (N_s × N_s) and solves for the source prestige
    vector π_S, then recovers π_I.

    alpha=1, mu=None:
        π_S via _eigs_rank on M_S (ARPACK). Returns lam1, lam2 of M_S.
        π_I = H_SI^T @ π_S, individually normalised.

    alpha<1, mu=None:
        Original Katz on M_S with round-trip damping alpha².
        π_I = alpha · H_SI^T @ π_S, individually normalised.

    alpha<1, mu given (joint prior, length N_s + N_u):
        Effective prior mu_eff on sources; Katz–Hubbell power iteration.
        π_I = alpha · H_SI^T @ π_S + (1−alpha) · mu_I.

    Parameters
    ----------
    H_SI : csr_matrix, shape (N_s, N_u). Row-stochastic; dangling rows as zeros.
    H_IS : csr_matrix, shape (N_u, N_s). Row-stochastic; dangling rows as zeros.
    alpha : float in (0, 1]. Per-hop damping (round-trip is alpha²).
    mu : joint prior, shape (N_s + N_u,), or None.
    tol : convergence tolerance (default 1e-9).
    max_iter : maximum iterations for alpha<1 path (default 500).

    Returns
    -------
    pi_s       : np.ndarray, shape (N_s,), individually L1-normalised.
    pi_u       : np.ndarray, shape (N_u,), individually L1-normalised.
    lam1       : float — dominant eigenvalue of M_S (0.0 for alpha<1)
    lam2       : float — second eigenvalue of M_S  (0.0 for alpha<1)
    iters      : int   — power-iteration count (0 for alpha=1 eigs path)
    final_norm : float — L1 residual (0.0 for alpha=1 eigs path)
    """
    N_s = H_SI.shape[0]
    N_u = H_SI.shape[1]

    if alpha < 1.0 and mu is None:
        raise ValueError(
            "bipartite(): mu must be provided when alpha < 1. "
            "Use mu_type='uniform' (μ = 1/N for all N_s+N_u units) or "
            "'unit_scaled' (μ = 1/(2N_S) for sources, 1/(2N_I) for institutions, sums to 1)."
        )

    # One-mode projection: M_S = H_SI @ H_IS  (N_s × N_s)
    M_S = H_SI.dot(H_IS)

    # alpha² is the round-trip damping for M_S; alpha is per-hop
    alpha_rt = alpha * alpha

    # Construct effective prior for power_iteration on M_S
    if mu is None:
        mu_eff = None
        mu_u   = None
    else:
        mu_s = mu[:N_s]
        mu_u = mu[N_s:]
        mu_eff = mu_s + alpha * H_IS.T.dot(mu_u)
        s = mu_eff.sum()
        if s > 0:
            mu_eff = mu_eff / s

    # Solve for pi_S
    pi_s, lam1, lam2, iters, final_norm = power_iteration(
        M_S, alpha_rt, mu=mu_eff, tol=tol, max_iter=max_iter,
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

    return pi_s, pi_u, lam1, lam2, iters, final_norm


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

def rank(csr_data, m: tuple, chi: float, alpha: float,
         mu: Optional[np.ndarray] = None) -> RankResult:
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
    alpha : damping factor ∈ (0,1]
    mu : prior vector, or None.
        For m=(0,1,1,0): joint vector of length n_s + n_u.  Required when
        alpha < 1.  Ignored (must be None) when alpha = 1.
        Two standard choices (constructed by run_rankings._make_mu):
          'uniform'     — μ_p = 1/(N_S+N_U)  for all units  (Katz)
          'unit_scaled' — μ_p = 1/(2N_S) for sources, 1/(2N_I) for institutions

    Returns
    -------
    RankResult with pi_s, pi_u, v_s, v_u, lam1, lam2, iters, final_norm.
    """
    # Sentinel masks (None for ε=0 runs; bool arrays for ε=1 runs).
    is_s = csr_data.is_sentinel_s
    is_u = csr_data.is_sentinel_u

    def _A_real_s():
        return csr_data.a_s[~is_s].sum() if (is_s is not None and is_s.any()) else csr_data.a_s.sum()

    def _A_real_u():
        return csr_data.a_u[~is_u].sum() if (is_u is not None and is_u.any()) else csr_data.a_u.sum()

    def _mask_nan(v, mask):
        if mask is not None and mask.any():
            v = v.astype(float, copy=True)
            v[mask] = np.nan
        return v

    def _pi_real(pi, mask):
        # Sum of π over real units only.  For ε=0 (mask=None) this is 1.0 and
        # cancels; for ε=1 it renormalises v so the a-weighted mean over real
        # units equals 1, matching the baseline scale.
        return pi[~mask].sum() if (mask is not None and mask.any()) else pi.sum()

    if m == (1, 0, 0, 0):
        # Source-only: power iteration on (N_s × N_s) H_SS
        H, _ = _row_normalise(csr_data.C_SS)
        pi, lam1, lam2, iters, norm = power_iteration(H, alpha)
        A       = _A_real_s()
        pi_real = _pi_real(pi, is_s)
        v_s = _mask_nan(A * pi / (pi_real * csr_data.a_s), is_s)
        return RankResult(pi_s=pi, pi_u=None, v_s=v_s, v_u=None,
                          lam1=lam1, lam2=lam2, iters=iters, final_norm=norm)

    elif m == (0, 0, 0, 1):
        # Institution-only: power iteration on (N_u × N_u) H_II
        H, _ = _row_normalise(csr_data.C_II)
        pi, lam1, lam2, iters, norm = power_iteration(H, alpha)
        A       = _A_real_u()
        pi_real = _pi_real(pi, is_u)
        v_u = _mask_nan(A * pi / (pi_real * csr_data.a_u), is_u)
        return RankResult(pi_s=None, pi_u=pi, v_s=None, v_u=v_u,
                          lam1=lam1, lam2=lam2, iters=iters, final_norm=norm)

    elif m == (0, 1, 1, 0):
        # Bipartite SI/IS: power iteration on M_S = H_SI @ H_IS
        H_SI, _ = _row_normalise(csr_data.C_SI)
        H_IS, _ = _row_normalise(csr_data.C_IS)
        pi_s_ind, pi_u_ind, lam1, lam2, iters, norm = bipartite(H_SI, H_IS, alpha, mu=mu)
        # Joint-normalise (each individually sums to 1; divide by 2 to sum jointly to 1)
        pi_s = pi_s_ind / 2.0
        pi_u = pi_u_ind / 2.0
        A       = _A_real_s() + _A_real_u()
        pi_real = _pi_real(pi_s, is_s) + _pi_real(pi_u, is_u)
        v_s = _mask_nan(A * pi_s / (pi_real * csr_data.a_s), is_s)
        v_u = _mask_nan(A * pi_u / (pi_real * csr_data.a_u), is_u)
        return RankResult(pi_s=pi_s, pi_u=pi_u, v_s=v_s, v_u=v_u,
                          lam1=lam1, lam2=lam2, iters=iters, final_norm=norm)

    elif m == (1, 1, 1, 1):
        # Full joint: assemble χ-scaled N×N matrix, then power iteration
        C = bmat(
            [[csr_data.C_SS,                    chi * (1 - chi) * csr_data.C_SI],
             [chi * (1 - chi) * csr_data.C_IS,  csr_data.C_II                  ]],
            format='csr',
        )
        H, _ = _row_normalise(C)
        pi, lam1, lam2, iters, norm = power_iteration(H, alpha)
        pi_s = pi[:csr_data.n_s]
        pi_u = pi[csr_data.n_s:]
        A       = _A_real_s() + _A_real_u()
        pi_real = _pi_real(pi_s, is_s) + _pi_real(pi_u, is_u)
        v_s = _mask_nan(A * pi_s / (pi_real * csr_data.a_s), is_s)
        v_u = _mask_nan(A * pi_u / (pi_real * csr_data.a_u), is_u)
        return RankResult(pi_s=pi_s, pi_u=pi_u, v_s=v_s, v_u=v_u,
                          lam1=lam1, lam2=lam2, iters=iters, final_norm=norm)

    else:
        raise ValueError(f"Unsupported block mask m={m}. "
                         "Expected one of (1,0,0,0), (0,0,0,1), (0,1,1,0), (1,1,1,1).")
