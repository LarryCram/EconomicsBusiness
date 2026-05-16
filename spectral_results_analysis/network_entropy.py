"""
network_entropy.py — Diffusion diagnostics via row-wise Shannon entropy.

Purpose
-------
Quantify how diffusely connected the direct source network (H_SS),
direct institution network (H_II), and institution-mediated source
network (M_S = H_SI @ H_IS) are in the baseline run.

For each row-stochastic kernel H, we compute row entropy
  h_i = -sum_j H_ij log(H_ij),
and normalized row entropy
  h_i_norm = h_i / log(n),
where n is the number of nodes in that kernel.

We also report an "effective branching" count per row,
  m_i = exp(h_i),
interpretable as the effective number of destinations reached from row i.

Outputs
-------
plots/network_entropy_summary.csv
  One row per kernel with scalar diagnostics.
plots/network_entropy_rows.csv
  One row per node-row with row-level entropy diagnostics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs
from spectral_ranking.build_csr import build_csr
from spectral_ranking.katz_ranker import _row_normalise


def _row_entropy(H) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row entropy h_i and active-row mask for CSR matrix H."""
    H = H.tocsr()
    n = H.shape[0]

    indptr = H.indptr
    data = H.data
    row_idx = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))

    contrib = np.zeros_like(data)
    nz = data > 0
    contrib[nz] = -data[nz] * np.log10(data[nz])

    h = np.bincount(row_idx, weights=contrib, minlength=n).astype(float)
    row_sum = np.asarray(H.sum(axis=1)).ravel()
    active = row_sum > 0
    return h, active


def _stationary_weights(H, active: np.ndarray) -> np.ndarray:
    """Approximate stationary weights for H via dominant eigenvector of H^T."""
    n = H.shape[0]
    w = np.zeros(n, dtype=float)

    if active.any():
        w[active] = 1.0 / active.sum()

    try:
        from scipy.sparse.linalg import eigs

        vals, vecs = eigs(H.T, k=1, which='LM')
        _ = vals  # silence linters; value not needed directly
        v = np.real(vecs[:, 0])
        if np.abs(v).sum() == 0:
            return w
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        v = np.clip(v, 0.0, None)
        s = float(v.sum())
        if s <= 0:
            return w
        v /= s
        if np.isfinite(v).all() and v.sum() > 0:
            return v
    except Exception as exc:
        print(f"  Warning: stationary-weight fallback to uniform ({exc})")

    return w


def _kernel_metrics(name: str, H, node_ids: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Compute row-level and summary entropy metrics for one kernel."""
    n = H.shape[0]
    h, active = _row_entropy(H)
    log_n = np.log10(n) if n > 1 else 1.0

    h_norm = np.zeros(n, dtype=float)
    if n > 1:
        h_norm = h / log_n

    m_eff = 10 ** h
    m_eff_frac = m_eff / max(n, 1)

    w_uniform = np.zeros(n, dtype=float)
    if active.any():
        w_uniform[active] = 1.0 / active.sum()

    w_stat = _stationary_weights(H, active)

    def _wmean(x, w):
        s = float(w.sum())
        if s <= 0:
            return np.nan
        return float(np.dot(x, w) / s)

    active_h_norm = h_norm[active]
    active_m_eff = m_eff[active]

    summary = {
        'kernel': name,
        'n_nodes': int(n),
        'n_active_rows': int(active.sum()),
        'dangling_rows': int((~active).sum()),
        'dangling_share': float((~active).mean()),
        'mean_h_norm_uniform': _wmean(h_norm, w_uniform),
        'mean_h_norm_stationary': _wmean(h_norm, w_stat),
        'median_h_norm_active': float(np.median(active_h_norm)) if active.any() else np.nan,
        'q10_h_norm_active': float(np.quantile(active_h_norm, 0.10)) if active.any() else np.nan,
        'q90_h_norm_active': float(np.quantile(active_h_norm, 0.90)) if active.any() else np.nan,
        'mean_m_eff_uniform': _wmean(m_eff, w_uniform),
        'mean_m_eff_stationary': _wmean(m_eff, w_stat),
        'median_m_eff_active': float(np.median(active_m_eff)) if active.any() else np.nan,
        'mean_m_eff_frac_uniform': _wmean(m_eff_frac, w_uniform),
        'entropy_rate_raw_stationary': _wmean(h, w_stat),
        'entropy_rate_norm_stationary': _wmean(h_norm, w_stat),
    }

    rows = pd.DataFrame({
        'kernel': name,
        'node_idx': node_ids.astype(int),
        'active_row': active,
        'row_entropy': h,
        'row_entropy_norm': h_norm,
        'effective_branching': m_eff,
        'effective_branching_frac': m_eff_frac,
        'weight_stationary': w_stat,
    })

    return rows, summary


def _build_kernels(db, run_code: str, fx: str, tau_u: int, tau_s: int, rho: int):
    """Build baseline H_SS, H_II, and M_S kernels from edge-list DB."""
    csr_ss = build_csr(db, run_code, fx, tau_u, tau_s, rho, (1, 0, 0, 0))
    H_ss, _ = _row_normalise(csr_ss.C_SS)

    csr_ii = build_csr(db, run_code, fx, tau_u, tau_s, rho, (0, 0, 0, 1))
    H_ii, _ = _row_normalise(csr_ii.C_II)

    csr_bi = build_csr(db, run_code, fx, tau_u, tau_s, rho, (0, 1, 1, 0))
    H_SI, _ = _row_normalise(csr_bi.C_SI)
    H_IS, _ = _row_normalise(csr_bi.C_IS)
    M_s = H_SI.dot(H_IS).tocsr()

    return (
        ('H_SS', H_ss, csr_ss.source_ids),
        ('H_II', H_ii, csr_ii.inst_ids),
        ('M_S', M_s, csr_bi.source_ids),
    )


def main() -> None:
    paths = load_config()
    baseline = next(r for r in load_runs() if r['label'] == 'baseline')

    run_code = baseline['run_code']
    tau_u = baseline['tau_u']
    tau_s = baseline['tau_s']
    fx = baseline['fx']
    rho = 0

    el_path = paths.working / 'edge_lists.duckdb'
    print(f"Loading kernels from {el_path}")

    with duckdb.connect(str(el_path), read_only=True) as db:
        kernels = _build_kernels(db, run_code, fx, tau_u, tau_s, rho)

    row_frames = []
    summary_rows = []
    for name, H, node_ids in kernels:
        rows, summary = _kernel_metrics(name, H, node_ids)
        row_frames.append(rows)
        summary_rows.append(summary)

    df_rows = pd.concat(row_frames, ignore_index=True)
    df_summary = pd.DataFrame(summary_rows)

    out_summary = paths.plots / 'network_entropy_summary.csv'
    out_rows = paths.plots / 'network_entropy_rows.csv'

    df_summary.to_csv(out_summary, index=False)
    df_rows.to_csv(out_rows, index=False)

    print('\nSummary diagnostics')
    print(df_summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f"\nSaved {out_summary}")
    print(f"Saved {out_rows}")


if __name__ == '__main__':
    main()
