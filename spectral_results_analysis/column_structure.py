"""
column_structure.py — Column-sum concentration diagnostics per kernel.

The column sums of a row-stochastic H are the first iterate of power iteration
from uniform: s_j = sum_i H_ij. Their Gini coefficient is the primary predictor
of π-shape (high Gini → steep ranking). Mean row perplexity is a second-order
modifier (high perplexity washes out column asymmetry).

Kernels computed: H_SS, H_II, H_SI, H_IS, M_S = H_SI@H_IS, M_I = H_IS@H_SI.

Outputs:
  plots/column_structure.csv  — one row per kernel with scalar diagnostics
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


# ─── Statistics ───────────────────────────────────────────────────────────────

def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative array."""
    x = np.sort(x[x > 0])
    n = len(x)
    if n == 0:
        return float('nan')
    cumx = np.cumsum(x)
    return float((2 * np.dot(np.arange(1, n + 1), x) - (n + 1) * cumx[-1]) / (n * cumx[-1]))


def _row_perplexity(H) -> np.ndarray:
    """Effective branching m_i = exp(row entropy) for each active row."""
    H = H.tocsr()
    n = H.shape[0]
    indptr, data = H.indptr, H.data
    row_idx = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    contrib = np.zeros_like(data)
    nz = data > 0
    contrib[nz] = -data[nz] * np.log10(data[nz])
    h = np.bincount(row_idx, weights=contrib, minlength=n).astype(float)
    active = np.asarray(H.sum(axis=1)).ravel() > 0
    m = np.full(n, np.nan)
    m[active] = 10 ** h[active]
    return m


def _col_sums(H) -> np.ndarray:
    return np.asarray(H.sum(axis=0)).ravel()


def _kernel_stats(name: str, H) -> dict:
    cs = _col_sums(H)
    m  = _row_perplexity(H)
    m_active = m[~np.isnan(m)]
    return {
        'kernel':        name,
        'n_rows':        H.shape[0],
        'n_cols':        H.shape[1],
        'gini_col_sum':  _gini(cs),
        'mean_col_sum':  float(cs[cs > 0].mean()) if (cs > 0).any() else np.nan,
        'max_col_sum':   float(cs.max()),
        'mean_row_perp': float(m_active.mean())   if len(m_active) else np.nan,
        'std_row_perp':  float(m_active.std())    if len(m_active) else np.nan,
        'median_row_perp': float(np.median(m_active)) if len(m_active) else np.nan,
    }


# ─── Kernels ──────────────────────────────────────────────────────────────────

def build_kernels(db, run_code, fx, tau_u, tau_s, rho):
    csr_ss = build_csr(db, run_code, fx, tau_u, tau_s, rho, (1, 0, 0, 0))
    H_SS, _ = _row_normalise(csr_ss.C_SS)

    csr_ii = build_csr(db, run_code, fx, tau_u, tau_s, rho, (0, 0, 0, 1))
    H_II, _ = _row_normalise(csr_ii.C_II)

    csr_bi = build_csr(db, run_code, fx, tau_u, tau_s, rho, (0, 1, 1, 0))
    H_SI, _ = _row_normalise(csr_bi.C_SI)
    H_IS, _ = _row_normalise(csr_bi.C_IS)
    M_S = H_SI.dot(H_IS).tocsr()
    M_I = H_IS.dot(H_SI).tocsr()

    return [
        ('H_SS', H_SS),
        ('H_II', H_II),
        ('H_SI', H_SI),
        ('H_IS', H_IS),
        ('M_S',  M_S),
        ('M_I',  M_I),
    ]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    paths    = load_config()
    baseline = next(r for r in load_runs() if r['label'] == 'baseline')
    run_code = baseline['run_code']
    tau_u    = baseline['tau_u']
    tau_s    = baseline['tau_s']
    fx       = baseline['fx']
    rho      = 0

    el_path = paths.working / 'edge_lists.duckdb'
    print(f'Loading kernels from {el_path}')
    with duckdb.connect(str(el_path), read_only=True) as db:
        kernels = build_kernels(db, run_code, fx, tau_u, tau_s, rho)

    rows = [_kernel_stats(name, H) for name, H in kernels]
    df = pd.DataFrame(rows)

    print('\n' + df.to_string(index=False, float_format='{:.4f}'.format))

    out = paths.plots / 'column_structure.csv'
    df.to_csv(out, index=False)
    print(f'\nSaved {out}')

    # Key ratio: mean row perplexity II vs SS — the "3.5x" generalised
    m_ss = df.loc[df['kernel'] == 'H_SS', 'mean_row_perp'].values[0]
    m_ii = df.loc[df['kernel'] == 'H_II', 'mean_row_perp'].values[0]
    print(f'\nm_bar(H_II) / m_bar(H_SS) = {m_ii / m_ss:.3f}')

    g_ss = df.loc[df['kernel'] == 'H_SS', 'gini_col_sum'].values[0]
    g_ms = df.loc[df['kernel'] == 'M_S',  'gini_col_sum'].values[0]
    print(f'Gini(col_sum M_S) / Gini(col_sum H_SS) = {g_ms / g_ss:.3f}  '
          f'[> 1 → bipartite steepens; < 1 → flattens]')


if __name__ == '__main__':
    main()
