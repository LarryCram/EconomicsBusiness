"""
table_kernel_structure.py — Row-dispersion / column-concentration diagnostics.

Combines: column_structure.py + network_entropy.py

Builds all six stochastic kernels for the baseline run and computes per-kernel
statistics that motivate the paper's discussion of v^S vs v^B and v^I vs v^B.

  K̄_row  = mean row perplexity = mean exp(-Σ_j H_ij log H_ij)
             effective fan-out of a single Markov step

  G_col   = Gini coefficient of column sums d_j = Σ_i H_ij
             concentration of inward references

Outputs
-------
plots/kernel_structure.csv   full scalar diagnostics per kernel
plots/kernel_structure.tex   LaTeX 3-column table (paper-ready)
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


# ── Statistics ────────────────────────────────────────────────────────────────

def _j_col(cs: np.ndarray, n_cols: int) -> float:
    """Column evenness: normalised entropy of column-sum distribution."""
    total = cs.sum()
    if total == 0 or n_cols <= 1:
        return float('nan')
    p = cs / total
    nz = p > 0
    h = float(-np.dot(p[nz], np.log10(p[nz])))
    return h / np.log10(n_cols)


def _row_perplexity(H) -> np.ndarray:
    """Per-row perplexity K_i = exp(-Σ_j H_ij log H_ij) for CSR matrix H."""
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


def _kernel_stats(name: str, H) -> dict:
    cs       = np.asarray(H.sum(axis=0)).ravel()
    m        = _row_perplexity(H)
    m_active = m[~np.isnan(m)]
    n_cols   = H.shape[1]

    # J_bar = mean row evenness = mean(log K_i) / log n_cols
    # K_bar = n_cols^J_bar = geometric mean of per-row perplexities
    if len(m_active) > 0 and n_cols > 1:
        log_k_mean = float(np.mean(np.log10(m_active)))
        J_bar = log_k_mean / np.log10(n_cols)
        K_bar = float(10 ** log_k_mean)   # = n_cols ** J_bar
    else:
        J_bar = K_bar = np.nan
    return {
        'kernel':          name,
        'n_rows':          H.shape[0],
        'n_cols':          n_cols,
        'J_bar':           J_bar,
        'J_col':           _j_col(cs, n_cols),
        'K_bar':           K_bar,
        'mean_row_perp':   float(m_active.mean())     if len(m_active) else np.nan,
        'median_row_perp': float(np.median(m_active)) if len(m_active) else np.nan,
        'max_col_sum':     float(cs.max()),
        'mean_col_sum':    float(cs[cs > 0].mean())   if (cs > 0).any() else np.nan,
    }


# ── Kernel building ───────────────────────────────────────────────────────────

def build_kernels(db, run_code, fx, tau_u, tau_s, rho):
    csr_ss = build_csr(db, run_code, fx, tau_u, tau_s, rho, (1, 0, 0, 0))
    C_SS, _ = _row_normalise(csr_ss.C_SS)

    csr_ii = build_csr(db, run_code, fx, tau_u, tau_s, rho, (0, 0, 0, 1))
    C_II, _ = _row_normalise(csr_ii.C_II)

    csr_bi = build_csr(db, run_code, fx, tau_u, tau_s, rho, (0, 1, 1, 0))
    C_SI, _ = _row_normalise(csr_bi.C_SI)
    C_IS, _ = _row_normalise(csr_bi.C_IS)
    M_S = C_SI.dot(C_IS).tocsr()
    M_I = C_IS.dot(C_SI).tocsr()

    return [
        ('C_SS', C_SS),
        ('C_II', C_II),
        ('C_SI', C_SI),
        ('C_IS', C_IS),
        ('M_S',  M_S),
        ('M_I',  M_I),
    ]


# ── LaTeX table ───────────────────────────────────────────────────────────────

_LATEX_NAME = {
    'C_SS': r'$C_{SS}$',
    'C_II': r'$C_{II}$',
    'C_SI': r'$C_{SI}$',
    'C_IS': r'$C_{IS}$',
    'M_S':  r'$M_S$',
    'M_I':  r'$M_I$',
}


def make_latex_table(df: pd.DataFrame) -> str:
    lines = [
        r'\begin{tabular}{lccc}',
        r'\toprule',
        r'Kernel & $\bar{J}$ & $J_{\text{col}}$ & $\bar{K}$ \\',
        r'\midrule',
    ]
    sub = {'C_SS', 'C_II', 'C_SI', 'C_IS'}
    comp = {'M_S', 'M_I'}
    for _, row in df.iterrows():
        if row['kernel'] in comp and lines[-1] != r'\midrule':
            lines.append(r'\midrule')
        name = _LATEX_NAME.get(row['kernel'], row['kernel'])
        j    = f"{row['J_bar']:.2f}"
        jc   = f"{row['J_col']:.2f}"
        k    = f"{round(row['K_bar'])}"
        lines.append(rf'{name} & {j} & {jc} & {k} \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    paths    = load_config()
    baseline = next(r for r in load_runs() if r['label'] == 'baseline')
    run_code = baseline['run_code']
    tau_u    = baseline['tau_u']
    tau_s    = baseline['tau_s']
    fx       = baseline['fx']

    el_path = paths.working / 'edge_lists.duckdb'
    print(f'Loading kernels from {el_path}')
    with duckdb.connect(str(el_path), read_only=True) as db:
        kernels = build_kernels(db, run_code, fx, tau_u, tau_s, rho=0)

    df = pd.DataFrame([_kernel_stats(name, H) for name, H in kernels])

    print('\n' + df[['kernel', 'n_rows', 'n_cols',
                      'J_bar', 'J_col', 'K_bar']].to_string(
        index=False, float_format='{:.4f}'.format))

    out_csv = paths.plots / 'kernel_structure.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nSaved {out_csv}')

    out_tex = paths.tables / 'table_kernel_structure.tex'
    out_tex.write_text(make_latex_table(df))
    print(f'Saved {out_tex}')

    m_ss = df.loc[df.kernel == 'C_SS', 'mean_row_perp'].values[0]
    m_ii = df.loc[df.kernel == 'C_II', 'mean_row_perp'].values[0]
    print(f'\nm̄(C_II) / m̄(C_SS) = {m_ii / m_ss:.3f}')

    jc_ss = df.loc[df.kernel == 'C_SS', 'J_col'].values[0]
    jc_ms = df.loc[df.kernel == 'M_S',  'J_col'].values[0]
    print(f'J_col(M_S) / J_col(C_SS) = {jc_ms / jc_ss:.3f}'
          '  [<1 → two-layer steepens column concentration]')

    k_ss = df.loc[df.kernel == 'C_SS', 'K_bar'].values[0]
    k_ii = df.loc[df.kernel == 'C_II', 'K_bar'].values[0]
    print(f'K_bar(C_II) / K_bar(C_SS) = {k_ii / k_ss:.2f}x  [geometric fan-out ratio]')


if __name__ == '__main__':
    main()
