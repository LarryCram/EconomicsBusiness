"""
fig_jbar_hist.py — Distributions of per-row evenness J_i and column marginal p_j.

For each of the six baseline kernels (C_SS, C_II, C_SI, C_IS, M_S, M_I):
  top row:    histogram of J_i = row entropy / log(n_col)  (one value per row)
  bottom row: histogram of p_j = col_sum_j / n_rows        (one value per col, log x-axis)

Output: plots/fig_jbar_hist.pdf
"""

from __future__ import annotations
import sys
from pathlib import Path

import duckdb
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs
from spectral_ranking.build_csr import build_csr
from spectral_ranking.katz_ranker import _row_normalise


def _row_ji(H) -> np.ndarray:
    """J_i = row entropy / log(n_col) for each active row."""
    H = H.tocsr()
    n_rows, n_cols = H.shape
    data = H.data
    indptr = H.indptr
    row_idx = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(indptr))
    contrib = np.zeros_like(data)
    nz = data > 0
    contrib[nz] = -data[nz] * np.log10(data[nz])
    h = np.bincount(row_idx, weights=contrib, minlength=n_rows).astype(float)
    active = np.asarray(H.sum(axis=1)).ravel() > 0
    ji = np.full(n_rows, np.nan)
    ji[active] = h[active] / np.log10(n_cols)
    return ji[active]


def _col_pj(H) -> np.ndarray:
    """p_j = col_sum_j / n_rows — marginal distribution over columns."""
    n_rows = H.shape[0]
    cs = np.asarray(H.sum(axis=0)).ravel()
    return cs / n_rows


def build_kernels(db, run_code, fx, tau_u, tau_s):
    csr_ss = build_csr(db, run_code, fx, tau_u, tau_s, 0, (1, 0, 0, 0))
    C_SS, _ = _row_normalise(csr_ss.C_SS)

    csr_ii = build_csr(db, run_code, fx, tau_u, tau_s, 0, (0, 0, 0, 1))
    C_II, _ = _row_normalise(csr_ii.C_II)

    csr_bi = build_csr(db, run_code, fx, tau_u, tau_s, 0, (0, 1, 1, 0))
    C_SI, _ = _row_normalise(csr_bi.C_SI)
    C_IS, _ = _row_normalise(csr_bi.C_IS)
    M_S = C_SI.dot(C_IS).tocsr()
    M_I = C_IS.dot(C_SI).tocsr()

    return [
        ('$C_{SS}$', C_SS),
        ('$C_{II}$', C_II),
        ('$C_{SI}$', C_SI),
        ('$C_{IS}$', C_IS),
        ('$M_S$',    M_S),
        ('$M_I$',    M_I),
    ]


def plot(kernels: list, out_path: Path) -> None:
    n = len(kernels)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 5.5))

    for col, (name, H) in enumerate(kernels):
        ji    = _row_ji(H)
        pj    = _col_pj(H)
        n_col = H.shape[1]

        # top: J_i histogram
        ax = axes[0, col]
        ax.hist(ji, bins=40, range=(0, 1), color='steelblue', edgecolor='none')
        ax.set_yscale('log')
        jbar = float(np.mean(ji))
        ax.axvline(jbar, color='crimson', lw=1.2, ls='--')
        ax.set_title(name, fontsize=10)
        ax.set_xlim(0, 1)
        ax.text(0.04, 0.93, f'$\\bar{{J}}={jbar:.3f}$\nn={len(ji):,}',
                transform=ax.transAxes, va='top', fontsize=8)
        if col == 0:
            ax.set_ylabel('count (rows, log)', fontsize=9)

        # bottom: p_j histogram (log-log)
        ax = axes[1, col]
        pj_nz = pj[pj > 0]
        log_lo = np.floor(np.log10(pj_nz.min()))
        log_hi = np.ceil(np.log10(pj_nz.max()))
        bins = np.logspace(log_lo, log_hi, 50)
        ax.hist(pj_nz, bins=bins, color='darkorange', edgecolor='none')
        ax.set_xscale('log')
        ax.set_yscale('log')
        uniform_pj = 1.0 / n_col
        ax.axvline(uniform_pj, color='crimson', lw=1.2, ls='--')
        # J_col from entropy of pj
        nz = pj_nz > 0
        jcol = float(-np.dot(pj_nz[nz] / pj_nz[nz].sum(),
                             np.log10(pj_nz[nz] / pj_nz[nz].sum())) / np.log10(n_col))
        ax.text(0.96, 0.93, f'$J_{{\\rm col}}={jcol:.3f}$\nn={n_col:,}',
                transform=ax.transAxes, va='top', ha='right', fontsize=8)
        if col == 0:
            ax.set_ylabel('count (cols, log)', fontsize=9)

    axes[0, 0].set_xlabel('$J_i$')
    axes[1, 0].set_xlabel('$p_j$  (log scale)')
    fig.suptitle('Row evenness $J_i$ and column marginal $p_j$ — baseline kernels',
                 fontsize=11)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.10,
                        hspace=0.45, wspace=0.35)
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved {out_path}')


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
        kernels = build_kernels(db, run_code, fx, tau_u, tau_s)

    plot(kernels, paths.plots / 'fig_jbar_hist.pdf')


if __name__ == '__main__':
    main()
