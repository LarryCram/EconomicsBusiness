"""
cross_mask_gain.py — Cross-mask influence comparison for top-N units.

For the top-N sources by v^B: scatter v^B (x) vs v^S (y).
For the top-N institutions by v^B: scatter v^B (x) vs v^I (y).
Diagonal y=x marks no change from cross-layer routing.

Outputs:
  plots/cross_mask_gain.pdf
  plots/cross_mask_gain_latex.pdf
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

_baseline  = next(r for r in load_runs() if r['label'] == 'baseline')
_run_code  = _baseline['run_code']
_tau_u     = _baseline['tau_u']
_tau_s     = _baseline['tau_s']
_fx        = _baseline['fx']

BASELINE_TABLE = f'rk_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m0110_chi50_alpha100'
SS_TABLE       = f'rk_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m1000_chi50_alpha100'
II_TABLE       = f'rk_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m0001_chi50_alpha100'

TOP_N = 20


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_v(db, table):
    return db.execute(
        f"SELECT unit_idx, unit_type, v FROM {table}"
    ).df()


def fetch_data(db):
    tables = {r[0] for r in db.execute('SHOW TABLES').fetchall()}
    for t in (BASELINE_TABLE, SS_TABLE, II_TABLE):
        if t not in tables:
            raise RuntimeError(f'Table not found: {t}')

    base = load_v(db, BASELINE_TABLE)
    ss   = load_v(db, SS_TABLE)
    ii   = load_v(db, II_TABLE)

    # Sources: top-N by v^B, joined to v^S
    src_b = base[base['unit_type'] == 'S'][['unit_idx', 'v']].rename(columns={'v': 'v_B'})
    src_s = ss[ss['unit_type'] == 'S'][['unit_idx', 'v']].rename(columns={'v': 'v_S'})
    sources = (src_b.merge(src_s, on='unit_idx')
                    .sort_values('v_B', ascending=False)
                    .head(TOP_N)
                    .reset_index(drop=True))

    # Institutions: top-N by v^B, joined to v^I
    ins_b = base[base['unit_type'] == 'U'][['unit_idx', 'v']].rename(columns={'v': 'v_B'})
    ins_i = ii[ii['unit_type'] == 'U'][['unit_idx', 'v']].rename(columns={'v': 'v_I'})
    insts = (ins_b.merge(ins_i, on='unit_idx')
                  .sort_values('v_B', ascending=False)
                  .head(TOP_N)
                  .reset_index(drop=True))

    return sources, insts


# ─── Plot ─────────────────────────────────────────────────────────────────────

def _draw_panel(ax, df, x_col, y_col, xlabel, ylabel):
    x = df[x_col].values
    y = df[y_col].values

    ax.scatter(x, y, s=30, color='#333333', zorder=3)

    # Diagonal y = x over the joint range
    lo = min(x.min(), y.min()) * 0.9
    hi = max(x.max(), y.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], color='#aaaaaa', linewidth=0.8, zorder=1)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_aspect('equal')


def plot(sources, insts):
    paths = load_config()
    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.35)

    _draw_panel(axes[0], sources, 'v_B', 'v_S',
                '$v^B$ (bipartite)', '$v^S$ (source-only)')
    _draw_panel(axes[1], insts,   'v_B', 'v_I',
                '$v^B$ (bipartite)', '$v^I$ (institution-only)')

    axes[0].set_title(f'Sources  (top {TOP_N})', fontsize=10, pad=6)
    axes[1].set_title(f'Institutions  (top {TOP_N})', fontsize=10, pad=6)

    sup = fig.suptitle(
        'Cross-mask influence: bipartite vs within-layer',
        fontsize=10, y=1.02,
    )

    out = paths.plots / 'cross_mask_gain.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

    sup.set_visible(False)
    for ax in axes:
        ax.set_title('')
    latex_out = paths.plots / 'cross_mask_gain_latex.pdf'
    fig.savefig(latex_out, bbox_inches='tight')
    print(f'Saved {latex_out}')
    sup.set_visible(True)
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    paths = load_config()
    rk_path = paths.working / 'rankings.duckdb'
    with duckdb.connect(str(rk_path), read_only=True) as db:
        sources, insts = fetch_data(db)

    print(f'Sources  top-{TOP_N}  v^B range: {sources.v_B.min():.3f} – {sources.v_B.max():.3f}')
    print(f'Insts    top-{TOP_N}  v^B range: {insts.v_B.min():.3f}  – {insts.v_B.max():.3f}')
    print(sources[['unit_idx', 'v_B', 'v_S']].to_string(index=False))
    print(insts[['unit_idx',   'v_B', 'v_I']].to_string(index=False))

    plot(sources, insts)
    print('FINISHED!')


if __name__ == '__main__':
    main()
