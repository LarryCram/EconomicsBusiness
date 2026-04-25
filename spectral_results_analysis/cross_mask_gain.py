"""
cross_mask_gain.py — Cross-mask influence comparison, all units.

Row 0: scatter v^B (x) vs v^S or v^I (y) for all units, uncoloured.
Row 1: same scatter, coloured by field label (E/B/A/X).

Diagonal y=x in all panels. Units above diagonal: within-layer ranking higher;
units below: bipartite ranking higher.

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

COLOR  = {'E': '#e41a1c', 'B': '#377eb8', 'A': '#ff7f00', 'X': '#999999'}
MARKER = {'E': 'o', 'B': 's', 'A': 'D', 'X': 'x'}
FIELD_ORDER = ['X', 'A', 'B', 'E']   # plotted bottom-to-top


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_v(db, table):
    return db.execute(f"SELECT unit_idx, unit_type, v FROM {table}").df()


def fetch_data(db):
    tables = {r[0] for r in db.execute('SHOW TABLES').fetchall()}
    for t in (BASELINE_TABLE, SS_TABLE, II_TABLE):
        if t not in tables:
            raise RuntimeError(f'Table not found: {t}')

    base = load_v(db, BASELINE_TABLE)
    ss   = load_v(db, SS_TABLE)
    ii   = load_v(db, II_TABLE)

    src_b = base[base['unit_type'] == 'S'][['unit_idx', 'v']].rename(columns={'v': 'v_B'})
    src_s = ss[ss['unit_type'] == 'S'][['unit_idx', 'v']].rename(columns={'v': 'v_S'})
    sources = (src_b.merge(src_s, on='unit_idx')
                    .sort_values('v_B', ascending=False)
                    .reset_index(drop=True))

    ins_b = base[base['unit_type'] == 'U'][['unit_idx', 'v']].rename(columns={'v': 'v_B'})
    ins_i = ii[ii['unit_type'] == 'U'][['unit_idx', 'v']].rename(columns={'v': 'v_I'})
    insts = (ins_b.merge(ins_i, on='unit_idx')
                  .sort_values('v_B', ascending=False)
                  .reset_index(drop=True))

    return sources, insts


def attach_field_labels(sources, insts, paths):
    sm = pd.read_csv(paths.data / 'source_master.csv',
                     usecols=['source_idx', 'field_eb'])
    src_labels = dict(zip(sm['source_idx'].astype(int), sm['field_eb']))
    sources['field'] = sources['unit_idx'].map(src_labels).fillna('X')

    inst_df = pd.read_parquet(str(paths.parquet / 'institution_field_eb.parquet'),
                              columns=['unit_idx', 'field_eb'])
    ins_labels = dict(zip(inst_df['unit_idx'].astype(int), inst_df['field_eb']))
    insts['field'] = insts['unit_idx'].map(ins_labels).fillna('X')

    return sources, insts


# ─── Plot ─────────────────────────────────────────────────────────────────────

def _draw_diagonal(ax, x, y):
    lo = min(x.min(), y.min()) * 0.9
    hi = max(x.max(), y.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], color='#aaaaaa', linewidth=0.8, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def _plain_panel(ax, df, x_col, y_col, xlabel, ylabel):
    x, y = df[x_col].values, df[y_col].values
    ax.scatter(x, y, s=8, color='#333333', alpha=0.4, zorder=3)
    _draw_diagonal(ax, x, y)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_aspect('equal')


def _field_panel(ax, df, x_col, y_col, xlabel, ylabel):
    x_all, y_all = df[x_col].values, df[y_col].values
    _draw_diagonal(ax, x_all, y_all)
    for fld in FIELD_ORDER:
        sub = df[df['field'] == fld]
        if sub.empty:
            continue
        ax.scatter(sub[x_col].values, sub[y_col].values,
                   s=8, color=COLOR[fld], marker=MARKER[fld],
                   alpha=0.6, zorder=3,
                   label=f'{fld}  (n={len(sub):,})')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_aspect('equal')
    ax.legend(fontsize=7.5, framealpha=0.85, loc='upper left', markerscale=1.8)


def plot(sources, insts):
    paths = load_config()
    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.28, wspace=0.35)

    _plain_panel(axes[0, 0], sources, 'v_B', 'v_S',
                 '$v^B$', '$v^S$  (source-only)')
    _plain_panel(axes[0, 1], insts,   'v_B', 'v_I',
                 '$v^B$', '$v^I$  (institution-only)')

    _field_panel(axes[1, 0], sources, 'v_B', 'v_S',
                 '$v^B$', '$v^S$  (source-only)')
    _field_panel(axes[1, 1], insts,   'v_B', 'v_I',
                 '$v^B$', '$v^I$  (institution-only)')

    for ax, title in zip(axes[0], ['Sources', 'Institutions']):
        ax.set_title(title, fontsize=10, pad=6)

    sup = fig.suptitle(
        'Cross-mask influence: bipartite vs within-layer',
        fontsize=10, y=1.01,
    )

    out = paths.plots / 'cross_mask_gain.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

    sup.set_visible(False)
    for ax in axes[0]:
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

    sources, insts = attach_field_labels(sources, insts, paths)

    print(f'Sources  n={len(sources)}  v^B range: {sources.v_B.min():.3f} – {sources.v_B.max():.3f}')
    print(f'Insts    n={len(insts)}    v^B range: {insts.v_B.min():.3f}  – {insts.v_B.max():.3f}')

    plot(sources, insts)
    print('FINISHED!')


if __name__ == '__main__':
    main()
