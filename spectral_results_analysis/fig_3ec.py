"""
fig_3ec.py — Combined bipartite vs separate field single-unit rankings (sources).

x-axis: log(v_S^{EB,BIP})  — E and B ranked together, m=0110, F=EB
y-axis: log(v_S^{E,SS})    — E sources ranked in E-only corpus, m=1000
        log(v_S^{B,SS})    — B sources ranked in B-only corpus, m=1000

Points coloured by field_eb.  Diagonal y=x marks no change.

Outputs:
  plots/fig_3ec.pdf / fig_3ec_latex.pdf
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

# ── Run parameters ─────────────────────────────────────────────────────────────
_bl    = next(r for r in load_runs() if r['label'] == 'baseline')
_rc    = _bl['run_code']
_tau_u = _bl['tau_u']
_tau_s = _bl['tau_s']

def _tname(fx, m):
    return (f'rk_{_rc}_{fx}_tauU{_tau_u}_tauS{_tau_s}_vartau'
            f'_rho0_m{m}_chi50_alpha100')

T_EB_BIP = _tname('EB', '0110')   # combined EB bipartite — x-axis
T_E      = _tname('E',  '1000')   # E-only SS             — y-axis for E sources
T_B      = _tname('B',  '1000')   # B-only SS             — y-axis for B sources

# ── Visual spec ─────────────────────────────────────────────────────────────────
E_COLOR = '#e41a1c'
B_COLOR = '#377eb8'
_V_LO, _V_HI   = 0.005, 20
_V_TICKS        = [0.01, 0.1, 1, 10]
_PT_S, _PT_A    = 12, 0.5


def _pub_theme():
    sns.set_theme(style='whitegrid', font_scale=1.05)
    plt.rcParams.update({
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.major.size': 5,    'ytick.major.size': 5,
    })


def _lv(v):
    return np.log10(np.clip(v, 1e-10, None))


def _log_axes(ax, xlabel, ylabel):
    lo, hi = _lv(_V_LO), _lv(_V_HI)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ticks = [_lv(t) for t in _V_TICKS]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in _V_TICKS])
    ax.set_yticklabels([str(t) for t in _V_TICKS])
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.plot([lo, hi], [lo, hi], color='black', lw=0.9, ls='--', zorder=1)


def _save(fig, sup, stem, paths):
    out = paths.plots / f'{stem}.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')
    sup.set_visible(False)
    fig.savefig(paths.plots / f'{stem}_latex.pdf', bbox_inches='tight')
    print(f'Saved {paths.plots / f"{stem}_latex.pdf"}')
    sup.set_visible(True)
    plt.close(fig)


def main():
    paths = load_config()
    rk_path = paths.working / 'rankings.duckdb'
    par     = paths.working / 'parquet'

    src_eb = pd.read_parquet(str(par / 'source_master.parquet'),
                             columns=['source_idx', 'field_eb'])
    field_map = dict(zip(src_eb['source_idx'].astype(int), src_eb['field_eb']))

    with duckdb.connect(str(rk_path), read_only=True) as db:
        tables = {r[0] for r in db.execute('SHOW TABLES').fetchall()}
        for t in [T_EB_BIP, T_E, T_B]:
            if t not in tables:
                raise RuntimeError(f'Table {t} not found')

        v_bip = db.execute(f"SELECT unit_idx, v FROM {T_EB_BIP} WHERE unit_type='S'").df()
        v_e   = db.execute(f"SELECT unit_idx, v FROM {T_E}      WHERE unit_type='S'").df()
        v_b   = db.execute(f"SELECT unit_idx, v FROM {T_B}      WHERE unit_type='S'").df()

    v_e   = v_e.rename(columns={'v': 'v_sep'})
    v_b   = v_b.rename(columns={'v': 'v_sep'})
    v_bip = v_bip.rename(columns={'v': 'v_comb'})

    df_e = v_bip.merge(v_e, on='unit_idx', how='inner')
    df_b = v_bip.merge(v_b, on='unit_idx', how='inner')

    df_e['field'] = df_e['unit_idx'].map(field_map)
    df_b['field'] = df_b['unit_idx'].map(field_map)

    df_e = df_e[df_e['field'] == 'E'].copy()
    df_b = df_b[df_b['field'] == 'B'].copy()

    _pub_theme()
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))

    _log_axes(ax,
              xlabel=r'$\log(v_S^{EB,\mathrm{BIP}})$  — combined E+B bipartite ranking',
              ylabel=r'$\log(v_S^{f,SS})$  — separate field SS ranking')

    for df, color, label in [
        (df_b, B_COLOR, f'$F_B$  (n={len(df_b):,})'),
        (df_e, E_COLOR, f'$F_E$  (n={len(df_e):,})'),
    ]:
        ax.scatter(_lv(df['v_comb'].values), _lv(df['v_sep'].values),
                   c=color, s=_PT_S, alpha=_PT_A, linewidths=0.8, zorder=3,
                   label=label)

    ax.legend(fontsize=10, framealpha=0.85, loc='upper left', markerscale=1.6)
    ax.set_title('Sources: combined bipartite vs separate field SS ranking', fontsize=10, pad=6)

    sup = fig.suptitle(
        r'$\log(v_S^{f,SS})$ vs $\log(v_S^{EB,\mathrm{BIP}})$: SS vs bipartite combined ranking',
        fontsize=9, y=1.01,
    )
    _save(fig, sup, 'fig_3ec', paths)

    print(f'\nE sources: n={len(df_e)}')
    print(f'  median log(v_sep/v_comb) = {(_lv(df_e.v_sep) - _lv(df_e.v_comb)).median():+.3f}')
    print(f'B sources: n={len(df_b)}')
    print(f'  median log(v_sep/v_comb) = {(_lv(df_b.v_sep) - _lv(df_b.v_comb)).median():+.3f}')


if __name__ == '__main__':
    main()
    print('FINISHED!')
