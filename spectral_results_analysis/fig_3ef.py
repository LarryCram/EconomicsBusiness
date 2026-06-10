"""
fig_3ef.py — Two-panel comparison: counterfactual SS and bipartite BIP.

Left  panel: v_S^{E||B,SS} vs v_S^{E+B,SS} — top-10 SS sources omitted
             (counterfactual; reproduces RHS of counterfactual_zs10_zi0)
Right panel: v_S^{E||B,BIP} vs v_S^{E+B,BIP} — full corpus, bipartite
             (reproduces fig_3ed)

Tick labels show log10 values (-2, -1, 0, 1).

Outputs:
  plots/fig_3ef.pdf / fig_3ef_latex.pdf
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

sys.path.insert(0, str(Path(__file__).parent.parent / 'spectral_ranking'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from util import load_config, load_runs
from build_csr import build_csr
from katz_ranker import rank
from counterfactual_ranking import _slice_csr

# ── Parameters ──────────────────────────────────────────────────────────────────
_Z_S = 10                       # top sources to omit in left panel

E_COLOR  = '#e41a1c'
B_COLOR  = '#377eb8'
_V_LO, _V_HI = 0.005, 20
_V_TICKS     = [-2, -1, 0, 1]  # tick positions in log10 space
_PT_S, _PT_A = 12, 0.5


def _bl(runs):
    return next(r for r in runs if r['label'] == 'baseline')


def _tname(bl, fx, m):
    rc, u, s = bl['run_code'], bl['tau_u'], bl['tau_s']
    return f'rk_{rc}_{fx}_tauU{u}_tauS{s}_vartau_rho0_m{m}_chi50_alpha100'


def _lv(v):
    return np.log10(np.clip(v, 1e-10, None))


def _pub_theme():
    sns.set_theme(style='whitegrid', font_scale=1.05)
    plt.rcParams.update({
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.major.size': 5,    'ytick.major.size': 5,
    })


def _log_axes(ax, xlabel, ylabel=None):
    lo, hi = _lv(_V_LO), _lv(_V_HI)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xticks(_V_TICKS)
    ax.set_yticks(_V_TICKS)
    ax.set_xticklabels([str(t) for t in _V_TICKS])
    ax.set_yticklabels([str(t) for t in _V_TICKS])
    ax.set_xlabel(xlabel, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=4)
    else:
        ax.tick_params(labelleft=False)
    ax.plot([lo, hi], [lo, hi], color='black', lw=0.9, ls='--', zorder=1)


def _scatter(ax, df_e, df_b, panel_label):
    for df, color, lab in [
        (df_b, B_COLOR, f'$F_B$  (n={len(df_b):,})'),
        (df_e, E_COLOR, f'$F_E$  (n={len(df_e):,})'),
    ]:
        ax.scatter(_lv(df['v_comb'].values), _lv(df['v_sep'].values),
                   c=color, s=_PT_S, alpha=_PT_A, linewidths=0.8, zorder=3,
                   label=lab)
    ax.legend(fontsize=9, framealpha=0.85, loc='upper left', markerscale=1.6)
    ax.text(0.03, 0.83, panel_label, transform=ax.transAxes,
            ha='left', va='top', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=2))


def _run_cf(el_db, bl, fx, excl_src):
    """Run one SS mode with excl_src removed; return {unit_idx: v}."""
    rc, tau_u, tau_s = bl['run_code'], bl['tau_u'], bl['tau_s']
    data = build_csr(el_db, rc, fx, tau_u, tau_s, rho=0, m=(1,0,0,0), ref_units='')
    data = _slice_csr(data, excl_src, set())
    result = rank(data, (1,0,0,0), chi=0.5, alpha=1.0, mu=None)
    return dict(zip(data.source_ids.astype(int), result.v_s))


def _merge(v_x, v_y, field_map, keep_field):
    rows = [{'unit_idx': idx, 'v_comb': vx, 'v_sep': v_y[idx]}
            for idx, vx in v_x.items()
            if idx in v_y and field_map.get(idx) == keep_field]
    return pd.DataFrame(rows)


def main():
    paths = load_config()
    runs  = load_runs()
    bl    = _bl(runs)
    par   = paths.working / 'parquet'

    src_eb  = pd.read_parquet(str(par / 'source_master.parquet'),
                              columns=['source_idx', 'field_eb'])
    field_s = dict(zip(src_eb['source_idx'].astype(int), src_eb['field_eb']))

    # ── Left panel: counterfactual SS (top-_Z_S sources omitted) ────────────────
    rk_db = duckdb.connect(str(paths.working / 'rankings.duckdb'), read_only=True)
    bip_t = _tname(bl, 'EB', '0110')
    excl_src = set(rk_db.execute(
        f"SELECT unit_idx FROM {bip_t} WHERE unit_type='S' "
        f"ORDER BY v DESC LIMIT {_Z_S}"
    ).df()['unit_idx'].astype(int))

    # Right panel: bipartite from rankings.duckdb
    def _load(t):
        return dict(zip(*rk_db.execute(
            f"SELECT unit_idx, v FROM {t} WHERE unit_type='S'"
        ).df().values.T))

    v_eb_bip = _load(_tname(bl, 'EB', '0110'))
    v_e_bip  = _load(_tname(bl, 'E',  '0110'))
    v_b_bip  = _load(_tname(bl, 'B',  '0110'))
    rk_db.close()

    el_db = duckdb.connect(str(paths.working / 'edge_lists.duckdb'), read_only=True)
    print(f'Running counterfactual SS (top-{_Z_S} omitted) ...', flush=True)
    v_eb_cf = _run_cf(el_db, bl, 'EB', excl_src)
    v_e_cf  = _run_cf(el_db, bl, 'E',  excl_src)
    v_b_cf  = _run_cf(el_db, bl, 'B',  excl_src)
    el_db.close()
    print('  Done.')

    df_e_cf  = _merge(v_eb_cf,  v_e_cf,  field_s, 'E')
    df_b_cf  = _merge(v_eb_cf,  v_b_cf,  field_s, 'B')
    df_e_bip = _merge(v_eb_bip, v_e_bip, field_s, 'E')
    df_b_bip = _merge(v_eb_bip, v_b_bip, field_s, 'B')

    # ── Plot ─────────────────────────────────────────────────────────────────────
    _pub_theme()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.12)

    _log_axes(ax_l,
              xlabel=r'$\log(v_S^{E+B,SS})$',
              ylabel=(r'$\log(v_S^{E{\|}B,SS})$'
                      r' or $\log(v_S^{E{\|}B,\mathrm{Bipartite}})$'))
    _scatter(ax_l, df_e_cf, df_b_cf,
             panel_label=f'Counterfactual: top-{_Z_S} sources omitted')

    _log_axes(ax_r,
              xlabel=r'$\log(v_S^{E+B,\mathrm{Bipartite}})$')
    _scatter(ax_r, df_e_bip, df_b_bip,
             panel_label='EB corpus: bipartite')

    sup = fig.suptitle(
        r'Sources: counterfactual SS vs bipartite BIP — combined vs separate fields',
        fontsize=10, y=1.02,
    )

    out = paths.plots / 'fig_3ef.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')
    sup.set_visible(False)
    fig.savefig(paths.plots / 'fig_3ef_latex.pdf', bbox_inches='tight')
    print(f'Saved {paths.plots / "fig_3ef_latex.pdf"}')
    sup.set_visible(True)
    plt.close(fig)

    for lab, df_e, df_b in [('Counterfactual SS', df_e_cf, df_b_cf),
                             ('Bipartite BIP',    df_e_bip, df_b_bip)]:
        print(f'\n{lab}:')
        print(f'  E n={len(df_e)}  median log(v_sep/v_comb)='
              f'{(_lv(df_e.v_sep) - _lv(df_e.v_comb)).median():+.3f}')
        print(f'  B n={len(df_b)}  median log(v_sep/v_comb)='
              f'{(_lv(df_b.v_sep) - _lv(df_b.v_comb)).median():+.3f}')


if __name__ == '__main__':
    main()
    print('FINISHED!')
