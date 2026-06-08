"""
fig_3eb.py — Combined vs separate field single-unit rankings.

Left panel  (sources):
  x-axis: log(v_S^{EB,SS})  — E and B ranked together, m=1000, F=EB
  y-axis: log(v_S^{f,SS})   — E sources ranked in E-only, B in B-only, m=1000

Right panel (institutions):
  x-axis: log(v_I^{EB,II})  — E and B ranked together, m=0001, F=EB
  y-axis: log(v_I^{f,II})   — E institutions ranked in E-only, B in B-only, m=0001

Points coloured by field_eb (E=red, B=blue).  Diagonal y=x marks no change.

Outputs:
  plots/fig_3eb.pdf / fig_3eb_latex.pdf
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

T_EB_SS = _tname('EB', '1000')   # combined EB SS  — source x-axis
T_E_SS  = _tname('E',  '1000')   # E-only SS       — source y-axis
T_B_SS  = _tname('B',  '1000')   # B-only SS       — source y-axis

T_EB_II = _tname('EB', '0001')   # combined EB II  — institution x-axis
T_E_II  = _tname('E',  '0001')   # E-only II       — institution y-axis
T_B_II  = _tname('B',  '0001')   # B-only II       — institution y-axis

# ── Visual spec ─────────────────────────────────────────────────────────────────
E_COLOR = '#e41a1c'
B_COLOR = '#377eb8'
_V_LO, _V_HI = 0.005, 20
_V_TICKS     = [0.01, 0.1, 1, 10]
_PT_S, _PT_A = 12, 0.5


def _pub_theme():
    sns.set_theme(style='whitegrid', font_scale=1.05)
    plt.rcParams.update({
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.major.size': 5,    'ytick.major.size': 5,
    })


def _lv(v):
    return np.log10(np.clip(v, 1e-10, None))


def _log_axes(ax, xlabel):
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
    ax.plot([lo, hi], [lo, hi], color='black', lw=0.9, ls='--', zorder=1)


def _draw_panel(ax, df_e, df_b, xlabel, title):
    _log_axes(ax, xlabel)
    for df, color, label in [
        (df_b, B_COLOR, f'$F_B$  (n={len(df_b):,})'),
        (df_e, E_COLOR, f'$F_E$  (n={len(df_e):,})'),
    ]:
        ax.scatter(_lv(df['v_comb'].values), _lv(df['v_sep'].values),
                   c=color, s=_PT_S, alpha=_PT_A, linewidths=0.8, zorder=3,
                   label=label)
    ax.legend(fontsize=9, framealpha=0.85, loc='upper left', markerscale=1.6)
    ax.set_title(title, fontsize=10, pad=6)


def _merge(v_comb, v_sep, field_map, keep_field):
    v_comb = v_comb.rename(columns={'v': 'v_comb'})
    v_sep  = v_sep.rename(columns={'v': 'v_sep'})
    df = v_comb.merge(v_sep, on='unit_idx', how='inner')
    df['field'] = df['unit_idx'].map(field_map)
    return df[df['field'] == keep_field].copy()


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
    paths   = load_config()
    rk_path = paths.working / 'rankings.duckdb'
    par     = paths.working / 'parquet'

    # Field classifications
    src_eb  = pd.read_parquet(str(par / 'source_master.parquet'),
                              columns=['source_idx', 'field_eb'])
    src_field = dict(zip(src_eb['source_idx'].astype(int), src_eb['field_eb']))

    inst_eb  = pd.read_parquet(str(par / 'institution_field_eb.parquet'),
                               columns=['unit_idx', 'field_eb'])
    inst_field = dict(zip(inst_eb['unit_idx'].astype(int), inst_eb['field_eb']))

    with duckdb.connect(str(rk_path), read_only=True) as db:
        tables = {r[0] for r in db.execute('SHOW TABLES').fetchall()}
        for t in [T_EB_SS, T_E_SS, T_B_SS, T_EB_II, T_E_II, T_B_II]:
            if t not in tables:
                raise RuntimeError(f'Table {t} not found')

        def _load(t, utype):
            col = 'S' if utype == 'S' else 'U'
            return db.execute(
                f"SELECT unit_idx, v FROM {t} WHERE unit_type='{col}'"
            ).df()

        # Sources
        v_eb_s = _load(T_EB_SS, 'S')
        v_e_s  = _load(T_E_SS,  'S')
        v_b_s  = _load(T_B_SS,  'S')

        # Institutions
        v_eb_i = _load(T_EB_II, 'U')
        v_e_i  = _load(T_E_II,  'U')
        v_b_i  = _load(T_B_II,  'U')

    # Sources
    df_e_s = _merge(v_eb_s, v_e_s, src_field, 'E')
    df_b_s = _merge(v_eb_s, v_b_s, src_field, 'B')

    # Institutions
    df_e_i = _merge(v_eb_i, v_e_i, inst_field, 'E')
    df_b_i = _merge(v_eb_i, v_b_i, inst_field, 'B')

    _pub_theme()
    fig, (ax_s, ax_i) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig.subplots_adjust(wspace=0.05)

    _draw_panel(ax_s, df_e_s, df_b_s,
                xlabel=r'$\log(v_S^{EB,SS})$',
                title='Sources')
    ax_s.set_ylabel(r'$\log(v_S^{SS})$ or $\log(v_I^{II})$', labelpad=4)

    _draw_panel(ax_i, df_e_i, df_b_i,
                xlabel=r'$\log(v_I^{EB,II})$',
                title='Institutions')
    ax_i.set_ylabel('')
    ax_i.tick_params(labelleft=False)

    sup = fig.suptitle(
        r'Combined vs separate field single-unit rankings: effect of separating fields',
        fontsize=10, y=1.02,
    )
    _save(fig, sup, 'fig_3eb', paths)

    for label, df_e, df_b in [('Sources', df_e_s, df_b_s),
                               ('Institutions', df_e_i, df_b_i)]:
        print(f'\n{label}:')
        print(f'  E n={len(df_e)}  median log(v_sep/v_comb) = '
              f'{(_lv(df_e.v_sep) - _lv(df_e.v_comb)).median():+.3f}')
        print(f'  B n={len(df_b)}  median log(v_sep/v_comb) = '
              f'{(_lv(df_b.v_sep) - _lv(df_b.v_comb)).median():+.3f}')


if __name__ == '__main__':
    main()
    print('FINISHED!')
