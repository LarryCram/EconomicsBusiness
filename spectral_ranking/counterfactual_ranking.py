"""
counterfactual_ranking.py — Spectral ranking with top-z units omitted.

Identifies the top z_S sources and z_I institutions by the EB bipartite
baseline ranking, removes them from the corpus, and re-runs the six modes
needed for the fig_3eb-style plot (EB-SS, E-SS, B-SS, EB-II, E-II, B-II).

Produces:
  plots/counterfactual_zs{z_s}_zi{z_i}.pdf
  plots/counterfactual_zs{z_s}_zi{z_i}_latex.pdf

Usage:
  python spectral_ranking/counterfactual_ranking.py [--z-s N] [--z-i N]
"""

import argparse
import sys
from pathlib import Path
from dataclasses import replace

import matplotlib
matplotlib.use('Agg')

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))          # spectral_ranking/
sys.path.insert(0, str(Path(__file__).parent.parent))   # project root

from util import load_config, load_runs
from build_csr import build_csr, CSRData
from katz_ranker import rank


# ── Defaults ───────────────────────────────────────────────────────────────────
_DEFAULT_Z_S = 10
_DEFAULT_Z_I = 0

# ── Visual spec (matches fig_3eb) ──────────────────────────────────────────────
E_COLOR  = '#e41a1c'
B_COLOR  = '#377eb8'
CF_COLOR = '#4daf4a'  # green for counterfactual E
_V_LO, _V_HI = 0.005, 70
_V_TICKS     = [0.01, 0.1, 1, 10]
_PT_S, _PT_A = 12, 0.5


# ── Run parameter helpers ───────────────────────────────────────────────────────

def _baseline(runs):
    return next(r for r in runs if r['label'] == 'baseline')


def _tname(bl, fx, m_str):
    rc, u, s = bl['run_code'], bl['tau_u'], bl['tau_s']
    return f'rk_{rc}_{fx}_tauU{u}_tauS{s}_vartau_rho0_m{m_str}_chi50_alpha100'


def _tname_bip(bl):
    return _tname(bl, 'EB', '0110')


# ── CSR slicing ────────────────────────────────────────────────────────────────

def _slice_csr(data: CSRData, excl_src: set, excl_inst: set) -> CSRData:
    """Return a new CSRData with excluded source and institution indices removed."""
    keep_s = ~np.isin(data.source_ids, list(excl_src)) if excl_src else np.ones(data.n_s, bool)
    keep_u = ~np.isin(data.inst_ids,   list(excl_inst)) if excl_inst else np.ones(data.n_u, bool)

    def _sl(M, kr, kc):
        if M is None:
            return None
        return M[kr, :][:, kc]

    return CSRData(
        C_SS       = _sl(data.C_SS, keep_s, keep_s),
        C_SI       = _sl(data.C_SI, keep_s, keep_u),
        C_IS       = _sl(data.C_IS, keep_u, keep_s),
        C_II       = _sl(data.C_II, keep_u, keep_u),
        source_ids = data.source_ids[keep_s],
        inst_ids   = data.inst_ids[keep_u],
        a_s        = data.a_s[keep_s],
        a_u        = data.a_u[keep_u],
        n_s        = int(keep_s.sum()),
        n_u        = int(keep_u.sum()),
    )


# ── One ranking pass ────────────────────────────────────────────────────────────

def _run_mode(el_db, bl, fx: str, m: tuple,
              excl_src: set, excl_inst: set) -> dict:
    """
    Build CSR for (fx, m), exclude specified units, rank, and return
    {unit_idx -> v} for the appropriate unit type.
    """
    rc, tau_u, tau_s = bl['run_code'], bl['tau_u'], bl['tau_s']
    data = build_csr(el_db, rc, fx, tau_u, tau_s,
                     rho=0, m=m, ref_units='')
    data = _slice_csr(data, excl_src, excl_inst)

    chi  = 0.5
    result = rank(data, m, chi, alpha=1.0, mu=None)

    unit_type = m[0]  # 1 for SS, 0 for II (first element)
    if m == (1, 0, 0, 0):          # SS
        ids = data.source_ids
        v   = result.v_s
    else:                           # II  m=(0,0,0,1)
        ids = data.inst_ids
        v   = result.v_u

    return dict(zip(ids.astype(int), v))


# ── Plot (fig_3eb layout) ───────────────────────────────────────────────────────

def _lv(v):
    return np.log10(np.clip(v, 1e-10, None))


def _pub_theme():
    sns.set_theme(style='whitegrid', font_scale=1.05)
    plt.rcParams.update({
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.major.size': 5,    'ytick.major.size': 5,
    })


def _log_axes(ax, xlabel):
    lo, hi = _lv(_V_LO), _lv(_V_HI)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ticks = [_lv(t) for t in _V_TICKS]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in _V_TICKS])
    ax.set_yticklabels([str(t) for t in _V_TICKS])
    ax.set_xlabel(xlabel, labelpad=4)
    ax.plot([lo, hi], [lo, hi], color='black', lw=0.9, ls='--', zorder=1)


def _merge(v_x_map, v_y_map, field_map, keep_field):
    """Join x-axis map with y-axis map; inner join so only shared unit_idxs appear."""
    rows = []
    for idx, v_x in v_x_map.items():
        if idx in v_y_map and field_map.get(idx) == keep_field:
            rows.append({'unit_idx': idx, 'v_comb': v_x, 'v_sep': v_y_map[idx]})
    return pd.DataFrame(rows)


_YLABEL_S = r'$\log(v_S^{E{\|}B,SS})$'
_XLABEL_S = r'$\log(v_S^{E+B,SS})$'


def _add_encircle(ax, df, idx_set, color):
    """Draw a circle enclosing all points in idx_set; may spill outside axes."""
    sub = df[df['unit_idx'].isin(idx_set)]
    if len(sub) == 0:
        return
    xs = _lv(sub['v_comb'].values)
    ys = _lv(sub['v_sep'].values)
    cx, cy = xs.mean(), ys.mean()
    r = np.sqrt((xs - cx)**2 + (ys - cy)**2).max() * 1.08
    circle = mpatches.Circle((cx, cy), r, color=color, fill=False,
                              linewidth=0.8, clip_on=False, zorder=5)
    ax.add_patch(circle)


def _draw_panel(ax, df_e, df_b, xlabel, show_ylabel, panel_label=None):
    """One panel: E red, B blue, diagonal."""
    _log_axes(ax, xlabel)
    for df, color, label in [
        (df_b, B_COLOR, f'$F_B$  (n={len(df_b):,})'),
        (df_e, E_COLOR, f'$F_E$  (n={len(df_e):,})'),
    ]:
        ax.scatter(_lv(df['v_comb'].values), _lv(df['v_sep'].values),
                   c=color, s=_PT_S, alpha=_PT_A, linewidths=0.8, zorder=3,
                   label=label)
    ax.legend(fontsize=9, framealpha=0.85, loc='upper left', markerscale=1.6)
    if show_ylabel:
        ax.set_ylabel(_YLABEL_S, labelpad=4)
    else:
        ax.tick_params(labelleft=False)
    if panel_label:
        ax.text(0.03, 0.83, panel_label, transform=ax.transAxes,
                ha='left', va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=2))


def _plot(v_eb_s_cf, v_e_s_cf, v_b_s_cf,
          v_eb_i_cf, v_e_i_cf, v_b_i_cf,
          v_eb_s_full, v_e_s_full, v_b_s_full,
          v_eb_i_full, v_e_i_full, v_b_i_full,
          src_field, inst_field, z_s, z_i, paths,
          top_e_idx=None, top_b_idx=None):
    # Sources only
    df_e_s_full = _merge(v_eb_s_full, v_e_s_full, src_field, 'E')
    df_b_s_full = _merge(v_eb_s_full, v_b_s_full, src_field, 'B')
    df_e_s_cf   = _merge(v_eb_s_cf,   v_e_s_cf,   src_field, 'E')
    df_b_s_cf   = _merge(v_eb_s_cf,   v_b_s_cf,   src_field, 'B')

    _pub_theme()
    fig, (ax_full, ax_cf) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig.subplots_adjust(wspace=0.05)

    _draw_panel(ax_full, df_e_s_full, df_b_s_full,
                xlabel=_XLABEL_S, show_ylabel=True,
                panel_label='Full corpus')
    if top_e_idx is not None:
        _add_encircle(ax_full, df_e_s_full, top_e_idx, E_COLOR)
    if top_b_idx is not None:
        _add_encircle(ax_full, df_b_s_full, top_b_idx, B_COLOR)

    _draw_panel(ax_cf, df_e_s_cf, df_b_s_cf,
                xlabel=_XLABEL_S, show_ylabel=False,
                panel_label=f'Counterfactual (top-{z_s} sources omitted)')

    sup = fig.suptitle(
        rf'Sources: full corpus vs counterfactual (top-{z_s} omitted)',
        fontsize=10, y=1.01,
    )

    stem = f'counterfactual_zs{z_s}_zi{z_i}'
    out  = paths.plots / f'{stem}.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')
    sup.set_visible(False)
    fig.savefig(paths.plots / f'{stem}_latex.pdf', bbox_inches='tight')
    print(f'Saved {paths.plots / f"{stem}_latex.pdf"}')
    sup.set_visible(True)
    plt.close(fig)

    for label, df_e, df_b in [('Sources (full)', df_e_s_full, df_b_s_full),
                               ('Sources (cf)',   df_e_s_cf,   df_b_s_cf)]:
        print(f'\n{label}:')
        print(f'  E n={len(df_e)}  median log(v_sep/v_comb) = '
              f'{(_lv(df_e.v_sep) - _lv(df_e.v_comb)).median():+.3f}')
        print(f'  B n={len(df_b)}  median log(v_sep/v_comb) = '
              f'{(_lv(df_b.v_sep) - _lv(df_b.v_comb)).median():+.3f}')


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--z-s', type=int, default=_DEFAULT_Z_S,
                    help='number of top sources to omit (default %(default)s)')
    ap.add_argument('--z-i', type=int, default=_DEFAULT_Z_I,
                    help='number of top institutions to omit (default %(default)s)')
    args = ap.parse_args()
    z_s, z_i = args.z_s, args.z_i

    paths = load_config()
    runs  = load_runs()
    bl    = _baseline(runs)
    par   = paths.working / 'parquet'

    # Field classifications
    src_eb  = pd.read_parquet(str(par / 'source_master.parquet'),
                              columns=['source_idx', 'field_eb'])
    src_field = dict(zip(src_eb['source_idx'].astype(int), src_eb['field_eb']))

    inst_eb  = pd.read_parquet(str(par / 'institution_field_eb.parquet'),
                               columns=['unit_idx', 'field_eb'])
    inst_field = dict(zip(inst_eb['unit_idx'].astype(int), inst_eb['field_eb']))

    # Top-z units from EB bipartite baseline
    rk_db = duckdb.connect(str(paths.working / 'rankings.duckdb'), read_only=True)
    bip_tname = _tname_bip(bl)
    top_src  = rk_db.execute(
        f"SELECT unit_idx FROM {bip_tname} WHERE unit_type='S' "
        f"ORDER BY v DESC LIMIT {z_s}"
    ).df()['unit_idx'].astype(int).tolist() if z_s > 0 else []
    top_inst = rk_db.execute(
        f"SELECT unit_idx FROM {bip_tname} WHERE unit_type='U' "
        f"ORDER BY v DESC LIMIT {z_i}"
    ).df()['unit_idx'].astype(int).tolist() if z_i > 0 else []
    rk_db.close()

    excl_src  = set(top_src)
    excl_inst = set(top_inst)

    print(f'Omitting {len(excl_src)} sources and {len(excl_inst)} institutions')
    if excl_src:
        # Print names for confirmation
        sm = pd.read_parquet(str(par / 'source_master.parquet'),
                             columns=['source_idx', 'source_name'])
        names = sm[sm['source_idx'].isin(excl_src)]['source_name'].tolist()
        print('  Sources omitted:', ', '.join(names))

    el_db = duckdb.connect(str(paths.working / 'edge_lists.duckdb'), read_only=True)

    print('\nRunning EB-SS ...', flush=True)
    v_eb_s = _run_mode(el_db, bl, 'EB', (1,0,0,0), excl_src, excl_inst)
    print(f'  n_S = {len(v_eb_s)}')

    print('Running E-SS  ...', flush=True)
    v_e_s  = _run_mode(el_db, bl, 'E',  (1,0,0,0), excl_src, excl_inst)

    print('Running B-SS  ...', flush=True)
    v_b_s  = _run_mode(el_db, bl, 'B',  (1,0,0,0), excl_src, excl_inst)

    print('Running EB-II ...', flush=True)
    v_eb_i = _run_mode(el_db, bl, 'EB', (0,0,0,1), excl_src, excl_inst)
    print(f'  n_I = {len(v_eb_i)}')

    print('Running E-II  ...', flush=True)
    v_e_i  = _run_mode(el_db, bl, 'E',  (0,0,0,1), excl_src, excl_inst)

    print('Running B-II  ...', flush=True)
    v_b_i  = _run_mode(el_db, bl, 'B',  (0,0,0,1), excl_src, excl_inst)

    el_db.close()

    # Full-corpus rankings (background) — loaded from existing duckdb tables
    print('\nLoading full-corpus rankings ...', flush=True)
    rk_db2 = duckdb.connect(str(paths.working / 'rankings.duckdb'), read_only=True)

    def _load_full(fx, m_str, utype):
        t = _tname(bl, fx, m_str)
        col = 'S' if utype == 'S' else 'U'
        df = rk_db2.execute(
            f"SELECT unit_idx, v FROM {t} WHERE unit_type='{col}'"
        ).df()
        return dict(zip(df['unit_idx'].astype(int), df['v']))

    v_eb_s_full = _load_full('EB', '1000', 'S')
    v_e_s_full  = _load_full('E',  '1000', 'S')
    v_b_s_full  = _load_full('B',  '1000', 'S')
    v_eb_i_full = _load_full('EB', '0001', 'U')
    v_e_i_full  = _load_full('E',  '0001', 'U')
    v_b_i_full  = _load_full('B',  '0001', 'U')

    # Top-10 E and B by field-specific SS ranking (for encircling on full corpus panel)
    top_e_idx = set(rk_db2.execute(
        f"SELECT unit_idx FROM {_tname(bl,'E','1000')} WHERE unit_type='S' ORDER BY v DESC LIMIT 10"
    ).df()['unit_idx'].astype(int))
    top_b_idx = set(rk_db2.execute(
        f"SELECT unit_idx FROM {_tname(bl,'B','1000')} WHERE unit_type='S' ORDER BY v DESC LIMIT 10"
    ).df()['unit_idx'].astype(int))
    rk_db2.close()

    _plot(v_eb_s,      v_e_s,      v_b_s,
          v_eb_i,      v_e_i,      v_b_i,
          v_eb_s_full, v_e_s_full, v_b_s_full,
          v_eb_i_full, v_e_i_full, v_b_i_full,
          src_field, inst_field, z_s, z_i, paths,
          top_e_idx=top_e_idx, top_b_idx=top_b_idx)


if __name__ == '__main__':
    main()
    print('FINISHED!')
