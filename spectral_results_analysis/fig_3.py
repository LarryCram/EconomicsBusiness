"""
fig_3.py  — Rank curves across network mode m  (left panels, paper figure).
fig_3a.py — vSS/vII vs v_bipartite scatter     (right panels, paper figure).

  m=0110  bipartite SI/IS   — black line  (baseline, both panels)
  m=1000  source-only SS    — red X       (S panel only)
  m=0001  institution-only  — red X       (I panel only)
  m=1111  full joint χ*     — green X     (both panels; χ* resolved from _catalog)

All runs use baseline F from params.csv, τ_U=τ_S=20, ρ=0, α=1.
field_eb=X units excluded.

Outputs:
  plots/fig_3.pdf / fig_3_latex.pdf         — rank curves (2×1)
  plots/fig_3a.pdf / fig_3a_latex.pdf       — scatter panels (2×1)
  spectral_ranking_latex/tables/table_unit_effects.tex — illustrative v≈1 cases
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.colors import LogNorm

_V_LO, _V_HI = 0.005, 20
_V_TICKS = [0.01, 0.1, 1, 10]

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

_baseline      = next(r for r in load_runs() if r['label'] == 'baseline')
_run_code      = _baseline['run_code']
_tau_u         = _baseline['tau_u']
_tau_s         = _baseline['tau_s']
_fx            = _baseline['fx']

BASELINE_TABLE = f'rk_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m0110_chi50_alpha100'
SS_TABLE       = f'rk_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m1000_chi50_alpha100'
II_TABLE       = f'rk_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m0001_chi50_alpha100'
EL_TABLE       = f'el_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau'

V1_BAND = 0.05  # |log(v)| < 0.05  →  v in (e^-0.05, e^+0.05) ≈ (0.951, 1.051)

# Visual spec per label
STYLE = {
    'm=0110': dict(color='black',   marker=None, zorder=4, lw=1.8, alpha=1.0),
    'm=1000': dict(color='#d62728', marker='x',  zorder=3, s=55,   lw=1.0, alpha=0.5),
    'm=0001': dict(color='#d62728', marker='x',  zorder=3, s=55,   lw=1.0, alpha=0.5),
    'm=1111': dict(color='#2ca02c', marker='x',  zorder=2, s=55,   lw=1.0, alpha=0.5),
}

S_LABELS = ['m=0110', 'm=1000', 'm=1111']
I_LABELS = ['m=0110', 'm=0001', 'm=1111']


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_run(db, table_name: str) -> pd.DataFrame:
    return db.execute(
        f"SELECT unit_idx, unit_type, v, a_p FROM {table_name}"
    ).df()


def wmean_v(df: pd.DataFrame) -> float:
    """a_p-weighted mean of v. Should equal 1 by construction."""
    if df.empty:
        return float('nan')
    return float(np.average(df['v'].values, weights=df['a_p'].values))


def resolve_chi_star_table(db) -> str | None:
    """
    Look up the full-joint χ* run from _catalog.
    Matches on m=(1,1,1,1), same corpus as baseline, chi != 0.5.
    Returns table_name, or None with a warning if not found.
    """
    tables = {row[0] for row in db.execute('SHOW TABLES').fetchall()}
    if '_catalog' not in tables:
        print('  WARNING: _catalog not found — cannot resolve χ* table')
        return None
    rows = db.execute(
        "SELECT table_name, chi, label FROM _catalog "
        "WHERE m_SS=1 AND m_SI=1 AND m_IS=1 AND m_II=1 "
        f"  AND run_code='{_run_code}' AND fx='{_fx}' AND tau_u={_tau_u} AND tau_s={_tau_s} AND rho=0 "
        "  AND round(alpha*100)=100 AND round(chi*100) != 50 "
        "ORDER BY created_at DESC LIMIT 1"
    ).fetchall()
    if not rows:
        print('  WARNING: no full-joint χ* entry in _catalog — skipping m=1111')
        return None
    tname, chi, label = rows[0]
    if tname not in tables:
        print(f'  WARNING: {tname} in _catalog but not in database — skipping m=1111')
        return None
    print(f'  m=1111 χ*={chi:.4f}  label={label}  table={tname}')
    return tname


def fetch_data(db) -> tuple:
    """
    Returns
    -------
    src_rank_map  : dict  unit_idx -> baseline_rank  (sources, m=0110)
    inst_rank_map : dict  unit_idx -> baseline_rank  (institutions, m=0110)
    series        : dict  label -> {'S': DataFrame, 'I': DataFrame}
                    Each DataFrame has columns [unit_idx, baseline_rank, v].
                    Baseline series is included (label 'm=0110').
    """
    tables = {row[0] for row in db.execute('SHOW TABLES').fetchall()}

    # ── Baseline: establish x-axis lock ──────────────────────────────────────
    if BASELINE_TABLE not in tables:
        raise RuntimeError(
            f'Baseline table {BASELINE_TABLE} not found in rankings.duckdb.'
        )
    df_base = load_run(db, BASELINE_TABLE)

    df_s_base = (df_base[df_base['unit_type'] == 'S']
                 .sort_values('v', ascending=False)
                 .reset_index(drop=True))
    df_s_base['baseline_rank'] = np.arange(1, len(df_s_base) + 1)
    src_rank_map = df_s_base.set_index('unit_idx')['baseline_rank'].to_dict()

    df_i_base = (df_base[df_base['unit_type'] == 'U']
                 .sort_values('v', ascending=False)
                 .reset_index(drop=True))
    df_i_base['baseline_rank'] = np.arange(1, len(df_i_base) + 1)
    inst_rank_map = df_i_base.set_index('unit_idx')['baseline_rank'].to_dict()

    def project(df, rank_map):
        out = df.copy()
        out['baseline_rank'] = out['unit_idx'].map(rank_map)
        return out.dropna(subset=['baseline_rank']).sort_values('baseline_rank')

    def report(label, df_s, df_i):
        both = pd.concat([df_s, df_i]) if not df_s.empty and not df_i.empty else pd.DataFrame()
        joint = f'  wmean_v_joint={wmean_v(both):.4f}' if not both.empty else ''
        print(f'  {label:<22}  '
              f'N_s={len(df_s):>5,}  N_u={len(df_i):>5,}  '
              f'wmean_v_S={wmean_v(df_s):.4f}  wmean_v_I={wmean_v(df_i):.4f}'
              f'{joint}')

    series = {
        'm=0110': {
            'S': project(df_s_base.rename(columns={'baseline_rank': 'baseline_rank'}),
                         src_rank_map),
            'I': project(df_i_base.rename(columns={'baseline_rank': 'baseline_rank'}),
                         inst_rank_map),
        }
    }
    report('m=0110 (baseline)', series['m=0110']['S'], series['m=0110']['I'])

    # ── SS run (sources only) ─────────────────────────────────────────────────
    if SS_TABLE in tables:
        df = load_run(db, SS_TABLE)
        df_s = project(df[df['unit_type'] == 'S'], src_rank_map)
        series['m=1000'] = {'S': df_s, 'I': pd.DataFrame()}
        report('m=1000 (SS)', df_s, pd.DataFrame())
    else:
        print(f'  WARNING: {SS_TABLE} not found — skipping m=1000')

    # ── II run (institutions only) ────────────────────────────────────────────
    if II_TABLE in tables:
        df = load_run(db, II_TABLE)
        df_i = project(df[df['unit_type'] == 'U'], inst_rank_map)
        series['m=0001'] = {'S': pd.DataFrame(), 'I': df_i}
        report('m=0001 (II)', pd.DataFrame(), df_i)
    else:
        print(f'  WARNING: {II_TABLE} not found — skipping m=0001')

    # ── Full-joint χ* run ─────────────────────────────────────────────────────
    chi_star_table = resolve_chi_star_table(db)
    if chi_star_table:
        df = load_run(db, chi_star_table)
        df_s = project(df[df['unit_type'] == 'S'], src_rank_map)
        df_i = project(df[df['unit_type'] == 'U'], inst_rank_map)
        series['m=1111'] = {'S': df_s, 'I': df_i}
        report('m=1111 (χ*)', df_s, df_i)

    return src_rank_map, inst_rank_map, series


# ─── Plot ─────────────────────────────────────────────────────────────────────

def _draw_panel(ax, series: dict, unit_key: str, panel_labels: list,
                n_baseline: int, panel_title: str) -> None:
    for label in panel_labels:
        if label not in series:
            continue
        df = series[label][unit_key]
        if df.empty:
            continue

        style = STYLE[label]
        is_baseline = style['marker'] is None
        n_overlap = len(df)

        if is_baseline:
            ax.plot(
                df['baseline_rank'].values,
                df['v'].values,
                color=style['color'],
                linewidth=style['lw'],
                alpha=style['alpha'],
                zorder=style['zorder'],
                label=label,
            )
        else:
            ax.scatter(
                df['baseline_rank'].values,
                df['v'].values,
                color=style['color'],
                marker=style['marker'],
                s=style['s'],
                linewidths=style['lw'],
                zorder=style['zorder'],
                alpha=style.get('alpha', 0.5),
                label=f'{label}  ({n_overlap:,}/{n_baseline:,})',
            )

    ax.set_yscale('log')
    ax.set_ylim(_V_LO, _V_HI)
    ax.set_yticks(_V_TICKS)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.axhline(1.0, color='#999999', linewidth=1.0, linestyle='--', zorder=0)
    ax.text(
        n_baseline * 0.98, 1.0,
        '$v=1$',
        ha='right', va='bottom',
        fontsize=9, color='#999999',
    )
    ax.set_xlim(1, n_baseline)
    ax.set_xlabel('Baseline rank  (m=0110)', labelpad=4)
    ax.set_ylabel('Influence per work $v$', labelpad=4)
    ax.set_title(panel_title, fontsize=11, pad=6)
    ax.legend(fontsize=9, framealpha=0.85, loc='upper right')


def filter_non_x(series: dict, src_rank_map: dict,
                 inst_rank_map: dict) -> tuple:
    """
    Drop field_eb='X' sources and all-X institutions from every series DataFrame
    and the rank maps.  Returns updated (src_rank_map, inst_rank_map, series)
    and prints counts dropped.
    """
    paths = load_config()
    par   = paths.working / 'parquet'

    src_eb = pd.read_parquet(str(par / 'source_master.parquet'),
                             columns=['source_idx', 'field_eb'])
    non_x_src  = set(src_eb.loc[src_eb['field_eb'] != 'X', 'source_idx'])

    inst_eb = pd.read_parquet(str(par / 'institution_field_eb.parquet'),
                              columns=['unit_idx', 'field_eb'])
    non_x_inst = set(inst_eb.loc[inst_eb['field_eb'] != 'X', 'unit_idx'])

    n_src_before  = len(src_rank_map)
    n_inst_before = len(inst_rank_map)

    src_rank_map  = {k: v for k, v in src_rank_map.items()  if k in non_x_src}
    inst_rank_map = {k: v for k, v in inst_rank_map.items() if k in non_x_inst}

    def filt(df, keep_set):
        return df[df['unit_idx'].isin(keep_set)].copy() if not df.empty else df

    new_series = {}
    for mode, panels in series.items():
        new_series[mode] = {
            'S': filt(panels['S'], non_x_src),
            'I': filt(panels['I'], non_x_inst),
        }

    n_src_drop  = n_src_before  - len(src_rank_map)
    n_inst_drop = n_inst_before - len(inst_rank_map)
    print(f'  Dropped field_eb=X: {n_src_drop} sources '
          f'({n_src_before}→{len(src_rank_map)}), '
          f'{n_inst_drop} institutions '
          f'({n_inst_before}→{len(inst_rank_map)})')

    return src_rank_map, inst_rank_map, new_series


def load_cross_type_colors(series: dict) -> tuple:
    """
    For each source i:      weighted-avg v_I^bipartite of cited institutions.
    For each institution k:  weighted-avg v_S^bipartite of citing sources.
    Weights = cited_inst_weight from the baseline edge list.
    By the bipartite fixed-point equation these equal the unit's own v_bipartite,
    but computing via the edge list makes the cross-type interpretation explicit.
    """
    paths = load_config()
    el_path = paths.working / 'edge_lists.duckdb'
    el_table = f'el_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau'

    v_s = series['m=0110']['S'][['unit_idx', 'v']].rename(columns={'v': 'v_s'})
    v_i = series['m=0110']['I'][['unit_idx', 'v']].rename(columns={'v': 'v_i'})

    with duckdb.connect(str(el_path), read_only=True) as el_db:
        el = el_db.execute(f"""
            SELECT citer_source_idx, cited_inst_idx, SUM(cited_inst_weight) AS w
            FROM {el_table}
            WHERE citer_source_idx IS NOT NULL AND cited_inst_idx IS NOT NULL
            GROUP BY citer_source_idx, cited_inst_idx
        """).df()

    # Source → weighted avg v_I of cited institutions
    el_s = el.merge(v_i.rename(columns={'unit_idx': 'cited_inst_idx'}),
                    on='cited_inst_idx', how='inner')
    el_s['wv'] = el_s['w'] * el_s['v_i']
    agg_s = el_s.groupby('citer_source_idx')[['wv', 'w']].sum()
    agg_s['color_v'] = agg_s['wv'] / agg_s['w']
    src_colors = agg_s['color_v'].to_dict()

    # Institution → weighted avg v_S of sources that cite it
    el_i = el.merge(v_s.rename(columns={'unit_idx': 'citer_source_idx'}),
                    on='citer_source_idx', how='inner')
    el_i['wv'] = el_i['w'] * el_i['v_s']
    agg_i = el_i.groupby('cited_inst_idx')[['wv', 'w']].sum()
    agg_i['color_v'] = agg_i['wv'] / agg_i['w']
    inst_colors = agg_i['color_v'].to_dict()

    return src_colors, inst_colors


def _draw_scatter_panel(ax, fig, series: dict, unit_key: str,
                        alt_mode: str, y_label: str, title: str,
                        cmap: str = 'RdBu_r', cbar_label: str = '$v$ bipartite',
                        color_map: dict | None = None) -> None:
    """
    Scatter v_bipartite (x) vs v_alt (y).
    colour = color_map[unit_idx] if provided, else v_bipartite (x).
    """
    df_bip = series['m=0110'][unit_key]
    if alt_mode not in series or series[alt_mode][unit_key].empty:
        ax.text(0.5, 0.5, f'{alt_mode} not available',
                transform=ax.transAxes, ha='center', va='center', fontsize=9)
        ax.set_title(title, fontsize=10, pad=6)
        return

    df_alt = series[alt_mode][unit_key]
    df = (df_bip[['unit_idx', 'v']].rename(columns={'v': 'v_bip'})
          .merge(df_alt[['unit_idx', 'v']].rename(columns={'v': 'v_alt'}),
                 on='unit_idx', how='inner'))
    df = df[(df['v_bip'] > 0) & (df['v_alt'] > 0)].copy()

    if color_map is not None:
        df['color_v'] = df['unit_idx'].map(color_map)
        df = df.dropna(subset=['color_v'])
        c = df['color_v'].values
    else:
        c = df['v_bip'].values

    x = df['v_bip'].values
    y = df['v_alt'].values

    norm = LogNorm(vmin=c[c > 0].min(), vmax=c.max())
    sc = ax.scatter(x, y, c=c, cmap=cmap, norm=norm,
                    s=12, alpha=0.55, linewidths=0, zorder=2, rasterized=True)

    ax.plot([_V_LO, _V_HI], [_V_LO, _V_HI], color='#888888', lw=1.0, ls='--', zorder=1)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(_V_LO, _V_HI)
    ax.set_ylim(_V_LO, _V_HI)
    ax.set_xticks(_V_TICKS)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_yticks(_V_TICKS)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel('$v$  bipartite  (m=0110)', labelpad=4)
    ax.set_ylabel(y_label, labelpad=4)
    ax.set_title(title, fontsize=11, pad=6)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.text(0.03, 0.97, f'n={len(df):,}', transform=ax.transAxes,
            fontsize=9, ha='left', va='top', color='#555555')


def _pub_theme():
    sns.set_theme(style='whitegrid', font_scale=1.05)
    plt.rcParams.update({
        'axes.linewidth':    1.2,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.major.size':  5,   'ytick.major.size':  5,
    })


def _save(fig, sup, stem: str, paths) -> None:
    out = paths.plots / f'{stem}.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')
    sup.set_visible(False)
    fig.savefig(paths.plots / f'{stem}_latex.pdf', bbox_inches='tight')
    print(f'Saved {paths.plots / f"{stem}_latex.pdf"}')
    sup.set_visible(True)
    plt.close(fig)


def plot3(src_rank_map: dict, inst_rank_map: dict, series: dict) -> None:
    """Left panels: rank curves."""
    paths = load_config()
    _pub_theme()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.subplots_adjust(wspace=0.35)

    _draw_panel(axes[0], series, 'S', S_LABELS,
                n_baseline=len(src_rank_map),  panel_title='Sources')
    _draw_panel(axes[1], series, 'I', I_LABELS,
                n_baseline=len(inst_rank_map), panel_title='Institutions')

    sup = fig.suptitle(
        'Influence per work — sensitivity to network mode $m$',
        fontsize=9, y=1.01,
    )
    _save(fig, sup, 'fig_3', paths)


def plot3a(src_rank_map: dict, inst_rank_map: dict, series: dict,
           src_highlights: pd.DataFrame | None = None,
           inst_highlights: pd.DataFrame | None = None) -> None:
    """Right panels: vSS/vII vs v_bipartite scatter with cross-type colour."""
    paths = load_config()
    _pub_theme()

    print('Computing cross-type partner colours ...')
    src_colors, inst_colors = load_cross_type_colors(series)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.45)

    _draw_scatter_panel(axes[0], fig, series, 'S', 'm=1000',
                        '$v$  SS only  (m=1000)',
                        'Sources — $v_{\\rm SS}$ vs $v_{\\rm bip}$',
                        cmap='viridis_r',
                        cbar_label='mean $v_I$ (bipartite)',
                        color_map=src_colors)
    if src_highlights is not None and not src_highlights.empty:
        axes[0].scatter(src_highlights['v_bip'].values, src_highlights['v_ss'].values,
                        marker='+', s=150, color='black', linewidths=1.5, zorder=6)
    _draw_scatter_panel(axes[1], fig, series, 'I', 'm=0001',
                        '$v$  II only  (m=0001)',
                        'Institutions — $v_{\\rm II}$ vs $v_{\\rm bip}$',
                        cmap='plasma_r',
                        cbar_label='mean $v_S$ (bipartite)',
                        color_map=inst_colors)
    if inst_highlights is not None and not inst_highlights.empty:
        axes[1].scatter(inst_highlights['v_bip'].values, inst_highlights['v_ii'].values,
                        marker='+', s=150, color='black', linewidths=1.5, zorder=6)

    sup = fig.suptitle(
        '$v_{\\rm SS}$ vs $v_{\\rm bip}$ and $v_{\\rm II}$ vs $v_{\\rm bip}$',
        fontsize=9, y=1.01,
    )
    _save(fig, sup, 'fig_3a', paths)


# ─── Unit-effects table (table_unit_effects.tex) ─────────────────────────────

def _explain_source(row) -> str:
    v_bip, v_ss, mv_i = row['v_bip'], row['v_ss'], row['mean_v_I_bip']
    return (
        f"v_SB ({v_bip:.3f}) is {'above' if v_bip > v_ss else 'below'} v_SS ({v_ss:.3f}). "
        f"In the bipartite model v_SB equals the H_SI-weighted mean v_IB of cited "
        f"institutions ({mv_i:.3f}, {'above' if mv_i > 1.0 else 'below'} the institutional mean of 1), "
        f"placing this source {'above' if v_bip > 1.0 else 'below'} the bipartite average. "
        f"v_SS reflects source-source citations, which are absent from the bipartite model."
    )


def _explain_institution(row) -> str:
    v_bip, v_ii, mv_s = row['v_bip'], row['v_ii'], row['mean_v_S_bip']
    return (
        f"v_IB ({v_bip:.3f}) is {'above' if v_bip > v_ii else 'below'} v_II ({v_ii:.3f}). "
        f"In the bipartite model v_IB equals the H_IS-weighted mean v_SB of citing "
        f"sources ({mv_s:.3f}, {'above' if mv_s > 1.0 else 'below'} the source mean of 1), "
        f"placing this institution {'above' if v_bip > 1.0 else 'below'} the bipartite average. "
        f"v_II reflects institution-institution citations, which are absent from the bipartite model."
    )


def _write_unit_effects_table(df: pd.DataFrame) -> None:
    src = df[df['type'] == 'Sources'].copy()
    ins = df[df['type'] == 'Institutions'].copy()

    case_order_s = ['min v_ss | v_bip≈1', 'max v_ss | v_bip≈1',
                    'min v_bip | v_ss≈1', 'max v_bip | v_ss≈1']
    case_order_i = ['min v_ii | v_bip≈1', 'max v_ii | v_bip≈1',
                    'min v_bip | v_ii≈1', 'max v_bip | v_ii≈1']
    case_label = {
        'min v_ss | v_bip≈1':  r'$\min v_{\rm SS}$',
        'max v_ss | v_bip≈1':  r'$\max v_{\rm SS}$',
        'min v_bip | v_ss≈1':  r'$\min v_{\rm bip}$',
        'max v_bip | v_ss≈1':  r'$\max v_{\rm bip}$',
        'min v_ii | v_bip≈1':  r'$\min v_{\rm II}$',
        'max v_ii | v_bip≈1':  r'$\max v_{\rm II}$',
        'min v_bip | v_ii≈1':  r'$\min v_{\rm bip}$',
        'max v_bip | v_ii≈1':  r'$\max v_{\rm bip}$',
    }

    def fmt(x):
        return f'{float(x):.2f}'

    def src_row(r):
        return (f'{r["name"]} & {fmt(r["v_bip"])} & {fmt(r["v_ss"])} & {fmt(r["mean_v_I_bip"])}'
                f' & {case_label[r["case"]]} \\\\\n')

    def ins_row(r):
        return (f'{r["name"]} & {fmt(r["v_bip"])} & {fmt(r["v_ii"])} & {fmt(r["mean_v_S_bip"])}'
                f' & {case_label[r["case"]]} \\\\\n')

    src_rows = ''.join(src_row(src[src['case'] == c].iloc[0]) for c in case_order_s)
    ins_rows = ''.join(ins_row(ins[ins['case'] == c].iloc[0]) for c in case_order_i)

    tex = (
        r'\begin{table}[htbp]' + '\n'
        r'\centering\small' + '\n'
        r'\caption{Illustrative sources and institutions near $v=1$, showing how' + '\n'
        r'$v_{\rm bip}$ diverges from $v_{\rm SS}$ and $v_{\rm II}$.' + '\n'
        r'$\bar{v}^{\rm bip}$ is the $H$-weighted mean bipartite rank of cross-type' + '\n'
        r'partners. Field: E\,=\,economics, B\,=\,business, A\,=\,ambiguous.}' + '\n'
        r'\label{tab:unit_effects}' + '\n'
        r'\begin{tabular}{lrrrl}' + '\n'
        r'\toprule' + '\n'
        r'\multicolumn{5}{l}{\textit{Sources}} \\[2pt]' + '\n'
        r'Name & $v_{\rm bip}$ & $v_{\rm SS}$ & $\bar{v}_{I}^{\rm bip}$ & Case \\' + '\n'
        r'\midrule' + '\n'
        + src_rows
        + r'\midrule' + '\n'
        r'\multicolumn{5}{l}{\textit{Institutions}} \\[2pt]' + '\n'
        r'Name & $v_{\rm bip}$ & $v_{\rm II}$ & $\bar{v}_{S}^{\rm bip}$ & Case \\' + '\n'
        r'\midrule' + '\n'
        + ins_rows
        + r'\bottomrule' + '\n'
        r'\end{tabular}' + '\n'
        r'\end{table}' + '\n'
    )

    tables_dir = Path(__file__).parent.parent / 'spectral_ranking_latex' / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / 'table_unit_effects.tex'
    out.write_text(tex)
    print(f'Saved {out}')


def build_unit_effects_table(series: dict, paths) -> None:
    """Generate table_unit_effects.tex — illustrative units near v=1."""
    print('Building unit-effects table ...')
    el_path = paths.working / 'edge_lists.duckdb'
    par     = paths.working / 'parquet'

    df_s_bip = series['m=0110']['S'][['unit_idx', 'v']].rename(columns={'v': 'v_bip'})
    df_i_bip = series['m=0110']['I'][['unit_idx', 'v']].rename(columns={'v': 'v_bip'})
    df_ss    = series['m=1000']['S'][['unit_idx', 'v']].rename(columns={'v': 'v_ss'})
    df_ii    = series['m=0001']['I'][['unit_idx', 'v']].rename(columns={'v': 'v_ii'})

    if df_ss.empty or df_ii.empty:
        print('  WARNING: SS or II run missing — skipping unit-effects table')
        return

    with duckdb.connect(str(el_path), read_only=True) as db:
        el = db.execute(f"""
            SELECT citer_source_idx, cited_inst_idx, SUM(cited_inst_weight) AS w
            FROM {EL_TABLE}
            WHERE citer_source_idx IS NOT NULL AND cited_inst_idx IS NOT NULL
            GROUP BY citer_source_idx, cited_inst_idx
        """).df()

    # Source → H_SI-weighted mean v_IB
    el_s = el.merge(df_i_bip.rename(columns={'unit_idx': 'cited_inst_idx', 'v_bip': 'v_i'}),
                    on='cited_inst_idx', how='inner')
    el_s['wv'] = el_s['w'] * el_s['v_i']
    agg_s = el_s.groupby('citer_source_idx')[['wv', 'w']].sum()
    agg_s['mean_v_I_bip'] = agg_s['wv'] / agg_s['w']
    agg_s = agg_s[['mean_v_I_bip']].reset_index().rename(columns={'citer_source_idx': 'unit_idx'})

    # Institution → H_IS-weighted mean v_SB
    el_i = el.merge(df_s_bip.rename(columns={'unit_idx': 'citer_source_idx', 'v_bip': 'v_s'}),
                    on='citer_source_idx', how='inner')
    el_i['wv'] = el_i['w'] * el_i['v_s']
    agg_i = el_i.groupby('cited_inst_idx')[['wv', 'w']].sum()
    agg_i['mean_v_S_bip'] = agg_i['wv'] / agg_i['w']
    agg_i = agg_i[['mean_v_S_bip']].reset_index().rename(columns={'cited_inst_idx': 'unit_idx'})

    sm = pd.read_parquet(str(par / 'source_master.parquet'),
                         columns=['source_idx', 'source_name', 'field_eb'])
    sm = sm.rename(columns={'source_idx': 'unit_idx'})

    inst_eb = pd.read_parquet(str(par / 'institution_field_eb.parquet'),
                              columns=['unit_idx', 'field_eb'])
    corp_inst = pd.read_parquet(str(par / 'corpus_institutions.parquet'),
                                columns=['institution_idx', 'institution_name'])
    corp_inst = corp_inst.rename(columns={'institution_idx': 'unit_idx'})

    non_x_src  = set(sm.loc[sm['field_eb']  != 'X', 'unit_idx'])
    non_x_inst = set(inst_eb.loc[inst_eb['field_eb'] != 'X', 'unit_idx'])

    src = (df_s_bip
           .merge(df_ss,  on='unit_idx', how='inner')
           .merge(agg_s,  on='unit_idx', how='left')
           .merge(sm[['unit_idx', 'source_name', 'field_eb']], on='unit_idx', how='inner'))
    src = src[src['unit_idx'].isin(non_x_src)].copy()

    inst = (df_i_bip
            .merge(df_ii,     on='unit_idx', how='inner')
            .merge(agg_i,     on='unit_idx', how='left')
            .merge(inst_eb,   on='unit_idx', how='inner')
            .merge(corp_inst, on='unit_idx', how='left'))
    inst = inst[inst['unit_idx'].isin(non_x_inst)].copy()
    inst = inst.rename(columns={'institution_name': 'source_name'})

    rows = []
    for band_col, alt_col, case_lo, case_hi, utype, data in [
        ('v_ss',  'v_bip', 'min v_bip | v_ss≈1',  'max v_bip | v_ss≈1',  'Sources',      src),
        ('v_bip', 'v_ss',  'min v_ss | v_bip≈1',   'max v_ss | v_bip≈1',  'Sources',      src),
        ('v_ii',  'v_bip', 'min v_bip | v_ii≈1',   'max v_bip | v_ii≈1',  'Institutions', inst),
        ('v_bip', 'v_ii',  'min v_ii | v_bip≈1',   'max v_ii | v_bip≈1',  'Institutions', inst),
    ]:
        band = data[np.abs(np.log(data[band_col])) < V1_BAND].copy()
        if not band.empty:
            rows.append((utype, case_lo, band.loc[band[alt_col].idxmin()]))
            rows.append((utype, case_hi, band.loc[band[alt_col].idxmax()]))

    records = []
    for unit_type, case, r in rows:
        is_src = unit_type == 'Sources'
        records.append({
            'type':         unit_type,
            'case':         case,
            'name':         r['source_name'],
            'field_eb':     r['field_eb'],
            'v_bip':        round(float(r['v_bip']), 3),
            'v_ss':         round(float(r['v_ss']),  3) if is_src  else '',
            'mean_v_I_bip': round(float(r['mean_v_I_bip']), 3) if is_src else '',
            'v_ii':         '' if is_src else round(float(r['v_ii']),  3),
            'mean_v_S_bip': '' if is_src else round(float(r['mean_v_S_bip']), 3),
            'explanation':  _explain_source(r) if is_src else _explain_institution(r),
        })

    out_df = pd.DataFrame(records)
    csv_path = paths.plots / 'fig3_v1_examples.csv'
    out_df.to_csv(str(csv_path), index=False)
    print(f'  Saved {csv_path}')
    _write_unit_effects_table(out_df)

    src_hl = out_df[out_df['type'] == 'Sources'][['v_bip', 'v_ss']].copy()
    src_hl['v_bip'] = src_hl['v_bip'].astype(float)
    src_hl['v_ss']  = src_hl['v_ss'].astype(float)

    inst_hl = out_df[out_df['type'] == 'Institutions'][['v_bip', 'v_ii']].copy()
    inst_hl['v_bip'] = inst_hl['v_bip'].astype(float)
    inst_hl['v_ii']  = inst_hl['v_ii'].astype(float)

    return src_hl, inst_hl


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    paths   = load_config()
    rk_path = paths.working / 'rankings.duckdb'

    if not rk_path.exists():
        raise FileNotFoundError(
            f'rankings.duckdb not found at {rk_path}. '
            'Run spectral_ranking/run_rankings.py first.'
        )

    with duckdb.connect(str(rk_path), read_only=True) as db:
        print('Loading runs:')
        src_rank_map, inst_rank_map, series = fetch_data(db)

    print('Including field_eb=X units in plots (no non-X filtering).')

    plot3(src_rank_map, inst_rank_map, series)
    src_highlights, inst_highlights = build_unit_effects_table(series, paths)
    plot3a(src_rank_map, inst_rank_map, series,
           src_highlights=src_highlights, inst_highlights=inst_highlights)


if __name__ == '__main__':
    main()
    print('FINISHED!')
