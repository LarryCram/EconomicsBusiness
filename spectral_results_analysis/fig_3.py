"""
fig_3.py — Prestige-per-work rank curves across network mode m.

Shows how the v distribution changes with m, with x-axis locked to the
bipartite baseline rank order (same convention as fig_2.py).

  m=0110  bipartite SI/IS   — black line  (baseline, both panels)
  m=1000  source-only SS    — red X       (S panel only; no institutions)
  m=0001  institution-only  — red X       (I panel only; no sources)
  m=1111  full joint χ*     — green X     (both panels; χ* resolved from _catalog)

x-axis: LOCKED to baseline (m=0110) rank order.
  Each unit sits at its baseline rank; units absent from an alternative run
  (e.g. institutions in m=1000) are simply omitted from that series.

All runs: F=A, t_x=5, τ_U=tau_u_floor['A'], ρ=fixed, α=0.85.

Outputs:
  plots/fig_3.pdf        — with title (exploration)
  plots/fig_3_latex.pdf  — without title (paper)
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_params

_tau = load_params()['tau_u_floor']['A']

BASELINE_TABLE = f'rk_t5_A_tau{_tau}_rho0_m0110_chi50_alpha85'
SS_TABLE       = f'rk_t5_A_tau{_tau}_rho0_m1000_chi50_alpha85'
II_TABLE       = f'rk_t5_A_tau{_tau}_rho0_m0001_chi50_alpha85'

# Visual spec per label
STYLE = {
    'm=0110': dict(color='black',   marker=None, zorder=3, lw=1.4, alpha=1.0),
    'm=1000': dict(color='#d62728', marker='x',  zorder=2, s=40,   lw=0.8),
    'm=0001': dict(color='#d62728', marker='x',  zorder=2, s=40,   lw=0.8),
    'm=1111': dict(color='#2ca02c', marker='x',  zorder=2, s=40,   lw=0.8),
}

S_LABELS = ['m=0110', 'm=1000', 'm=1111']
I_LABELS = ['m=0110', 'm=0001', 'm=1111']


# ─── Field labels ─────────────────────────────────────────────────────────────

def load_field_labels(paths) -> dict:
    """Return {source_idx (int): 'E' | 'B'} from source_master.csv in data/."""
    sm = pd.read_csv(paths.data / 'source_master.csv',
                     usecols=['source_idx', 'field_subset'])
    sm = sm.dropna(subset=['field_subset'])
    return dict(zip(sm['source_idx'].astype(int), sm['field_subset']))


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
        f"  AND tx=5 AND fx='A' AND tau_u={_tau} AND rho=0 "
        "  AND round(alpha*100)=85 AND round(chi*100) != 50 "
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
                n_baseline: int, panel_title: str,
                field_labels: dict | None = None) -> None:
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
                alpha=0.55,
                label=f'{label}  ({n_overlap:,}/{n_baseline:,})',
            )

    # ── E/B overlay on S panel (black circles = E, black squares = B) ─────────
    if unit_key == 'S' and field_labels and 'm=1000' in series:
        df_ss = series['m=1000']['S']
        if not df_ss.empty:
            df_ss = df_ss.copy()
            df_ss['F'] = df_ss['unit_idx'].map(field_labels)
            for f_val, marker, legend_label in [('E', 'o', 'E (SS)'),
                                                ('B', 's', 'B (SS)')]:
                sub = df_ss[df_ss['F'] == f_val]
                if sub.empty:
                    continue
                ax.scatter(
                    sub['baseline_rank'].values,
                    sub['v'].values,
                    color='black',
                    marker=marker,
                    s=18,
                    linewidths=0.5,
                    facecolors='none',
                    zorder=4,
                    label=f'{legend_label}  ({len(sub):,})',
                )

    ax.set_yscale('log')
    ax.axhline(1.0, color='#999999', linewidth=0.8, linestyle='--', zorder=0)
    ax.text(
        n_baseline * 0.98, 1.0,
        '$v=1$',
        ha='right', va='bottom',
        fontsize=7.5, color='#999999',
    )
    ax.set_xlim(1, n_baseline)
    ax.set_xlabel('Baseline rank  (m=0110)', labelpad=4)
    ax.set_ylabel('Prestige per work $v$', labelpad=4)
    ax.set_title(panel_title, fontsize=10, pad=6)
    ax.legend(fontsize=8, framealpha=0.85, loc='upper right')


def plot3(src_rank_map: dict, inst_rank_map: dict, series: dict,
          field_labels: dict) -> None:
    paths = load_config()

    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    fig.subplots_adjust(hspace=0.44)

    _draw_panel(axes[0], series, 'S', S_LABELS,
                n_baseline=len(src_rank_map),  panel_title='Sources',
                field_labels=field_labels)
    _draw_panel(axes[1], series, 'I', I_LABELS,
                n_baseline=len(inst_rank_map), panel_title='Institutions')

    sup = fig.suptitle(
        'Prestige per work — sensitivity to network mode $m$  '
        '(x-axis locked to bipartite baseline)',
        fontsize=9, y=1.01,
    )

    out = paths.plots / 'fig_3.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

    sup.set_visible(False)
    latex_out = paths.plots / 'fig_3_latex.pdf'
    fig.savefig(latex_out, bbox_inches='tight')
    print(f'Saved {latex_out}')
    sup.set_visible(True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    paths = load_config()
    rk_path = paths.working / 'rankings.duckdb'

    if not rk_path.exists():
        raise FileNotFoundError(
            f'rankings.duckdb not found at {rk_path}. '
            'Run spectral_ranking/run_rankings.py first.'
        )

    with duckdb.connect(str(rk_path), read_only=True) as db:
        print('Loading runs:')
        src_rank_map, inst_rank_map, series = fetch_data(db)

    paths = load_config()
    field_labels = load_field_labels(paths)
    print(f'Field labels loaded: {sum(v=="E" for v in field_labels.values())} E, '
          f'{sum(v=="B" for v in field_labels.values())} B')

    plot3(src_rank_map, inst_rank_map, series, field_labels)


if __name__ == '__main__':
    main()
    print('FINISHED!')
