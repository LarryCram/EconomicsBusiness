"""
fig_6.py — Time-series comparison: baseline (2020–24) vs t1–t4.

Baseline: run_code=20242024, F=A, τ_U=τ_S=20, ρ=0, m=0110, α=1, χ=0.5.
x-axis locked to baseline rank (same convention as fig_2/fig_3).

Four overlays (stage-2 runs):
  t1  2000–04   lightest blue
  t2  2005–09
  t3  2010–14
  t4  2015–19   darkest blue

All overlays use F=A, τ=20, ρ=0, m=0110, α=1, χ=0.5.
Units absent from an earlier-period corpus are omitted from that series.

Outputs:
  plots/fig_6.pdf        — with title (exploration)
  plots/fig_6_latex.pdf  — without title (paper)
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

_runs     = load_runs()
_baseline = next(r for r in _runs if r['label'] == 'baseline')
_overlays = [r for r in _runs if r['stage'] == 2]  # t1, t2, t3, t4 in CSV order

# Sequential blues: lightest (t1=oldest) → darkest (t4=most recent)
_BLUES = ['#bdd7e7', '#6baed6', '#2171b5', '#084594']

BASELINE_COLOUR = 'black'


def _table_name(r: dict) -> str:
    chi     = r['chi']
    chi_str = 'STAR' if chi == -1.0 else str(round(chi * 100))
    alpha_int = round(r['alpha'] * 100)
    return (f"rk_{r['run_code']}_{r['fx']}"
            f"_tauU{r['tau_u']}_tauS{r['tau_s']}"
            f"_rho{r['rho']}_m{r['m']}"
            f"_chi{chi_str}_alpha{alpha_int}")


def _period_label(r: dict) -> str:
    return f"{r['tc0']}–{str(r['tc1'])[-2:]}"


# ─── Data ─────────────────────────────────────────────────────────────────────

def fetch_data(db) -> tuple:
    """
    Returns
    -------
    src_rank_map, inst_rank_map : x-axis lock dicts
    series : list of (display_label, colour, df_s, df_i)
             baseline first, then t1→t4
    """
    baseline_tname = _table_name(_baseline)
    tables = {row[0] for row in db.execute('SHOW TABLES').fetchall()}

    if baseline_tname not in tables:
        raise RuntimeError(
            f'Table {baseline_tname} not found in rankings.duckdb. '
            'Run run_rankings.py first.'
        )

    # ── Baseline: establish x-axis lock ──────────────────────────────────────
    df_base = db.execute(
        f"SELECT unit_idx, unit_type, v FROM {baseline_tname}"
    ).df()

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

    def _project(df, rank_map):
        out = df.copy()
        out['baseline_rank'] = out['unit_idx'].map(rank_map)
        return out.dropna(subset=['baseline_rank']).sort_values('baseline_rank')

    series = []

    # ── Overlays: t1 → t4 (chronological, lightest → darkest blue) ───────────
    for run, colour in zip(_overlays, _BLUES):
        tname = _table_name(run)
        period = _period_label(run)
        if tname not in tables:
            print(f'  WARNING: {tname} not found — skipping {period}')
            continue
        df = db.execute(
            f"SELECT unit_idx, unit_type, v FROM {tname}"
        ).df()
        df_s = _project(df[df['unit_type'] == 'S'], src_rank_map)
        df_i = _project(df[df['unit_type'] == 'U'], inst_rank_map)
        series.append((period, colour, df_s, df_i))

    # ── Baseline last so it is drawn on top and appears last in legend ────────
    series.append((_period_label(_baseline), BASELINE_COLOUR, df_s_base, df_i_base))

    return src_rank_map, inst_rank_map, series


# ─── Plot ─────────────────────────────────────────────────────────────────────

def _draw_panel(ax, series: list, unit_idx: int,
                n_baseline: int, panel_title: str) -> None:
    """
    unit_idx: 2 = df_s (sources), 3 = df_i (institutions)
    """
    for i, (period, colour, df_s, df_i) in enumerate(series):
        df = df_s if unit_idx == 2 else df_i
        if df.empty:
            continue

        is_baseline = (colour == BASELINE_COLOUR)
        n_overlap   = len(df)

        if is_baseline:
            ax.plot(
                df['baseline_rank'].values,
                df['v'].values,
                color=colour,
                linewidth=1.4,
                alpha=1.0,
                zorder=3,
                label=period,
            )
        else:
            ax.scatter(
                df['baseline_rank'].values,
                df['v'].values,
                color=colour,
                marker='x',
                s=35,
                linewidths=0.8,
                alpha=0.65,
                zorder=2,
                label=f'{period}  ({n_overlap:,}/{n_baseline:,})',
            )

    ax.set_yscale('log')
    ax.axhline(1.0, color='#999999', linewidth=0.8, linestyle='--', zorder=0)
    ax.text(
        n_baseline * 0.98, 1.0, '$v=1$',
        ha='right', va='bottom', fontsize=7.5, color='#999999',
    )
    ax.set_xlim(1, n_baseline)
    ax.set_xlabel('Baseline rank', labelpad=4)
    ax.set_ylabel('Prestige per work $v$', labelpad=4)
    ax.set_title(panel_title, fontsize=10, pad=6)
    ax.legend(fontsize=7.5, framealpha=0.85, loc='upper right')


def plot6(src_rank_map: dict, inst_rank_map: dict, series: list) -> None:
    paths = load_config()
    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    fig.subplots_adjust(hspace=0.44)

    _draw_panel(axes[0], series, 2, len(src_rank_map),  'Sources')
    _draw_panel(axes[1], series, 3, len(inst_rank_map), 'Institutions')

    sup = fig.suptitle(
        'Time-series comparison — prestige per work  '
        '(x-axis locked to 2020–24 baseline)',
        fontsize=9, y=1.01,
    )

    out = paths.plots / 'fig_6.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

    sup.set_visible(False)
    latex_out = paths.plots / 'fig_6_latex.pdf'
    fig.savefig(latex_out, bbox_inches='tight')
    print(f'Saved {latex_out}')
    sup.set_visible(True)

    plt.close(fig)

    # Console summary
    print(f'\n{"Period":<12}  {"n_S":>7}  {"n_I":>7}')
    print('-' * 30)
    for period, _, df_s, df_i in series:
        print(f'{period:<12}  {len(df_s):>7,}  {len(df_i):>7,}')


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
        src_rank_map, inst_rank_map, series = fetch_data(db)

    print(f'Loaded {len(series)} series (baseline + {len(series) - 1} overlays)')
    plot6(src_rank_map, inst_rank_map, series)


if __name__ == '__main__':
    main()
    print('FINISHED!')
