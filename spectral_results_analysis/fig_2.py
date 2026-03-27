"""
fig_2.py — Prestige-per-work distribution with parameter sensitivity overlay.

Baseline parameters: t_x=5, F=A, τ_U=10, ρ=fixed, m=(0,1,1,0), α=0.85.

Two-panel figure:
  Top:    sources (journals)
  Bottom: institutions
  x-axis: LOCKED to baseline rank order (rank 1 = highest v in baseline)
  y-axis: prestige per work v  (log scale; v=1 is corpus-average)

The baseline curve is drawn in black.  Every other available run in
rankings.duckdb is overplotted at the same x-positions (each unit sits at
its baseline rank regardless of how its rank changes under the alternative
parameters).  Units absent from an alternative corpus (e.g. non-economics
sources when F=E) are simply omitted from that line.

Outputs:
  plots/fig_2.pdf        — with title (exploration)
  plots/fig_2_latex.pdf  — without title (paper)
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config

BASELINE_TABLE = 'rk_t5_A_tau10_rho0_m0110_chi50_alpha85'
BASELINE_LABEL = 'baseline'


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_run(db, table_name: str) -> pd.DataFrame:
    return db.execute(
        f"SELECT unit_idx, unit_type, v FROM {table_name}"
    ).df()


def fetch_all(db) -> tuple:
    """
    Returns
    -------
    src_rank_map  : dict  unit_idx -> baseline_rank  (sources)
    inst_rank_map : dict  unit_idx -> baseline_rank  (institutions)
    runs          : list of (label, df_s_projected, df_i_projected)
                    sorted so baseline comes first
    catalog       : DataFrame with one row per available run
    """
    catalog = db.execute(
        "SELECT table_name, label FROM _catalog ORDER BY label"
    ).df()

    # ── Baseline: establish the x-axis lock ──────────────────────────────
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

    # ── Load every run and project onto baseline ranks ────────────────────
    runs = []
    for _, row in catalog.iterrows():
        tname = row['table_name']
        label = row['label']
        df = load_run(db, tname)

        df_s = df[df['unit_type'] == 'S'].copy()
        df_s['baseline_rank'] = df_s['unit_idx'].map(src_rank_map)
        df_s = (df_s.dropna(subset=['baseline_rank'])
                .sort_values('baseline_rank'))

        df_i = df[df['unit_type'] == 'U'].copy()
        df_i['baseline_rank'] = df_i['unit_idx'].map(inst_rank_map)
        df_i = (df_i.dropna(subset=['baseline_rank'])
                .sort_values('baseline_rank'))

        runs.append((label, df_s, df_i))

    # Baseline first, rest alphabetical
    runs.sort(key=lambda x: (x[0] != BASELINE_LABEL, x[0]))
    return src_rank_map, inst_rank_map, runs


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot2(src_rank_map: dict, inst_rank_map: dict, runs: list) -> None:
    paths = load_config()

    n_s = len(src_rank_map)
    n_i = len(inst_rank_map)
    n_runs = len(runs)

    # Baseline in black; first two non-baseline runs in red and blue; rest skipped.
    OVERLAY_COLOURS = ['#e41a1c', '#377eb8']  # red, blue
    colour_iter = iter(OVERLAY_COLOURS)
    run_colours = {}
    for label, _, _ in runs:
        if label == BASELINE_LABEL:
            run_colours[label] = 'black'
        elif len(run_colours) - 1 < len(OVERLAY_COLOURS):
            run_colours[label] = next(colour_iter)
        else:
            run_colours[label] = None   # None = skip

    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    fig.subplots_adjust(hspace=0.44)

    panel_specs = [
        (axes[0], 'df_s', n_s,  'Sources'),
        (axes[1], 'df_i', n_i, 'Institutions'),
    ]

    for ax, df_key, n_baseline, panel_label in panel_specs:
        for label, df_s, df_i in runs:
            df = df_s if df_key == 'df_s' else df_i
            if df.empty:
                continue

            colour = run_colours.get(label)
            if colour is None:
                continue
            is_baseline = (label == BASELINE_LABEL)
            lw      = 1.4 if is_baseline else 0.75
            alpha   = 1.0 if is_baseline else 0.65
            zorder  = 3  if is_baseline else 2
            # Coverage annotation: how many baseline units appear in this run
            n_overlap = len(df)

            ax.plot(
                df['baseline_rank'].values,
                df['v'].values,
                color=colour,
                linewidth=lw,
                alpha=alpha,
                zorder=zorder,
                label=f'{label}  ({n_overlap:,}/{n_baseline:,})'
                      if not is_baseline else f'{label}',
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
        ax.set_xlabel('Baseline rank', labelpad=4)
        ax.set_ylabel('Prestige per work $v$', labelpad=4)
        ax.set_title(panel_label, fontsize=10, pad=6)

        ax.legend(
            fontsize=7,
            framealpha=0.85,
            loc='upper right',
            ncol=2 if n_runs > 5 else 1,
        )

    sup = fig.suptitle(
        'Parameter sensitivity — prestige per work  '
        f'(x-axis locked to {BASELINE_LABEL})',
        fontsize=9, y=1.01,
    )

    out = paths.plots / 'fig_2.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

    sup.set_visible(False)
    latex_out = paths.plots / 'fig_2_latex.pdf'
    fig.savefig(latex_out, bbox_inches='tight')
    print(f'Saved {latex_out}')
    sup.set_visible(True)

    # Console summary
    print(f'\n{"Label":<18}  {"n_S":>6}  {"n_I":>6}')
    print('-' * 34)
    for label, df_s, df_i in runs:
        print(f'{label:<18}  {len(df_s):>6,}  {len(df_i):>6,}')


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
        tables = {row[0] for row in db.execute('SHOW TABLES').fetchall()}
        if BASELINE_TABLE not in tables:
            raise RuntimeError(
                f'Table {BASELINE_TABLE} not found in rankings.duckdb. '
                'Run run_rankings.py --baseline-only first.'
            )
        src_rank_map, inst_rank_map, runs = fetch_all(db)

    print(f'Loaded {len(runs)} run(s) from rankings.duckdb')
    plot2(src_rank_map, inst_rank_map, runs)


if __name__ == '__main__':
    main()
    print('FINISHED!')
