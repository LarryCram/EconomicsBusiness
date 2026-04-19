"""
fig_6.py — Time-series comparison: baseline (2020–24) vs t1-fix–t4-fix.

1×2 layout with shared y-axis:

    Left panel:  Sources (fixed-universe runs; + markers, solid running mean)
    Right panel: Institutions (fixed-universe runs; + markers, solid running mean)

Outputs:
    plots/fig_6.pdf        — 1×2 with title (exploration)
    plots/fig_6_latex.pdf  — 1×2 without title (paper)

Console:
    Per-period table: n_tau, n_fix, n_dropped (SCC), Spearman ρ(v_fix, v_tau).
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

_runs     = load_runs()
_baseline = next(r for r in _runs if r['label'] == 'baseline')

_TAU_LABELS = {'t1', 't2', 't3', 't4'}
_FIX_LABELS = {'t1-fix', 't2-fix', 't3-fix', 't4-fix'}

_tau_runs = sorted(
    [r for r in _runs if r['label'] in _TAU_LABELS],
    key=lambda r: r['label'],
)
_fix_runs = sorted(
    [r for r in _runs if r['label'] in _FIX_LABELS],
    key=lambda r: r['label'],
)

_COLOURS      = ['#7b00d4', '#0057ff', '#00aa00', '#ff1500']
BASELINE_COLOUR = 'black'

_MU_SUFFIX = {'': '', 'uniform': '_muUniform', 'unit_scaled': '_muUnitScaled'}


def _table_name(r: dict) -> str:
    tau_sfx   = '_fixtau' if r.get('ref_units', '') else '_vartau'
    chi_str   = 'STAR' if r['chi'] == -1.0 else str(round(r['chi'] * 100))
    alpha_int = round(r['alpha'] * 100)
    mu_sfx    = _MU_SUFFIX.get(r.get('mu_type', ''), f"_mu{r.get('mu_type', '')}")
    return (f"rk_{r['run_code']}_{r['fx']}"
            f"_tauU{r['tau_u']}_tauS{r['tau_s']}{tau_sfx}"
            f"_rho{r['rho']}_m{r['m']}"
            f"_chi{chi_str}_alpha{alpha_int}{mu_sfx}")


def _period_label(r: dict) -> str:
    return f"{r['tc0']}–{str(r['tc1'])[-2:]}"


# ─── Data ─────────────────────────────────────────────────────────────────────

def fetch_data(db) -> tuple:
    """
    Returns
    -------
    src_rank_map, inst_rank_map : unit_idx → baseline_rank dicts
    df_s_base, df_i_base        : baseline DataFrames
    pairs : list of (period, colour, df_s_tau, df_i_tau, df_s_fix, df_i_fix)
            Each df has columns [unit_idx, baseline_rank, v].
            df_s_fix / df_i_fix are None when the fixed-universe run is absent.
    """
    baseline_tname = _table_name(_baseline)
    tables = {row[0] for row in db.execute('SHOW TABLES').fetchall()}

    if baseline_tname not in tables:
        raise RuntimeError(
            f'Table {baseline_tname} not found in rankings.duckdb. '
            'Run run_rankings.py first.'
        )

    # ── Baseline: x-axis lock ─────────────────────────────────────────────────
    df_base = db.execute(
        f'SELECT unit_idx, unit_type, v FROM {baseline_tname}'
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

    # ── Build τ-per-window / fixed-universe pairs ─────────────────────────────
    # Pair by sorted label: t1↔t1-fix, t2↔t2-fix, …
    # If a run is missing from the DB, its df is None.
    pairs = []
    for r_tau, r_fix, colour in zip(_tau_runs, _fix_runs, _COLOURS):
        period = _period_label(r_tau)
        tn_tau = _table_name(r_tau)
        tn_fix = _table_name(r_fix)

        if tn_tau not in tables:
            print(f'  WARNING: {tn_tau} not found — skipping {period}')
            continue

        df_tau = db.execute(f'SELECT unit_idx, unit_type, v FROM {tn_tau}').df()
        df_s_tau = _project(df_tau[df_tau['unit_type'] == 'S'], src_rank_map)
        df_i_tau = _project(df_tau[df_tau['unit_type'] == 'U'], inst_rank_map)

        if tn_fix in tables:
            df_fix = db.execute(f'SELECT unit_idx, unit_type, v FROM {tn_fix}').df()
            df_s_fix = _project(df_fix[df_fix['unit_type'] == 'S'], src_rank_map)
            df_i_fix = _project(df_fix[df_fix['unit_type'] == 'U'], inst_rank_map)
        else:
            print(f'  NOTE: {tn_fix} not found — fixed-universe series absent for {period}')
            df_s_fix = df_i_fix = None

        pairs.append((period, colour, df_s_tau, df_i_tau, df_s_fix, df_i_fix))

    return src_rank_map, inst_rank_map, df_s_base, df_i_base, pairs


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def _running_mean_log(v_series: np.ndarray, w: int) -> np.ndarray:
    log_v = np.log10(v_series)
    return np.power(10.0,
        np.asarray(pd.Series(log_v).rolling(w, center=True, min_periods=1).mean(),
                   dtype=float))


def _draw_scatter_panel(ax, df_s_base, df_i_base, pairs,
                        unit_idx: int, n_baseline: int,
                        panel_title: str, mode: str) -> None:
    """
    unit_idx : 0 = sources, 1 = institutions.
    mode     : 'tau' = τ-per-window series; 'fix' = fixed-universe series.
    Baseline curve always shown.
    """
    df_b = df_s_base if unit_idx == 0 else df_i_base
    ax.plot(df_b['baseline_rank'], df_b['v'],
            color=BASELINE_COLOUR, linewidth=1.4, alpha=1.0, zorder=4,
            label=_period_label(_baseline))

    any_series = False
    for period, colour, df_s_tau, df_i_tau, df_s_fix, df_i_fix in pairs:
        if mode == 'tau':
            df = df_s_tau if unit_idx == 0 else df_i_tau
            marker, ls = 'x', '--'
        else:
            df = df_s_fix if unit_idx == 0 else df_i_fix
            if df is None:
                continue
            marker, ls = '+', '-'

        if df.empty:
            continue
        any_series = True
        n = len(df)

        ax.scatter(df['baseline_rank'], df['v'],
                   color=colour, marker=marker, s=28, linewidths=0.7,
                   alpha=0.28, zorder=2,
                   label=f'{period}  ({n:,}/{n_baseline:,})')
        w = max(50, len(df) // 10)
        ax.plot(df['baseline_rank'],
                _running_mean_log(df['v'].values, w),
                color=colour, linewidth=1.3, linestyle=ls, alpha=0.85, zorder=3)

    if not any_series and mode == 'fix':
        ax.text(0.5, 0.5, 'Fixed-universe runs\nnot yet available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=9, color='grey')

    ax.set_yscale('log')
    ax.set_ylim(0.02, 20)
    ax.axhline(1.0, color='#999999', linewidth=0.8, linestyle='--', zorder=0)
    ax.text(n_baseline * 0.98, 1.0, '$v=1$',
            ha='right', va='bottom', fontsize=7.5, color='#999999')
    ax.set_xlim(1, n_baseline)
    ax.set_xlabel('Baseline rank', labelpad=4)
    ax.set_title(panel_title, fontsize=10, pad=6)
    handles, labels = ax.get_legend_handles_labels()
    # baseline (black line) was added first — move it to the bottom
    ax.legend(handles[1:] + handles[:1], labels[1:] + labels[:1],
              fontsize=6.0, framealpha=0.85, loc='upper right', ncol=1)


# ─── Console summary ──────────────────────────────────────────────────────────

def print_summary(pairs, n_base_s: int, n_base_i: int) -> None:
    print(f'\n{"Period":<10}  '
          f'{"τ-pw S":>7}  {"fix S":>7}  {"drop S":>7}  {"ρ_S":>6}  |  '
          f'{"τ-pw I":>7}  {"fix I":>7}  {"drop I":>7}  {"ρ_I":>6}')
    print('-' * 90)

    for period, _, df_s_tau, df_i_tau, df_s_fix, df_i_fix in pairs:
        def _rho(a, b):
            if a is None or b is None or len(a) < 3 or len(b) < 3:
                return float('nan')
            m = a.merge(b[['unit_idx', 'v']].rename(columns={'v': 'vb'}),
                        on='unit_idx', how='inner')
            return spearmanr(m['v'], m['vb']).statistic if len(m) >= 3 else float('nan')

        n_fix_s = len(df_s_fix) if df_s_fix is not None else 0
        n_fix_i = len(df_i_fix) if df_i_fix is not None else 0
        drop_s  = n_base_s - n_fix_s
        drop_i  = n_base_i - n_fix_i
        rho_s   = _rho(df_s_fix, df_s_tau)
        rho_i   = _rho(df_i_fix, df_i_tau)

        print(f'{period:<10}  '
              f'{len(df_s_tau):>7,}  {n_fix_s:>7,}  {drop_s:>7,}  {rho_s:>6.3f}  |  '
              f'{len(df_i_tau):>7,}  {n_fix_i:>7,}  {drop_i:>7,}  {rho_i:>6.3f}')

    print(f'\nBaseline universe: {n_base_s:,} sources, {n_base_i:,} institutions')
    print('drop = baseline units absent from fixed-universe ranking (SCC filter)')
    print('ρ    = Spearman correlation between v_fix and v_tau on common units')


# ─── Main ─────────────────────────────────────────────────────────────────────

def plot6(src_rank_map, inst_rank_map, df_s_base, df_i_base, pairs) -> None:
    paths = load_config()
    sns.set_theme(style='whitegrid', font_scale=0.95)

    n_base_s = len(src_rank_map)
    n_base_i = len(inst_rank_map)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    fig.subplots_adjust(wspace=0.10)

    _draw_scatter_panel(axes[0], df_s_base, df_i_base, pairs,
                        0, n_base_s, 'Sources', mode='fix')
    _draw_scatter_panel(axes[1], df_s_base, df_i_base, pairs,
                        1, n_base_i, 'Institutions', mode='fix')

    # Shared y-axis label on the left panel only
    axes[0].set_ylabel('Influence per work $v$', labelpad=4)
    axes[1].set_ylabel('')

    sup_text = 'Time-series comparison (fixed universe)'
    sup = fig.suptitle(sup_text, fontsize=9, y=1.01)

    out = paths.plots / 'fig_6.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

    sup.set_visible(False)
    latex_out = paths.plots / 'fig_6_latex.pdf'
    fig.savefig(latex_out, bbox_inches='tight')
    print(f'Saved {latex_out}')
    sup.set_visible(True)
    plt.close(fig)


def main():
    paths   = load_config()
    rk_path = paths.working / 'rankings.duckdb'

    if not rk_path.exists():
        raise FileNotFoundError(
            f'rankings.duckdb not found at {rk_path}. '
            'Run spectral_ranking/run_rankings.py first.'
        )

    with duckdb.connect(str(rk_path), read_only=True) as db:
        src_rank_map, inst_rank_map, df_s_base, df_i_base, pairs = fetch_data(db)

    n_base_s = len(src_rank_map)
    n_base_i = len(inst_rank_map)

    print_summary(pairs, n_base_s, n_base_i)
    plot6(src_rank_map, inst_rank_map, df_s_base, df_i_base, pairs)


if __name__ == '__main__':
    main()
    print('FINISHED!')
