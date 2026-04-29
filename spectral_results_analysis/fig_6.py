"""
fig_6.py — Time-series comparison: baseline (2020–24) vs t1-fix–t4-fix.

1×2 layout with shared y-axis:

    Left panel:  Sources      (fixed-universe runs; + markers, adaptive running mean)
    Right panel: Institutions (fixed-universe runs; + markers, adaptive running mean)

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

def _running_mean_log_adaptive(v_series: np.ndarray, ranks: np.ndarray,
                               n_baseline: int,
                               w_lo: float = 0.01,
                               w_hi: float = 0.10) -> np.ndarray:
    """
    Centered running geometric mean with rank-adaptive window width.

    Window (as fraction of n_present) grows linearly from w_lo at rank 1
    to w_hi at rank n_baseline, giving fine resolution at the steep top end.
    """
    log_v = np.log10(np.clip(v_series, 1e-10, None))
    n = len(log_v)
    result = np.empty(n)
    for i in range(n):
        frac = ranks[i] / n_baseline
        w = max(3, round(n * (w_lo + (w_hi - w_lo) * frac)))
        lo = max(0, i - w // 2)
        hi = min(n, i + w // 2 + 1)
        result[i] = log_v[lo:hi].mean()
    return np.power(10.0, result)


def _draw_scatter_panel(ax, df_s_base, df_i_base, pairs,
                        unit_idx: int, n_baseline: int,
                        panel_title: str, mode: str,
                        x_max: int | None = None) -> None:
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
        ax.plot(df['baseline_rank'],
                _running_mean_log_adaptive(df['v'].values,
                                           df['baseline_rank'].values,
                                           n_baseline),
                color=colour, linewidth=1.3, linestyle=ls, alpha=0.85, zorder=3)

    if not any_series and mode == 'fix':
        ax.text(0.5, 0.5, 'Fixed-universe runs\nnot yet available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=9, color='grey')

    x_lim = x_max if x_max is not None else n_baseline
    ax.set_yscale('log')
    ax.set_ylim(0.02, 20)
    ax.axhline(1.0, color='#999999', linewidth=0.8, linestyle='--', zorder=0)
    ax.text(x_lim * 0.98, 1.0, '$v=1$',
            ha='right', va='bottom', fontsize=7.5, color='#999999')
    ax.set_xlim(1, x_lim)
    ax.set_xlabel('Baseline rank', labelpad=4)
    ax.set_title(panel_title, fontsize=10, pad=6)
    handles, labels = ax.get_legend_handles_labels()
    # baseline (black line) was added first — move it to the bottom
    ax.legend(handles[1:] + handles[:1], labels[1:] + labels[:1],
              fontsize=6.0, framealpha=0.85, loc='upper right', ncol=1)


# ─── Network concentration diagnostics ───────────────────────────────────────

def _gini(arr: np.ndarray) -> float:
    arr = np.sort(arr.astype(float))
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n)


def print_network_stats(db_el, runs: list) -> None:
    """
    For each window, query the edge list and compute:
      Gini   — Gini coefficient of citation in-weight per source
               (in-weight = Σ 1/R_i over all references received)
      Top-5  — share of total in-weight absorbed by the top 5 sources
      mean_log_v (geometric mean proxy) — printed from the ranking if available,
               but network stats are from the raw edge list only.

    A lower Gini / higher n_src in 2000 relative to 2020 shows the network
    is more evenly arranged, explaining the lower max(v).
    """
    tables = {row[0] for row in db_el.execute('SHOW TABLES').fetchall()}
    print(f'\n{"Label":<10}  {"Period":<10}  {"n_src":>6}  '
          f'{"Gini in-wt":>11}  {"Top-5 shr":>10}  {"Top-1 shr":>10}')
    print('-' * 65)
    for r in runs:
        tau_sfx  = '_fixtau' if r.get('ref_units', '') else '_vartau'
        el_table = (f"el_{r['run_code']}_{r['fx']}"
                    f"_tauU{r['tau_u']}_tauS{r['tau_s']}{tau_sfx}")
        if el_table not in tables:
            continue
        df = db_el.execute(f"""
            SELECT cited_source_idx, SUM(1.0 / R_i) AS in_weight
            FROM {el_table}
            GROUP BY cited_source_idx
        """).df()
        w      = df['in_weight'].to_numpy()
        w_sort = np.sort(w)[::-1]
        g      = _gini(w)
        top5   = w_sort[:5].sum()  / w.sum()
        top1   = w_sort[:1].sum()  / w.sum()
        period = f"{r['tc0']}–{str(r['tc1'])[-2:]}"
        print(f"{r['label']:<10}  {period:<10}  {len(w):>6,}  "
              f"{g:>11.4f}  {top5:>10.4f}  {top1:>10.4f}")


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
    print()
    print('Column definitions:')
    print('  τ-pw  = citations from that period, sources = journals with ≥τ papers')
    print('          in THAT period  (e.g. 2000-04 active journals for t1)')
    print('  fix   = same citations, but sources restricted to the 2020-24 baseline')
    print('          journals only  (journals active then but not now are excluded)')
    print('  drop  = baseline journals absent from the fix ranking because they')
    print('          had too few citations in that period to enter the SCC')
    print()
    print('Spearman ρ  (comparing fix vs τ-pw on their common journals):')
    print('  Both runs use the same period\'s citations. They differ only in which')
    print('  journals are included. ρ measures whether including extra journals')
    print('  (those active in 2000-04 but gone by 2020-24, or vice versa) changes')
    print('  the rank order of the journals that appear in both rankings.')
    print('  ρ ≈ 1  → the ranking of surviving journals is unaffected by whether')
    print('           defunct journals are included in the network.')
    print('  ρ < 1  → defunct/new journals act as significant conduits and their')
    print('           presence reshuffles the rankings of the shared journals.')


# ─── v>1 works table ─────────────────────────────────────────────────────────

def print_v_gt1_table(db_rk, db_el, runs: list) -> None:
    """
    Work-level v>1 table.  For each run, counts distinct works (union of
    citer and cited sides of the edge list) and reports:
      src v>1        — works whose source has v > 1
      any inst v>1   — works with at least one ranked institution having v > 1
      all inst v>1   — works where every ranked institution has v > 1
    """
    rk_tables = {row[0] for row in db_rk.execute('SHOW TABLES').fetchall()}
    el_tables = {row[0] for row in db_el.execute('SHOW TABLES').fetchall()}

    print(f'\n{"Label":<12}  {"Period":<10}  {"n_works":>8}  '
          f'{"src>1":>7}  {"src%":>6}  '
          f'{"≥1 inst>1":>10}  {"≥1%":>6}  '
          f'{"all inst>1":>11}  {"all%":>6}')
    print('-' * 90)

    for r in runs:
        rk_tname = _table_name(r)
        tau_sfx  = '_fixtau' if r.get('ref_units', '') else '_vartau'
        el_tname = (f"el_{r['run_code']}_{r['fx']}"
                    f"_tauU{r['tau_u']}_tauS{r['tau_s']}{tau_sfx}")

        if rk_tname not in rk_tables or el_tname not in el_tables:
            continue

        df_rk = db_rk.execute(
            f"SELECT unit_idx, unit_type, v FROM {rk_tname}"
        ).df()
        df_src_v  = (df_rk[df_rk['unit_type'] == 'S'][['unit_idx', 'v']]
                     .rename(columns={'unit_idx': 'src_idx',  'v': 'src_v'}))
        df_inst_v = (df_rk[df_rk['unit_type'] == 'U'][['unit_idx', 'v']]
                     .rename(columns={'unit_idx': 'inst_idx', 'v': 'inst_v'}))

        db_el.register('_src_v_tmp',  df_src_v)
        db_el.register('_inst_v_tmp', df_inst_v)

        row = db_el.execute(f"""
            WITH
            all_edges AS (
                SELECT citer_work_idx AS work_idx,
                       citer_source_idx AS source_idx,
                       citer_inst_idx   AS inst_idx
                FROM {el_tname}
                UNION
                SELECT cited_work_idx,
                       cited_source_idx,
                       cited_inst_idx
                FROM {el_tname}
            ),
            work_src_v AS (
                SELECT DISTINCT ae.work_idx, s.src_v
                FROM all_edges ae
                JOIN _src_v_tmp s ON s.src_idx = ae.source_idx
            ),
            work_inst_agg AS (
                SELECT ae.work_idx,
                       MAX(iv.inst_v) AS max_inst_v,
                       MIN(iv.inst_v) AS min_inst_v
                FROM all_edges ae
                JOIN _inst_v_tmp iv ON iv.inst_idx = ae.inst_idx
                GROUP BY ae.work_idx
            )
            SELECT
                (SELECT COUNT(DISTINCT work_idx) FROM all_edges)                    AS n_works,
                (SELECT COUNT(DISTINCT work_idx) FROM work_src_v WHERE src_v > 1)   AS src_gt1,
                COUNT(DISTINCT work_idx) FILTER(WHERE max_inst_v > 1)               AS any_inst_gt1,
                COUNT(DISTINCT work_idx) FILTER(WHERE min_inst_v > 1)               AS all_inst_gt1
            FROM work_inst_agg
        """).fetchone()

        db_el.unregister('_src_v_tmp')
        db_el.unregister('_inst_v_tmp')

        n, src_gt1, any_gt1, all_gt1 = row
        def pct(x): return 100.0 * x / n if n else float('nan')
        period = f"{r['tc0']}–{str(r['tc1'])[-2:]}"
        print(f"{r['label']:<12}  {period:<10}  {n:>8,}  "
              f"{src_gt1:>7,}  {pct(src_gt1):>5.1f}%  "
              f"{any_gt1:>10,}  {pct(any_gt1):>5.1f}%  "
              f"{all_gt1:>11,}  {pct(all_gt1):>5.1f}%")

    print()
    print('n_works = distinct works appearing in the edge list (citer ∪ cited).')
    print('src>1   = source has v>1; ≥1/all inst>1 = ≥1 or all ranked institutions have v>1.')


# ─── Main ─────────────────────────────────────────────────────────────────────

def plot6(src_rank_map, inst_rank_map, df_s_base, df_i_base, pairs) -> None:
    paths = load_config()
    sns.set_theme(style='whitegrid', font_scale=0.95)

    n_base_s = len(src_rank_map)
    n_base_i = len(inst_rank_map)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.subplots_adjust(wspace=0.10)

    _draw_scatter_panel(axes[0], df_s_base, df_i_base, pairs,
                        0, n_base_s, 'Sources', mode='fix')
    _draw_scatter_panel(axes[1], df_s_base, df_i_base, pairs,
                        1, n_base_i, 'Institutions', mode='fix')

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

    el_path = paths.working / 'edge_lists.duckdb'

    with duckdb.connect(str(rk_path), read_only=True) as db_rk, \
         duckdb.connect(str(el_path), read_only=True) as db_el:
        src_rank_map, inst_rank_map, df_s_base, df_i_base, pairs = fetch_data(db_rk)
        all_runs = [_baseline] + _tau_runs + _fix_runs
        print_v_gt1_table(db_rk, db_el, all_runs)
        print_network_stats(db_el, [_baseline] + _tau_runs)

    n_base_s = len(src_rank_map)
    n_base_i = len(inst_rank_map)

    print_summary(pairs, n_base_s, n_base_i)
    plot6(src_rank_map, inst_rank_map, df_s_base, df_i_base, pairs)


if __name__ == '__main__':
    main()
    print('FINISHED!')
