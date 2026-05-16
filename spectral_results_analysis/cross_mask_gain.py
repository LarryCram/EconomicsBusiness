"""
cross_mask_gain.py — Cross-mask influence comparison, all units.

Compares the bipartite (m=0110) influence scores v^B against the
unipartite within-layer scores v^S (m=1000, sources only) and
v^I (m=0001, institutions only).

Key outputs
-----------
  Scalar:
    Spearman ρ(v^B, v^S) and ρ(v^B, v^I)
    OLS slope of log v^B on log v^S / v^I  (slope > 1 → bipartite steepens)
    N_core: units whose removal stabilises the slope (residualise-and-re-scatter)

  Plots:
    plots/cross_mask_gain.pdf / cross_mask_gain_latex.pdf
      2×1 scatter: log v^B vs log v^S (sources, left) and
                   log v^B vs log v^I (institutions, right)
      Colour = log v^B.  OLS regression line, Spearman ρ and slope annotated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import spearmanr
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

# ── Table names ───────────────────────────────────────────────────────────────

_baseline = next(r for r in load_runs() if r['label'] == 'baseline')
_run_code = _baseline['run_code']
_tau_u    = _baseline['tau_u']
_tau_s    = _baseline['tau_s']
_fx       = _baseline['fx']

_SFXB  = f'{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0'
B_TABLE  = f'rk_{_SFXB}_m0110_chi50_alpha100'   # bipartite
SS_TABLE = f'rk_{_SFXB}_m1000_chi50_alpha100'   # source-only
II_TABLE = f'rk_{_SFXB}_m0001_chi50_alpha100'   # institution-only

_V_LO, _V_HI   = 0.005, 20
_V_TICKS        = [0.01, 0.1, 1, 10]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(db, table: str, unit_type: str) -> pd.DataFrame:
    tables = [r[0] for r in db.execute('SHOW TABLES').fetchall()]
    if table not in tables:
        print(f'  WARNING: {table} not found')
        return pd.DataFrame(columns=['unit_idx', 'v'])
    return db.execute(f"""
        SELECT unit_idx, v FROM {table}
        WHERE unit_type = '{unit_type}'
    """).fetchdf()


def _merge_bip_alt(df_bip: pd.DataFrame,
                   df_alt: pd.DataFrame) -> pd.DataFrame:
    df = (df_bip[['unit_idx', 'v']].rename(columns={'v': 'v_bip'})
          .merge(df_alt[['unit_idx', 'v']].rename(columns={'v': 'v_alt'}),
                 on='unit_idx', how='inner'))
    return df[(df['v_bip'] > 0) & (df['v_alt'] > 0)].copy()


def _ols_loglog(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS slope and intercept of log y on log x."""
    lx = np.log10(x)
    ly = np.log10(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    return float(slope), float(intercept)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic)


def _n_core(df: pd.DataFrame, n_steps: int = 30) -> int:
    """
    Residualise-and-re-scatter: iteratively remove the top-k units by v_bip
    and recompute the log-log OLS slope.  N_core is the smallest k at which
    removing one more unit changes the slope by < 0.01 (stabilisation).
    """
    df_sorted = df.sort_values('v_bip', ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    step = max(1, n // n_steps)
    prev_slope = None
    for k in range(0, n - 10, step):
        sub = df_sorted.iloc[k:]
        if len(sub) < 10:
            break
        slope, _ = _ols_loglog(sub['v_bip'].values, sub['v_alt'].values)
        if prev_slope is not None and abs(slope - prev_slope) < 0.01:
            return k
        prev_slope = slope
    return 0


# ── Plot ──────────────────────────────────────────────────────────────────────

def _pub_theme():
    sns.set_theme(style='whitegrid', font_scale=1.05)
    plt.rcParams.update({
        'axes.linewidth':    1.2,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.major.size':  5,   'ytick.major.size':  5,
    })


def _draw_panel(ax, df: pd.DataFrame,
                xlabel: str, ylabel: str, title: str) -> dict:
    """Scatter log v^B (x) vs log v^alt (y).  Returns scalar diagnostics."""
    x = df['v_bip'].values
    y = df['v_alt'].values

    rho          = _spearman(x, y)
    slope, icept = _ols_loglog(x, y)
    n_core       = _n_core(df)

    # Scatter coloured by log v_bip
    from matplotlib.colors import LogNorm
    sc = ax.scatter(x, y,
                    c=x, cmap='viridis',
                    norm=LogNorm(vmin=max(x.min(), 1e-4), vmax=x.max()),
                    s=10, alpha=0.55, linewidths=0, zorder=2, rasterized=True)

    # Identity line
    ax.plot([_V_LO, _V_HI], [_V_LO, _V_HI],
            color='#888888', lw=1.0, ls='--', zorder=1)

    # OLS regression line
    x_line = np.logspace(np.log10(_V_LO), np.log10(_V_HI), 200)
    y_line = np.exp(icept) * x_line ** slope
    ax.plot(x_line, y_line, color='#d62728', lw=1.3, zorder=3)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(_V_LO, _V_HI); ax.set_ylim(_V_LO, _V_HI)
    ax.set_xticks(_V_TICKS); ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_yticks(_V_TICKS); ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_title(title, fontsize=11, pad=6)

    ax.text(0.04, 0.96,
            f'n={len(df):,}\n'
            f'ρ={rho:.3f}\n'
            f'slope={slope:.3f}\n'
            f'$N_{{core}}$={n_core}',
            transform=ax.transAxes, fontsize=9,
            ha='left', va='top', color='#333333',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label(
        r'$v^B$  (bipartite)', fontsize=9)

    return {'rho': rho, 'slope': slope, 'n': len(df), 'n_core': n_core}


def plot_gain(df_src: pd.DataFrame, df_inst: pd.DataFrame, paths) -> None:
    _pub_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.subplots_adjust(wspace=0.40)

    stats_s = _draw_panel(
        axes[0], df_src,
        xlabel=r'$v^B$  bipartite  (m=0110)',
        ylabel=r'$v^S$  source-only  (m=1000)',
        title='Sources')

    stats_i = _draw_panel(
        axes[1], df_inst,
        xlabel=r'$v^B$  bipartite  (m=0110)',
        ylabel=r'$v^I$  institution-only  (m=0001)',
        title='Institutions')

    sup = fig.suptitle(
        r'Cross-mask influence: bipartite vs within-layer  '
        r'(slope $> 1$ $\Rightarrow$ bipartite steepens rankings)',
        fontsize=9, y=1.01)

    was_interactive = plt.isinteractive()
    plt.ioff()
    out = paths.plots / 'cross_mask_gain.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')
    sup.set_visible(False)
    fig.savefig(paths.plots / 'cross_mask_gain_latex.pdf', bbox_inches='tight')
    print(f'Saved {paths.plots / "cross_mask_gain_latex.pdf"}')
    sup.set_visible(True)
    plt.close(fig)
    if was_interactive:
        plt.ion()

    return stats_s, stats_i


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    paths  = load_config()
    rk_path = paths.working / 'rankings.duckdb'

    with duckdb.connect(str(rk_path), read_only=True) as db:
        df_b_s  = _load(db, B_TABLE,  'S')
        df_b_i  = _load(db, B_TABLE,  'U')
        df_ss_s = _load(db, SS_TABLE, 'S')
        df_ii_i = _load(db, II_TABLE, 'U')

    df_src  = _merge_bip_alt(df_b_s,  df_ss_s)
    df_inst = _merge_bip_alt(df_b_i,  df_ii_i)

    print(f'Sources matched:      {len(df_src):,}  '
          f'(bipartite {len(df_b_s):,}, SS {len(df_ss_s):,})')
    print(f'Institutions matched: {len(df_inst):,}  '
          f'(bipartite {len(df_b_i):,}, II {len(df_ii_i):,})')

    # ── Scalar diagnostics ────────────────────────────────────────────────────
    rho_s  = _spearman(df_src['v_bip'].values,  df_src['v_alt'].values)
    rho_i  = _spearman(df_inst['v_bip'].values, df_inst['v_alt'].values)
    slp_s, _ = _ols_loglog(df_src['v_bip'].values,  df_src['v_alt'].values)
    slp_i, _ = _ols_loglog(df_inst['v_bip'].values, df_inst['v_alt'].values)
    nc_s   = _n_core(df_src)
    nc_i   = _n_core(df_inst)

    print()
    print('── Cross-mask influence gain ─────────────────────────────────────')
    print(f'  Sources:      Spearman ρ(v^B, v^S) = {rho_s:.4f}   '
          f'log-log slope = {slp_s:.4f}   N_core = {nc_s}')
    print(f'  Institutions: Spearman ρ(v^B, v^I) = {rho_i:.4f}   '
          f'log-log slope = {slp_i:.4f}   N_core = {nc_i}')
    print(f'  slope > 1 → bipartite steepens;  slope < 1 → bipartite flattens')

    plot_gain(df_src, df_inst, paths)


if __name__ == '__main__':
    main()
    print('FINISHED!')
