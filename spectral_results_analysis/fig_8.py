"""
fig_8.py — JCR indicators vs spectral ranking v.

Plots WoS JCR 2024 indicators against influence-per-work v from the
baseline ranking (m=0110, all sources, τ_U=τ_S=20, ρ=0, α=1).

Join path:
  ranking  (unit_idx = source_idx)
    ← source_master (source_idx, wos_journal_name)
    → wos_jcr_eb.csv (Journal name → JIF / 5yr JIF / JCI / AIS)
  Match: wos_journal_name vs Journal name, both uppercased.
  wos_jcr_eb has multi-category duplicates; deduplicate on name (indicators
  are journal-level and identical across category rows).
  The name → source_idx mapping via source_master is unique.

Only sources present in both the baseline ranking and the JCR file are plotted.

x-axis: rank by v (descending), restricted to JCR-matched sources
y-axis: v (black line) + four JCR indicators as scatter — all log scale

Outputs:
  plots/fig_8.pdf        — with title (exploration)
  plots/fig_8_latex.pdf  — without title (paper)
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

_baseline  = next(r for r in load_runs() if r['label'] == 'baseline')
_run_code  = _baseline['run_code']
_tau_u     = _baseline['tau_u']
_tau_s     = _baseline['tau_s']

BASELINE_TABLE = f'rk_{_run_code}_A_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m0110_chi50_alpha100'

JCR_PATH = Path(__file__).parent.parent / 'data' / 'wos_jcr_eb.csv'

# JCR indicators: (csv column, display label, colour, marker)
INDICATORS = [
    ('2024 JIF',                'JIF 2024',  '#d62728', 'o'),
    ('5 Year JIF',              'JIF 5yr',   '#ff7f0e', 's'),
    ('2024 JCI',                'JCI 2024',  '#2ca02c', '^'),
    ('Article Influence Score', 'Art. Inf.', '#9467bd', 'x'),
]
IND_COLS = [col for col, _, _, _ in INDICATORS]


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_jcr_with_source_idx(paths) -> pd.DataFrame:
    """
    Load wos_jcr_eb.csv, deduplicate on Journal name, join to source_master
    on wos_journal_name (both uppercased).
    Returns DataFrame with columns [source_idx] + IND_COLS.
    """
    jcr = pd.read_csv(JCR_PATH, dtype=str)

    # Normalise journal name for matching
    jcr['name_key'] = jcr['Journal name'].str.upper().str.strip()

    # Deduplicate: indicators are journal-level, identical across category rows
    jcr = jcr.drop_duplicates('name_key')

    # Parse numeric indicators
    for col in IND_COLS:
        jcr[col] = pd.to_numeric(
            jcr[col].str.replace(',', '', regex=False), errors='coerce'
        )

    # Load source_master
    sm_path = paths.parquet / 'source_master.parquet'
    if sm_path.exists():
        sm = pd.read_parquet(sm_path, columns=['source_idx', 'wos_journal_name'])
    else:
        sm = pd.read_csv(
            Path(__file__).parent.parent / 'data' / 'source_master.csv',
            usecols=['source_idx', 'wos_journal_name'],
        )

    sm = sm.dropna(subset=['wos_journal_name']).copy()
    sm['name_key'] = sm['wos_journal_name'].str.upper().str.strip()

    # Join: name_key → source_idx  (mapping is unique per source_master)
    merged = jcr.merge(sm[['source_idx', 'name_key']], on='name_key', how='inner')

    print(f'  JCR rows (after name dedup):   {len(jcr):,}')
    print(f'  source_master with wos name:   {len(sm):,}')
    print(f'  Matched JCR → source_idx:      {len(merged):,}')

    return merged[['source_idx'] + IND_COLS]


def fetch_data(db, paths) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      unit_idx, v, plot_rank, + IND_COLS
    Restricted to sources present in both ranking and JCR.
    """
    tables = {row[0] for row in db.execute('SHOW TABLES').fetchall()}
    if BASELINE_TABLE not in tables:
        raise RuntimeError(f'Baseline table {BASELINE_TABLE} not found.')

    # Baseline ranking — sources only
    df_rk = db.execute(
        f"SELECT unit_idx, v FROM {BASELINE_TABLE} WHERE unit_type='S'"
    ).df()

    # JCR joined to source_idx
    jcr = load_jcr_with_source_idx(paths)

    # Inner join: only sources present in both
    df = df_rk.merge(
        jcr.rename(columns={'source_idx': 'unit_idx'}),
        on='unit_idx', how='inner',
    )

    df = df.sort_values('v', ascending=False).reset_index(drop=True)
    df['plot_rank'] = np.arange(1, len(df) + 1)

    print(f'  Sources in baseline:           {len(df_rk):,}')
    print(f'  Sources matched to JCR:        {len(df):,}')
    for col in IND_COLS:
        n = df[col].notna().sum()
        print(f'    {col:<30} {n:,} non-null')

    return df


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot8(df: pd.DataFrame, paths) -> None:
    n = len(df)

    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, ax = plt.subplots(figsize=(9, 5))

    # Baseline v — black line
    ax.plot(
        df['plot_rank'].values,
        df['v'].values,
        color='black', linewidth=1.4, alpha=1.0, zorder=4,
        label=f'$v$ (baseline m=0110, n={n:,})',
    )

    # JCR indicators — scatter
    for col, label, colour, marker in INDICATORS:
        sub = df.dropna(subset=[col])
        is_cross = marker in ('x', '+')
        ax.scatter(
            sub['plot_rank'].values,
            sub[col].values,
            color=colour,
            marker=marker,
            s=45 if is_cross else 20,
            alpha=0.55,
            linewidths=0.8 if is_cross else 0.4,
            zorder=2,
            label=f'{label}  (n={len(sub):,})',
        )

    ax.set_yscale('log')
    ax.axhline(1.0, color='#999999', linewidth=0.8, linestyle='--', zorder=0)
    ax.text(n * 0.98, 1.0, '$v=1$',
            ha='right', va='bottom', fontsize=7.5, color='#999999')
    ax.set_xlim(1, n)
    ax.set_xlabel('Rank by $v$  (baseline m=0110, JCR-matched sources)',
                  labelpad=4)
    ax.set_ylabel('Value (log scale)', labelpad=4)
    ax.legend(fontsize=8, framealpha=0.85, loc='upper right')

    sup = fig.suptitle(
        'JCR indicators vs spectral ranking  '
        '(x-axis = rank by $v$, JCR-matched sources only)',
        fontsize=9, y=1.01,
    )

    out = paths.plots / 'fig_8.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

    sup.set_visible(False)
    latex_out = paths.plots / 'fig_8_latex.pdf'
    fig.savefig(latex_out, bbox_inches='tight')
    print(f'Saved {latex_out}')
    sup.set_visible(True)

    plt.close(fig)


def plot8a(df: pd.DataFrame, paths) -> None:
    """
    Scatter of each JCR indicator vs v, with OLS trend lines.
    x-axis: v, 0–50.  y-axis: indicator value (linear scale).
    Trend lines fitted on all matched data for each indicator.
    """
    x_lo, x_hi = 0.01, 20.0
    y_hi = 30.0
    # Evenly spaced on log scale for drawing trend lines
    x_line = np.logspace(np.log10(x_lo), np.log10(x_hi), 300)

    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, ax = plt.subplots(figsize=(9, 5))

    for col, label, colour, marker in INDICATORS:
        sub = df.dropna(subset=[col])
        is_cross = marker in ('x', '+')

        # Scatter within plotted window
        vis = sub[(sub['v'] >= x_lo) & (sub['v'] <= x_hi)]
        ax.scatter(
            vis['v'].values,
            vis[col].values,
            color=colour,
            marker=marker,
            s=45 if is_cross else 20,
            alpha=0.45,
            linewidths=0.8 if is_cross else 0.4,
            zorder=2,
        )

        # OLS in log-log space: log(y) = m * log(v) + b  (straight line on log-log axes)
        valid = sub[sub[col] > 0]
        lx = np.log(valid['v'].values)
        ly = np.log(valid[col].values)
        m, b = np.polyfit(lx, ly, 1)
        ly_hat = m * lx + b
        r2 = 1.0 - np.sum((ly - ly_hat) ** 2) / np.sum((ly - ly.mean()) ** 2)
        ax.plot(
            x_line,
            np.exp(b) * x_line ** m,
            color=colour,
            linewidth=1.6,
            zorder=3,
            label=f'{label}  $v^{{{m:.2f}}}$  $R^2={r2:.2f}$  (n={len(valid):,})',
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.001, y_hi)
    ax.set_xlabel('Influence per work $v$  (baseline m=0110, log scale)', labelpad=4)
    ax.set_ylabel('JCR indicator value (log scale)', labelpad=4)
    ax.legend(fontsize=8, framealpha=0.85, loc='upper left')

    sup = fig.suptitle(
        'JCR indicators vs $v$ with OLS trend lines  '
        '(JCR-matched sources only)',
        fontsize=9, y=1.01,
    )

    out = paths.plots / 'fig_8a.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

    sup.set_visible(False)
    latex_out = paths.plots / 'fig_8a_latex.pdf'
    fig.savefig(latex_out, bbox_inches='tight')
    print(f'Saved {latex_out}')
    sup.set_visible(True)

    plt.close(fig)


def plot8b(df: pd.DataFrame, paths) -> None:
    """
    x-axis: Article Influence Score (log).
    y-axis: v plus the three remaining JCR indicators (log).
    Power-law OLS trend line for each y series.
    """
    AIS_COL = 'Article Influence Score'
    Y_SERIES = [
        ('v',        '$v$',      'black',   'o'),
        ('2024 JIF', 'JIF 2024', '#d62728', 's'),
        ('5 Year JIF','JIF 5yr', '#ff7f0e', '^'),
        ('2024 JCI', 'JCI 2024', '#2ca02c', 'x'),
    ]

    x_lo, x_hi = 0.001, 30.0
    y_lo, y_hi = 0.001, 30.0
    x_line = np.logspace(np.log10(x_lo), np.log10(x_hi), 300)

    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, ax = plt.subplots(figsize=(9, 5))

    for col, label, colour, marker in Y_SERIES:
        sub = df.dropna(subset=[AIS_COL, col])
        sub = sub[(sub[AIS_COL] > 0) & (sub[col] > 0)]
        is_cross = marker in ('x', '+')

        ax.scatter(
            sub[AIS_COL].values,
            sub[col].values,
            color=colour,
            marker=marker,
            s=45 if is_cross else 20,
            alpha=0.45,
            linewidths=0.8 if is_cross else 0.4,
            zorder=2,
        )

        # Power-law OLS: log(y) = m * log(x) + b
        lx = np.log(sub[AIS_COL].values)
        ly = np.log(sub[col].values)
        m, b = np.polyfit(lx, ly, 1)
        ly_hat = m * lx + b
        ss_res = np.sum((ly - ly_hat) ** 2)
        ss_tot = np.sum((ly - ly.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        ax.plot(
            x_line,
            np.exp(b) * x_line ** m,
            color=colour,
            linewidth=1.6,
            zorder=3,
            label=f'{label}  $\\propto$ AIS$^{{{m:.2f}}}$  $R^2={r2:.2f}$  (n={len(sub):,})',
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel('Article Influence Score (log scale)', labelpad=4)
    ax.set_ylabel('Value (log scale)', labelpad=4)
    ax.legend(fontsize=8, framealpha=0.85, loc='upper left')

    sup = fig.suptitle(
        '$v$ and JCR indicators vs Article Influence Score  '
        '(power-law OLS trend lines)',
        fontsize=9, y=1.01,
    )

    out = paths.plots / 'fig_8b.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

    sup.set_visible(False)
    latex_out = paths.plots / 'fig_8b_latex.pdf'
    fig.savefig(latex_out, bbox_inches='tight')
    print(f'Saved {latex_out}')
    sup.set_visible(True)

    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    paths   = load_config()
    rk_path = paths.working / 'rankings.duckdb'

    if not rk_path.exists():
        raise FileNotFoundError(
            f'rankings.duckdb not found at {rk_path}. '
            'Run spectral_ranking/run_rankings.py first.'
        )

    print('Loading data...')
    with duckdb.connect(str(rk_path), read_only=True) as db:
        df = fetch_data(db, paths)

    plot8(df, paths)
    plot8a(df, paths)
    plot8b(df, paths)


if __name__ == '__main__':
    main()
    print('FINISHED!')
