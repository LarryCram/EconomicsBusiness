"""
fig_log_ratio_stability.py — Log-ratio stability across 25-year time series.

For each of five time windows (t1=2000-04, t2=2005-09, t3=2010-14,
t4=2015-19, baseline=2020-24) computes log2(v_1000/v_0110) per source,
then stratifies by Leiden community from the baseline A_SS network
(source_communities_A_SS.csv).

Hypothesis: if subdisciplinary clustering drives the scatter in Fig-3, the
same journals should show consistently elevated log ratios in every window —
this tests stability over 25 years of data as non-circular evidence.

Outputs:
  plots/fig_log_ratio_stability.pdf       — exploration (with title)
  plots/fig_log_ratio_stability_latex.pdf — paper (without title)
  prints median and IQR per community per window to stdout
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

# ─── Configuration ────────────────────────────────────────────────────────────

COMMUNITY_CSV = Path(__file__).parent.parent / 'plots' / 'source_communities_A_SS.csv'

# Community labels (community id → descriptive name)
COMM_LABELS = {
    0: 'Mgt/Mktg',
    1: 'Economics',
    2: 'Finance',
    3: 'Ops/Trans',
    4: 'Sociology',
    5: 'Tourism',
    6: 'Psychology',
    7: 'Proj.Mgt',
}

# Only plot these communities (omit tiny ones)
PLOT_COMMS = [1, 2, 0, 4]   # Economics, Finance, Mgt, Sociology

# Time windows: (window_label, run_code, label_0110, label_1000)
WINDOWS = [
    ('2000–04', '00040004', 't1',       't1-SS'),
    ('2005–09', '05090509', 't2',       't2-SS'),
    ('2010–14', '10141014', 't3',       't3-SS'),
    ('2015–19', '15191519', 't4',       't4-SS'),
    ('2020–24', '20242024', 'baseline', 'SS-only'),
]


# ─── Data loading ─────────────────────────────────────────────────────────────

def table_name(run_code, fx, tau_u, tau_s, rho, m, chi, alpha, ref_units=''):
    chi_str   = 'STAR' if chi == -1.0 else str(round(chi * 100))
    alpha_int = round(alpha * 100)
    tau_sfx   = '_fixtau' if ref_units else '_vartau'
    return (f'rk_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}{tau_sfx}_rho{rho}'
            f'_m{m}_chi{chi_str}_alpha{alpha_int}')


def load_window(db, run_code: str, label_0110: str, label_1000: str,
                runs_by_label: dict) -> pd.DataFrame | None:
    """
    Load v_0110 and v_1000 for a given time window.
    Returns DataFrame(unit_idx, v_0110, v_1000, log_ratio) for sources
    present in both rankings, or None if tables not found.
    """
    r0 = runs_by_label.get(label_0110)
    r1 = runs_by_label.get(label_1000)
    if r0 is None or r1 is None:
        print(f'  WARNING: missing run labels {label_0110!r} or {label_1000!r}')
        return None

    t0 = table_name(r0['run_code'], r0['fx'], r0['tau_u'], r0['tau_s'],
                    r0['rho'], r0['m'], r0['chi'], r0['alpha'])
    t1 = table_name(r1['run_code'], r1['fx'], r1['tau_u'], r1['tau_s'],
                    r1['rho'], r1['m'], r1['chi'], r1['alpha'])

    tables = {row[0] for row in db.execute('SHOW TABLES').fetchall()}
    if t0 not in tables:
        print(f'  WARNING: table {t0} not found')
        return None
    if t1 not in tables:
        print(f'  WARNING: table {t1} not found')
        return None

    df0 = db.execute(
        f"SELECT unit_idx, v FROM {t0} WHERE unit_type='S'"
    ).df().rename(columns={'v': 'v_0110'})
    df1 = db.execute(
        f"SELECT unit_idx, v FROM {t1} WHERE unit_type='S'"
    ).df().rename(columns={'v': 'v_1000'})

    df = df0.merge(df1, on='unit_idx', how='inner')
    df = df[(df['v_0110'] > 0) & (df['v_1000'] > 0)].copy()
    df['log_ratio'] = np.log10(df['v_1000'] / df['v_0110'])
    return df


# ─── Analysis ─────────────────────────────────────────────────────────────────

def build_panel(rk_path: Path) -> pd.DataFrame:
    """
    Returns long-format DataFrame with columns:
      window, unit_idx, v_0110, v_1000, log_ratio, community
    """
    comm = pd.read_csv(COMMUNITY_CSV, usecols=['unit_idx', 'community'])
    comm = comm.drop_duplicates('unit_idx')

    runs = load_runs()
    runs_by_label = {r['label']: r for r in runs}

    all_rows = []
    with duckdb.connect(str(rk_path), read_only=True) as db:
        for win_label, run_code, lbl0, lbl1 in WINDOWS:
            print(f'  Loading {win_label} ({lbl0} / {lbl1}) ...', end=' ')
            df = load_window(db, run_code, lbl0, lbl1, runs_by_label)
            if df is None:
                continue
            df['window'] = win_label
            print(f'{len(df):,} sources')
            all_rows.append(df)

    panel = pd.concat(all_rows, ignore_index=True)
    panel = panel.merge(comm, on='unit_idx', how='left')
    panel['community'] = panel['community'].astype('Int64')
    return panel


def print_summary(panel: pd.DataFrame) -> None:
    print(f'\n{"─"*72}')
    print('Median log2(v_1000/v_0110) by community and time window')
    print(f'{"─"*72}')

    windows = [w for w, *_ in WINDOWS]
    header = f'{"Community":<15}' + ''.join(f'{w:>10}' for w in windows)
    print(header)
    print('─' * len(header))

    for cid in sorted(panel['community'].dropna().unique()):
        sub = panel[panel['community'] == cid]
        name = COMM_LABELS.get(int(cid), f'Comm {cid}')
        row = f'{name:<15}'
        for win, *_ in WINDOWS:
            vals = sub[sub['window'] == win]['log_ratio'].dropna()
            if len(vals) > 0:
                row += f'{np.median(vals):>+10.3f}'
            else:
                row += f'{"n/a":>10}'
        print(row)

    print()
    print('(+) = SS-only ranking inflates v relative to bipartite baseline')
    print('(-) = SS-only ranking deflates v relative to bipartite baseline')


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_stability(panel: pd.DataFrame, paths) -> None:
    windows = [w for w, *_ in WINDOWS]
    n_comms = len(PLOT_COMMS)

    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(
        1, n_comms, figsize=(4.0 * n_comms, 5), sharey=True
    )
    fig.subplots_adjust(wspace=0.08)

    for ax, cid in zip(axes, PLOT_COMMS):
        sub = panel[panel['community'] == cid].copy()
        sub['window'] = pd.Categorical(sub['window'], categories=windows, ordered=True)

        # Box plot per window
        bp_data = [sub[sub['window'] == w]['log_ratio'].dropna().values
                   for w in windows]
        positions = list(range(len(windows)))

        bp = ax.boxplot(
            bp_data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            medianprops=dict(color='black', linewidth=1.5),
            boxprops=dict(facecolor='#aec7e8', alpha=0.8),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker='.', markersize=2, alpha=0.3,
                            markerfacecolor='#555555'),
        )

        ax.axhline(0, color='#d62728', linewidth=0.9, linestyle='--', zorder=0)

        # Median line connecting medians across windows
        medians = [np.median(d) if len(d) > 0 else np.nan for d in bp_data]
        ax.plot(positions, medians, color='#1f77b4', linewidth=1.2,
                marker='o', markersize=4, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(windows, rotation=35, ha='right', fontsize=7.5)
        ax.set_title(COMM_LABELS.get(cid, f'Comm {cid}'), fontsize=9, pad=5)
        n_sources = sub['unit_idx'].nunique()
        ax.text(0.97, 0.97, f'n≈{n_sources}', transform=ax.transAxes,
                fontsize=7, ha='right', va='top', color='#555555')

    axes[0].set_ylabel('$\\log(v_{1000}/v_{0110})$', labelpad=4)

    sup = fig.suptitle(
        '$\\log(v_{1000}/v_{0110})$ stability across 25 years  '
        '(by A$_{SS}$ Leiden community)',
        fontsize=9, y=1.02,
    )

    out = paths.plots / 'fig_log_ratio_stability.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'\nSaved {out}')

    sup.set_visible(False)
    fig.savefig(paths.plots / 'fig_log_ratio_stability_latex.pdf',
                bbox_inches='tight')
    sup.set_visible(True)

    plt.close(fig)


def plot_all_communities(panel: pd.DataFrame, paths) -> None:
    """
    Single panel: mean +/- SE of log ratio per community per window.
    All 8 communities, windows on x, one line per community.
    """
    windows = [w for w, *_ in WINDOWS]
    comm_ids = sorted(panel['community'].dropna().unique())

    palette = sns.color_palette('tab10', n_colors=len(comm_ids))

    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, ax = plt.subplots(figsize=(8, 5))

    for cid, colour in zip(comm_ids, palette):
        sub = panel[panel['community'] == cid]
        medians = []
        q25s, q75s = [], []
        for win in windows:
            vals = sub[sub['window'] == win]['log_ratio'].dropna().values
            if len(vals) >= 5:
                medians.append(np.median(vals))
                q25s.append(np.percentile(vals, 25))
                q75s.append(np.percentile(vals, 75))
            else:
                medians.append(np.nan)
                q25s.append(np.nan)
                q75s.append(np.nan)

        xs = list(range(len(windows)))
        name = COMM_LABELS.get(int(cid), f'Comm {cid}')
        ax.plot(xs, medians, color=colour, linewidth=1.4,
                marker='o', markersize=5, label=name, zorder=3)
        ax.fill_between(xs, q25s, q75s, color=colour, alpha=0.15, zorder=1)

    ax.axhline(0, color='#999999', linewidth=0.9, linestyle='--', zorder=0)
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows, fontsize=8)
    ax.set_xlabel('Time window', labelpad=4)
    ax.set_ylabel('Median $\\log(v_{1000}/v_{0110})$  (IQR shaded)', labelpad=4)
    ax.legend(fontsize=7.5, framealpha=0.85, loc='lower left', ncol=2)

    sup = fig.suptitle(
        'Stability of SS-vs-bipartite log-ratio across 25 years  '
        '(median ± IQR by community)',
        fontsize=9, y=1.01,
    )

    out = paths.plots / 'fig_log_ratio_stability_all.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

    sup.set_visible(False)
    fig.savefig(paths.plots / 'fig_log_ratio_stability_all_latex.pdf',
                bbox_inches='tight')
    sup.set_visible(True)

    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    paths   = load_config()
    rk_path = paths.working / 'rankings.duckdb'

    if not rk_path.exists():
        raise FileNotFoundError(f'rankings.duckdb not found at {rk_path}')
    if not COMMUNITY_CSV.exists():
        raise FileNotFoundError(
            f'{COMMUNITY_CSV} not found — run source_communities.py first'
        )

    print('=== Loading rankings for all windows ===')
    panel = build_panel(rk_path)

    n_total = len(panel)
    n_with_comm = panel['community'].notna().sum()
    print(f'\n  Total source×window rows: {n_total:,}')
    print(f'  Rows with community label: {n_with_comm:,} '
          f'({100*n_with_comm/n_total:.1f}%)')

    print_summary(panel)

    print('\n=== Plotting ===')
    plot_stability(panel, paths)
    plot_all_communities(panel, paths)


if __name__ == '__main__':
    main()
    print('FINISHED!')
