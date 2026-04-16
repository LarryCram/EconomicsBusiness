"""
fig_stability.py — Cross-window stability of spectral rankings.

Uses the five baseline (m=0110) time windows t1-t5 to measure:

  1. Threshold persistence: fraction of v>1 units in 2020-24 that were
     also v>1 in 2000-04 (and vice versa), restricted to units present
     in both windows.

  2. Systematic movers: for units present in all five windows, fit
     log(v) = a + b·t (t = 1..5) per unit. Report top-10 risers and
     fallers by slope b, separately for sources and institutions.

  3. Plots:
       fig_stability_sources.pdf  — scatter v_t1 vs v_t5 for common sources,
                                    coloured by v>1 status, with movers labelled
       fig_stability_insts.pdf    — same for institutions
       fig_stability_movers.pdf   — slope b ranked, top/bottom 10 labelled

Outputs all figures to plots/ and prints summary tables to stdout.
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

WINDOWS = [
    ('2000–04', 't1',       1),
    ('2005–09', 't2',       2),
    ('2010–14', 't3',       3),
    ('2015–19', 't4',       4),
    ('2020–24', 'baseline', 5),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def tname(r: dict) -> str:
    chi_str  = 'STAR' if r['chi'] == -1.0 else str(round(r['chi'] * 100))
    tau_sfx  = '_fixtau' if r.get('ref_units') else '_vartau'
    return (f"rk_{r['run_code']}_{r['fx']}_tauU{r['tau_u']}_tauS{r['tau_s']}{tau_sfx}"
            f"_rho{r['rho']}_m{r['m']}_chi{chi_str}_alpha{round(r['alpha']*100)}")


def load_all_windows(db, runs_by_label: dict) -> pd.DataFrame:
    """
    Returns long-format DataFrame:
      unit_idx, unit_type, v, log_v, window, t
    for all five windows.
    """
    frames = []
    for win_label, label, t in WINDOWS:
        r = runs_by_label[label]
        tn = tname(r)
        df = db.execute(
            f"SELECT unit_idx, unit_type, v FROM {tn} WHERE v > 0"
        ).df()
        df['window'] = win_label
        df['t'] = t
        df['log_v'] = np.log(df['v'])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_names(paths) -> dict:
    """Returns {unit_idx: name} for sources and institutions."""
    sm_path = paths.parquet / 'source_master.parquet'
    sm = pd.read_parquet(sm_path, columns=['source_idx', 'source_name'])
    names = {int(row['source_idx']): row['source_name'] for _, row in sm.iterrows()}

    econ_path = paths.working / 'econ.duckdb'
    with duckdb.connect(str(econ_path), read_only=True) as db:
        im = db.execute(
            "SELECT DISTINCT CAST(regexp_extract(institution_id, 'I([0-9]+)$', 1) AS BIGINT) AS institution_idx, "
            'institution_name FROM authorships WHERE institution_id IS NOT NULL'
        ).df()
    im = im.dropna(subset=['institution_idx'])
    for _, row in im.iterrows():
        names[int(row['institution_idx'])] = row['institution_name']
    return names


# ─── Persistence ──────────────────────────────────────────────────────────────

def persistence_table(panel: pd.DataFrame) -> None:
    """
    For each unit_type, print the 2×2 persistence table:
    v>1 in t1 × v>1 in t5, restricted to units present in both windows.
    Also prints Spearman ρ between v_t1 and v_t5.
    """
    print(f'\n{"─"*65}')
    print('Threshold persistence  (v > 1,  t1=2000–04  vs  t5=2020–24)')
    print(f'{"─"*65}')

    for utype, label in [('S', 'Sources'), ('U', 'Institutions')]:
        sub = panel[panel['unit_type'] == utype]
        t1 = sub[sub['t'] == 1][['unit_idx', 'v']].rename(columns={'v': 'v_t1'})
        t5 = sub[sub['t'] == 5][['unit_idx', 'v']].rename(columns={'v': 'v_t5'})
        both = t1.merge(t5, on='unit_idx')

        both['hi_t1'] = both['v_t1'] > 1
        both['hi_t5'] = both['v_t5'] > 1

        n = len(both)
        n_hi_t5 = both['hi_t5'].sum()
        n_hi_t1 = both['hi_t1'].sum()

        # Of those v>1 in t5, how many were v>1 in t1?
        sub5 = both[both['hi_t5']]
        pct_persist = 100 * sub5['hi_t1'].mean()

        # Of those v>1 in t1, how many are v>1 in t5?
        sub1 = both[both['hi_t1']]
        pct_survive = 100 * sub1['hi_t5'].mean()

        # Spearman on common set
        rho, pval = stats.spearmanr(both['v_t1'], both['v_t5'])

        print(f'\n  {label}  (n common to both windows: {n:,})')
        print(f'    v>1 in t1:  {n_hi_t1:,}  ({100*n_hi_t1/n:.1f}% of common set)')
        print(f'    v>1 in t5:  {n_hi_t5:,}  ({100*n_hi_t5/n:.1f}% of common set)')
        print(f'    Of v>1 in t5: {pct_persist:.1f}% were also v>1 in t1')
        print(f'    Of v>1 in t1: {pct_survive:.1f}% are still v>1 in t5')
        print(f'    Spearman ρ(v_t1, v_t5) = {rho:.3f}  (p={pval:.1e})')

        # Full 5-window Spearman chain
        print(f'    Spearman ρ by consecutive window:')
        for (_, la, ta), (_, lb, tb) in zip(WINDOWS[:-1], WINDOWS[1:]):
            va = sub[sub['t'] == ta][['unit_idx', 'v']].rename(columns={'v': 'va'})
            vb = sub[sub['t'] == tb][['unit_idx', 'v']].rename(columns={'v': 'vb'})
            pair = va.merge(vb, on='unit_idx')
            r2, _ = stats.spearmanr(pair['va'], pair['vb'])
            print(f'      t{ta}→t{tb}  ({la}→{lb.split("=")[-1]}): '
                  f'ρ={r2:.3f}  n={len(pair):,}')


# ─── Movers ───────────────────────────────────────────────────────────────────

def compute_movers(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For units present in all 5 windows, fit log(v) ~ t by OLS.
    Returns DataFrame with columns:
      unit_idx, unit_type, slope, intercept, r2, n_windows,
      v_t1, v_t5, mean_log_v
    sorted by |slope| descending within each unit_type.
    """
    rows = []
    for uid, grp in panel.groupby('unit_idx'):
        grp = grp.sort_values('t')
        if len(grp) < 5:
            continue
        utype = grp['unit_type'].iloc[0]
        ts = grp['t'].values.astype(float)
        lv = grp['log_v'].values
        slope, intercept, r, *_ = stats.linregress(ts, lv)
        r2 = r ** 2
        # v at t1 and t5
        v_t1 = grp[grp['t'] == 1]['v'].values
        v_t5 = grp[grp['t'] == 5]['v'].values
        rows.append({
            'unit_idx':   uid,
            'unit_type':  utype,
            'slope':      slope,
            'intercept':  intercept,
            'r2':         r2,
            'n_windows':  len(grp),
            'v_t1':       v_t1[0] if len(v_t1) else np.nan,
            'v_t5':       v_t5[0] if len(v_t5) else np.nan,
            'mean_log_v': lv.mean(),
        })
    return pd.DataFrame(rows)


def print_movers(movers: pd.DataFrame, names: dict, top_n: int = 10) -> None:
    print(f'\n{"─"*72}')
    print(f'Systematic movers  (log(v) linear trend over 5 windows)')
    print(f'{"─"*72}')

    for utype, label in [('S', 'Sources'), ('U', 'Institutions')]:
        sub = movers[movers['unit_type'] == utype].copy()
        sub['name'] = sub['unit_idx'].map(names).fillna('?')
        n_total = len(sub)
        print(f'\n  {label}  (n present in all 5 windows: {n_total:,})')

        for direction, heading in [('risers', True), ('fallers', False)]:
            ranked = sub.sort_values('slope', ascending=not heading)
            top = ranked.head(top_n)
            print(f'\n  Top {top_n} {direction}:')
            print(f'  {"name":<45}  {"slope":>7}  {"R²":>5}  {"v_t1":>6}  {"v_t5":>6}')
            print(f'  {"-"*45}  {"-------":>7}  {"-----":>5}  {"------":>6}  {"------":>6}')
            for _, row in top.iterrows():
                print(f'  {str(row["name"])[:45]:<45}  '
                      f'{row["slope"]:>+7.3f}  '
                      f'{row["r2"]:>5.2f}  '
                      f'{row["v_t1"]:>6.2f}  '
                      f'{row["v_t5"]:>6.2f}')


# ─── Plots ────────────────────────────────────────────────────────────────────

def _prepare_scatter_data(panel: pd.DataFrame, movers: pd.DataFrame,
                          names: dict, unit_type: str) -> tuple:
    """Prepare data and label_ids for one scatter panel."""
    sub = panel[panel['unit_type'] == unit_type]
    t1 = sub[sub['t'] == 1][['unit_idx', 'v']].rename(columns={'v': 'v_t1'})
    t5 = sub[sub['t'] == 5][['unit_idx', 'v']].rename(columns={'v': 'v_t5'})
    both = t1.merge(t5, on='unit_idx').copy()
    both['v>1_both']    = (both['v_t1'] > 1) & (both['v_t5'] > 1)
    both['v>1_neither'] = (both['v_t1'] <= 1) & (both['v_t5'] <= 1)
    both['riser']       = (both['v_t1'] <= 1) & (both['v_t5'] > 1)
    both['faller']      = (both['v_t1'] > 1)  & (both['v_t5'] <= 1)
    mv = movers[movers['unit_type'] == unit_type].copy()
    label_ids = (set(mv.nlargest(5, 'slope')['unit_idx']) |
                 set(mv.nsmallest(5, 'slope')['unit_idx']))
    return both, label_ids


def _draw_scatter_panel(ax, both: pd.DataFrame, label_ids: set,
                        names: dict, shared_lims: list,
                        title: str, show_ylabel: bool,
                        show_labels: bool = True) -> None:
    """Draw one v_t1 vs v_t5 scatter panel onto ax."""
    COLOURS = {
        'v>1_both':    ('#1f77b4', 'v>1 in both'),
        'v>1_neither': ('#aec7e8', 'v≤1 in both'),
        'riser':       ('#2ca02c', 'crossed $v=1$ upward'),
        'faller':      ('#d62728', 'crossed $v=1$ downward'),
    }
    for col, (c, lbl) in COLOURS.items():
        mask = both[col]
        ax.scatter(both.loc[mask, 'v_t1'], both.loc[mask, 'v_t5'],
                   c=c, s=12, alpha=0.6, linewidths=0,
                   label=f'{lbl} (n={mask.sum():,})', zorder=2)

    ax.axhline(1, color='#999999', linewidth=0.7, linestyle='--', zorder=0)
    ax.axvline(1, color='#999999', linewidth=0.7, linestyle='--', zorder=0)
    ax.plot(shared_lims, shared_lims, color='#555555',
            linewidth=0.7, linestyle=':', zorder=0)

    if show_labels:
        for _, row in both[both['unit_idx'].isin(label_ids)].iterrows():
            name = names.get(int(row['unit_idx']), '?')
            ax.annotate(name[:28], (row['v_t1'], row['v_t5']),
                        fontsize=5.5, ha='left', va='bottom',
                        xytext=(3, 2), textcoords='offset points',
                        color='#222222')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(shared_lims)
    ax.set_ylim(shared_lims)
    ax.set_xlabel('$v$  (2000–04)', labelpad=4)
    if show_ylabel:
        ax.set_ylabel('$v$  (2020–24)', labelpad=4)
    ax.set_title(title, fontsize=9, pad=5)
    ax.legend(fontsize=7, framealpha=0.85, loc='upper left')


def plot_v_scatter(panel: pd.DataFrame, movers: pd.DataFrame,
                   names: dict, paths) -> None:
    """
    Two-panel facet: Sources (left) and Institutions (right).
    Shared log y-axis.  v_t1 vs v_t5 scatter coloured by v>1 crossing status.
    """
    # Compute data for both panels first so we can set shared axis limits
    both_s, lids_s = _prepare_scatter_data(panel, movers, names, 'S')
    both_u, lids_u = _prepare_scatter_data(panel, movers, names, 'U')

    lo = 0.02
    hi = max(both_s['v_t1'].max(), both_s['v_t5'].max(),
             both_u['v_t1'].max(), both_u['v_t5'].max()) * 1.1
    shared_lims = [lo, hi]

    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.subplots_adjust(wspace=0.06)

    _draw_scatter_panel(axes[0], both_s, lids_s, names, shared_lims,
                        'Sources', show_ylabel=True, show_labels=False)
    _draw_scatter_panel(axes[1], both_u, lids_u, names, shared_lims,
                        'Institutions', show_ylabel=False, show_labels=False)

    sup = fig.suptitle(
        'Prestige per work $v$: 2000–04 vs 2020–24  '
        '(common units, log scale)',
        fontsize=9, y=1.02,
    )

    out = paths.plots / 'fig_stability_scatter.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

    sup.set_visible(False)
    fig.savefig(paths.plots / 'fig_stability_scatter_latex.pdf',
                bbox_inches='tight')
    sup.set_visible(True)
    plt.close(fig)


def plot_movers(movers: pd.DataFrame, names: dict, paths,
                top_n: int = 15) -> None:
    """Slope ranked bar chart for top/bottom movers, sources and institutions."""
    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.38)

    for ax, utype, ulab in [(axes[0], 'S', 'Sources'),
                             (axes[1], 'U', 'Institutions')]:
        sub = movers[movers['unit_type'] == utype].copy()
        sub['name'] = sub['unit_idx'].map(names).fillna('?')
        sub['name'] = sub['name'].str[:40]

        top = pd.concat([
            sub.nlargest(top_n, 'slope'),
            sub.nsmallest(top_n, 'slope'),
        ]).drop_duplicates('unit_idx').sort_values('slope')

        colours = ['#d62728' if s < 0 else '#1f77b4' for s in top['slope']]
        ax.barh(range(len(top)), top['slope'].values, color=colours, height=0.7)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top['name'].values, fontsize=6.5)
        ax.axvline(0, color='black', linewidth=0.7)
        ax.set_xlabel('Trend slope $b$  (Δlog $v$ per window)', labelpad=4)
        ax.set_title(f'{ulab}  (n in all 5 windows: {len(sub):,})',
                     fontsize=9, pad=5)

    sup = fig.suptitle(
        'Systematic movers: log($v$) trend slope over 25 years  '
        '(top/bottom 15 sources and institutions)',
        fontsize=9, y=1.02,
    )

    fig.savefig(paths.plots / 'fig_stability_movers.pdf', bbox_inches='tight')
    print(f'Saved {paths.plots / "fig_stability_movers.pdf"}')

    sup.set_visible(False)
    fig.savefig(paths.plots / 'fig_stability_movers_latex.pdf', bbox_inches='tight')
    sup.set_visible(True)
    plt.close(fig)


# ─── Trajectory analysis ─────────────────────────────────────────────────────

def classify_trajectory(seq: list[bool]) -> str:
    """
    Given a 5-element boolean sequence (True = v>1), return a descriptive class.
      stable_high   — v>1 in ≥4 windows
      stable_low    — v≤1 in ≥4 windows
      monotone_rise — first crossing never reversed (L…LH…H, at least one L and one H)
      monotone_fall — first fall never reversed (H…HL…L, at least one H and one L)
      oscillating   — any other pattern (crosses v=1 more than once)
    """
    n_high = sum(seq)
    if n_high >= 4:
        return 'stable_high'
    if n_high <= 1:
        return 'stable_low'
    # Find transitions
    transitions = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    if transitions == 1:
        if not seq[0]:   # starts low, ends high
            return 'monotone_rise'
        else:            # starts high, ends low
            return 'monotone_fall'
    return 'oscillating'


def trajectory_analysis(panel: pd.DataFrame, names: dict) -> pd.DataFrame:
    """
    For each unit_type, build trajectory table for units present in all 5 windows.
    Returns long DataFrame with one row per unit containing:
      unit_idx, unit_type, traj_str (e.g. 'LLHHH'), traj_class,
      v_t1..v_t5, first_cross (window t where v first exceeded 1, or None)
    """
    rows = []
    ts = [1, 2, 3, 4, 5]
    for uid, grp in panel.groupby('unit_idx'):
        if len(grp) < 5:
            continue
        grp = grp.set_index('t').reindex(ts)
        if grp['v'].isna().any():
            continue
        utype = grp['unit_type'].dropna().iloc[0]
        vs = grp['v'].values
        hi = [v > 1 for v in vs]
        traj_str   = ''.join('H' if h else 'L' for h in hi)
        traj_class = classify_trajectory(hi)
        # First window where v crossed above 1
        first_cross = next((ts[i] for i in range(len(hi)) if hi[i]), None)
        rows.append({
            'unit_idx':    uid,
            'unit_type':   utype,
            'traj_str':    traj_str,
            'traj_class':  traj_class,
            'first_cross': first_cross,
            **{f'v_t{t}': vs[i] for i, t in enumerate(ts)},
        })
    df = pd.DataFrame(rows)
    df['name'] = df['unit_idx'].map(names).fillna('?')
    return df


def print_trajectory_report(traj: pd.DataFrame) -> None:
    print(f'\n{"─"*72}')
    print('Trajectory classification  (units present in all 5 windows)')
    print(f'{"─"*72}')

    ORDER = ['stable_high', 'monotone_rise', 'oscillating',
             'monotone_fall', 'stable_low']
    CLASS_LABEL = {
        'stable_high':   'Stable high   (v>1 in ≥4/5 windows)',
        'monotone_rise': 'Monotone rise (crossed v=1, never fell back)',
        'oscillating':   'Oscillating   (crossed v=1 more than once)',
        'monotone_fall': 'Monotone fall (fell below v=1, never recovered)',
        'stable_low':    'Stable low    (v≤1 in ≥4/5 windows)',
    }

    for utype, ulab in [('S', 'Sources'), ('U', 'Institutions')]:
        sub = traj[traj['unit_type'] == utype]
        n   = len(sub)
        print(f'\n  {ulab}  (n={n:,})')
        for cls in ORDER:
            grp = sub[sub['traj_class'] == cls]
            pct = 100 * len(grp) / n
            print(f'    {CLASS_LABEL[cls]:<50}  {len(grp):>4,}  ({pct:>5.1f}%)')

        # Binary string frequency table (top 15)
        print(f'\n  Most common binary sequences ({ulab}):')
        counts = sub['traj_str'].value_counts().head(15)
        for seq, cnt in counts.items():
            cls = sub[sub['traj_str'] == seq]['traj_class'].iloc[0]
            print(f'    {seq}  {cnt:>4,}  ({cls})')

        # For monotone risers: when did they first cross?
        risers = sub[sub['traj_class'] == 'monotone_rise']
        if len(risers):
            win_labels = {1:'2000–04',2:'2005–09',3:'2010–14',
                          4:'2015–19',5:'2020–24'}
            print(f'\n  Monotone risers by first-crossing window ({ulab}):')
            for t, wl in win_labels.items():
                n_cross = (risers['first_cross'] == t).sum()
                print(f'    {wl}: {n_cross:>3,}')


def _draw_trajectory_panel(ax, sub_traj: pd.DataFrame, title: str,
                            show_ylabel: bool) -> None:
    """Draw one spaghetti panel onto ax (shared helper for the facet plot)."""
    COLOURS = {
        'monotone_rise': '#1f77b4',
        'oscillating':   '#ff7f0e',
    }
    ts = [1, 2, 3, 4, 5]
    win_labels = ['2000–04', '2005–09', '2010–14', '2015–19', '2020–24']

    # Individual paths — oscillating behind, monotone on top
    for cls in ['oscillating', 'monotone_rise']:
        grp = sub_traj[sub_traj['traj_class'] == cls]
        colour = COLOURS[cls]
        alpha  = 0.20 if cls == 'oscillating' else 0.35
        lw     = 0.6  if cls == 'oscillating' else 0.8
        for _, row in grp.iterrows():
            vs = [row[f'v_t{t}'] for t in ts]
            ax.plot(ts, vs, color=colour, alpha=alpha, linewidth=lw, zorder=2)

    # Median path per class
    for cls, colour in [('oscillating', '#ff7f0e'), ('monotone_rise', '#1f77b4')]:
        grp = sub_traj[sub_traj['traj_class'] == cls]
        if len(grp) == 0:
            continue
        med = [np.median(grp[f'v_t{t}'].values) for t in ts]
        ax.plot(ts, med, color=colour, linewidth=2.0, zorder=4,
                label=f'{cls.replace("_", " ")}  (n={len(grp):,})')

    ax.axhline(1.0, color='#d62728', linewidth=1.0, linestyle='--', zorder=3)
    ax.text(5.06, 1.0, '$v=1$', va='center', fontsize=7, color='#d62728')

    ax.set_yscale('log')
    ax.set_xticks(ts)
    ax.set_xticklabels(win_labels, rotation=20, ha='right', fontsize=7.5)
    ax.set_xlabel('Time window', labelpad=4)
    if show_ylabel:
        ax.set_ylabel('$v$  (log scale)', labelpad=4)
    ax.set_title(title, fontsize=9, pad=5)
    ax.legend(fontsize=7.5, framealpha=0.85, loc='upper left')


def plot_trajectories(traj: pd.DataFrame, paths) -> None:
    """
    Two-panel facet: Sources (left) and Institutions (right), shared y-axis.
    Each panel shows upward crossers (v≤1 in t1, v>1 in t5).
    """
    ts = [1, 2, 3, 4, 5]

    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.subplots_adjust(wspace=0.06)

    for ax, utype, title, show_y in [
        (axes[0], 'S', 'Sources', True),
        (axes[1], 'U', 'Institutions', False),
    ]:
        sub = traj[
            (traj['unit_type'] == utype) &
            (traj['v_t1'] <= 1) &
            (traj['v_t5'] > 1)
        ].copy()
        _draw_trajectory_panel(ax, sub, title, show_ylabel=show_y)

    sup = fig.suptitle(
        'Units that crossed $v=1$ upward (2000–04 → 2020–24): '
        'monotone vs oscillating paths',
        fontsize=9, y=1.02,
    )

    out = paths.plots / 'fig_stability_trajectories.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

    sup.set_visible(False)
    fig.savefig(paths.plots / 'fig_stability_trajectories_latex.pdf',
                bbox_inches='tight')
    sup.set_visible(True)
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    paths   = load_config()
    rk_path = paths.working / 'rankings.duckdb'

    if not rk_path.exists():
        raise FileNotFoundError(f'rankings.duckdb not found at {rk_path}')

    runs_by_label = {r['label']: r for r in load_runs()}

    print('=== Loading rankings ===')
    with duckdb.connect(str(rk_path), read_only=True) as db:
        panel = load_all_windows(db, runs_by_label)

    print(f'  Total rows: {len(panel):,}')

    print('\n=== Loading names ===')
    names = load_names(paths)

    print('\n=== Persistence analysis ===')
    persistence_table(panel)

    print('\n\n=== Mover analysis ===')
    movers = compute_movers(panel)
    print_movers(movers, names)

    print('\n\n=== Trajectory analysis ===')
    traj = trajectory_analysis(panel, names)
    print_trajectory_report(traj)

    print('\n=== Plotting ===')
    plot_v_scatter(panel, movers, names, paths)
    plot_movers(movers, names, paths)
    plot_trajectories(traj, paths)


if __name__ == '__main__':
    main()
    print('FINISHED!')
