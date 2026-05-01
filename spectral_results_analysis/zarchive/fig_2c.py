"""
fig_2c.py — v_S^S vs v_S^B and v_I^I vs v_I^B across corpus filters F.

For each available (F, m=0110, m=1000) run pair in the baseline window,
scatter log₁₀(v^B) [x] vs log₁₀(v^S) [y], coloured by F.
Left panel: Sources.  Right panel: Institutions.

Draw order (bottom → top): EBAX (all), A, E, B.
F=EBA, F=X have only bipartite (m=0110) runs — no SS/II counterpart.
F=A will appear once m=1000/m=0001 rows are added to params.csv and run.

Layout follows fig_3bz: equal-aspect log-linear axes, 1×2 panels.

Outputs:
  plots/fig_2c.pdf        — with suptitle (exploration)
  plots/fig_2c_latex.pdf  — without suptitle (paper)
"""

import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

_LOG_LO, _LOG_HI = -2.35, 1.35
_LOG_TICKS = [-2, -1, 0, 1]

_FX_COLOR = {
    'EBAX': '#333333',
    'EBA':  '#888888',
    'E':    '#e41a1c',
    'B':    '#377eb8',
    'A':    '#ff7f00',
    'X':    '#4daf4a',
}
_FX_MARKER = {
    'EBAX': 'o',
    'EBA':  's',
    'E':    'o',
    'B':    's',
    'A':    'D',
    'X':    'x',
}
_FX_LABEL = {
    'EBAX': r'$F{=}$all',
    'EBA':  r'$F{=}EBA$',
    'E':    r'$F{=}E$',
    'B':    r'$F{=}B$',
    'A':    r'$F{=}A$',
    'X':    r'$F{=}X$',
}
_FX_ORDER = ['EBAX', 'A', 'E', 'B']


def _lv(v: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(v, 1e-10, None))


def _table_name(r: dict) -> str:
    tau_sfx   = '_fixtau' if r.get('ref_units', '') else '_vartau'
    chi_str   = 'STAR' if float(r['chi']) == -1.0 else str(round(float(r['chi']) * 100))
    alpha_int = round(float(r['alpha']) * 100)
    return (f"rk_{r['run_code']}_{r['fx']}"
            f"_tauU{r['tau_u']}_tauS{r['tau_s']}{tau_sfx}"
            f"_rho{r['rho']}_m{r['m']}"
            f"_chi{chi_str}_alpha{alpha_int}")


# ─── Data ─────────────────────────────────────────────────────────────────────

def fetch_data(db) -> dict:
    """
    Find all available (bipartite m=0110, single-kernel m=1000/0001) run pairs
    in the baseline window, load v, merge on unit_idx.

    Returns:
        data['S'][fx] = DataFrame [unit_idx, v_bip, v_single]
        data['I'][fx] = DataFrame [unit_idx, v_bip, v_single]
    """
    tables = {row[0] for row in db.execute('SHOW TABLES').fetchall()}

    runs = load_runs()
    bl   = next(r for r in runs if r['label'] == 'baseline')

    # Baseline-window runs: same run_code/tau, vartau, rho=0, alpha=1, chi=0.5, omega=0
    window_runs = [
        r for r in runs
        if (r['run_code'] == bl['run_code']
            and int(r['tau_u']) == int(bl['tau_u'])
            and int(r['tau_s']) == int(bl['tau_s'])
            and not r.get('ref_units', '')
            and int(r.get('rho', 0)) == 0
            and round(float(r.get('alpha', 1.0)) * 100) == 100
            and round(float(r.get('chi', 0.5)) * 100) == 50
            and int(r.get('omega', 0)) == 0
            and not r.get('mu_type', ''))
    ]

    # Group by fx → m → run-dict
    by_fx: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in window_runs:
        by_fx[r['fx']][str(r['m'])] = r

    data: dict[str, dict] = {'S': {}, 'I': {}}

    def _load_v(tname: str, unit_type: str, col: str) -> pd.DataFrame | None:
        if tname not in tables:
            return None
        df = db.execute(
            f"SELECT unit_idx, v FROM {tname} WHERE unit_type='{unit_type}'"
        ).df()
        return df.rename(columns={'v': col})

    for fx in _FX_ORDER:
        if fx not in by_fx:
            continue
        m_runs = by_fx[fx]
        bip = m_runs.get('0110')
        ss  = m_runs.get('1000')
        ii  = m_runs.get('0001')

        if bip is None:
            continue
        bip_tname = _table_name(bip)

        # Sources: bipartite + SS
        if ss is not None:
            ss_tname = _table_name(ss)
            df_b = _load_v(bip_tname, 'S', 'v_bip')
            df_s = _load_v(ss_tname,  'S', 'v_single')
            if df_b is not None and df_s is not None:
                merged = df_b.merge(df_s, on='unit_idx', how='inner')
                merged = merged[(merged['v_bip'] > 0) & (merged['v_single'] > 0)]
                data['S'][fx] = merged
                print(f'  F={fx:5s} Sources:      n={len(merged):>5,}')
            else:
                print(f'  F={fx:5s} Sources:      table missing '
                      f'(bip={bip_tname in tables}, ss={ss_tname in tables})')
        else:
            print(f'  F={fx:5s} Sources:      no m=1000 run in params.csv')

        # Institutions: bipartite + II
        if ii is not None:
            ii_tname = _table_name(ii)
            df_b = _load_v(bip_tname, 'U', 'v_bip')
            df_i = _load_v(ii_tname,  'U', 'v_single')
            if df_b is not None and df_i is not None:
                merged = df_b.merge(df_i, on='unit_idx', how='inner')
                merged = merged[(merged['v_bip'] > 0) & (merged['v_single'] > 0)]
                data['I'][fx] = merged
                print(f'  F={fx:5s} Institutions: n={len(merged):>5,}')
            else:
                print(f'  F={fx:5s} Institutions: table missing '
                      f'(bip={bip_tname in tables}, ii={ii_tname in tables})')
        else:
            print(f'  F={fx:5s} Institutions: no m=0001 run in params.csv')

    return data


# ─── Plot ─────────────────────────────────────────────────────────────────────

def _log_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlim(_LOG_LO, _LOG_HI)
    ax.set_ylim(_LOG_LO, _LOG_HI)
    ax.set_xticks(_LOG_TICKS)
    ax.set_yticks(_LOG_TICKS)
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4, rotation=90)
    ax.axvline(0, color='#cccccc', lw=0.7, ls=':', zorder=0)
    ax.axhline(0, color='#cccccc', lw=0.7, ls=':', zorder=0)
    ax.plot([_LOG_LO, _LOG_HI], [_LOG_LO, _LOG_HI],
            color='#888888', lw=0.8, ls='--', zorder=1)
    ax.set_aspect('equal')


def _draw_panel(ax, data: dict, xlabel: str, ylabel: str, title: str) -> None:
    _log_axes(ax, xlabel, ylabel)

    for fx in _FX_ORDER:
        if fx not in data:
            continue
        df = data[fx]
        x  = _lv(df['v_bip'].values)
        y  = _lv(df['v_single'].values)
        ax.scatter(x, y,
                   c=_FX_COLOR[fx], marker=_FX_MARKER[fx],
                   s=6, alpha=0.50, linewidths=0.3, zorder=3,
                   label=f'{_FX_LABEL[fx]}  (n={len(df):,})')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, framealpha=0.85,
              loc='upper left', markerscale=1.6)
    ax.set_title(title, fontsize=11, pad=6)


def _pub_theme() -> None:
    sns.set_theme(style='whitegrid', font_scale=1.05)
    plt.rcParams.update({
        'axes.linewidth':    1.2,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.major.size':  5,   'ytick.major.size':  5,
    })


def _save(fig, sup, stem: str, paths) -> None:
    was_interactive = plt.isinteractive()
    plt.ioff()
    out = paths.plots / f'{stem}.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')
    sup.set_visible(False)
    fig.savefig(paths.plots / f'{stem}_latex.pdf', bbox_inches='tight')
    print(f'Saved {paths.plots / f"{stem}_latex.pdf"}')
    sup.set_visible(True)
    plt.close(fig)
    if was_interactive:
        plt.ion()


def plot2c(data: dict) -> None:
    paths = load_config()
    _pub_theme()

    fig, (ax_s, ax_i) = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(wspace=0.35)

    _draw_panel(ax_s, data['S'],
                xlabel=r'$\log(v_S^B)$',
                ylabel=r'$\log(v_S^S)$',
                title='Sources')

    _draw_panel(ax_i, data['I'],
                xlabel=r'$\log(v_I^B)$',
                ylabel=r'$\log(v_I^I)$',
                title='Institutions')

    sup = fig.suptitle(
        r'Within-layer vs bipartite influence by corpus $F$'
        r'   ($F{=}$all, A, E, B;  draw order bottom$\to$top)',
        fontsize=9, y=1.01,
    )
    _save(fig, sup, 'fig_2c', paths)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    paths   = load_config()
    rk_path = paths.working / 'rankings.duckdb'

    if not rk_path.exists():
        raise FileNotFoundError(
            f'rankings.duckdb not found at {rk_path}. '
            'Run spectral_ranking/run_rankings.py first.'
        )

    print('Loading run pairs from rankings.duckdb ...')
    with duckdb.connect(str(rk_path), read_only=True) as db:
        data = fetch_data(db)

    if not data['S'] and not data['I']:
        print('No (bipartite, single-kernel) pairs found — nothing to plot.')
        return

    plot2c(data)


if __name__ == '__main__':
    main()
    print('FINISHED!')
