"""
fig_2.py — Prestige-per-work rank curves across field scope (F), m=0110.
           Also produces fig_2c (within-layer vs bipartite influence scatter).

Baseline: all sources, m=0110, τ_U=τ_S=20, ρ=0, α=1.
x-axis locked to baseline rank order.

Source panel — four overlays (all m=0110), plotted bottom-to-top:
  F=X   — residual (neither E nor B)          (purple)
    F=A   — ambiguous sources (field_eb='A')    (orange)
  F=B   — business sources only               (blue)
  F=E   — economics sources only              (red)

Institution panel — baseline v, colour-coded by institution field label:
    Institution field label read from institution_field_eb.parquet (C_IS
    citation-weight fractions, quota-based assignment):
        X = 30% of institutions by frac_X,
        A = 10% of institutions by frac_A,
        remaining non-X split E:B = 5:8.

Outputs:
  plots/fig_2.pdf         — with title (exploration)
  plots/fig_2_latex.pdf   — without title (paper)
  plots/fig_2c.pdf        — v_S^S vs v_S^B and v_I^I vs v_I^B, with suptitle
  plots/fig_2c_latex.pdf  — same, without suptitle (paper)
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset_axes
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

_baseline      = next(r for r in load_runs() if r['label'] == 'baseline')
_run_code      = _baseline['run_code']
_tau_u         = _baseline['tau_u']
_tau_s         = _baseline['tau_s']

BASELINE_TABLE = (
    f"rk_{_run_code}_{_baseline['fx']}_tauU{_tau_u}_tauS{_tau_s}_vartau"
    "_rho0_m0110_chi50_alpha100"
)
EL_TABLE       = f"el_{_run_code}_{_baseline['fx']}_tauU{_tau_u}_tauS{_tau_s}_vartau"

INST_D      = 0.1   # vertical offset: E → v*(1+D), B → v*(1-D), A → v, X dropped

# Display label → (catalog label, colour, marker)
# Order: plotted bottom-to-top (last = top layer)
OVERLAYS = [
    ('F=A', 'F=A', '#ff7f0e', '+'),   # orange
    ('F=B', 'F=B', '#377eb8', '+'),   # blue
    ('F=E', 'F=E', '#e41a1c', '+'),   # red     — top layer
]

# Institution field categories: plotted bottom-to-top, colours match source overlays
# X is dropped from the institution panel
INST_FIELD_STYLE = {
    'A': ('#2ca02c', 'o', 1.0),    # green, no offset
    'B': ('#377eb8', 'o', 1.0),    # blue,  shifted down
    'E': ('#e41a1c', 'o', 1.0),    # red,   shifted up
}
INST_FIELD_ORDER = ['B', 'E']        # A and X omitted

# y-axis: integer log10 tick labels
_LOG_FMT = ticker.FuncFormatter(lambda x, _: str(int(round(np.log10(x)))))

INSET_XLIM  = (0, 50)
INSET_YLIM  = (0.8, 30)
INSET_YTICKS = [1, 10]


# ─── Institution field labels ─────────────────────────────────────────────────

def fetch_inst_field_labels(parquet_path: Path) -> dict:
    """Read institution E/B/A/X labels from institution_field_eb.parquet."""
    df = pd.read_parquet(str(parquet_path / 'institution_field_eb.parquet'),
                         columns=['unit_idx', 'field_eb'])
    counts = df['field_eb'].value_counts().to_dict()
    print('  Institution field labels: '
          + '  '.join(f'{c}={counts.get(c, 0):,}' for c in ['E', 'B', 'A', 'X']))
    return dict(zip(df['unit_idx'].astype(int), df['field_eb']))


# ─── Ranking data ─────────────────────────────────────────────────────────────

def fetch_data(db) -> tuple:
    """
    Returns
    -------
    src_rank_map  : dict  unit_idx -> baseline_rank  (sources)
    df_i_base     : DataFrame  baseline institutions with baseline_rank and v
    series        : list of (display_label, colour, marker, df_s, df_i)
                    baseline first, then overlays in OVERLAYS order
                    (df_i entries are unused in the new institution panel)
    """
    tables = {row[0] for row in db.execute('SHOW TABLES').fetchall()}

    if BASELINE_TABLE not in tables:
        raise RuntimeError(
            f'Baseline table {BASELINE_TABLE} not found in rankings.duckdb. '
            'Run run_rankings.py first.'
        )

    # ── Baseline: establish x-axis lock ──────────────────────────────────────
    df_base = db.execute(
        f"SELECT unit_idx, unit_type, v FROM {BASELINE_TABLE}"
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

    def project(df, rank_map):
        out = df.copy()
        out['baseline_rank'] = out['unit_idx'].map(rank_map)
        return out.dropna(subset=['baseline_rank']).sort_values('baseline_rank')

    series = [('baseline', 'black', None, df_s_base, df_i_base)]

    # ── Resolve catalog label → table name ───────────────────────────────────
    cat = db.execute(
        "SELECT label, table_name FROM ("
        "  SELECT label, table_name,"
        "         ROW_NUMBER() OVER (PARTITION BY label ORDER BY created_at DESC) AS rn"
        "  FROM _catalog"
        "  WHERE m_SI=1 AND m_IS=1 AND m_SS=0 AND m_II=0"
        f"   AND run_code='{_run_code}' AND tau_u={_tau_u} AND tau_s={_tau_s} AND rho=0"
        "    AND round(alpha*100)=100"
        ") WHERE rn=1"
    ).df()
    label_to_table = dict(zip(cat['label'], cat['table_name']))

    for disp_label, cat_label, colour, marker in OVERLAYS:
        tname = label_to_table.get(cat_label)
        if tname is None or tname not in tables:
            print(f'  WARNING: {cat_label} not found in catalog — skipping {disp_label}')
            continue
        df = db.execute(f"SELECT unit_idx, unit_type, v FROM {tname}").df()
        df_s = project(df[df['unit_type'] == 'S'], src_rank_map)
        df_i = project(df[df['unit_type'] == 'U'], inst_rank_map)
        series.append((disp_label, colour, marker, df_s, df_i))

    return src_rank_map, df_i_base, series


# ─── Axis helpers ─────────────────────────────────────────────────────────────

def _setup_log_yaxis(ax, ylim=(0.005, 20), yticks=(0.01, 0.1, 1, 10)):
    ax.set_yscale('log')
    ax.set_ylim(*ylim)
    ax.set_yticks(list(yticks))
    ax.yaxis.set_major_formatter(_LOG_FMT)
    ax.set_ylabel('$\\log(v)$', rotation=90, labelpad=4)


def _add_inset(ax, series, df_key):
    """Bottom-left inset zoomed to INSET_XLIM ranks, INSET_YLIM v-range."""
    axins = _inset_axes(
        ax, width='36%', height='36%', loc='lower left',
        bbox_to_anchor=(0.08, 0.10, 1, 1),
        bbox_transform=ax.transAxes, borderpad=0,
    )

    for label, colour, marker, df_s, df_i in series:
        df = df_s if df_key == 'S' else df_i
        if df.empty:
            continue
        is_baseline = marker is None
        d = df[(df['baseline_rank'] >= INSET_XLIM[0]) &
               (df['baseline_rank'] <= INSET_XLIM[1])]

        if is_baseline:
            axins.plot(d['baseline_rank'].values, d['v'].values,
                       color=colour, linewidth=1.4, alpha=0.5, zorder=3)
        else:
            if d.empty:
                continue
            is_lm = marker in ('x', '+')
            axins.scatter(d['baseline_rank'].values, d['v'].values,
                          color=colour, marker=marker,
                          s=40 if is_lm else 25, alpha=0.6, zorder=2,
                          edgecolors=colour if is_lm else 'white',
                          linewidths=0.8 if is_lm else 0.4)
            # Fit LOWESS on a wider window for a stable curve
            wide = df[df['baseline_rank'] <= INSET_XLIM[1] * 3]
            if len(wide) >= 5:
                frac = max(0.5, 20 / max(len(wide), 20))
                sm = lowess(np.log10(wide['v'].values), wide['baseline_rank'].values,
                            frac=frac, return_sorted=True)
                mask = ((sm[:, 0] >= INSET_XLIM[0]) &
                        (sm[:, 0] <= INSET_XLIM[1]))
                if mask.sum() >= 2:
                    axins.plot(sm[mask, 0], 10 ** sm[mask, 1],
                               color=colour, linewidth=1.8, alpha=1.0, zorder=3)

    axins.set_yscale('log')
    axins.set_ylim(*INSET_YLIM)
    axins.set_yticks(INSET_YTICKS)
    axins.yaxis.set_major_formatter(_LOG_FMT)
    axins.yaxis.set_minor_formatter(ticker.NullFormatter())
    axins.set_xlim(*INSET_XLIM)
    axins.set_xticks([0, INSET_XLIM[1]])
    axins.axhline(1.0, color='#999999', linewidth=0.6, linestyle='--', zorder=0)
    axins.tick_params(labelsize=9.0, which='both')
    axins.tick_params(which='minor', length=0)
    for spine in axins.spines.values():
        spine.set_linewidth(0.6)


# ─── Panel drawing ────────────────────────────────────────────────────────────

def _draw_src_panel(ax, series: list, n_baseline: int, max_rank=None) -> None:
    """Source panel: baseline line + restricted-corpus overlays."""
    xlim = max_rank if max_rank else n_baseline
    for label, colour, marker, df_s, _ in series:
        if df_s.empty:
            continue
        is_baseline = marker is None
        d = df_s[df_s['baseline_rank'] <= xlim] if max_rank else df_s
        n_overlap = len(df_s)

        if is_baseline:
            ax.plot(
                d['baseline_rank'].values,
                d['v'].values,
                color=colour,
                linewidth=1.4,
                alpha=0.5,
                zorder=3,
                label='baseline (all)',
            )
        else:
            if d.empty:
                continue
            is_line_marker = marker in ('x', '+')
            ax.scatter(
                d['baseline_rank'].values,
                d['v'].values,
                color=colour,
                marker=marker,
                s=45 if is_line_marker else 30,
                alpha=0.5,
                zorder=2,
                edgecolors='none' if is_line_marker else 'white',
                linewidths=0.8 if is_line_marker else 0.4,
                label=f'{label}  ({n_overlap:,}/{n_baseline:,})',
            )
            frac = max(0.2, 5 / max(len(d), 5))
            sm = lowess(np.log10(d['v'].values), d['baseline_rank'].values,
                        frac=frac, return_sorted=True)
            ax.plot(sm[:, 0], 10 ** sm[:, 1], color=colour,
                    linewidth=2.8, alpha=1.0, zorder=3)

    _setup_log_yaxis(ax)
    ax.axhline(1.0, color='#999999', linewidth=0.8, linestyle='--', zorder=0)
    ax.text(xlim * 0.98, 1.0, '$v=1$',
            ha='right', va='bottom', fontsize=7.5, color='#999999')
    ax.set_xlim(0, xlim)
    ax.set_xlabel('Baseline rank', labelpad=4)
    ax.set_title('Sources', fontsize=10, pad=6)
    ax.legend(fontsize=7.5, framealpha=0.85, loc='upper right')
    if not max_rank:
        _add_inset(ax, series, 'S')


def _draw_inst_panel(ax, series: list, n_baseline: int, max_rank=None) -> None:
    """Institution panel: baseline line + restricted-corpus overlays."""
    xlim = max_rank if max_rank else n_baseline
    for label, colour, marker, _, df_i in series:
        if df_i.empty:
            continue
        is_baseline = marker is None
        d = df_i[df_i['baseline_rank'] <= xlim] if max_rank else df_i
        n_overlap = len(df_i)

        if is_baseline:
            ax.plot(
                d['baseline_rank'].values,
                d['v'].values,
                color=colour,
                linewidth=1.4,
                alpha=0.5,
                zorder=3,
                label='baseline (all)',
            )
        else:
            if d.empty:
                continue
            is_line_marker = marker in ('x', '+')
            ax.scatter(
                d['baseline_rank'].values,
                d['v'].values,
                color=colour,
                marker=marker,
                s=45 if is_line_marker else 30,
                alpha=0.5,
                zorder=2,
                edgecolors='none' if is_line_marker else 'white',
                linewidths=0.8 if is_line_marker else 0.4,
                label=f'{label}  ({n_overlap:,}/{n_baseline:,})',
            )
            frac = max(0.2, 5 / max(len(d), 5))
            sm = lowess(np.log10(d['v'].values), d['baseline_rank'].values,
                        frac=frac, return_sorted=True)
            ax.plot(sm[:, 0], 10 ** sm[:, 1], color=colour,
                    linewidth=2.8, alpha=1.0, zorder=3)

    _setup_log_yaxis(ax)
    ax.axhline(1.0, color='#999999', linewidth=0.8, linestyle='--', zorder=0)
    ax.text(xlim * 0.98, 1.0, '$v=1$',
            ha='right', va='bottom', fontsize=7.5, color='#999999')
    ax.set_xlim(0, xlim)
    ax.set_xlabel('Baseline rank', labelpad=4)
    ax.set_title('Institutions', fontsize=10, pad=6)
    ax.legend(fontsize=7.5, framealpha=0.85, loc='upper right')
    if not max_rank:
        _add_inset(ax, series, 'I')


def plot2a(src_rank_map: dict, df_i_base: pd.DataFrame,
           series: list, top_n: int = 100) -> None:
    paths = load_config()
    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    fig.subplots_adjust(wspace=0.18)

    _draw_src_panel(axes[0], series, n_baseline=len(src_rank_map), max_rank=top_n)
    _draw_inst_panel(axes[1], series, n_baseline=len(df_i_base), max_rank=top_n)
    axes[1].set_ylabel('')

    sup = fig.suptitle(
        f'Field scope sensitivity — top {top_n} sources and institutions  '
        '(m=0110, x-axis locked to all-sources baseline)',
        fontsize=9, y=1.01,
    )

    out = paths.plots / 'fig_2a.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

    sup.set_visible(False)
    fig.savefig(paths.plots / 'fig_2a_latex.pdf', bbox_inches='tight')
    sup.set_visible(True)

    plt.close(fig)


def plot2(src_rank_map: dict, df_i_base: pd.DataFrame,
          series: list) -> None:
    paths = load_config()
    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    fig.subplots_adjust(wspace=0.05)

    _draw_src_panel(axes[0], series, n_baseline=len(src_rank_map))
    _draw_inst_panel(axes[1], series, n_baseline=len(df_i_base))
    axes[1].set_ylabel('')

    sup = fig.suptitle(
        'Field scope sensitivity — influence per work  '
        '(m=0110, x-axis locked to all-sources baseline)',
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

    plt.close(fig)

    # Console summary
    print(f'\n{"Label":<12}  {"n_S":>7}')
    print('-' * 22)
    for label, _, _, df_s, _ in series:
        print(f'{label:<12}  {len(df_s):>7,}')


# ─── Fig 2c: within-layer vs bipartite influence ──────────────────────────────

_2C_LOG_LO, _2C_LOG_HI = -2.35, 1.35
_2C_LOG_TICKS = [-1, 0, 1]

_2C_FX_COLOR  = {'EBAX': '#333333', 'EBA': '#888888', 'E': '#e41a1c',
                 'B': '#377eb8', 'A': '#ff7f00', 'X': '#4daf4a'}
_2C_FX_MARKER = {'EBAX': 'o', 'EBA': 's', 'E': 'o', 'B': 's', 'A': 'D', 'X': 'x'}
_2C_FX_LABEL  = {'EBAX': r'$F{=}$all', 'EBA': r'$F{=}EBA$', 'E': r'$F{=}E$',
                 'B': r'$F{=}B$', 'A': r'$F{=}A$', 'X': r'$F{=}X$'}
_2C_FX_ORDER  = ['EBAX', 'A', 'E', 'B']


def _2c_table_name(r: dict) -> str:
    tau_sfx   = '_fixtau' if r.get('ref_units', '') else '_vartau'
    chi_str   = 'STAR' if float(r['chi']) == -1.0 else str(round(float(r['chi']) * 100))
    alpha_int = round(float(r['alpha']) * 100)
    return (f"rk_{r['run_code']}_{r['fx']}"
            f"_tauU{r['tau_u']}_tauS{r['tau_s']}{tau_sfx}"
            f"_rho{r['rho']}_m{r['m']}"
            f"_chi{chi_str}_alpha{alpha_int}")


def _2c_fetch(db) -> dict:
    """Load (bipartite m=0110, single-kernel m=1000/0001) run pairs."""
    from collections import defaultdict
    tables = {row[0] for row in db.execute('SHOW TABLES').fetchall()}
    runs   = load_runs()
    bl     = next(r for r in runs if r['label'] == 'baseline')

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

    by_fx: dict = defaultdict(dict)
    for r in window_runs:
        by_fx[r['fx']][str(r['m'])] = r

    data: dict = {'S': {}, 'I': {}}

    def _load_v(tname, unit_type, col):
        if tname not in tables:
            return None
        df = db.execute(
            f"SELECT unit_idx, v FROM {tname} WHERE unit_type='{unit_type}'"
        ).df()
        return df.rename(columns={'v': col})

    for fx in _2C_FX_ORDER:
        if fx not in by_fx:
            continue
        m_runs = by_fx[fx]
        bip    = m_runs.get('0110')
        ss     = m_runs.get('1000')
        ii     = m_runs.get('0001')
        if bip is None:
            continue
        bip_t = _2c_table_name(bip)

        if ss is not None:
            df_b = _load_v(bip_t, 'S', 'v_bip')
            df_s = _load_v(_2c_table_name(ss), 'S', 'v_single')
            if df_b is not None and df_s is not None:
                m = df_b.merge(df_s, on='unit_idx', how='inner')
                m = m[(m['v_bip'] > 0) & (m['v_single'] > 0)]
                data['S'][fx] = m
                print(f'  fig_2c F={fx:5s} Sources:      n={len(m):,}')

        if ii is not None:
            df_b = _load_v(bip_t, 'U', 'v_bip')
            df_i = _load_v(_2c_table_name(ii), 'U', 'v_single')
            if df_b is not None and df_i is not None:
                m = df_b.merge(df_i, on='unit_idx', how='inner')
                m = m[(m['v_bip'] > 0) & (m['v_single'] > 0)]
                data['I'][fx] = m
                print(f'  fig_2c F={fx:5s} Institutions: n={len(m):,}')

    return data


def _2c_log_axes(ax, xlabel, ylabel):
    ax.set_xlim(_2C_LOG_LO, _2C_LOG_HI)
    ax.set_ylim(_2C_LOG_LO, _2C_LOG_HI)
    ax.set_xticks(_2C_LOG_TICKS)
    ax.set_yticks(_2C_LOG_TICKS)
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4, rotation=90)
    ax.axvline(0, color='#cccccc', lw=0.7, ls=':', zorder=0)
    ax.axhline(0, color='#cccccc', lw=0.7, ls=':', zorder=0)
    ax.plot([_2C_LOG_LO, _2C_LOG_HI], [_2C_LOG_LO, _2C_LOG_HI],
            color='#888888', lw=0.8, ls='--', zorder=1)
    ax.set_aspect('equal')


def _2c_draw_panel(ax, data, xlabel, ylabel, title):
    _2c_log_axes(ax, xlabel, ylabel)
    for fx in _2C_FX_ORDER:
        if fx not in data:
            continue
        df = data[fx]
        x  = np.log10(np.clip(df['v_bip'].values,    1e-10, None))
        y  = np.log10(np.clip(df['v_single'].values, 1e-10, None))
        ax.scatter(x, y,
                   c=_2C_FX_COLOR[fx], marker=_2C_FX_MARKER[fx],
                   s=12, alpha=0.5, linewidths=1.0, zorder=3,
                   label=f'{_2C_FX_LABEL[fx]}  (n={len(df):,})')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=10, framealpha=0.85, loc='upper left', markerscale=1.6)
    ax.set_title(title, fontsize=11, pad=6)


def plot2c(db) -> None:
    paths = load_config()
    data  = _2c_fetch(db)
    if not data['S'] and not data['I']:
        print('fig_2c: no (bipartite, single-kernel) pairs found — skipping.')
        return

    sns.set_theme(style='whitegrid', font_scale=1.05)
    plt.rcParams.update({'axes.linewidth': 1.2,
                         'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
                         'xtick.major.size':  5,   'ytick.major.size':  5})

    fig, (ax_s, ax_i) = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.05)

    _2c_draw_panel(ax_s, data['S'],
                   xlabel=r'$\log(v_S^B)$', ylabel=r'$\log(v_S^S)$',
                   title='Sources')
    _2c_draw_panel(ax_i, data['I'],
                   xlabel=r'$\log(v_I^B)$', ylabel=r'',
                   title='Institutions')
    ax_i.tick_params(labelleft=False)

    sup = fig.suptitle(
        r'Within-layer vs bipartite influence by corpus $F$'
        r'   ($F{=}$all, A, E, B;  draw order bottom$\to$top)',
        fontsize=9, y=1.01,
    )
    was_interactive = plt.isinteractive()
    plt.ioff()
    out = paths.plots / 'fig_2c.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')
    sup.set_visible(False)
    fig.savefig(paths.plots / 'fig_2c_latex.pdf', bbox_inches='tight')
    print(f'Saved {paths.plots / "fig_2c_latex.pdf"}')
    sup.set_visible(True)
    plt.close(fig)
    if was_interactive:
        plt.ion()


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
        src_rank_map, df_i_base, series = fetch_data(db)
        print(f'Loaded {len(series)} series (baseline + {len(series)-1} overlays)')
        plot2(src_rank_map, df_i_base, series)
        plot2a(src_rank_map, df_i_base, series)
        plot2c(db)


if __name__ == '__main__':
    main()
    print('FINISHED!')
