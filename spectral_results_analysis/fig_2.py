"""
fig_2.py — Prestige-per-work rank curves across field scope (F), m=0110.

Baseline: F=ALL (full corpus), m=0110, τ_U=τ_S=20, ρ=0, α=1.
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
    ('F=A', 'F=A', '#ff7f0e', '+'),   # orange  — top layer
]

# Institution field categories: plotted bottom-to-top, colours match source overlays
# X is dropped from the institution panel
INST_FIELD_STYLE = {
    'A': ('#2ca02c', 'o', 1.0),    # green, no offset
    'B': ('#377eb8', 'o', 1.0),    # blue,  shifted down
    'E': ('#e41a1c', 'o', 1.0),    # red,   shifted up
}
INST_FIELD_ORDER = ['B', 'E']        # A and X omitted


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


# ─── Plot ─────────────────────────────────────────────────────────────────────

def _draw_src_panel(ax, series: list, n_baseline: int) -> None:
    """Source panel: baseline line + restricted-corpus overlays."""
    for label, colour, marker, df_s, _ in series:
        if df_s.empty:
            continue
        is_baseline = marker is None
        n_overlap = len(df_s)

        if is_baseline:
            ax.plot(
                df_s['baseline_rank'].values,
                df_s['v'].values,
                color=colour,
                linewidth=1.4,
                alpha=1.0,
                zorder=3,
                label='baseline (all)',
            )
        else:
            is_line_marker = marker in ('x', '+')
            ax.scatter(
                df_s['baseline_rank'].values,
                df_s['v'].values,
                color=colour,
                marker=marker,
                s=45 if is_line_marker else 30,
                alpha=1.0,
                zorder=2,
                edgecolors=colour if is_line_marker else 'white',
                linewidths=0.8 if is_line_marker else 0.4,
                label=f'{label}  ({n_overlap:,}/{n_baseline:,})',
            )

    ax.set_yscale('log')
    ax.set_ylim(0.005, 20)
    ax.set_yticks([0.01, 0.1, 1, 10])
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.axhline(1.0, color='#999999', linewidth=0.8, linestyle='--', zorder=0)
    ax.text(n_baseline * 0.98, 1.0, '$v=1$',
            ha='right', va='bottom', fontsize=7.5, color='#999999')
    ax.set_xlim(1, n_baseline)
    ax.set_xlabel('Baseline rank  F=A', labelpad=4)
    ax.set_ylabel('Influence per work $v$', labelpad=4)
    ax.set_title('Sources', fontsize=10, pad=6)
    ax.legend(fontsize=7.5, framealpha=0.85, loc='upper right')


def _draw_inst_panel(ax, series: list, n_baseline: int) -> None:
    """Institution panel: baseline line + restricted-corpus overlays."""
    for label, colour, marker, _, df_i in series:
        if df_i.empty:
            continue
        is_baseline = marker is None
        n_overlap = len(df_i)

        if is_baseline:
            ax.plot(
                df_i['baseline_rank'].values,
                df_i['v'].values,
                color=colour,
                linewidth=1.4,
                alpha=1.0,
                zorder=3,
                label='baseline (all)',
            )
        else:
            is_line_marker = marker in ('x', '+')
            ax.scatter(
                df_i['baseline_rank'].values,
                df_i['v'].values,
                color=colour,
                marker=marker,
                s=45 if is_line_marker else 30,
                alpha=1.0,
                zorder=2,
                edgecolors=colour if is_line_marker else 'white',
                linewidths=0.8 if is_line_marker else 0.4,
                label=f'{label}  ({n_overlap:,}/{n_baseline:,})',
            )

    ax.set_yscale('log')
    ax.set_ylim(0.005, 20)
    ax.set_yticks([0.01, 0.1, 1, 10])
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.axhline(1.0, color='#999999', linewidth=0.8, linestyle='--', zorder=0)
    ax.text(n_baseline * 0.98, 1.0, '$v=1$',
            ha='right', va='bottom', fontsize=7.5, color='#999999')
    ax.set_xlim(1, n_baseline)
    ax.set_xlabel('Baseline rank  F=A', labelpad=4)
    ax.set_ylabel('Influence per work $v$', labelpad=4)
    ax.set_title('Institutions', fontsize=10, pad=6)
    ax.legend(fontsize=7.5, framealpha=0.85, loc='upper right')


def plot2(src_rank_map: dict, df_i_base: pd.DataFrame,
        series: list, inst_field_map: dict) -> None:
    paths = load_config()
    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    fig.subplots_adjust(wspace=0.18)

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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    paths   = load_config()
    rk_path = paths.working / 'rankings.duckdb'

    if not rk_path.exists():
        raise FileNotFoundError(
            f'rankings.duckdb not found at {rk_path}. '
            'Run spectral_ranking/run_rankings.py first.'
        )

    print('Loading institution field labels...')
    inst_field_map = fetch_inst_field_labels(paths.parquet)

    with duckdb.connect(str(rk_path), read_only=True) as db:
        src_rank_map, df_i_base, series = fetch_data(db)

    print(f'Loaded {len(series)} series (baseline + {len(series)-1} overlays)')
    plot2(src_rank_map, df_i_base, series, inst_field_map)


if __name__ == '__main__':
    main()
    print('FINISHED!')
