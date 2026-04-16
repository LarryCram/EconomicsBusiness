"""
fig_2.py — Prestige-per-work rank curves across field scope (F), m=0110.

Baseline: F=A (all sources), m=0110, τ_U=τ_S=20, ρ=0, α=1.
x-axis locked to baseline rank order.

Source panel — four overlays (all m=0110), plotted bottom-to-top:
  F=X   — residual (neither E nor B)          (purple)
  F=M   — mixed sources (field_eb='A')        (orange)
  F=B   — business sources only               (blue)
  F=E   — economics sources only              (red)

Institution panel — baseline v, colour-coded by institution field label:
  Institution field label derived from E_signal / B_signal computed from
  citer works in the F=A edge list joined to source_master.parquet.
  E_signal = n_E_works + Σ econ_score/(econ_score+bus_score) over M-source works
  B_signal = n_B_works + Σ bus_score/(econ_score+bus_score) over M-source works
  Classification (k = INST_K):
    E: E_signal >= k AND B_signal < k
    B: B_signal >= k AND E_signal < k
    A: E_signal >= k AND B_signal >= k
    X: E_signal < k  AND B_signal < k

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

BASELINE_TABLE = f'rk_{_run_code}_A_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m0110_chi50_alpha100'
EL_TABLE       = f'el_{_run_code}_A_tauU{_tau_u}_tauS{_tau_s}_vartau'

INST_N_E    = 750   # top-N by E/B ratio → E
INST_N_B    = 750   # bottom-N by E/B ratio → B
INST_K_MIN  = 1.0   # minimum total (E_signal + B_signal) to be non-X
INST_D      = 0.1   # vertical offset: E → v*(1+D), B → v*(1-D), A → v, X dropped

# Display label → (catalog label, colour, marker)
# Order: plotted bottom-to-top (last = top layer)
OVERLAYS = [
    ('F=X', 'F=X', '#984ea3', 's'),   # purple  — bottom layer
    ('F=M', 'F=M', '#ff7f00', 'o'),   # orange  (mixed: field_eb='A')
    ('F=B', 'F=B', '#377eb8', 'x'),   # blue
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


# ─── Institution field labels ─────────────────────────────────────────────────

def fetch_inst_field_labels(el_path: Path, parquet_path: Path) -> dict:
    """
    Classify each institution in the F=A edge list as E / B / A / X.

    E_signal = n works in E-sources + Σ econ_score/(econ+bus) for M-source works
    B_signal = n works in B-sources + Σ bus_score/(econ+bus)  for M-source works

    Returns dict: inst_idx -> 'E'|'B'|'A'|'X'
    """
    sm = str(parquet_path / 'source_master.parquet')

    with duckdb.connect(str(el_path), read_only=True) as db:
        df = db.execute(f"""
            SELECT
                e.citer_inst_idx AS inst_idx,
                SUM(CASE
                    WHEN sm.field_eb = 'E' THEN 1.0
                    WHEN sm.field_eb = 'A' AND (sm.econ_score + sm.bus_score) > 0
                        THEN sm.econ_score::DOUBLE / (sm.econ_score + sm.bus_score)
                    ELSE 0.0
                END) AS e_signal,
                SUM(CASE
                    WHEN sm.field_eb = 'B' THEN 1.0
                    WHEN sm.field_eb = 'A' AND (sm.econ_score + sm.bus_score) > 0
                        THEN sm.bus_score::DOUBLE / (sm.econ_score + sm.bus_score)
                    ELSE 0.0
                END) AS b_signal
            FROM (
                SELECT DISTINCT citer_work_idx, citer_inst_idx, citer_source_idx
                FROM {EL_TABLE}
            ) e
            JOIN '{sm}' sm ON e.citer_source_idx = sm.source_idx
            GROUP BY e.citer_inst_idx
        """).df()

    # X: insufficient total signal
    active = df[(df['e_signal'] + df['b_signal']) >= INST_K_MIN].copy()

    # Sort by E/B ratio descending; B=0 → inf (pure E), E=0 → 0 (pure B)
    active['eb_ratio'] = active.apply(
        lambda r: (r['e_signal'] / r['b_signal']) if r['b_signal'] > 0 else float('inf'),
        axis=1,
    )
    active = active.sort_values('eb_ratio', ascending=False).reset_index(drop=True)

    n = len(active)
    n_e = min(INST_N_E, n)
    n_b = min(INST_N_B, n - n_e)   # can't overlap with E slice

    labels = ['A'] * n
    for i in range(n_e):
        labels[i] = 'E'
    for i in range(n - n_b, n):
        labels[i] = 'B'
    active['field_eb_inst'] = labels

    label_map = active.set_index('inst_idx')['field_eb_inst'].to_dict()
    df['field_eb_inst'] = df['inst_idx'].map(label_map).fillna('X')

    counts = df['field_eb_inst'].value_counts().to_dict()
    print(f'  Institution field labels (N_E={INST_N_E}, N_B={INST_N_B}, min_total={INST_K_MIN}): '
          + '  '.join(f'{c}={counts.get(c, 0):,}' for c in INST_FIELD_ORDER))

    return df.set_index('inst_idx')['field_eb_inst'].to_dict()


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
    ax.axhline(1.0, color='#999999', linewidth=0.8, linestyle='--', zorder=0)
    ax.text(n_baseline * 0.98, 1.0, '$v=1$',
            ha='right', va='bottom', fontsize=7.5, color='#999999')
    ax.set_xlim(1, n_baseline)
    ax.set_xlabel('Baseline rank  (F=A)', labelpad=4)
    ax.set_ylabel('Influence per work $v$', labelpad=4)
    ax.set_title('Sources', fontsize=10, pad=6)
    ax.legend(fontsize=7.5, framealpha=0.85, loc='upper right')


def _draw_inst_panel(ax, df_i_base: pd.DataFrame,
                     inst_field_map: dict, n_baseline: int) -> None:
    """
    Institution panel: baseline v coloured by institution field label (E/B/A).
    X institutions are omitted.
    Vertical offsets separate groups: E → v*(1+D), B → v*(1-D), A → v.
    """
    OFFSET = {'E': 1.0 + INST_D, 'B': 1.0 - INST_D, 'A': 1.0}

    df = df_i_base.copy()
    df['field_eb_inst'] = df['unit_idx'].map(inst_field_map).fillna('X')

    for cat in INST_FIELD_ORDER:   # A first (bottom), E last (top)
        colour, marker, _ = INST_FIELD_STYLE[cat]
        sub = df[df['field_eb_inst'] == cat]
        if sub.empty:
            continue
        is_line_marker = marker in ('x', '+')
        v_plot = sub['v'].values * OFFSET[cat]
        ax.scatter(
            sub['baseline_rank'].values,
            v_plot,
            color=colour,
            marker=marker,
            s=45 if is_line_marker else 15,
            alpha=1.0,
            zorder=INST_FIELD_ORDER.index(cat) + 2,
            edgecolors=colour if is_line_marker else 'white',
            linewidths=0.8 if is_line_marker else 0.4,
            label=f'{cat}  ({len(sub):,}/{n_baseline:,})',
        )

    ax.set_yscale('log')
    ax.axhline(1.0, color='#999999', linewidth=0.8, linestyle='--', zorder=0)
    ax.text(n_baseline * 0.98, 1.0, '$v=1$',
            ha='right', va='bottom', fontsize=7.5, color='#999999')
    ax.set_xlim(1, n_baseline)
    ax.set_xlabel('Baseline rank  (F=A)', labelpad=4)
    ax.set_ylabel('Influence per work $v$', labelpad=4)
    ax.set_title('Institutions', fontsize=10, pad=6)
    ax.legend(fontsize=7.5, framealpha=0.85, loc='upper right')


def plot2(src_rank_map: dict, df_i_base: pd.DataFrame,
          series: list, inst_field_map: dict) -> None:
    paths = load_config()
    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    fig.subplots_adjust(hspace=0.44)

    _draw_src_panel(axes[0], series, n_baseline=len(src_rank_map))
    _draw_inst_panel(axes[1], df_i_base, inst_field_map, n_baseline=len(df_i_base))

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
    el_path = paths.working / 'edge_lists.duckdb'

    if not rk_path.exists():
        raise FileNotFoundError(
            f'rankings.duckdb not found at {rk_path}. '
            'Run spectral_ranking/run_rankings.py first.'
        )
    if not el_path.exists():
        raise FileNotFoundError(
            f'edge_lists.duckdb not found at {el_path}. '
            'Run prepare_data/build_edge_lists.py first.'
        )

    print('Computing institution field labels...')
    inst_field_map = fetch_inst_field_labels(el_path, paths.parquet)

    with duckdb.connect(str(rk_path), read_only=True) as db:
        src_rank_map, df_i_base, series = fetch_data(db)

    print(f'Loaded {len(series)} series (baseline + {len(series)-1} overlays)')
    plot2(src_rank_map, df_i_base, series, inst_field_map)


if __name__ == '__main__':
    main()
    print('FINISHED!')
