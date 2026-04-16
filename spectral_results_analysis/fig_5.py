"""
fig_5.py — Phase 2 parameter sensitivity: τ, ρ, and (α, μ).

Baseline: F=A, τ_U=τ_S=20, ρ=0, m=0110, α=1.
x-axis locked to baseline rank (same convention as fig_2/fig_3).

Five overlays:
  τ=40              : raise both τ_U and τ_S to 40; α=1, ρ=0.
                      Units dropped by the higher threshold are absent.

  ρ=1               : full reference count (equal attention per reference);
                      τ=20, α=1.

  census=1yr        : census window 2024 only (tc0=tc1=2024), target 2020–24.
                      τ is applied to a single year; closer to AIS formula
                      and reduces overcounting of near-year references.

  α=0.85, μ=1/N     : Katz–Hubbell, uniform prior — μ_p = 1/(N_S+N_U) for
                      all units; τ=20, ρ=0.

  α=0.85, μ=1/N_p   : Katz–Hubbell, unit-scaled prior — μ_p = 1/N_S for
                      sources, 1/N_U for institutions; τ=20, ρ=0.

All computations use the bipartite m=0110 mode.
χ is not material for the bipartite mode and is not varied here.
α=0.85 cases are computed on-the-fly via bipartite(); not stored in rankings.duckdb.

Outputs:
  plots/fig_5.pdf        — with title (exploration)
  plots/fig_5_latex.pdf  — without title (paper)
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
from spectral_ranking.build_csr import build_csr
from spectral_ranking.katz_ranker import _row_normalise, bipartite

_baseline      = next(r for r in load_runs() if r['label'] == 'baseline')
_run_code      = _baseline['run_code']
_tau_u         = _baseline['tau_u']
_tau_s         = _baseline['tau_s']

BASELINE_TABLE = f'rk_{_run_code}_A_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m0110_chi50_alpha100'

# Visual style per overlay label (ordered: baseline drawn first)
STYLE = {
    'baseline':        dict(color='black',   marker=None, lw=1.4, alpha_vis=1.0,  zorder=5),
    'τ=40':            dict(color='#9467bd', marker='x',  lw=0.8, alpha_vis=0.65, zorder=4, s=40),
    'ρ=1':             dict(color='#ff7f0e', marker='x',  lw=0.8, alpha_vis=0.65, zorder=3, s=40),
    'census=1yr':      dict(color='#1f77b4', marker='x',  lw=0.8, alpha_vis=0.65, zorder=4, s=40),
    'α=0.85, μ=1/N':   dict(color='#d62728', marker='x', lw=0.8, alpha_vis=0.65, zorder=2, s=40),
    'α=0.85, μ=1/N_p': dict(color='#2ca02c', marker='x', lw=0.8, alpha_vis=0.65, zorder=2, s=40),
}


# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_baseline(rk_db) -> tuple:
    """
    Load v from the stored baseline ranking.

    Returns
    -------
    src_rank_map  : dict  unit_idx -> baseline_rank  (sources)
    inst_rank_map : dict  unit_idx -> baseline_rank  (institutions)
    df_s_base     : DataFrame [unit_idx, baseline_rank, v, a_p]
    df_u_base     : DataFrame [unit_idx, baseline_rank, v, a_p]
    """
    df = rk_db.execute(
        f"SELECT unit_idx, unit_type, v, a_p FROM {BASELINE_TABLE}"
    ).df()

    df_s = (df[df['unit_type'] == 'S']
            .sort_values('v', ascending=False)
            .reset_index(drop=True))
    df_s['baseline_rank'] = np.arange(1, len(df_s) + 1)
    src_rank_map = df_s.set_index('unit_idx')['baseline_rank'].to_dict()

    df_u = (df[df['unit_type'] == 'U']
            .sort_values('v', ascending=False)
            .reset_index(drop=True))
    df_u['baseline_rank'] = np.arange(1, len(df_u) + 1)
    inst_rank_map = df_u.set_index('unit_idx')['baseline_rank'].to_dict()

    return src_rank_map, inst_rank_map, df_s, df_u


def _compute_bipartite_v(
    el_db,
    run_code: str, fx: str, tau_u: int, tau_s: int,
    rho: int, alpha: float,
    mu: np.ndarray | None = None,
) -> tuple:
    """
    Compute bipartite v_s, v_u directly from edge_lists.duckdb.

    Returns (df_s, df_u) with columns [unit_idx, v], or (None, None)
    if the required edge-list or units table is absent.
    """
    tname = f'el_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_vartau'
    uname = f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_vartau_m0110'
    tables = {r[0] for r in el_db.execute('SHOW TABLES').fetchall()}
    missing = [t for t in (tname, uname) if t not in tables]
    if missing:
        print(f'  WARNING: tables not found: {missing} — skipping')
        return None, None

    csr = build_csr(el_db, run_code, fx, tau_u, tau_s, rho, (0, 1, 1, 0))
    H_SI, _ = _row_normalise(csr.C_SI)
    H_IS, _ = _row_normalise(csr.C_IS)

    pi_s_ind, pi_u_ind, _lam1, _lam2, iters, norm = bipartite(H_SI, H_IS, alpha=alpha, mu=mu)

    # Joint-normalise matching rank() convention
    pi_s = pi_s_ind / 2.0
    pi_u = pi_u_ind / 2.0
    A = csr.a_s.sum() + csr.a_u.sum()
    v_s = A * pi_s / csr.a_s
    v_u = A * pi_u / csr.a_u

    df_s = pd.DataFrame({'unit_idx': csr.source_ids.astype(int), 'v': v_s})
    df_u = pd.DataFrame({'unit_idx': csr.inst_ids.astype(int),   'v': v_u})

    print(f'    n_s={csr.n_s:,}  n_u={csr.n_u:,}  iters={iters}  norm={norm:.2e}')
    return df_s, df_u


def fetch_data(rk_db, el_db) -> tuple:
    """
    Returns
    -------
    src_rank_map, inst_rank_map : x-axis lock dicts
    series : dict  label -> {'S': DataFrame, 'I': DataFrame}
             Each DataFrame has columns [unit_idx, baseline_rank, v].
    """
    src_rank_map, inst_rank_map, df_s_base, df_u_base = load_baseline(rk_db)

    def project(df, rank_map):
        out = df.copy()
        out['baseline_rank'] = out['unit_idx'].map(rank_map)
        return out.dropna(subset=['baseline_rank']).sort_values('baseline_rank')

    series = {'baseline': {'S': df_s_base, 'I': df_u_base}}

    # ── τ=40 ──────────────────────────────────────────────────────────────────
    print('  τ=40 ...')
    df_s, df_u = _compute_bipartite_v(el_db, _run_code, 'A', 40, 40, 0, 1.0)
    if df_s is not None:
        series['τ=40'] = {
            'S': project(df_s, src_rank_map),
            'I': project(df_u, inst_rank_map),
        }

    # ── ρ=1 ───────────────────────────────────────────────────────────────────
    print('  ρ=1 ...')
    df_s, df_u = _compute_bipartite_v(el_db, _run_code, 'A', _tau_u, _tau_s, 1, 1.0)
    if df_s is not None:
        series['ρ=1'] = {
            'S': project(df_s, src_rank_map),
            'I': project(df_u, inst_rank_map),
        }

    # ── census=1yr (tc0=tc1=2024, tt0=2020, tt1=2024) ───────────────────────
    print('  census=1yr ...')
    df_s, df_u = _compute_bipartite_v(el_db, '24242024', 'A', _tau_u, _tau_s, 0, 1.0)
    if df_s is not None:
        series['census=1yr'] = {
            'S': project(df_s, src_rank_map),
            'I': project(df_u, inst_rank_map),
        }

    # ── α=0.85: build both mu vectors from the baseline corpus ───────────────
    N_s = len(df_s_base)
    N_u = len(df_u_base)
    N   = N_s + N_u

    # μ=1/N  — uniform (Katz)
    print('  α=0.85, μ=1/N (uniform) ...')
    mu_uniform = np.full(N, 1.0 / N)
    df_s, df_u = _compute_bipartite_v(
        el_db, _run_code, 'A', _tau_u, _tau_s, 0, 0.85, mu=mu_uniform
    )
    if df_s is not None:
        series['α=0.85, μ=1/N'] = {
            'S': project(df_s, src_rank_map),
            'I': project(df_u, inst_rank_map),
        }

    # μ=1/N_p — unit-scaled (1/N_S for sources, 1/N_U for institutions)
    print('  α=0.85, μ=1/N_p (unit_scaled) ...')
    mu_unit = np.concatenate([
        np.full(N_s, 1.0 / N_s),
        np.full(N_u, 1.0 / N_u),
    ])
    df_s, df_u = _compute_bipartite_v(
        el_db, _run_code, 'A', _tau_u, _tau_s, 0, 0.85, mu=mu_unit
    )
    if df_s is not None:
        series['α=0.85, μ=1/N_p'] = {
            'S': project(df_s, src_rank_map),
            'I': project(df_u, inst_rank_map),
        }

    return src_rank_map, inst_rank_map, series


# ─── Plot ─────────────────────────────────────────────────────────────────────

def _draw_panel(ax, series: dict, unit_key: str,
                n_baseline: int, panel_title: str) -> None:
    for label, style in STYLE.items():
        if label not in series:
            continue
        df = series[label][unit_key]
        if df.empty:
            continue

        n_overlap = len(df)
        is_baseline = style['marker'] is None

        if is_baseline:
            ax.plot(
                df['baseline_rank'].values,
                df['v'].values,
                color=style['color'],
                linewidth=style['lw'],
                alpha=style['alpha_vis'],
                zorder=style['zorder'],
                label='baseline',
            )
        else:
            ax.scatter(
                df['baseline_rank'].values,
                df['v'].values,
                color=style['color'],
                marker=style['marker'],
                s=style['s'],
                linewidths=style['lw'],
                alpha=style['alpha_vis'],
                zorder=style['zorder'],
                label=f'{label}  ({n_overlap:,}/{n_baseline:,})',
            )

    ax.set_yscale('log')
    ax.axhline(1.0, color='#999999', linewidth=0.8, linestyle='--', zorder=0)
    ax.text(
        n_baseline * 0.98, 1.0, '$v=1$',
        ha='right', va='bottom', fontsize=7.5, color='#999999',
    )
    ax.set_xlim(1, n_baseline)
    ax.set_xlabel('Baseline rank', labelpad=4)
    ax.set_ylabel('Influence per work $v$', labelpad=4)
    ax.set_title(panel_title, fontsize=10, pad=6)
    ax.legend(fontsize=7.5, framealpha=0.85, loc='lower left')


def plot5(src_rank_map: dict, inst_rank_map: dict, series: dict) -> None:
    paths = load_config()
    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    fig.subplots_adjust(hspace=0.44)

    _draw_panel(axes[0], series, 'S', len(src_rank_map),  'Sources')
    _draw_panel(axes[1], series, 'I', len(inst_rank_map), 'Institutions')

    sup = fig.suptitle(
        'Parameter sensitivity — influence per work  '
        '(x-axis locked to bipartite baseline)',
        fontsize=9, y=1.01,
    )

    out = paths.plots / 'fig_5.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

    sup.set_visible(False)
    latex_out = paths.plots / 'fig_5_latex.pdf'
    fig.savefig(latex_out, bbox_inches='tight')
    print(f'Saved {latex_out}')
    sup.set_visible(True)

    plt.close(fig)

    # Console summary
    print(f'\n{"Label":<14}  {"n_S":>7}  {"n_I":>7}')
    print('-' * 32)
    for label, d in series.items():
        print(f'{label:<14}  {len(d["S"]):>7,}  {len(d["I"]):>7,}')


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

    with duckdb.connect(str(rk_path), read_only=True) as rk_db, \
         duckdb.connect(str(el_path), read_only=True) as el_db:
        print('Computing overlays:')
        src_rank_map, inst_rank_map, series = fetch_data(rk_db, el_db)

    plot5(src_rank_map, inst_rank_map, series)


if __name__ == '__main__':
    main()
    print('FINISHED!')
