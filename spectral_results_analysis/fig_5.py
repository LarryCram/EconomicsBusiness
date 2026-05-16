"""
fig_5.py — Phase 2 parameter sensitivity: τ, ρ, (α, μ), ω, and ε.

Baseline: all sources, τ_U=τ_S=20, ρ=0, m=0110, α=1, ω=0 (author-fractional).
x-axis locked to baseline rank (same convention as fig_2/fig_3).

Seven overlays:
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

  no self-ref       : zero diagonal of M_S=H_SI@H_IS and M_I=H_IS@H_SI,
                      removing the self-loop through own institutional
                      affiliation.  Imprimitive so uses α=0.85, μ=1/N_p.
                      Shown on fig_5d alongside the other α=0.85 cases.

  ω=1               : direct 1/N_inst institution weights instead of
                      author-fractional; τ=20, ρ=0, α=1.

  ε=1               : sentinel nodes absorb cross-boundary references
                      (corpus→outside and outside→corpus); sentinel rows
                      (unit_idx=1) are excluded from the comparison.
                      Loaded from pre-computed baseline-eps ranking.

All computations use the bipartite m=0110 mode.
χ is not material for the bipartite mode and is not varied here.
α=0.85 cases are computed on-the-fly via bipartite(); not stored in rankings.duckdb.
ε=1 is loaded from rankings.duckdb (pre-computed baseline-eps run).

Outputs:
  plots/fig_5.pdf        — all overlays combined, with title (exploration)
  plots/fig_5_latex.pdf  — all overlays combined, without title (paper)
  plots/fig_5a.pdf/latex — τ sensitivity
  plots/fig_5b.pdf/latex — ρ sensitivity
  plots/fig_5c.pdf/latex — census window sensitivity
  plots/fig_5d.pdf/latex — α=0.85 Katz–Hubbell sensitivity
  plots/fig_5e.pdf/latex — ω=1 institution weighting sensitivity
  plots/fig_5f.pdf/latex — ε=1 sentinel-boundary sensitivity
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend: saves to file, never blocks on show()

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_LOG_LO, _LOG_HI = -2.35, 1.35
_LOG_TICKS = [-1, 0, 1]


def _lv(v: np.ndarray) -> np.ndarray:
    """log₁₀, clipped to avoid -inf."""
    return np.log10(np.clip(v, 1e-10, None))


def _pub_theme():
    sns.set_theme(style='whitegrid', font_scale=1.05)
    plt.rcParams.update({
        'axes.linewidth':    1.2,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.major.size':  5,   'ytick.major.size':  5,
    })


def _save(fig, sup, stem: str, paths) -> None:
    out = paths.plots / f'{stem}.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')
    sup.set_visible(False)
    fig.savefig(paths.plots / f'{stem}_latex.pdf', bbox_inches='tight')
    print(f'Saved {paths.plots / f"{stem}_latex.pdf"}')
    sup.set_visible(True)
    plt.close(fig)

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs
from spectral_ranking.build_csr import build_csr
from spectral_ranking.katz_ranker import _row_normalise, bipartite, power_iteration

_baseline      = next(r for r in load_runs() if r['label'] == 'baseline')
_run_code      = _baseline['run_code']
_tau_u         = _baseline['tau_u']
_tau_s         = _baseline['tau_s']

BASELINE_TABLE = f"rk_{_run_code}_{_baseline['fx']}_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m0110_chi50_alpha100"
EPS_TABLE      = BASELINE_TABLE + '_eps1'

# Visual style per overlay label (ordered: baseline drawn first)
STYLE = {
    'baseline':        dict(color='black',   marker=None, lw=1.4, alpha_vis=1.0,  zorder=5),
    'τ=40':            dict(color='#e41a1c', marker='o',  lw=0.0, alpha_vis=0.50, zorder=4, s=12),
    'ρ=1':             dict(color='#377eb8', marker='o',  lw=0.0, alpha_vis=0.50, zorder=3, s=12),
    'census=1yr':      dict(color='#4daf4a', marker='o',  lw=0.0, alpha_vis=0.50, zorder=4, s=12),
    'α=0.85, μ=1/N_p': dict(color='#377eb8', marker='o', lw=0.0, alpha_vis=0.55, zorder=3, s=12),
    'H_SS/II, no self': dict(color='#4daf4a', marker='o', lw=0.0, alpha_vis=0.55, zorder=3, s=12),
    'no self-ref':     dict(color='#e41a1c', marker='o', lw=0.0, alpha_vis=0.65, zorder=6, s=12),
    'ω=1':             dict(color='#ff7f00', marker='o',  lw=0.0, alpha_vis=0.50, zorder=3, s=12),
    'ε=1':             dict(color='#984ea3', marker='o',  lw=0.0, alpha_vis=0.55, zorder=4, s=12),
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
    direct_inst: bool = False,
    zero_diag: bool = False,
) -> tuple:
    """
    Compute bipartite v_s, v_u directly from edge_lists.duckdb.

    zero_diag=True zeros the diagonal of M_S=H_SI@H_IS and M_I=H_IS@H_SI
    before re-normalising, removing the self-loop where a unit reaches itself
    via its own institutional affiliation.  Makes both kernels imprimitive so
    requires alpha < 1 with a prior mu.

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

    csr = build_csr(el_db, run_code, fx, tau_u, tau_s, rho, (0, 1, 1, 0),
                    direct_inst=direct_inst)
    H_SI, _ = _row_normalise(csr.C_SI)
    H_IS, _ = _row_normalise(csr.C_IS)

    if zero_diag:
        M_S = H_SI.dot(H_IS).tocsr(); M_S.setdiag(0); M_S.eliminate_zeros()
        M_I = H_IS.dot(H_SI).tocsr(); M_I.setdiag(0); M_I.eliminate_zeros()
        H_S, _ = _row_normalise(M_S)
        H_I, _ = _row_normalise(M_I)
        n_s, n_u = csr.n_s, csr.n_u
        if mu is not None:
            mu_s = mu[:n_s]; mu_s = mu_s / mu_s.sum()
            mu_u = mu[n_s:]; mu_u = mu_u / mu_u.sum()
        else:
            mu_s = mu_u = None
        pi_s, _, _, iters_s, norm_s = power_iteration(H_S, alpha, mu=mu_s)
        pi_u, _, _, iters_u, norm_u = power_iteration(H_I, alpha, mu=mu_u)
        print(f'    n_s={n_s:,}  n_u={n_u:,}  '
              f'iters_s={iters_s}  norm_s={norm_s:.2e}  '
              f'iters_u={iters_u}  norm_u={norm_u:.2e}')
    else:
        pi_s, pi_u, _, _, iters, norm = bipartite(H_SI, H_IS, alpha=alpha, mu=mu)
        print(f'    n_s={csr.n_s:,}  n_u={csr.n_u:,}  iters={iters}  norm={norm:.2e}')

    # Joint-normalise matching rank() convention
    pi_s = pi_s / 2.0
    pi_u = pi_u / 2.0
    A = csr.a_s.sum() + csr.a_u.sum()
    v_s = A * pi_s / csr.a_s
    v_u = A * pi_u / csr.a_u

    df_s = pd.DataFrame({'unit_idx': csr.source_ids.astype(int), 'v': v_s})
    df_u = pd.DataFrame({'unit_idx': csr.inst_ids.astype(int),   'v': v_u})

    return df_s, df_u


def _compute_diag_single_v(
    el_db,
    run_code: str, fx: str, tau_u: int, tau_s: int,
    alpha: float,
) -> tuple:
    """
    Compute v_S from H_SS (zero diagonal) and v_U from H_II (zero diagonal).

    Uses m=1000 unit set for sources and m=0001 unit set for institutions.
    Prior: μ = 1/N_p (uniform within each unit type).
    Normalised as A_type · π / a so the a-weighted mean v = 1 in each type.
    Returns (df_s, df_u) or (None, None) if either unit table is absent.
    """
    tables = {r[0] for r in el_db.execute('SHOW TABLES').fetchall()}

    ss_uname = f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_vartau_m1000'
    ii_uname = f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_vartau_m0001'
    missing = [t for t in (ss_uname, ii_uname) if t not in tables]
    if missing:
        print(f'  WARNING: unit tables not found: {missing} — skipping')
        return None, None

    # ── Sources: H_SS with zero diagonal ─────────────────────────────────────
    csr_ss = build_csr(el_db, run_code, fx, tau_u, tau_s, 0, (1, 0, 0, 0))
    C_SS = csr_ss.C_SS.tocsr(); C_SS.setdiag(0); C_SS.eliminate_zeros()
    H_SS, _ = _row_normalise(C_SS)
    mu_s = np.full(csr_ss.n_s, 1.0 / csr_ss.n_s)
    pi_s, _, _, iters_s, norm_s = power_iteration(H_SS, alpha, mu=mu_s)
    v_s = float(csr_ss.a_s.sum()) * pi_s / csr_ss.a_s
    df_s = pd.DataFrame({'unit_idx': csr_ss.source_ids.astype(int), 'v': v_s})

    # ── Institutions: H_II with zero diagonal ─────────────────────────────────
    csr_ii = build_csr(el_db, run_code, fx, tau_u, tau_s, 0, (0, 0, 0, 1))
    C_II = csr_ii.C_II.tocsr(); C_II.setdiag(0); C_II.eliminate_zeros()
    H_II, _ = _row_normalise(C_II)
    mu_u = np.full(csr_ii.n_u, 1.0 / csr_ii.n_u)
    pi_u, _, _, iters_u, norm_u = power_iteration(H_II, alpha, mu=mu_u)
    v_u = float(csr_ii.a_u.sum()) * pi_u / csr_ii.a_u
    df_u = pd.DataFrame({'unit_idx': csr_ii.inst_ids.astype(int), 'v': v_u})

    print(f'    SS: n_s={csr_ss.n_s:,}  iters={iters_s}  norm={norm_s:.2e}')
    print(f'    II: n_u={csr_ii.n_u:,}  iters={iters_u}  norm={norm_u:.2e}')
    return df_s, df_u


def _load_stored_variant(rk_db, table: str) -> tuple:
    """
    Load v from a pre-computed ranking table, excluding sentinel rows (unit_idx=1).

    Returns (df_s, df_u) with columns [unit_idx, v], or (None, None) if absent.
    """
    tables = {r[0] for r in rk_db.execute('SHOW TABLES').fetchall()}
    if table not in tables:
        print(f'  WARNING: {table} not found — skipping')
        return None, None
    df = rk_db.execute(
        f"SELECT unit_idx, unit_type, v FROM {table} WHERE unit_idx != 1"
    ).df()
    df = df.dropna(subset=['v'])
    df_s = df[df['unit_type'] == 'S'][['unit_idx', 'v']].copy()
    df_u = df[df['unit_type'] == 'U'][['unit_idx', 'v']].copy()
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
    df_s, df_u = _compute_bipartite_v(el_db, _run_code, _baseline['fx'], 40, 40, 0, 1.0)
    if df_s is not None:
        series['τ=40'] = {
            'S': project(df_s, src_rank_map),
            'I': project(df_u, inst_rank_map),
        }

    # ── ρ=1 ───────────────────────────────────────────────────────────────────
    print('  ρ=1 ...')
    df_s, df_u = _compute_bipartite_v(el_db, _run_code, _baseline['fx'], _tau_u, _tau_s, 1, 1.0)
    if df_s is not None:
        series['ρ=1'] = {
            'S': project(df_s, src_rank_map),
            'I': project(df_u, inst_rank_map),
        }

    # ── census=1yr (tc0=tc1=2024, tt0=2020, tt1=2024) ───────────────────────
    print('  census=1yr ...')
    df_s, df_u = _compute_bipartite_v(el_db, '24242024', _baseline['fx'], _tau_u, _tau_s, 0, 1.0)
    if df_s is not None:
        series['census=1yr'] = {
            'S': project(df_s, src_rank_map),
            'I': project(df_u, inst_rank_map),
        }

    # ── α=0.85: build mu vector from the baseline corpus ────────────────────
    N_s = len(df_s_base)
    N_u = len(df_u_base)

    # μ=1/N_p — unit-scaled (1/N_S for sources, 1/N_U for institutions)
    print('  α=0.85, μ=1/N_p (unit_scaled) ...')
    mu_unit = np.concatenate([
        np.full(N_s, 1.0 / N_s),
        np.full(N_u, 1.0 / N_u),
    ])
    df_s, df_u = _compute_bipartite_v(
        el_db, _run_code, _baseline['fx'], _tau_u, _tau_s, 0, 0.85, mu=mu_unit
    )
    if df_s is not None:
        series['α=0.85, μ=1/N_p'] = {
            'S': project(df_s, src_rank_map),
            'I': project(df_u, inst_rank_map),
        }

    # H_SS/II no self-ref: zero diagonal of C_SS and C_II; α=0.85, μ=1/N_p
    print('  H_SS/II, no self-ref (zero diag, α=0.85, μ=1/N_p) ...')
    df_s, df_u = _compute_diag_single_v(
        el_db, _run_code, _baseline['fx'], _tau_u, _tau_s, 0.85,
    )
    if df_s is not None:
        series['H_SS/II, no self'] = {
            'S': project(df_s, src_rank_map),
            'I': project(df_u, inst_rank_map),
        }

    # no self-ref: zero diagonal of M_S and M_I; α=0.85, μ=1/N_p (unit-scaled)
    print('  no self-ref (zero diag M_S/M_I, α=0.85, μ=1/N_p) ...')
    df_s, df_u = _compute_bipartite_v(
        el_db, _run_code, _baseline['fx'], _tau_u, _tau_s, 0, 0.85,
        mu=mu_unit, zero_diag=True,
    )
    if df_s is not None:
        series['no self-ref'] = {
            'S': project(df_s, src_rank_map),
            'I': project(df_u, inst_rank_map),
        }

    # ── ω=1: direct 1/N_inst institution weights ──────────────────────────────
    print('  ω=1 (direct inst weights) ...')
    df_s, df_u = _compute_bipartite_v(
        el_db, _run_code, _baseline['fx'], _tau_u, _tau_s, 0, 1.0,
        direct_inst=True,
    )
    if df_s is not None:
        series['ω=1'] = {
            'S': project(df_s, src_rank_map),
            'I': project(df_u, inst_rank_map),
        }

    # ── ε=1: sentinel nodes for cross-boundary references ─────────────────────
    print('  ε=1 (sentinel boundary) ...')
    df_s, df_u = _load_stored_variant(rk_db, EPS_TABLE)
    if df_s is not None:
        series['ε=1'] = {
            'S': project(df_s, src_rank_map),
            'I': project(df_u, inst_rank_map),
        }

    return src_rank_map, inst_rank_map, series


# ─── Plot ─────────────────────────────────────────────────────────────────────

def _draw_panel(ax, series: dict, unit_key: str,
                n_baseline: int, panel_title: str,
                ylabel: str = '', show_legend: bool = True) -> None:
    for label, style in STYLE.items():
        if label not in series:
            continue
        df = series[label][unit_key]
        if df.empty:
            continue

        n_overlap = len(df)
        is_baseline = style['marker'] is None
        yv = _lv(df['v'].values)

        if is_baseline:
            ax.plot(
                df['baseline_rank'].values,
                yv,
                color=style['color'],
                linewidth=style['lw'],
                alpha=style['alpha_vis'],
                zorder=style['zorder'],
                label='baseline',
            )
        else:
            ax.scatter(
                df['baseline_rank'].values,
                yv,
                color=style['color'],
                marker=style['marker'],
                s=style['s'],
                linewidths=style['lw'],
                alpha=style['alpha_vis'],
                zorder=style['zorder'],
                label=f'{label}  ({n_overlap:,}/{n_baseline:,})',
            )

    ax.set_ylim(_LOG_LO, _LOG_HI)
    ax.set_yticks(_LOG_TICKS)
    ax.axhline(0.0, color='#999999', linewidth=0.8, linestyle='--', zorder=0)
    ax.set_xlim(1, n_baseline)
    ax.set_xlabel('Baseline rank', labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=4)
    ax.set_title(panel_title, fontsize=10, pad=6)
    if show_legend:
        ax.legend(fontsize=7.5, framealpha=0.85, loc='lower left')


def plot5(src_rank_map: dict, inst_rank_map: dict, series: dict) -> None:
    paths = load_config()
    _pub_theme()

    _SHOW = ['baseline', 'τ=40', 'ρ=1', 'census=1yr', 'ω=1', 'ε=1']
    sub = {k: series[k] for k in _SHOW if k in series}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.subplots_adjust(wspace=0.04)

    _draw_panel(axes[0], sub, 'S', len(src_rank_map),
                'Sources', ylabel=r'$\log(v)$', show_legend=True)
    _draw_panel(axes[1], sub, 'I', len(inst_rank_map), 'Institutions',
                show_legend=True)
    axes[1].tick_params(labelleft=False)

    sup = fig.suptitle(
        'Parameter sensitivity — influence per work  '
        '(x-axis locked to bipartite baseline)',
        fontsize=9, y=1.01,
    )

    _save(fig, sup, 'fig_5', paths)

    # Console summary
    print(f'\n{"Label":<14}  {"n_S":>7}  {"n_I":>7}')
    print('-' * 32)
    for label, d in series.items():
        print(f'{label:<14}  {len(d["S"]):>7,}  {len(d["I"]):>7,}')


# ─── Stability report ────────────────────────────────────────────────────────

def report_stability(series: dict) -> None:
    """
    For each non-baseline variant and each unit type report:
      n        — units common to baseline and variant
      spear_r  — Spearman ρ(v_var, v_base): primary rank-preservation summary
      med_lr   — median log(v_var / v_base): systematic log-level shift
      sd_lr    — SD of log(v_var / v_base): spread (dominated by rank shuffling)
      wil_p    — Wilcoxon p (H0: median log-ratio = 0)
      med_dr   — median |rank_var − rank_base| (variant rank within its own corpus)
      med_rel  — median |rank_var/n_var − rank_base/n_base| (fractional percentile shift)
    """
    from scipy import stats as _stats

    base_s = (series['baseline']['S']
              .set_index('unit_idx')[['v', 'baseline_rank']]
              .rename(columns={'v': 'v_base', 'baseline_rank': 'rank_base'}))
    base_i = (series['baseline']['I']
              .set_index('unit_idx')[['v', 'baseline_rank']]
              .rename(columns={'v': 'v_base', 'baseline_rank': 'rank_base'}))

    hdr = (f'{"Variant":<24}  {"Type":<5}  {"n":>6}  '
           f'{"Spear ρ":>8}  {"med log-r":>10}  {"sd log-r":>9}  {"Wil p":>10}  '
           f'{"med|Δrank|":>11}  {"med|Δpct|":>10}')
    print('\n' + '─' * len(hdr))
    print('Parameter stability — log-ratio and rank displacement')
    print('─' * len(hdr))
    print(hdr)
    print('─' * len(hdr))

    rows = []
    for label, d in series.items():
        if label == 'baseline':
            continue
        for ukey, base_ref in [('S', base_s), ('I', base_i)]:
            var = (d[ukey]
                   .set_index('unit_idx')[['v', 'baseline_rank']]
                   .rename(columns={'v': 'v_var', 'baseline_rank': 'rank_base_var'}))

            mg = base_ref.join(var, how='inner')
            n  = len(mg)
            if n < 5:
                continue

            lr = np.log10(mg['v_var'].values) - np.log10(mg['v_base'].values)
            med_lr = float(np.median(lr))
            sd_lr  = float(np.std(lr, ddof=1))
            _, wil_p = _stats.wilcoxon(lr, alternative='two-sided')
            spear_r = float(_stats.spearmanr(mg['v_var'].values, mg['v_base'].values).statistic)

            # variant rank within its own corpus (rank by v_var descending)
            var_ranked = (d[ukey]
                          .sort_values('v', ascending=False)
                          .reset_index(drop=True))
            var_ranked['rank_var'] = np.arange(1, len(var_ranked) + 1)
            rank_map = var_ranked.set_index('unit_idx')['rank_var']
            mg = mg.copy()
            mg['rank_var'] = mg.index.map(rank_map)
            mg = mg.dropna(subset=['rank_var'])
            med_dr = float(np.median(np.abs(mg['rank_var'].values - mg['rank_base'].values)))

            n_base = len(base_ref)
            n_var  = len(d[ukey])
            pct_base = mg['rank_base'].values / n_base
            pct_var  = mg['rank_var'].values  / n_var
            med_rel  = float(np.median(np.abs(pct_var - pct_base)))

            print(f'{label:<24}  {ukey:<5}  {n:>6,}  '
                  f'{spear_r:>8.4f}  {med_lr:>+10.4f}  {sd_lr:>9.4f}  {wil_p:>10.3e}  '
                  f'{med_dr:>11.1f}  {med_rel:>10.4f}')
            rows.append(dict(variant=label, unit_type=ukey, n=n,
                             spearman_r=spear_r,
                             med_log_ratio=med_lr, sd_log_ratio=sd_lr,
                             wilcoxon_p=wil_p, med_abs_delta_rank=med_dr,
                             med_rel_rank_shift=med_rel))

    print('─' * len(hdr))
    print()
    print('Column definitions:')
    print('  Spear ρ    Spearman rank correlation of v_var vs v_base on common units;')
    print('             primary summary of rank-order preservation.')
    print('  med log-r  Median of log(v_var/v_base) across common units;')
    print('             the systematic vertical displacement in log(v).')
    print('  sd log-r   SD of log(v_var/v_base); spread of the vertical displacement,')
    print('             dominated by rank shuffling on the steep part of the v curve.')
    print('  Wil p      Wilcoxon signed-rank p-value (H0: median log-ratio = 0).')
    print('  med|Δrank| Median absolute rank shift between variant and baseline')
    print('             rank within each corpus (units).')
    print('  med|Δpct|  Median absolute rank shift expressed as a fraction of corpus')
    print('             size: |rank_var/n_var − rank_base/n_base|; scale-free.')

    paths = load_config()
    out = paths.plots / 'fig_5_stability.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\nSaved {out}')


# ─── Individual parameter plots ───────────────────────────────────────────────

def _plot5_single(src_rank_map: dict, inst_rank_map: dict,
                  series: dict, keys: list[str],
                  stem: str, title: str) -> None:
    """
    Side-by-side Sources | Institutions for baseline + selected overlays.
    Saves stem.pdf and stem_latex.pdf to plots/.
    """
    paths = load_config()
    sub = {k: series[k] for k in ['baseline'] + keys if k in series}

    _pub_theme()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.subplots_adjust(wspace=0.04)

    _draw_panel(axes[0], sub, 'S', len(src_rank_map),  'Sources',
                ylabel=r'$\log(v)$', show_legend=True)
    _draw_panel(axes[1], sub, 'I', len(inst_rank_map), 'Institutions',
                show_legend=True)
    axes[1].tick_params(labelleft=False)

    sup = fig.suptitle(title, fontsize=9, y=1.01)
    _save(fig, sup, stem, paths)


def plot5a(src_rank_map, inst_rank_map, series):
    _plot5_single(src_rank_map, inst_rank_map, series,
                  keys=['τ=40'],
                  stem='fig_5a',
                  title='Parameter sensitivity: threshold τ=40  (x-axis locked to baseline)')


def plot5b(src_rank_map, inst_rank_map, series):
    _plot5_single(src_rank_map, inst_rank_map, series,
                  keys=['ρ=1'],
                  stem='fig_5b',
                  title='Parameter sensitivity: reference weighting ρ=1  (x-axis locked to baseline)')


def plot5c(src_rank_map, inst_rank_map, series):
    _plot5_single(src_rank_map, inst_rank_map, series,
                  keys=['census=1yr'],
                  stem='fig_5c',
                  title='Parameter sensitivity: census window 1 yr  (x-axis locked to baseline)')


def plot5d(src_rank_map, inst_rank_map, series):
    _plot5_single(src_rank_map, inst_rank_map, series,
                  keys=['α=0.85, μ=1/N_p', 'H_SS/II, no self', 'no self-ref'],
                  stem='fig_5d',
                  title='Parameter sensitivity: Katz–Hubbell α=0.85 — bipartite vs single-kernel, no self-reference  (x-axis locked to baseline)')


def plot5e(src_rank_map, inst_rank_map, series):
    _plot5_single(src_rank_map, inst_rank_map, series,
                  keys=['ω=1'],
                  stem='fig_5e',
                  title='Parameter sensitivity: institution weighting ω=1 (direct 1/N_inst)  (x-axis locked to baseline)')


def plot5f(src_rank_map, inst_rank_map, series):
    _plot5_single(src_rank_map, inst_rank_map, series,
                  keys=['ε=1'],
                  stem='fig_5f',
                  title='Parameter sensitivity: sentinel boundary ε=1  (x-axis locked to baseline)')


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
    report_stability(series)
    plot5a(src_rank_map, inst_rank_map, series)
    plot5b(src_rank_map, inst_rank_map, series)
    plot5c(src_rank_map, inst_rank_map, series)
    plot5d(src_rank_map, inst_rank_map, series)
    plot5e(src_rank_map, inst_rank_map, series)
    plot5f(src_rank_map, inst_rank_map, series)


if __name__ == '__main__':
    main()
    print('FINISHED!')
