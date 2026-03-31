"""
fig_4.py — Community structure (φ₂/√v) vs prestige (v).

Four panels arranged 2×2:
  Row 0 — Sources:      SS (m=1000)  |  Bipartite (m=0110)
  Row 1 — Institutions: SS (m=0110, φ₂_I recovered from H_SI^T φ₂_S)
                        |  Bipartite (same run, same φ₂_I)

y-axis: φ₂/√v.  Dividing by √v approximately removes the eigenvector
normalisation artefact that shrinks |φ₂| for low-prestige units, giving
a per-unit community identity measure that is stable across the prestige range.

x-axis: v from the bipartite baseline (m=0110, α=1, F=A).
Coloured by field subset F ∈ {E, B} for sources; institutions have no
field label so the I panels use a single colour.

Outputs:
  plots/fig_4.pdf        — with title (exploration)
  plots/fig_4_latex.pdf  — without title (paper)
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_params
from spectral_ranking.build_csr import build_csr
from spectral_ranking.katz_ranker import _row_normalise

_p     = load_params()
_tau_u = _p['tau_u_floor']['A']
_tau_s = _p['tau_s_floor']['A']

BASELINE_TABLE = f'rk_t5_A_tauU{_tau_u}_tauS{_tau_s}_rho0_m0110_chi50_alpha100'

COLOR  = {'E': '#1f77b4', 'B': '#d62728', 'other': '#999999'}
MARKER = {'E': 'o',       'B': 's',       'other': 'x'}
LABEL  = {'E': 'Economics (E)', 'B': 'Business (B)', 'other': 'Other'}


# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_baseline_v(db) -> tuple:
    """Load v for sources and institutions from the bipartite baseline."""
    df_s = db.execute(
        f"SELECT unit_idx, v FROM {BASELINE_TABLE} WHERE unit_type='S'"
    ).df().rename(columns={'unit_idx': 'source_idx'})
    df_u = db.execute(
        f"SELECT unit_idx, v FROM {BASELINE_TABLE} WHERE unit_type='U'"
    ).df().rename(columns={'unit_idx': 'inst_idx'})
    return df_s, df_u


def load_field_labels(paths) -> tuple:
    sm = pd.read_csv(paths.data / 'source_master.csv',
                     usecols=['source_idx', 'field_subset', 'source_name'])
    sm = sm.dropna(subset=['field_subset'])
    field_labels = dict(zip(sm['source_idx'].astype(int), sm['field_subset']))
    source_names = dict(zip(sm['source_idx'].astype(int), sm['source_name']))
    return field_labels, source_names


def load_inst_field_labels(el_db, tau_u, tau_s) -> dict:
    """
    Return {inst_idx (int): 'E' | 'B' | 'EB'} by checking which field-subset
    unit tables the institution appears in.

    F=E and F=B run on distinct networks; an institution can be in both.
    'EB'   → present in both E and B networks
    'E'    → present in E network only
    'B'    → present in B network only
    'other'→ present in A network only (not in E or B sub-networks)
    """
    def inst_set(fx):
        tname = f'_units_t5_{fx}_tauU{tau_u}_tauS{tau_s}'
        tables = {r[0] for r in el_db.execute('SHOW TABLES').fetchall()}
        if tname not in tables:
            return set()
        rows = el_db.execute(
            f"SELECT unit_idx FROM {tname} WHERE unit_type='U'"
        ).fetchall()
        return {int(r[0]) for r in rows}

    e_set = inst_set('E')
    b_set = inst_set('B')

    inst_field = {}
    for idx in e_set | b_set:
        if idx in e_set and idx in b_set:
            inst_field[idx] = 'other'   # present in both; not exclusively E or B
        elif idx in e_set:
            inst_field[idx] = 'E'
        else:
            inst_field[idx] = 'B'
    return inst_field


# ─── Eigenpair computation ────────────────────────────────────────────────────

def _eigenpair2(M, alpha_eff):
    """
    λ₂ and φ₂ of (α_eff·M^T + (1−α_eff)·μ·1^T).
    Sign: max |φ₂| element is positive.
    """
    from scipy.sparse.linalg import eigs, LinearOperator
    N = M.shape[0]
    mu = np.full(N, 1.0 / N)

    def matvec(x):
        return alpha_eff * M.T.dot(x) + (1.0 - alpha_eff) * mu * x.sum()
    Op = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    vals, vecs = eigs(Op, k=2, which='LM')
    order = np.argsort(-vals.real)
    lam2 = vals[order[1]].real
    phi2 = vecs[:, order[1]].real
    if phi2[np.argmax(np.abs(phi2))] < 0:
        phi2 = -phi2
    return lam2, phi2


def compute_eigenpairs(db, tx, fx, tau_u, tau_s, rho, alpha):
    """
    Compute φ₂ for all four panels.

    Returns
    -------
    lam2_ss  : float
    lam2_ii  : float
    lam2_bi  : float
    df_ss_s  : DataFrame  source_idx, phi2  (from H_SS)
    df_ii_i  : DataFrame  inst_idx,   phi2  (from H_II)
    df_bi_s  : DataFrame  source_idx, phi2  (from M_S, round-trip α²)
    df_bi_i  : DataFrame  inst_idx,   phi2  (H_SI^T φ₂_S, unit-norm)
    """
    # SS
    csr_ss = build_csr(db, tx, fx, tau_u, tau_s, rho, (1, 0, 0, 0))
    H_ss, _ = _row_normalise(csr_ss.C_SS)
    lam2_ss, phi2_ss = _eigenpair2(H_ss, alpha)
    df_ss_s = pd.DataFrame({'source_idx': csr_ss.source_ids.astype(int),
                            'phi2': phi2_ss})

    # II
    csr_ii = build_csr(db, tx, fx, tau_u, tau_s, rho, (0, 0, 0, 1))
    H_ii, _ = _row_normalise(csr_ii.C_II)
    lam2_ii, phi2_ii = _eigenpair2(H_ii, alpha)
    df_ii_i = pd.DataFrame({'inst_idx': csr_ii.inst_ids.astype(int),
                            'phi2': phi2_ii})

    # Bipartite
    csr_bi = build_csr(db, tx, fx, tau_u, tau_s, rho, (0, 1, 1, 0))
    H_SI, _ = _row_normalise(csr_bi.C_SI)
    H_IS, _ = _row_normalise(csr_bi.C_IS)
    M_S = H_SI.dot(H_IS)
    lam2_bi, phi2_bi_s = _eigenpair2(M_S, alpha ** 2)
    df_bi_s = pd.DataFrame({'source_idx': csr_bi.source_ids.astype(int),
                            'phi2': phi2_bi_s})

    # Recover institution φ₂ from bipartite: φ₂_I = H_SI^T φ₂_S, unit-normed
    phi2_bi_i = H_SI.T.dot(phi2_bi_s)
    norm = np.linalg.norm(phi2_bi_i)
    if norm > 0:
        phi2_bi_i /= norm
    if phi2_bi_i[np.argmax(np.abs(phi2_bi_i))] < 0:
        phi2_bi_i = -phi2_bi_i
    df_bi_i = pd.DataFrame({'inst_idx': csr_bi.inst_ids.astype(int),
                            'phi2': phi2_bi_i})

    return lam2_ss, lam2_ii, lam2_bi, df_ss_s, df_ii_i, df_bi_s, df_bi_i


# ─── Plot ─────────────────────────────────────────────────────────────────────

def _draw_panel(ax, df, v_col, phi2_col, id_col, group_map, panel_title):
    """Scatter φ₂/√v vs v (log x-axis), coloured by group."""
    df = df.copy()
    df['v'] = df[v_col]
    df['phi2'] = df[phi2_col]
    df['score'] = df['phi2'] / np.sqrt(df['v'].clip(lower=1e-12))
    df['grp'] = df[id_col].map(group_map).fillna('other')

    _GRP_STYLE = {
        'other': dict(alpha=0.80, zorder=2, linewidths=2.0),
        'E':     dict(alpha=1.0,  zorder=4, linewidths=0.4),
        'B':     dict(alpha=1.0,  zorder=3, linewidths=0.4),
    }
    for grp in ['other', 'B', 'E']:   # Other first so E/B paint over it
        sub = df[df['grp'] == grp]
        if sub.empty:
            continue
        ax.scatter(
            sub['v'].values,
            sub['score'].values,
            c=COLOR[grp],
            marker=MARKER[grp],
            s=18,
            label=f"{LABEL[grp]}  (n={len(sub):,})",
            **_GRP_STYLE[grp],
        )

    ax.axhline(0, color='#aaaaaa', linewidth=0.6, zorder=1)

    ax.set_xscale('log')
    ax.set_xlabel('$v$  (bipartite baseline)', labelpad=4)
    ax.set_ylabel('$\\phi_2 / \\sqrt{v}$', labelpad=4)
    ax.legend(title=panel_title, fontsize=7.5, title_fontsize=8.5,
              framealpha=0.85, loc='upper left', markerscale=1.4)


def plot4(lam2_ss, lam2_ii, lam2_bi,
          df_ss_s, df_ii_i, df_bi_s, df_bi_i,
          df_v_s, df_v_u,
          field_labels, inst_field_labels):
    paths = load_config()
    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.06, wspace=0.06)

    # Merge v into φ₂ frames
    ms_s  = df_ss_s.merge(df_v_s, on='source_idx', how='inner')
    mii_i = df_ii_i.merge(df_v_u, on='inst_idx',   how='inner')
    mb_s  = df_bi_s.merge(df_v_s, on='source_idx', how='inner')
    mb_i  = df_bi_i.merge(df_v_u, on='inst_idx',   how='inner')

    # Shared axis limits
    x_min = 0.01
    x_max = float(np.nanmax(np.concatenate([
        ms_s['v'].values, mii_i['v'].values,
        mb_s['v'].values, mb_i['v'].values,
    ])))
    y_min = -0.2
    y_max =  0.2

    # Report all units outside y bounds (all v, not filtered by x_min)
    def _report_outside_y(label, df, v_col, phi2_col, id_col):
        v      = df[v_col].values.clip(1e-12)
        scores = df[phi2_col].values / np.sqrt(v)
        mask   = (scores < y_min) | (scores > y_max)
        if not mask.any():
            return
        out = df[mask].copy()
        out['score'] = scores[mask]
        out = out.sort_values('score')
        print(f"  {label}: {mask.sum()} units outside [{y_min}, {y_max}]")
        for _, row in out.iterrows():
            print(f"    {id_col}={int(row[id_col])}  v={row[v_col]:.4f}  φ₂/√v={row['score']:.4f}")

    print("\nUnits outside y-axis bounds:")
    _report_outside_y('Sources SS',        ms_s,  'v', 'phi2', 'source_idx')
    _report_outside_y('Sources Bipartite', mb_s,  'v', 'phi2', 'source_idx')
    _report_outside_y('Institutions II',   mii_i, 'v', 'phi2', 'inst_idx')
    _report_outside_y('Institutions Bip',  mb_i,  'v', 'phi2', 'inst_idx')

    # Row 0 — Sources
    _draw_panel(axes[0, 0], ms_s, 'v', 'phi2', 'source_idx',
                field_labels, 'Sources — SS')
    _draw_panel(axes[0, 1], mb_s, 'v', 'phi2', 'source_idx',
                field_labels, 'Sources — Bipartite')

    # Row 1 — Institutions coloured by field-network membership (E / B / EB)
    _draw_panel(axes[1, 0], mii_i, 'v', 'phi2', 'inst_idx',
                inst_field_labels, 'Institutions — II')
    _draw_panel(axes[1, 1], mb_i, 'v', 'phi2', 'inst_idx',
                inst_field_labels, 'Institutions — Bipartite')

    # Apply shared axes to all panels
    for ax in axes.flat:
        ax.set_xlim(x_min * 0.9, x_max * 1.1)
        ax.set_ylim(y_min, y_max)

    # Remove redundant axis labels and tick labels on shared edges
    for ax in axes[0, :]:          # upper row: no x-axis
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)
    for ax in axes[:, 1]:          # right column: no y-axis
        ax.set_ylabel('')
        ax.tick_params(labelleft=False)

    sup = fig.suptitle(
        'Community identity $\\phi_2/\\sqrt{v}$ vs prestige $v$',
        fontsize=10, y=1.01,
    )

    out = paths.plots / 'fig_4.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

    sup.set_visible(False)
    latex_out = paths.plots / 'fig_4_latex.pdf'
    fig.savefig(latex_out, bbox_inches='tight')
    print(f'Saved {latex_out}')
    sup.set_visible(True)
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    paths  = load_config()
    params = load_params()

    tau_u = params['tau_u_floor']['A']
    tau_s = params['tau_s_floor']['A']
    alpha = 1.0
    tx, fx, rho = 5, 'A', 0

    field_labels, _ = load_field_labels(paths)
    print(f"Field labels: {sum(v=='E' for v in field_labels.values())} E, "
          f"{sum(v=='B' for v in field_labels.values())} B")

    rk_path = paths.working / 'rankings.duckdb'
    el_path = paths.working / 'edge_lists.duckdb'

    with duckdb.connect(str(rk_path), read_only=True) as db:
        df_v_s, df_v_u = load_baseline_v(db)
    print(f"Baseline v: {len(df_v_s):,} sources, {len(df_v_u):,} institutions")

    with duckdb.connect(str(el_path), read_only=True) as db:
        print("Computing eigenpairs ...")
        lam2_ss, lam2_ii, lam2_bi, df_ss_s, df_ii_i, df_bi_s, df_bi_i = \
            compute_eigenpairs(db, tx, fx, tau_u, tau_s, rho, alpha)
        print("Loading institution field-network membership ...")
        inst_field_labels = load_inst_field_labels(db, tau_u, tau_s)
    print(f"  λ₂(SS)={lam2_ss:.4f}  λ₂(II)={lam2_ii:.4f}  λ₂(bipartite)={lam2_bi:.4f}")
    e_only = sum(v == 'E'     for v in inst_field_labels.values())
    b_only = sum(v == 'B'     for v in inst_field_labels.values())
    other  = sum(v == 'other' for v in inst_field_labels.values())
    print(f"  Institutions: E={e_only}  B={b_only}  both/other={other}")

    plot4(lam2_ss, lam2_ii, lam2_bi,
          df_ss_s, df_ii_i, df_bi_s, df_bi_i,
          df_v_s, df_v_u,
          field_labels, inst_field_labels)
    print('FINISHED!')


if __name__ == '__main__':
    main()
