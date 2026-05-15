"""
Quick diagnostic: compare raw pi and a_p between baseline and eps=1.

x-axis locked to baseline v rank (descending).
Panels: log pi | a_p | log v  x  sources / institutions.

Prints confirmed Sigma-pi totals (real + sentinel = 1) and a-weighted sums.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')

import duckdb
import numpy as np
import matplotlib.pyplot as plt

from util import load_config, load_runs

_bl = next(r for r in load_runs() if r['label'] == 'baseline')
BASELINE_TABLE = f"rk_{_bl['run_code']}_{_bl['fx']}_tauU{_bl['tau_u']}_tauS{_bl['tau_s']}_vartau_rho0_m0110_chi50_alpha100"
EPS_TABLE      = BASELINE_TABLE + '_eps1'


def load_table(db, table):
    """Return (df_s, df_u) including sentinel rows (unit_idx=1)."""
    df = db.execute(
        f'SELECT unit_idx, unit_type, pi, v, a_p FROM {table}'
    ).df()
    s = df[df['unit_type'] == 'S'].copy()
    u = df[df['unit_type'] == 'U'].copy()
    return s, u


def diag():
    paths   = load_config()
    rk_path = paths.working / 'rankings.duckdb'

    with duckdb.connect(str(rk_path), read_only=True) as db:
        base_s, base_u = load_table(db, BASELINE_TABLE)
        eps_s,  eps_u  = load_table(db, EPS_TABLE)

    # split sentinel from real for eps
    sent_s = eps_s[eps_s['unit_idx'] == 1]
    sent_u = eps_u[eps_u['unit_idx'] == 1]
    real_s = eps_s[eps_s['unit_idx'] != 1].dropna(subset=['pi'])
    real_u = eps_u[eps_u['unit_idx'] != 1].dropna(subset=['pi'])

    # ── Sigma-pi confirmation ─────────────────────────────────────────────────
    pi_s_base = base_s['pi'].sum()
    pi_u_base = base_u['pi'].sum()
    pi_s_real = real_s['pi'].sum()
    pi_u_real = real_u['pi'].sum()
    pi_s_sent = float(sent_s['pi'].iloc[0]) if not sent_s.empty else 0.0
    pi_u_sent = float(sent_u['pi'].iloc[0]) if not sent_u.empty else 0.0

    A_base = base_s['a_p'].sum() + base_u['a_p'].sum()
    A_real = real_s['a_p'].sum() + real_u['a_p'].sum()

    print(f'\n  {"":28}  {"baseline":>10}  {"eps1 real":>10}  {"eps1 sent":>10}  {"eps1 total":>11}')
    print(f'  {"n_sources":<28}  {len(base_s):>10,}  {len(real_s):>10,}')
    print(f'  {"n_insts":<28}  {len(base_u):>10,}  {len(real_u):>10,}')
    print(f'  {"Sigma a_s":<28}  {base_s["a_p"].sum():>10.0f}  {real_s["a_p"].sum():>10.0f}')
    print(f'  {"Sigma a_u":<28}  {base_u["a_p"].sum():>10.0f}  {real_u["a_p"].sum():>10.0f}')
    print(f'  {"A (real)":<28}  {A_base:>10.0f}  {A_real:>10.0f}')
    print()
    print(f'  {"Sigma pi_s":<28}  {pi_s_base:>10.6f}  {pi_s_real:>10.6f}  {pi_s_sent:>10.6f}  {pi_s_real+pi_s_sent:>11.6f}')
    print(f'  {"Sigma pi_u":<28}  {pi_u_base:>10.6f}  {pi_u_real:>10.6f}  {pi_u_sent:>10.6f}  {pi_u_real+pi_u_sent:>11.6f}')
    total_base = pi_s_base + pi_u_base
    total_eps  = pi_s_real + pi_u_real + pi_s_sent + pi_u_sent
    print(f'  {"Sigma pi (S+U)":<28}  {total_base:>10.6f}  {"":>10}  {"":>10}  {total_eps:>11.6f}')
    print()
    # a-weighted mean of v over real units (should be ~1 after fix)
    v_base_real = base_s['v'].fillna(0)  # baseline has no NaN
    aw_v_base = (base_s['a_p'].values @ base_s['v'].values +
                 base_u['a_p'].values @ base_u['v'].values) / A_base
    real_s_v = real_s.dropna(subset=['v'])
    real_u_v = real_u.dropna(subset=['v'])
    aw_v_eps = (real_s_v['a_p'].values @ real_s_v['v'].values +
                real_u_v['a_p'].values @ real_u_v['v'].values) / A_real
    print(f'  {"Sigma(a*v)/A  [real units]":<28}  {aw_v_base:>10.6f}  {aw_v_eps:>10.6f}')

    # ── Merge on common unit_idx, sort by baseline v rank ────────────────────
    def merge_by_v(bdf, edf):
        m = (bdf[['unit_idx', 'pi', 'a_p', 'v']]
             .merge(edf[['unit_idx', 'pi', 'a_p', 'v']],
                    on='unit_idx', suffixes=('_base', '_eps'))
             .sort_values('v_base', ascending=False)
             .reset_index(drop=True))
        return m

    ms = merge_by_v(base_s, real_s)
    mu = merge_by_v(base_u, real_u)
    rk_s = np.arange(1, len(ms) + 1)
    rk_u = np.arange(1, len(mu) + 1)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    for row, (m, rk, utype) in enumerate(
            [(ms, rk_s, 'Source'), (mu, rk_u, 'Institution')]):

        ax_pi, ax_ap, ax_v = axes[row]

        ax_pi.scatter(rk, np.log10(np.clip(m['pi_base'], 1e-15, None)),
                      s=4, alpha=0.5, color='black',   label='baseline', zorder=3)
        ax_pi.scatter(rk, np.log10(np.clip(m['pi_eps'],  1e-15, None)),
                      s=4, alpha=0.5, color='#984ea3',  label='eps1',    zorder=2)
        ax_pi.set_title(f'{utype}: log pi')
        ax_pi.set_xlabel('Baseline v rank')
        ax_pi.set_ylabel('log10 pi')
        ax_pi.legend(fontsize=8)

        ax_ap.scatter(rk, m['a_p_base'], s=4, alpha=0.5, color='black',   label='baseline', zorder=3)
        ax_ap.scatter(rk, m['a_p_eps'],  s=4, alpha=0.5, color='#984ea3',  label='eps1',    zorder=2)
        ax_ap.set_title(f'{utype}: a_p')
        ax_ap.set_xlabel('Baseline v rank')
        ax_ap.set_ylabel('a_p')
        ax_ap.legend(fontsize=8)

        ax_v.scatter(rk, np.log10(np.clip(m['v_base'], 1e-10, None)),
                     s=4, alpha=0.5, color='black',   label='baseline', zorder=3)
        ax_v.scatter(rk, np.log10(np.clip(m['v_eps'],  1e-10, None)),
                     s=4, alpha=0.5, color='#984ea3',  label='eps1',    zorder=2)
        ax_v.set_title(f'{utype}: log v')
        ax_v.set_xlabel('Baseline v rank')
        ax_v.set_ylabel('log10 v')
        ax_v.legend(fontsize=8)

    fig.suptitle(
        'Diagnostic: baseline vs eps=1 -- pi, a_p, v  (x-axis = baseline v rank)',
        fontsize=10)

    out = paths.plots / 'diag_epsilon_pi.pdf'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    diag()
