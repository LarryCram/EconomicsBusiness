"""
fig_boot_envelope.py — Bootstrap rank-uncertainty envelope.

For each unit, the bootstrap produces B fractional ranks (rank / SCC size per
replicate; NaN units assigned fractional rank 1.0).  Plotting the 2.5th /
50th / 97.5th percentile of those fractional ranks against the unit's baseline
fractional rank gives an envelope whose width is a direct proxy for the
derivative of the v-distribution:

  • Top tier (steep v curve): scores are well-separated → small rank swaps
    → envelope pinches tightly around the diagonal.
  • Bulk (flat v curve): scores are densely packed → many rank swaps per
    unit of v → envelope blows out wide.

Uses the 'all combined' stage4 bootstrap (stage4/).

Outputs
-------
  plots/fig_boot_envelope.pdf / _latex.pdf   — Sources | Institutions
  plots/fig_boot_envelope_S.pdf / _latex.pdf — Sources only (wider)
  plots/fig_boot_envelope_I.pdf / _latex.pdf — Institutions only (wider)
"""

import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

_LOG_LO, _LOG_HI = -2.35, 1.35


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


def _fractional_ranks(v_boot: np.ndarray) -> np.ndarray:
    """
    Convert (B, n) v-value matrix to (B, n) fractional-rank matrix.

    Within each replicate, rank non-NaN units by v descending (rank 1 = best).
    Fractional rank = rank / n_valid  (so rank-1 unit → 1/n_valid ≈ 0).
    NaN units (outside SCC) remain NaN — excluded from quantile computation.
    """
    B, n = v_boot.shape
    frac = np.full((B, n), np.nan, dtype=np.float32)
    for b in range(B):
        row  = v_boot[b]
        pos  = np.where(~np.isnan(row))[0]
        n_v  = len(pos)
        if n_v == 0:
            continue
        # sorted_idx[0] is the index (within pos) of the best unit
        sorted_idx = np.argsort(-row[pos])
        rank_of_k  = np.empty(n_v, dtype=np.float32)
        rank_of_k[sorted_idx] = (np.arange(n_v, dtype=np.float32) + 1) / n_v
        frac[b, pos] = rank_of_k
    return frac


def _envelope(frac_ranks: np.ndarray, baseline_order: np.ndarray,
              q_lo: float = 0.025, q_hi: float = 0.975) -> tuple:
    """
    For each unit (sorted by baseline rank), compute percentiles of its
    bootstrap fractional ranks.

    Parameters
    ----------
    frac_ranks    : (B, n) fractional ranks from _fractional_ranks
    baseline_order: (n,) indices that sort units by baseline rank ascending
                    (i.e. baseline_order[0] is the index of rank-1 unit)

    Returns x, y_lo, y_med, y_hi — all length n, x in (0,1].
    """
    n = len(baseline_order)
    x     = (np.arange(1, n + 1, dtype=np.float32)) / n
    y_lo  = np.empty(n, dtype=np.float32)
    y_med = np.empty(n, dtype=np.float32)
    y_hi  = np.empty(n, dtype=np.float32)

    for rank_pos, unit_i in enumerate(baseline_order):
        col = frac_ranks[:, unit_i]
        y_lo[rank_pos]  = np.nanquantile(col, q_lo)
        y_med[rank_pos] = np.nanquantile(col, 0.5)
        y_hi[rank_pos]  = np.nanquantile(col, q_hi)

    return x, y_lo, y_med, y_hi


def _draw_panel(ax, x, y_lo, y_med, y_hi, title, ylabel='', n_units=None, error_label=''):
    ax.fill_between(x, y_lo, y_hi,
                    color='#e41a1c', alpha=0.45, linewidth=0,
                    label=f'2.5–97.5 pct ({error_label})')
    ax.plot(x, y_med, color='#377eb8', linewidth=0.8,
            label=f'median ({error_label})')
    ax.plot([0, 1], [0, 1], color='black', linewidth=0.9,
            linestyle='--', zorder=5, label='diagonal')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Baseline fractional rank  (0 = best)', labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=4)
    label_n = f'  (n={n_units:,})' if n_units else ''
    ax.set_title(title + label_n, fontsize=11, pad=6)
    ax.legend(fontsize=10, framealpha=0.85, loc='upper left')


STAGES = [
    ('stage4_resample',    'resample'),
    ('stage4_year',        'year errors'),
    ('stage4_source',      'source errors'),
    ('stage4_institution', 'institution errors'),
    ('stage4_reference',   'reference errors'),
    ('stage4',             'all errors combined'),
]


def compute_envelopes(paths, stage: str) -> dict:
    """
    Load bootstrap data for one stage directory and return envelope arrays
    for Sources (key 'S') and Institutions (key 'I').
    """
    bl = next(r for r in load_runs() if r['label'] == 'baseline')
    BASELINE_TABLE = (f"rk_{bl['run_code']}_{bl['fx']}"
                      f"_tauU{bl['tau_u']}_tauS{bl['tau_s']}"
                      f"_vartau_rho0_m0110_chi50_alpha100_omega1")

    boot_dir = paths.working / 'bootstrap_oa_errors' / stage
    print(f'  Loading {stage} ...', flush=True)
    v_s = np.load(str(boot_dir / 'v_s_boot.npy')).astype(np.float32)
    v_u = np.load(str(boot_dir / 'v_u_boot.npy')).astype(np.float32)
    with open(boot_dir / 'meta.json') as f:
        meta = json.load(f)
    B = meta.get('completed', v_s.shape[0])
    v_s = v_s[:B]; v_u = v_u[:B]
    print(f'    B={B}  n_s={v_s.shape[1]}  n_u={v_u.shape[1]}')

    with duckdb.connect(str(paths.working / 'rankings.duckdb'), read_only=True) as db:
        df = db.execute(
            f'SELECT unit_idx, unit_type, v FROM {BASELINE_TABLE}'
        ).df()

    out = {}
    for unit_type, v_boot, id_list, panel_label in [
        ('S', v_s, meta['source_ids'],  'Sources'),
        ('U', v_u, meta['inst_ids'],    'Institutions'),
    ]:
        base = df[df['unit_type'] == unit_type].set_index('unit_idx')['v']
        ids  = np.array(id_list, dtype=int)
        mask = np.array([uid in base.index for uid in ids])
        idx  = np.where(mask)[0]
        v_base = base.loc[ids[mask]].values.astype(np.float32)
        baseline_order = np.argsort(-v_base)
        frac = _fractional_ranks(v_boot[:, idx])
        x, y_lo, y_med, y_hi = _envelope(frac, baseline_order)
        key = 'I' if unit_type == 'U' else unit_type
        out[key] = dict(x=x, y_lo=y_lo, y_med=y_med, y_hi=y_hi, n=len(v_base))

    return out


def plot_envelope(env: dict, error_label: str, stem: str, paths) -> None:
    _pub_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig.subplots_adjust(wspace=0.05)

    _draw_panel(axes[0], **{k: env['S'][k] for k in ('x','y_lo','y_med','y_hi')},
                title='Sources', ylabel='Bootstrap fractional rank',
                n_units=env['S']['n'], error_label=error_label)
    _draw_panel(axes[1], **{k: env['I'][k] for k in ('x','y_lo','y_med','y_hi')},
                title='Institutions', n_units=env['I']['n'], error_label=error_label)
    axes[1].tick_params(labelleft=False)
    axes[0].xaxis.get_major_ticks()[-1].label1.set_visible(False)

    sup = fig.suptitle(
        f'Bootstrap rank-uncertainty envelope — {error_label}  '
        f'(shaded band = 2.5–97.5 pct, 1,000 replicates)',
        fontsize=9, y=1.01,
    )
    _save(fig, sup, stem, paths)


def main():
    paths = load_config()
    for stage_dir, error_label in STAGES:
        boot_dir = paths.working / 'bootstrap_oa_errors' / stage_dir
        if not (boot_dir / 'v_s_boot.npy').exists():
            print(f'  Skipping {stage_dir} — not found')
            continue
        env  = compute_envelopes(paths, stage_dir)
        stem = f'fig_boot_envelope_{stage_dir.replace("stage4", "").strip("_") or "all"}'
        plot_envelope(env, error_label, stem, paths)


if __name__ == '__main__':
    main()
    print('FINISHED!')
