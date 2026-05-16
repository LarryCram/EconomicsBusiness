"""
fig_7.py — Bootstrap uncertainty in baseline spectral ranking.

Loads B bootstrap replicates of v_s and v_u from $WORKING/bootstrap/,
plots them as a scatter cloud against baseline rank alongside the baseline
curve and 5th–95th percentile bands.

Two-panel layout (top = sources, bottom = institutions), matching fig_2/fig_3.

Source colouring by field_eb:
  'E'   → red
  'B'   → blue
  'A'   → orange
  NULL  → grey

Outputs:
  plots/fig_7.pdf        — with title (exploration)
  plots/fig_7_latex.pdf  — without title (paper)
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

_baseline      = next(r for r in load_runs() if r['label'] == 'baseline')
_run_code      = _baseline['run_code']
_tau_u         = _baseline['tau_u']
_tau_s         = _baseline['tau_s']
_fx            = _baseline['fx']
BASELINE_TABLE         = f'rk_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m0110_chi50_alpha100'
BASELINE_DIRECT_TABLE  = f'rk_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m0110_chi50_alpha100_omega1'
EL_TABLE               = f'el_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau'

FIELD_COLOURS = {
    'E':   '#e41a1c',   # red
    'B':   '#377eb8',   # blue
    'A':   '#ff7f0e',   # orange
    None:  '#aaaaaa',   # grey
}

INST_FIELD_COLOURS = {
    'E': '#e41a1c',   # red
    'B': '#377eb8',   # blue
    'A': '#ff7f0e',   # orange — matches source A (mixed/middle)
    'X': '#aaaaaa',   # grey
}



# ─── Data ─────────────────────────────────────────────────────────────────────

def load_bootstrap(boot_dir: Path) -> tuple:
    """Load v_s_boot, v_u_boot, lam_ratio_boot arrays and meta from boot_dir."""
    v_s_boot = np.load(boot_dir / 'v_s_boot.npy')   # (B, n_s)
    v_u_boot = np.load(boot_dir / 'v_u_boot.npy')   # (B, n_u)
    lam_path = boot_dir / 'lam_ratio_boot.npy'
    lam_ratio_boot = np.load(lam_path) if lam_path.exists() else None
    with open(boot_dir / 'meta.json') as f:
        meta = json.load(f)
    completed = meta.get('completed', v_s_boot.shape[0])
    v_s_boot = v_s_boot[:completed]
    v_u_boot = v_u_boot[:completed]
    if lam_ratio_boot is not None:
        lam_ratio_boot = lam_ratio_boot[:completed]
    return v_s_boot, v_u_boot, lam_ratio_boot, meta


def load_baseline(rk_path: Path, table: str = BASELINE_TABLE) -> pd.DataFrame:
    """Load baseline v and unit_type from rankings.duckdb."""
    with duckdb.connect(str(rk_path), read_only=True) as db:
        df = db.execute(
            f"SELECT unit_idx, unit_type, v FROM {table}"
        ).df()
    return df


def load_field_labels(paths) -> tuple:
    """Return ({source_idx: field_eb}, {source_idx: source_name})."""
    sm = pd.read_csv(paths.data / 'source_master.csv',
                     usecols=['source_idx', 'field_eb', 'source_name'])
    idx = sm['source_idx'].astype(int)
    return (dict(zip(idx, sm['field_eb'])),
            dict(zip(idx, sm['source_name'])))


def fetch_inst_field_labels(parquet_path: Path) -> dict:
    """Read institution E/B/A/X labels from institution_field_eb.parquet."""
    df = pd.read_parquet(str(parquet_path / 'institution_field_eb.parquet'),
                         columns=['unit_idx', 'field_eb'])
    return dict(zip(df['unit_idx'].astype(int), df['field_eb']))


def build_panel_data(v_boot: np.ndarray,
                     baseline_v: np.ndarray,
                     unit_ids: np.ndarray,
                     meta_ids: list) -> tuple:
    """
    Align bootstrap array columns with baseline units via meta_ids.

    Parameters
    ----------
    v_boot      : (B, N_boot) float32 — bootstrap replicates
    baseline_v  : (n_base,) float64   — baseline v, ordered by baseline_rank
    unit_ids    : (n_base,) int64     — unit_idx for each baseline unit
    meta_ids    : list of int         — dense-index → unit_idx from meta.json

    Returns
    -------
    baseline_rank  : (n_base,) int   — 1..n_base
    v_base_ordered : (n_base,)       — baseline v sorted by rank
    boot_flat      : (B * n_aligned,) — flattened bootstrap v values
    rank_flat      : (B * n_aligned,) — repeated baseline ranks
    n_aligned      : int             — number of units present in both
    """
    meta_idx = pd.Index(meta_ids)
    dense_pos = meta_idx.get_indexer(unit_ids)
    present   = dense_pos >= 0

    unit_ids_aligned  = unit_ids[present]
    dense_pos_aligned = dense_pos[present]
    n_aligned = len(unit_ids_aligned)

    # Sort by descending baseline v to assign ranks
    v_base_aligned = baseline_v[present]
    order          = np.argsort(-v_base_aligned)
    ranks          = np.arange(1, n_aligned + 1)

    v_base_ranked  = v_base_aligned[order]
    dense_ranked   = dense_pos_aligned[order]

    # Bootstrap values for aligned units in rank order: shape (B, n_aligned)
    boot_ranked = v_boot[:, dense_ranked]   # (B, n_aligned)

    # Flatten: B copies of each rank
    rank_flat = np.tile(ranks, v_boot.shape[0])            # (B * n_aligned,)
    boot_flat = boot_ranked.ravel(order='C')               # (B * n_aligned,)

    return ranks, v_base_ranked, boot_flat, rank_flat, n_aligned, dense_ranked, order


# ─── Plot ─────────────────────────────────────────────────────────────────────

def _draw_panel(ax, ranks, v_base, boot_flat, rank_flat,
                n_aligned, B, panel_title,
                colours=None, show_ylabel=True) -> None:
    """Draw one panel: scatter + blue min/max dots + running mean + baseline.

    y-axis is linear log10(v): ylim=(_LOG_LO, _LOG_HI), yticks=[-1, 0, 1].
    """
    lv_base = np.log10(np.clip(v_base, 1e-10, None))
    lv_boot = np.log10(np.clip(boot_flat, 1e-10, None))

    ax.scatter(rank_flat, lv_boot,
               c=colours if colours is not None else '#aaaaaa',
               s=1, alpha=0.04, linewidths=0, rasterized=True, zorder=1)

    boot_2d_log = lv_boot.reshape(B, n_aligned)
    ax.scatter(ranks, boot_2d_log.max(axis=0),
               s=4, c=_BLUE, linewidths=0, zorder=6)
    ax.scatter(ranks, boot_2d_log.min(axis=0),
               s=4, c=_BLUE, linewidths=0, zorder=6)

    # Running geometric mean of per-unit median, plotted in log space
    med_v = np.nanmedian(boot_flat.reshape(B, n_aligned), axis=0)
    lv_running = np.log10(np.maximum(_boot_running_mean(med_v), 1e-10))
    ax.plot(ranks, lv_running,
            color='#e41a1c', linewidth=1.4, alpha=0.9, zorder=4)

    ax.plot(ranks, lv_base,
            color='black', linewidth=1.4, alpha=1.0, zorder=5)

    ax.set_ylim(_LOG_LO, _LOG_HI)
    ax.set_yticks(_LOG_TICKS)
    ax.set_xlim(1, n_aligned)
    ax.axhline(0.0, color='#999999', linewidth=0.8, linestyle='--', zorder=0)
    ax.text(n_aligned * 0.98, 0.0, '$v=1$',
            ha='right', va='bottom', fontsize=7.5, color='#999999')
    ax.set_xlabel('Baseline rank', labelpad=4)
    ax.set_title(panel_title, fontsize=10, pad=6)

    if show_ylabel:
        ax.set_ylabel(r'$\log(v)$', labelpad=4)
    else:
        ax.tick_params(labelleft=False)


Y_LO, Y_HI = 0.02, 20.0

_LOG_LO, _LOG_HI = -2.35, 1.35
_LOG_TICKS = [-1, 0, 1]
_BLUE = '#377eb8'

OA_ERROR_TYPES  = ['year', 'source', 'institution', 'reference', 'resample']
OA_ERROR_LABELS = {
    'year':        'Year',
    'source':      'Source',
    'institution': 'Institution',
    'reference':   'Reference',
    'resample':    'Sampling',
}


def _boot_running_mean(v: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """Centered running geometric mean; window grows as min(rank, frac*n)."""
    log_v = np.log10(np.clip(v, 1e-10, None))
    n = len(log_v)
    w_max = max(3, round(n * frac))
    result = np.empty(n)
    for i in range(n):
        w = min(i + 1, w_max)          # rank-1 at top, caps at frac*n
        lo = max(0, i - w // 2)
        hi = min(n, i + w // 2 + 1)
        result[i] = log_v[lo:hi].mean()
    return np.power(10.0, result)


def _draw_cell(ax, ranks, v_base, boot_flat, rank_flat,
               n_aligned, B,
               show_xlabel: bool = True,
               show_ylabel: bool = False,
               unit_tag: str = '') -> None:
    """Draw one facet cell: scatter cloud + blue min/max dots + baseline.

    y-axis is linear log10(v): ylim=(_LOG_LO, _LOG_HI), yticks=[-1, 0, 1].
    Blue dots mark the per-rank maximum and minimum across all B replicates.
    """
    lv_base = np.log10(np.clip(v_base, 1e-10, None))
    lv_boot = np.log10(np.clip(boot_flat, 1e-10, None))

    ax.scatter(rank_flat, lv_boot,
               c='#e41a1c', s=1, alpha=0.04, linewidths=0, rasterized=True, zorder=1)

    boot_2d_log = lv_boot.reshape(B, n_aligned)
    ax.scatter(ranks, boot_2d_log.max(axis=0),
               s=4, c=_BLUE, linewidths=0, zorder=6)
    ax.scatter(ranks, boot_2d_log.min(axis=0),
               s=4, c=_BLUE, linewidths=0, zorder=6)

    ax.plot(ranks, lv_base, color='black', linewidth=1.2, alpha=1.0, zorder=5)

    ax.set_ylim(_LOG_LO, _LOG_HI)
    ax.set_yticks(_LOG_TICKS)
    ax.set_xlim(1, n_aligned)
    ax.axhline(0.0, color='#bbbbbb', linewidth=0.7, linestyle='--', zorder=0)

    if show_xlabel:
        ax.set_xlabel('Baseline rank', fontsize=8, labelpad=3)

    if show_ylabel:
        ax.set_ylabel(r'$\log(v)$', fontsize=8, labelpad=3)
    else:
        ax.tick_params(labelleft=False)

    if unit_tag:
        ax.text(0.03, 0.97, unit_tag, transform=ax.transAxes,
                fontsize=7, va='top', ha='left', color='#555555')


def plot7a_oa_errors(paths, df_base) -> None:
    """
    2-row × 4-column facet: rows = Sources / Institutions,
    columns = year / source / institution / reference error types.
    Shared y-axis.  Omits the 'all' composite run.
    Saves fig7a_oa_errors.pdf and fig7a_oa_errors_latex.pdf.
    """
    boot_base = paths.working / 'bootstrap_oa_errors'

    # Load available error-type runs (skip missing)
    et_data: dict[str, tuple] = {}
    for et in OA_ERROR_TYPES:
        d = boot_base / f'stage4_{et}'
        npy = d / 'v_s_boot.npy'
        if not npy.exists():
            print(f'  Missing {d} — skipping', flush=True)
            continue
        et_data[et] = load_bootstrap(d)
        B = et_data[et][0].shape[0]
        print(f'  {et}: B={B}', flush=True)

    if not et_data:
        print('No per-error-type bootstrap outputs found; skipping fig7a_oa_errors.')
        return

    col_keys = [et for et in OA_ERROR_TYPES if et in et_data]
    n_cols   = len(col_keys)

    df_s     = df_base[df_base['unit_type'] == 'S'].copy()
    df_i     = df_base[df_base['unit_type'] == 'U'].copy()
    src_ids  = df_s['unit_idx'].to_numpy(dtype=np.int64)
    src_v    = df_s['v'].to_numpy(dtype=np.float64)
    inst_ids = df_i['unit_idx'].to_numpy(dtype=np.int64)
    inst_v   = df_i['v'].to_numpy(dtype=np.float64)

    sns.set_theme(style='whitegrid', font_scale=0.85)
    fig, axes = plt.subplots(
        1, n_cols * 2,
        figsize=(2.6 * n_cols * 2, 3.2),
        sharey=True, sharex=False,
    )
    if n_cols * 2 == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.05)

    for col, et in enumerate(col_keys):
        v_s_boot, v_u_boot, _, meta = et_data[et]
        B = v_s_boot.shape[0]

        ranks_s, v_base_s, boot_flat_s, rank_flat_s, n_s, *_ = \
            build_panel_data(v_s_boot, src_v, src_ids, meta['source_ids'])
        ranks_i, v_base_i, boot_flat_i, rank_flat_i, n_i, *_ = \
            build_panel_data(v_u_boot, inst_v, inst_ids, meta['inst_ids'])

        ax_s = axes[2 * col]
        ax_i = axes[2 * col + 1]

        _draw_cell(ax_s, ranks_s, v_base_s, boot_flat_s, rank_flat_s, n_s, B,
                   show_xlabel=True, show_ylabel=(col == 0), unit_tag='S')
        _draw_cell(ax_i, ranks_i, v_base_i, boot_flat_i, rank_flat_i, n_i, B,
                   show_xlabel=True, show_ylabel=False, unit_tag='I')

        ax_s.set_title(OA_ERROR_LABELS[et], fontsize=9, pad=4)

    out_stem = 'fig_7b_oa_errors'
    sup = fig.suptitle('OA error bootstrap — influence per work', fontsize=9, y=1.02)

    out = paths.plots / f'{out_stem}.pdf'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f'Saved {out}')

    sup.set_visible(False)
    fig.savefig(paths.plots / f'{out_stem}_latex.pdf', bbox_inches='tight', dpi=150)
    print(f'Saved {paths.plots / (out_stem + "_latex.pdf")}')
    sup.set_visible(True)

    plt.close(fig)


def report_outliers(v_boot: np.ndarray, unit_ids: list, label: str,
                    name_map: dict = None) -> None:
    """
    Print units that have any bootstrap replicate outside [Y_LO, Y_HI].
    v_boot : (B, N) array aligned to unit_ids.
    name_map : optional {unit_idx: name} for display.
    """
    lo_mask = (v_boot < Y_LO).any(axis=0)
    hi_mask = (v_boot > Y_HI).any(axis=0)
    outlier_mask = lo_mask | hi_mask
    n_out = outlier_mask.sum()
    if n_out == 0:
        print(f'  {label}: no outliers outside [{Y_LO}, {Y_HI}]')
        return
    print(f'  {label}: {n_out} unit(s) with replicates outside [{Y_LO}, {Y_HI}]')
    print(f'    {"unit_idx":>12}  {"name":<35}  {"v_min":>8}  {"v_max":>8}  {"n_lo":>5}  {"n_hi":>5}')
    for i in np.where(outlier_mask)[0]:
        uid  = unit_ids[i]
        col  = v_boot[:, i]
        name = (name_map or {}).get(uid, '')
        print(f'    {uid:>12}  {name:<35}  {col.min():>8.4f}  {col.max():>8.4f}'
              f'  {int((col < Y_LO).sum()):>5}  {int((col > Y_HI).sum()):>5}')


def report_eigenvalues(lam_ratio_boot: np.ndarray) -> None:
    """
    Print summary of λ₂/λ₁ ratios across all replicates.
    Near-primitivity (λ₂/λ₁ close to 1) indicates near-reducibility.
    """
    valid = lam_ratio_boot[~np.isnan(lam_ratio_boot)]
    if len(valid) == 0:
        print('  No λ₂/λ₁ data available.')
        return
    print(f'\nλ₂/λ₁ ratio across {len(valid)} replicates:')
    print(f'  mean={valid.mean():.6f}  median={np.median(valid):.6f}')
    print(f'  min={valid.min():.6f}  max={valid.max():.6f}')
    print(f'  p5={np.percentile(valid,5):.6f}  p95={np.percentile(valid,95):.6f}')
    # Distribution by threshold
    for thresh in [0.90, 0.95, 0.98, 0.99, 0.999]:
        n = (valid >= thresh).sum()
        print(f'  λ₂/λ₁ ≥ {thresh}: {n} replicates ({100*n/len(valid):.1f}%)')


def report_boot_stability(
    boot_flat_s: np.ndarray, v_base_s: np.ndarray, n_s: int,
    boot_flat_i: np.ndarray, v_base_i: np.ndarray, n_i: int,
    B: int, label: str = '',
) -> None:
    """
    Print rank-stability statistics for bootstrap replicates, analogous to
    report_stability() in fig_5.py.  Statistics are averaged over B replicates.
    """
    from scipy import stats as _stats

    hdr = (f'{"Type":<5}  {"n":>6}  {"B":>5}  '
           f'{"Spear ρ":>8}  {"med log-r":>10}  {"sd log-r":>9}  '
           f'{"med|Δrank|":>11}  {"med|Δpct|":>10}')
    title = f'Bootstrap rank stability{(" — " + label) if label else ""}'
    print('\n' + '─' * len(hdr))
    print(title)
    print('─' * len(hdr))
    print(hdr)
    print('─' * len(hdr))

    strat_hdr = (f'  {"Type":<5}  {"quartile":<12}  {"n_units":>8}  '
                 f'{"v range":<16}  {"med log-r":>10}  {"sd log-r":>9}')

    for ukey, boot_flat, v_base, n in [
        ('S', boot_flat_s, v_base_s, n_s),
        ('I', boot_flat_i, v_base_i, n_i),
    ]:
        boot_2d = boot_flat.reshape(B, n)        # (B, n), units in baseline rank order
        baseline_ranks = np.arange(1, n + 1)

        # Spearman ρ per replicate → mean across replicates
        rhos = np.array([
            _stats.spearmanr(boot_2d[b], v_base).statistic
            for b in range(B)
        ])
        mean_rho = float(np.nanmean(rhos))

        # Log-ratio: (B, n) array
        with np.errstate(divide='ignore', invalid='ignore'):
            lr = np.log10(boot_2d / v_base[np.newaxis, :])   # (B, n)

        lr_valid = lr[np.isfinite(lr)]
        med_lr = float(np.median(lr_valid))
        sd_lr  = float(np.std(lr_valid, ddof=1))

        # Rank displacement per replicate → mean of per-replicate medians
        med_dr_list = []
        for b in range(B):
            row = boot_2d[b]
            finite = np.isfinite(row)
            if finite.sum() < 2:
                continue
            order = np.argsort(-np.where(finite, row, -np.inf))
            boot_ranks = np.empty(n, dtype=np.int32)
            boot_ranks[order] = np.arange(1, n + 1)
            med_dr_list.append(float(np.median(np.abs(boot_ranks - baseline_ranks))))

        med_dr  = float(np.mean(med_dr_list))
        med_rel = med_dr / n

        print(f'{ukey:<5}  {n:>6,}  {B:>5}  '
              f'{mean_rho:>8.4f}  {med_lr:>+10.4f}  {sd_lr:>9.4f}  '
              f'{med_dr:>11.1f}  {med_rel:>10.4f}')

        # ── Rank-stratified log-ratio (units already sorted by baseline rank) ──
        # units at position 0..n-1 are in descending v order (rank 1 = highest v)
        # v_base[0] > v_base[1] > … so quartile boundaries are positions
        q_bounds = [0, n // 4, n // 2, 3 * n // 4, n]
        q_labels = ['Q1 (top 25%)', 'Q2', 'Q3', 'Q4 (bot 25%)']
        print(strat_hdr)
        for qi, (lo, hi, qlabel) in enumerate(
                zip(q_bounds[:-1], q_bounds[1:], q_labels)):
            lr_q = lr[:, lo:hi]
            lr_q_valid = lr_q[np.isfinite(lr_q)]
            med_q = float(np.median(lr_q_valid))
            sd_q  = float(np.std(lr_q_valid, ddof=1))
            v_lo  = float(v_base[hi - 1])
            v_hi  = float(v_base[lo])
            print(f'  {ukey:<5}  {qlabel:<12}  {hi - lo:>8}  '
                  f'{v_lo:>6.3f}–{v_hi:<6.3f}    {med_q:>+10.4f}  {sd_q:>9.4f}')

    print('─' * len(hdr))
    print('Spear ρ     mean over B replicates of Spearman(v_boot, v_base).')
    print('med log-r   median of log(v_boot/v_base) pooled over B and units;')
    print('            positive for low-v units and negative for high-v units cancel')
    print('            in the pooled median — see rank-stratified breakdown above.')
    print('sd log-r    SD of log(v_boot/v_base) pooled.')
    print('med|Δrank|  mean over B replicates of median |rank_boot − rank_base|.')
    print('med|Δpct|   med|Δrank| / n; fractional percentile displacement.')


def plot7(paths, v_s_boot, v_u_boot, lam_ratio_boot, meta, df_base,
          field_labels: tuple,
          out_stem: str = 'fig_7', boot_label: str = '80% resample') -> None:
    B = v_s_boot.shape[0]

    # ── Sources ──────────────────────────────────────────────────────────────
    df_s = df_base[df_base['unit_type'] == 'S'].copy()
    src_ids  = df_s['unit_idx'].to_numpy(dtype=np.int64)
    src_v    = df_s['v'].to_numpy(dtype=np.float64)

    ranks_s, v_base_s, boot_flat_s, rank_flat_s, n_s, dense_s, order_s = \
        build_panel_data(v_s_boot, src_v, src_ids, meta['source_ids'])

    src_ids_ranked = [meta['source_ids'][dense_s[i]] for i in range(n_s)]
    boot_s_ranked  = v_s_boot[:, [dense_s[i] for i in range(n_s)]]
    print('\nBootstrap outliers (replicates outside [0.02, 20]):')
    report_outliers(boot_s_ranked, src_ids_ranked, 'Sources', name_map=field_labels[1])

    # ── Institutions ─────────────────────────────────────────────────────────
    df_i = df_base[df_base['unit_type'] == 'U'].copy()
    inst_ids = df_i['unit_idx'].to_numpy(dtype=np.int64)
    inst_v   = df_i['v'].to_numpy(dtype=np.float64)

    ranks_i, v_base_i, boot_flat_i, rank_flat_i, n_i, dense_i, _ = \
        build_panel_data(v_u_boot, inst_v, inst_ids, meta['inst_ids'])

    inst_ids_ranked = [meta['inst_ids'][dense_i[i]] for i in range(n_i)]
    boot_i_ranked   = v_u_boot[:, [dense_i[i] for i in range(n_i)]]
    report_outliers(boot_i_ranked, inst_ids_ranked, 'Institutions')

    if lam_ratio_boot is not None:
        report_eigenvalues(lam_ratio_boot)

    report_boot_stability(
        boot_flat_s, v_base_s, n_s,
        boot_flat_i, v_base_i, n_i,
        B, label=boot_label,
    )

    # ── Plot ─────────────────────────────────────────────────────────────────
    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.subplots_adjust(wspace=0.05)

    _draw_panel(axes[0], ranks_s, v_base_s, boot_flat_s, rank_flat_s,
                n_s, B, 'Sources', show_ylabel=True)
    _draw_panel(axes[1], ranks_i, v_base_i, boot_flat_i, rank_flat_i,
                n_i, B, 'Institutions', show_ylabel=False)

    skipped = meta.get('skipped', 0)
    sup = fig.suptitle(
        f'Bootstrap uncertainty — influence per work  '
        f'(B={B}, {boot_label}'
        + (f', {skipped} skipped' if skipped else '') + ')',
        fontsize=9, y=1.01,
    )

    out = paths.plots / f'{out_stem}.pdf'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f'Saved {out}')

    sup.set_visible(False)
    latex_out = paths.plots / f'{out_stem}_latex.pdf'
    fig.savefig(latex_out, bbox_inches='tight', dpi=150)
    print(f'Saved {latex_out}')
    sup.set_visible(True)

    plt.close(fig)

    # Console summary
    print(f'\nB={B}  skipped={skipped}')
    print(f'Sources:      n={n_s}  p05/p95 band computed')
    print(f'Institutions: n={n_i}  p05/p95 band computed')


# ─── Main ─────────────────────────────────────────────────────────────────────

BOOT_LABELS = {
    'bootstrap':           '80% resample',
    'bootstrap_oa_errors': 'OA metadata errors',
}

BOOT_OUT_STEMS = {
    'bootstrap':           'fig_7',
    'bootstrap_oa_errors': 'fig_7a_oa_errors',
}

# Sub-directory within $WORKING/<boot>/ where the npy files live
BOOT_STAGE_DIRS = {
    'bootstrap':           '',
    'bootstrap_oa_errors': 'stage4',
}

# Baseline ranking table to compare against for each bootstrap type.
# Both bootstrap types use direct 1/N_inst weights (omega=1) — no author model
# needed when resampling works — so both compare against baseline-direct.
BOOT_BASELINE_TABLES = {
    'bootstrap':           BASELINE_DIRECT_TABLE,
    'bootstrap_oa_errors': BASELINE_DIRECT_TABLE,
}


def main():
    parser = argparse.ArgumentParser(
        description='Plot bootstrap uncertainty in spectral ranking.'
    )
    parser.add_argument(
        '--boot', default='bootstrap_oa_errors',
        choices=list(BOOT_LABELS.keys()),
        help='Bootstrap subdirectory under $WORKING/ (default: bootstrap)',
    )
    args = parser.parse_args()

    paths   = load_config()
    rk_path = paths.working / 'rankings.duckdb'

    stage_sub = BOOT_STAGE_DIRS.get(args.boot, '')
    boot_dir  = paths.working / args.boot
    if stage_sub:
        boot_dir = boot_dir / stage_sub
    if not (boot_dir / 'v_s_boot.npy').exists():
        raise FileNotFoundError(
            f'Bootstrap arrays not found in {boot_dir}. '
            f'Run the corresponding bootstrap script first.'
        )

    print(f'Loading bootstrap arrays from {boot_dir} ...', flush=True)
    v_s_boot, v_u_boot, lam_ratio_boot, meta = load_bootstrap(boot_dir)
    B = v_s_boot.shape[0]
    print(f'  B={B}  n_s={meta["n_s"]}  n_u={meta["n_u"]}  '
          f'skipped={meta.get("skipped", 0)}')

    baseline_tbl = BOOT_BASELINE_TABLES[args.boot]
    print(f'Loading baseline ranking ({baseline_tbl}) ...', flush=True)
    df_base = load_baseline(rk_path, baseline_tbl)

    print('Loading field labels ...', flush=True)
    field_labels = load_field_labels(paths)   # (field_eb_dict, name_dict)

    print('Plotting ...', flush=True)
    plot7(paths, v_s_boot, v_u_boot, lam_ratio_boot, meta, df_base,
          field_labels,
          out_stem=BOOT_OUT_STEMS[args.boot],
          boot_label=BOOT_LABELS[args.boot])

    if args.boot == 'bootstrap_oa_errors':
        print('\nPlotting per-error-type facet (fig_7b_oa_errors) ...', flush=True)
        plot7a_oa_errors(paths, df_base)


if __name__ == '__main__':
    main()
    print('FINISHED!')
