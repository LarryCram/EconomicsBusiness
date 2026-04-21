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
BASELINE_TABLE = f'rk_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m0110_chi50_alpha100'
EL_TABLE       = f'el_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau'

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

def load_bootstrap(paths) -> tuple:
    """Load v_s_boot, v_u_boot, lam_ratio_boot arrays and meta from $WORKING/bootstrap/."""
    boot_dir = paths.working / 'bootstrap'
    v_s_boot = np.load(boot_dir / 'v_s_boot.npy')   # (B, n_s)
    v_u_boot = np.load(boot_dir / 'v_u_boot.npy')   # (B, n_u)
    lam_path = boot_dir / 'lam_ratio_boot.npy'
    lam_ratio_boot = np.load(lam_path) if lam_path.exists() else None
    with open(boot_dir / 'meta.json') as f:
        meta = json.load(f)
    completed = meta.get('completed', v_s_boot.shape[0])
    # Only use completed replicates (rows with non-zero v_s)
    v_s_boot = v_s_boot[:completed]
    v_u_boot = v_u_boot[:completed]
    if lam_ratio_boot is not None:
        lam_ratio_boot = lam_ratio_boot[:completed]
    return v_s_boot, v_u_boot, lam_ratio_boot, meta


def load_baseline(rk_path: Path) -> pd.DataFrame:
    """Load baseline v and unit_type from rankings.duckdb."""
    with duckdb.connect(str(rk_path), read_only=True) as db:
        df = db.execute(
            f"SELECT unit_idx, unit_type, v FROM {BASELINE_TABLE}"
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
                colours=None) -> None:
    """Draw one panel: bootstrap scatter + percentile band + baseline."""

    # Bootstrap scatter
    ax.scatter(
        rank_flat, boot_flat,
        c=colours if colours is not None else '#aaaaaa',
        s=1, alpha=0.04, linewidths=0, rasterized=True,
        zorder=1,
    )

    # 5th–95th percentile band per unit
    boot_2d = boot_flat.reshape(B, n_aligned)
    p05 = np.nanpercentile(boot_2d, 5,  axis=0)
    p95 = np.nanpercentile(boot_2d, 95, axis=0)
    ax.fill_between(ranks, p05, p95,
                    color='#888888', alpha=0.18, linewidth=0, zorder=2)

    # Baseline curve
    ax.plot(ranks, v_base,
            color='black', linewidth=1.4, alpha=1.0, zorder=3,
            label='baseline')

    ax.set_yscale('log')
    ax.set_ylim(0.02, 20)
    ax.axhline(1.0, color='#999999', linewidth=0.8, linestyle='--', zorder=0)
    ax.text(n_aligned * 0.98, 1.0, '$v=1$',
            ha='right', va='bottom', fontsize=7.5, color='#999999')
    ax.set_xlim(1, n_aligned)
    ax.set_xlabel('Baseline rank', labelpad=4)
    ax.set_ylabel('Influence per work $v$', labelpad=4)
    ax.set_title(panel_title, fontsize=10, pad=6)


Y_LO, Y_HI = 0.02, 20.0


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


def plot7(paths, v_s_boot, v_u_boot, lam_ratio_boot, meta, df_base,
          field_labels: tuple, inst_field_map=None) -> None:
    B = v_s_boot.shape[0]

    # ── Sources ──────────────────────────────────────────────────────────────
    df_s = df_base[df_base['unit_type'] == 'S'].copy()
    src_ids  = df_s['unit_idx'].to_numpy(dtype=np.int64)
    src_v    = df_s['v'].to_numpy(dtype=np.float64)

    ranks_s, v_base_s, boot_flat_s, rank_flat_s, n_s, dense_s, order_s = \
        build_panel_data(v_s_boot, src_v, src_ids, meta['source_ids'])

    # Outlier report — aligned boot array is (B, n_s) in rank order
    src_ids_ranked = [meta['source_ids'][dense_s[i]] for i in range(n_s)]
    boot_s_ranked  = v_s_boot[:, [dense_s[i] for i in range(n_s)]]
    print('\nBootstrap outliers (replicates outside [0.02, 20]):')
    report_outliers(boot_s_ranked, src_ids_ranked, 'Sources', name_map=field_labels[1])

    # Per-point colour by field_eb (repeated B times)
    field_per_unit = np.array([
        field_labels[0].get(meta['source_ids'][dense_s[i]], None)
        for i in range(n_s)
    ])
    colours_per_unit = np.array([
        FIELD_COLOURS.get(f, FIELD_COLOURS[None]) for f in field_per_unit
    ])
    colours_flat_s = np.tile(colours_per_unit, B)

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

    # Per-point colour by institution field (repeated B times)
    if inst_field_map is not None:
        inst_field_per_unit = np.array([
            inst_field_map.get(meta['inst_ids'][dense_i[i]], 'X')
            for i in range(n_i)
        ])
        colours_per_inst = np.array([
            INST_FIELD_COLOURS.get(f, INST_FIELD_COLOURS['X'])
            for f in inst_field_per_unit
        ])
        colours_flat_i = np.tile(colours_per_inst, B)
    else:
        colours_flat_i = None

    # ── Plot ─────────────────────────────────────────────────────────────────
    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    fig.subplots_adjust(hspace=0.44)

    _draw_panel(axes[0], ranks_s, v_base_s, boot_flat_s, rank_flat_s,
                n_s, B, 'Sources', colours=colours_flat_s)
    _draw_panel(axes[1], ranks_i, v_base_i, boot_flat_i, rank_flat_i,
                n_i, B, 'Institutions', colours=colours_flat_i)

    # Legend for source colours
    for label, colour in [('E', FIELD_COLOURS['E']),
                           ('B', FIELD_COLOURS['B']),
                           ('A', FIELD_COLOURS['A']),
                           ('X', FIELD_COLOURS[None])]:
        axes[0].scatter([], [], c=colour, s=20, label=label)
    axes[0].legend(fontsize=7, framealpha=0.85, loc='upper right')

    # Legend for institution colours
    if inst_field_map is not None:
        for label, colour in [('E', INST_FIELD_COLOURS['E']),
                               ('B', INST_FIELD_COLOURS['B']),
                               ('A', INST_FIELD_COLOURS['A']),
                               ('X', INST_FIELD_COLOURS['X'])]:
            axes[1].scatter([], [], c=colour, s=20, label=label)
        axes[1].legend(fontsize=7, framealpha=0.85, loc='upper right')

    skipped = meta.get('skipped', 0)
    sup = fig.suptitle(
        f'Bootstrap uncertainty — influence per work  '
        f'(B={B}, 80% resample'
        + (f', {skipped} skipped' if skipped else '') + ')',
        fontsize=9, y=1.01,
    )

    out = paths.plots / 'fig_7.pdf'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f'Saved {out}')

    sup.set_visible(False)
    latex_out = paths.plots / 'fig_7_latex.pdf'
    fig.savefig(latex_out, bbox_inches='tight', dpi=150)
    print(f'Saved {latex_out}')
    sup.set_visible(True)

    plt.close(fig)

    # Console summary
    print(f'\nB={B}  skipped={skipped}')
    print(f'Sources:      n={n_s}  p05/p95 band computed')
    print(f'Institutions: n={n_i}  p05/p95 band computed')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    paths   = load_config()
    rk_path = paths.working / 'rankings.duckdb'

    boot_dir = paths.working / 'bootstrap'
    if not (boot_dir / 'v_s_boot.npy').exists():
        raise FileNotFoundError(
            f'Bootstrap arrays not found in {boot_dir}. '
            'Run spectral_ranking_bootstrap/bootstrap_baseline.py first.'
        )

    print('Loading bootstrap arrays ...', flush=True)
    v_s_boot, v_u_boot, lam_ratio_boot, meta = load_bootstrap(paths)
    B = v_s_boot.shape[0]
    print(f'  B={B}  n_s={meta["n_s"]}  n_u={meta["n_u"]}  '
          f'skipped={meta.get("skipped", 0)}')

    print('Loading baseline ranking ...', flush=True)
    df_base = load_baseline(rk_path)

    print('Loading field labels ...', flush=True)
    field_labels = load_field_labels(paths)   # (field_eb_dict, name_dict)

    print('Loading institution field labels ...', flush=True)
    inst_field_map = fetch_inst_field_labels(paths.parquet)
    print(f'  {len(inst_field_map)} institutions classified')

    print('Plotting ...', flush=True)
    plot7(paths, v_s_boot, v_u_boot, lam_ratio_boot, meta, df_base,
          field_labels, inst_field_map)


if __name__ == '__main__':
    main()
    print('FINISHED!')
