"""
fig_field_joint_risk.py — Field-sensitivity of unit rankings across mask and corpus.

3-row × 2-col figure (Sources left, Institutions right).
x-axis: EB bipartite rank (m=0110, F=EB — E+B units only).
y-axis: log10(v_alt) − log10(v_EB_bip)  (offset from EB bipartite baseline).
Points coloured by field_eb: E (red) and B (blue) only.
100-unit sliding-window running mean overlaid per field.

Rows:
  (a) E+B, single unit  — F=EB (no A/X), m=1000 (sources) / m=0001 (institutions)
  (b) E||B, single unit — F=E or F=B separately, m=1000 (sources) / m=0001 (institutions)
  (c) E||B, bipartite   — F=E and F=B separately, m=0110

Outputs:
  plots/fig_field_joint_risk.pdf / _latex.pdf
  plots/field_joint_risk_summary.csv
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

# ── Run parameters ─────────────────────────────────────────────────────────────
_bl        = next(r for r in load_runs() if r['label'] == 'baseline')
_rc        = _bl['run_code']   # '20242024'
_tau_u     = _bl['tau_u']      # 20
_tau_s     = _bl['tau_s']      # 10

def _tname(fx, m, rho=0, chi=50, alpha=100):
    return (f'rk_{_rc}_{fx}_tauU{_tau_u}_tauS{_tau_s}_vartau'
            f'_rho{rho}_m{m}_chi{chi}_alpha{alpha}')

# (a) E+B corpus (no A/X), single-unit
SS_EB = _tname('EB', '1000')
II_EB = _tname('EB', '0001')
# (b) field-separate, single-unit
SS_E = _tname('E', '1000')
SS_B = _tname('B', '1000')
II_E = _tname('E', '0001')
II_B = _tname('B', '0001')
# (c) E||B separately, bipartite
BIP_E  = _tname('E',  '0110')
BIP_B  = _tname('B',  '0110')
# baseline
BIP_EB = _tname('EB', '0110')

# ── Visual spec ─────────────────────────────────────────────────────────────────
E_COLOR = '#e41a1c'
B_COLOR = '#377eb8'
FIELD_COLOR = {'E': E_COLOR, 'B': B_COLOR}

Y_LO, Y_HI = -1.2, 1.2
Y_TICKS = [-1.0, -0.5, 0.0, 0.5, 1.0]
ROLL_WIN = 100   # running-mean window (units)
PT_SIZE  = 8
PT_ALPHA = 0.25
LINE_LW  = 2.0


def _pub_theme():
    sns.set_theme(style='whitegrid', font_scale=1.05)
    plt.rcParams.update({
        'axes.linewidth':    1.2,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.major.size':  5,   'ytick.major.size':  5,
    })


def _load_v(db, tname: str, unit_type: str) -> pd.DataFrame:
    """Return DataFrame(unit_idx, v) for a given table and unit_type ('S' or 'U')."""
    return db.execute(
        f"SELECT unit_idx, v FROM {tname} WHERE unit_type='{unit_type}'"
    ).df()


def _rolling_mean(x_sorted: np.ndarray, y_sorted: np.ndarray,
                  win: int = ROLL_WIN) -> tuple:
    """Return (x, y_smooth) using a centred rolling mean of size win."""
    s = pd.Series(y_sorted, index=x_sorted)
    smooth = s.rolling(win, min_periods=max(10, win // 5), center=True).mean()
    return x_sorted, smooth.values


def _lv(v):
    return np.log10(np.clip(v, 1e-10, None))



def _draw_strip(ax, df_base: pd.DataFrame, data_series: list,
                n_src: int, legend_title: str, ylabel: bool,
                bottom_row: bool, show_denom: bool = False) -> list:
    ax.axhline(0, color='black', linewidth=0.9, linestyle='--', zorder=5)
    ax.set_xlim(0, n_src)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_yticks(Y_TICKS)
    if bottom_row:
        ax.set_xlabel('EB bipartite rank', labelpad=4)
    else:
        ax.tick_params(labelbottom=False)
    if ylabel:
        ax.set_ylabel(r'$\Delta\log(v)$', labelpad=4)
    else:
        ax.tick_params(labelleft=False)

    # Pre-merge all series so we can compute the denominator before labelling
    prepared = []
    for df_v, field in data_series:
        merged = (df_base[['unit_idx', 'baseline_rank', 'v_base']]
                  .merge(df_v.rename(columns={'v': 'v_alt'}),
                         on='unit_idx', how='inner'))
        merged = merged[(merged['v_base'] > 0) & (merged['v_alt'] > 0)].copy()
        prepared.append((merged, field))

    total_n = sum(len(m) for m, _ in prepared) if show_denom else None

    for merged, field in prepared:
        c = FIELD_COLOR.get(field, '#888888')
        if merged.empty:
            continue
        merged.sort_values('baseline_rank', inplace=True)
        delta = _lv(merged['v_alt'].values) - _lv(merged['v_base'].values)
        x = merged['baseline_rank'].values
        n_label = (f'n={len(merged):,}/{total_n:,}' if show_denom
                   else f'n={len(merged):,}')
        ax.scatter(x, delta, color=c, s=PT_SIZE, alpha=PT_ALPHA,
                   linewidths=0, zorder=2,
                   label=f'$F_{{{field}}}$  ({n_label})')

        x_sm, y_sm = _rolling_mean(x.astype(float), delta)
        valid = ~np.isnan(y_sm)
        if valid.sum() > 1:
            ax.plot(x_sm[valid], y_sm[valid], color=c,
                    lw=LINE_LW, alpha=1.0, zorder=3)

    leg = ax.legend(title=legend_title, fontsize=9, framealpha=0.85,
                    loc='upper left', bbox_to_anchor=(0.03, 0.97),
                    markerscale=1.8)
    leg.get_title().set_fontsize(9)
    return []


def _save(fig, sup, stem: str, paths) -> None:
    out = paths.plots / f'{stem}.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')
    sup.set_visible(False)
    fig.savefig(paths.plots / f'{stem}_latex.pdf', bbox_inches='tight')
    print(f'Saved {paths.plots / f"{stem}_latex.pdf"}')
    sup.set_visible(True)
    plt.close(fig)


def _write_summary(all_rows: list, paths) -> None:
    """Save a brief numerical summary CSV."""
    rows = []
    for panel, field, deltas in all_rows:
        deltas = np.array(deltas)
        rows.append({
            'panel': panel,
            'field': field,
            'n': len(deltas),
            'median_delta_log10': float(np.median(deltas)),
            'mean_delta_log10': float(np.mean(deltas)),
            'median_ratio': float(10 ** np.median(deltas)),
        })
    pd.DataFrame(rows).to_csv(str(paths.plots / 'field_joint_risk_summary.csv'), index=False)
    print(f'Saved {paths.plots / "field_joint_risk_summary.csv"}')


def main():
    paths   = load_config()
    rk_path = paths.working / 'rankings.duckdb'
    par     = paths.working / 'parquet'

    if not rk_path.exists():
        raise FileNotFoundError(f'rankings.duckdb not found at {rk_path}')

    # ── Field labels ─────────────────────────────────────────────────────────
    src_eb = pd.read_parquet(str(par / 'source_master.parquet'),
                             columns=['source_idx', 'field_eb'])
    src_field = dict(zip(src_eb['source_idx'].astype(int), src_eb['field_eb']))

    inst_eb = pd.read_parquet(str(par / 'institution_field_eb.parquet'),
                              columns=['unit_idx', 'field_eb'])
    inst_field = dict(zip(inst_eb['unit_idx'].astype(int), inst_eb['field_eb']))

    # ── Load all needed ranking tables ─────────────────────────────────────
    with duckdb.connect(str(rk_path), read_only=True) as db:
        tables = {row[0] for row in db.execute('SHOW TABLES').fetchall()}

        def need(t):
            if t not in tables:
                raise RuntimeError(f'Table {t} not found in rankings.duckdb')

        for t in [SS_EB, II_EB, SS_E, SS_B, II_E, II_B,
                  BIP_E, BIP_B, BIP_EB]:
            need(t)

        v_base_s  = _load_v(db, BIP_EB, 'S')
        v_base_i  = _load_v(db, BIP_EB, 'U')
        v_ss_eb    = _load_v(db, SS_EB,  'S')
        v_ii_eb    = _load_v(db, II_EB,  'U')
        v_ss_e     = _load_v(db, SS_E,   'S')
        v_ss_b     = _load_v(db, SS_B,   'S')
        v_ii_e     = _load_v(db, II_E,   'U')
        v_ii_b     = _load_v(db, II_B,   'U')
        v_bip_e_s  = _load_v(db, BIP_E,  'S')
        v_bip_e_i  = _load_v(db, BIP_E,  'U')
        v_bip_b_s  = _load_v(db, BIP_B,  'S')
        v_bip_b_i  = _load_v(db, BIP_B,  'U')
        v_bip_eb_s = _load_v(db, BIP_EB, 'S')
        v_bip_eb_i = _load_v(db, BIP_EB, 'U')

    # ── Build baseline rank maps and annotate with field ──────────────────
    def _mk_base(v_df, field_map) -> pd.DataFrame:
        df = (v_df.sort_values('v', ascending=False)
                  .reset_index(drop=True)
                  .copy())
        df['baseline_rank'] = np.arange(1, len(df) + 1)
        df['v_base']  = df['v']
        df['field_eb'] = df['unit_idx'].map(field_map).fillna('X')
        return df[['unit_idx', 'baseline_rank', 'v_base', 'field_eb']]

    base_s = _mk_base(v_base_s, src_field)
    base_i = _mk_base(v_base_i, inst_field)

    n_s = len(base_s)
    n_i = len(base_i)
    print(f'Baseline: {n_s} sources, {n_i} institutions')

    # EB-filtered baseline: E and B units only, re-ranked 1..n preserving EBAX ordering
    def _mk_base_eb(base):
        df = (base[base['field_eb'].isin(['E', 'B'])]
              .sort_values('baseline_rank')
              .copy())
        df['baseline_rank'] = np.arange(1, len(df) + 1)
        return df[['unit_idx', 'baseline_rank', 'v_base', 'field_eb']]

    base_s_eb = _mk_base_eb(base_s)
    base_i_eb = _mk_base_eb(base_i)
    n_s_eb = len(base_s_eb)
    n_i_eb = len(base_i_eb)
    print(f'EB subset: {n_s_eb} sources, {n_i_eb} institutions')

    # ── Assemble panel data ────────────────────────────────────────────────
    #
    # data_series for _draw_strip: list of (DataFrame(unit_idx, v), field_str)
    # The field_str is used for colour; for panel (a) we split the single table
    # by field_eb using the base field map.

    def _split_by_field(df_v, base, field):
        """Keep only units with matching field from base."""
        keep = base.loc[base['field_eb'] == field, 'unit_idx']
        return df_v[df_v['unit_idx'].isin(set(keep))]

    # Panel (a): E+B corpus (no A/X), single-unit mask
    pa_s = [(_split_by_field(v_ss_eb, base_s, 'E'), 'E'),
            (_split_by_field(v_ss_eb, base_s, 'B'), 'B')]
    pa_i = [(_split_by_field(v_ii_eb, base_i, 'E'), 'E'),
            (_split_by_field(v_ii_eb, base_i, 'B'), 'B')]

    # Panel (b): field-separate, single-unit mask
    pb_s = [(v_ss_e, 'E'), (v_ss_b, 'B')]
    pb_i = [(v_ii_e, 'E'), (v_ii_b, 'B')]

    # Panel (c): E||B separately, bipartite
    pc_s = [(v_bip_e_s, 'E'), (v_bip_b_s, 'B')]
    pc_i = [(v_bip_e_i, 'E'), (v_bip_b_i, 'B')]

    PANEL_ROWS = [
        ('E+B, single unit', pa_s, pa_i, n_s_eb, n_i_eb, True),
        ('E||B, single unit', pb_s, pb_i, n_s_eb, n_i_eb, False),
        ('E||B, bipartite',  pc_s, pc_i, n_s_eb, n_i_eb, False),
    ]

    # ── Summary stats ───────────────────────────────────────────────────────
    summary_rows = []

    def _collect_deltas(base, data_series, panel_name):
        for df_v, field in data_series:
            merged = (base[['unit_idx', 'v_base']]
                      .merge(df_v.rename(columns={'v': 'v_alt'}),
                             on='unit_idx', how='inner'))
            merged = merged[(merged['v_base'] > 0) & (merged['v_alt'] > 0)]
            if not merged.empty:
                deltas = (_lv(merged['v_alt'].values) -
                          _lv(merged['v_base'].values)).tolist()
                summary_rows.append((panel_name, field, deltas))

    _collect_deltas(base_s, pa_s, 'Source-only')
    _collect_deltas(base_i, pa_i, 'Inst-only')
    _collect_deltas(base_s, pc_s, 'E||B-bipartite-S')
    _collect_deltas(base_i, pc_i, 'E||B-bipartite-I')

    # ── Plot ────────────────────────────────────────────────────────────────
    _pub_theme()
    n_rows = len(PANEL_ROWS)
    _lsz = plt.rcParams.get('axes.labelsize', 10)

    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3 * n_rows),
                             sharey=True, sharex=False, squeeze=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.12)

    # Column titles
    axes[0, 0].set_title('Sources',      fontsize=_lsz, pad=5)
    axes[0, 1].set_title('Institutions', fontsize=_lsz, pad=5)

    for row_i, (legend_title, src_series, inst_series, n_s_panel, n_i_panel, denom) in enumerate(PANEL_ROWS):
        is_bottom = (row_i == n_rows - 1)
        _draw_strip(axes[row_i, 0], base_s_eb, src_series,
                    n_src=n_s_panel, legend_title=legend_title,
                    ylabel=True, bottom_row=is_bottom, show_denom=denom)
        _draw_strip(axes[row_i, 1], base_i_eb, inst_series,
                    n_src=n_i_panel, legend_title=legend_title,
                    ylabel=False, bottom_row=is_bottom, show_denom=denom)

    # Shared y-label (already set per strip for left column)
    # Add bottom-strip x-tick visibility fix
    axes[-1, 0].xaxis.get_major_ticks()[-1].label1.set_visible(False)

    sup = fig.suptitle(
        r'$\Delta\log(v)$ vs bipartite baseline: field-sensitivity by unit mask',
        fontsize=_lsz, y=1.01,
    )

    _save(fig, sup, 'fig_field_joint_risk', paths)
    _write_summary(summary_rows, paths)

    # ── Console report ──────────────────────────────────────────────────────
    print('\nSummary (median Δlog10(v)):')
    print(f'{"Panel":<40}  {"Field":>5}  {"n":>6}  {"median":>8}')
    for panel, field, deltas in summary_rows:
        arr = np.array(deltas)
        print(f'{panel:<40}  {field:>5}  {len(arr):>6,}  {np.median(arr):>+8.3f}')


if __name__ == '__main__':
    main()
    print('FINISHED!')
