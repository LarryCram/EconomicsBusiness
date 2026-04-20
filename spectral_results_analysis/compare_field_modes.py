"""
compare_field_modes.py — Test whether F=E/F=B source shifts are institution-mediated.

For sources present in all relevant corpora, computes v_F / v_base ratios under:
  m=0110 (bipartite — institutions in the loop)
  m=1000 (SS only  — institutions excluded)

If the F=E shift is institution-mediated, the ratio should be larger under m=0110
than under m=1000.  Wilcoxon signed-rank test on paired ratios.

Outputs console table + CSV to plots/.
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

_baseline = next(r for r in load_runs() if r['label'] == 'baseline')
_run_code = _baseline['run_code']
_tau_u    = _baseline['tau_u']
_tau_s    = _baseline['tau_s']

BASE_0110 = f'rk_{_run_code}_ALL_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m0110_chi50_alpha100'
BASE_1000 = f'rk_{_run_code}_ALL_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m1000_chi50_alpha100'

OVERLAY_PAIRS = [
    # (field, label_0110, label_1000, label_0001)
    ('E', 'F=E',    'F=E-SS', 'F=E-II'),
    ('B', 'F=B',    'F=B-SS', 'F=B-II'),
]


def load_table(db, table: str, unit_type: str = 'S') -> pd.DataFrame | None:
    tables = {r[0] for r in db.execute('SHOW TABLES').fetchall()}
    if table not in tables:
        print(f'  WARNING: {table} not found — skipping')
        return None
    q = f"SELECT label, table_name FROM _catalog WHERE table_name='{table}'"
    rows = db.execute(q).fetchall()
    label = rows[0][0] if rows else '?'
    df = db.execute(
        f"SELECT unit_idx, v FROM {table} WHERE unit_type='{unit_type}'"
    ).df()
    print(f'  {table}  ({label})  n_{unit_type}={len(df):,}')
    return df


def ratio_stats(v_F: np.ndarray, v_base: np.ndarray, label: str) -> dict:
    ratio = v_F / v_base
    w, p = stats.wilcoxon(ratio - 1.0, alternative='two-sided')
    return {
        'label':    label,
        'n':        len(ratio),
        'mean_ratio': float(np.mean(ratio)),
        'median_ratio': float(np.median(ratio)),
        'pct_above_1': float(100 * np.mean(ratio > 1.0)),
        'wilcoxon_W': float(w),
        'wilcoxon_p': float(p),
    }


def main():
    paths   = load_config()
    rk_path = paths.working / 'rankings.duckdb'

    rows = []
    with duckdb.connect(str(rk_path), read_only=True) as db:
        print('Loading baseline sources...')
        df_base_0110 = load_table(db, BASE_0110, 'S')
        df_base_1000 = load_table(db, BASE_1000, 'S')

        if df_base_0110 is None or df_base_1000 is None:
            raise RuntimeError('Baseline tables missing.')

        # Look up overlay table names from catalog
        cat = db.execute(
            "SELECT label, table_name FROM _catalog"
        ).df()
        label_to_table = dict(zip(cat['label'], cat['table_name']))

        for field, lbl_0110, lbl_1000, lbl_0001 in OVERLAY_PAIRS:
            print(f'\n=== Field {field} ===')
            tname_0110 = label_to_table.get(lbl_0110)
            tname_1000 = label_to_table.get(lbl_1000)
            tname_0001 = label_to_table.get(lbl_0001)

            df_F_0110 = load_table(db, tname_0110, 'S') if tname_0110 else None
            df_F_1000 = load_table(db, tname_1000, 'S') if tname_1000 else None
            df_F_0001 = load_table(db, tname_0001, 'U') if tname_0001 else None  # institutions

            # ── Sources: m=0110 vs m=1000 ──────────────────────────────────────
            if df_F_0110 is not None and df_F_1000 is not None:
                # Triple-merge: sources in F_0110 ∩ F_1000 ∩ base_0110 ∩ base_1000
                mg = (df_base_0110
                      .rename(columns={'v': 'v_base_0110'})
                      .merge(df_base_1000.rename(columns={'v': 'v_base_1000'}),
                             on='unit_idx', how='inner')
                      .merge(df_F_0110.rename(columns={'v': 'v_F_0110'}),
                             on='unit_idx', how='inner')
                      .merge(df_F_1000.rename(columns={'v': 'v_F_1000'}),
                             on='unit_idx', how='inner'))

                print(f'  Common sources (0110∩1000∩F_0110∩F_1000): {len(mg):,}')

                r0110 = ratio_stats(mg['v_F_0110'].values, mg['v_base_0110'].values,
                                    f'F={field} / base  [m=0110 bipartite]')
                r1000 = ratio_stats(mg['v_F_1000'].values, mg['v_base_1000'].values,
                                    f'F={field} / base  [m=1000 SS-only]')

                for r in (r0110, r1000):
                    rows.append({'field': field, **r})

                print(f'\n  {"":40s}  {"mean_ratio":>10}  {"median":>8}  {"pct>1":>6}  {"Wilcoxon p":>12}')
                print(f'  {"-"*40}  {"-"*10}  {"-"*8}  {"-"*6}  {"-"*12}')
                for r in (r0110, r1000):
                    print(f'  {r["label"]:<40}  {r["mean_ratio"]:10.4f}  '
                          f'{r["median_ratio"]:8.4f}  {r["pct_above_1"]:6.1f}%  '
                          f'{r["wilcoxon_p"]:12.3e}')

                # Paired test: is ratio_0110 > ratio_1000?
                diff = mg['v_F_0110'].values / mg['v_base_0110'].values \
                     - mg['v_F_1000'].values / mg['v_base_1000'].values
                w2, p2 = stats.wilcoxon(diff, alternative='greater')
                print(f'\n  Paired Wilcoxon (ratio_0110 > ratio_1000):  W={w2:.0f}  p={p2:.3e}')
                rows.append({
                    'field': field,
                    'label': f'PAIRED: F={field} ratio_0110 > ratio_1000',
                    'n': len(mg),
                    'mean_ratio': float(np.mean(diff)),
                    'median_ratio': float(np.median(diff)),
                    'pct_above_1': float(100 * np.mean(diff > 0)),
                    'wilcoxon_W': float(w2),
                    'wilcoxon_p': float(p2),
                })

            # ── Institutions: m=0110 baseline vs m=0001 ────────────────────────
            if df_F_0001 is not None:
                df_base_0110_U = load_table(db, BASE_0110, 'U')
                if df_base_0110_U is not None:
                    mg_i = (df_base_0110_U
                            .rename(columns={'v': 'v_base'})
                            .merge(df_F_0001.rename(columns={'v': 'v_F_0001'}),
                                   on='unit_idx', how='inner'))
                    print(f'\n  Common institutions (base∩F_{field}_II): {len(mg_i):,}')
                    ri = ratio_stats(mg_i['v_F_0001'].values, mg_i['v_base'].values,
                                     f'F={field} / base  [m=0001 II-only, institutions]')
                    rows.append({'field': field, **ri})
                    print(f'  {ri["label"]:<40}  {ri["mean_ratio"]:10.4f}  '
                          f'{ri["median_ratio"]:8.4f}  {ri["pct_above_1"]:6.1f}%  '
                          f'{ri["wilcoxon_p"]:12.3e}')

    out_csv = paths.plots / 'compare_field_modes.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'\nSaved {out_csv}')


if __name__ == '__main__':
    main()
    print('FINISHED!')
