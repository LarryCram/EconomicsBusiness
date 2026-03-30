"""
csv_3.py — Source-level comparison table across network modes (Figure 3 diagnostics).

Columns
-------
  baseline_rank   rank by v in m=0110
  source_idx
  source_name
  F               field label: E / B / (blank = unlabelled)
  a_p             work count
  v_0110          prestige per work, bipartite baseline
  v_1000          prestige per work, SS-only
  v_1111          prestige per work, full-joint χ*
  r_1000          v_1000 / v_0110
  r_1111          v_1111 / v_0110

Default sort: baseline_rank ascending.
Re-sort as needed in any spreadsheet or with pandas.

Output: data/csv_3_mode_comparison.csv
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_params

_tau = load_params()['tau_u_floor']['A']

BASELINE_TABLE = f'rk_t5_A_tau{_tau}_rho0_m0110_chi50_alpha85'
SS_TABLE       = f'rk_t5_A_tau{_tau}_rho0_m1000_chi50_alpha85'
OUT_NAME       = 'csv_3_mode_comparison.csv'


# ─── Helpers ──────────────────────────────────────────────────────────────────

def resolve_chi_star_table(db) -> str | None:
    tables = {row[0] for row in db.execute('SHOW TABLES').fetchall()}
    if '_catalog' not in tables:
        print('  WARNING: _catalog not found')
        return None
    rows = db.execute(
        "SELECT table_name FROM _catalog "
        "WHERE m_SS=1 AND m_SI=1 AND m_IS=1 AND m_II=1 "
        f"  AND tx=5 AND fx='A' AND tau_u={_tau} AND rho=0 "
        "  AND round(alpha*100)=85 AND round(chi*100) != 50 "
        "ORDER BY created_at DESC LIMIT 1"
    ).fetchall()
    if not rows:
        print('  WARNING: no full-joint χ* entry in _catalog — r_1111 will be NaN')
        return None
    tname = rows[0][0]
    if tname not in tables:
        print(f'  WARNING: {tname} not in database — r_1111 will be NaN')
        return None
    print(f'  χ* table: {tname}')
    return tname


def load_sources(db, tname: str) -> pd.DataFrame:
    return db.execute(
        f"SELECT unit_idx AS source_idx, v, a_p "
        f"FROM {tname} WHERE unit_type = 'S'"
    ).df()


def load_field_labels_and_names(data_dir: Path) -> pd.DataFrame:
    sm = pd.read_csv(data_dir / 'source_master.csv',
                     usecols=['source_idx', 'source_name', 'field_subset'])
    sm['source_idx'] = sm['source_idx'].astype(int)
    sm = sm.rename(columns={'field_subset': 'F'})
    return sm


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    paths   = load_config()
    rk_path = paths.working / 'rankings.duckdb'

    if not rk_path.exists():
        raise FileNotFoundError(f'rankings.duckdb not found at {rk_path}')

    with duckdb.connect(str(rk_path), read_only=True) as db:
        tables = {row[0] for row in db.execute('SHOW TABLES').fetchall()}

        if BASELINE_TABLE not in tables:
            raise RuntimeError(f'Baseline table {BASELINE_TABLE} not found')
        df_base = load_sources(db, BASELINE_TABLE)
        df_base = df_base.sort_values('v', ascending=False).reset_index(drop=True)
        df_base['baseline_rank'] = np.arange(1, len(df_base) + 1)
        df_base = df_base.rename(columns={'v': 'v_0110'})

        if SS_TABLE not in tables:
            raise RuntimeError(f'SS table {SS_TABLE} not found')
        df_ss = load_sources(db, SS_TABLE)[['source_idx', 'v']].rename(columns={'v': 'v_1000'})

        chi_star_table = resolve_chi_star_table(db)
        if chi_star_table:
            df_full = load_sources(db, chi_star_table)[['source_idx', 'v']].rename(columns={'v': 'v_1111'})
        else:
            df_full = pd.DataFrame(columns=['source_idx', 'v_1111'])

    # ── Join ─────────────────────────────────────────────────────────────────
    df = (df_base
          .merge(df_ss,   on='source_idx', how='left')
          .merge(df_full, on='source_idx', how='left'))

    # ── Ratios ───────────────────────────────────────────────────────────────
    df['r_1000'] = df['v_1000'] / df['v_0110']
    df['r_1111'] = df['v_1111'] / df['v_0110']

    # ── Names and field labels ────────────────────────────────────────────────
    meta = load_field_labels_and_names(paths.data)
    df = df.merge(meta, on='source_idx', how='left')

    # ── Final column order ────────────────────────────────────────────────────
    df = df[['baseline_rank', 'source_idx', 'source_name', 'F', 'a_p',
             'v_0110', 'v_1000', 'v_1111', 'r_1000', 'r_1111']]

    # ── Write ─────────────────────────────────────────────────────────────────
    out = paths.data / OUT_NAME
    df.to_csv(out, index=False, float_format='%.6f')
    print(f'\nSaved {out}  ({len(df):,} sources)')

    # ── Quick diagnostics ─────────────────────────────────────────────────────
    print(f'\nTop 10 by r_1000 (v_1000 / v_0110):')
    print(df.nlargest(10, 'r_1000')
            [['baseline_rank', 'source_name', 'F', 'a_p', 'v_0110', 'v_1000', 'r_1000']]
            .to_string(index=False))

    print(f'\nTop 10 by r_1111 (v_1111 / v_0110):')
    print(df.nlargest(10, 'r_1111')
            [['baseline_rank', 'source_name', 'F', 'a_p', 'v_0110', 'v_1111', 'r_1111']]
            .to_string(index=False))


if __name__ == '__main__':
    main()
    print('FINISHED!')
