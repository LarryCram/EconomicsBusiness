"""
compare_rankings_db.py — Compare every ranking table in two rankings.duckdb files.

Used for the QSS-revision reproduction check (rerun with an empty correction table
must reproduce the locked baseline) and for the corrected-vs-uncorrected comparison.

For each rk_* table present in both databases reports: units only in A / only in B,
and for common units max and s.d. of |log10 v_B - log10 v_A| by unit type.

Usage:
  .venv/bin/python spectral_ranking/compare_rankings_db.py A/rankings.duckdb B/rankings.duckdb
"""

import sys

import duckdb
import numpy as np
import pandas as pd


def tables(db):
    return {r[0] for r in db.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'rk_%'").fetchall()}


def main(a_path, b_path):
    a = duckdb.connect(a_path, read_only=True)
    b = duckdb.connect(b_path, read_only=True)
    ta, tb = tables(a), tables(b)
    print(f"tables: A={len(ta)}  B={len(tb)}  common={len(ta & tb)}")
    for t in sorted(ta - tb):
        print(f"  only in A: {t}")
    for t in sorted(tb - ta):
        print(f"  only in B: {t}")
    rows = []
    for t in sorted(ta & tb):
        q = f"SELECT unit_idx, unit_type, v FROM {t}"
        da, db_ = a.execute(q).df(), b.execute(q).df()
        m = da.merge(db_, on=['unit_idx', 'unit_type'], how='outer', suffixes=('_a', '_b'),
                     indicator=True)
        for ut, g in m.groupby('unit_type'):
            both = g[g['_merge'] == 'both']
            ok = both['v_a'].notna() & both['v_b'].notna() & (both['v_a'] > 0) & (both['v_b'] > 0)
            d = np.log10(both.loc[ok, 'v_b'] / both.loc[ok, 'v_a'])
            rows.append(dict(table=t, unit=ut, n_common=len(both),
                             only_a=int((g['_merge'] == 'left_only').sum()),
                             only_b=int((g['_merge'] == 'right_only').sum()),
                             max_abs_dlog=float(d.abs().max()) if len(d) else np.nan,
                             sd_dlog=float(d.std()) if len(d) > 1 else np.nan))
    r = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_colwidth', 90)
    print(r.to_string(index=False))
    return r


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
