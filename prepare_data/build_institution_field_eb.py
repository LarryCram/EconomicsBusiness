"""
build_institution_field_eb.py — Field classification for institutions.

Uses H_SI (row-normalised C_SI) from the baseline edge list to classify each
institution by the field_eb composition of the sources whose works cite it.
Row-normalising C_SI gives each source equal total weight regardless of volume,
consistent with how the ranking model distributes prestige via H_SI.

Classification rule:
  field_eb = dominant label among {E, B, A} by H_SI-weighted fraction;
  X if the X fraction >= 0.5 (majority of H_SI weight comes from X sources).

Output
------
  WORKING/parquet/institution_field_eb.parquet
    unit_idx, w_total, frac_E, frac_B, frac_A, frac_X, field_eb
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs


def main():
    paths = load_config()
    par   = paths.working / 'parquet'

    b        = next(r for r in load_runs() if r['label'] == 'baseline')
    el_table = f"el_{b['run_code']}_A_tauU{b['tau_u']}_tauS{b['tau_s']}_vartau"

    print(f'Using edge list: {el_table}')

    with duckdb.connect(str(paths.working / 'edge_lists.duckdb'), read_only=True) as db:
        el = db.execute(f"""
            WITH raw AS (
                SELECT cited_inst_idx AS unit_idx,
                       citer_source_idx AS source_idx,
                       SUM(cited_inst_weight) AS w
                FROM {el_table}
                WHERE cited_inst_idx IS NOT NULL AND citer_source_idx IS NOT NULL
                GROUP BY cited_inst_idx, citer_source_idx
            )
            SELECT unit_idx,
                   source_idx,
                   w / SUM(w) OVER (PARTITION BY source_idx) AS w
            FROM raw
        """).df()

    src_eb = pd.read_parquet(str(par / 'source_master.parquet'),
                             columns=['source_idx', 'field_eb'])
    el = el.merge(src_eb, on='source_idx', how='left')
    el['field_eb'] = el['field_eb'].fillna('X')

    total = el.groupby('unit_idx')['w'].sum().rename('w_total')
    fracs = (el.groupby(['unit_idx', 'field_eb'])['w']
               .sum()
               .unstack(fill_value=0.0)
               .div(total, axis=0))
    for col in ['E', 'B', 'A', 'X']:
        if col not in fracs:
            fracs[col] = 0.0
    fracs = fracs[['E', 'B', 'A', 'X']].copy()
    fracs.columns = ['frac_E', 'frac_B', 'frac_A', 'frac_X']
    fracs = fracs.join(total).reset_index()

    def classify(row):
        if row['frac_X'] >= 0.5:
            return 'X'
        return row[['frac_E', 'frac_B', 'frac_A']].idxmax().replace('frac_', '')

    fracs['field_eb'] = fracs.apply(classify, axis=1)

    out = par / 'institution_field_eb.parquet'
    fracs.to_parquet(str(out), index=False)

    print(fracs['field_eb'].value_counts().to_string())
    print(f'Total institutions: {len(fracs):,}')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
    print('FINISHED!')
