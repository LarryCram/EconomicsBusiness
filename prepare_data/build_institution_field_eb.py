"""
build_institution_field_eb.py — Field classification for institutions.

Uses H_SI (row-normalised C_SI) from the baseline edge list to classify each
institution by the field_eb composition of the sources whose works cite it.
Row-normalising C_SI gives each source equal total weight regardless of volume,
consistent with how the ranking model distributes prestige via H_SI.

Classification rule (quota-based):
    1) Assign top 30% of institutions by frac_X to X.
    2) From the remainder, assign top 10% of all institutions by frac_A to A.
    3) Split the rest into E:B = 5:8 using frac_E ranking for E;
         all remaining institutions are B.

This enforces global composition targets while still using fractional scores
to rank units within each class.

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


TARGET_X_SHARE = 0.30
TARGET_A_SHARE = 0.10
TARGET_E_RATIO = 5
TARGET_B_RATIO = 8


def main():
    paths = load_config()
    par   = paths.working / 'parquet'

    b = next(r for r in load_runs() if r['label'] == 'baseline')
    el_table = (
        f"el_{b['run_code']}_{b['fx']}_tauU{b['tau_u']}_tauS{b['tau_s']}_vartau"
    )

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

    n_total = len(fracs)
    n_x = int(round(TARGET_X_SHARE * n_total))
    n_a = int(round(TARGET_A_SHARE * n_total))

    # Step 1: X by highest frac_X.
    ranked_x = fracs.sort_values(
        ['frac_X', 'frac_B', 'frac_E', 'frac_A', 'unit_idx'],
        ascending=[False, False, False, False, True],
    )
    x_ids = set(ranked_x.head(n_x)['unit_idx'].astype(int))

    # Step 2: A by highest frac_A from non-X remainder.
    rem_after_x = fracs.loc[~fracs['unit_idx'].isin(x_ids)].copy()
    ranked_a = rem_after_x.sort_values(
        ['frac_A', 'frac_E', 'frac_B', 'frac_X', 'unit_idx'],
        ascending=[False, False, False, False, True],
    )
    a_ids = set(ranked_a.head(n_a)['unit_idx'].astype(int))

    # Step 3: E:B = 5:8 on the remaining pool, with the guardrail that
    # B labels are assigned only when frac_B >= frac_E.
    rem_after_ax = rem_after_x.loc[~rem_after_x['unit_idx'].isin(a_ids)].copy()
    rem_n = len(rem_after_ax)
    n_e = int(round(rem_n * TARGET_E_RATIO / (TARGET_E_RATIO + TARGET_B_RATIO)))

    ranked_e = rem_after_ax.sort_values(
        ['frac_E', 'frac_B', 'frac_A', 'frac_X', 'unit_idx'],
        ascending=[False, False, False, False, True],
    )

    # E-preferred candidates satisfy frac_E >= frac_B.
    e_pref = ranked_e[ranked_e['frac_E'] >= ranked_e['frac_B']]
    n_e_actual = min(n_e, len(e_pref))
    e_ids = set(e_pref.head(n_e_actual)['unit_idx'].astype(int))

    if n_e_actual < n_e:
        print(
            'Warning: E quota reduced to satisfy frac_E >= frac_B guardrail '
            f'({n_e_actual} < target {n_e}).'
        )

    fracs['field_eb'] = 'B'
    fracs.loc[fracs['unit_idx'].isin(e_ids), 'field_eb'] = 'E'
    fracs.loc[fracs['unit_idx'].isin(a_ids), 'field_eb'] = 'A'
    fracs.loc[fracs['unit_idx'].isin(x_ids), 'field_eb'] = 'X'

    out = par / 'institution_field_eb.parquet'
    fracs.to_parquet(str(out), index=False)

    counts = fracs['field_eb'].value_counts()
    print('field_eb')
    for c in ['E', 'B', 'A', 'X']:
        n = int(counts.get(c, 0))
        p = 100.0 * n / n_total if n_total else 0.0
        print(f'{c}  {n:5d}  ({p:5.2f}%)')

    # Sanity check: no B label should have frac_E > frac_B.
    b_bad = fracs[(fracs['field_eb'] == 'B') & (fracs['frac_E'] > fracs['frac_B'])]
    print(f'B with frac_E > frac_B: {len(b_bad)}')
    print(f'Total institutions: {len(fracs):,}')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
    print('FINISHED!')
