"""
institution_retention.py — Select τ_U (institution retention threshold).

For each of the 21 (t_x, F_x) corpus definitions and each τ_U ∈ {0, 5, 10, 15, 20},
reports the number of institutions retained (mean works/year ≥ τ_U) and the
proportion of total corpus works those institutions account for.

Rows:    21 (t_x, F_x) pairs
Columns: τ_U ∈ {0, 5, 10, 15, 20} × (inst_count, pct_works)

Requires source_master.parquet to have the field_subset column ('E'/'B'/NULL),
produced by journal_filter_match_oa.py.

Output: data/institution_retention.csv
"""

import sys
from pathlib import Path
import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_params

paths  = load_config()
params = load_params()
PARQUET = paths.parquet

# Time windows from params.yaml: t_x → (census_start, census_end, target_start, target_end)
_tw = params['time_windows']
TIME_WINDOWS = {
    tx: (w['census'][0], w['census'][1], w['target'][0], w['target'][1])
    for tx, w in _tw.items()
}

# Recommended τ_U floor per field subset (informational — logged at runtime)
TAU_U_FLOOR = params['tau_u_floor']   # {'E': 5, 'B': 5, 'A': 10}

TAU_U_VALUES = [0, 1, 2, 3, 4, 5, 10, 15, 20]

# Additional WHERE clause on source_master.field_subset for each F_x
FIELD_COND = {
    'E': "AND sm.field_subset = 'E'",
    'B': "AND sm.field_subset = 'B'",
    'A': "",
}


def compute_retention(db, tx: int, fx: str) -> pd.DataFrame:
    cs, ce, ts, te = TIME_WINDOWS[tx]
    min_year = min(cs, ts)
    max_year = max(ce, te)
    n_years  = max_year - min_year + 1
    fc = FIELD_COND[fx]

    return db.sql(f"""
        WITH inst_works AS (
            -- Works per institution within the (t_x, F_x) window
            SELECT a.institution_idx,
                   COUNT(DISTINCT a.work_idx)              AS works_count,
                   COUNT(DISTINCT a.work_idx) / {n_years}.0 AS works_per_year
            FROM '{PARQUET}/corpus_works.parquet' w
            JOIN '{PARQUET}/source_master.parquet' sm ON w.source_idx = sm.source_idx
            JOIN '{PARQUET}/corpus_authorships.parquet' a  ON w.work_idx = a.work_idx
            WHERE w.publication_year BETWEEN {min_year} AND {max_year}
              AND a.institution_idx IS NOT NULL
              {fc}
            GROUP BY a.institution_idx
        ),
        total AS (
            SELECT COUNT(DISTINCT a.work_idx) AS total_works
            FROM '{PARQUET}/corpus_works.parquet' w
            JOIN '{PARQUET}/source_master.parquet' sm ON w.source_idx = sm.source_idx
            JOIN '{PARQUET}/corpus_authorships.parquet' a  ON w.work_idx = a.work_idx
            WHERE w.publication_year BETWEEN {min_year} AND {max_year}
              AND a.institution_idx IS NOT NULL
              {fc}
        ),
        thresholds AS (
            SELECT unnest([{', '.join(str(v) for v in TAU_U_VALUES)}]) AS tau_u
        ),
        -- For each τ_U: set of retained institution_idxs
        retained_inst AS (
            SELECT t.tau_u, iw.institution_idx
            FROM inst_works iw
            CROSS JOIN thresholds t
            WHERE iw.works_per_year >= t.tau_u
        ),
        -- For each τ_U: distinct works and sources covered by any retained institution
        retained_works AS (
            SELECT ri.tau_u,
                   COUNT(DISTINCT a.work_idx)  AS ret_works,
                   COUNT(DISTINCT w.source_idx) AS ret_sources
            FROM retained_inst ri
            JOIN '{PARQUET}/corpus_authorships.parquet' a  ON ri.institution_idx = a.institution_idx
            JOIN '{PARQUET}/corpus_works.parquet' w        ON a.work_idx = w.work_idx
            JOIN '{PARQUET}/source_master.parquet' sm      ON w.source_idx = sm.source_idx
            WHERE w.publication_year BETWEEN {min_year} AND {max_year}
              {fc}
            GROUP BY ri.tau_u
        )
        SELECT
            ri.tau_u,
            COUNT(DISTINCT ri.institution_idx)            AS retained_inst,
            rw.ret_works                                  AS retained_works,
            ROUND(rw.ret_works * 100.0 / t.total_works, 1) AS pct_works,
            rw.ret_sources                                AS retained_sources
        FROM retained_inst ri
        JOIN retained_works rw USING (tau_u)
        CROSS JOIN total t
        GROUP BY ri.tau_u, rw.ret_works, rw.ret_sources, t.total_works
        ORDER BY ri.tau_u
    """).df()


def build_table(db) -> pd.DataFrame:
    rows = []
    for tx in range(1, 8):
        for fx in ['E', 'B', 'A']:
            print(f"  t_x={tx}  F_x={fx} ...", end='  ', flush=True)
            df = compute_retention(db, tx, fx)
            row = {'t_x': tx, 'F_x': fx}
            for _, r in df.iterrows():
                tau = int(r['tau_u'])
                row[f'tau{tau}_inst']    = int(r['retained_inst'])
                row[f'tau{tau}_pct']     = round(float(r['pct_works']), 1)
                row[f'tau{tau}_sources'] = int(r['retained_sources'])
            floor = TAU_U_FLOOR[fx]
            floor_row = df.loc[df.tau_u == floor]
            floor_str = (f"  τ_U≥{floor}: {int(floor_row['retained_inst'].iloc[0]):,} inst, "
                         f"{floor_row['pct_works'].iloc[0]:.1f}% works"
                         if not floor_row.empty else "")
            print(f"total works = {int(df.loc[df.tau_u == 0, 'retained_works'].iloc[0]):,}{floor_str}")
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    with duckdb.connect() as db:
        db.sql(f"""
            SET temp_directory = '{paths.working}/.tmp';
            SET memory_limit   = '56GB';
        """)
        print("Computing institution retention table (21 rows × 5 τ_U values)...")
        table = build_table(db)

    out_path = paths.data / 'institution_retention.csv'
    table.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
    print('FINISHED!')
