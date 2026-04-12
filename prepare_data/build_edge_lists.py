"""
build_edge_lists.py — Build pre-projection citer–cited edge lists.

For each corpus configuration derived from params.csv, writes one edge list
table and one unit index table to WORKING/edge_lists.duckdb.

Edge list table schema
----------------------
citer_work_idx       BIGINT   -- citing work
citer_source_idx     BIGINT   -- source of citing work
citer_inst_idx       BIGINT   -- institution of citing work (one row per retained inst)
cited_work_idx       BIGINT   -- cited work
cited_source_idx     BIGINT   -- source of cited work
cited_inst_idx       BIGINT   -- institution of cited work (one row per retained inst)
inst_weight          DOUBLE   -- ω_iu author-fractional (paper eq. 1), citing side
direct_inst_weight   DOUBLE   -- 1 / n_retained_institutions_of_citing_work
cited_inst_weight    DOUBLE   -- ω_jv author-fractional (paper eq. 1), cited side
R_i                  BIGINT   -- intra-corpus reference count of citing work
a_citer_source       BIGINT   -- work count of citer source in this corpus
a_cited_source       BIGINT   -- work count of cited source in this corpus
a_citer_inst         DOUBLE   -- fractional work count of citer institution (Σ_i ω_iu)
a_cited_inst         DOUBLE   -- fractional work count of cited institution

Table naming
------------
  el_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}
  _units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}

  run_code  8-char string: last-2-digits of tc0,tc1,tt0,tt1  e.g. '20242024'

At matrix build time supply:
  ρ ∈ {0,1}  →  full: weight 1; fixed: weight R̄/R_i
  m ∈ {0,1}⁴ →  block mask for SS/SI/IS/II
  χ ∈ [0,1]  →  source–institution mixing

Institution retention
---------------------
For fx='A' the retained institution set is computed from the A corpus itself.
For all field subsets (E, B, EB, NEB, X) the institution set is inherited from
the corresponding A corpus (_units_{run_code}_A_tauU{tau_u}_tauS{tau_s}).
A must therefore be built before its field subsets within each
(run_code, tau_u, tau_s) group.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import duckdb
from scipy.sparse import csr_matrix, bmat as sp_bmat
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

paths   = load_config()
PARQUET = paths.parquet
DB_PATH = paths.working / 'edge_lists.duckdb'

FIELD_COND = {
    'E':   "AND sm.field_eb = 'E'",
    'B':   "AND sm.field_eb = 'B'",
    'EB':  "AND sm.field_eb IN ('E', 'B', 'A')",
    'NEB': "AND sm.field_eb IS NULL",
    'X':   "AND sm.field_eb = 'X'",
    'A':   "",
}


def table_name(run_code: str, fx: str, tau_u: int, tau_s: int) -> str:
    return f'el_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}'


def corpus_configs_from_csv() -> list:
    """
    Derive unique corpus configurations from params.csv.
    Returns list of dicts ordered so that fx='A' precedes non-A within
    each (run_code, tau_u, tau_s) group.
    """
    rows = load_runs()
    seen = set()
    configs = []
    for r in rows:
        key = (r['run_code'], r['tc0'], r['tc1'], r['tt0'], r['tt1'],
               r['fx'], r['tau_u'], r['tau_s'])
        if key not in seen:
            seen.add(key)
            configs.append({
                'run_code': r['run_code'],
                'tc0': r['tc0'], 'tc1': r['tc1'],
                'tt0': r['tt0'], 'tt1': r['tt1'],
                'fx':  r['fx'],
                'tau_u': r['tau_u'],
                'tau_s': r['tau_s'],
            })

    def sort_key(c):
        # A first within each (run_code, tau_u, tau_s) group
        return (c['run_code'], c['tau_u'], c['tau_s'],
                0 if c['fx'] == 'A' else 1, c['fx'])

    return sorted(configs, key=sort_key)


def build_one(db, run_code: str, tc0: int, tc1: int, tt0: int, tt1: int,
              fx: str, tau_u: int, tau_s: int,
              inherited_inst_table: str = None) -> int:
    """
    Build one edge list table.

    Parameters
    ----------
    inherited_inst_table : str or None
        If provided, institution retention is read from this table
        (unit_type='U' rows) instead of being computed from the corpus.
        Pass the _units table of the corresponding A corpus.
    """
    cs, ce = tc0, tc1   # census window
    ts, te = tt0, tt1   # target window
    min_year     = min(cs, ts)
    max_year     = max(ce, te)
    census_years = ce - cs + 1
    fc           = FIELD_COND[fx]
    tname        = table_name(run_code, fx, tau_u, tau_s)

    db.execute(f"""
        CREATE OR REPLACE TEMP TABLE _fw_tmp AS
        SELECT w.work_idx, w.source_idx, w.publication_year
        FROM '{PARQUET}/corpus_works.parquet' w
        JOIN '{PARQUET}/source_master.parquet' sm ON w.source_idx = sm.source_idx
        WHERE w.publication_year BETWEEN {min_year} AND {max_year}
          AND sm.has_corpus_refs = true
        {fc}
    """)

    db.execute(f"""
        CREATE OR REPLACE TEMP TABLE _auths_tmp AS
        SELECT DISTINCT work_idx, author_idx, institution_idx
        FROM '{PARQUET}/corpus_authorships.parquet'
        WHERE institution_idx IS NOT NULL
          AND work_idx IN (SELECT work_idx FROM _fw_tmp)
    """)

    if inherited_inst_table:
        retained_inst_sql = f"""        retained_inst AS (
            SELECT unit_idx AS institution_idx
            FROM {inherited_inst_table}
            WHERE unit_type = 'U'
        ),"""
    else:
        retained_inst_sql = f"""        retained_inst AS (
            SELECT institution_idx
            FROM iw_raw
            WHERE work_idx IN (SELECT work_idx FROM fw_census)
            GROUP BY institution_idx
            HAVING COUNT(DISTINCT work_idx) / {census_years}.0 >= {tau_u}
        ),"""

    db.execute(f"""
        CREATE OR REPLACE TABLE {tname} AS
        WITH
        fw AS (SELECT * FROM _fw_tmp),
        fw_census AS (
            SELECT work_idx FROM fw
            WHERE publication_year BETWEEN {cs} AND {ce}
        ),
        work_author_counts AS (
            SELECT work_idx,
                   COUNT(DISTINCT author_idx)      AS n_authors,
                   COUNT(DISTINCT institution_idx) AS n_institutions
            FROM _auths_tmp
            GROUP BY work_idx
        ),
        author_inst_counts AS (
            SELECT work_idx, author_idx,
                   COUNT(DISTINCT institution_idx) AS n_inst_per_author
            FROM _auths_tmp
            GROUP BY work_idx, author_idx
        ),
        iw_raw AS (
            SELECT a.work_idx,
                   a.institution_idx,
                   SUM(1.0 / wac.n_authors / aic.n_inst_per_author) AS inst_weight,
                   ANY_VALUE(1.0 / wac.n_institutions)               AS direct_inst_weight
            FROM _auths_tmp a
            JOIN work_author_counts wac ON a.work_idx = wac.work_idx
            JOIN author_inst_counts aic ON a.work_idx = aic.work_idx
                                       AND a.author_idx = aic.author_idx
            GROUP BY a.work_idx, a.institution_idx
        ),
{retained_inst_sql}
        retained_source AS (
            SELECT source_idx
            FROM fw
            WHERE work_idx IN (SELECT work_idx FROM fw_census)
            GROUP BY source_idx
            HAVING COUNT(DISTINCT work_idx) / {census_years}.0 >= {tau_s}
        ),
        iw AS (
            SELECT * FROM iw_raw
            WHERE institution_idx IN (SELECT institution_idx FROM retained_inst)
        ),
        retained_works AS (
            SELECT DISTINCT work_idx FROM iw
        ),
        rr AS (
            SELECT r.citer_idx, r.cited_idx
            FROM '{PARQUET}/corpus_references.parquet' r
            JOIN fw wc ON r.citer_idx = wc.work_idx
            JOIN fw wd ON r.cited_idx  = wd.work_idx
            WHERE r.citer_idx IN (SELECT work_idx FROM retained_works)
              AND r.cited_idx  IN (SELECT work_idx FROM retained_works)
              AND r.citer_idx != r.cited_idx
              AND wc.publication_year BETWEEN {cs} AND {ce}
              AND wd.publication_year BETWEEN {ts} AND {te}
              AND wd.publication_year <= wc.publication_year + 1
        ),
        R_i AS (
            SELECT citer_idx AS work_idx, COUNT(*) AS ref_count
            FROM rr GROUP BY citer_idx
        ),
        a_source AS (
            SELECT fw.source_idx,
                   COUNT(DISTINCT fw.work_idx) AS source_works
            FROM fw
            WHERE fw.work_idx IN (SELECT work_idx FROM retained_works)
              AND fw.source_idx IN (SELECT source_idx FROM retained_source)
            GROUP BY fw.source_idx
        ),
        a_inst AS (
            SELECT institution_idx,
                   SUM(inst_weight) AS inst_frac_works
            FROM iw GROUP BY institution_idx
        ),
        citer AS (
            SELECT iw.work_idx,
                   fw.source_idx,
                   iw.institution_idx,
                   iw.inst_weight,
                   iw.direct_inst_weight,
                   ri.ref_count,
                   acs.source_works    AS a_citer_source,
                   ain.inst_frac_works AS a_citer_inst
            FROM iw
            JOIN fw        ON iw.work_idx        = fw.work_idx
            JOIN R_i ri    ON iw.work_idx        = ri.work_idx
            JOIN a_source acs ON fw.source_idx    = acs.source_idx
            JOIN a_inst ain   ON iw.institution_idx = ain.institution_idx
        ),
        cited AS (
            SELECT iw.work_idx,
                   fw.source_idx,
                   iw.institution_idx,
                   iw.inst_weight      AS cited_inst_weight,
                   acs.source_works    AS a_cited_source,
                   ain.inst_frac_works AS a_cited_inst
            FROM iw
            JOIN fw        ON iw.work_idx        = fw.work_idx
            JOIN a_source acs ON fw.source_idx    = acs.source_idx
            JOIN a_inst ain   ON iw.institution_idx = ain.institution_idx
        )
        SELECT
            r.citer_idx          AS citer_work_idx,
            ci.source_idx        AS citer_source_idx,
            ci.institution_idx   AS citer_inst_idx,
            r.cited_idx          AS cited_work_idx,
            cj.source_idx        AS cited_source_idx,
            cj.institution_idx   AS cited_inst_idx,
            ci.inst_weight,
            ci.direct_inst_weight,
            cj.cited_inst_weight,
            ci.ref_count         AS R_i,
            ci.a_citer_source,
            cj.a_cited_source,
            ci.a_citer_inst,
            cj.a_cited_inst
        FROM rr r
        JOIN citer ci ON r.citer_idx = ci.work_idx
        JOIN cited cj ON r.cited_idx = cj.work_idx
    """)

    db.execute("DROP TABLE IF EXISTS _fw_tmp")
    db.execute("DROP TABLE IF EXISTS _auths_tmp")

    return db.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]


def build_units(db, run_code: str, fx: str, tau_u: int, tau_s: int) -> int:
    """
    Build the unit index table _units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}.

    Derives all sources and institutions that appear in the edge list together
    with their a_p work counts.
    """
    tname = table_name(run_code, fx, tau_u, tau_s)
    uname = f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}'

    db.execute(f"""
        CREATE OR REPLACE TABLE {uname} AS
        SELECT unit_idx, unit_type, MAX(a_p) AS a_p
        FROM (
            SELECT citer_source_idx AS unit_idx, 'S' AS unit_type,
                   CAST(a_citer_source AS DOUBLE) AS a_p
            FROM {tname}
            UNION ALL
            SELECT cited_source_idx  AS unit_idx, 'S' AS unit_type,
                   CAST(a_cited_source AS DOUBLE) AS a_p
            FROM {tname}
            UNION ALL
            SELECT citer_inst_idx AS unit_idx, 'U' AS unit_type,
                   a_citer_inst AS a_p
            FROM {tname}
            UNION ALL
            SELECT cited_inst_idx AS unit_idx, 'U' AS unit_type,
                   a_cited_inst AS a_p
            FROM {tname}
        )
        GROUP BY unit_idx, unit_type
        ORDER BY unit_type, unit_idx
    """)
    return db.execute(f"SELECT COUNT(*) FROM {uname}").fetchone()[0]


def filter_singletons(db, run_code: str, fx: str, tau_u: int, tau_s: int) -> tuple:
    """
    Remove units not in the giant SCC of their governing graph, then rebuild
    the units table.  Iterates until stable.

    Returns (total_sources_dropped, total_insts_dropped).
    """
    tname = table_name(run_code, fx, tau_u, tau_s)
    uname = f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}'
    total_s, total_u = 0, 0

    for iteration in range(20):
        units = db.execute(
            f"SELECT unit_idx, unit_type FROM {uname}"
        ).fetchdf()
        src_ids  = units[units['unit_type'] == 'S']['unit_idx'].to_numpy(dtype=np.int64)
        inst_ids = units[units['unit_type'] == 'U']['unit_idx'].to_numpy(dtype=np.int64)
        n_s, n_u = len(src_ids), len(inst_ids)

        if n_s == 0 and n_u == 0:
            break

        src_index  = pd.Index(src_ids)
        inst_index = pd.Index(inst_ids)

        def _block(q, row_ix, col_ix, shape):
            df = db.execute(q).fetchdf()
            if len(df) == 0:
                return csr_matrix(shape)
            r = row_ix.get_indexer(df.iloc[:, 0].to_numpy(dtype=np.int64))
            c = col_ix.get_indexer(df.iloc[:, 1].to_numpy(dtype=np.int64))
            v = df.iloc[:, 2].to_numpy(dtype=np.float64)
            mask = (r >= 0) & (c >= 0)
            return csr_matrix((v[mask], (r[mask], c[mask])), shape=shape)

        C_SS = _block(
            f"SELECT citer_source_idx, cited_source_idx, COUNT(*) FROM {tname} GROUP BY 1,2",
            src_index, src_index, (n_s, n_s))
        C_SI = _block(
            f"SELECT citer_source_idx, cited_inst_idx, COUNT(*) FROM {tname} GROUP BY 1,2",
            src_index, inst_index, (n_s, n_u))
        C_IS = _block(
            f"SELECT citer_inst_idx, cited_source_idx, COUNT(*) FROM {tname} GROUP BY 1,2",
            inst_index, src_index, (n_u, n_s))
        C_II = _block(
            f"SELECT citer_inst_idx, cited_inst_idx, COUNT(*) FROM {tname} GROUP BY 1,2",
            inst_index, inst_index, (n_u, n_u))

        # Single SCC on the full node set: sources 0..n_s-1, institutions n_s..n_s+n_u-1.
        # Using C_full (all four blocks) means connectivity through any path —
        # SS, SI, IS, II — is respected.  A source with no SS edges but SI/IS
        # connections is correctly kept; previously it was wrongly dropped by
        # a separate connected_components(C_SS) call.
        from collections import Counter
        C_full = sp_bmat([[C_SS, C_SI], [C_IS, C_II]], format='csr')
        _, labels_full = connected_components(C_full, directed=True, connection='strong')
        giant_full = Counter(labels_full).most_common(1)[0][0]
        drop_src  = src_ids[labels_full[:n_s] != giant_full]
        drop_inst = inst_ids[labels_full[n_s:] != giant_full]

        if len(drop_src) == 0 and len(drop_inst) == 0:
            print(f"    filter_singletons: stable after {iteration} pass(es)")
            break

        print(f"    filter_singletons pass {iteration+1}: "
              f"drop {len(drop_src)} sources, {len(drop_inst)} institutions")
        total_s += len(drop_src)
        total_u += len(drop_inst)

        if len(drop_src) > 0:
            drop_src_df = pd.DataFrame({'idx': drop_src})
            db.register('_drop_src', drop_src_df)
            db.execute(f"""
                DELETE FROM {tname}
                WHERE citer_source_idx IN (SELECT idx FROM _drop_src)
                   OR cited_source_idx  IN (SELECT idx FROM _drop_src)
            """)
            db.unregister('_drop_src')

        if len(drop_inst) > 0:
            drop_inst_df = pd.DataFrame({'idx': drop_inst})
            db.register('_drop_inst', drop_inst_df)
            db.execute(f"""
                DELETE FROM {tname}
                WHERE citer_inst_idx IN (SELECT idx FROM _drop_inst)
                   OR cited_inst_idx  IN (SELECT idx FROM _drop_inst)
            """)
            db.unregister('_drop_inst')

        build_units(db, run_code, fx, tau_u, tau_s)

    return total_s, total_u


def ensure_catalog(db):
    db.execute("""
        CREATE TABLE IF NOT EXISTS _catalog (
            table_name     VARCHAR PRIMARY KEY,
            run_code       VARCHAR,
            F_x            VARCHAR,
            tau_u          INTEGER,
            tau_s          INTEGER,
            n_rows         BIGINT,
            n_sources      INTEGER,
            n_institutions INTEGER,
            created_at     VARCHAR
        )
    """)
    # Migrate pre-run_code schema
    cols = {row[0] for row in db.execute("DESCRIBE _catalog").fetchall()}
    if 'run_code' not in cols:
        db.execute("ALTER TABLE _catalog ADD COLUMN run_code VARCHAR")
    if 'tau_s' not in cols:
        db.execute("ALTER TABLE _catalog ADD COLUMN tau_s INTEGER")
        db.execute("UPDATE _catalog SET tau_s = 0")


def update_catalog(db, run_code: str, fx: str, tau_u: int, tau_s: int, n_rows: int):
    tname = table_name(run_code, fx, tau_u, tau_s)
    n_sources = db.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT citer_source_idx AS s FROM {tname}
            UNION
            SELECT DISTINCT cited_source_idx FROM {tname}
        )
    """).fetchone()[0]
    n_inst = db.execute(
        f"SELECT COUNT(DISTINCT citer_inst_idx) FROM {tname}"
    ).fetchone()[0]
    db.execute(
        """INSERT OR REPLACE INTO _catalog
           (table_name, run_code, F_x, tau_u, tau_s, n_rows, n_sources, n_institutions, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        [tname, run_code, fx, tau_u, tau_s, n_rows, n_sources, n_inst,
         datetime.now().isoformat(timespec='seconds')]
    )


def clean_stale(db) -> None:
    """Drop edge-list and units tables not in the current schedule."""
    configs = corpus_configs_from_csv()
    expected = set()
    for c in configs:
        expected.add(table_name(c['run_code'], c['fx'], c['tau_u'], c['tau_s']))
        expected.add(f"_units_{c['run_code']}_{c['fx']}_tauU{c['tau_u']}_tauS{c['tau_s']}")

    import re
    _mode_suffix = re.compile(r'_m[01]{4}$')
    all_tables = {row[0] for row in db.execute('SHOW TABLES').fetchall()}
    stale = [t for t in all_tables
             if (t.startswith('el_') or t.startswith('_units_'))
             and not _mode_suffix.search(t)   # leave mode-specific tables to filter_mode_units.py
             and t not in expected]

    for t in sorted(stale):
        db.execute(f'DROP TABLE IF EXISTS {t}')
        db.execute("DELETE FROM _catalog WHERE table_name = ?", [t])
        print(f'  Dropped stale table: {t}')

    if not stale:
        print('  No stale tables found.')


def main():
    configs = corpus_configs_from_csv()

    with duckdb.connect(str(DB_PATH)) as db:
        db.execute(f"SET temp_directory = '{paths.working}/.tmp'")
        db.execute("SET memory_limit = '56GB'")
        ensure_catalog(db)
        print('=== Cleaning stale tables ===')
        clean_stale(db)

        # Group configs by (run_code, tau_u, tau_s); A is already first within each group
        from itertools import groupby
        key_fn = lambda c: (c['run_code'], c['tau_u'], c['tau_s'])
        for group_key, group in groupby(configs, key=key_fn):
            run_code, tau_u, tau_s = group_key
            a_units_table = f'_units_{run_code}_A_tauU{tau_u}_tauS{tau_s}'

            for c in group:
                fx    = c['fx']
                tc0, tc1, tt0, tt1 = c['tc0'], c['tc1'], c['tt0'], c['tt1']
                tname = table_name(run_code, fx, tau_u, tau_s)
                inherited = None if fx == 'A' else a_units_table

                print(f"  Building {tname} ...", end='  ', flush=True)
                build_one(db, run_code, tc0, tc1, tt0, tt1, fx, tau_u, tau_s,
                          inherited_inst_table=inherited)
                build_units(db, run_code, fx, tau_u, tau_s)
                n_s, n_u = filter_singletons(db, run_code, fx, tau_u, tau_s)
                uname = f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}'
                n_units_final = db.execute(f"SELECT COUNT(*) FROM {uname}").fetchone()[0]
                n_rows_final  = db.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
                update_catalog(db, run_code, fx, tau_u, tau_s, n_rows_final)
                print(f"{n_rows_final:,} rows  Units: {n_units_final}  "
                      f"(dropped {n_s} sources, {n_u} insts as non-giant-SCC)",
                      flush=True)

        print("\n=== Catalog ===")
        db.sql("SELECT * FROM _catalog ORDER BY run_code, F_x, tau_u").show()

        # Sample baseline edge list
        baseline_tname = table_name('20242024', 'A', 20, 20)
        print(f"\n=== Baseline edge list sample ({baseline_tname}) ===")
        db.sql(f"SELECT * FROM {baseline_tname} LIMIT 20").show()


if __name__ == '__main__':
    main()
    print('FINISHED!')
