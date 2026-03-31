"""
build_edge_lists.py — Build pre-projection citer–cited edge lists.

For each (t_x, F_x, τ_U) corpus definition, writes one table to
WORKING/edge_lists.duckdb at (citer_work × citer_institution ×
cited_work × cited_institution) granularity.

Schema
------
citer_work_idx       BIGINT   -- citing work
citer_source_idx      BIGINT   -- source of citing work
citer_inst_idx       BIGINT   -- institution of citing work (one row per retained inst)
cited_work_idx       BIGINT   -- cited work
cited_source_idx      BIGINT   -- source of cited work
cited_inst_idx       BIGINT   -- institution of cited work (one row per retained inst)
inst_weight          DOUBLE   -- ω_iu author-fractional (paper eq. 1), citing side
direct_inst_weight   DOUBLE   -- 1 / n_retained_institutions_of_citing_work
cited_inst_weight    DOUBLE   -- ω_jv author-fractional (paper eq. 1), cited side
R_i                  BIGINT   -- intra-corpus reference count of citing work
a_citer_source       BIGINT   -- work count of citer source in this corpus
a_cited_source       BIGINT   -- work count of cited source in this corpus
a_citer_inst         DOUBLE   -- fractional work count of citer institution (Σ_i ω_iu)
a_cited_inst         DOUBLE   -- fractional work count of cited institution

At matrix build time supply:
  ρ ∈ {0,1}  →  full: weight 1; fixed: weight R̄/R_i  (R̄ = corpus mean)
  m ∈ {0,1}⁴ →  block mask for SS/SI/IS/II
  χ ∈ [0,1]  →  source–institution mixing

SS block: GROUP BY (citer_source, cited_source)  — deduplicate on (work pair)
          to avoid counting once per institution combination
SI block: GROUP BY (citer_source, cited_inst)    — deduplicate on citer work
IS block: GROUP BY (citer_inst,   cited_source)  — deduplicate on cited work
II block: no deduplication needed (full cross product intended)

Tables are named  el_t{t_x}_{F_x}_tau{τ_U},  e.g. el_t5_A_tau20.
A _catalog table records parameters and summary statistics.

Institution retention
---------------------
For F=A the retained institution set is computed from the A corpus itself
(mean census works ≥ τ_U).  For all field subsets (E, B, EB, NEB) the
institution set is *inherited* from the corresponding A corpus — i.e. from
_units_t{tx}_A_tauU{tau_u}_tauS{tau_s} — so that the source and institution sets of the
field subsets are exact partitions of the A corpus.  A must therefore be
built before its field subsets.
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
from util import load_config, load_params

paths  = load_config()
params = load_params()
PARQUET = paths.parquet
DB_PATH = paths.working / 'edge_lists.duckdb'

_tw = params['time_windows']
TIME_WINDOWS = {
    tx: (w['census'][0], w['census'][1], w['target'][0], w['target'][1])
    for tx, w in _tw.items()
}
TAU_U_FLOOR = params['tau_u_floor']
TAU_S_FLOOR = params['tau_s_floor']

FIELD_COND = {
    'E':   "AND sm.field_subset = 'E'",
    'B':   "AND sm.field_subset = 'B'",
    'EB':  "AND sm.field_subset IN ('E', 'B')",
    'NEB': "AND sm.field_subset IS NULL",
    'A':   "",
}

# Field subsets that inherit their institution set from the A corpus.
# A must be built first for each (tx, tau_u) before these are processed.
FIELD_SUBSETS = ['E', 'B', 'EB', 'NEB']


def table_name(tx: int, fx: str, tau_u: int, tau_s: int) -> str:
    return f'el_t{tx}_{fx}_tauU{tau_u}_tauS{tau_s}'


def build_one(db, tx: int, fx: str, tau_u: int, tau_s: int,
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
    cs, ce, ts, te = TIME_WINDOWS[tx]
    min_year     = min(cs, ts)
    max_year     = max(ce, te)
    census_years = ce - cs + 1
    fc           = FIELD_COND[fx]
    tname        = table_name(tx, fx, tau_u, tau_s)

    # ── Pre-materialise filtered parquets (one scan each) ──────────────────
    # _fw_tmp:    works in the year range and field subset; referenced 4×
    #             inside the main CTE so worth materialising.
    # _auths_tmp: deduplicated authorships for those works.  The original CTE
    #             scanned corpus_authorships.parquet 3× (work_author_counts,
    #             author_inst_counts, iw_raw); this reduces it to one scan.
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
        -- ── Works (from temp table) ───────────────────────────────────────
        fw AS (SELECT * FROM _fw_tmp),
        -- ── Census-window works only — used for τ_U retention filter ─────────
        fw_census AS (
            SELECT work_idx FROM fw
            WHERE publication_year BETWEEN {cs} AND {ce}
        ),
        -- ── Per-work author and institution counts (for weight computation) ─
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
        -- ── Institution weights per (work, institution) — pre-τ_U ──────────
        -- inst_weight      = ω_iu = Σ_ℓ (1/a_i)(1/u_iℓ)  [paper eq. 1]
        -- direct_inst_weight = 1 / n_distinct_institutions_of_work
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
        -- ── Retained institutions ────────────────────────────────────────────
{retained_inst_sql}
        -- ── Retained sources (mean annual census works ≥ τ_S) ───────────────
        retained_source AS (
            SELECT source_idx
            FROM fw
            WHERE work_idx IN (SELECT work_idx FROM fw_census)
            GROUP BY source_idx
            HAVING COUNT(DISTINCT work_idx) / {census_years}.0 >= {tau_s}
        ),
        -- ── Restrict weights to retained institutions ───────────────────────
        iw AS (
            SELECT * FROM iw_raw
            WHERE institution_idx IN (SELECT institution_idx FROM retained_inst)
        ),
        -- ── Retained works: at least one retained institution ───────────────
        retained_works AS (
            SELECT DISTINCT work_idx FROM iw
        ),
        -- ── Intra-corpus references: citer in census, cited in target ─────────
        -- Excludes self-loops and cited works > 1 year newer than citer
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
        -- ── R_i: intra-corpus reference count of each citing work ───────────
        R_i AS (
            SELECT citer_idx AS work_idx, COUNT(*) AS ref_count
            FROM rr GROUP BY citer_idx
        ),
        -- ── a_p: integer work count per source ─────────────────────────────
        a_source AS (
            SELECT fw.source_idx,
                   COUNT(DISTINCT fw.work_idx) AS source_works
            FROM fw
            WHERE fw.work_idx IN (SELECT work_idx FROM retained_works)
              AND fw.source_idx IN (SELECT source_idx FROM retained_source)
            GROUP BY fw.source_idx
        ),
        -- ── a_p: fractional work count per institution (Σ_i ω_iu) ──────────
        a_inst AS (
            SELECT institution_idx,
                   SUM(inst_weight) AS inst_frac_works
            FROM iw GROUP BY institution_idx
        ),
        -- ── Citer side: one row per (work, institution) ─────────────────────
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
        -- ── Cited side: one row per (work, institution) ─────────────────────
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
        -- ── Final cross product: one row per (citer_work × citer_inst ×
        --                                      cited_work × cited_inst) ──────
        SELECT
            r.citer_idx          AS citer_work_idx,
            ci.source_idx         AS citer_source_idx,
            ci.institution_idx   AS citer_inst_idx,
            r.cited_idx          AS cited_work_idx,
            cj.source_idx         AS cited_source_idx,
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


def build_units(db, tx: int, fx: str, tau_u: int, tau_s: int) -> int:
    """
    Build the unit index table _units_t{tx}_{fx}_tauU{tau_u}_tauS{tau_s} in edge_lists.duckdb.

    Derives all sources and institutions that appear in the edge list together
    with their a_p work counts (integer for sources, fractional for institutions).
    Called immediately after build_one() for each corpus.

    Note: units that have retained works but zero intra-corpus references (isolated
    nodes) are absent from the edge list and therefore absent from this table.
    They are very rare in a dense citation corpus; add a parquet-based query here
    if isolated-node correctness becomes necessary.
    """
    tname = table_name(tx, fx, tau_u, tau_s)
    uname = f'_units_t{tx}_{fx}_tauU{tau_u}_tauS{tau_s}'

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


def filter_singletons(db, tx: int, fx: str, tau_u: int, tau_s: int) -> tuple:
    """
    Remove units not in the giant SCC of their governing graph, then rebuild
    the units table.  Iterates until no further removals occur.

    Criterion:
      Sources     : must be in giant SCC of C_SS (strict).  This also excludes
                    sources that are connected only through institutional paths
                    (IN/OUT-component in C_SS) — they cite or are cited via
                    institutions but have no source-to-source cycle with the
                    main literature.
      Institutions: must be in giant SCC of the full joint C (lenient).

    Returns (total_sources_dropped, total_insts_dropped).
    """
    tname = table_name(tx, fx, tau_u, tau_s)
    uname = f'_units_t{tx}_{fx}_tauU{tau_u}_tauS{tau_s}'
    total_s, total_u = 0, 0

    for iteration in range(20):   # safety cap; typically 1–2 passes
        # ── Load current unit index ───────────────────────────────────────
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

        C_full = sp_bmat([[C_SS, C_SI], [C_IS, C_II]], format='csr')

        from collections import Counter

        # Sources: giant SCC of C_SS
        _, labels_ss = connected_components(C_SS, directed=True, connection='strong')
        giant_ss = Counter(labels_ss).most_common(1)[0][0]
        drop_src = src_ids[labels_ss != giant_ss]

        # Institutions: giant SCC of full joint C
        _, labels_full = connected_components(C_full, directed=True, connection='strong')
        giant_full = Counter(labels_full).most_common(1)[0][0]
        drop_inst = inst_ids[labels_full[n_s:] != giant_full]

        if len(drop_src) == 0 and len(drop_inst) == 0:
            print(f"    filter_singletons: stable after {iteration} pass(es)")
            break

        print(f"    filter_singletons pass {iteration+1}: "
              f"drop {len(drop_src)} sources, {len(drop_inst)} institutions")
        total_s += len(drop_src)
        total_u += len(drop_inst)

        # ── Delete edges involving dropped units ──────────────────────────
        # Use a temp table to avoid huge IN(...) lists
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

        # ── Rebuild units table ───────────────────────────────────────────
        build_units(db, tx, fx, tau_u, tau_s)

    return total_s, total_u


def ensure_catalog(db):
    db.execute("""
        CREATE TABLE IF NOT EXISTS _catalog (
            table_name     VARCHAR PRIMARY KEY,
            t_x            INTEGER,
            F_x            VARCHAR,
            tau_u          INTEGER,
            tau_s          INTEGER,
            n_rows         BIGINT,
            n_sources      INTEGER,
            n_institutions INTEGER,
            created_at     VARCHAR
        )
    """)
    # Migrate pre-tau_s schema: add column if absent
    cols = {row[0] for row in db.execute("DESCRIBE _catalog").fetchall()}
    if 'tau_s' not in cols:
        db.execute("ALTER TABLE _catalog ADD COLUMN tau_s INTEGER")
        db.execute("UPDATE _catalog SET tau_s = 0")


def update_catalog(db, tx: int, fx: str, tau_u: int, tau_s: int, n_rows: int):
    tname = table_name(tx, fx, tau_u, tau_s)
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
           (table_name, t_x, F_x, tau_u, tau_s, n_rows, n_sources, n_institutions, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        [tname, tx, fx, tau_u, tau_s, n_rows, n_sources, n_inst,
         datetime.now().isoformat(timespec='seconds')]
    )


def clean_stale(db) -> None:
    """Drop edge-list and units tables not in the current schedule."""
    expected = set()
    for tx in range(1, 8):
        for fx in ['A'] + FIELD_SUBSETS:
            tau_u = TAU_U_FLOOR[fx]
            tau_s = TAU_S_FLOOR[fx]
            expected.add(table_name(tx, fx, tau_u, tau_s))
            expected.add(f'_units_t{tx}_{fx}_tauU{tau_u}_tauS{tau_s}')

    all_tables = {row[0] for row in db.execute('SHOW TABLES').fetchall()}
    stale = [t for t in all_tables
             if (t.startswith('el_') or t.startswith('_units_'))
             and t not in expected]

    for t in sorted(stale):
        db.execute(f'DROP TABLE IF EXISTS {t}')
        db.execute("DELETE FROM _catalog WHERE table_name = ?", [t])
        print(f'  Dropped stale table: {t}')

    if not stale:
        print('  No stale tables found.')


def main():
    with duckdb.connect(str(DB_PATH)) as db:
        db.execute(f"SET temp_directory = '{paths.working}/.tmp'")
        db.execute("SET memory_limit = '56GB'")
        ensure_catalog(db)
        print('=== Cleaning stale tables ===')
        clean_stale(db)

        for tx in range(1, 8):
            # ── 1. Build A first to establish the canonical institution set ──
            tau_u_A = TAU_U_FLOOR['A']
            tau_s_A = TAU_S_FLOOR['A']
            tname_A = table_name(tx, 'A', tau_u_A, tau_s_A)
            a_units_table = f'_units_t{tx}_A_tauU{tau_u_A}_tauS{tau_s_A}'
            print(f"  Building {tname_A} ...", end='  ', flush=True)
            n = build_one(db, tx, 'A', tau_u_A, tau_s_A)
            build_units(db, tx, 'A', tau_u_A, tau_s_A)
            n_s, n_u = filter_singletons(db, tx, 'A', tau_u_A, tau_s_A)
            n_units_final = db.execute(f"SELECT COUNT(*) FROM {a_units_table}").fetchone()[0]
            n_rows_final  = db.execute(f"SELECT COUNT(*) FROM {tname_A}").fetchone()[0]
            update_catalog(db, tx, 'A', tau_u_A, tau_s_A, n_rows_final)
            print(f"{n_rows_final:,} rows  Units: {n_units_final}  "
                  f"(dropped {n_s} sources, {n_u} insts as non-giant-SCC)",
                  flush=True)

            # ── 2. Build field subsets inheriting institution set from A ─────
            for fx in FIELD_SUBSETS:
                tau_u = TAU_U_FLOOR[fx]
                tau_s = TAU_S_FLOOR[fx]
                tname = table_name(tx, fx, tau_u, tau_s)
                print(f"  Building {tname} ...", end='  ', flush=True)
                n = build_one(db, tx, fx, tau_u, tau_s,
                              inherited_inst_table=a_units_table)
                build_units(db, tx, fx, tau_u, tau_s)
                n_s, n_u = filter_singletons(db, tx, fx, tau_u, tau_s)
                n_units_final = db.execute(
                    f"SELECT COUNT(*) FROM _units_t{tx}_{fx}_tauU{tau_u}_tauS{tau_s}"
                ).fetchone()[0]
                n_rows_final = db.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
                update_catalog(db, tx, fx, tau_u, tau_s, n_rows_final)
                print(f"{n_rows_final:,} rows  Units: {n_units_final}  "
                      f"(dropped {n_s} sources, {n_u} insts as non-giant-SCC)",
                      flush=True)

        print("\n=== Catalog ===")
        db.sql("SELECT * FROM _catalog ORDER BY t_x, F_x, tau_u").show()

        _tau_u_a = TAU_U_FLOOR['A']
        _tau_s_a = TAU_S_FLOOR['A']
        _sample = table_name(5, 'A', _tau_u_a, _tau_s_a)
        print(f"\n=== Baseline edge list sample ({_sample}) ===")
        db.sql(f"SELECT * FROM {_sample} LIMIT 20").show()


if __name__ == '__main__':
    main()
    print('FINISHED!')
