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

Tables are named  el_t{t_x}_{F_x}_tau{τ_U},  e.g. el_t5_A_tau10.
A _catalog table records parameters and summary statistics.
"""

import sys
from pathlib import Path
from datetime import datetime
import duckdb

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

FIELD_COND = {
    'E': "AND sm.field_subset = 'E'",
    'B': "AND sm.field_subset = 'B'",
    'A': "",
}


def table_name(tx: int, fx: str, tau_u: int) -> str:
    return f'el_t{tx}_{fx}_tau{tau_u}'


def build_one(db, tx: int, fx: str, tau_u: int) -> int:
    cs, ce, ts, te = TIME_WINDOWS[tx]
    min_year     = min(cs, ts)
    max_year     = max(ce, te)
    census_years = ce - cs + 1
    fc           = FIELD_COND[fx]
    tname    = table_name(tx, fx, tau_u)

    db.execute(f"""
        CREATE OR REPLACE TABLE {tname} AS
        WITH
        -- ── Works spanning both census and target windows ────────────────────
        fw AS (
            SELECT w.work_idx, w.source_idx, w.publication_year
            FROM '{PARQUET}/corpus_works.parquet' w
            JOIN '{PARQUET}/source_master.parquet' sm ON w.source_idx = sm.source_idx
            WHERE w.publication_year BETWEEN {min_year} AND {max_year}
            {fc}
        ),
        -- ── Census-window works only — used for τ_U retention filter ─────────
        fw_census AS (
            SELECT work_idx FROM fw
            WHERE publication_year BETWEEN {cs} AND {ce}
        ),
        -- ── Per-work author and institution counts (for weight computation) ─
        work_author_counts AS (
            SELECT work_idx,
                   COUNT(DISTINCT author_idx)     AS n_authors,
                   COUNT(DISTINCT institution_idx) AS n_institutions
            FROM '{PARQUET}/corpus_authorships.parquet'
            WHERE institution_idx IS NOT NULL
              AND work_idx IN (SELECT work_idx FROM fw)
            GROUP BY work_idx
        ),
        author_inst_counts AS (
            SELECT work_idx, author_idx,
                   COUNT(DISTINCT institution_idx) AS n_inst_per_author
            FROM '{PARQUET}/corpus_authorships.parquet'
            WHERE institution_idx IS NOT NULL
              AND work_idx IN (SELECT work_idx FROM fw)
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
            FROM (SELECT DISTINCT work_idx, author_idx, institution_idx
                  FROM '{PARQUET}/corpus_authorships.parquet'
                  WHERE institution_idx IS NOT NULL
                    AND work_idx IN (SELECT work_idx FROM fw)) a
            JOIN work_author_counts wac ON a.work_idx = wac.work_idx
            JOIN author_inst_counts aic ON a.work_idx = aic.work_idx
                                       AND a.author_idx = aic.author_idx
            GROUP BY a.work_idx, a.institution_idx
        ),
        -- ── Retained institutions: mean census works / census year ≥ τ_U ─────
        retained_inst AS (
            SELECT institution_idx
            FROM iw_raw
            WHERE work_idx IN (SELECT work_idx FROM fw_census)
            GROUP BY institution_idx
            HAVING COUNT(DISTINCT work_idx) / {census_years}.0 >= {tau_u}
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

    return db.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]


def build_units(db, tx: int, fx: str, tau_u: int) -> int:
    """
    Build the unit index table _units_t{tx}_{fx}_tau{tau_u} in edge_lists.duckdb.

    Derives all sources and institutions that appear in the edge list together
    with their a_p work counts (integer for sources, fractional for institutions).
    Called immediately after build_one() for each corpus.

    Note: units that have retained works but zero intra-corpus references (isolated
    nodes) are absent from the edge list and therefore absent from this table.
    They are very rare in a dense citation corpus; add a parquet-based query here
    if isolated-node correctness becomes necessary.
    """
    tname = table_name(tx, fx, tau_u)
    uname = f'_units_t{tx}_{fx}_tau{tau_u}'

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


def ensure_catalog(db):
    db.execute("""
        CREATE TABLE IF NOT EXISTS _catalog (
            table_name     VARCHAR PRIMARY KEY,
            t_x            INTEGER,
            F_x            VARCHAR,
            tau_u          INTEGER,
            n_rows         BIGINT,
            n_sources      INTEGER,
            n_institutions INTEGER,
            created_at     VARCHAR
        )
    """)


def update_catalog(db, tx: int, fx: str, tau_u: int, n_rows: int):
    tname = table_name(tx, fx, tau_u)
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
        "INSERT OR REPLACE INTO _catalog VALUES (?,?,?,?,?,?,?,?)",
        [tname, tx, fx, tau_u, n_rows, n_sources, n_inst,
         datetime.now().isoformat(timespec='seconds')]
    )


def main():
    with duckdb.connect(str(DB_PATH)) as db:
        db.execute(f"SET temp_directory = '{paths.working}/.tmp'")
        db.execute("SET memory_limit = '56GB'")
        ensure_catalog(db)

        for tx in range(1, 8):
            for fx in ['E', 'B', 'A']:
                tau_u = TAU_U_FLOOR[fx]
                tname = table_name(tx, fx, tau_u)
                print(f"  Building {tname} ...", end='  ', flush=True)
                n = build_one(db, tx, fx, tau_u)
                update_catalog(db, tx, fx, tau_u, n)
                n_units = build_units(db, tx, fx, tau_u)
                print(f"  Units: {n_units} (sources + institutions)", flush=True)
                print(f"{n:,} rows", flush=True)

        print("\n=== Catalog ===")
        db.sql("SELECT * FROM _catalog ORDER BY t_x, F_x, tau_u").show()

        print("\n=== Baseline edge list sample (el_t5_A_tau10) ===")
        db.sql("SELECT * FROM el_t5_A_tau10 LIMIT 20").show()


if __name__ == '__main__':
    main()
    print('FINISHED!')
