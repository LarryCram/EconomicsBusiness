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
direct_cited_inst_weight DOUBLE -- 1 / n_retained_institutions_of_cited_work
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
For fx='EBAX' the retained institution set is computed from the full corpus.
For all field subsets (E, B, A, EBA, X) the institution set is inherited
from the corresponding EBAX corpus
(_units_{run_code}_EBAX_tauU{tau_u}_tauS{tau_s}).
EBAX must therefore be built before its field subsets within each
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

SX_IDX = 1   # sentinel source_idx for ε=1 runs (never equal to a real OA id)
IX_IDX = 1   # sentinel institution_idx for ε=1 runs

def _field_cond(fx: str) -> str:
    """SQL WHERE fragment derived from the set of letters in fx.
    fx letters must be a subset of {'E','B','A','X'}; EBAX → no filter."""
    letters = set(fx) & {'E', 'B', 'A', 'X'}
    if letters == {'E', 'B', 'A', 'X'}:
        return ""
    elif len(letters) == 1:
        return f"AND sm.field_eb = '{next(iter(letters))}'"
    else:
        quoted = ', '.join(f"'{c}'" for c in 'EBAX' if c in letters)
        return f"AND sm.field_eb IN ({quoted})"


def _tau_sfx(ref_units: str) -> str:
    """'_fixtau' when the unit set is inherited from a reference window;
    '_vartau' when derived from this window's own τ filter."""
    return '_fixtau' if ref_units else '_vartau'


def table_name(run_code: str, fx: str, tau_u: int, tau_s: int,
               ref_units: str = '', epsilon: int = 0) -> str:
    """Canonical edge-list table name.
    _vartau: unit set from this window's τ filter (default).
    _fixtau: unit set inherited from a reference window.
    _eps1:   includes cross-boundary sentinel edges (ε=1)."""
    eps_sfx = '_eps1' if epsilon else ''
    return f'el_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}{_tau_sfx(ref_units)}{eps_sfx}'


def _units_name(run_code: str, fx: str, tau_u: int, tau_s: int,
                ref_units: str = '', epsilon: int = 0) -> str:
    """Canonical raw (C_full SCC) units table name."""
    eps_sfx = '_eps1' if epsilon else ''
    return f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}{_tau_sfx(ref_units)}{eps_sfx}'


def corpus_configs_from_csv() -> list:
    """
    Derive unique corpus configurations from params.csv.
    Returns list of dicts ordered so that fx='EBAX' precedes non-EBAX within
    each (run_code, tau_u, tau_s) group.
    """
    rows = load_runs()
    seen = set()
    configs = []
    for r in rows:
        ref     = r.get('ref_units', '')
        epsilon = int(r.get('epsilon', 0))
        # ref_units and epsilon are part of the dedup key: eps1 runs for the
        # same (run_code, fx, tau_u, tau_s) produce distinct edge list tables.
        key = (r['run_code'], r['tc0'], r['tc1'], r['tt0'], r['tt1'],
               r['fx'], r['tau_u'], r['tau_s'], ref, epsilon)
        if key not in seen:
            seen.add(key)
            configs.append({
                'run_code':  r['run_code'],
                'tc0': r['tc0'], 'tc1': r['tc1'],
                'tt0': r['tt0'], 'tt1': r['tt1'],
                'fx':        r['fx'],
                'tau_u':     r['tau_u'],
                'tau_s':     r['tau_s'],
                'ref_units': ref,
                'epsilon':   epsilon,
            })

    def sort_key(c):
        # fixtau configs must come after all vartau configs so the reference
        # _units_..._vartau table is guaranteed to exist when they run.
        # Within each (run_code, tau_u, tau_s) group, EBAX comes before others.
        # epsilon=1 configs come after epsilon=0 of the same fx.
        has_ref = 1 if c['ref_units'] else 0
        return (has_ref, c['run_code'], c['tau_u'], c['tau_s'],
            0 if c['fx'] == 'EBAX' else 1, c['fx'], c['epsilon'])

    return sorted(configs, key=sort_key)


def build_one(db, run_code: str, tc0: int, tc1: int, tt0: int, tt1: int,
              fx: str, tau_u: int, tau_s: int,
              inherited_inst_table: str = None,
              inherited_src_table: str = None,
              ref_units: str = '', epsilon: int = 0) -> int:
    """
    Build one edge list table.

    Parameters
    ----------
    inherited_inst_table : str or None
        If provided, institution retention is read from this table
        (unit_type='U' rows) instead of being computed from the corpus.
        Used for field subsets (E/B/A/EBA/X) which inherit the EBAX-corpus
        institution set of the same window.
    inherited_src_table : str or None
        If provided, source retention is read from this table
        (unit_type='S' rows) instead of being computed from the corpus.
        Used for fixtau runs which inherit both source and institution sets
        from the reference window (ref_units non-empty).
    """
    cs, ce = tc0, tc1   # census window
    ts, te = tt0, tt1   # target window
    min_year     = min(cs, ts)
    max_year     = max(ce, te)
    census_years = ce - cs + 1
    fc           = _field_cond(fx)
    tname        = table_name(run_code, fx, tau_u, tau_s, ref_units, epsilon)

    db.execute(f"""
        CREATE OR REPLACE TEMP TABLE _fw_tmp AS
        SELECT w.work_idx, w.source_idx, w.publication_year
        FROM '{PARQUET}/corpus_works.parquet' w
        JOIN '{PARQUET}/source_master.parquet' sm ON w.source_idx = sm.source_idx
        WHERE w.publication_year BETWEEN {min_year} AND {max_year}
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
              AND institution_idx IN (
                  SELECT institution_idx FROM '{PARQUET}/corpus_institutions.parquet'
              )
            GROUP BY institution_idx
            HAVING COUNT(DISTINCT work_idx) / {census_years}.0 >= {tau_u}
        ),"""

    if inherited_src_table:
        retained_src_sql = f"""        retained_source AS (
            SELECT unit_idx AS source_idx
            FROM {inherited_src_table}
            WHERE unit_type = 'S'
        ),"""
    else:
        retained_src_sql = f"""        retained_source AS (
            SELECT source_idx
            FROM fw
            WHERE work_idx IN (SELECT work_idx FROM fw_census)
            GROUP BY source_idx
            HAVING COUNT(DISTINCT work_idx) / {census_years}.0 >= {tau_s}
        ),"""

    db.execute(f"""
        CREATE OR REPLACE TABLE {tname} AS
        WITH
        fw AS (SELECT * FROM _fw_tmp),
        fw_census AS (
            SELECT work_idx FROM fw
            WHERE publication_year BETWEEN {cs} AND {ce}
        ),
{retained_src_sql}
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
                   iw.inst_weight        AS cited_inst_weight,
                   iw.direct_inst_weight AS direct_cited_inst_weight,
                   acs.source_works      AS a_cited_source,
                   ain.inst_frac_works   AS a_cited_inst
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
            cj.direct_cited_inst_weight,
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

    if epsilon == 1:
        _add_epsilon_edges(db, tname, tc0, tc1, tt0, tt1)

    return db.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]


def _add_epsilon_edges(db, tname: str, tc0: int, tc1: int,
                       tt0: int, tt1: int) -> None:
    """
    Append cross-boundary sentinel edges to an existing edge list table.

    Two edge types are added:
      type1 — corpus citer  →  SX_IDX/IX_IDX cited   (supp_role='cited')
      type2 — SX_IDX/IX_IDX citer  →  corpus cited   (supp_role='citer')

    R_i for type1 citer rows is the intra-corpus reference count already in
    the table.  R_i for type2 (supp citer) rows is referenced_works_count
    from corpus_works_supp (total OA reference list length).

    a_p values for SX_IDX/IX_IDX are set to the count of distinct
    supplementary works of each boundary type; v for sentinels is set to
    NaN in katz_ranker.py so these values do not affect rankings.
    """
    cs, ce = tc0, tc1
    ts, te = tt0, tt1
    db.execute(f"""
        INSERT INTO {tname}
        WITH
        citer_info AS (
            SELECT DISTINCT
                citer_work_idx, citer_source_idx, citer_inst_idx,
                inst_weight, direct_inst_weight, R_i, a_citer_source, a_citer_inst
            FROM {tname}
        ),
        cited_info AS (
            SELECT DISTINCT
                cited_work_idx, cited_source_idx, cited_inst_idx,
                cited_inst_weight, direct_cited_inst_weight, a_cited_source, a_cited_inst
            FROM {tname}
        ),
        supp_cited_pairs AS (
            SELECT r.citer_idx, r.cited_idx
            FROM '{PARQUET}/corpus_references_supp.parquet' r
            JOIN '{PARQUET}/corpus_works.parquet'      wc ON r.citer_idx = wc.work_idx
            JOIN '{PARQUET}/corpus_works_supp.parquet' ws ON r.cited_idx  = ws.work_idx
            WHERE r.supp_role = 'cited'
              AND wc.publication_year BETWEEN {cs} AND {ce}
              AND ws.publication_year BETWEEN {ts} AND {te}
              AND ws.publication_year <= wc.publication_year + 1
              AND r.citer_idx IN (SELECT citer_work_idx FROM citer_info)
        ),
        supp_citer_pairs AS (
            SELECT r.citer_idx, r.cited_idx
            FROM '{PARQUET}/corpus_references_supp.parquet' r
            JOIN '{PARQUET}/corpus_works_supp.parquet' ws ON r.citer_idx = ws.work_idx
            JOIN '{PARQUET}/corpus_works.parquet'      wd ON r.cited_idx  = wd.work_idx
            WHERE r.supp_role = 'citer'
              AND ws.publication_year BETWEEN {cs} AND {ce}
              AND wd.publication_year BETWEEN {ts} AND {te}
              AND wd.publication_year <= ws.publication_year + 1
              AND r.cited_idx IN (SELECT cited_work_idx FROM cited_info)
        ),
        sx_cited_count AS (
            SELECT COUNT(DISTINCT cited_idx)::BIGINT AS n FROM supp_cited_pairs
        ),
        sx_citer_count AS (
            SELECT COUNT(DISTINCT citer_idx)::BIGINT AS n FROM supp_citer_pairs
        ),
        ri_supp AS (
            SELECT DISTINCT r.citer_idx AS work_idx,
                   COALESCE(ws.referenced_works_count, 1) AS ref_count
            FROM supp_citer_pairs r
            JOIN '{PARQUET}/corpus_works_supp.parquet' ws ON r.citer_idx = ws.work_idx
        ),
        type1 AS (
            SELECT
                r.citer_idx                          AS citer_work_idx,
                ci.citer_source_idx,
                ci.citer_inst_idx,
                r.cited_idx                          AS cited_work_idx,
                {SX_IDX}                             AS cited_source_idx,
                {IX_IDX}                             AS cited_inst_idx,
                ci.inst_weight,
                ci.direct_inst_weight,
                1.0                                  AS cited_inst_weight,
                1.0                                  AS direct_cited_inst_weight,
                ci.R_i,
                ci.a_citer_source,
                sx_cited_count.n                     AS a_cited_source,
                ci.a_citer_inst,
                sx_cited_count.n::DOUBLE             AS a_cited_inst
            FROM supp_cited_pairs r
            JOIN citer_info ci ON r.citer_idx = ci.citer_work_idx
            CROSS JOIN sx_cited_count
        ),
        type2 AS (
            SELECT
                r.citer_idx                          AS citer_work_idx,
                {SX_IDX}                             AS citer_source_idx,
                {IX_IDX}                             AS citer_inst_idx,
                r.cited_idx                          AS cited_work_idx,
                cj.cited_source_idx,
                cj.cited_inst_idx,
                1.0                                  AS inst_weight,
                1.0                                  AS direct_inst_weight,
                cj.cited_inst_weight,
                cj.direct_cited_inst_weight,
                ri_supp.ref_count                    AS R_i,
                sx_citer_count.n                     AS a_citer_source,
                cj.a_cited_source,
                sx_citer_count.n::DOUBLE             AS a_citer_inst,
                cj.a_cited_inst
            FROM supp_citer_pairs r
            JOIN cited_info cj ON r.cited_idx = cj.cited_work_idx
            JOIN ri_supp ON r.citer_idx = ri_supp.work_idx
            CROSS JOIN sx_citer_count
        )
        SELECT * FROM type1
        UNION ALL
        SELECT * FROM type2
    """)


def build_units(db, run_code: str, fx: str, tau_u: int, tau_s: int,
                ref_units: str = '', epsilon: int = 0) -> int:
    """
    Build the unit index table for one corpus configuration.
    """
    tname = table_name(run_code, fx, tau_u, tau_s, ref_units, epsilon)
    uname = _units_name(run_code, fx, tau_u, tau_s, ref_units, epsilon)

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


def filter_singletons(db, run_code: str, fx: str, tau_u: int, tau_s: int,
                      ref_units: str = '', epsilon: int = 0) -> tuple:
    """
    Remove units not in the giant SCC of their governing graph, then rebuild
    the units table.  Iterates until stable.

    Sentinel units (idx==1, i.e. SX_IDX/IX_IDX) are always exempted from
    removal regardless of SCC membership.

    Returns (total_sources_dropped, total_insts_dropped).
    """
    tname = table_name(run_code, fx, tau_u, tau_s, ref_units, epsilon)
    uname = _units_name(run_code, fx, tau_u, tau_s, ref_units, epsilon)
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

        # Sentinels (idx==1) are never dropped regardless of SCC membership.
        drop_src  = drop_src[drop_src != 1]
        drop_inst = drop_inst[drop_inst != 1]

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

        build_units(db, run_code, fx, tau_u, tau_s, ref_units, epsilon)

    return total_s, total_u


def ensure_catalog(db):
    db.execute("""
        CREATE TABLE IF NOT EXISTS _catalog (
            table_name     VARCHAR PRIMARY KEY,
            run_code       VARCHAR,
            F_x            VARCHAR,
            tau_u          INTEGER,
            tau_s          INTEGER,
            epsilon        INTEGER,
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
    if 'epsilon' not in cols:
        db.execute("ALTER TABLE _catalog ADD COLUMN epsilon INTEGER")
        db.execute("UPDATE _catalog SET epsilon = 0")


def update_catalog(db, run_code: str, fx: str, tau_u: int, tau_s: int, n_rows: int,
                   ref_units: str = '', epsilon: int = 0):
    tname = table_name(run_code, fx, tau_u, tau_s, ref_units, epsilon)
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
           (table_name, run_code, F_x, tau_u, tau_s, epsilon, n_rows, n_sources, n_institutions, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        [tname, run_code, fx, tau_u, tau_s, epsilon, n_rows, n_sources, n_inst,
         datetime.now().isoformat(timespec='seconds')]
    )


def clean_stale(db) -> None:
    """Drop edge-list and units tables not in the current schedule."""
    configs = corpus_configs_from_csv()
    expected = set()
    for c in configs:
        ref_units = c.get('ref_units', '')
        epsilon   = c.get('epsilon', 0)
        expected.add(table_name(c['run_code'], c['fx'], c['tau_u'], c['tau_s'], ref_units, epsilon))
        expected.add(_units_name(c['run_code'], c['fx'], c['tau_u'], c['tau_s'], ref_units, epsilon))

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

        # Group configs by (run_code, tau_u, tau_s); EBAX is first within each group
        from itertools import groupby
        key_fn = lambda c: (c['run_code'], c['tau_u'], c['tau_s'])
        for group_key, group in groupby(configs, key=key_fn):
            run_code, tau_u, tau_s = group_key
            all_units_table = f'_units_{run_code}_EBAX_tauU{tau_u}_tauS{tau_s}_vartau'

            for c in group:
                fx        = c['fx']
                ref_units = c.get('ref_units', '')
                epsilon   = c.get('epsilon', 0)
                tc0, tc1, tt0, tt1 = c['tc0'], c['tc1'], c['tt0'], c['tt1']
                tname = table_name(run_code, fx, tau_u, tau_s, ref_units, epsilon)

                if ref_units:
                    # Fixed-universe run: inherit both sources and institutions
                    # from the named reference units table (typically the baseline).
                    inherited_units = f'_units_{ref_units}_vartau'
                    inh_inst = inherited_units
                    inh_src  = inherited_units
                elif fx != 'EBAX':
                    # Field subset: inherit institutions from EBAX corpus of same window
                    inh_inst = all_units_table
                    inh_src  = None
                else:
                    inh_inst = None
                    inh_src  = None

                print(f"  Building {tname} ...", end='  ', flush=True)
                build_one(db, run_code, tc0, tc1, tt0, tt1, fx, tau_u, tau_s,
                          inherited_inst_table=inh_inst,
                          inherited_src_table=inh_src,
                          ref_units=ref_units, epsilon=epsilon)
                build_units(db, run_code, fx, tau_u, tau_s, ref_units, epsilon)
                n_s, n_u = filter_singletons(db, run_code, fx, tau_u, tau_s, ref_units, epsilon)
                uname = _units_name(run_code, fx, tau_u, tau_s, ref_units, epsilon)
                n_units_final = db.execute(f"SELECT COUNT(*) FROM {uname}").fetchone()[0]
                n_rows_final  = db.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
                update_catalog(db, run_code, fx, tau_u, tau_s, n_rows_final, ref_units, epsilon)
                print(f"{n_rows_final:,} rows  Units: {n_units_final}  "
                      f"(dropped {n_s} sources, {n_u} insts as non-giant-SCC)",
                      flush=True)

        print("\n=== Catalog ===")
        db.sql("SELECT * FROM _catalog ORDER BY run_code, F_x, tau_u").show()

        # Sample baseline edge list (derive from params rather than hardcoding)
        baseline_rows = [r for r in load_runs() if r['label'] == 'baseline']
        if baseline_rows:
            r = baseline_rows[0]
            baseline_tname = table_name(r['run_code'], r['fx'], r['tau_u'], r['tau_s'])
            print(f"\n=== Baseline edge list sample ({baseline_tname}) ===")
            db.sql(f"SELECT * FROM {baseline_tname} LIMIT 20").show()


if __name__ == '__main__':
    main()
    print('FINISHED!')
