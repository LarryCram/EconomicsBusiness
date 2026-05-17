"""
load_corpus_entities.py — Extract corpus entities from the OpenAlex snapshot.

Run after journal_filter_match_oa.py has produced source_master.parquet.
Produces parquet files in WORKING/parquet/:

    corpus_works.parquet        -- articles and reviews in OAS sources, 2000-2024,
                                   excluding paratext and retracted works
    corpus_authorships.parquet  -- author-institution pairs for corpus works
                                   (both author_idx and institution_idx must be present)
    corpus_references.parquet   -- intra-corpus reference pairs (citer_idx, cited_idx)
    corpus_institutions.parquet -- one row per institution in corpus_authorships,
                                   enriched with OpenAlex metadata and works_per_year
                                   (used as the τ_U institution-retention filter)

    corpus_works_supp.parquet        -- non-corpus works in the publication window that
                                        cite corpus works (supp_role='citer') or are cited
                                        by corpus works (supp_role='cited'/'both');
                                        includes referenced_works_count for ρ=0 weighting
    corpus_authorships_supp.parquet  -- author-institution pairs for supplementary works
    corpus_references_supp.parquet   -- cross-boundary reference pairs; supp_role indicates
                                        which end is supplementary ('citer' or 'cited')

Sentinel units for ε=1 runs:
    SX_IDX = 1  (source_idx for all supplementary sources)
    IX_IDX = 1  (institution_idx for all supplementary institutions)
Sentinel rows with these IDs are added to source_master.parquet and
corpus_institutions.parquet so downstream joins succeed.

Institution retention analysis (selecting τ_U) is in institution_retention.py.
"""

from pathlib import Path
import duckdb
import yaml

# Load configuration
config_path = Path('./config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)
    WORKING  = Path(config.get('WORKING'))
    OPENALEX = Path(config.get('OPENALEX'))

PARQUET = WORKING / 'parquet'

# Census window — update to match model parameters
YEAR_MIN = 2000
YEAR_MAX = 2024

# Sentinel IDs for external (non-corpus) sources and institutions used in ε=1 runs.
# These are safe: real OpenAlex IDs are large positive integers never equal to 1.
SX_IDX = 1   # sentinel source_idx  →  'source_external'
IX_IDX = 1   # sentinel institution_idx  →  'institution_external'

# Corpus span in years (used to compute works_per_year)
CORPUS_YEARS = YEAR_MAX - YEAR_MIN + 1   # 25


def load_works(db):
    db.sql(f"""
        COPY (
            SELECT w.* EXCLUDE (source_id),
                   w.source_id AS source_idx
            FROM '{OPENALEX}/works/*.parquet' w
            JOIN '{PARQUET}/source_master.parquet' sm
                ON w.source_id = sm.source_idx
            WHERE w.publication_year BETWEEN {YEAR_MIN} AND {YEAR_MAX}
              AND list_contains(['article', 'review'], w."type")
              AND w.is_paratext = false
              AND w.is_retracted = false
        ) TO '{PARQUET}/corpus_works.parquet' (FORMAT PARQUET)
    """)
    db.sql(f"SELECT * FROM '{PARQUET}/corpus_works.parquet'").show()
    print("WORKS EXTRACT COMPLETE!")


def load_authorships(db):
    db.sql(f"""
        COPY (
            SELECT a.*
            FROM '{PARQUET}/corpus_works.parquet' w
            JOIN '{OPENALEX}/authorships/*.parquet' a USING (work_idx)
            WHERE author_idx IS NOT NULL AND institution_idx IS NOT NULL
        ) TO '{PARQUET}/corpus_authorships.parquet' (FORMAT PARQUET)
    """)
    db.sql(f"SELECT * FROM '{PARQUET}/corpus_authorships.parquet'").show()
    print("AUTHORSHIPS EXTRACT COMPLETE!")


def load_references(db):
    db.sql(f"""
        COPY (
            SELECT r.citer_idx, r.cited_idx
            FROM '{OPENALEX}/references/*.parquet' r
            JOIN '{PARQUET}/corpus_works.parquet' cw1 ON r.citer_idx = cw1.work_idx
            JOIN '{PARQUET}/corpus_works.parquet' cw2 ON r.cited_idx  = cw2.work_idx
        ) TO '{PARQUET}/corpus_references.parquet' (FORMAT PARQUET)
    """)
    db.sql(f"SELECT * FROM '{PARQUET}/corpus_references.parquet'").show()
    print("REFERENCES EXTRACT COMPLETE!")


INSTITUTION_TYPES = ('education', 'nonprofit', 'government', 'other')

# Confirmed OpenAlex authorship disambiguation errors: small institutions
# absorbing papers from famous namesakes.  Excluded from corpus_institutions
# so they propagate out of all downstream edge lists and rankings.
#   175594653  John Brown University (AR)      ← Brown University papers
#   87182695   Universidad del Noreste (MX)    ← Northeastern University papers
#   68812265   Anderson University – SC        ← UCLA Anderson School papers
INSTITUTION_EXCLUSIONS = (175594653, 87182695, 68812265)


def load_institutions(db):
    """
    Build corpus_institutions.parquet.

    Aggregates corpus_authorships by institution, enriches with OpenAlex
    institution metadata, and filters to type IN ('education', 'nonprofit',
    'government', 'other').  This excludes companies, healthcare facilities,
    archives, and other non-research entities that enter via OpenAlex
    authorship errors or are otherwise out of scope.

    INSTITUTION_EXCLUSIONS removes confirmed OA disambiguation errors where
    a small institution absorbs papers from a famous namesake.

    institution_idx is derived from the OpenAlex institution id by stripping
    the 'https://openalex.org/I' prefix.
    """
    type_list      = ', '.join(f"'{t}'"  for t in INSTITUTION_TYPES)
    exclusion_list = ', '.join(str(i)    for i in INSTITUTION_EXCLUSIONS)
    db.sql(f"""
        COPY (
            WITH inst_works AS (
                SELECT institution_idx,
                       COUNT(DISTINCT work_idx) AS works_count
                FROM '{PARQUET}/corpus_authorships.parquet'
                WHERE institution_idx IS NOT NULL
                GROUP BY institution_idx
            ),
            inst_meta AS (
                SELECT CAST(REGEXP_REPLACE(id, 'https://openalex.org/I', '') AS BIGINT)
                           AS institution_idx,
                       display_name AS institution_name,
                       country_code,
                       type,
                       ror
                FROM '{OPENALEX}/institutions.parquet'
                WHERE type IN ({type_list})
                  AND CAST(REGEXP_REPLACE(id, 'https://openalex.org/I', '') AS BIGINT)
                      NOT IN ({exclusion_list})
            )
            SELECT iw.institution_idx,
                   im.institution_name,
                   im.country_code,
                   im.type,
                   im.ror,
                   iw.works_count,
                   ROUND(iw.works_count / {CORPUS_YEARS}.0, 4) AS works_per_year
            FROM inst_works iw
            INNER JOIN inst_meta im USING (institution_idx)
            ORDER BY iw.works_count DESC
        ) TO '{PARQUET}/corpus_institutions.parquet' (FORMAT PARQUET)
    """)
    n = db.sql(f"SELECT COUNT(*) FROM '{PARQUET}/corpus_institutions.parquet'").fetchone()[0]
    print(f"corpus_institutions.parquet: {n:,} total institutions")
    print("INSTITUTIONS EXTRACT COMPLETE!")


def flag_no_refs(db):
    """
    Add has_corpus_refs boolean to source_master.parquet.

    True if the source has at least one work appearing as citer_idx in
    corpus_references.parquet.  Sources with no outgoing intra-corpus
    references are flagged False and excluded from edge-list construction
    and published tables.
    """
    db.sql(f"""
        CREATE OR REPLACE TEMP TABLE _sm_flagged AS
        SELECT sm.*,
               sm.source_idx IN (
                   SELECT DISTINCT cw.source_idx
                   FROM '{PARQUET}/corpus_references.parquet' cr
                   JOIN '{PARQUET}/corpus_works.parquet' cw ON cr.citer_idx = cw.work_idx
               ) AS has_corpus_refs
        FROM '{PARQUET}/source_master.parquet' sm
    """)
    n_flagged = db.sql(
        "SELECT COUNT(*) FROM _sm_flagged WHERE has_corpus_refs = false"
    ).fetchone()[0]
    db.sql(f"""
        COPY (SELECT * FROM _sm_flagged)
        TO '{PARQUET}/source_master.parquet' (FORMAT PARQUET)
    """)
    print(f"FLAG NO-REFS COMPLETE — {n_flagged} sources flagged has_corpus_refs=false")



def load_works_supp(db):
    """
    Build corpus_works_supp.parquet — non-corpus works in the publication window
    that are cited by corpus works ('cited') or that cite corpus works ('citer'),
    or both ('both').  Includes referenced_works_count for ρ=0 weighting.

    Real source_idx is kept; replacement by SX_IDX happens in build_edge_lists.py.
    """
    db.sql(f"""
        COPY (
            WITH
            corpus_ids AS (
                SELECT work_idx FROM '{PARQUET}/corpus_works.parquet'
            ),
            cited_supp AS (
                SELECT DISTINCT r.cited_idx AS work_idx
                FROM '{OPENALEX}/references/*.parquet' r
                WHERE r.citer_idx IN (SELECT work_idx FROM corpus_ids)
                  AND r.cited_idx NOT IN (SELECT work_idx FROM corpus_ids)
            ),
            citer_supp AS (
                SELECT DISTINCT r.citer_idx AS work_idx
                FROM '{OPENALEX}/references/*.parquet' r
                WHERE r.cited_idx IN (SELECT work_idx FROM corpus_ids)
                  AND r.citer_idx NOT IN (SELECT work_idx FROM corpus_ids)
            ),
            combined AS (
                SELECT work_idx,
                       TRUE  AS is_cited,
                       FALSE AS is_citer
                FROM cited_supp
                UNION ALL
                SELECT work_idx,
                       FALSE AS is_cited,
                       TRUE  AS is_citer
                FROM citer_supp
            ),
            aggregated AS (
                SELECT work_idx,
                       MAX(CAST(is_cited AS INTEGER)) = 1 AS is_cited,
                       MAX(CAST(is_citer AS INTEGER)) = 1 AS is_citer
                FROM combined
                GROUP BY work_idx
            )
            SELECT w.work_idx,
                   w.source_id                AS source_idx,
                   w.publication_year,
                   w.referenced_works_count,
                   CASE
                       WHEN a.is_cited AND a.is_citer THEN 'both'
                       WHEN a.is_cited               THEN 'cited'
                       ELSE                               'citer'
                   END                        AS supp_role
            FROM aggregated a
            JOIN '{OPENALEX}/works/*.parquet' w ON a.work_idx = w.work_idx
            WHERE w.publication_year BETWEEN {YEAR_MIN} AND {YEAR_MAX}
              AND list_contains(['article', 'review'], w."type")
              AND w.is_paratext  = false
              AND w.is_retracted = false
        ) TO '{PARQUET}/corpus_works_supp.parquet' (FORMAT PARQUET)
    """)
    n = db.sql(f"SELECT COUNT(*) FROM '{PARQUET}/corpus_works_supp.parquet'").fetchone()[0]
    n_cited = db.sql(f"""
        SELECT COUNT(*) FROM '{PARQUET}/corpus_works_supp.parquet'
        WHERE supp_role IN ('cited','both')
    """).fetchone()[0]
    n_citer = db.sql(f"""
        SELECT COUNT(*) FROM '{PARQUET}/corpus_works_supp.parquet'
        WHERE supp_role IN ('citer','both')
    """).fetchone()[0]
    print(f"corpus_works_supp.parquet: {n:,} works  "
          f"(cited={n_cited:,}  citer={n_citer:,})")
    print("WORKS SUPP EXTRACT COMPLETE!")


def load_authorships_supp(db):
    """
    Build corpus_authorships_supp.parquet — author-institution pairs for
    supplementary works (real institution_idx values; replacement by IX_IDX
    happens in build_edge_lists.py).
    """
    db.sql(f"""
        COPY (
            SELECT a.work_idx,
                   a.author_idx,
                   a.institution_idx,
                   ws.supp_role
            FROM '{OPENALEX}/authorships/*.parquet' a
            JOIN '{PARQUET}/corpus_works_supp.parquet' ws ON a.work_idx = ws.work_idx
            WHERE a.author_idx      IS NOT NULL
              AND a.institution_idx IS NOT NULL
        ) TO '{PARQUET}/corpus_authorships_supp.parquet' (FORMAT PARQUET)
    """)
    n = db.sql(
        f"SELECT COUNT(*) FROM '{PARQUET}/corpus_authorships_supp.parquet'"
    ).fetchone()[0]
    print(f"corpus_authorships_supp.parquet: {n:,} rows")
    print("AUTHORSHIPS SUPP EXTRACT COMPLETE!")


def load_references_supp(db):
    """
    Build corpus_references_supp.parquet — cross-boundary reference pairs.

    supp_role='cited': citer is a corpus work, cited is a supplementary work.
    supp_role='citer': citer is a supplementary work, cited is a corpus work.
    """
    db.sql(f"""
        COPY (
            WITH
            corpus_ids     AS (SELECT work_idx FROM '{PARQUET}/corpus_works.parquet'),
            cited_supp_ids AS (
                SELECT work_idx FROM '{PARQUET}/corpus_works_supp.parquet'
                WHERE supp_role IN ('cited', 'both')
            ),
            citer_supp_ids AS (
                SELECT work_idx FROM '{PARQUET}/corpus_works_supp.parquet'
                WHERE supp_role IN ('citer', 'both')
            )
            SELECT r.citer_idx,
                   r.cited_idx,
                   CASE
                       WHEN r.citer_idx IN (SELECT work_idx FROM corpus_ids)
                        AND r.cited_idx  IN (SELECT work_idx FROM cited_supp_ids)
                       THEN 'cited'
                       ELSE 'citer'
                   END AS supp_role
            FROM '{OPENALEX}/references/*.parquet' r
            WHERE (
                r.citer_idx IN (SELECT work_idx FROM corpus_ids)
                AND r.cited_idx IN (SELECT work_idx FROM cited_supp_ids)
            ) OR (
                r.citer_idx IN (SELECT work_idx FROM citer_supp_ids)
                AND r.cited_idx IN (SELECT work_idx FROM corpus_ids)
            )
        ) TO '{PARQUET}/corpus_references_supp.parquet' (FORMAT PARQUET)
    """)
    n = db.sql(
        f"SELECT COUNT(*) FROM '{PARQUET}/corpus_references_supp.parquet'"
    ).fetchone()[0]
    print(f"corpus_references_supp.parquet: {n:,} rows")
    print("REFERENCES SUPP EXTRACT COMPLETE!")


def add_sentinel_rows(db):
    """
    Insert sentinel rows (SX_IDX, IX_IDX) into source_master.parquet and
    corpus_institutions.parquet so that ε=1 edge lists can join against them.

    All domain columns are NULL; the idx and name are the only meaningful fields.
    Idempotent — removes any pre-existing sentinel row before inserting.
    """
    import pandas as pd

    # ── source_master sentinel ────────────────────────────────────────────────
    sm_cols = db.execute(
        f"DESCRIBE SELECT * FROM '{PARQUET}/source_master.parquet'"
    ).fetchdf()
    sm_sentinel = {col: None for col in sm_cols['column_name']}
    sm_sentinel['source_idx']     = SX_IDX
    sm_sentinel['source_name']    = 'source_external'
    sm_sentinel['has_corpus_refs'] = True
    sm_df = pd.DataFrame([sm_sentinel])
    for col, dtype in zip(sm_cols['column_name'], sm_cols['column_type']):
        if 'INT' in dtype.upper() or 'BIGINT' in dtype.upper():
            sm_df[col] = sm_df[col].astype('Int64')
        elif 'BOOL' in dtype.upper():
            sm_df[col] = sm_df[col].astype(object)
    db.register('_sm_sentinel', sm_df)
    db.sql(f"""
        COPY (
            SELECT * FROM '{PARQUET}/source_master.parquet'
            WHERE source_idx != {SX_IDX}
            UNION ALL
            SELECT * FROM _sm_sentinel
        ) TO '{PARQUET}/source_master.parquet' (FORMAT PARQUET)
    """)
    db.unregister('_sm_sentinel')
    print(f"source_master.parquet: sentinel source_idx={SX_IDX} added")

    # ── corpus_institutions sentinel ──────────────────────────────────────────
    ci_cols = db.execute(
        f"DESCRIBE SELECT * FROM '{PARQUET}/corpus_institutions.parquet'"
    ).fetchdf()
    ci_sentinel = {col: None for col in ci_cols['column_name']}
    ci_sentinel['institution_idx']  = IX_IDX
    ci_sentinel['institution_name'] = 'institution_external'
    ci_df = pd.DataFrame([ci_sentinel])
    for col, dtype in zip(ci_cols['column_name'], ci_cols['column_type']):
        if 'INT' in dtype.upper() or 'BIGINT' in dtype.upper():
            ci_df[col] = ci_df[col].astype('Int64')
    db.register('_ci_sentinel', ci_df)
    db.sql(f"""
        COPY (
            SELECT * FROM '{PARQUET}/corpus_institutions.parquet'
            WHERE institution_idx != {IX_IDX}
            UNION ALL
            SELECT * FROM _ci_sentinel
        ) TO '{PARQUET}/corpus_institutions.parquet' (FORMAT PARQUET)
    """)
    db.unregister('_ci_sentinel')
    print(f"corpus_institutions.parquet: sentinel institution_idx={IX_IDX} added")
    print("SENTINEL ROWS COMPLETE!")


def main():
    with duckdb.connect() as db:
        db.sql(f"""
            SET temp_directory = '{WORKING}/.tmp';
            SET memory_limit = '56GB';
            SET preserve_insertion_order = false;
        """)
        load_works(db)
        load_authorships(db)
        load_references(db)
        load_institutions(db)
        flag_no_refs(db)
        load_works_supp(db)
        load_authorships_supp(db)
        load_references_supp(db)
        add_sentinel_rows(db)

if __name__ == "__main__":
    main()
    print("FINISHED!")
