"""
load_corpus_entities.py — Extract corpus entities from the OpenAlex snapshot.

Run after journal_filter_match_oa.py has produced source_master.parquet.
Produces three parquet files in WORKING/parquet/:

    corpus_works.parquet        -- articles and reviews in OAS sources, 2000-2024,
                                   excluding paratext and retracted works
    corpus_authorships.parquet  -- author-institution pairs for corpus works
                                   (both author_idx and institution_idx must be present)
    corpus_references.parquet   -- intra-corpus reference pairs (citer_idx, cited_idx)

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

# Corpus span in years (used to compute works_per_year)
CORPUS_YEARS = YEAR_MAX - YEAR_MIN + 1   # 25


def load_works(db):
    db.sql(f"""
        COPY (
            SELECT w.* EXCLUDE (source_id),
                   CAST(regexp_replace(w.source_id, 'https://openalex.org/S', '') AS BIGINT) AS source_idx
            FROM '{OPENALEX}/works/*.parquet' w
            JOIN '{PARQUET}/source_master.parquet' sm
                ON CAST(regexp_replace(w.source_id, 'https://openalex.org/S', '') AS BIGINT) = sm.source_idx
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
        flag_no_refs(db)

if __name__ == "__main__":
    main()
    print("FINISHED!")
