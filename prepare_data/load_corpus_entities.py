"""
load_corpus_entities.py — Extract corpus entities from the OpenAlex snapshot.

Run after journal_filter_match_oa.py has produced source_master.parquet.
Produces four parquet files in WORKING/parquet/:

    corpus_works.parquet        -- articles and reviews in OAS sources, 2000-2024,
                                   excluding paratext and retracted works
    corpus_authorships.parquet  -- author-institution pairs for corpus works
                                   (both author_idx and institution_idx must be present)
    corpus_references.parquet   -- intra-corpus reference pairs (citer_idx, cited_idx)
    corpus_institutions.parquet -- one row per institution appearing in corpus_authorships,
                                   enriched with OpenAlex metadata and works_per_year
                                   (used as the τ_U institution-retention filter)
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
            SELECT w.*
            FROM '{PARQUET}/source_master.parquet' s
            JOIN '{OPENALEX}/works/*.parquet' w USING (source_id)
            WHERE publication_year BETWEEN {YEAR_MIN} AND {YEAR_MAX}
              AND list_contains(['article', 'review'], "type")
              AND is_paratext = false
              AND is_retracted = false
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


def load_institutions(db):
    """
    Build corpus_institutions.parquet.

    Aggregates corpus_authorships by institution, computes works_count and
    works_per_year = works_count / CORPUS_YEARS, and enriches with OpenAlex
    institution metadata.  institution_idx is derived from the OpenAlex
    institution id by stripping the 'https://openalex.org/I' prefix.
    """
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
                       type
                FROM '{OPENALEX}/institutions.parquet'
            )
            SELECT iw.institution_idx,
                   im.institution_name,
                   im.country_code,
                   im.type,
                   iw.works_count,
                   ROUND(iw.works_count / {CORPUS_YEARS}.0, 4) AS works_per_year
            FROM inst_works iw
            LEFT JOIN inst_meta im USING (institution_idx)
            ORDER BY iw.works_count DESC
        ) TO '{PARQUET}/corpus_institutions.parquet' (FORMAT PARQUET)
    """)
    n = db.sql(f"SELECT COUNT(*) FROM '{PARQUET}/corpus_institutions.parquet'").fetchone()[0]
    print(f"corpus_institutions.parquet: {n:,} total institutions")
    print("INSTITUTIONS EXTRACT COMPLETE!")


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

if __name__ == "__main__":
    main()
    print("FINISHED!")
