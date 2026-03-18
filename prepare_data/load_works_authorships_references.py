"""
load_works_authorships_references.py

Extracts corpus works, authorships, and intra-corpus references from the
OpenAlex parquet snapshot for all sources in source_master.parquet (OAS).

    Corpus works:  articles and reviews in OAS sources, published within the
                   census window [YEAR_MIN, YEAR_MAX], excluding paratext and
                   retracted works.
    Authorships:   author-institution pairs for corpus works (both author and
                   institution must be present).
    References:    intra-corpus reference pairs (citer_idx, cited_idx) where
                   both the citing and cited work are in the corpus.

Run after journal_filter_match_oa.py has produced source_master.parquet.
"""

from pathlib import Path
import duckdb
import yaml

# Load configuration
config_path = Path('./config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)
    WORKING = Path(config.get('WORKING'))
    OPENALEX = Path(config.get('OPENALEX'))

PARQUET = WORKING / 'parquet'

# Census window — update to match model parameters
YEAR_MIN = 2000
YEAR_MAX = 2025


def load_works_authorships_references(db):
    db.sql(f"""
        SET temp_directory = '{WORKING}/.tmp';
        SET memory_limit = '56GB';
        SET preserve_insertion_order = false;
    """)

    # --- Works ---
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

    # --- Authorships ---
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

    # --- Intra-corpus references ---
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


def main():
    with duckdb.connect() as db:
        load_works_authorships_references(db)

if __name__ == "__main__":
    main()
    print("FINISHED!")
