from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import time

PARQUET_FOLDER = '/home/lc/m/openalex_feb26/parquet'
JSON_FOLDER = '/home/lc/m/openalex_feb26/data'
print(f'{Path(PARQUET_FOLDER).exists() = }')
print(f'{Path(JSON_FOLDER).exists() = }')

def convert_works_to_parquet(db):
    sql = """
        SET preserve_insertion_order=FALSE;

        CREATE OR REPLACE TABLE works AS (
        WITH 
            loader AS
            (SELECT id AS work_id,
                id[23:]::BIGINT AS work_idx,
                doi,
                title,
                institutions_distinct_count,
                publication_year, 
                referenced_works_count, 
                cited_by_count, 
                type,          
                is_retracted,  
                is_paratext,
                biblio.*, 
                primary_location.source.id AS source_id,
                primary_location.source.id[23:]::BIGINT AS source_idx,        
                primary_location.source.display_name AS source_name, 
                primary_location.source.host_organization AS source_host,
                referenced_works,
                authorships,    
            FROM '/home/lc/m/openalex_feb26/json/**/*.json'   
            -- LIMIT 16
            )
        SELECT * FROM loader);

        COPY works TO '/home/lc/m/openalex_feb26/parquet/works.parquet' (FORMAT PARQUET);

        COPY (
        WITH 
            authorship_reducer AS
            (SELECT work_id, work_idx, author_id, author_idx, author_name,
                    institution.id AS institution_id,
                    institution.id[23:]::BIGINT AS institution_idx,
                    institution.display_name AS institution_name,
                    institution.ror AS ror,
                    institution.country_code
            FROM 
                (SELECT work_id, 
                        authorship.author.id AS author_id, authorship.author.id[23:]::BIGINT AS author_idx, 
                        authorship.author.display_name AS author_name, unnest(authorship.institutions) AS institution
                    FROM (SELECT work_id, unnest(authorships) AS authorship FROM works))
            )
        SELECT * FROM authorship_reducer)
        TO '/home/lc/m/openalex_feb26/parquet/authorships.parquet' (FORMAT PARQUET); 
        """
    db.sql(sql)
    return

def convert_to_parquet(db):
    
    for kind in ['topics', 'authors', 'sources', 'institutions']:
        print(f'{kind = }')
        sql = f"SELECT * FROM {kind}.{kind}"
        db.sql(sql).show()
        sql = f"COPY (SELECT * FROM {kind}.{kind}) TO '/home/lc/m/openalex_june25/parquet/{kind}.parquet' (FORMAT parquet)"
        db.sql(sql)
    
    return

def main():
    with duckdb.connect() as db:
            
        db.sql("ATTACH IF NOT EXISTS '/home/lc/m/openalex_june25/authors.duckdb' AS authors")
        db.sql("ATTACH IF NOT EXISTS '/home/lc/m/openalex_june25/sources.duckdb' AS sources")
        db.sql("ATTACH IF NOT EXISTS '/home/lc/m/openalex_june25/institutions.duckdb' AS institutions")
        db.sql("ATTACH IF NOT EXISTS '/home/lc/m/openalex_june25/works.duckdb' AS works")
        db.sql("ATTACH IF NOT EXISTS '/home/lc/m/openalex_june25/topics.duckdb' AS topics")
        db.sql("SHOW ALL TABLES").show()

        convert_to_parquet(db)
        convert_works_to_parquet(db)
        return

if __name__ == "__main__":
    start = time.time()
    # main()
    print(f'FINISHED {time.time() - start = }')
