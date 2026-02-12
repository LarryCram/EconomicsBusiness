from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import time

PARQUET_FOLDER = '/home/lc/m/openalex_feb26/parquet'
JSON_FOLDER = '/home/lc/m/openalex_feb26/json'
print(f'{Path(PARQUET_FOLDER).exists() = }')
print(f'{Path(JSON_FOLDER).exists() = }')

def convert_works_to_parquet(db):
    sql = f"""
        -- 0 CONVERT THE apenalex CLI return for econ_bus
        -- ++++++++++++++++++++++++++++++++++++++++++++
        SET preserve_insertion_order=FALSE;

        CREATE OR REPLACE TEMP TABLE works AS (
        WITH 
            loader AS
            (SELECT id AS work_id,
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
            (SELECT work_id, author_id, author_name,
                    institution.id AS institution_id,
                    institution.display_name AS institution_name,
                    institution.ror AS ror,
                    institution.country_code
            FROM 
                (SELECT work_id,
                        authorship.author.id AS author_id,
                        authorship.author.display_name AS author_name, unnest(authorship.institutions) AS institution
                    FROM (SELECT work_id, unnest(authorships) AS authorship FROM works))
            )
        SELECT * FROM authorship_reducer)
        TO '/home/lc/m/openalex_feb26/parquet/authorships.parquet' (FORMAT PARQUET); 

        SELECT * FROM works LIMIT 4;
        SHOW TABLES;
        """
    db.sql(sql)
  
    return

def main():
    with duckdb.connect() as db:
        convert_works_to_parquet(db)
        db.sql(f"SELECT count(*) FROM '{PARQUET_FOLDER}/works.parquet' GROUP BY ALL").show()
        db.sql(f"SELECT * FROM '{PARQUET_FOLDER}/authorships.parquet'").show()  
        return

if __name__ == "__main__":
    start = time.time()
    main()
    print(f'FINISHED {time.time() - start = }')
