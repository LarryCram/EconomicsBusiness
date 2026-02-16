from pathlib import Path
import duckdb

def process_works(db):
    sql = """
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
            FROM read_json_auto('/home/lc/m/openalex_feb26/json/**/*.json', ignore_errors = true)  
            -- LIMIT 16
            )
        SELECT * FROM loader);

        -- Works parquet from filtered works 
        COPY (SELECT * EXCLUDE (referenced_works, authorships) FROM works) TO '/home/lc/m/openalex_feb26/parquet/works.parquet' (FORMAT PARQUET);

        -- References parquet from unnested referenced_works
        COPY (
                SELECT w.work_id AS citer_work, r.cited_work
                FROM works w
                LEFT JOIN LATERAL unnest(w.referenced_works) AS r(cited_work) ON TRUE
            ) TO '/home/lc/m/openalex_feb26/parquet/references.parquet' (FORMAT PARQUET);

        -- Authorships parquet from unnsted authors and unnested institutions
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
    """
    db.sql(sql)
    return

def main():
    with duckdb.connect() as db:
        process_works(db)
    return

if __name__ == "__main__":

    main()
    print("FINISHED!")