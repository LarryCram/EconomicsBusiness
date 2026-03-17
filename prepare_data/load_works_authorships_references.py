import duckdb

def load_works_authorships(db):
    sql = """
        SET temp_directory = '/home/lc/s/.tmp';
        SET memory_limit = '56GB'; 
        SET preserve_insertion_order = false; 
        -- FETCH CORPUS OF WORKS FOR SOURCES
        -- =================================
        COPY (
        SELECT c.*, w.* EXCLUDE (w.source_name, w.source_id) 
            FROM read_xlsx('/home/lc/Projects/EconomicsBusiness/data/corpus_sources_master.xlsx', sheet='10_LongList_OAS') c
            LEFT JOIN '/home/lc/m/openalex_feb26/parquet/works/*.parquet' w
            USING (source_id)
            WHERE (publication_year BETWEEN 2000 AND 2025) AND list_contains(['article', 'review'], "type") AND out_of_scope='In scope' AND is_paratext=false AND is_retracted=false
        ) TO '/home/lc/m/working/econ_bus/parquet/corpus_works.parquet' (FORMAT PARQUET)
        """
    db.sql(sql)
    db.sql("SELECT * FROM '/home/lc/m/working/econ_bus/parquet/corpus_works.parquet'").show()
    print("WORKS EXTRACT COMPLETE!")
    
    sql = """
        -- FETCH AUTHORSHIPS FOR SOURCES
        -- =============================
        COPY (
        SELECT *
            FROM '/home/lc/m/working/econ_bus/parquet/corpus_works.parquet'
            LEFT JOIN '/home/lc/m/openalex_feb26/parquet/authorships/*.parquet'
            USING (work_idx)
            WHERE author_idx NOT NULL AND institution_idx NOT NULL
        ) TO '/home/lc/m/working/econ_bus/parquet/corpus_authorships.parquet' (FORMAT PARQUET)
        """
    db.sql(sql)
    db.sql("SELECT * FROM '/home/lc/m/working/econ_bus/parquet/corpus_authorships.parquet'").show()
    print("FINISHED AUTHORSHIPS EXTRACT!")
    
    sql = """
        -- FETCH REFERENCES DROPPING REFERENCED WORKS THAT ARE NOT IN CORPUS
        -- =================================================================        
        COPY (
        SELECT 
            r.citer_idx, 
            r.cited_idx
        FROM '/home/lc/m/openalex_feb26/parquet/references/*.parquet' r
        -- JOIN 1: Filters so 'citer_id' MUST be in your corpus
        JOIN '/home/lc/m/working/econ_bus/parquet/corpus_works.parquet' cw1 
            ON r.citer_idx = cw1.work_idx
        -- JOIN 2: Filters so 'cited_id' MUST also be in your corpus
        JOIN '/home/lc/m/working/econ_bus/parquet/corpus_works.parquet' cw2 
            ON r.cited_idx = cw2.work_idx
        ) TO '/home/lc/m/working/econ_bus/parquet/corpus_references.parquet' (FORMAT PARQUET);
        """
    db.sql(sql)
    db.sql("SELECT * FROM '/home/lc/m/working/econ_bus/parquet/corpus_references.parquet'").show()
    print("FINISHED REFERENCES EXTRACT!")
    # db.sql("SELECT count(DISTINCT work_id) as works_count_econ_bus FROM '/home/lc/m/working/econ_bus/parquet/works.parquet'").show()
    # db.sql("SELECT count(DISTINCT source_id) as source_count_econ_bus FROM '/home/lc/m/working/econ_bus/parquet/works.parquet'").show()
    # db.sql("SELECT count(DISTINCT author_id) as author_count_econ_bus FROM '/home/lc/m/working/econ_bus/parquet/authorships.parquet'").show()
    # db.sql("SELECT count(DISTINCT institution_id) as institution_count_econ_bus FROM '/home/lc/m/working/econ_bus/parquet/authorships.parquet'").show()
    return

def main():
    with duckdb.connect() as db:
        load_works_authorships(db)

if __name__ == "__main__":
    main()
    print("FINISHED !")