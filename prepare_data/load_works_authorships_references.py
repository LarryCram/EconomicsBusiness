import duckdb

def load_works_authorships(db):
    sql = """
        -- 1 LOAD works and authorships for the JOURNAL LIST
        -- =================================================
        SET preserve_insertion_order=FALSE;

        COPY (
        SELECT w.*
                FROM read_parquet('/home/lc/m/working/econ_bus/parquet/source_master.parquet')
                JOIN (SELECT * FROM read_parquet('/home/lc/s/openalex_feb26/parquet/works/*.parquet')) w
                USING (source_id)
        ) TO '/home/lc/m/working/econ_bus/parquet/works.parquet' (FORMAT PARQUET);

        COPY (
        SELECT a.*
            FROM read_parquet('/home/lc/m/working/econ_bus/parquet/works.parquet')
            JOIN (SELECT * FROM read_parquet('/home/lc/s/openalex_feb26/parquet/authorships/*.parquet')) a
            USING (work_id)
        ) TO '/home/lc/m/working/econ_bus/parquet/authorships.parquet' (FORMAT PARQUET);
    """
    db.sql(sql)
    db.sql("SELECT count(DISTINCT work_id) as works_count_econ_bus FROM '/home/lc/m/working/econ_bus/parquet/works.parquet'").show()
    db.sql("SELECT count(DISTINCT source_id) as source_count_econ_bus FROM '/home/lc/m/working/econ_bus/parquet/works.parquet'").show()
    db.sql("SELECT count(DISTINCT author_id) as author_count_econ_bus FROM '/home/lc/m/working/econ_bus/parquet/authorships.parquet'").show()
    db.sql("SELECT count(DISTINCT institution_id) as institution_count_econ_bus FROM '/home/lc/m/working/econ_bus/parquet/authorships.parquet'").show()
    return

def main():
    with duckdb.connect() as db:
        load_works_authorships(db)

if __name__ == "__main__":
    main()
    print("FINISHED !")