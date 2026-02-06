from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import time

PARQUET_FOLDER = '/home/lc/m/openalex_feb26/parquet'
JSON_FOLDER = '/home/lc/m/openalex_feb26/data'
print(f'{Path(PARQUET_FOLDER).exists() = }')
print(f'{Path(JSON_FOLDER).exists() = }')

def convert_to_parquet(db):
    
    for kind in ['topics', 'authors', 'sources', 'institutions', 'works']:
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
        return

if __name__ == "__main__":
    start = time.time()
    # main()
    print(f'FINISHED {time.time() - start = }')
