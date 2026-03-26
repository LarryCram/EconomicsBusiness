#!/usr/bin/env python3
"""
Assemble edge list with institution weighting for spectral ranking analysis
"""

import os
import sys
import duckdb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config

paths = load_config()

def create_works_clean(parquet_dir):
    """Create works_clean.parquet with institution weighting calculations"""
    
    print("=== Assembling Works with Institution Weights ===")
    
    # Connect to DuckDB
    conn = duckdb.connect()
    
    try:
        # Set DuckDB configuration
        conn.execute("SET preserve_insertion_order=FALSE;")
        
        # Define file paths
        works_path = os.path.join(parquet_dir, 'works.parquet')
        authorships_path = os.path.join(parquet_dir, 'authorships.parquet')
        works_clean_path = os.path.join(parquet_dir, 'works_clean.parquet')
        
        print(f"Reading from: {works_path}")
        print(f"Reading from: {authorships_path}")
        print(f"Writing to: {works_clean_path}")
        
        # Execute the SQL to create work_items with weights
        works_clean_sql = f"""
        COPY (
            SELECT 
                work_id, 
                w.source_id,
                author_id,
                institution_id, 
                w.publication_year,
                
                -- Source weight: 1.0 total per work, split across rows
                1.0 / COUNT(*) OVER (PARTITION BY work_id) AS source_weight,
                
                -- Author weight: 1.0/(authors per work) total per author, split across their institution rows
                (1.0 / COUNT(DISTINCT author_id) OVER (PARTITION BY work_id)) 
                / COUNT(*) OVER (PARTITION BY work_id, author_id) AS author_weight,
                
                -- Institution weight: Same as author_weight in this formulation
                -- Because author's share is already split among institutions
                (1.0 / COUNT(DISTINCT author_id) OVER (PARTITION BY work_id)) 
                / COUNT(*) OVER (PARTITION BY work_id, author_id) AS institution_weight
                
            FROM read_parquet('{works_path}') w
            LEFT JOIN (SELECT work_id, source_id, author_id, institution_id, title, source_name, 
                              author_name, institution_name, institutions_distinct_count, publication_year
                        FROM read_parquet('{authorships_path}')) sub
            USING (work_id)
            WHERE author_id IS NOT NULL 
              AND list_contains(['article', 'review'], "type") 
              AND is_paratext=false 
              AND is_retracted=false
            ORDER BY work_id
        ) TO '{works_clean_path}' (FORMAT PARQUET);
        """
        
        print("Executing SQL query...")
        conn.execute(works_clean_sql)
        
        # Get row count for verification
        count_sql = f"SELECT COUNT(*) FROM read_parquet('{works_clean_path}')"
        row_count = conn.execute(count_sql).fetchone()[0]
        
        print(f"✓ Successfully created works_clean.parquet")
        print(f"✓ Total rows: {row_count:,}")
        
        return True
        
    except Exception as e:
        print(f"Error creating works_clean: {e}")
        return False
    finally:
        conn.close()

def create_citer_cited_table(parquet_dir):
    """Create unprojected citer-cited table with institution weights"""
    
    print("\n=== Assembling Citer-Cited Relationships ===")
    
    # Connect to DuckDB
    conn = duckdb.connect()
    
    try:
        # Set DuckDB configuration
        conn.execute("SET preserve_insertion_order=FALSE;")
        
        # Define file paths
        references_path = os.path.join(parquet_dir, 'references.parquet')
        works_clean_path = os.path.join(parquet_dir, 'works_clean.parquet')
        unprojected_path = os.path.join(parquet_dir, 'unprojected.parquet')
        
        print(f"Reading references from: {references_path}")
        print(f"Reading works_clean from: {works_clean_path}")
        print(f"Writing to: {unprojected_path}")
        
        # Execute the SQL to create unprojected table
        unprojected_sql = f"""
        COPY (
            WITH refs AS (
                SELECT citer_work, cited_work 
                FROM read_parquet('{references_path}')   
            ),
            unprojected AS (
                SELECT 
                    citer_work, 
                    i1.source_id AS citer_source, 
                    i1.author_id AS citer_author, 
                    i1.institution_id AS citer_institution, 
                    i1.publication_year AS citer_year,
                    i1.source_weight AS citer_source_weight, 
                    i1.author_weight AS citer_author_weight, 
                    i1.institution_weight AS citer_institution_weight, 
                    
                    cited_work, 
                    i2.source_id AS cited_source, 
                    i2.author_id AS cited_author, 
                    i2.institution_id AS cited_institution, 
                    i2.publication_year AS cited_year,
                    i2.source_weight AS cited_source_weight, 
                    i2.author_weight AS cited_author_weight, 
                    i2.institution_weight AS cited_institution_weight

                FROM refs 
                -- INNER JOIN is safe because we know works_clean is clean now
                INNER JOIN read_parquet('{works_clean_path}') i1 ON i1.work_id = citer_work
                INNER JOIN read_parquet('{works_clean_path}') i2 ON i2.work_id = cited_work
            )
            SELECT * FROM unprojected
        ) TO '{unprojected_path}' (FORMAT PARQUET);
        """
        
        print("Executing SQL query...")
        conn.execute(unprojected_sql)
        
        # Get row count for verification
        count_sql = f"SELECT COUNT(*) FROM read_parquet('{unprojected_path}')"
        row_count = conn.execute(count_sql).fetchone()[0]
        
        print(f"✓ Successfully created unprojected.parquet")
        print(f"✓ Total citation relationships: {row_count:,}")
        print(f"✓ Includes citer_year and cited_year for later filtering")
        
        return True
        
    except Exception as e:
        print(f"Error creating citer-cited table: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    # Set up paths from config
    parquet_dir = str(paths.parquet)
    
    print(f"Parquet directory: {parquet_dir}")
    print(f"Time windows will be applied later in pipeline")
    
    # Create works_clean parquet file
    success1 = create_works_clean(parquet_dir)
    
    # Create citer-cited relationships if works_clean succeeded
    success2 = False
    if success1:
        success2 = create_citer_cited_table(parquet_dir)
    
    if success1 and success2:
        print("\n✓ Edge list assembly completed successfully")
        print("✓ Created: works_clean.parquet and unprojected.parquet")
    else:
        print("\n✗ Edge list assembly failed")