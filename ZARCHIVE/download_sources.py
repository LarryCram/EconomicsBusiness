#!/usr/bin/env python3
"""
Parse OpenAlex JSON files to parquet format using DuckDB SQL
"""

import os
import duckdb
import yaml
from pathlib import Path

# Load configuration relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'data.yaml')

with open(config_path) as f:
    config = yaml.safe_load(f)

def parse_json_to_parquets(json_dir, parquet_dir):
    """Parse OpenAlex JSON using DuckDB SQL to create works, references, and authorships parquet files"""
    json_files = list(Path(json_dir).glob("**/*.json"))
    if not json_files:
        print("No JSON files to convert")
        return False
    
    print(f"Parsing {len(json_files)} JSON files with DuckDB...")
    
    # Ensure output directory exists
    os.makedirs(parquet_dir, exist_ok=True)
    
    # Connect to DuckDB
    conn = duckdb.connect()
    
    try:
        # Set DuckDB configuration
        conn.execute("SET preserve_insertion_order=FALSE;")
        
        print("Creating works table from JSON...")
        works_sql = f"""
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
            FROM read_json_auto('{json_dir}/**/*.json', ignore_errors = true)  
            )
        SELECT * FROM loader);
        """
        conn.execute(works_sql)
        
        # Export works parquet (excluding nested fields)
        works_parquet_path = os.path.join(parquet_dir, 'works.parquet')
        print(f"Exporting works to {works_parquet_path}")
        conn.execute(f"""
            COPY (SELECT * EXCLUDE (referenced_works, authorships) FROM works) 
            TO '{works_parquet_path}' (FORMAT PARQUET);
        """)
        
        # Export references parquet
        references_parquet_path = os.path.join(parquet_dir, 'references.parquet')
        print(f"Exporting references to {references_parquet_path}")
        conn.execute(f"""
            COPY (
                SELECT w.work_id AS citer_work, r.cited_work
                FROM works w
                LEFT JOIN LATERAL unnest(w.referenced_works) AS r(cited_work) ON TRUE
            ) TO '{references_parquet_path}' (FORMAT PARQUET);
        """)
        
        # Export authorships parquet
        authorships_parquet_path = os.path.join(parquet_dir, 'authorships.parquet')
        print(f"Exporting authorships to {authorships_parquet_path}")
        conn.execute(f"""
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
                            authorship.author.display_name AS author_name, 
                            unnest(authorship.institutions) AS institution
                        FROM (SELECT work_id, unnest(authorships) AS authorship FROM works))
                )
            SELECT * FROM authorship_reducer)
            TO '{authorships_parquet_path}' (FORMAT PARQUET);
        """)
        
        # Get summary statistics
        works_count = conn.execute("SELECT COUNT(*) FROM works").fetchone()[0]
        refs_count = conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT w.work_id AS citer_work, r.cited_work
                FROM works w
                LEFT JOIN LATERAL unnest(w.referenced_works) AS r(cited_work) ON TRUE
            )
        """).fetchone()[0]
        auths_count = conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT work_id,
                        authorship.author.id AS author_id,
                        authorship.author.display_name AS author_name, 
                        unnest(authorship.institutions) AS institution
                    FROM (SELECT work_id, unnest(authorships) AS authorship FROM works)
            )
        """).fetchone()[0]
        
        print(f"✓ Successfully parsed {works_count} works")
        print(f"✓ Successfully parsed {refs_count} references")
        print(f"✓ Successfully parsed {auths_count} authorships")
        
        return True
        
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return False
    finally:
        conn.close()

def verify_parquet_files(parquet_dir):
    """Examine the created parquet files to verify structure"""
    conn = duckdb.connect()
    
    try:
        parquet_files = ['works.parquet', 'references.parquet', 'authorships.parquet']
        
        for filename in parquet_files:
            filepath = os.path.join(parquet_dir, filename)
            if os.path.exists(filepath):
                print(f"\\n--- {filename} ---")
                
                # Get row count and sample data
                count = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{filepath}')").fetchone()[0]
                print(f"Rows: {count:,}")
                
                # Get column info
                schema = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{filepath}')").fetchall()
                print(f"Columns ({len(schema)}): {', '.join([col[0] for col in schema[:10]])}{'...' if len(schema) > 10 else ''}")
                
                # Show sample
                if count > 0:
                    sample = conn.execute(f"SELECT * FROM read_parquet('{filepath}') LIMIT 2").fetchall()
                    print(f"Sample: {len(sample)} rows shown")
            else:
                print(f"\\n--- {filename} --- NOT FOUND")
    
    except Exception as e:
        print(f"Error verifying parquet files: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    # Set up paths from config
    json_dir = config['JSON_DIR']
    parquet_dir = config['PARQUET_DIR']
    
    print("=== Parsing JSON to Parquet Files ===")
    print(f"JSON directory: {json_dir}")
    print(f"Output directory: {parquet_dir}")
    
    # Ensure output directory exists
    os.makedirs(parquet_dir, exist_ok=True)
    
    # Parse JSON to parquet files
    success = parse_json_to_parquets(json_dir, parquet_dir)
    
    if success:
        print("\n✓ JSON parsing completed successfully")
        print(f"Parquet files created in: {parquet_dir}")
    else:
        print("\n✗ JSON parsing failed")