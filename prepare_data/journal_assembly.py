from pathlib import Path
import duckdb
import yaml

# Load configuration relative to this script
script_dir = Path(__file__)
config_path = Path('./config.yaml')

with open(config_path) as f:
    config = yaml.safe_load(f)
    print(f'{config = }')
    PROJECT_FOLDER = config['PROJECT_ROOT']
    DATA = PROJECT_FOLDER / Path(config.get('DATA'))

def load_journals():
    with duckdb.connect() as db:
        # Create temp table for ERA data
        db.sql(f"""
            CREATE TEMP TABLE era AS
            SELECT DISTINCT list_distinct(["ISSN 1", "ISSN 2", "ISSN 3"]) AS era_ISSN,
                    Title AS era_name, 
                    "FoR 1 Name" as FoR_name,  
            FROM read_xlsx('{DATA}/source_masters/ecobus_journal_harzing_era.xlsx', 
                            sheet='ERA2023 Submission Journal List', 
                            range='A:P', 
                            header=true, 
                            all_varchar = true)
            WHERE "FoR 1"[1:2] = '35' OR "FoR 1"[1:2] = '38'
            GROUP BY ALL
        """)
        print("=== ERA BASE LIST ===")
        db.sql("SELECT COUNT(*) as era_count FROM era").show()
        db.sql("SELECT * FROM era LIMIT 10").show()
        db.sql("SELECT era_name, COUNT(DISTINCT era_name) AS count_name FROM ERA GROUP BY ALL ORDER BY count_name DESC ").show()
        
        # Harzing table with consistent structure
        db.sql(f"""
            CREATE TEMP TABLE harzing AS
            SELECT DISTINCT
                [ISSN] as harzing_issn,
                Journal as journal_name,
                Subject_areas as field,
            FROM read_xlsx('{DATA}/source_masters/ecobus_journal_harzing_era.xlsx', sheet='Harzing', range='A:C', header=true) h
            WHERE ISSN IS NOT NULL
        """)
        
        print("=== HARZING CLEAN TABLE ===")
        db.sql("SELECT COUNT(*) as harzing_count FROM harzing").show()
        db.sql("SELECT * FROM harzing LIMIT 5").show()

        
        # Simple join of ERA and Harzing on ISSN overlaps
        db.sql("""
            CREATE TEMP TABLE comprehensive_journals AS
            SELECT DISTINCT
                list_distinct(COALESCE(e.era_ISSN, []) || COALESCE(h.harzing_issn, [])) as unique_issn_list,
                e.era_name as era_journal_name,
                e.FoR_name as era_field,
                h.journal_name as harzing_journal_name,
                h.field as harzing_field
            FROM era e
            FULL OUTER JOIN harzing h 
                ON len(list_intersect(e.era_ISSN, h.harzing_issn)) > 0
        """)
        
        # Show summary statistics  
        print("=== COMBINED ERA AND HARZING RESULTS ===")
        db.sql("SELECT COUNT(*) as count FROM comprehensive_journals").show()
        
        print("\n=== BREAKDOWN BY SOURCE PRESENCE ===")
        db.sql("""
            SELECT 
                CASE 
                    WHEN era_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL THEN 'Both sources'
                    WHEN era_journal_name IS NOT NULL THEN 'ERA only'  
                    WHEN harzing_journal_name IS NOT NULL THEN 'Harzing only'
                    ELSE 'Neither (error)'
                END as source_type,
                COUNT(*) as count
            FROM comprehensive_journals
            GROUP BY source_type
        """).show()
        
        print("\n=== EXAMPLES OF EACH TYPE ===")
        print("ERA only:")
        db.sql("""
            SELECT unique_issn_list, era_journal_name, era_field
            FROM comprehensive_journals 
            WHERE era_journal_name IS NOT NULL AND harzing_journal_name IS NULL
            LIMIT 3
        """).show()
        
        print("Harzing only:")
        db.sql("""
            SELECT unique_issn_list, harzing_journal_name, harzing_field
            FROM comprehensive_journals 
            WHERE era_journal_name IS NULL AND harzing_journal_name IS NOT NULL
            LIMIT 3
        """).show()
        
        print("Both sources (overlapping):")
        db.sql("""
            SELECT unique_issn_list, era_journal_name, harzing_journal_name, era_field, harzing_field
            FROM comprehensive_journals 
            WHERE era_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL
            LIMIT 3
        """).show()
        
        print("=== TOTAL JOURNALS ===")
        db.sql("SELECT COUNT(*) as total_journals FROM comprehensive_journals").show()
        
        # Save the result
        print("=== SAVING COMPREHENSIVE JOURNAL LIST ===")
        db.sql(f"""
            COPY comprehensive_journals 
            TO '{DATA}/comprehensive_journal_list.csv' 
            (FORMAT 'csv', HEADER true)
        """)
        
        print("Results saved to comprehensive_journal_list.csv")
    return

def main():
    load_journals()
    return

if __name__ == "__main__":
    main()
    print("FINISHED!")