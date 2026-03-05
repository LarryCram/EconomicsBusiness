"""
Journal Database Assembly Script

This script combines journal data from three major academic sources:
- WOS/JCR (Web of Science Journal Citation Reports): Economics and Business journals
- ERA 2023 (Excellence in Research for Australia): Field of Research codes 35 & 38
- Harzing Journal Quality List: Economics and business journal rankings

The script performs ISSN-based matching to merge overlapping journals across sources,
creating a comprehensive journal database with metadata from all three sources.
Uses DuckDB for efficient SQL operations on Excel files and produces a unified
parquet file containing journal names, abbreviations, ISSN lists, field classifications,
and category information for downstream bibliometric analysis.

Output: comprehensive_journal_list.parquet in the configured DATA directory
"""
from pathlib import Path
import duckdb
import yaml
import pandas as pd

# Configure pandas display options for wider output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Load configuration relative to this script
script_dir = Path(__file__)
config_path = Path('./config.yaml')

with open(config_path) as f:
    config = yaml.safe_load(f)
    print(f'{config = }')
    PROJECT_FOLDER = config['PROJECT_ROOT']
    DATA = PROJECT_FOLDER / Path(config.get('DATA'))
    WORKING = Path(config.get('WORKING'))

def assemble_wos_journal_list(db):
    """Create WOS temp table from xlsx files"""
    wos_files = list(Path(DATA/'source_masters/WOS_JCR').glob('*.xlsx'))
    print(f'{wos_files = }')

    # Build UNION ALL query
    union_parts = []
    for file in wos_files:
        union_parts.append(f"""
                            SELECT list_distinct(list_filter([ISSN, eISSN], x -> x IS NOT NULL AND x != 'N/A')) AS wos_issn, ISSN, eISSN, 
                                "Journal name" AS jcr_name, 
                                "JCR Abbreviation" AS jcr_abrv, 
                                Category
                            FROM read_xlsx('{file}', range='A3:AZ999', header=true)
                            WHERE jcr_name NOT NULL AND 
                                not list_contains(
                                    ['ETHNIC STUDIES', 'DEMOGRAPHY', 'DEVELOPMENT STUDIES', 
                                    'URBAN STUDIES', 'FORESTRY', 'HEALTH POLICY & SERVICES',
                                    'GEOGRAPHY', 'AREA STUDIES', 'HEALTH CARE SCIENCES & SERVICES',
                                    'ENVIRONMENTAL STUDIES', 'ECOLOGY', 'COMMUNICATION',
                                    'PUBLIC ADMINISTRATION'], Category)
                            GROUP BY ALL
                        """)

    # Create a view with all data
    create_view_sql = "CREATE OR REPLACE VIEW all_jcr_data AS " + " UNION ALL ".join(union_parts)
    db.execute(create_view_sql)

    db.sql("SELECT Category, count(Category) AS counts FROM all_jcr_data GROUP BY ALL ORDER BY counts DESC").show()

    # Create temp table with aggregated WOS data
    db.sql("""
        CREATE TEMP TABLE wos AS
        SELECT 
            list_distinct(flatten(list(wos_issn))) AS wos_issn,
            ANY_VALUE(jcr_name) AS jcr_name,
            jcr_abrv,
            list_distinct(list(Category)) AS categories
        FROM all_jcr_data 
        WHERE jcr_abrv IS NOT NULL
        GROUP BY jcr_abrv
        ORDER BY jcr_abrv
    """)
    
    print("=== WOS BASE LIST ===")
    db.sql("SELECT COUNT(*) as wos_count FROM wos").show()
    db.sql("SELECT * FROM wos LIMIT 10").show()
    
    return

def load_journals(db):
    # Create temp table for WOS data
    assemble_wos_journal_list(db)
    
    # Create temp table for ERA data
    db.sql(f"""
        CREATE TEMP TABLE era AS
        SELECT DISTINCT list_distinct(list_filter(["ISSN 1", "ISSN 2", "ISSN 3"], x -> x IS NOT NULL AND x != 'N/A')) AS era_ISSN,
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
            list_filter([ISSN], x -> x IS NOT NULL AND x != 'N/A') as harzing_issn,
            Journal as journal_name,
            Subject_areas as field,
        FROM read_xlsx('{DATA}/source_masters/ecobus_journal_harzing_era.xlsx', sheet='Harzing', range='A:C', header=true) h
        WHERE ISSN IS NOT NULL AND ISSN != 'N/A'
    """)
    
    print("=== HARZING CLEAN TABLE ===")
    db.sql("SELECT COUNT(*) as harzing_count FROM harzing").show()
    db.sql("SELECT * FROM harzing LIMIT 5").show()

    
# First join: ERA and Harzing on ISSN overlaps
    db.sql("""
        CREATE TEMP TABLE era_harzing AS
        SELECT DISTINCT
            list_distinct(list_filter(COALESCE(e.era_ISSN, []) || COALESCE(h.harzing_issn, []), x -> x IS NOT NULL AND x != 'N/A')) as combined_issn_list,
            e.era_name as era_journal_name,
            e.FoR_name as era_field,
            h.journal_name as harzing_journal_name,
            h.field as harzing_field
        FROM era e
        FULL OUTER JOIN harzing h 
            ON len(list_intersect(e.era_ISSN, h.harzing_issn)) > 0
    """)
    
    print("=== ERA + HARZING INTERMEDIATE RESULTS ===")
    db.sql("SELECT COUNT(*) as era_harzing_count FROM era_harzing").show()
    
    print("\\n=== ERA+HARZING BREAKDOWN ===")
    db.sql("""
        SELECT 
            CASE 
                WHEN era_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL THEN 'Both ERA and Harzing'
                WHEN era_journal_name IS NOT NULL THEN 'ERA only'  
                WHEN harzing_journal_name IS NOT NULL THEN 'Harzing only'
                ELSE 'Neither (error)'
            END as source_type,
            COUNT(*) as count
        FROM era_harzing
        GROUP BY source_type
    """).show()

    # Second join: Add WOS to the ERA+Harzing combination
    db.sql("""
        CREATE TEMP TABLE comprehensive_journals AS
        SELECT DISTINCT
            list_distinct(list_filter(COALESCE(eh.combined_issn_list, []) || COALESCE(w.wos_issn, []), x -> x IS NOT NULL AND x != 'N/A')) as unique_issn_list,
            eh.era_journal_name,
            eh.era_field,
            eh.harzing_journal_name,
            eh.harzing_field,
            w.jcr_name as wos_journal_name,
            w.jcr_abrv as wos_abbreviation,
            w.categories as wos_categories
        FROM era_harzing eh
        FULL OUTER JOIN wos w
            ON len(list_intersect(eh.combined_issn_list, w.wos_issn)) > 0
    """)
    
    # Show summary statistics  
    print("=== FINAL COMPREHENSIVE RESULTS ===")
    db.sql("SELECT COUNT(*) as final_count FROM comprehensive_journals").show()
    
    print("\n=== FINAL BREAKDOWN BY SOURCE PRESENCE ===")
    db.sql("""
        SELECT 
            CASE 
                WHEN era_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL AND wos_journal_name IS NOT NULL THEN 'All three sources'
                WHEN era_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL THEN 'ERA + Harzing'
                WHEN era_journal_name IS NOT NULL AND wos_journal_name IS NOT NULL THEN 'ERA + WOS'
                WHEN harzing_journal_name IS NOT NULL AND wos_journal_name IS NOT NULL THEN 'Harzing + WOS'
                WHEN era_journal_name IS NOT NULL THEN 'ERA only'  
                WHEN harzing_journal_name IS NOT NULL THEN 'Harzing only'
                WHEN wos_journal_name IS NOT NULL THEN 'WOS only'
                ELSE 'None (error)'
            END as source_type,
            COUNT(*) as count
        FROM comprehensive_journals
        GROUP BY source_type
        ORDER BY count DESC
    """).show()
    
    print("\n=== EXAMPLES OF EACH TYPE ===")
    print("ERA only:")
    db.sql("""
        SELECT unique_issn_list, era_journal_name, era_field
        FROM comprehensive_journals 
        WHERE era_journal_name IS NOT NULL AND harzing_journal_name IS NULL AND wos_journal_name IS NULL
        LIMIT 3
    """).show()
    
    print("Harzing only:")
    db.sql("""
        SELECT unique_issn_list, harzing_journal_name, harzing_field
        FROM comprehensive_journals 
        WHERE era_journal_name IS NULL AND harzing_journal_name IS NOT NULL AND wos_journal_name IS NULL
        LIMIT 3
    """).show()
    
    print("WOS only:")
    db.sql("""
        SELECT unique_issn_list, wos_journal_name, wos_abbreviation, wos_categories
        FROM comprehensive_journals 
        WHERE era_journal_name IS NULL AND harzing_journal_name IS NULL AND wos_journal_name IS NOT NULL
        LIMIT 3
    """).show()
    
    print("All three sources:")
    db.sql("""
        SELECT unique_issn_list, era_journal_name, harzing_journal_name, wos_journal_name, wos_abbreviation
        FROM comprehensive_journals 
        WHERE era_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL AND wos_journal_name IS NOT NULL
        LIMIT 3
    """).show()
    
    print("=== TOTAL JOURNALS ===")
    db.sql("SELECT COUNT(*) as total_journals FROM comprehensive_journals").show()
    
    # Save the result
    print("=== SAVING COMPREHENSIVE JOURNAL LIST ===")
    db.sql(f"""
        COPY comprehensive_journals 
        TO '{WORKING}/parquet/comprehensive_journal_list.parquet' 
        (FORMAT PARQUET) --(FORMAT 'csv', HEADER true)
    """)
    
    print("Results saved to comprehensive_journal_list.parquet")
    return

def main():
    with duckdb.connect() as db:
        load_journals(db)  # Pass db connection to load_journals
        sql = f"""
            COPY (
            SELECT * 
                FROM '{WORKING}/parquet/comprehensive_journal_list.parquet' 
            ) TO '{DATA}/comprehensive_journal_list.csv' (FORMAT CSV)
        """
        db.sql(sql)
    return

if __name__ == "__main__":
    main()
    print("FINISHED!")