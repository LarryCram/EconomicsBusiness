"""
Journal Database Assembly Script

Combines journal data from four major academic sources:
- WOS/JCR (Web of Science Journal Citation Reports): Economics and Business journals
- ERA 2023 (Excellence in Research for Australia): Field of Research codes 35 & 38
- Harzing Journal Quality List: Economics and business journal rankings
- Scopus (Elsevier): ASJC codes 1400 (Business/Management/Accounting) or 2000 (Economics/EF)

ISSN normalization: Scopus ISSNs lack the dash (8 raw digits); converted to XXXX-XXXX
to match ERA/WOS/Harzing format before joining.

Output: comprehensive_journal_list.parquet in the configured WORKING/parquet/ directory.
Columns: unique_issn_list, era_journal_name, era_for_codes, harzing_journal_name,
         harzing_field, wos_journal_name, wos_abbreviation, wos_categories,
         scopus_journal_name, scopus_asjc
"""
from pathlib import Path
import duckdb
import yaml
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

config_path = Path('./config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)
    print(f'{config = }')
    PROJECT_FOLDER = Path(config['PROJECT_ROOT'])
    DATA = PROJECT_FOLDER / Path(config.get('DATA'))
    WORKING = Path(config.get('WORKING'))


def assemble_wos_journal_list(db):
    """Create wos temp table from xlsx files."""
    wos_files = list(Path(DATA / 'source_masters/WOS_JCR').glob('*.xlsx'))
    print(f'{wos_files = }')

    union_parts = []
    for file in wos_files:
        union_parts.append(f"""
            SELECT list_distinct(list_filter([ISSN, eISSN], x -> x IS NOT NULL AND x != 'N/A')) AS wos_issn,
                   ISSN, eISSN,
                   "Journal name" AS jcr_name,
                   "JCR Abbreviation" AS jcr_abrv,
                   Category
            FROM read_xlsx('{file}', range='A3:AZ999', header=true)
            WHERE jcr_name NOT NULL
              AND Category IN ('ECONOMICS', 'MANAGEMENT', 'BUSINESS', 'BUSINESS, FINANCE', 'TRANSPORTATION')
            GROUP BY ALL
        """)

    db.execute("CREATE OR REPLACE VIEW all_jcr_data AS " + " UNION ALL ".join(union_parts))
    db.sql("SELECT Category, count(Category) AS counts FROM all_jcr_data GROUP BY ALL ORDER BY counts DESC").show()

    db.sql("""
        CREATE TEMP TABLE wos AS
        SELECT
            list_distinct(flatten(list(wos_issn))) AS wos_issn,
            MIN(jcr_name) AS jcr_name,
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


def assemble_scopus_journal_list(db):
    """Create scopus temp table from Scopus_Mar_2026.xlsx, filtered to ASJC 1400 or 2000.

    Scopus ISSNs are 8 raw digits (no dash); converted to XXXX-XXXX to match other sources.
    Includes all source types (Journal, Book Series, Trade Journal) and both Active/Inactive.
    """
    xl_path = DATA / 'source_masters/Scopus_Mar_2026.xlsx'
    df = pd.read_excel(xl_path, sheet_name='Scopus Sources Mar. 2026')

    col_1400 = next(c for c in df.columns if str(c).startswith('1400'))
    col_2000 = next(c for c in df.columns if str(c).startswith('2000'))

    df = df[(df[col_1400].notna() | df[col_2000].notna()) & (df['Source Type'] == 'Journal')].copy()
    df = df[['Sourcerecord ID', 'Source Title', 'ISSN', 'EISSN',
             'Source Type', 'Active or Inactive', col_1400, col_2000]].rename(columns={
        'Sourcerecord ID':   'source_id',
        'Source Title':      'scopus_name',
        'ISSN':              'print_issn',
        'EISSN':             'eissn',
        'Source Type':       'source_type',
        'Active or Inactive': 'active_status',
        col_1400:            'asjc_1400',
        col_2000:            'asjc_2000',
    })
    db.register('scopus_raw', df)

    db.sql("""
        CREATE TEMP TABLE scopus AS
        SELECT
            list_distinct(list_filter([
                CASE WHEN print_issn IS NOT NULL AND len(TRIM(CAST(print_issn AS VARCHAR))) = 8
                     THEN SUBSTRING(TRIM(CAST(print_issn AS VARCHAR)), 1, 4) || '-' ||
                          SUBSTRING(TRIM(CAST(print_issn AS VARCHAR)), 5, 4) END,
                CASE WHEN eissn IS NOT NULL AND len(TRIM(CAST(eissn AS VARCHAR))) = 8
                     THEN SUBSTRING(TRIM(CAST(eissn AS VARCHAR)), 1, 4) || '-' ||
                          SUBSTRING(TRIM(CAST(eissn AS VARCHAR)), 5, 4) END
            ], x -> x IS NOT NULL)) AS scopus_issn,
            scopus_name,
            list_distinct(list_filter([
                CASE WHEN asjc_1400 IS NOT NULL THEN '1400' END,
                CASE WHEN asjc_2000 IS NOT NULL THEN '2000' END
            ], x -> x IS NOT NULL)) AS scopus_asjc,
            source_type,
            active_status
        FROM scopus_raw
    """)

    print("=== SCOPUS BASE LIST ===")
    db.sql("SELECT COUNT(*) AS scopus_count FROM scopus").show()
    db.sql("SELECT * FROM scopus LIMIT 5").show()
    db.sql("""
        SELECT source_type, active_status, COUNT(*) AS n
        FROM scopus GROUP BY ALL ORDER BY n DESC
    """).show()
    db.sql("""
        SELECT
            list_contains(scopus_asjc, '1400') AS has_1400,
            list_contains(scopus_asjc, '2000') AS has_2000,
            COUNT(*) AS n
        FROM scopus GROUP BY ALL ORDER BY n DESC
    """).show()


def load_journals(db):
    assemble_wos_journal_list(db)

    # ERA temp table
    db.sql(f"""
        CREATE TEMP TABLE era AS
        SELECT list_distinct(list_filter(flatten(list(["ISSN 1", "ISSN 2", "ISSN 3"])),
                             x -> x IS NOT NULL AND x != 'N/A')) AS era_ISSN,
               Title AS era_name,
               list_filter([
                   CASE WHEN "FoR 1" IS NOT NULL AND "FoR 1" != '' THEN "FoR 1" END,
                   CASE WHEN "FoR 2" IS NOT NULL AND "FoR 2" != '' THEN "FoR 2" END,
                   CASE WHEN "FoR 3" IS NOT NULL AND "FoR 3" != '' THEN "FoR 3" END
               ], x -> x IS NOT NULL) AS era_for_codes,
        FROM read_xlsx('{DATA}/source_masters/ecobus_journal_harzing_era.xlsx',
                       sheet='ERA2023 Submission Journal List',
                       range='A:P', header=true, all_varchar=true)
        WHERE LEFT("FoR 1", 2) IN ('35', '38')
          AND ("FoR 2" IS NULL OR "FoR 2" = '' OR LEFT("FoR 2", 2) IN ('35', '38'))
          AND ("FoR 3" IS NULL OR "FoR 3" = '' OR LEFT("FoR 3", 2) IN ('35', '38'))
        GROUP BY "ERA Journal Id", Title, "FoR 1", "FoR 2", "FoR 3"
    """)
    print("=== ERA BASE LIST ===")
    db.sql("SELECT COUNT(*) as era_count FROM era").show()
    db.sql("SELECT * FROM era LIMIT 10").show()
    db.sql("SELECT era_name, COUNT(DISTINCT era_name) AS count_name FROM ERA GROUP BY ALL ORDER BY count_name DESC").show()

    # Harzing temp table
    db.sql(f"""
        CREATE TEMP TABLE harzing AS
        SELECT DISTINCT
            list_filter([ISSN], x -> x IS NOT NULL AND x != 'N/A') as harzing_issn,
            Journal as journal_name,
            Subject_areas as field,
        FROM read_xlsx('{DATA}/source_masters/ecobus_journal_harzing_era.xlsx',
                       sheet='Harzing', range='A:C', header=true) h
        WHERE ISSN IS NOT NULL AND ISSN != 'N/A'
    """)
    print("=== HARZING CLEAN TABLE ===")
    db.sql("SELECT COUNT(*) as harzing_count FROM harzing").show()
    db.sql("SELECT * FROM harzing LIMIT 5").show()

    # Scopus temp table
    assemble_scopus_journal_list(db)

    # Join 1: ERA + Harzing
    db.sql("""
        CREATE TEMP TABLE era_harzing AS
        SELECT DISTINCT
            list_distinct(list_filter(COALESCE(e.era_ISSN, []) || COALESCE(h.harzing_issn, []),
                                      x -> x IS NOT NULL AND x != 'N/A')) AS combined_issn_list,
            e.era_name as era_journal_name,
            e.era_for_codes as era_for_codes,
            h.journal_name as harzing_journal_name,
            h.field as harzing_field
        FROM era e
        FULL OUTER JOIN harzing h ON len(list_intersect(e.era_ISSN, h.harzing_issn)) > 0
    """)

    print("=== ERA + HARZING INTERMEDIATE RESULTS ===")
    db.sql("SELECT COUNT(*) as era_harzing_count FROM era_harzing").show()
    n_era = db.sql("SELECT COUNT(*) FROM era").fetchone()[0]
    n_harzing = db.sql("SELECT COUNT(*) FROM harzing").fetchone()[0]
    n_eh = db.sql("SELECT COUNT(*) FROM era_harzing").fetchone()[0]
    if n_eh > n_era + n_harzing:
        print(f"WARNING: many-to-many fan-out in ERA+Harzing join "
              f"({n_eh} rows > {n_era} ERA + {n_harzing} Harzing).")
        db.sql("""SELECT era_journal_name, COUNT(*) AS n FROM era_harzing
                  WHERE era_journal_name IS NOT NULL
                  GROUP BY era_journal_name HAVING n > 1 ORDER BY n DESC""").show()

    print("\n=== ERA+HARZING BREAKDOWN ===")
    db.sql("""
        SELECT
            CASE
                WHEN era_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL THEN 'Both ERA and Harzing'
                WHEN era_journal_name IS NOT NULL THEN 'ERA only'
                WHEN harzing_journal_name IS NOT NULL THEN 'Harzing only'
                ELSE 'Neither (error)'
            END as source_type, COUNT(*) as count
        FROM era_harzing GROUP BY source_type
    """).show()

    # Join 2: ERA+Harzing + WOS
    db.sql("""
        CREATE TEMP TABLE era_harzing_wos AS
        SELECT DISTINCT
            list_distinct(list_filter(COALESCE(eh.combined_issn_list, []) || COALESCE(w.wos_issn, []),
                                      x -> x IS NOT NULL AND x != 'N/A')) AS unique_issn_list,
            eh.era_journal_name,
            eh.era_for_codes,
            eh.harzing_journal_name,
            eh.harzing_field,
            w.jcr_name as wos_journal_name,
            w.jcr_abrv as wos_abbreviation,
            w.categories as wos_categories
        FROM era_harzing eh
        FULL OUTER JOIN wos w ON len(list_intersect(eh.combined_issn_list, w.wos_issn)) > 0
    """)

    print("=== ERA+HARZING+WOS INTERMEDIATE RESULTS ===")
    n_wos = db.sql("SELECT COUNT(*) FROM wos").fetchone()[0]
    n_ehw = db.sql("SELECT COUNT(*) FROM era_harzing_wos").fetchone()[0]
    db.sql("SELECT COUNT(*) AS era_harzing_wos_count FROM era_harzing_wos").show()
    if n_ehw > n_eh + n_wos:
        print(f"WARNING: many-to-many fan-out in +WOS join "
              f"({n_ehw} rows > {n_eh} ERA+Harzing + {n_wos} WOS).")
        db.sql("""SELECT wos_journal_name, COUNT(*) AS n FROM era_harzing_wos
                  WHERE wos_journal_name IS NOT NULL
                  GROUP BY wos_journal_name HAVING n > 1 ORDER BY n DESC""").show()

    # Join 3: ERA+Harzing+WOS + Scopus → final comprehensive_journals
    db.sql("""
        CREATE TEMP TABLE comprehensive_journals AS
        SELECT DISTINCT
            list_distinct(list_filter(COALESCE(ehw.unique_issn_list, []) || COALESCE(s.scopus_issn, []),
                                      x -> x IS NOT NULL AND x != 'N/A')) AS unique_issn_list,
            ehw.era_journal_name,
            ehw.era_for_codes,
            ehw.harzing_journal_name,
            ehw.harzing_field,
            ehw.wos_journal_name,
            ehw.wos_abbreviation,
            ehw.wos_categories,
            s.scopus_name AS scopus_journal_name,
            s.scopus_asjc
        FROM era_harzing_wos ehw
        FULL OUTER JOIN scopus s ON len(list_intersect(ehw.unique_issn_list, s.scopus_issn)) > 0
    """)

    print("=== FINAL COMPREHENSIVE RESULTS ===")
    n_scopus = db.sql("SELECT COUNT(*) FROM scopus").fetchone()[0]
    n_final = db.sql("SELECT COUNT(*) FROM comprehensive_journals").fetchone()[0]
    db.sql("SELECT COUNT(*) as final_count FROM comprehensive_journals").show()
    if n_final > n_ehw + n_scopus:
        print(f"WARNING: many-to-many fan-out in +Scopus join "
              f"({n_final} rows > {n_ehw} ERA+Harzing+WOS + {n_scopus} Scopus).")
        db.sql("""SELECT scopus_journal_name, COUNT(*) AS n FROM comprehensive_journals
                  WHERE scopus_journal_name IS NOT NULL
                  GROUP BY scopus_journal_name HAVING n > 1 ORDER BY n DESC""").show()

    print("\n=== FINAL BREAKDOWN BY SOURCE PRESENCE ===")
    db.sql("""
        SELECT
            CASE
                WHEN era_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL
                     AND wos_journal_name IS NOT NULL AND scopus_journal_name IS NOT NULL THEN 'All four sources'
                WHEN era_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL
                     AND wos_journal_name IS NOT NULL THEN 'ERA + Harzing + WOS'
                WHEN era_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL
                     AND scopus_journal_name IS NOT NULL THEN 'ERA + Harzing + Scopus'
                WHEN era_journal_name IS NOT NULL AND wos_journal_name IS NOT NULL
                     AND scopus_journal_name IS NOT NULL THEN 'ERA + WOS + Scopus'
                WHEN harzing_journal_name IS NOT NULL AND wos_journal_name IS NOT NULL
                     AND scopus_journal_name IS NOT NULL THEN 'Harzing + WOS + Scopus'
                WHEN era_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL THEN 'ERA + Harzing'
                WHEN era_journal_name IS NOT NULL AND wos_journal_name IS NOT NULL THEN 'ERA + WOS'
                WHEN era_journal_name IS NOT NULL AND scopus_journal_name IS NOT NULL THEN 'ERA + Scopus'
                WHEN harzing_journal_name IS NOT NULL AND wos_journal_name IS NOT NULL THEN 'Harzing + WOS'
                WHEN harzing_journal_name IS NOT NULL AND scopus_journal_name IS NOT NULL THEN 'Harzing + Scopus'
                WHEN wos_journal_name IS NOT NULL AND scopus_journal_name IS NOT NULL THEN 'WOS + Scopus'
                WHEN era_journal_name IS NOT NULL THEN 'ERA only'
                WHEN harzing_journal_name IS NOT NULL THEN 'Harzing only'
                WHEN wos_journal_name IS NOT NULL THEN 'WOS only'
                WHEN scopus_journal_name IS NOT NULL THEN 'Scopus only'
                ELSE 'None (error)'
            END AS source_combination,
            COUNT(*) AS n
        FROM comprehensive_journals
        GROUP BY source_combination
        ORDER BY n DESC
    """).show()

    print("=== TOTAL JOURNALS ===")
    db.sql("SELECT COUNT(*) as total_journals FROM comprehensive_journals").show()

    print("=== SAVING COMPREHENSIVE JOURNAL LIST ===")
    db.sql(f"""
        COPY comprehensive_journals
        TO '{WORKING}/parquet/comprehensive_journal_list.parquet'
        (FORMAT PARQUET)
    """)
    print("Results saved to comprehensive_journal_list.parquet")


def main():
    with duckdb.connect() as db:
        load_journals(db)
        db.sql(f"""
            COPY (SELECT * FROM '{WORKING}/parquet/comprehensive_journal_list.parquet')
            TO '{DATA}/comprehensive_journal_list.csv' (FORMAT CSV)
        """)


if __name__ == "__main__":
    main()
    print("FINISHED!")
