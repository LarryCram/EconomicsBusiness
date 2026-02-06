from pathlib import Path
import pandas as pd
import duckdb

WORK_FOLDER = '/home/lc/Projects/EconomicsBusiness/2026_study/DATA'
print(f'{Path(WORK_FOLDER).exists() = }')

def read_journals():
    journals = pd.read_excel(Path(WORK_FOLDER)/'ecobus_journal_institution_results.xlsx')
    print(f'{journals.shape = }\n{journals.head()}')
    return journals

def read_incites():
    incites = pd.read_csv(Path(WORK_FOLDER)/'wos_jcr.csv')
    print(f'{incites.shape = }\n{incites.head()}\n{incites.info()}')
    return incites

def match_journals(journals, incites):
    with duckdb.connect() as db:
        sql = """
            SELECT DISTINCT o.id as source_id, o.id[23:]::BIGINT as source_idx, 
                    i."Journal name" as inCites, journal, o.display_name as journal_name, i.ISSN, list_sort(list(Category))
            FROM journals j
            LEFT JOIN incites i
            ON lower(journal) = lower("Journal name")
            LEFT JOIN '/home/lc/m/openalex_june25/parquet/sources.parquet' o
            ON list_contains(o.issn, i.issn)
            WHERE o.id IS NOT NULL
            GROUP BY ALL
            """
        db.sql(sql).show()
        out_file = str(Path(WORK_FOLDER)/'econ_bus_journal_oa.csv') 
        db.sql(f"COPY ( {sql} ) TO '{out_file}'")
    return


def main():
    journals = read_journals()
    incites = read_incites()
    match_journals(journals, incites)


if __name__ == "__main__":
    main()
    print("FINISHED!")