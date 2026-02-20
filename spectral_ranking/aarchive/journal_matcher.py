from pathlib import Path
import pandas as pd
import duckdb

WORK_FOLDER = '/home/lc/Projects/EconomicsBusiness/data'
print(f'{Path(WORK_FOLDER).exists() = }')

def read_journals():
    journals = pd.read_excel(Path(WORK_FOLDER)/'ecobus_journal_institution_results.xlsx')
    print(f'{journals.shape = }\n{journals.head()}')
    return journals

def read_incites():
    incites = pd.read_csv(Path(WORK_FOLDER)/'journals_incites.csv')
    print(f'{incites.shape = }\n{incites.head()}\n{incites.info()}')
    return incites

def append_difficult_journals(df):
        difficult = pd.read_csv(Path(WORK_FOLDER)/'sources_difficult.csv')
        print(f'{difficult.shape = }\n{difficult.head()}')
        df = pd.concat([df, difficult], axis=0)
        return df

def match_journals(journals, incites):
    with duckdb.connect() as db:
        sql = """
            SELECT DISTINCT o.id as source_id, 
                    i.journal as dd_name, j.journal AS incites_name, o.display_name as journal_name, 
                    i.ISSN, categories
            FROM journals j
            LEFT JOIN incites i
            ON lower(i.journal) = lower(j.journal)
            LEFT JOIN '/home/lc/m/openalex_june25/parquet/sources.parquet' o
            ON list_contains(o.issn, i.issn)
            WHERE o.id IS NOT NULL
            GROUP BY ALL
            """
        df = db.sql(sql).df()
        print(f'{df.shape = }\n{df.head()}')
        df = append_difficult_journals(df)
        print(f'{df.shape = }\n{df.head()}')
        out_file = str(Path(WORK_FOLDER)/'journals_to_sources_oa.csv') 
        db.sql(f"COPY (SELECT * FROM df) TO '{out_file}'")

    return df

def main():
    journals = read_journals()
    incites = read_incites()
    match_journals(journals, incites)

if __name__ == "__main__":
    main()
    print("FINISHED!")