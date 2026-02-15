from pathlib import Path
import duckdb
import glob

# 1. Find all xlsx files
DATA_PATH = Path('/home/lc/Projects/EconomicsBusiness/data/')
print(f'{DATA_PATH.exists() = }')
files = list(DATA_PATH.glob('WOS_JCR/*.xlsx'))
print('{files = }')

# 2. Create the table from the first file
with duckdb.connect() as db:
    file = files[0]
    sql = f"""CREATE TABLE all_data AS SELECT * FROM read_xlsx('{file}', range="A3:F:999", header=true) WHERE "JCR Abbreviation" NOT NULL"""
    db.sql(sql)
    db.sql("SELECT * FROM all_data").show()


    # 3. Insert the rest
    for file in files[1:]:
        sql = f""" INSERT INTO all_data SELECT * FROM read_xlsx('{file}', range="A3:F:999", header=true) WHERE "JCR Abbreviation" NOT NULL"""
        db.sql(sql)
        db.sql("SELECT * FROM all_data").show()

    # 4. Pack Categories into multi-category journals
    df = db.sql("""SELECT "Journal name" AS journal, ISSN, eISSN, list(Category) AS categories FROM all_data GROUP BY ALL""").df().drop_duplicates('journal')
    print(f'{df.shape = }\n{df.head()}')

    # 5. Save to combined journal spreadsheet
    out_file = DATA_PATH/'journals_incites.csv'
    sql = f"""COPY (
                SELECT "Journal name" AS journal, ISSN, eISSN, list_sort(list(Category)) AS categories FROM all_data GROUP BY ALL
                ) TO '{out_file}' (FORMAT csv)"""
    db.sql(sql)