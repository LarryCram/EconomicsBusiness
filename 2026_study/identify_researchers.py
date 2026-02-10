import pandas as pd
from pyalex import Authors, autocomplete, config
import duckdb

config.api_key = "OchtksdohLaziRq08C4IJP"

def loader():
    df = pd.read_excel('/home/lc/Dropbox/ECONOMICS_BUSINESS/DATAFILES/researchers_sample.xlsx')
    print(f'{df.shape = }\n{df.head()}')
    return df

def matcher(df):
    results = []
    for row in df.itertuples():
        authors = pd.DataFrame(Authors().autocomplete(row.NAME))
        if len(authors) > 1:
            authors = authors.sort_values('works_count', ascending=False)
            authors.insert(0, 'NAME', row.NAME)
            authors.insert(1, 'Group', row.Group)
            results.append(authors.iloc[[0]])
        else:
            print(f'DID NOT FIND {row.NAME = }')
    df = pd.concat(results)
    print(df)
    return df

def db_inspector():
    with duckdb.connect() as db:
        db.sql("ATTACH IF NOT EXISTS '/home/lc/m/working/econ.duckdb'")
        df = db.sql("SHOW ALL TABLES").df()
        df = db.sql("SELECT * FROM econ.authors").df()
        print(df.info())
        return

def main():
    db_inspector()
    # df = loader()
    # df = matcher(df)
    # df.to_csv('./DATA/matched_authors.csv')
    return

if __name__ == "__main__":
    main()
    print("FINISHED !")