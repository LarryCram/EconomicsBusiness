from pathlib import Path
import pandas as pd
import numpy as np
from pyalex import Authors, autocomplete, config
import duckdb

config.api_key = "OchtksdohLaziRq08C4IJP"
DATA_PATH = '/home/lc/Projects/EconomicsBusiness/data'
print(f'{Path(DATA_PATH).exists() = }')
PARQUET_PATH = '/home/lc/m/openalex_feb26/parquet'
print(f'{Path(PARQUET_PATH).exists() = }')

def loader():
    sample = pd.read_excel(f'{DATA_PATH}/researchers_results.xlsx').sort_values('PUB', ascending=False).iloc[:, :10]
    print(f'{sample.shape = }\n{sample.head()}')
    return sample

def matcher(df):
    results = []
    for row in df.itertuples():
        parts = row.Research_Profile.split(',', maxsplit=1)
        if len(parts) == 2:
            name = ' '.join([parts[1].strip().replace('Maryan', 'Maryam').replace('Arvin', 'Arvind').replace('Luis, V', 'Luis V.'), parts[0].strip().replace('Casalo', 'Casaló')])
        else:
            name = row.Research_Profile
        authors = pd.DataFrame(Authors().autocomplete(name))
        if len(authors) > 0:
            authors = authors.sort_values('works_count', ascending=False)
            authors.insert(0, 'Name', row.Research_Profile)
            authors.insert(1, 'Group', row.Group)
            authors.insert(2, 'Class', row.Class)
            results.append(authors.iloc[[0]])
            authors_template = authors.iloc[[0]]
        else:
            authors = authors_template.reset_index(drop=True)
            print(authors.head())
            authors.loc[0] = np.nan
            authors.at[0, 'Name'] = row.Research_Profile
            authors.at[0, 'Group'] = row.Group
            authors.at[0, 'Class'] = row.Class
            print(authors.head())
            results.append(authors.iloc[[0]])
            print(f'DID NOT FIND {row.Research_Profile = }')
    df = pd.concat(results)
    print(df)
    return df

def load_researchers():
    with duckdb.connect() as db:
        sql = f"""
            SELECT DISTINCT author_id, author_name, count(DISTINCT work_id) AS works_count 
                FROM '{PARQUET_PATH}/authorships.parquet'
                GROUP BY ALL
                ORDER BY works_count DESC
            """
        db.sql(sql).show()
        researchers = db.sql(sql).df()
        return researchers

def main():
    researchers = load_researchers()
    sample = loader()
    df = matcher(sample)
    df.to_csv(f'{DATA_PATH}/verified_researchers.csv')
    return

if __name__ == "__main__":
    main()
    print("FINISHED !")