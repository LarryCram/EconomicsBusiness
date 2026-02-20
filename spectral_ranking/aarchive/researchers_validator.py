from pathlib import Path
import duckdb
import pandas as pd

DATA_PATH = '/home/lc/Projects/EconomicsBusiness/data'
print(f'{Path(DATA_PATH).exists() = }')
PARQUET_PATH = '/home/lc/m/openalex_feb26/parquet'
print(f'{Path(PARQUET_PATH).exists() = }')

def load_all_authors():
    # Load authors from authorships in a specified year range
    with duckdb.connect() as db:
        sql = f"""
                SELECT author_id, author_name, count(DISTINCT work_id) AS works_count
                    FROM '{PARQUET_PATH}/authorships.parquet' a
                    lEFT JOIN (SELECT work_id, publication_year FROM '{PARQUET_PATH}/works.parquet') s
                    USING (work_id)
                    WHERE publication_year BETWEEN 2000 AND 2024
                    GROUP BY ALL 
                    ORDER BY works_count DESC
            """
        authors = db.sql(sql).df()
        print(f'{authors.shape = }\n{authors.head()}')
        linked = link_authors(db, authors)

    return linked

def link_authors(db, authors):
    # Link the researchers to the full authors
    researchers = pd.read_csv(f"{DATA_PATH}/verified_researchers.csv")
    print(f'{researchers.shape = }\n{researchers.head()}')
    sql = """
        SELECT r.Name, r.Group, r.Class, r.id, r.works_count as works_count_sample, a.author_id, a.author_name, a.works_count
            FROM authors a
            RIGHT JOIN researchers r
            ON author_id = id
            ORDER BY a.works_count DESC
        """
    linked = db.sql(sql).df()
    print(f'{linked.shape = }\n{linked.head()}')
    return linked

def select_authors(linked):
    mask = [True if r and n else False for r, n in zip(linked.Name, linked.author_name)]
    print(f'Number of linked researchers {len(linked[mask]) = }')
    mask = [True if r and (not n) else False for r, n in zip(linked.Name, linked.author_name)]
    print(f'Number of UN-linked researchers {len(linked[mask]) = }')
    unlinked_researchers = linked[mask]
    print(f'{unlinked_researchers.shape = }\n{unlinked_researchers.head(46)}')
    researchers = linked[~linked.Name.isna()]
    print(f'{researchers.shape = }\n{researchers.head()}')
    return

def main():
    linked = load_all_authors()
    select_authors(linked)
    # linked.to_csv(f'{DATA_PATH}/linked_authors.csv')
    return

if __name__ == "__main__":
    main()
    print("FINISHED !")