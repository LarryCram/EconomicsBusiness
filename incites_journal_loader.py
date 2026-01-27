from logging.config import fileConfig
from pathlib import Path
import time
import duckdb
import pandas as pd

# incites_path = Path(r'C:\Users\LC.RSPE-056310\Dropbox\ECONOMICS_BUSINESS\DATA\WOS_JCR\\')
incites_path = Path(r'/home/lc/Projects/EconomicsBusiness/DATA/WOS_JCR')
assembled_file = Path(r'/home/lc/Projects/EconomicsBusiness/DATAFILES/esi_eco_bus.xlsx')

def load_incites(db):
    print(f'{incites_path.exists() = }')
    file_list = list(incites_path.glob('*.xlsx'))
    dfs = [pd.read_excel(f, header=2) for f in file_list]
    df = pd.concat(dfs, ignore_index=True)
    db.sql('CREATE OR REPLACE TABLE incites AS SELECT * FROM df')
    print(f"Loaded {len(df)} rows from {len(file_list)} files into DuckDB table 'incites'.")
    return

def incites_econ_bus(db):
    print(f'{Path(assembled_file).exists() = }')
    sql = f"""
        SELECT DISTINCT d.eco_bus_incites, i."Journal name", i.issn AS issn_1, Category-- eco_bus_incites, d.issn, d.eissn, s.issn, id, display_name
        FROM read_xlsx('{assembled_file}', sheet='ALL') d
        LEFT JOIN econ_bus.incites i
        ON lower(eco_bus_incites) = lower("Journal name")
        WHERE list_contains(['ECONOMICS', 'MANAGEMENT', 'BUSINESS, FINANCE'], Category)
        ORDER BY eco_bus_incites
        """
    db.sql(sql).show()
    return

def main():
    with duckdb.connect('/home/lc/m/working/econ_bus.duckdb') as db:
        load_incites(db)
        incites_econ_bus(db)
    return

if __name__ == "__main__":
    start = time.time()
    main()
    print(f'finished at {time.time() - start = }')