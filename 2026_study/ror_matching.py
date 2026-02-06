import requests
import json
import pandas as pd
from typing import Dict, Optional, Union, List
import time
import duckdb
import requests
import json
from pyalex import config, Institutions

config.email = "Lawrence.Cram@anu.edu.au"
config.api_key = "OchtksdohLaziRq08C4IJP"

def read_incites():
    with duckdb.connect() as db:
        sql = """
            SELECT *
                FROM read_xlsx('/home/lc/Dropbox/ECONOMICS_BUSINESS/DATAFILES/data_eco_bus.xlsx', sheet='institutions')
                --LEFT JOIN '/home/lc/m/openalex_june25/parquet/institutions.parquet'
            """
        db.sql(sql).show()
        df = db.sql(sql).df()
    return df

def process_df(df):
    col_keep = []
    for text in ['index', 'inCites', 'substring', 'score', 'matching_type', 'chosen', 'established', 'organization.id', 'links', 'types']:
        for col in df.columns:
            if text in col:
                col_keep.append(col)
                break
    col_drop = [c for c in df.columns if c not in col_keep]
    print(f'{col_drop = }')
    df = df.drop(columns=col_drop)
    df.columns = [c.replace('.', '_') for c in df.columns]
    for row in df.itertuples():
        if row.score == 1 or row.matching_type == 'EXACT':
            df = df.iloc[[row.Index]]
            df.insert(0, 'selector', len(df))
            return df
    df.insert(0, 'selector', len(df))        
    return df

def match_openalex(df):
    sql = """
        SELECT index, selector, 
                inCites, id, ror, display_name, country_code, works_count, cited_by_count, 
                substring, score, matching_type, chosen, organization_established, 
                organization_id, organization_types
            FROM df d 
            LEFT JOIN '/home/lc/m/openalex_june25/parquet/institutions.parquet' 
            ON organization_id = ror
        """
    with duckdb.connect() as db:
        ddf = db.sql(sql).df().sort_values(['inCites', 'index']).reset_index(drop=True)
    return ddf

def match_institution_to_ror(institution_name, min_score=80):
    """Match institution name to ROR ID with fuzzy matching"""
    
    try:
        url = "https://api.ror.org/organizations?affiliation"
        
        response = requests.post(
            url,
            json={"affiliation_string": f'{requests.utils.quote(institution_name)}'},
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        url = f"https://api.ror.org/v2/organizations?affiliation={requests.utils.quote(institution_name)}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()      
        data = pd.json_normalize(response.json()['items'])
        data.insert(0, 'inCites', institution_name)
        data = process_df(data[:3].reset_index())
        print(f'{data.shape = }\n{data.head()}')
        return data
            
    except Exception as e:
        print(f'ERROR {e = }')
        return

def main():

    clarivate_org_names = read_incites()
    print(f'{clarivate_org_names.shape = }\n{clarivate_org_names.head()}')
    results = []
    for kount, test_affiliation in enumerate(clarivate_org_names.institution):
        # test_affiliation = "Dept. of Physics, Univ. of Oxford, Oxford OX1 3RH, UK"
        results.append(match_institution_to_ror(test_affiliation))
        print(f'Processed {kount = }')
        # if kount > 4:
        #     break
    df = pd.concat(results)
    print(f'{df.shape = }\n{df.info()}\n{df.head(99)}')
    df = match_openalex(df)
    print(f'{df.shape = }\n{df.info()}\n{df.head(99)}')
    df.to_csv('./2026_study/DATA/test_ror.csv')
    return

if __name__ == "__main__":
    start = time.time()
    main()
    print(f'FINISHED {time.time() - start = }')

