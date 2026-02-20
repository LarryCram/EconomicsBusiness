from pathlib import Path
import pandas as pd
import duckdb
import requests
import json

DATA_PATH = Path('/home/lc/Projects/EconomicsBusiness/data')
print(f'{DATA_PATH.exists() = }')

def load_incites():
    # Load Domingos's institution data
    with duckdb.connect() as db:
        sql = f"""
            SELECT *
                FROM read_xlsx('{DATA_PATH}/ecobus_journal_institution_results.xlsx', sheet='eco_bus_institutions')
            """
        db.sql(sql).show()
        df = db.sql(sql).df()
    return df

def process_ror(df):
    col_keep = []
    for text in ['index', 'inCites', 'substring', 'score', 'matching_type', 'chosen', 'established', 'organization.id', 'links', 'types']:
        for col in df.columns:
            if text in col:
                col_keep.append(col)
                break
    col_drop = [c for c in df.columns if c not in col_keep]
    df = df.drop(columns=col_drop)
    df.columns = [c.replace('.', '_') for c in df.columns]
    for row in df.itertuples():
        if row.score == 1 or row.matching_type == 'EXACT':
            df = df.iloc[[row.Index]]
            df.insert(0, 'selector', len(df))
            return df
    df.insert(0, 'selector', len(df))        
    return df

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
        ror = pd.json_normalize(response.json()['items'])
        ror.insert(0, 'inCites', institution_name)
        ror = process_ror(ror[:3].reset_index())
        print(f'{ror.shape = }\n{ror.loc[:,['inCites', 'organization_id', 'substring', 'score']].head()}')
        return ror
            
    except Exception as e:
        print(f'ERROR {e = }')
        return

def match_openalex(df):
    sql = """
        SELECT index, selector, 
                inCites, id, ror, display_name, country_code, works_count, cited_by_count, 
                substring, score, matching_type, chosen, organization_established, 
                organization_id, organization_types
            FROM df d 
            LEFT JOIN '/home/lc/m/openalex_feb26/parquet/institutions.parquet' 
            ON organization_id = ror
        """
    with duckdb.connect() as db:
        matched = db.sql(sql).df().sort_values(['inCites', 'index']).reset_index(drop=True)
    matched.to_csv(f'{DATA_PATH}/institution_matched.csv')
    return matched

def main():

    print("=== Validation Report ===")
    incites = load_incites()
    print("Load inCites institutions")
    print(f'{incites.shape = }\n{incites.head()}')
    results = []
    for kount, test_affiliation in enumerate(incites.institution):
        # test_affiliation = "Dept. of Physics, Univ. of Oxford, Oxford OX1 3RH, UK"
        results.append(match_institution_to_ror(test_affiliation))
        print(f'Processed {kount = }')
        # if kount > 4:
        #     break
    incites_to_ror = pd.concat(results)
    print("Find ror for inCites names")
    print(f'{incites_to_ror.shape = }\n{incites_to_ror.head()}')
    incites_to_oa = match_openalex(incites_to_ror)
    print(f'{incites_to_oa.shape = }\n{incites_to_oa.head()}')
    print("=== Validation Complete ===")
    return

if __name__ == "__main__":
    main()