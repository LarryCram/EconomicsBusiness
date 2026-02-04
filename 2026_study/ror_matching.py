import requests
import json
from typing import Dict, Optional, Union, List
import time
import duckdb

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

import requests
import json

def match_institution_to_ror(institution_name, min_score=80):
    """Match institution name to ROR ID with fuzzy matching"""
    
    try:
        # Use GET method which is more reliable
        # url = f"https://api.ror.org/organizations/affiliation/?query={requests.utils.quote(institution_name)}"
        
        # response = requests.get(url, timeout=10)

        url = "https://api.ror.org/organizations?affiliation"
        
        response = requests.post(
            url,
            json={"affiliation_string": f'{requests.utils.quote(institution_name)}'},
            headers={"Content-Type": "application/json"},
            timeout=10
        )

# curl 'https://api.ror.org/v2/organizations?affiliation=Arizona State University' | json_pp
        url = f"https://api.ror.org/v2/organizations?affiliation={requests.utils.quote(institution_name)}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        print(f'{response = }')
        
        data = response.json()
        # print(f'{data = }')
        
        if data.get('items'):
            best_match = data['items'][0]

            score = best_match.get('score', 0)
            print(f'{score = }')
            print(f'{best_match = }')
            
            if score >= min_score:
                org = best_match.get('organization', best_match)
                return {
                    'institution': institution_name,
                    'ror_id': org.get('id'),
                    'ror_name': org.get('name'),
                    'match_score': score,
                    'country': org.get('country', {}).get('country_name'),
                    'confidence': 'high' if score > 90 else 'medium' if score > 80 else 'low'
                }
        
        return {'institution': institution_name, 'ror_id': None, 'error': 'No good match found'}
        
    except Exception as e:
        response.raise_for_status()
        print(f'{response = }')
        return {'institution': institution_name, 'ror_id': None, 'error': str(e)}

def main():

# Example usage
    clarivate_org_names = read_incites()
    print(f'{clarivate_org_names.shape = }\n{clarivate_org_names.head()}')
    for test_affiliation in clarivate_org_names.institution:
        # test_affiliation = "Dept. of Physics, Univ. of Oxford, Oxford OX1 3RH, UK"
        result = match_institution_to_ror(test_affiliation)
        
        if result and result.get("matched_organization"):
            org = result["matched_organization"]
            print(f"Matched: {org.get('name')}")
            print(f"ROR ID: {org.get('id')}")
            print(f"Score: {org.get('score')}")
        else:
            print("No match found or error occurred")
        break
    
    # # Batch processing example
    # sample_affiliations = [
    #     "Department of Computer Science, Stanford University, CA 94305, USA",
    #     "MIT, Cambridge, Massachusetts 02139, USA",
    #     "CNRS, Paris, France",
    #     "Harvard Medical School, Boston, MA 02115, USA",
    #     "Max Planck Institute for Informatics, Saarbrücken, Germany"
    # ]
    
    # # Uncomment to run batch processing
    # # batch_results = batch_match_affiliations(sample_affiliations, batch_delay=0.5)
    # # save_results_to_csv(batch_results, "ror_matches.csv")

if __name__ == "__main__":
    start = time.time()
    main()
    print(f'FINISHED {time.time() - start = }')

