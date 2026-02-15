from pathlib import Path
import pandas as pd

# Set pandas display options for wider output
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)       # Wider display width
pd.set_option('display.max_colwidth', None)  # Max width per column

DATA_PATH = '/home/lc/Projects/EconomicsBusiness/data/'
print(f'{Path(DATA_PATH).exists() = }')

# validate sources
def validate_sources():
    source_file = f'{DATA_PATH}/journals_institutions_from_dd.xlsx'
    original = pd.read_excel(source_file, sheet_name='journals', skiprows=1)
    print(f'{original.shape = }\n{original.head()}')
    print(f'{original.journal.nunique() = }')
    
    matched_file = f'{DATA_PATH}/journals_to_sources_oa.csv'
    matched = pd.read_csv(matched_file)  #.drop_duplicates(['journal'])
    print(f'{matched.shape = }\n{matched.head()}')
    print(f'{matched.journal_name.nunique() = }')
    from collections import defaultdict
    dd = defaultdict(list)
    for row in matched.itertuples():
        dd[row.source_id].append(row.dd_name)
    print("Are there any duplicate entries for the dd list?")
    [print(k, len(v), v) for k, v in dd.items() if len(v) != 1]
    
    # Get unique journals and merge
    unique_journals = original[['journal']].drop_duplicates()
    print(f'{unique_journals.journal.nunique() = }')

    result = unique_journals.merge(matched[['source_id', 'incites_name', 'journal_name', 'ISSN']], 
                                  left_on=unique_journals['journal'].str.lower(), 
                                  right_on=matched['incites_name'].str.lower(), 
                                  how='left', suffixes=('', '_matched'))
    
    result['verified'] = ~result['source_id'].isna()
    result = result[['journal', 'source_id', 'journal_name', 'ISSN', 'verified']]
    
    print(f'Total: {len(result)}, Verified: {result.verified.sum()}, Missing: {(~result.verified).sum()}')
    result.to_csv(f'{DATA_PATH}/verified_sources.csv', index=False)
    print(f'{result.journal.nunique() = } {result.source_id.nunique() = }')

def main():
    print("=== Validation Report ===")
    validate_sources()
    print("=== Validation Complete ===")
    return


if __name__ == "__main__":
    main()