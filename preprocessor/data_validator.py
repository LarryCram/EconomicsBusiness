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
    
    matched_file = f'{DATA_PATH}/econ_bus_journal_oa.csv'
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
    return

# validate institutions
def validate_institutions():

# load Domingo institutions
    source_file = f'{DATA_PATH}/journals_institutions_from_dd.xlsx'
    original = pd.read_excel(source_file, sheet_name='institutions', skiprows=1)
    print(f'Domingo institutions: {len(original)} institutions ({original['institution'].nunique()} unique)')

# load matched list
    matched_file = f'{DATA_PATH}/econ_bus_ror_oa_SAVE.csv'
    matched = pd.read_csv(matched_file).drop_duplicates(['inCites'])
    print(f'Matched institutions: {len(matched)} institutions ({matched['inCites'].nunique()} unique)')

# Find institutions from Domingo's list that are not in the matched list
    domingo_institutions_lower = set(original.institution.str.lower())
    matched_institutions_lower = set(matched.inCites.str.lower())
    
    successfully_matched = len(domingo_institutions_lower & matched_institutions_lower)
    missing_from_domingo = original[~original.institution.str.lower().isin(matched.inCites.str.lower())]
    
    print(f'Successfully matched from Domingo\'s list: {successfully_matched}')
    print(f'Missing from Domingo\'s list: {len(missing_from_domingo)}')
    
    if len(missing_from_domingo) > 0:
        print('\nUnmatched institutions:')
        print(f'{missing_from_domingo.shape = }\n{missing_from_domingo['institution'].head(32)}')
        missing_from_domingo.to_excel('/home/lc/Projects/EconomicsBusiness/2026_study/DATA/unverified_institutions.xlsx')
    return


# validate researchers
def validate_researchers():

# load Domingo researchers
    source_file = f'{DATA_PATH}/researchers_results.xlsx'
    original = pd.read_excel(source_file, sheet_name='average_influence', skiprows=0)
    print(f'Domingo researchers: {len(original)} researchers ({original['Research_Profile'].nunique()} unique)')

# load matched list
    matched_file = f'{DATA_PATH}/matched_authors.csv'
    matched = pd.read_csv(matched_file).drop_duplicates(['NAME'])
    print(f'Matched authors: {len(matched)} authors ({matched['NAME'].nunique()} unique)')


# Find researchers from Domingo's list that are not in the matched list
    domingo_researchers_lower = set(original.Research_Profile.str.lower())
    matched_researchers_lower = set(matched.NAME.str.lower())
    
    successfully_matched = len(domingo_researchers_lower & matched_researchers_lower)
    missing_from_domingo = original[~original.Research_Profile.str.lower().isin(matched.NAME.str.lower())]
    
    print(f'Successfully matched from Domingo\'s list: {successfully_matched}')
    print(f'Missing from Domingo\'s list: {len(missing_from_domingo)}')
    
    if len(missing_from_domingo) > 0:
        print('\nUnmatched researchers:')
        print(f'{missing_from_domingo.shape = }\n{missing_from_domingo['Research_Profile'].head(32)}')
        missing_from_domingo.to_excel('/home/lc/Projects/EconomicsBusiness/2026_study/DATA/unverified_researchers.xlsx')


def main():
    print("=== Validation Report ===")
    validate_sources()
    # validate_institutions()
    # validate_researchers()
    print("=== Validation Complete ===")
    return


if __name__ == "__main__":
    main()