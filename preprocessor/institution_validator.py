from pathlib import Path
import pandas as pd

# Set pandas display options for wider output
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)       # Wider display width
pd.set_option('display.max_colwidth', None)  # Max width per column

DATA_PATH = '/home/lc/Projects/EconomicsBusiness/data/'
print(f'{Path(DATA_PATH).exists() = }')

# validate institutions
def validate_institutions():

# load Domingo institutions
    source_file = f'{DATA_PATH}/journals_institutions_from_dd.xlsx'
    original = pd.read_excel(source_file, sheet_name='institutions', skiprows=1)
    print(f'Domingo institutions: {len(original)} institutions ({original['institution'].nunique()} unique)')

# load matched list
    matched_file = f'{DATA_PATH}/institutions_matched_SAVE.csv'
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
        missing_from_domingo.to_csv(f'{DATA_PATH}/unverified_institutions.csv')
    return

def main():
    print("=== Validation Report ===")
    validate_institutions()
    print("=== Validation Complete ===")
    return

if __name__ == "__main__":
    main()