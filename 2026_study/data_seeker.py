from pathlib import Path
import pandas as pd
from pyalex import config, Authors, autocomplete

config.api_key = "OchtksdohLaziRq08C4IJP"

DATA_PATH = '/home/lc/Projects/EconomicsBusiness/2026_study/DATA/'
print(f'{Path(DATA_PATH).exists() = }')

def load_researchers():
# load Domingo researchers
    source_file = f'{DATA_PATH}/researchers_results.xlsx'
    original = pd.read_excel(source_file, sheet_name='average_influence', skiprows=0).iloc[:, 0:10]
    print(f'Domingo researchers: {len(original)} researchers ({original['Research_Profile'].nunique()} unique)')
    return original

def seek_researchers(df):

    # df = df[:4]
    id, display_name, works_count = [], [], []
    for row in df.itertuples():
        # print(row)
        flip = ' '.join(reversed(row.Research_Profile.lower().split(',')))
        response = Authors().autocomplete(flip)
        if response:
            authors = pd.DataFrame(response).sort_values('works_count', ascending=False).reset_index(drop=True)
            print(authors.iloc[:, 0:4].head(1))
            id.append(authors.at[0, 'id'])
            display_name.append(authors.at[0, 'display_name'])
            works_count.append(authors.at[0, 'works_count'])
        else:
            id.append('')
            display_name.append('')
            works_count.append(0)

    df.insert(1, 'author_id', id)
    df.insert(2, 'display_name', display_name)
    df.insert(2, 'works_count', works_count)
    print(f'{df.shape = }\n{df.head(16)}')
    return df

def main():
    print("=== Seeker Report ===")
    df = load_researchers()
    adjusted = seek_researchers(df)
    adjusted.to_csv(f'{DATA_PATH}.adjusted_researchers_results.csv')
    print("=== Seeker Complete ===")
    return


if __name__ == "__main__":
    main()