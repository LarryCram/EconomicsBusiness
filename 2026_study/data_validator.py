from pathlib import Path
import pandas as pd

# validate sources
def validate_sources():
    source_file = './DATA/journals_institutions_from_dd.xlsx'
    print(f'{Path(source_file).exists() = }')
    original = pd.read_excel(source_file, work_sheet='journals')
    print(f'{original.shape = }')
    return
          

# validate institutions

# validate researchers

def main():
    validate_sources()


if __name__ == "__ main__":
    main()
    print("FINISHED !")