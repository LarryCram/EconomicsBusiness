from pathlib import Path
import shutil

base = '/home/lc/MyLibrary/files'
files = Path(base).glob("**/*.pdf")
# print(list(files))
for source_file in files:
    destination_file = Path(f'/home/lc/PDFS/{source_file.name}')
    print(f'{source_file = } {destination_file = }')
    shutil.copy(source_file, destination_file)

