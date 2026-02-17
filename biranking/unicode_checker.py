import glob
import json
import os

json_files = glob.glob('/home/lc/m/openalex_feb26/json/**/*.json', recursive=True)
print(f'Checking {len(json_files)} JSON files for unicode errors...')

for i, file in enumerate(json_files):
    if i % 10000 == 0:
        print(f'Processed {i}/{len(json_files)} files')
    
    try:
        with open(file, 'r', encoding='utf-8') as f:
            json.load(f)
    except UnicodeDecodeError as e:
        print(f'Unicode error in file: {file}')
        print(f'Error: {e}')
        
        # Read raw bytes around the error
        with open(file, 'rb') as f:
            f.seek(max(0, e.start - 50))
            context = f.read(100)
            print(f'Context around error: {context}')
        
        # Rename file
        new_name = file.replace('.json', '.json_')
        os.rename(file, new_name)
        print(f'Renamed to: {new_name}')
        break
    except:
        pass

print('Done')