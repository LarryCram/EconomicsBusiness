import pandas as pd
import json

def make_model(verbose=False):
    """
    Creates toy model data with unnested authorships and institutions.
    
    Args:
        verbose (bool): If True, prints detailed information about the data
        
    Returns:
        pd.DataFrame: Unnested DataFrame with work, authorship, and institution data
    """
    model_json = """{"work_id": "W1", "journal_id": "J1", "publication_year": 1, "referenced_works": [], "authorships": [{"author_id": "A1", "institution_id": ["I1"]}, {"author_id": "A2", "institution_id": ["I2"]}]}
                    {"work_id": "W2", "journal_id": "J2", "publication_year": 1, "referenced_works": ["W1"], "authorships": [{"author_id": "A3", "institution_id": ["I3"]}]}
                    {"work_id": "W3", "journal_id": "J1", "publication_year": 2, "referenced_works": ["W1", "W2"], "authorships": [{"author_id": "A1", "institution_id": ["I1"]}]}
                    {"work_id": "W4", "journal_id": "J3", "publication_year": 2, "referenced_works": ["W1", "W3"], "authorships": [{"author_id": "A2", "institution_id": ["I2"]}, {"author_id": "A4", "institution_id": ["I1", "I3"]}]}
                    {"work_id": "W5", "journal_id": "J2", "publication_year": 2, "referenced_works": ["W2"], "authorships": [{"author_id": "A3", "institution_id": ["I3"]}]}
                    {"work_id": "W6", "journal_id": "J1", "publication_year": 3, "referenced_works": ["W3", "W4", "W5"], "authorships": [{"author_id": "A1", "institution_id": ["I1"]}, {"author_id": "A4", "institution_id": ["I1", "I3"]}]}
                    {"work_id": "W7", "journal_id": "J3", "publication_year": 3, "referenced_works": ["W4", "W6"], "authorships": [{"author_id": "A2", "institution_id": ["I2"]}]}
                    {"work_id": "W8", "journal_id": "J2", "publication_year": 3, "referenced_works": ["W1", "W5"], "authorships": [{"author_id": "A3", "institution_id": ["I3"]}, {"author_id": "A4", "institution_id": ["I1"]}]}"""
    
    # Parse JSON Lines format - each line is a separate JSON object
    json_objects = []
    for line in model_json.strip().split('\n'):
        if line.strip():  # Skip empty lines
            json_objects.append(json.loads(line.strip()))
    
    # Create DataFrame from the list of JSON objects
    model_df = pd.json_normalize(json_objects)
    
    # Unnest the authorships column
    # First explode the authorships list to create one row per authorship
    model_df_exploded = model_df.explode('authorships').reset_index(drop=True)
    
    # Then normalize the authorship dictionaries into separate columns
    authorships_normalized = pd.json_normalize(model_df_exploded['authorships'])
    
    # Combine the original columns (excluding authorships) with the normalized authorship data
    model_df_unnested = pd.concat([
        model_df_exploded.drop('authorships', axis=1).reset_index(drop=True),
        authorships_normalized.reset_index(drop=True)
    ], axis=1)
    
    # Further unnest institution_id if it contains lists
    if 'institution_id' in model_df_unnested.columns:
        model_df_unnested = model_df_unnested.explode('institution_id').reset_index(drop=True)
    
    if verbose:
        print(f'Original shape: {model_df.shape}')
        print(f'Unnested shape: {model_df_unnested.shape}')
        print(f'\nUnnested DataFrame:\n{model_df_unnested.head(10)}')
    
    return model_df_unnested 