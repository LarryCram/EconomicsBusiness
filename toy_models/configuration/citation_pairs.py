import pandas as pd

def create_citer_cited_df(df):
    """
    Unpacks the referenced_works column to create a citer-cited dataframe using pandas explode.
    
    Args:
        df: DataFrame with columns work_id and referenced_works
        
    Returns:
        DataFrame with columns citer_id and cited_id
    """
    # Filter out rows with empty referenced_works lists
    df_with_refs = df[df['referenced_works'].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    
    # Explode the referenced_works column to create one row per citation
    exploded_df = df_with_refs.explode('referenced_works').reset_index(drop=True)
    
    # Rename columns to match expected output
    result_df = exploded_df.rename(columns={
        'work_id': 'citer_id',
        'referenced_works': 'cited_id'
    })[['citer_id', 'cited_id']]
    
    return result_df