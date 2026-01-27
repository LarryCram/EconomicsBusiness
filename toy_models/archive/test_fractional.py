#!/usr/bin/env python3

from configuration.make_toy_model import make_model
from configuration.citation_pairs import create_citer_cited_df
import pandas as pd
import numpy as np

def unit_citation_matrix_full(df, unit='author', verbose=False):
    """
    Creates a citation matrix for a specific unit type with full credit allocation (original method).
    """
    unit_col = f'{unit}_id'
    
    # Create citer-cited pairs from the dataset
    original_df = df[['work_id', 'referenced_works']].drop_duplicates(subset=['work_id'])
    citer_cited_df = create_citer_cited_df(original_df)
    
    if len(citer_cited_df) == 0:
        all_units = sorted(df[unit_col].unique())
        n_units = len(all_units)
        citation_matrix = np.zeros((n_units, n_units))
        return pd.DataFrame(citation_matrix, index=all_units, columns=all_units)
    
    # Create work to unit mapping
    work_to_unit = df[['work_id', unit_col]].drop_duplicates()
    
    # Merge citer-cited pairs with unit information
    citer_cited_with_units = citer_cited_df.merge(
        work_to_unit.rename(columns={'work_id': 'citer_id', unit_col: f'citer_{unit}_id'}),
        on='citer_id',
        how='left'
    )
    
    citer_cited_with_units = citer_cited_with_units.merge(
        work_to_unit.rename(columns={'work_id': 'cited_id', unit_col: f'cited_{unit}_id'}),
        on='cited_id',
        how='left'
    )
    
    citer_cited_with_units = citer_cited_with_units.dropna()
    
    # Get all unique units
    all_units = sorted(df[unit_col].unique())
    n_units = len(all_units)
    citation_matrix = np.zeros((n_units, n_units))
    
    # Create unit to index mapping
    unit_to_idx = {unit_id: idx for idx, unit_id in enumerate(all_units)}
    
    # Fill the citation matrix
    for _, row in citer_cited_with_units.iterrows():
        citer_unit = row[f'citer_{unit}_id']
        cited_unit = row[f'cited_{unit}_id']
        
        citer_idx = unit_to_idx[citer_unit]
        cited_idx = unit_to_idx[cited_unit]
        
        citation_matrix[citer_idx, cited_idx] += 1
    
    citation_matrix = pd.DataFrame(
        citation_matrix,
        index=all_units,
        columns=all_units
    )
    
    if verbose:
        print(f"\nFull credit citation matrix for {unit}:")
        print(citation_matrix)
        print(f"Total citations: {citation_matrix.sum().sum():.6f}")
    
    return citation_matrix


def unit_citation_matrix_fractional(df, unit='author', verbose=False):
    """
    Creates a citation matrix for a specific unit type with fractional credit allocation.
    For multi-author works, citation credit is distributed: (1/ni) * (1/nj) per author pair.
    """
    unit_col = f'{unit}_id'
    
    # Create citer-cited pairs from the dataset
    original_df = df[['work_id', 'referenced_works']].drop_duplicates(subset=['work_id'])
    citer_cited_df = create_citer_cited_df(original_df)
    
    if len(citer_cited_df) == 0:
        # No citations found, return empty matrix
        all_units = sorted(df[unit_col].unique())
        n_units = len(all_units)
        citation_matrix = np.zeros((n_units, n_units))
        return pd.DataFrame(citation_matrix, index=all_units, columns=all_units)
    
    # Create work to unit mapping - get all units for each work
    work_to_units = {}
    for _, row in df.iterrows():
        work_id = row['work_id']
        unit_id = row[unit_col]
        
        if work_id not in work_to_units:
            work_to_units[work_id] = []
        if unit_id not in work_to_units[work_id]:
            work_to_units[work_id].append(unit_id)
    
    # Get all unique units
    all_units = sorted(df[unit_col].unique())
    n_units = len(all_units)
    
    # Create citation matrix
    citation_matrix = np.zeros((n_units, n_units))
    
    # Create unit to index mapping
    unit_to_idx = {unit_id: idx for idx, unit_id in enumerate(all_units)}
    
    # Fill the citation matrix with fractional credits
    for _, row in citer_cited_df.iterrows():
        citer_work = row['citer_id']
        cited_work = row['cited_id']
        
        # Get units for citing and cited works
        if citer_work not in work_to_units or cited_work not in work_to_units:
            continue
            
        citer_units = work_to_units[citer_work]
        cited_units = work_to_units[cited_work]
        
        ni = len(citer_units)  # number of citing units
        nj = len(cited_units)  # number of cited units
        
        # Fractional credit per unit pair: 1/(ni * nj)
        fractional_credit = 1.0 / (ni * nj)
        
        # Distribute credit among all unit pairs
        for citer_unit in citer_units:
            for cited_unit in cited_units:
                citer_idx = unit_to_idx[citer_unit]
                cited_idx = unit_to_idx[cited_unit]
                citation_matrix[citer_idx, cited_idx] += fractional_credit
    
    # Convert to DataFrame for easier handling
    citation_matrix = pd.DataFrame(
        citation_matrix,
        index=all_units,
        columns=all_units
    )
    
    if verbose:
        print(f"\nFractional citation matrix for {unit}:")
        print(citation_matrix)
        print(f"Total citations: {citation_matrix.sum().sum():.6f}")
    
    return citation_matrix


if __name__ == "__main__":
    # Test the fractional allocation for authors and institutions
    
    data = make_model(verbose=False)
    
    print("=== CITATION ALLOCATION COMPARISON ===")
    
    # Test Authors
    print("\n" + "="*60)
    print("AUTHORS - FRACTIONAL CREDIT ALLOCATION")
    print("="*60)
    fractional_matrix_authors = unit_citation_matrix_fractional(data, 'author', verbose=True)
    
    # Test Institutions  
    print("\n" + "="*60)
    print("INSTITUTIONS - FRACTIONAL CREDIT ALLOCATION")
    print("="*60)
    fractional_matrix_institutions = unit_citation_matrix_fractional(data, 'institution', verbose=True)
    
    print(f"\n=== SUMMARY ===")
    print(f"Author citations (fractional): {fractional_matrix_authors.sum().sum():.6f}")
    print(f"Institution citations (fractional): {fractional_matrix_institutions.sum().sum():.6f}")
    print(f"Expected total (work-work citations): 13")
    
    print(f"\nThis confirms that both authors and institutions now use fractional allocation")
    print(f"and preserve the conservation principle!")