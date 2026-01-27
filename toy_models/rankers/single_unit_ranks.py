#!/usr/bin/env python3

from configuration.make_toy_model import make_model
from configuration.citation_pairs import create_citer_cited_df
from utils.algorithms import pinski_narin, geller, pagerank, analyze_matrix_properties, verify_pn_geller
import pandas as pd
import numpy as np


def unit_citation_matrix(df, unit, verbose=False):
    """
    Creates a citation matrix for a specific unit type with appropriate credit allocation.
    - For journals: full credit (no multi-journal works assumed)
    - For authors: fractional credit (1/ni * 1/nj for multi-author works)
    - For institutions: fractional credit (1/ni * 1/nj for multi-institution works)
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
    
    # Get all unique units
    all_units = sorted(df[unit_col].unique())
    n_units = len(all_units)
    
    # Create citation matrix
    citation_matrix = np.zeros((n_units, n_units))
    
    # Create unit to index mapping
    unit_to_idx = {unit_id: idx for idx, unit_id in enumerate(all_units)}
    
    if unit == 'journal':
        # For journals, use full credit (assuming no multi-journal works)
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
        
        # Fill the citation matrix with full credit
        for _, row in citer_cited_with_units.iterrows():
            citer_unit = row[f'citer_{unit}_id']
            cited_unit = row[f'cited_{unit}_id']
            
            citer_idx = unit_to_idx[citer_unit]
            cited_idx = unit_to_idx[cited_unit]
            
            citation_matrix[citer_idx, cited_idx] += 1
            
    else:
        # For authors and institutions, use fractional credit
        # Create work to unit mapping - get all units for each work
        work_to_units = {}
        for _, row in df.iterrows():
            work_id = row['work_id']
            unit_id = row[unit_col]
            
            if work_id not in work_to_units:
                work_to_units[work_id] = []
            if unit_id not in work_to_units[work_id]:
                work_to_units[work_id].append(unit_id)
        
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
        allocation_type = "full credit" if unit == 'journal' else "fractional credit"
        print(f"\nCitation matrix for {unit} ({allocation_type}):")
        print(citation_matrix)
        print(f"Total citations: {citation_matrix.sum().sum():.6f}")
    
    return citation_matrix


def unit_driver(df, rankers=['pinski_narin', 'geller', 'pagerank'], verbose=False):
    """
    Driver function to compute citation matrices and rankings for all units.
    """
    if verbose:
        print(f"\n{'='*80}")
        print("SINGLE-UNIT CITATION ANALYSIS")
        print(f"{'='*80}")
    
    results = {}
    
    for unit in ['journal', 'author', 'institution']:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing unit: {unit}")
            print(f"{'='*60}")
        
        unit_results = {}
        citation_matrix = unit_citation_matrix(df, unit, verbose=verbose)
        unit_results['matrix'] = citation_matrix
        
        # Analyze matrix properties
        matrix_analysis = analyze_matrix_properties(citation_matrix, verbose=verbose)
        unit_results['matrix_analysis'] = matrix_analysis
        
        # Apply ranking algorithms
        unit_results['rankings'] = {}
        unit_results['check_geller'] = {}
        
        for ranker in rankers:
            if verbose:
                print(f"\n{'-'*60}")
                print(f"Computing {ranker} rankings for {unit}")
                print(f"{'-'*60}")
            
            if ranker == 'pinski_narin':
                ranking_result = pinski_narin(citation_matrix)
            elif ranker == 'geller':
                ranking_result = geller(citation_matrix)
            elif ranker == 'pagerank':
                ranking_result = pagerank(citation_matrix, alpha=1.0)
            else:
                print(f"Unknown ranker: {ranker}")
                continue
            
            unit_results['rankings'][ranker] = ranking_result
            print(f" ranking = {ranking_result} {sum(ranking_result) = }")
            print(" for citation matrix ")
            print(unit_results['matrix'])
            
        
        # Using the Geller conversion from PN to Markov/pagerank
        print(f"\n{'-'*60}")
        print(f"Comparing Pinski-Narin and Geller rankings for {unit.upper()}")
        print(f"{'-'*60}")

        verify_pn_geller(unit_results['matrix'], unit_results['rankings']['pinski_narin'], unit_results['rankings']['geller'])

        # check_geller = [float(x)/float(y) for x, y in zip(unit_results['rankings']['pinski_narin'], unit_results['matrix_analysis']['row_sums'])]
        # norm = sum(check_geller)
        # check_geller = [(x/norm)/y for x, y  in zip(check_geller, unit_results['rankings']['pinski_narin'])]
        # unit_results['check_geller']['geller'] = check_geller
        # print(f"    Check Geller relation for PN/Markov {check_geller}")

        results[unit] = unit_results
    
    return results
