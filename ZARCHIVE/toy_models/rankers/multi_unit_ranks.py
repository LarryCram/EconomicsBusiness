import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'configuration'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from citation_pairs import create_citer_cited_df
from utils.algorithms import pinski_narin, geller, pagerank, analyze_matrix_properties, verify_pn_geller


def calculate_geller_from_pn(citation_matrix, pn_result):
    """Calculate Geller weights from PN weights using the relationship Geller[i] = PN[i] / output[i]."""
    import numpy as np
    
    # Calculate outputs (row sums)
    outputs = citation_matrix.sum(axis=1)
    
    # Calculate ratios: PN[i] / output[i]
    ratios = []
    units = []
    
    for unit in citation_matrix.index:
        output_i = outputs[unit]
        pn_weight_i = pn_result['pi'][unit]
        
        if output_i > 0:
            ratio = pn_weight_i / output_i
        else:
            ratio = 0
        
        ratios.append(ratio)
        units.append(unit)
    
    # Normalize ratios to get Geller weights
    ratios = np.array(ratios)
    if ratios.sum() > 0:
        geller_weights = ratios / ratios.sum()
    else:
        geller_weights = ratios
    
    return {
        'pi': {unit: geller_weights[i] for i, unit in enumerate(units)},
        'ratios': {unit: ratios[i] for i, unit in enumerate(units)},
        'outputs': {unit: outputs[unit] for unit in units}
    }


def multi_unit_citation_matrix(df, weights=None, verbose=False):
    """
    Creates a comprehensive citation matrix with all units (journals, authors, institutions) combined.
    
    Args:
        df: DataFrame with unnested data containing work_id, journal_id, author_id, institution_id, and referenced_works
        weights (dict): Weights for different citation types. Must sum to 1.0. 
                       Default: {'journal': 1/3, 'author': 1/3, 'institution': 1/3}
        verbose (bool): If True, prints detailed information
        
    Returns:
        dict: {
            'matrix': Combined citation matrix with all units,
            'unit_mapping': Mapping of matrix indices to unit types and IDs,
            'unit_types': List indicating the type of each unit,
            'weights': The weights used for citation types
        }
    """
    
    # Set default weights if not provided
    if weights is None:
        weights = {'journal': 1/3, 'author': 1/3, 'institution': 1/3}
    
    # Validate weights
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-10:
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
    
    required_types = {'journal', 'author', 'institution'}
    if set(weights.keys()) != required_types:
        raise ValueError(f"Weights must contain exactly {required_types}, got {set(weights.keys())}")
    
    if verbose:
        print(f"Citation type weights:")
        for unit_type, weight in weights.items():
            print(f"  {unit_type}: {weight:.3f}")
        print()
    
    # Get unique units for each type
    journals = sorted(df['journal_id'].unique())
    authors = sorted(df['author_id'].unique())
    institutions = sorted(df['institution_id'].unique())
    
    # Create combined unit list with prefixes for clarity
    all_units = []
    unit_types = []
    unit_mapping = {}
    
    # Add journals
    for j in journals:
        unit_id = f"J_{j}"
        all_units.append(unit_id)
        unit_types.append('journal')
        unit_mapping[unit_id] = {'type': 'journal', 'id': j, 'original_id': j}
    
    # Add authors
    for a in authors:
        unit_id = f"A_{a}"
        all_units.append(unit_id)
        unit_types.append('author')
        unit_mapping[unit_id] = {'type': 'author', 'id': a, 'original_id': a}
    
    # Add institutions
    for i in institutions:
        unit_id = f"I_{i}"
        all_units.append(unit_id)
        unit_types.append('institution')
        unit_mapping[unit_id] = {'type': 'institution', 'id': i, 'original_id': i}
    
    n_units = len(all_units)
    
    if verbose:
        print(f"Creating multi-unit citation matrix:")
        print(f"  Journals: {len(journals)} ({journals})")
        print(f"  Authors: {len(authors)} ({authors})")
        print(f"  Institutions: {len(institutions)} ({institutions})")
        print(f"  Total units: {n_units}")
    
    # Initialize citation matrix
    citation_matrix = np.zeros((n_units, n_units))
    
    # Create citer-cited pairs
    original_df = df[['work_id', 'referenced_works']].drop_duplicates(subset=['work_id'])
    citer_cited_df = create_citer_cited_df(original_df)
    
    if len(citer_cited_df) == 0:
        if verbose:
            print("No citations found in the dataset")
        return {
            'matrix': pd.DataFrame(citation_matrix, index=all_units, columns=all_units),
            'unit_mapping': unit_mapping,
            'unit_types': unit_types
        }
    
    # Create work to units mapping - handle multiple authors/institutions per work
    work_to_units = {}
    for _, row in df.iterrows():
        work_id = row['work_id']
        if work_id not in work_to_units:
            work_to_units[work_id] = {
                'journal': [f"J_{row['journal_id']}"],
                'author': [f"A_{row['author_id']}"],
                'institution': [f"I_{row['institution_id']}"]
            }
        else:
            # Handle multiple authors/institutions for the same work
            journal_unit = f"J_{row['journal_id']}"
            author_unit = f"A_{row['author_id']}"
            institution_unit = f"I_{row['institution_id']}"
            
            if journal_unit not in work_to_units[work_id]['journal']:
                work_to_units[work_id]['journal'].append(journal_unit)
            if author_unit not in work_to_units[work_id]['author']:
                work_to_units[work_id]['author'].append(author_unit)
            if institution_unit not in work_to_units[work_id]['institution']:
                work_to_units[work_id]['institution'].append(institution_unit)
    
    # Create unit to index mapping
    unit_to_idx = {unit: idx for idx, unit in enumerate(all_units)}
    
    # Process each citation relationship with weighting
    for _, row in citer_cited_df.iterrows():
        citer_work = row['citer_id']
        cited_work = row['cited_id']
        
        # Get all units for citing work
        if citer_work in work_to_units:
            citer_units = work_to_units[citer_work]
        else:
            continue
            
        # Get all units for cited work
        if cited_work in work_to_units:
            cited_units = work_to_units[cited_work]
        else:
            continue
        
        # Add citations between all unit combinations with appropriate weights and fractional credit
        for unit_type in ['journal', 'author', 'institution']:
            # Get weight for this citation type
            weight = weights[unit_type]
            
            # Get all citer units of this type
            citer_units_for_type = citer_units[unit_type]
            
            # Get all cited units of this type  
            cited_units_for_type = cited_units[unit_type]
            
            # Calculate fractional credit based on unit type
            if unit_type == 'journal':
                # For journals: full credit (assuming no multi-journal works)
                fractional_credit = weight
            else:
                # For authors and institutions: fractional credit
                # Credit = weight * (1/ni) * (1/nj) where ni = citing units, nj = cited units
                ni = len(citer_units_for_type)
                nj = len(cited_units_for_type)
                fractional_credit = weight * (1.0 / ni) * (1.0 / nj)
            
            # Create weighted and fractionated citations from all citer units to all cited units of this type
            for citer_unit in citer_units_for_type:
                for cited_unit in cited_units_for_type:
                    if citer_unit in unit_to_idx and cited_unit in unit_to_idx:
                        citer_idx = unit_to_idx[citer_unit]
                        cited_idx = unit_to_idx[cited_unit]
                        citation_matrix[citer_idx, cited_idx] += fractional_credit
    
    # Convert to DataFrame
    citation_df = pd.DataFrame(
        citation_matrix,
        index=all_units,
        columns=all_units
    )
    
    if verbose:
        print(f"\nMulti-unit citation matrix ({n_units}x{n_units}):")
        print(citation_df)
        print(f"\nTotal citations: {citation_matrix.sum()}")
        
        # Show citations by type (weighted)
        print(f"\nWeighted citations by unit type:")
        for unit_type in ['journal', 'author', 'institution']:
            type_units = [u for u, t in zip(all_units, unit_types) if t == unit_type]
            type_indices = [unit_to_idx[u] for u in type_units]
            type_citations = citation_matrix[np.ix_(type_indices, type_indices)].sum()
            print(f"  {unit_type} to {unit_type}: {type_citations:.3f} (weight: {weights[unit_type]:.3f})")
    
    return {
        'matrix': citation_df,
        'unit_mapping': unit_mapping,
        'unit_types': unit_types,
        'weights': weights
    }


def multi_unit_driver(df, weights=None, rankers=['pinski_narin', 'geller', 'pagerank', ], verbose=False):
    """
    Driver function to compute rankings for the combined multi-unit citation matrix.
    
    Args:
        df: DataFrame with unnested data
        weights (dict): Weights for different citation types. Must sum to 1.0.
                       Default: {'journal': 1/3, 'author': 1/3, 'institution': 1/3}
        rankers: List of ranking algorithms to apply
        verbose (bool): If True, prints detailed processing information
        
    Returns:
        dict: Results including matrix, weights, and rankings
    """
    from utils.algorithms import pinski_narin, geller, pagerank, analyze_matrix_properties
    import numpy as pd
    
    if verbose:
        print(f"\n{'='*80}")
        print("MULTI-UNIT CITATION ANALYSIS")
        print(f"{'='*80}")
    
    # Create multi-unit citation matrix
    multi_result = multi_unit_citation_matrix(df, weights=weights, verbose=verbose)
    citation_matrix = multi_result['matrix']
    unit_mapping = multi_result['unit_mapping']
    unit_types = multi_result['unit_types']
    weights_used = multi_result['weights']
    
    # Analyze matrix properties
    if verbose:
        matrix_analysis = analyze_matrix_properties(citation_matrix, verbose=True)
    
    results = {
        'matrix': citation_matrix,
        'citation_matrix': citation_matrix,  # Also store under this key for network visualization
        'unit_mapping': unit_mapping,
        'unit_types': unit_types,
        'weights': weights_used,
        'rankings': {}
    }
    
    # Apply ranking algorithms
    for ranker in rankers:
        if verbose:
            print(f"\n{'-'*60}")
            print(f"Computing {ranker} rankings")
            print(f"{'-'*60}")
        
        if ranker == 'pinski_narin':
            ranking_result = pinski_narin(citation_matrix)
        elif ranker == 'geller':
            ranking_result = geller(citation_matrix)
        elif ranker == 'pagerank':
            ranking_result = pagerank(citation_matrix, alpha=0.85)
        else:
            print(f"Unknown ranker: {ranker}")
            continue
        
        results['rankings'][ranker] = ranking_result
        print(f" ranking = {ranking_result} {sum(ranking_result) = }")
        print(" for citation matrix ")
        print(results['matrix'])
        
    if verbose:
        # Using the Geller conversion from PN to Markov/pagerank
        print(f"\n{'-'*60}")
        print(f"Comparing Pinski-Narin and Geller rankings")
        print(f"{'-'*60}")

        verify_pn_geller(results['matrix'], results['rankings']['pinski_narin'], results['rankings']['geller'])
    
    return results


if __name__ == "__main__":
    # Test the multi-unit functionality
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from configuration.make_toy_model import make_model
    from utils.algorithms import verify_pn_geller
    
    print("Testing Multi-Unit Citation Analysis")
    print("="*50)
    
    data = make_model(verbose=False)
    results = multi_unit_driver(data, verbose=True)
    print(results['matrix'])
    df_list = results['matrix'].values.tolist()
    print(df_list)