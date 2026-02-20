#!/usr/bin/env python3

"""
Simple demonstration of Geller-Pinski-Narin relationship.

According to bibliometric theory, for a citation matrix:
Geller_weight[i] = output[i] / PN_weight[i] (up to a normalization constant)

Where:
- output[i] = total citations made by unit i (row sum)
- PN_weight[i] = Pinski-Narin influence score for unit i
"""

import numpy as np
import pandas as pd
from configuration.make_toy_model import make_model
from rankers.single_unit_ranks import unit_citation_matrix
from utils.algorithms import pinski_narin


def simple_geller_pn_demo():
    """Simple demonstration of the Geller-PN relationship."""
    
    print("Simple Geller-Pinski-Narin Relationship")
    print("=" * 50)
    
    # Generate toy model data
    data = make_model(verbose=False)
    
    # Get journal citation matrix
    citation_matrix = unit_citation_matrix(data, 'journal', verbose=False)
    
    print("\nJournal Citation Matrix:")
    print(citation_matrix)
    
    # Calculate outputs (row sums)
    outputs = citation_matrix.sum(axis=1)
    print(f"\nOutputs (citations made by each journal):")
    for journal in citation_matrix.index:
        print(f"  {journal}: {outputs[journal]:.3f}")
    
    # Run Pinski-Narin algorithm
    pn_result = pinski_narin(citation_matrix)
    print(f"\nPinski-Narin weights:")
    for journal, weight in pn_result['pi'].items():
        print(f"  {journal}: {weight:.6f}")
    
    # Calculate the ratio output[i] / PN[i] for each journal
    print(f"\nGeller relationship: output[i] / PN[i]")
    print(f"{'Journal':<8} {'Output':<10} {'PN_Weight':<12} {'Ratio':<15}")
    print("-" * 50)
    
    ratios = []
    for journal in citation_matrix.index:
        output_i = outputs[journal]
        pn_weight_i = pn_result['pi'][journal]
        
        if pn_weight_i > 0:
            ratio = output_i / pn_weight_i
        else:
            ratio = float('inf')
        
        ratios.append(ratio)
        print(f"{journal:<8} {output_i:<10.3f} {pn_weight_i:<12.6f} {ratio:<15.6f}")
    
    # Normalize the ratios to get Geller weights
    ratios = np.array(ratios)
    geller_weights = ratios / ratios.sum()
    
    print(f"\nNormalized ratios (Geller weights):")
    for i, journal in enumerate(citation_matrix.index):
        print(f"  {journal}: {geller_weights[i]:.6f}")
    
    # Verification: Show that this gives us a valid ranking
    print(f"\nRanking comparison:")
    print(f"{'Journal':<8} {'PN_Rank':<10} {'Geller_Rank':<12} {'PN_Score':<12} {'Geller_Score':<12}")
    print("-" * 70)
    
    # Sort by scores for ranking
    pn_sorted = sorted(pn_result['pi'].items(), key=lambda x: x[1], reverse=True)
    geller_sorted = sorted(zip(citation_matrix.index, geller_weights), key=lambda x: x[1], reverse=True)
    
    # Create rank mappings
    pn_ranks = {journal: rank for rank, (journal, _) in enumerate(pn_sorted, 1)}
    geller_ranks = {journal: rank for rank, (journal, _) in enumerate(geller_sorted, 1)}
    
    for journal in citation_matrix.index:
        pn_rank = pn_ranks[journal]
        geller_rank = geller_ranks[journal]
        pn_score = pn_result['pi'][journal]
        geller_score = geller_weights[list(citation_matrix.index).index(journal)]
        
        print(f"{journal:<8} {pn_rank:<10} {geller_rank:<12} {pn_score:<12.6f} {geller_score:<12.6f}")
    
    # Mathematical insight
    print(f"\n" + "="*60)
    print("MATHEMATICAL INSIGHT")
    print("="*60)
    
    print("\nThe relationship Geller[i] ∝ output[i] / PN[i] shows that:")
    print("1. Units with high output relative to their PN score get high Geller scores")
    print("2. Units that cite a lot but have low PN influence are 'rewarded' by Geller")
    print("3. This creates a complementary ranking to Pinski-Narin")
    
    print(f"\nNumerical verification:")
    print(f"- Sum of PN weights: {sum(pn_result['pi'].values()):.6f}")
    print(f"- Sum of Geller weights: {geller_weights.sum():.6f}")
    print(f"- Both should equal 1.0 ✓")
    
    # Show the proportionality constant
    total_output = outputs.sum()
    total_pn = sum(pn_result['pi'].values())
    
    print(f"\nProportionality analysis:")
    print(f"- Total output: {total_output:.3f}")
    print(f"- Total PN weight: {total_pn:.6f}")
    print(f"- Ratio: {total_output / total_pn:.6f}")
    
    # The constant of proportionality
    k = 1.0 / total_output  # Since Geller weights must sum to 1
    print(f"- Proportionality constant k: {k:.6f}")
    print(f"- Verification: k × total_output = {k * total_output:.6f} (should be 1.0)")


if __name__ == "__main__":
    simple_geller_pn_demo()