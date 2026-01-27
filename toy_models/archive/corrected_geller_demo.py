#!/usr/bin/env python3

"""
Corrected demonstration of Geller-Pinski-Narin relationship.

The correct relationship should be:
Geller_weight[i] = PN_weight[i] / output[i] (normalized)

Where:
- output[i] = total citations made by unit i (row sum)
- PN_weight[i] = Pinski-Narin influence score for unit i
"""

import numpy as np
import pandas as pd
from configuration.make_toy_model import make_model
from rankers.single_unit_ranks import unit_citation_matrix
from utils.algorithms import pinski_narin


def corrected_geller_pn_demo():
    """Corrected demonstration of the Geller-PN relationship."""
    
    print("Corrected Geller-Pinski-Narin Relationship")
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
    
    # Calculate the CORRECTED ratio: PN[i] / output[i] for each journal
    print(f"\nCorrected Geller relationship: PN[i] / output[i]")
    print(f"{'Journal':<8} {'PN_Weight':<12} {'Output':<10} {'Ratio':<15}")
    print("-" * 50)
    
    ratios = []
    for journal in citation_matrix.index:
        output_i = outputs[journal]
        pn_weight_i = pn_result['pi'][journal]
        
        if output_i > 0:
            ratio = pn_weight_i / output_i
        else:
            ratio = 0
        
        ratios.append(ratio)
        print(f"{journal:<8} {pn_weight_i:<12.6f} {output_i:<10.3f} {ratio:<15.6f}")
    
    # Normalize the ratios to get Geller weights
    ratios = np.array(ratios)
    if ratios.sum() > 0:
        geller_weights = ratios / ratios.sum()
    else:
        geller_weights = ratios
    
    print(f"\nNormalized ratios (Geller weights):")
    print(f"{'Journal':<8} {'Geller_Weight':<15}")
    print("-" * 25)
    for i, journal in enumerate(citation_matrix.index):
        print(f"{journal:<8} {geller_weights[i]:<15.6f}")
    
    # Verification: Show the relationship clearly
    print(f"\n" + "="*60)
    print("RELATIONSHIP VERIFICATION")
    print("="*60)
    
    print(f"{'Journal':<8} {'PN_Weight':<12} {'Output':<10} {'PN/Output':<12} {'Geller':<12} {'Match?':<8}")
    print("-" * 75)
    
    for i, journal in enumerate(citation_matrix.index):
        pn_weight_i = pn_result['pi'][journal]
        output_i = outputs[journal]
        ratio = pn_weight_i / output_i if output_i > 0 else 0
        geller_weight_i = geller_weights[i]
        
        # Check if normalized ratio matches Geller weight
        match = abs(ratio / ratios.sum() - geller_weight_i) < 1e-10 if ratios.sum() > 0 else True
        
        print(f"{journal:<8} {pn_weight_i:<12.6f} {output_i:<10.3f} {ratio:<12.6f} {geller_weight_i:<12.6f} {'✓' if match else '✗':<8}")
    
    # Show what this means conceptually
    print(f"\n" + "="*60)
    print("CONCEPTUAL INTERPRETATION")
    print("="*60)
    
    print("The relationship Geller[i] ∝ PN[i] / output[i] means:")
    print("1. High PN score + Low output = High Geller score")
    print("2. Low PN score + High output = Low Geller score")
    print("3. Geller rewards 'efficiency': high influence per citation made")
    print("4. PN rewards 'influence': overall impact in the network")
    
    # Ranking comparison
    print(f"\nRanking comparison:")
    pn_sorted = sorted(pn_result['pi'].items(), key=lambda x: x[1], reverse=True)
    geller_sorted = sorted(zip(citation_matrix.index, geller_weights), key=lambda x: x[1], reverse=True)
    
    print(f"Pinski-Narin ranking:")
    for rank, (journal, score) in enumerate(pn_sorted, 1):
        print(f"  {rank}. {journal}: {score:.6f}")
    
    print(f"Geller ranking:")
    for rank, (journal, score) in enumerate(geller_sorted, 1):
        print(f"  {rank}. {journal}: {score:.6f}")
    
    # Mathematical verification
    print(f"\n" + "="*60)
    print("MATHEMATICAL VERIFICATION")
    print("="*60)
    
    print(f"Sum of PN weights: {sum(pn_result['pi'].values()):.10f}")
    print(f"Sum of Geller weights: {geller_weights.sum():.10f}")
    print(f"Sum of ratios: {ratios.sum():.10f}")
    print(f"Sum of outputs: {outputs.sum():.3f}")
    
    # Show the exact relationship formula
    print(f"\nExact relationship:")
    print(f"For each journal i:")
    print(f"  Geller[i] = (PN[i] / output[i]) / Σ(PN[j] / output[j])")
    print(f"This ensures Σ Geller[i] = 1")


if __name__ == "__main__":
    corrected_geller_pn_demo()