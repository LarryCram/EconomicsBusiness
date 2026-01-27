#!/usr/bin/env python3

"""
Comprehensive Geller-Pinski-Narin relationship demonstration.

Shows the relationship Geller[i] = PN[i] / output[i] (normalized) for:
1. All single-unit cases: journals, authors, institutions
2. Multi-unit case with different weightings
"""

import numpy as np
import pandas as pd
from configuration.make_toy_model import make_model
from rankers.single_unit_ranks import unit_citation_matrix, unit_driver
from rankers.multi_unit_ranks import multi_unit_citation_matrix, multi_unit_driver
from utils.algorithms import pinski_narin


def calculate_geller_from_pn(citation_matrix, pn_result):
    """Calculate Geller weights from PN weights using the relationship."""
    
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
        'geller': {unit: geller_weights[i] for i, unit in enumerate(units)},
        'ratios': {unit: ratios[i] for i, unit in enumerate(units)},
        'outputs': {unit: outputs[unit] for unit in units}
    }


def demonstrate_single_unit_cases():
    """Demonstrate Geller-PN relationship for all single unit types."""
    
    print("SINGLE-UNIT GELLER-PINSKI-NARIN RELATIONSHIPS")
    print("=" * 60)
    
    # Generate toy model data
    data = make_model(verbose=False)
    
    for unit_type in ['journal', 'author', 'institution']:
        print(f"\n{unit_type.upper()} ANALYSIS")
        print("-" * 40)
        
        # Get citation matrix for this unit type
        citation_matrix = unit_citation_matrix(data, unit_type, verbose=False)
        
        # Run Pinski-Narin
        pn_result = pinski_narin(citation_matrix)
        
        # Calculate Geller using the relationship
        geller_result = calculate_geller_from_pn(citation_matrix, pn_result)
        
        print(f"\nCitation Matrix ({citation_matrix.shape[0]}x{citation_matrix.shape[1]}):")
        print(citation_matrix)
        
        print(f"\nRelationship: Geller[i] = PN[i] / output[i] (normalized)")
        print(f"{'Unit':<8} {'PN_Weight':<12} {'Output':<10} {'PN/Output':<12} {'Geller':<12}")
        print("-" * 65)
        
        for unit in citation_matrix.index:
            pn_weight = pn_result['pi'][unit]
            output = geller_result['outputs'][unit]
            ratio = geller_result['ratios'][unit]
            geller_weight = geller_result['geller'][unit]
            
            print(f"{unit:<8} {pn_weight:<12.6f} {output:<10.3f} {ratio:<12.6f} {geller_weight:<12.6f}")
        
        # Show rankings
        pn_sorted = sorted(pn_result['pi'].items(), key=lambda x: x[1], reverse=True)
        geller_sorted = sorted(geller_result['geller'].items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nRanking Comparison:")
        print(f"PN Ranking: {' > '.join([f'{unit}({score:.3f})' for unit, score in pn_sorted])}")
        print(f"Geller Ranking: {' > '.join([f'{unit}({score:.3f})' for unit, score in geller_sorted])}")
        
        # Verification
        pn_sum = sum(pn_result['pi'].values())
        geller_sum = sum(geller_result['geller'].values())
        print(f"Verification: PN sum = {pn_sum:.10f}, Geller sum = {geller_sum:.10f}")


def demonstrate_multi_unit_case():
    """Demonstrate Geller-PN relationship for multi-unit analysis."""
    
    print(f"\n\n{'='*60}")
    print("MULTI-UNIT GELLER-PINSKI-NARIN RELATIONSHIPS")
    print("=" * 60)
    
    # Generate toy model data
    data = make_model(verbose=False)
    
    # Test different weight scenarios
    weight_scenarios = [
        ("Equal weights", {'journal': 1/3, 'author': 1/3, 'institution': 1/3}),
        ("Journal-focused", {'journal': 0.6, 'author': 0.2, 'institution': 0.2}),
        ("Author-focused", {'journal': 0.1, 'author': 0.8, 'institution': 0.1}),
    ]
    
    for scenario_name, weights in weight_scenarios:
        print(f"\n{scenario_name.upper()}")
        print("-" * 40)
        print(f"Weights: J={weights['journal']:.1f}, A={weights['author']:.1f}, I={weights['institution']:.1f}")
        
        # Get multi-unit citation matrix
        multi_result = multi_unit_citation_matrix(data, weights=weights, verbose=False)
        citation_matrix = multi_result['matrix']
        
        # Run Pinski-Narin on multi-unit matrix
        pn_result = pinski_narin(citation_matrix)
        
        # Calculate Geller using the relationship
        geller_result = calculate_geller_from_pn(citation_matrix, pn_result)
        
        print(f"\nMulti-unit matrix ({citation_matrix.shape[0]}x{citation_matrix.shape[1]}):")
        print(f"Total citations: {citation_matrix.sum().sum():.3f}")
        
        print(f"\nRelationship: Geller[i] = PN[i] / output[i] (normalized)")
        print(f"{'Unit':<8} {'PN_Weight':<12} {'Output':<10} {'PN/Output':<12} {'Geller':<12}")
        print("-" * 65)
        
        # Group by unit type for cleaner display
        for unit_prefix in ['J_', 'A_', 'I_']:
            print(f"\n{unit_prefix[0]} units:")
            for unit in citation_matrix.index:
                if unit.startswith(unit_prefix):
                    pn_weight = pn_result['pi'][unit]
                    output = geller_result['outputs'][unit]
                    ratio = geller_result['ratios'][unit]
                    geller_weight = geller_result['geller'][unit]
                    
                    print(f"{unit:<8} {pn_weight:<12.6f} {output:<10.3f} {ratio:<12.6f} {geller_weight:<12.6f}")
        
        # Show top-ranked units for each algorithm
        pn_sorted = sorted(pn_result['pi'].items(), key=lambda x: x[1], reverse=True)
        geller_sorted = sorted(geller_result['geller'].items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 3 Rankings:")
        print(f"PN: {', '.join([f'{unit}({score:.3f})' for unit, score in pn_sorted[:3]])}")
        print(f"Geller: {', '.join([f'{unit}({score:.3f})' for unit, score in geller_sorted[:3]])}")
        
        # Verification
        pn_sum = sum(pn_result['pi'].values())
        geller_sum = sum(geller_result['geller'].values())
        print(f"Verification: PN sum = {pn_sum:.10f}, Geller sum = {geller_sum:.10f}")


def demonstrate_conceptual_interpretation():
    """Show what the Geller-PN relationship means conceptually."""
    
    print(f"\n\n{'='*60}")
    print("CONCEPTUAL INTERPRETATION")
    print("=" * 60)
    
    print("\nThe relationship Geller[i] = PN[i] / output[i] (normalized) reveals:")
    print()
    print("1. PINSKI-NARIN ALGORITHM:")
    print("   - Measures overall influence/prestige in the citation network")
    print("   - High PN score = frequently cited by other influential units")
    print("   - Focus: 'How much influence does this unit have?'")
    print()
    print("2. GELLER ALGORITHM:")
    print("   - Measures influence efficiency (influence per citation made)")
    print("   - High Geller score = high PN score relative to citing behavior")
    print("   - Focus: 'How efficiently does this unit use its citations?'")
    print()
    print("3. MATHEMATICAL RELATIONSHIP:")
    print("   - Geller[i] ∝ PN[i] / output[i]")
    print("   - Units with high PN but low output get high Geller scores")
    print("   - Units with low PN but high output get low Geller scores")
    print()
    print("4. PRACTICAL IMPLICATIONS:")
    print("   - PN rewards 'hub' behavior (citing many influential sources)")
    print("   - Geller rewards 'authority' behavior (being influential with few citations)")
    print("   - Different rankings reveal different aspects of network structure")
    print()
    print("5. BIBLIOMETRIC INTERPRETATION:")
    print("   - PN: 'Which journals/authors/institutions have the most impact?'")
    print("   - Geller: 'Which ones are most selective in their citing behavior?'")
    print("   - Complementary measures for comprehensive evaluation")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_single_unit_cases()
    demonstrate_multi_unit_case()
    demonstrate_conceptual_interpretation()