#!/usr/bin/env python3

"""
Demonstration of citation conservation between single-unit and multi-unit analyses.

This script shows that the total number of citations is properly conserved
between single-unit analysis and multi-unit analysis with different weighting schemes.
"""

from configuration.make_toy_model import make_model
from rankers.single_unit_ranks import unit_citation_matrix
from rankers.multi_unit_ranks import multi_unit_citation_matrix

def demonstrate_conservation():
    """Demonstrate that citation totals are conserved."""
    
    # Generate toy model data
    data = make_model(verbose=False)
    
    print("Citation Conservation Demonstration")
    print("=" * 50)
    
    # Single-unit analysis - all should have same total
    print("\n1. Single-Unit Analysis:")
    
    for unit_type in ['journal', 'author', 'institution']:
        matrix = unit_citation_matrix(data, unit_type)
        total = matrix.sum().sum()
        print(f"   {unit_type.capitalize()}: {total:.6f} citations")
    
    # Multi-unit analysis with different weights
    print("\n2. Multi-Unit Analysis with Different Weights:")
    
    weight_scenarios = [
        ("Equal weights", {'journal': 1/3, 'author': 1/3, 'institution': 1/3}),
        ("Journal-focused", {'journal': 0.6, 'author': 0.2, 'institution': 0.2}),
        ("Author-focused", {'journal': 0.1, 'author': 0.8, 'institution': 0.1}),
        ("Institution-focused", {'journal': 0.2, 'author': 0.2, 'institution': 0.6}),
        ("Extreme - Journal only", {'journal': 1.0, 'author': 0.0, 'institution': 0.0}),
    ]
    
    for name, weights in weight_scenarios:
        result = multi_unit_citation_matrix(data, weights=weights)
        matrix = result['matrix']
        total = matrix.sum().sum()
        
        print(f"   {name}: {total:.6f} citations")
        
        # Show distribution
        journals = [col for col in matrix.columns if col.startswith('J_')]
        authors = [col for col in matrix.columns if col.startswith('A_')]
        institutions = [col for col in matrix.columns if col.startswith('I_')]
        
        j_total = matrix.loc[journals, journals].sum().sum()
        a_total = matrix.loc[authors, authors].sum().sum()
        i_total = matrix.loc[institutions, institutions].sum().sum()
        
        print(f"     → J-J: {j_total:.3f}, A-A: {a_total:.3f}, I-I: {i_total:.3f}")
    
    print("\n3. Mathematical Verification:")
    print("   All totals should be identical, demonstrating proper")
    print("   conservation of citations between single and multi-unit analyses.")
    
    # Verify fractional credit within multi-unit
    print("\n4. Fractional Credit Details:")
    result = multi_unit_citation_matrix(data, verbose=False)
    matrix = result['matrix']
    
    print("   Multi-unit matrix shows fractional values where multiple")
    print("   authors/institutions share credit for the same citation.")
    print("   Example entries from author block:")
    
    authors = [col for col in matrix.columns if col.startswith('A_')]
    author_block = matrix.loc[authors, authors]
    
    for i, (idx, row) in enumerate(author_block.iterrows()):
        if i < 3:  # Show first 3 rows
            non_zero = [(col, val) for col, val in row.items() if val > 0]
            if non_zero:
                print(f"     {idx}: {non_zero[:3]}")  # Show first 3 non-zero entries

if __name__ == "__main__":
    demonstrate_conservation()