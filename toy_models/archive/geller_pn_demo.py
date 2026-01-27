#!/usr/bin/env python3

"""
Demonstration of the relationship between Geller and Pinski-Narin algorithms.

This script shows that for any unit i:
Geller_weight[i] = output[i] / PN_weight[i]

Where:
- output[i] = total citations made by unit i (row sum of citation matrix)
- PN_weight[i] = Pinski-Narin influence score for unit i
- Geller_weight[i] = Geller influence score for unit i
"""

import numpy as np
import pandas as pd
from configuration.make_toy_model import make_model
from rankers.single_unit_ranks import unit_citation_matrix
from utils.algorithms import pinski_narin, pagerank

def demonstrate_geller_pn_relationship():
    """Demonstrate the relationship between Geller and Pinski-Narin algorithms."""
    
    print("Geller-Pinski-Narin Relationship Demonstration")
    print("=" * 60)
    
    # Generate toy model data
    data = make_model(verbose=False)
    
    # Get journal citation matrix
    citation_matrix = unit_citation_matrix(data, 'journal', verbose=False)
    
    print("\nJournal Citation Matrix:")
    print(citation_matrix)
    print(f"Total citations: {citation_matrix.sum().sum()}")
    
    # Run Pinski-Narin algorithm
    print("\n" + "="*60)
    print("PINSKI-NARIN ALGORITHM")
    print("="*60)
    
    pn_result = pinski_narin(citation_matrix)
    print(f"PN converged in {pn_result['iter']} iterations")
    print(f"PN weights {pn_result['pi_vector'] = } {sum(pn_result['pi_vector']) = }")
    print(f"PN weights\n{pn_result['pi']}")
    
    # Run Geller algorithm
    print("\n" + "="*60)
    print("GELLER ALGORITHM")
    print("="*60)
    
    geller_result = pagerank(citation_matrix)
    print(f"Geller/Markov converged in {geller_result['iter']} iterations")    
    print(f"Geller/Markov weights {geller_result['pi_vector'] = } {sum(geller_result['pi_vector']) = }")
    print(f"Geller/Markov weights\n{geller_result['pi']}")

    row_sums = citation_matrix.sum(axis=1).to_list()
    geller_check = [x/y for x, y in zip(pn_result['pi_vector'], row_sums)]
    norm = sum(geller_check)
    geller_ratio = [(x/(norm))/y for x, y in zip(geller_check, geller_result['pi_vector'])]
    print(f"Row sums = {row_sums = }")
    print(f'PN/Geller ratio after adjustment {geller_ratio = } {sum(geller_ratio) = }')


if __name__ == "__main__":
    demonstrate_geller_pn_relationship()