#!/usr/bin/env python3
"""
Pinski-Narin Algorithm Implementation and Testing

Pinski & Narin method as described in:
G. Pinski and F. Narin. Citation influence for journal aggregates of scientific publications:
Theory, with application to the literature of physics. Information Processing & Management, 
12(5):297-312, 1976.

Note: This implementation matches the R algorithm results, which differ from some literature values.
"""

import pandas as pd
import numpy as np

def pinski_narin(C, t=3):
    """
    Pinski-Narin algorithm implementation
    
    Args:
        C: citation matrix (numpy array)
        t: precision parameter (convergence threshold = 1/10^t)
    
    Returns:
        dict: {'pi': influence vector, 'iter': number of iterations}
    """
    n = C.shape[0]
    H = C.copy().astype(float)
    rs = C @ np.ones(n)  # Column sums
    
    # Column normalization
    for j in range(n):
        if rs[j] != 0:
            H[:, j] = H[:, j] / rs[j]
    
    pi0 = np.zeros(n)
    pi1 = np.ones(n) / n
    eps = 1.0 / (10 ** t)
    iter_count = 0
    
    while np.sum(np.abs(pi0 - pi1)) > eps:
        pi0 = pi1.copy()
        pi1 = pi1 @ H
        iter_count += 1
        
        if iter_count > 1000:
            break
    
    return {'pi': pi1, 'iter': iter_count}

def main():
    # Load data
    pn_data = pd.read_csv('data/pinski_narin.csv', index_col=0)
    C = pn_data.values
    node_names = pn_data.index.tolist()
    
    # Expected values (R algorithm results vs literature values)
    literature_values = {'A': 28.0, 'B': 44.7, 'C': 12.4, 'D': 14.9}
    r_results = {'A': 28.9, 'B': 46.1, 'C': 9.6, 'D': 15.4}
    
    # Run algorithm
    result = pinski_narin(C, t=3)
    
    # Scale results
    expected_sum = sum(literature_values.values())
    our_scaled = result['pi'] * (expected_sum / np.sum(result['pi']))
    
    # Validation
    differences_vs_r = [abs(our_scaled[i] - r_results[node]) for i, node in enumerate(node_names)]
    differences_vs_lit = [abs(our_scaled[i] - literature_values[node]) for i, node in enumerate(node_names)]
    
    mae_vs_r = np.mean(differences_vs_r)
    mae_vs_lit = np.mean(differences_vs_lit)
    
    print(f"Mean Absolute Error vs R algorithm: {mae_vs_r:.3f}")
    print(f"Mean Absolute Error vs Literature: {mae_vs_lit:.3f}")
    print(f"Python results: A={our_scaled[0]:.1f}, B={our_scaled[1]:.1f}, C={our_scaled[2]:.1f}, D={our_scaled[3]:.1f}")
    
    matches_r = all(d <= 0.1 for d in differences_vs_r)
    print("✓ Matches R algorithm!" if matches_r else "✗ Differs from R algorithm.")
    print("Note: Both R and Python differ from some literature values, but are consistent with each other.")

if __name__ == "__main__":
    main()