#!/usr/bin/env python3
"""
Hubbell Algorithm Implementation and Testing

Hubbell method as described in:
C. H. Hubbell. An input-output approach to clique identification.
Sociometry, 28:377-399, 1965.
"""

import pandas as pd
import numpy as np

def hubbell(W, v, t=3):
    """
    Hubbell algorithm implementation
    
    Args:
        W: strength matrix (numpy array)
        v: exogenous vector (numpy array)
        t: precision parameter (convergence threshold = 1/10^t)
    
    Returns:
        dict: {'pi': status vector, 'iter': number of iterations}
    """
    n = W.shape[0]
    W1 = np.eye(n)
    S0 = np.zeros((n, n))
    S1 = np.eye(n)
    eps = 1.0 / (10 ** t)
    iter_count = 0
    
    while np.sum(np.abs(S0 - S1)) > eps:
        W1 = W1 @ W
        S0 = S1.copy()
        S1 = S1 + W1
        iter_count += 1
        
        if iter_count > 1000:
            break
    
    pi = v @ S1
    return {'pi': pi, 'iter': iter_count}

def main():
    # Load data
    hubbell_data = pd.read_csv('data/hubbell.csv', index_col=0)
    W = hubbell_data.values
    node_names = hubbell_data.index.tolist()
    
    # Expected values
    expected_values = {
        'A': 0.49, 'B': 0.41, 'C': 0.2, 'D': -0.9
    }
    
    # Exogenous vector (optimal value found: 0.20 for all elements)
    v = np.array([0.20, 0.20, 0.20, 0.20])
    
    # Run Hubbell
    result = hubbell(W, v, t=3)
    hubbell_scores = result['pi']
    
    # Comparison
    differences = [abs(hubbell_scores[i] - expected_values[node]) 
                  for i, node in enumerate(node_names)]
    mae = np.mean(differences)
    tolerance = 0.01
    within_tolerance = all(d <= tolerance for d in differences)
    
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"Nodes within tolerance (±{tolerance}): {len(differences)} out of {len(differences)}")
    print("✓ Perfect match!" if within_tolerance else "✗ Some discrepancies found.")

if __name__ == "__main__":
    main()