#!/usr/bin/env python3
"""
HITS Algorithm Implementation and Testing

HITS method as described in:
J. M. Kleinberg. Authoritative sources in a hyperlinked environment.
Journal of the ACM, 46(5):604-632, 1999.
"""

import pandas as pd
import numpy as np

def hits(L, t=3):
    """
    HITS algorithm implementation
    
    Args:
        L: adjacency matrix (numpy array)
        t: precision parameter (convergence threshold = 1/10^t)
    
    Returns:
        dict: {'a': authority vector, 'h': hub vector, 'val': dominant eigenvalue, 'iter': iterations}
    """
    n = L.shape[0]
    A = L.T @ L  # Authority matrix

    x0 = np.zeros(n)
    x1 = np.ones(n) / n
    eps = 1.0 / (10 ** t)
    iter_count = 0
    
    while np.sum(np.abs(x0 - x1)) > eps:
        x0 = x1.copy()
        x1 = A @ x1
        m = x1[np.argmax(np.abs(x1))]
        x1 = x1 / m
        iter_count += 1
        
        if iter_count > 1000:
            break
    
    y = L @ x1
    return {'a': x1, 'h': y, 'val': m, 'iter': iter_count}

def main():
    # Load data
    pagerank_data = pd.read_csv('data/pagerank.csv', index_col=0)
    L = pagerank_data.values
    node_names = pagerank_data.index.tolist()
    
    # Expected values
    expected_authority = {
        'A': 4.7, 'B': 45.9, 'C': 0.0, 'D': 5.3, 'E': 38.9, 'F': 5.3,
        'G': 0.0, 'H': 0.0, 'I': 0.0, 'L': 0.0, 'M': 0.0
    }
    
    expected_hub = {
        'A': 0.0, 'B': 0.0, 'C': 8.1, 'D': 8.9, 'E': 9.9, 'F': 14.9,
        'G': 14.9, 'H': 14.9, 'I': 14.9, 'L': 6.8, 'M': 6.8
    }
    
    # Run HITS
    result = hits(L, t=3)
    
    # Scale results (using B for authority, G for hub as reference points)
    auth_scale = expected_authority['B'] / result['a'][1]  # B is index 1
    hub_scale = expected_hub['G'] / result['h'][6]  # G is index 6
    
    our_authority_scaled = result['a'] * auth_scale
    our_hub_scaled = result['h'] * hub_scale
    
    # Comparison
    auth_differences = [abs(our_authority_scaled[i] - expected_authority[node]) 
                       for i, node in enumerate(node_names)]
    hub_differences = [abs(our_hub_scaled[i] - expected_hub[node]) 
                      for i, node in enumerate(node_names)]
    
    auth_mae = np.mean(auth_differences)
    hub_mae = np.mean(hub_differences)
    tolerance = 0.1
    auth_within_tolerance = len([d for d in auth_differences if d <= tolerance])
    hub_within_tolerance = len([d for d in hub_differences if d <= tolerance])
    
    print(f"Mean Absolute Error (Authority): {auth_mae:.3f}")
    print(f"Mean Absolute Error (Hub): {hub_mae:.3f}")
    print(f"Authority nodes within tolerance (±{tolerance}): {auth_within_tolerance}/{len(node_names)}")
    print(f"Hub nodes within tolerance (±{tolerance}): {hub_within_tolerance}/{len(node_names)}")
    
    perfect_match = (auth_within_tolerance == len(node_names) and 
                    hub_within_tolerance == len(node_names))
    print("✓ Perfect match!" if perfect_match else "✗ Some discrepancies found.")

if __name__ == "__main__":
    main()