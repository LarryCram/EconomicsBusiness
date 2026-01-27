#!/usr/bin/env python3
"""
PageRank Algorithm Implementation and Testing

PageRank method as described in:
A. N. Langville and C. D. Meyer. Google's PageRank and Beyond: 
The Science of Search Engine Rankings. Princeton University Press, 2006.
"""

import pandas as pd
import numpy as np

def pagerank(H, v=None, alpha=0.85, t=3):
    """
    PageRank algorithm implementation
    
    Args:
        H: adjacency matrix (numpy array)
        v: personalization vector (default: uniform)
        alpha: damping factor (default: 0.85)
        t: precision parameter (convergence threshold = 1/10^t)
    
    Returns:
        dict: {'pi': PageRank vector, 'iter': number of iterations}
    """
    n = H.shape[0]
    a = np.zeros(n)
    rs = H @ np.ones(n)
    H_work = H.copy().astype(float)
    
    # Row normalization and dangling node detection
    for i in range(n):
        if rs[i] == 0:
            a[i] = 1
        else:
            H_work[i, :] = H_work[i, :] / rs[i]
    
    e = np.ones(n)
    if v is None:
        v = np.ones(n) / n
    
    pi0 = np.zeros(n)
    pi1 = np.ones(n) / n
    eps = 1.0 / (10 ** t)
    iter_count = 0
    
    while np.sum(np.abs(pi0 - pi1)) > eps:
        pi0 = pi1.copy()
        term1 = alpha * (pi1 @ H_work)
        term2 = (alpha * (pi1 @ a) + (1 - alpha) * (pi1 @ e)) * v
        pi1 = term1 + term2
        iter_count += 1
        
        if iter_count > 1000:
            break
    
    return {'pi': pi1, 'iter': iter_count}

def main():
    # Load data
    pagerank_data = pd.read_csv('data/pagerank.csv', index_col=0)
    H = pagerank_data.values
    node_names = pagerank_data.index.tolist()
    
    # Expected values
    expected_values = {
        'A': 3.3, 'B': 38.4, 'C': 34.3, 'D': 3.9, 'E': 8.1, 'F': 3.9,
        'G': 1.6, 'H': 1.6, 'I': 1.6, 'L': 1.6, 'M': 1.6
    }
    
    # Run PageRank
    result = pagerank(H, alpha=0.85, t=3)
    pi_scaled = (result['pi'] / result['pi'].sum()) * 100
    
    # Comparison
    differences = [abs(pi_scaled[i] - expected_values[node]) for i, node in enumerate(node_names)]
    mae = np.mean(differences)
    tolerance = 0.1
    within_tolerance = len([d for d in differences if d <= tolerance])
    
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"Nodes within tolerance (±{tolerance}): {within_tolerance}/{len(node_names)}")
    print("✓ Perfect match!" if within_tolerance == len(node_names) else "✗ Some discrepancies found.")

if __name__ == "__main__":
    main()