"""
Spectral normalization and Katz iteration for bibliometric scoring.
Loads CSR matrices, applies normalizations, runs power iteration.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, diags
import pickle
from pathlib import Path

# Configuration
CONFIG = {
    'input_dir': '/home/lc/m/working',
    'output_dir': '/home/lc/m/working',
    'projection_types': ['s', 'a', 'i', 'si', 'sa', 'ai'],
    'normalizations': ['PN', 'G', 'S', 'H'],  # Pinski-Narin, Geller, Scimago, HITS
    'katz_alpha': 0.85,
    'max_iterations': 1000,
    'convergence_tol': 1e-6
}


def load_csr_data(projection_type):
    """Load CSR matrix and mappings for a projection type."""
    base_path = f"{CONFIG['input_dir']}/csr_{projection_type}"
    
    # Load matrix
    matrix = load_npz(f"{base_path}_matrix.npz")
    
    # Load mappings
    with open(f"{base_path}_mappings.pkl", 'rb') as f:
        mappings = pickle.load(f)
    
    print(f"Loaded {projection_type}: {matrix.shape[0]}×{matrix.shape[1]} matrix, {matrix.nnz} edges")
    return matrix, mappings['id_to_idx'], mappings['idx_to_id']


def normalize_matrix(matrix, normalization_type):
    """Apply matrix normalization according to type."""
    C = matrix.copy()
    n = C.shape[0]
    
    if normalization_type == 'PN':  # Pinski-Narin: column-normalize
        col_sums = np.array(C.sum(axis=0)).flatten()
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        D1 = diags(1.0 / col_sums)
        return C @ D1
        
    elif normalization_type == 'G':  # Geller: row-normalize  
        row_sums = np.array(C.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        D2 = diags(1.0 / row_sums)
        return D2 @ C
        
    elif normalization_type == 'S':  # Scimago: needs article counts (placeholder for now)
        # TODO: Load article counts per node when available
        # For now, use row normalization as approximation
        print(f"  Warning: Scimago normalization using row-normalize approximation")
        row_sums = np.array(C.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        D2 = diags(1.0 / row_sums)
        return D2 @ C
        
    elif normalization_type == 'H':  # HITS: C^T @ C
        return C.T @ C
        
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")


def katz_iteration(matrix, alpha=0.85, max_iter=1000, tol=1e-6):
    """Run Katz iteration to find principal eigenvector."""
    n = matrix.shape[0]
    
    # Initialize with uniform vector
    x = np.ones(n) / n
    
    for i in range(max_iter):
        x_new = alpha * (matrix.T @ x) + (1 - alpha) * np.ones(n) / n
        
        # Normalize
        x_new = x_new / np.sum(x_new)
        
        # Check convergence
        diff = np.linalg.norm(x_new - x)
        if diff < tol:
            print(f"  Converged after {i+1} iterations (diff: {diff:.2e})")
            return x_new
            
        x = x_new
    
    print(f"  Warning: Did not converge after {max_iter} iterations")
    return x


def save_results(scores, projection_type, normalization_type, idx_to_id):
    """Save spectral scores with node IDs."""
    # Create results DataFrame
    results = pd.DataFrame({
        'node_id': [idx_to_id[i] for i in range(len(scores))],
        'score': scores,
        'rank': range(1, len(scores) + 1)  # Will be re-ranked after sorting
    })
    
    # Sort by score (descending) and assign ranks
    results = results.sort_values('score', ascending=False).reset_index(drop=True)
    results['rank'] = range(1, len(results) + 1)
    
    # Save
    output_file = f"{CONFIG['output_dir']}/scores_{projection_type}_{normalization_type}.parquet"
    results.to_parquet(output_file, index=False)
    
    print(f"  Saved results to {output_file}")
    print(f"  Top score: {results.iloc[0]['score']:.6f}")
    
    return results


def main():
    """Main processing pipeline."""
    
    for proj_type in CONFIG['projection_types']:
        print(f"\nProcessing {proj_type}...")
        
        try:
            # Load CSR data
            matrix, id_to_idx, idx_to_id = load_csr_data(proj_type)
            
            # Process each normalization
            for norm_type in CONFIG['normalizations']:
                print(f"  Normalization: {norm_type}")
                
                # Apply normalization
                normalized_matrix = normalize_matrix(matrix, norm_type)
                
                # Run Katz iteration
                scores = katz_iteration(
                    normalized_matrix, 
                    alpha=CONFIG['katz_alpha'],
                    max_iter=CONFIG['max_iterations'],
                    tol=CONFIG['convergence_tol']
                )
                
                # Save results
                save_results(scores, proj_type, norm_type, idx_to_id)
                
        except FileNotFoundError:
            print(f"  Warning: CSR data not found for {proj_type}")
            continue
    
    print(f"\nSpectral scoring complete. Results saved to {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()