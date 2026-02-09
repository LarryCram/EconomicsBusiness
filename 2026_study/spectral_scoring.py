"""
Spectral scoring for bibliometric networks.
Loads edge lists, converts to CSR, applies normalizations, runs Katz iteration.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle

# Configuration
CONFIG = {
    'edge_list_file': '/home/lc/m/working/econ_bus_edge_lists.parquet',
    'output_dir': '/home/lc/m/working',
    'projection_types': ['s', 'a', 'i', 'si', 'sa', 'ai'],  # Updated to match actual data
    'normalizations': ['PN', 'G', 'S', 'H'],  # Pinski-Narin, Geller, Scimago, HITS
    'katz_alpha': 0.85,
    'max_iterations': 1000,
    'convergence_tol': 1e-6
}


def load_edge_lists():
    """Load parquet edge list data."""
    print(f"Loading edge lists from {CONFIG['edge_list_file']}")
    df = pd.read_parquet(CONFIG['edge_list_file'])
    print(f"Loaded {len(df)} edges")
    print(f"Columns: {list(df.columns)}")
    print(f"First few rows:")
    print(df.head())
    
    if 'projection_type' in df.columns:
        print(f"Projection type counts:")
        print(df['projection_type'].value_counts().sort_index())
    else:
        print("Available columns for grouping:")
        for col in df.columns:
            if df[col].dtype == 'object':
                print(f"  {col}: {df[col].nunique()} unique values")
    
    return df


def build_node_mappings(df, projection_type):
    """Create bidirectional mappings between openalex IDs and CSR indices."""
    subset = df[df['projection_type'] == projection_type].copy()
    
    # Get unique nodes from both citer and cited columns
    all_nodes = pd.concat([subset['citer'], subset['cited']]).unique()
    
    # Sort lexically for reproducibility (also groups /I before /S in hybrid matrices)
    all_nodes = np.sort(all_nodes)
    
    # Create mappings
    id_to_idx = {node_id: idx for idx, node_id in enumerate(all_nodes)}
    idx_to_id = {idx: node_id for node_id, idx in id_to_idx.items()}
    
    print(f"{projection_type}: {len(all_nodes)} unique nodes")
    return id_to_idx, idx_to_id


def build_csr_matrix(df, projection_type, id_to_idx):
    """Convert edge list to CSR sparse matrix."""
    subset = df[df['projection_type'] == projection_type].copy()
    
    # Map IDs to indices
    subset['citer_idx'] = subset['citer'].map(id_to_idx)
    subset['cited_idx'] = subset['cited'].map(id_to_idx)
    
    # Check for unmapped nodes before dropping
    initial_rows = len(subset)
    unmapped_citer = subset['citer_idx'].isna().sum()
    unmapped_cited = subset['cited_idx'].isna().sum()
    
    if unmapped_citer > 0:
        print(f"  Warning: {unmapped_citer} edges with unmapped citer nodes")
    if unmapped_cited > 0:
        print(f"  Warning: {unmapped_cited} edges with unmapped cited nodes")
    
    # Remove any edges with unmapped nodes
    subset = subset.dropna(subset=['citer_idx', 'cited_idx'])
    
    dropped_rows = initial_rows - len(subset)
    if dropped_rows > 0:
        print(f"  Dropped {dropped_rows} edges due to unmapped nodes")
    
    n_nodes = len(id_to_idx)
    
    # Create sparse matrix
    matrix = csr_matrix(
        (subset['weight'], (subset['citer_idx'], subset['cited_idx'])),
        shape=(n_nodes, n_nodes)
    )
    
    print(f"{projection_type}: {matrix.shape[0]}×{matrix.shape[1]} matrix, {matrix.nnz} non-zero entries")
    return matrix


def save_csr_data(matrix, id_to_idx, idx_to_id, projection_type):
    """Save CSR matrix and mappings."""
    base_path = f"{CONFIG['output_dir']}/csr_{projection_type}"
    
    # Save matrix in scipy format
    from scipy.sparse import save_npz
    save_npz(f"{base_path}_matrix.npz", matrix)
    
    # Save mappings
    with open(f"{base_path}_mappings.pkl", 'wb') as f:
        pickle.dump({'id_to_idx': id_to_idx, 'idx_to_id': idx_to_id}, f)
    
    print(f"Saved CSR data for {projection_type}")


def main():
    """Main processing pipeline."""
    # Load data
    df = load_edge_lists()
    
    # Process each projection type
    for proj_type in CONFIG['projection_types']:
        if proj_type not in df['projection_type'].values:
            print(f"Warning: {proj_type} not found in data")
            continue
            
        print(f"\nProcessing {proj_type}...")
        
        # Build mappings
        id_to_idx, idx_to_id = build_node_mappings(df, proj_type)
        
        # Build CSR matrix
        matrix = build_csr_matrix(df, proj_type, id_to_idx)
        
        # Save results
        save_csr_data(matrix, id_to_idx, idx_to_id, proj_type)
    
    print(f"\nCSR conversion complete. Data saved to {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()