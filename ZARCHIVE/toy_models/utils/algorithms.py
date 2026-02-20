import numpy as np
import pandas as pd
import networkx as nx

def pinski_narin(C):
    """
    Compute Pinski-Narin influence weights
    
    Parameters:
    C : numpy array (n x n)
        Citation matrix where C[i,j] = citations from i to j
        
    Returns:
    w : numpy array (n,)
        PN influence weights normalized such that w^T @ s = R
        where s = row sums of C, R = total citations
    """
    n = C.shape[0]
    
    # Row sums: s_j = sum_k C[j,k] (output of unit j)
    s = C.sum(axis=1)
    Lambda = np.diag(s)
    Lambda_inv = np.diag(1 / s)
    R = s.sum()  # Total citations
    
    # PN matrix: Γ = C Λ^{-1}, Γ_ij = C[i,j] / s_j
    Gamma = C @ Lambda_inv
    Gamma_T = Gamma.T  # For eigen equation: Γ^T w = w
    
    # Find dominant eigenvector
    try:
        eigvals, eigvecs = np.linalg.eig(Gamma_T)
        idx = np.argmax(np.real(eigvals))
        w = np.real(eigvecs[:, idx])
    except Exception as e:
        print(f"Exception in pinski_narin call {e = }")
        print(Gamma_T)
        return [0.0]*n

    # Apply scaling: w^T s = R
    scaling_factor = R / (w @ s)
    w = w * scaling_factor
    
    return w

def geller(C):
    """
    Compute Geller influence weights
    
    Parameters:
    C : numpy array (n x n)
        Citation matrix where C[i,j] = citations from i to j
        
    Returns:
    v : numpy array (n,)
        Geller influence weights normalized to sum to 1
    """
    n = C.shape[0]
    
    # Row sums: r_i = sum_k C[i,k] (output of unit i)
    r = C.sum(axis=1)
    R_mat = np.diag(r)
    R_inv = np.diag(1 / r)
    
    # Geller matrix: P = R^{-1} C, P_ij = C[i,j] / r_i
    P = R_inv @ C
    P_T = P.T  # For eigen equation: P^T v = v
    
    # Find dominant eigenvector (stationary distribution)
    try:
        eigvals, eigvecs = np.linalg.eig(P_T)
        idx = np.argmax(np.real(eigvals))
        v = np.real(eigvecs[:, idx])
    except Exception as e:
        print(f"Exception in geller call {e = }")
        print(P_T)
        return [0.0]*n
    
    # Normalize to sum to 1 (standard for Markov chains)
    v = v / v.sum()
    
    return v

def pagerank(C, alpha=0.85):
    """
    Compute PageRank weights
    
    Parameters:
    C : numpy array (n x n)
        Citation matrix where C[i,j] = citations from i to j
    alpha : float (default=0.85)
        Damping factor
        
    Returns:
    v : numpy array (n,)
        PageRank weights normalized to sum to 1
    """
    n = C.shape[0]
    
    # Row sums: r_i = sum_k C[i,k] (output of unit i)
    r = C.sum(axis=1)
    R_inv = np.diag(1 / r)
    
    # Geller matrix: P = R^{-1} C, P_ij = C[i,j] / r_i
    P = R_inv @ C
    
    # Handle dangling nodes (rows with zero sum) - replace with uniform distribution
    # For our matrix, no dangling nodes exist since all row sums > 0
    
    # Create teleportation matrix: E = ee^T / n where e is vector of ones
    e = np.ones(n)
    E = np.outer(e, e) / n
    
    # PageRank matrix: G = αP + (1-α)E
    G = alpha * P + (1 - alpha) * E
    G_T = G.T  # For eigen equation: G^T v = v
    
    # Find dominant eigenvector
    try:
        eigvals, eigvecs = np.linalg.eig(G_T)
        idx = np.argmax(np.real(eigvals))
        v = np.real(eigvecs[:, idx])
    except Exception as e:
        print(f"Exception in pagerank call {e = }")
        print(G_T)
        return [0.0]*n
    
    # Normalize to sum to 1
    v = v / v.sum()
    
    return v


def is_primitive_matrix(C):
    """
    Checks if a matrix M is non-negative, irreducible, and aperiodic.
    If it is, it computes the primitivity exponent.
    
    Args:
        M: A square numpy array.

    Returns:
        A string reporting the status and reason.
    """
    
    # === PRELIMINARY CHECKS ===
    M = C.to_numpy() 
    
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return "FAIL: Matrix is not square."
    
    n = M.shape[0]
    if n == 0:
        return "FAIL: Matrix is empty."

    # 1. Check for Non-Negativity
    if (M < 0).any():
        return "FAIL: Matrix is not non-negative (contains negative elements)."

    # 2. Check for Irreducibility
    # We use NetworkX to build a graph and check its properties.
    # An edge (i, j) exists if M[i, j] > 0.
    G = nx.from_numpy_array(M, create_using=nx.DiGraph)
    
    if not nx.is_strongly_connected(G):
        return "NOT PRIMITIVE (Reason: Matrix is REDUCIBLE)."

    # 3. Check for Aperiodicity
    if not nx.is_aperiodic(G):
        return "NOT PRIMITIVE (Reason: Matrix is IRREDUCIBLE but PERIODIC)."

    # === COMPUTE PRIMITIVITY EXPONENT ===
    # If we get here, the matrix IS primitive. We just need to find 
    # the exponent k by checking powers of M.
    
    # We only need to check up to Wielandt's bound which might be nnormous so include cut off. 
    # Each iteration is equivalent to counting inter-node paths at that separation.
    max_iterations = min((n - 1)**2 + 1, 32)
    
    # Start with M^1
    M_k = M.copy() 

    for k in range(1, max_iterations + 1):
        # Check if all elements are strictly positive
        if (M_k > 0).all():
            return f"Matrix is PRIMITIVE with exponent k = {k}."
        
        # Calculate the next power: M^(k+1) = M^k @ M
        if k < max_iterations:
            M_k = M_k @ M

    # This line should be unreachable if the graph checks are correct,
    # but it's good practice to include it.
    return "Error: Matrix passed graph checks but no positive power was found."

    # # --- Example Usage ---

    # # 1. Primitive Matrix (exponent k=3)
    # M_primitive = np.array([
    #     [0, 1, 0],
    #     [0, 0, 1],
    #     [1, 1, 0]
    # ])

    # # 2. Irreducible but Periodic Matrix
    # M_periodic = np.array([
    #     [0, 1],
    #     [1, 0]
    # ])

    # # 3. Reducible Matrix
    # M_reducible = np.array([
    #     [1, 1],
    #     [0, 1]
    # ])

    # # 4. Matrix with negative elements
    # M_negative = np.array([
    #     [1, -1],
    #     [1,  1]
    # ])

    # print(f"M1: {analyze_primitivity(M_primitive)}")
    # print(f"M2: {analyze_primitivity(M_periodic)}")
    # print(f"M3: {analyze_primitivity(M_reducible)}")
    # print(f"M4: {analyze_primitivity(M_negative)}")


def analyze_matrix_properties(C, verbose=False):
    """
    Analyze properties of a citation matrix.
    
    Args:
        C: citation matrix (pandas DataFrame)
        verbose: if True, print detailed analysis
        
    Returns:
        dict: analysis results
    """
    # Basic properties
    n_rows, n_cols = C.shape
    total_citations = C.sum().sum()
    density = np.count_nonzero(C.values) / (n_rows * n_cols)
    has_self_citations = np.any(np.diag(C.values) > 0)
    
    # Zero rows and columns
    row_sums = C.sum(axis=1).tolist()
    col_sums = C.sum(axis=0).tolist()
    zero_rows = sum(row_sums)
    zero_cols = sum(col_sums)
    
    # Primitivity analysis
    primitivity_result = is_primitive_matrix(C)
    
    results = {
        'size': (n_rows, n_cols),
        'total_citations': total_citations,
        'density': density,
        'has_self_citations': has_self_citations,
        'zero_rows': zero_rows,
        'zero_cols': zero_cols,
        'primitivity': primitivity_result,
        'row_sums': row_sums
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print("MATRIX ANALYSIS")
        print(f"{'='*50}")
        print(f"Matrix size: {n_rows}×{n_cols}")
        print(f"Total citations: {total_citations}")
        print(f"Density: {density:.3f}")
        print(f"Has self-citations: {has_self_citations}")
        print(f"Zero rows (units with no outgoing citations): {zero_rows}")
        print(f"Row sum vector: {row_sums}")
        print(f"Zero columns (units with no incoming citations): {zero_cols}")
        
        print(f"\nPrimitivity Analysis:")
        print(f"  {primitivity_result = }")

    
    return results

def verify_pn_geller(C, w_pn, v_geller):

    # Final verification
    print(f"\n" + "=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)

    s = C.sum(axis=1)
    R = s.sum()

    # Direct computation of relationships
    v_from_w = (s / R) * w_pn
    v_from_w = v_from_w / v_from_w.sum()

    w_from_v = R * (v_geller / s) 
    w_from_v = w_from_v / (w_from_v @ s) * R

    print(f"Max difference PN→Geller: {np.max(np.abs(v_geller - v_from_w)):.2e}")
    print(f"Max difference Geller→PN: {np.max(np.abs(w_pn - w_from_v)):.2e}")
    print(f"Max difference Geller→PageRank(α=1.0): {np.max(np.abs(v_geller - pagerank(C, 1.0))):.2e}")