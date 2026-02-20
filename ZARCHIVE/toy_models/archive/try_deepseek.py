import numpy as np
import pandas as pd

def pinsky_narin(C):
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
    eigvals, eigvecs = np.linalg.eig(Gamma_T)
    idx = np.argmax(np.real(eigvals))
    w = np.real(eigvecs[:, idx])
    
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
    eigvals, eigvecs = np.linalg.eig(P_T)
    idx = np.argmax(np.real(eigvals))
    v = np.real(eigvecs[:, idx])
    
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
    eigvals, eigvecs = np.linalg.eig(G_T)
    idx = np.argmax(np.real(eigvals))
    v = np.real(eigvecs[:, idx])
    
    # Normalize to sum to 1
    v = v / v.sum()
    
    return v

def test_relationship(C, units, alpha=0.85):
    """
    Test the relationship between PN, Geller, and PageRank weights
    """
    # Compute weights using all methods
    w_pn = pinsky_narin(C)
    v_geller = geller(C)
    v_pagerank = pagerank(C, alpha=alpha)
    
    # Get row sums and total citations
    s = C.sum(axis=1)
    R = s.sum()
    
    print("Citation Matrix C:")
    print(pd.DataFrame(C, index=units, columns=units))
    print("Row sums s_j:", dict(zip(units, s)))
    print(f"Total citations R: {R:.6f}")
    
    print(f"\nPinski-Narin weights:")
    print(dict(zip(units, w_pn)))
    print(f"Verification - ∑ w_k s_k = {w_pn @ s:.6f} (should equal R = {R:.6f})")
    
    print(f"\nGeller weights:")
    print(dict(zip(units, v_geller)))
    print(f"Verification - ∑ v_k = {v_geller.sum():.6f}")
    
    print(f"\nPageRank weights (α={alpha}):")
    print(dict(zip(units, v_pagerank)))
    print(f"Verification - ∑ v_k = {v_pagerank.sum():.6f}")
    
    # Test PN-Geller relationship
    print(f"\n=== Testing PN-Geller Relationship: v = (1/R) * Λ * w ===")
    v_predicted = (s / R) * w_pn
    v_predicted = v_predicted / v_predicted.sum()  # Normalize for comparison
    
    print("Actual Geller:   ", dict(zip(units, v_geller)))
    print("Predicted Geller:", dict(zip(units, v_predicted)))
    
    ratios = v_geller / v_predicted
    print("Ratios (actual/predicted):", dict(zip(units, ratios)))
    print(f"Ratio std: {np.std(ratios):.6f}")
    
    if np.std(ratios) < 1e-10:
        print("✓ Perfect PN-Geller relationship!")
    elif np.std(ratios) < 0.01:
        print("✓ PN-Geller relationship holds well!")
        scaling = np.mean(ratios)
        print(f"Actual: v_geller = {scaling:.6f} * (1/R) * Λ * w_pn")
    else:
        print("✗ PN-Geller relationship doesn't hold well")
    
    # Test Geller-PageRank relationship
    print(f"\n=== Testing Geller-PageRank Relationship ===")
    print("Geller weights:    ", dict(zip(units, v_geller)))
    print("PageRank weights:  ", dict(zip(units, v_pagerank)))
    
    differences = np.abs(v_geller - v_pagerank)
    print("Absolute differences:", dict(zip(units, differences)))
    print(f"Max difference: {np.max(differences):.2e}")
    
    if np.max(differences) < 1e-10:
        print("✓ Geller and PageRank are identical!")
    elif np.max(differences) < 0.01:
        print("✓ Geller and PageRank are very similar!")
    else:
        print("✗ Geller and PageRank differ significantly")
    
    return w_pn, v_geller, v_pagerank

# Test with our citation matrix
C = np.array([
    [2.0, 2.0, 1.0],
    [2.0, 2.0, 0.0], 
    [3.0, 0.0, 1.0]
])

# C = np.array([
#     [0.6666666666666666, 0.6666666666666666, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
#     [0.6666666666666666, 0.6666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
#     [1.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
#     [0.0, 0.0, 0.0, 0.3333333333333333, 0.25, 0.5, 0.08333333333333333, 0.0, 0.0, 0.0], 
#     [0.0, 0.0, 0.0, 0.41666666666666663, 0.25, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0], 
#     [0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.0, 0.0, 0.0, 0.0], 
#     [0.0, 0.0, 0.0, 0.49999999999999994, 0.25, 0.3333333333333333, 0.08333333333333333, 0.0, 0.0, 0.0], 
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.638888888888889, 0.3611111111111111, 0.7222222222222221], 
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4444444444444444, 0.16666666666666666, 0.2777777777777778], 
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.638888888888889, 0.3611111111111111, 0.7222222222222221]
# ])


units = [f'u{i}' for i in range(len(C))]

print("=" * 60)
print("TEST WITH STANDARD PARAMETERS (α=0.85)")
print("=" * 60)
w_pn, v_geller, v_pagerank = test_relationship(C, units, alpha=0.85)

# Additional test: what happens with different alpha values?
print("\n" + "=" * 60)
print("TESTING DIFFERENT ALPHA VALUES")
print("=" * 60)

for alpha in [0.0, 0.5, 0.99, 1.0]:
    v_pr = pagerank(C, alpha=alpha)
    print(f"\nPageRank (α={alpha}): {dict(zip(units, v_pr))}")
    
    if alpha == 1.0:
        diff = np.max(np.abs(v_pr - v_geller))
        print(f"  Difference from Geller: {diff:.2e}")
        if diff < 1e-10:
            print("  ✓ PageRank(α=1.0) = Geller")

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