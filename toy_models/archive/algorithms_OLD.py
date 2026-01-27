import pandas as pd
import numpy as np


def pinski_narin(C, t=3):
    """
    Pinski-Narin algorithm implementation
    
    Args:
        C: citation matrix (pandas DataFrame or numpy array)
        t: precision parameter (convergence threshold = 1/10^t)
    
    Returns:
        dict: {'pi': influence vector, 'iter': number of iterations}
    """
    # Convert to numpy array if it's a DataFrame
    if isinstance(C, pd.DataFrame):
        C_matrix = C.values
        index_labels = C.index.tolist()
    else:
        C_matrix = C.copy()
        index_labels = list(range(C_matrix.shape[0]))
    
    n = C_matrix.shape[0]
    s = np.sum(C_matrix, axis=0)  # Column sums (total citations received)
    
    # Create H matrix: H[i,j] = C[i,j] / s(j) if s(j) > 0, else 0
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if s[j] > 0:
                H[i, j] = C_matrix[i, j] / s[j]
            else:
                H[i, j] = 0
    
    pi0 = np.zeros(n)
    pi1 = np.ones(n) / n
    eps = 1.0 / (10 ** t)
    iter_count = 0
    
    while np.sum(np.abs(pi0 - pi1)) > eps:
        pi0 = pi1.copy()
        pi1 = H @ pi1  # Matrix-vector multiplication
        iter_count += 1
        
        if iter_count > 1000:
            break
    
    return {
        'pi': dict(zip(index_labels, [float(x) for x in pi1])), 
        'pi_vector': [float(x) for x in pi1],
        'iter': iter_count
    }


def pagerank(C, v=None, alpha=1.0, t=3):
    """
    PageRank algorithm implementation
    
    Args:
        C: citation matrix (pandas DataFrame or numpy array)
        v: personalization vector (default: uniform)
        alpha: damping factor (default: 0.85)
        t: precision parameter (convergence threshold = 1/10^t)
    
    Returns:
        dict: {'pi': PageRank vector, 'iter': number of iterations}
    """
    # Convert to numpy array if it's a DataFrame
    if isinstance(C, pd.DataFrame):
        H = C.values
        index_labels = C.index.tolist()
    else:
        H = C.copy()
        index_labels = list(range(H.shape[0]))
    
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
    
    return {
        'pi': dict(zip(index_labels, [float(x) for x in pi1])), 
        'pi_vector': [float(x) for x in pi1],
        'iter': iter_count
    }


def is_primitive_matrix(C, max_power=20):
    """
    Check if a citation matrix is primitive.
    
    A matrix is primitive if:
    1. It is non-negative (all entries >= 0)
    2. It is irreducible (strongly connected)
    3. It is aperiodic (gcd of cycle lengths = 1)
    
    Equivalently, a matrix is primitive if some power of the matrix has all positive entries.
    
    Args:
        C: citation matrix (pandas DataFrame or numpy array)
        max_power: maximum power to check (default: 20)
        
    Returns:
        dict: {
            'is_primitive': bool,
            'is_non_negative': bool,
            'positive_power': int or None (power where all entries > 0),
            'analysis': str
        }
    """
    # Convert to numpy array if it's a DataFrame
    if isinstance(C, pd.DataFrame):
        matrix = C.values
        index_labels = C.index.tolist()
    else:
        matrix = C.copy()
        index_labels = list(range(matrix.shape[0]))
    
    n = matrix.shape[0]
    
    # Check if matrix is non-negative
    is_non_negative = np.all(matrix >= 0)
    if not is_non_negative:
        return {
            'is_primitive': False,
            'is_non_negative': False,
            'positive_power': None,
            'analysis': 'Matrix contains negative entries'
        }
    
    # Check powers of the matrix
    current_power = matrix.copy()
    
    for power in range(1, max_power + 1):
        if power > 1:
            current_power = current_power @ matrix
        
        # Check if all entries are positive
        if np.all(current_power > 0):
            has_self_loops = np.any(np.diag(matrix) > 0)
            analysis = f"Matrix is primitive (all entries positive at power {power})"
            if has_self_loops:
                analysis += ". Matrix has self-loops which helps with aperiodicity."
            
            return {
                'is_primitive': True,
                'is_non_negative': True,
                'positive_power': power,
                'analysis': analysis
            }
    
    # If we get here, no power up to max_power had all positive entries
    has_self_loops = np.any(np.diag(matrix) > 0)
    analysis = f"Matrix may not be primitive (no power up to {max_power} has all positive entries)"
    if has_self_loops:
        analysis += ". Matrix has self-loops which helps with aperiodicity."
    
    return {
        'is_primitive': False,
        'is_non_negative': True,
        'positive_power': None,
        'analysis': analysis
    }


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
        print(f"  Is primitive: {primitivity_result['is_primitive']}")
        print(f"  Analysis: {primitivity_result['analysis']}")
        if primitivity_result['positive_power']:
            print(f"  First positive power: {primitivity_result['positive_power']}")
    
    return results