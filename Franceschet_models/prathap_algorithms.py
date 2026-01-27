#!/usr/bin/env python3
"""
Pinski-Narin Algorithm Implementation and Testing

Pinski & Narin method as described in:
G. Pinski and F. Narin. Citation influence for journal aggregates of scientific publications:
Theory, with application to the literature of physics. Information Processing & Management, 
12(5):297-312, 1976.

Note: This implementation matches the R algorithm results.
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
    rs = C.sum(axis=1) # Column sums
    
    # Column normalization
    for j in range(n):
        if rs[j] != 0:
            H[:, j] = H[:, j] / rs[j]
    print(f'Pinski-Narin method H[:3,:3] :: \n{H[:3, :3]}')
    
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
    prathap_data = pd.read_csv('data/prathap_tab_1.csv', header=0, index_col=[0], usecols=[1,2,3,4,5,6,7,8,9,10,], nrows=10).reset_index() 
    n_size = prathap_data.shape[0]
    print(f'{prathap_data.shape = }\n{prathap_data.head(12)}')
    column_sum = prathap_data.sum(axis=0)
    row_sum = prathap_data.T.sum(axis=0)
    tlc = prathap_data.iloc[0, 0]
    print(column_sum.iloc[0]/row_sum.iloc[0], row_sum.iloc[0]/column_sum.iloc[0], tlc/row_sum.iloc[0], tlc/column_sum.iloc[0])
    print(f'Column sum\n{column_sum}')
    print(f'Row sum\n{row_sum}')

    prathap_normal = prathap_data.iloc[:n_size, :n_size].astype(float)
    prathap_normal.iloc[:n_size, :n_size] = 0.0
    print(f'{prathap_normal.shape = }\n{prathap_normal.head(12)}')
    
    for j in range(n_size):
        col_sum = column_sum.iloc[j]
        prathap_normal.iloc[j, :] = prathap_data.iloc[j, :]/col_sum
    prathap_normal['row_sum'] = prathap_normal.sum(axis=1)
    column_sums = pd.Series(prathap_normal.sum())
    prathap_normal = pd.concat([prathap_normal, pd.DataFrame([column_sums])], ignore_index=False)
    print(f'{prathap_normal.shape = }\n{prathap_normal.head(12)}')

    node_names = prathap_data.columns.tolist()
    
    # Run algorithm
    C = prathap_data.iloc[:n_size, :n_size].values.T # < --- Prathap is working with the convention that i, j is column, row 
                                                     #       where i gives a reference to j and equivalently j is cited by i. 
    result = pinski_narin(C, t=6)
    print(f'{result = }')
    print(f'{result['pi'].sum() = }')

if __name__ == "__main__":
    main()