from configuration.make_toy_model import make_model
from rankers.multi_unit_ranks import multi_unit_citation_matrix, calculate_geller_from_pn
from utils.algorithms import pagerank, pinski_narin


def fmt_sorted(pi_dict):
    return sorted(pi_dict.items(), key=lambda x: x[1], reverse=True)


def main():
    data = make_model(verbose=False)
    multi = multi_unit_citation_matrix(data)
    C = multi['matrix']

    # Compute PN
    pn = pinski_narin(C)

    # Compute PageRank with alpha=1.0 (no teleportation)
    pr = pagerank(C, alpha=1.0)

    # Compute Geller from PN
    geller = calculate_geller_from_pn(C, pn)

    # Compare
    units = list(C.index)

    print('\nUnit\tPageRank(alpha=1.0)\tGeller(predicted)\tDiff')
    print('-'*60)
    for u in units:
        pr_val = pr['pi'].get(u, 0.0)
        g_val = geller['pi'].get(u, 0.0)
        diff = pr_val - g_val
        print(f"{u}\t{pr_val:.6f}\t{g_val:.6f}\t{diff:.6e}")

    # Print summary norms
    pr_vec = [pr['pi'][u] for u in units]
    g_vec = [geller['pi'][u] for u in units]

    import numpy as np
    l1 = np.sum(np.abs(np.array(pr_vec) - np.array(g_vec)))
    print('\nL1 norm of difference:', l1)

    print('\nTop 5 PageRank (alpha=1.0):')
    for k, v in fmt_sorted(pr['pi'])[:5]:
        print(f"  {k}: {v:.6f}")

    print('\nTop 5 Geller:')
    for k, v in fmt_sorted(geller['pi'])[:5]:
        print(f"  {k}: {v:.6f}")


if __name__ == '__main__':
    main()
