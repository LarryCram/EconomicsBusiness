# ALPHA.md — Notes on α, μ, primitivity, and the bipartite pipeline

## power_iteration
- signature power_iteration(M: CSR, alpha: float, mu: Optional[ndarray]=None, tol: float, max_iter: int)
- three valid regimes: (alpha=1, mu=None), (alpha<1, mu=None), (alpha<1, mu>0)
- test primitivity of M before iterating:
  - compute repeated boolean powers M, M^2, M^3, ... until all entries non-zero
  - the first such k is the primitivity index; report it
  - if k > 5 raise SystemExit
  - work with boolean sparsity pattern for efficiency
- implement power iteration with renormalisation after each step to control floating point error

## modes
- modes are 1000 (SS), 0001 (II), 1111 (SSII) and 0110 (SI/IS)

## modes 1000 0001 1111
- constructing H: CSR from edge list
- default is alpha=1 and mu=None
- alpha is a parameter
- chi is relevant only for 1111; use chi^* as default in that case
- after convergence compute pi_s, pi_i, v_s, v_i

## mode 0110
- bipartite() builds M_S = H_SI @ H_IS: N_s x N_s CSR from edge list
- H_SI and H_IS must be retained for recovery of pi_I
- default is alpha=1 and mu=None
- alpha is a per-hop parameter with the same meaning as in 1000 and 0001:
  each traversal of one directed edge is attenuated by alpha
- a round trip S->I->S traverses two edges, so M_S is called with alpha^2 as its
  round-trip damping: power_iteration(M_S, alpha^2, mu_eff)
- this ensures v_S from 0110 is directly comparable with v_S from 1000 at the same alpha
- chi is not relevant for 0110 (cancels in row normalisation)
- chi^* is not relevant for 0110
- call power_iteration(M_S, alpha^2, mu_eff) for pi_S
  - at alpha=1, mu=None: mu_eff = None
  - at alpha<1, mu!=None: mu_eff = mu_S + alpha * H_IS.T @ mu_I  (effective prior)
- recover pi_I:
  - at alpha=1: pi_I = H_SI.T @ pi_S
  - at alpha<1, mu!=None: pi_I = alpha * H_SI.T @ pi_S + (1-alpha) * mu_I
- pi_S and pi_I are each individually normalised probability distributions on exit;
  joint renormalisation (divide each by 2) is a convention for the combined ranking only
- compute v_s and v_i using joint A = a_s.sum() + a_u.sum()
