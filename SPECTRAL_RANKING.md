# Spectral Ranking Pipeline

## Purpose
#### Spectral ranking
Read pre-built edge lists from `edge_lists.duckdb`, assemble sparse block citation
matrices C(χ, m, ρ), row-normalise to H, run katz and katz_bipartite power iteration as required by parameters, and output prestige scores π and prestige-per-work v.
#### Parameter space exploration
-- Parameter exploration is in two stages.
-- Stage 1: compare to baseline [t_5, \rho = \bar{R}/R_i, F=A, \tau_u = 10, \alpha=0.85, SI/IS katz_biparite] effects of one-at-at time changes \rho = 1, \tau_u = 8, \alpha=0.5, F=E, F=B, C=SS/II and C=SS/SI/IS/II.
-- Stage 2: compare baseline to t_1...4 and t_6
#### Diagnostic displays
-- An early task is to explore ways to illustrate the findings. 
-- Work on this as soon as we have the baseline and one comparison \rho=1.

## Notation summary (from paper)

- **Units** P = S ∪ U, N = N_s + N_u. Sources occupy indices 1…N_s, institutions N_s+1…N.
- **b^S_i** ∈ {0,1}^N_s: source membership one-hot for work i.
- **b^U_i** ∈ R^N_u: institution weight vector, [b^U_i]_u = ω_{iu} (eq. 1).
- **ρ_i**: reference normalisation weight. ρ=1 → full count (ρ_i=1); ρ=0 → fixed count (ρ_i=R̄/R_i).
- **Raw blocks** (eq. 3): C_XY = Σ_{(i,j)∈ℜ} ρ_i b^X_i (b^Y_j)^T
- **Assembled matrix** (eq. 4): C(χ, m, ρ) with block scalings (1-χ)², χ(1-χ), χ²
  and binary mask m = (m_SS, m_SI, m_IS, m_II) ∈ {0,1}^4.
- **Row-stochastic H** = D_r^{-1} C, where D_r = diag(row sums). Dangling rows get μ.
- **Modified matrix** H̃ = αH + (1-α)μ1^T, μ_p = 1/N.
- **Katz ranking** (eq. 5): π = H̃^T π, i.e. π^(k+1) = H̃^T π^(k).
- **Prestige per work** (eq. 7): v_p = π_p / a_p, where a_p = work count of unit p.

---

## Block aggregation from the edge list

Each edge list table has one row per (citer_work, citer_inst, cited_work, cited_inst).
Raw blocks are computed by SQL aggregation with careful de-duplication to avoid
counting the institution cross-product where it should not appear.

### ρ weight
For ρ=0 (fixed count), first compute R̄ = AVG(R_i) over distinct citer works.
Then `rho_i = 1.0` if ρ=1, else `R_bar / R_i`. Apply as a multiplier in each query.

### C_SS — source × source
De-duplicate on (citer_work, cited_work): one reference contributes ρ_i once to
(citer_source, cited_source) regardless of how many institution combinations exist.

```sql
SELECT citer_source_idx, cited_source_idx, SUM(rho_i) AS weight
FROM (SELECT DISTINCT citer_work_idx, citer_source_idx, cited_work_idx,
                      cited_source_idx, rho_i FROM el)
GROUP BY citer_source_idx, cited_source_idx
```

### C_IS — institution × source
De-duplicate over cited_inst: for a given (citer_work, citer_inst, cited_work),
the contribution ρ_i × ω_{iu} to (citer_inst, cited_source) is counted once.

```sql
SELECT citer_inst_idx, cited_source_idx, SUM(rho_i * inst_weight) AS weight
FROM (SELECT DISTINCT citer_work_idx, citer_inst_idx, cited_work_idx,
                      cited_source_idx, rho_i, inst_weight FROM el)
GROUP BY citer_inst_idx, cited_source_idx
```

### C_SI — source × institution
De-duplicate over citer_inst: for a given (citer_work, cited_work, cited_inst),
the contribution ρ_i × ω_{jv} to (citer_source, cited_inst) is counted once.

```sql
SELECT citer_source_idx, cited_inst_idx, SUM(rho_i * cited_inst_weight) AS weight
FROM (SELECT DISTINCT citer_work_idx, citer_source_idx, cited_work_idx,
                      cited_inst_idx, rho_i, cited_inst_weight FROM el)
GROUP BY citer_source_idx, cited_inst_idx
```

### C_II — institution × institution
No de-duplication: every (citer_inst, cited_inst) combination across the reference
pair is a genuine cross-product contribution ρ_i × ω_{iu} × ω_{jv}.

```sql
SELECT citer_inst_idx, cited_inst_idx,
       SUM(rho_i * inst_weight * cited_inst_weight) AS weight
FROM el
GROUP BY citer_inst_idx, cited_inst_idx
```

---

## Three CSR constructions

The block mask m determines which blocks are built and how they are assembled.
χ is **absorbed by row-normalisation** whenever only one block or one symmetric
off-diagonal pair is active — it only has real effect in the full joint case where
multiple blocks contribute differently to each row sum.

### m = (1,0,0,0) — source-only
Build C_SS → row-normalise → H_SS of shape (N_s × N_s).
χ irrelevant. μ = (1/N_s) · **1**_{N_s}.
Output: π_S, v_S only.

### m = (0,0,0,1) — institution-only
Build C_II → row-normalise → H_II of shape (N_u × N_u).
χ irrelevant. μ = (1/N_u) · **1**_{N_u}. The prior is simply uniform over
institutions; ω plays no role in μ.
Output: π_I, v_I only.

### m = (0,1,1,0) — bipartite SI/IS
Build C_SI → row-normalise → H_SI of shape (N_s × N_u).
Build C_IS → row-normalise → H_IS of shape (N_u × N_s).
Keep as **two separate matrices**; do not assemble into one N×N block matrix.
χ irrelevant (common factor on both off-diagonal blocks cancels in normalisation).
μ_S = (1/N) · **1**_{N_s}, μ_I = (1/N) · **1**_{N_u} where N = N_s + N_u.
Use the Katz–Hubbell resolvent (see below). Output: π_S, π_I, v_S, v_I.

### m = (1,1,1,1) — full joint
Build all four raw blocks. Apply χ scaling before assembling into one N×N matrix:
```
C = bmat([[(1-χ)**2 * C_SS,  χ*(1-χ) * C_SI],
          [χ*(1-χ) * C_IS,   χ**2    * C_II]])
```
Row-normalise the assembled N×N matrix → H. μ = (1/N) · **1**_N.
Use standard Katz. Output: π_S, π_I, v_S, v_I.

### Unit index
Nodes with corpus works but zero intra-corpus references do not appear in any edge
list aggregation. They are dangling nodes that receive score only from the prior and
must still appear in the output. Add a `_units_t{tx}_{fx}_tau{tau_u}` table to
`edge_lists.duckdb` (from `build_edge_lists.py`) with columns `(unit_idx, unit_type,
a_p)` covering all retained sources and institutions. `build_csr.py` reads this table
to set matrix dimensions and the a_p denominators for v.

---

## Katz — standard iteration (SS, II, full joint)

H̃ is never formed explicitly. Dangling rows (zero rows of H, i.e. units with no
outgoing in-corpus citations) are handled by redistributing their probability mass
through the prior at each step:

```
dangling_idx ← {p : row p of H sums to 0}
π ← uniform μ
repeat:
    dangling_mass = π[dangling_idx].sum()
    π_new = α * (H.T @ π) + (α * dangling_mass + (1-α)) * μ
    π_new = np.maximum(π_new, 0.0)    # guard against FP underflow
    π_new /= π_new.sum()              # renormalise every step
    if ||π_new - π||_1 < tol: break
    π ← π_new
```

Renormalisation is mandatory every step. In exact arithmetic the map preserves the
L1 norm; in floating point, errors accumulate across hundreds of iterations without it.
Clipping before renormalising prevents underflow from propagating. tol = 1e-9,
max_iter = 500. `H.T @ π` is a standard CSR transpose-multiply.

---

## bipartite_resolvent() — SI/IS case only

H is block off-diagonal and periodic with period 2. The undamped fixed point lacks
a unique solution; with α < 1 naive power iteration converges but is governed by α
rather than |λ_2|. The resolvent eliminates institutions analytically.

Partitioning the fixed point π = α H^T π + (1−α) μ:

```
π_S = α H_IS^T π_I + (1−α) μ_S          (A)
π_I = α H_SI^T π_S + (1−α) μ_I          (B)
```

Substituting (B) into (A) and defining M_S = H_SI @ H_IS (N_s × N_s,
source–source one-mode projection through institutions):

```
(I − α² M_S^T) π_S = (1−α)(μ_S + α H_IS^T μ_I)
```

With N_s ≈ 1,600 this is a small system. Form the LHS explicitly as a dense or sparse
matrix and solve directly with `scipy.sparse.linalg.spsolve`. No inner iteration needed.

Recover institutions from (B): `π_I = α H_SI^T π_S + (1−α) μ_I`.

Concatenate [π_S; π_I], clip negatives, normalise jointly to sum to 1, then compute v.

μ_S and μ_I both use μ_p = 1/N, consistent with eq. (5) of the paper. The H_IS^T μ_I
term in the RHS transports institution prior mass back to sources, making the effective
boundary condition well-posed.

---

## Prestige per work

After convergence (eq. 7): v_p = π_p / a_p.
- Sources: a_s = census work count of source s (integer), from the units table.
- Institutions: a_u = Σ_i ω_{iu} (fractional work count), from the units table.

For units that appear in the edge list, a_p can be cross-checked against
`a_citer_source` / `a_citer_inst`. Isolated (dangling) nodes have a_p from the
units table only.

---

## Output schema

Store results in `WORKING/rankings.duckdb`. One table per parameter combination:

```
Table name: rk_t{tx}_{fx}_tau{tau_u}_rho{rho}_m{mstr}_chi{chi_int}_alpha{alpha_int}
  e.g. rk_t5_A_tau10_rho0_m0110_chi50_alpha85   (baseline)

Columns:
  unit_idx       BIGINT    -- source_idx or institution_idx
  unit_type      VARCHAR   -- 'S' or 'U'
  pi             DOUBLE    -- Katz ranking score (sums to 1 across all units in run)
  v              DOUBLE    -- prestige per work
  rank_pi        INTEGER   -- ordinal rank by pi descending
  rank_v         INTEGER   -- ordinal rank by v descending
  a_p            DOUBLE    -- work count (denominator of v)
```

Maintain a `_catalog` table: one row per run with all parameter values, N_s, N_u,
iterations to convergence, final L1 norm, and timestamp.

---

## Parameter sweep

### Baseline
t_x=5, F=A, τ_U=10, ρ=R̄/R_i (fixed count), m=(0,1,1,0), α=0.85.
χ is not a free parameter for the bipartite case (no effect after normalisation).

### Stage 1 — one-at-a-time from baseline
Each run changes one parameter; all others held at baseline:

| Variant | Change |
|---|---|
| ρ=1 | full reference count |
| τ_U=8 | relaxed institution threshold |
| α=0.5 | lower damping |
| F=E | economics sources only |
| F=B | business sources only |
| m=(1,0,0,0) | source-only SS |
| m=(0,0,0,1) | institution-only II |
| m=(1,1,1,1), χ=0.5 | full joint |

Run baseline and ρ=1 first. Begin diagnostic display work as soon as these two runs
are available.

### Stage 2 — time series
Hold all parameters at baseline values; sweep t_x ∈ {1, 2, 3, 4, 6}.
t_x=7 (reference spectroscopy) is qualitatively different and treated separately.

### Stage 3 — sensitivity grid (later)
χ ∈ {0.25, 0.75, 1.00} for m=(1,1,1,1); α ∈ {0.75, 0.95}; t_x=7.

---

## Script structure

```
spectral_ranking/              — pipeline: edge_lists.duckdb → rankings.duckdb
  build_csr.py                 -- reads edge list + unit index for one corpus case;
                                  returns raw CSR blocks (SS, SI, IS, II) and unit
                                  index arrays; applies ρ weighting; no assembly
  katz_ranker.py               -- assembles H per m and χ; routes to katz() or
                                  bipartite_resolvent(); returns (π, v, iters, norm)
  run_rankings.py              -- parameter driver; calls build_csr + katz_ranker;
                                  writes results and _catalog to rankings.duckdb

spectral_results_analysis/     — analysis: rankings.duckdb → plots, tables, paper
  diagnostics.py               -- rank correlation tables, sensitivity summaries
  plots.py                     -- all result figures (time series, sensitivity, etc.)
  tables.py                    -- LaTeX ranking tables for paper sections 4–5
```

`rankings.duckdb` is the interface between the two folders. The pipeline writes it
and has no knowledge of how results are used. The analysis folder reads it and has
no knowledge of how rankings were computed.

`build_csr.py` and `katz_ranker.py` are pure functions with no I/O.
All DuckDB access is in `run_rankings.py` and the analysis scripts.
