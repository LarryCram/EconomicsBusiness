# COMMUNITY_ANALYSIS.md — Parameter Space and Community Structure Analysis — **COMPLETED**

## Purpose — **COMPLETED**

This document specified the analytical work plan agreed in the session of 2026-03-28,
covering (a) the step-wise comparison of community structure across network modes
(SS → II → SI/IS → full joint) and (b) the use of the second eigenpair of the
ranking operator as the diagnostic tool within the existing spectral ranking framework.
All analytical objectives have been successfully implemented.

---

## Analytical objective

The paper's dual-ranking framework (sources and institutions, modes SS/II/SI–IS/full
joint) induces different degrees of disciplinary community mixing. The Economics (E)
and Business (B) communities are the primary test case. The goal is to show, using
objects already present in the spectral ranking framework, that:

- SS preserves E/B community separation most strongly.
- II dissolves it because institutions co-locate E and B.
- The SI/IS bipartite walk is intermediate, with the institution step acting as a
  partial mixer.
- The full joint case interpolates, with χ controlling the blend.
- The second eigenpair of the ranking operator, not a separate Laplacian, is the
  diagnostic tool throughout. It is commensurable with π because it lives in the
  same directed space.

---

## Key decisions from the session

**Second eigenpair is analysis-only.** It is not computed in the main pipeline
(run_rankings.py / katz_ranker.py) and is not stored in rankings.duckdb. It is
computed on-the-fly in the analysis layer, reading CSR blocks directly via build_csr.

**Eigenvalue computation.** Use `scipy.sparse.linalg.eigs(H.T, k=2, which='LM')`
for all non-symmetric matrices (SS, II, full joint). Take the real part of both
λ₂ and φ₂. Emit a warning if `abs(imag(λ₂)) / abs(λ₂) > 0.01`. Do not
symmetrise — symmetrisation discards the asymmetry between citing and cited roles
that the directed H encodes, and the real part of the directed second eigenpair is
directly interpretable as the dominant correction to the Katz ranking.

**Sign convention for φ₂.** Fix the sign so that the element of φ₂ corresponding
to the unit with the largest absolute φ₂ value is positive. This is self-contained
(requires no external labels) and reproducible.

**F-label lookup.** Colouring φ₂ by field (E/B) requires a mapping from source_idx
to F ∈ {E, B, A}. This should be read from a reference table in data/ or from the
source list used in corpus construction. Note: source identifiers were converted from
source_id (OpenAlex URI) to source_idx by stripping the "https://openalex.org/S"
prefix. Verify that source_idx values in rankings.duckdb join cleanly to any
F-label reference table before colouring plots. If the join fails, check the ID
stripping logic in prepare_data/.

**χ* calibration.** χ* = N_u / (N_s + N_u) is the dimensionally neutral mixing
weight at which the citation-flow contribution per unit is equal across sources and
institutions. It cannot be pre-computed because N_u depends on τ_U and F. It must
be computed dynamically in run_rankings.py by reading n_s and n_u from the units
table before building the Stage 1 parameter schedule, then added as an additional
RunParams entry for m=(1,1,1,1).

**τ_U policy.** τ_U=10 is the primary parameter. τ_U=8 is a Stage 1 sensitivity
check only and does not alter the primary analysis.

---

## Step-wise analysis plan

### Step 1 — SS case: E/B community structure in the source network

**Network**: m=(1,0,0,0), F=A, τ_U=10, ρ=0, α=0.85 (baseline SS run).
**Also run**: F=E and F=B restricted corpora (already in Stage 1).

**Computation**:
1. Load C_SS from build_csr (call with m=(1,0,0,0)).
2. Row-normalise to H_SS via _row_normalise.
3. Compute second eigenpair: `vals, vecs = eigs(H_SS.T, k=2, which='LM')`.
4. Take real parts; apply sign convention; record λ₂ and φ₂ (length N_s).
5. Record spectral gap g_SS = 1 − Re(λ₂) and amplification A_SS = 1/(1 − α·Re(λ₂)).

**Expected result**: φ₂ has opposite signs on E and B journal sets, near zero on
finance bridge journals (JF, JFE, RFS). The sign partition of φ₂ is the spectral
fingerprint of the E/B community, derived entirely from the ranking operator.

**Comparison within Step 1**: Rank-shift plot between SS F=A and SS F=E (and F=B).
Finance journals drop substantially in rank under F=E restriction. These are the
bridge journals near φ₂=0.

### Step 2 — II case: community diffusion in the institution network

**Network**: m=(0,0,0,1), F=A, τ_U=10, ρ=0, α=0.85.

**Computation**:
1. Load C_II; row-normalise to H_II.
2. Compute `eigs(H_II.T, k=2, which='LM')`; take real parts.
3. Record λ₂, φ₂ (length N_u), g_II = 1 − Re(λ₂), A_II = 1/(1 − α·Re(λ₂)).
4. Label institutions by dominant field where possible (see F-label note above).

**Expected result**: g_II > g_SS (larger spectral gap, faster mixing). φ₂ is
noisier with more institutions near zero, because multi-faculty universities
simultaneously host E and B researchers, making the II network more integrated.

### Step 3 — SI/IS bipartite case: institution-mediated community mixing

**Network**: m=(0,1,1,0), F=A, τ_U=10, ρ=0, α=0.85 (baseline).

**Computation**:
1. Load C_SI and C_IS; row-normalise to H_SI and H_IS.
2. Form one-mode projection M_S = H_SI @ H_IS  (N_s × N_s).
   Note: M_S is already computed inside bipartite_resolvent() but is not returned.
   For the analysis script, recompute it independently from the CSR blocks.
3. Compute `eigs(M_S.T, k=2, which='LM')`; take real parts.
4. Record λ₂^{M_S}, φ₂^S (length N_s).
5. Recover institution community vector: φ₂^I = H_SI.T @ φ₂^S, then normalise to
   unit Euclidean norm. This is the institution analogue, parallel to how π_I is
   recovered from π_S in bipartite_resolvent.
6. Effective damping is α² not α. Record:
   - g_bip = 1 − Re(λ₂^{M_S})
   - A_bip = 1/(1 − α²·Re(λ₂^{M_S}))

**Key comparison**: λ₂^{M_S} vs λ₂^{H_SS} on the same N_s-dimensional source
space. The institution-mediated projection M_S blurs community boundaries relative
to direct journal-to-journal citation H_SS. Even if λ₂^{M_S} ≈ λ₂^{H_SS}, the
effective community amplification is smaller in the bipartite case because α² < α.

### Step 4 — Full joint case at χ=0.5 and χ*

**Networks**: m=(1,1,1,1) at χ=0.5 and χ* = N_u/(N_s+N_u) (computed dynamically).

**Computation**:
1. Assemble full N×N matrix C per eq.(4) at each χ; row-normalise to H.
2. Compute `eigs(H.T, k=2, which='LM')`; take real parts.
3. The second eigenvector φ₂ is length N = N_s + N_u; partition into φ₂^S and φ₂^I.
4. Record λ₂, g = 1 − Re(λ₂), A = 1/(1 − α·Re(λ₂)).
5. At χ=0: recovers SS spectrum. At χ=1: recovers II spectrum. At intermediate χ:
   the second eigenvector may encode E/B community or S/I mode separation or a mix.
   At χ*, the flow-prior balance is dimensionally neutral (equal prestige per unit
   in expectation).

**Note on φ₂ interpretation in full joint**: the second eigenvector can reflect
either the E/B community split, the S/I mode split, or both. Examine the loading
of φ₂ on the E/B label (for source subvector) and on the S vs I unit type (for
the full vector) to determine which dominates at each χ.

### Step 5 — Summary comparison across cases

Assemble the spectral gap and amplification series:

| Case          | λ₂     | gap    | eff. damping | amplification        |
|---------------|--------|--------|--------------|----------------------|
| SS            | λ₂^SS  | g_SS   | α            | 1/(1 − α·λ₂^SS)     |
| bipartite M_S | λ₂^bip | g_bip  | α²           | 1/(1 − α²·λ₂^bip)   |
| II            | λ₂^II  | g_II   | α            | 1/(1 − α·λ₂^II)     |
| full χ=0.5    | λ₂^f05 | g_f05  | α            | 1/(1 − α·λ₂^f05)    |
| full χ=χ*     | λ₂^f*  | g_f*   | α            | 1/(1 − α·λ₂^f*)     |

This table is the quantitative spine of the community analysis section of the paper.

---

## Code changes required

### New file: `spectral_results_analysis/community.py`

Purpose: compute the second eigenpair for each network mode and produce the
community analysis plots and summary table. Reads directly from edge_lists.duckdb
(via build_csr) and rankings.duckdb (for π, v, a_p). No writes to rankings.duckdb.

Structure:

```
community.py
  load_blocks(db, tx, fx, tau_u, rho, m)
      -> calls build_csr; returns CSRData

  second_eigenpair_unipartite(C, alpha)
      -> row-normalise C to H
      -> eigs(H.T, k=2, which='LM')
      -> check imag warning threshold 0.01
      -> apply sign convention (max |φ₂| element positive)
      -> return (lambda2_real, phi2_real, gap, amplification)

  second_eigenpair_bipartite(C_SI, C_IS, alpha)
      -> row-normalise to H_SI, H_IS
      -> M_S = H_SI @ H_IS
      -> eigs(M_S.T, k=2, which='LM')
      -> check imag warning
      -> apply sign convention
      -> phi2_I = H_SI.T @ phi2_S, normalised
      -> effective damping is alpha**2
      -> return (lambda2_real, phi2_S, phi2_I, gap, amplification)

  load_field_labels(data_dir)
      -> read source F-label table from data/
      -> return dict {source_idx: 'E'|'B'|'A'}
      -> warn if join to rankings source_idx is incomplete (see ID note below)

  run_community_analysis(paths)
      -> top-level driver; calls above for SS, II, bipartite, full χ=0.5, full χ*
      -> prints summary table of (case, λ₂, gap, amplification)
      -> calls plot functions

  main()  — argparse entry point
```

**ID note for load_field_labels**: source_idx values in the database were derived
from OpenAlex source URIs by stripping the "https://openalex.org/S" prefix to give
integer IDs. Verify the F-label reference file in data/ uses the same integer form.
If not, apply the same stripping transformation before joining.

### Modified file: `spectral_ranking/run_rankings.py`

Add χ* as a dynamic Stage 1 variant for m=(1,1,1,1).

In `main()`, after opening the edge-list database and before building the schedule,
read n_s and n_u from the units table for the baseline corpus
(`_units_20242024_A_tauU20_tauS20`):

```python
units = el_db.execute(
    "SELECT unit_type, COUNT(*) AS n FROM _units_20242024_A_tauU20_tauS20 GROUP BY unit_type"
).fetchdf()
n_s_base = int(units.loc[units.unit_type=='S', 'n'].iloc[0])
n_u_base = int(units.loc[units.unit_type=='U', 'n'].iloc[0])
chi_star = n_u_base / (n_s_base + n_u_base)
```

Then add to params.csv as `full-joint-chi-star` row with `chi=-1` (resolved at runtime).

χ* will vary with corpus but for the primary baseline corpus (run_code=20242024, A, tau20) it is
fixed for the session. Log its value at run time.

---

## Data dependency note

The step-wise analysis above uses runs defined in params.csv:
- SS-only:      rk_20242024_A_tauU20_tauS20_rho0_m1000_chi50_alpha100
- II-only:      rk_20242024_A_tauU20_tauS20_rho0_m0001_chi50_alpha100
- baseline SI/IS: rk_20242024_A_tauU20_tauS20_rho0_m0110_chi50_alpha100
- full-joint:   rk_20242024_A_tauU20_tauS20_rho0_m1111_chi50_alpha100
- full-joint χ*: rk_20242024_A_tauU20_tauS20_rho0_m1111_chiSTAR_alpha100
- F=E:          rk_20242024_E_tauU20_tauS20_rho0_m0110_chi50_alpha100
- F=B:          rk_20242024_B_tauU20_tauS20_rho0_m0110_chi50_alpha100

The community.py script needs the edge-list database (for build_csr) and
rankings.duckdb (for π and v). It does not modify either.

---

## Open question (pending)

**Sign convention confirmation**: the convention adopted above (max |φ₂| element
positive) is self-contained and reproducible. If cross-run comparison of φ₂ signs
is needed (e.g. to track the same community across F=E/F=A/F=B runs), a label-based
convention (mean φ₂ over F=E sources positive) is more semantically meaningful but
requires the F-label join. Confirm which is preferred once F-label availability
is verified.

**M_S return from bipartite_resolvent**: the community analysis recomputes M_S
independently (does not require a change to katz_ranker.py). If M_S is needed
elsewhere, consider adding it to RankResult, but this is not required for the
current plan.
