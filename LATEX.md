# LaTeX
# EconomicsBusiness project — spectral_ranking_latex/

## Location
`spectral_ranking_latex/` — multi-file project, master = `main.tex`, sections in `sections/`.
Bibliography fed by Zotero (`MyLibrary.bib`).

## Current paper status (April 2026)

| Section | Title | Status |
|---|---|---|
| 1 | Introduction | Minor edits |
| 2 | Journal–institution citation networks | Minor edits |
| 3 | Corpus selection | Minor edits |
| 4 | Results | Large revision in this document |
| 5 | Discussion | Placeholder |
| 6 | Prospects | Placeholder |
| Supplement | — | Placeholder |

`main.tex` compiles cleanly.

---

## Convention reminder

$C_{ij}$ = attention from $i$ (citing, row) to $j$ (cited, column). Row sum = references given.
Prefer "reference" over "citation" in prose unless the citation direction is being
emphasised. This is the transpose of the economist's convention.

---

## Replace most of the existing 04 tex document with the following plan. Where there is conflict take the plan not 04.

## Section 04_results.tex
- Opening paragraph explains the layout of this section
 - Explain parameters: summary here. earlier partial summaries to be made coherent here only
 - Corpus selection: t_x, F in EBA, \tau_s/i, after specified \chi* is available
 - Mode selection: 
  - 1000 and 0001 as analogues of journal eigenvector methods which are common and institution wich are not
  - 1111 is the analgue of these when all paths between units are present
  - 0110 is the bipartite case that is a main focus of this paper
 - parameters to explore 
  - \alpha to close explanatory loops
  - \rho to test fractionation
  - \tau_s/i = 40 to test cutoff effect
  - t_x: t_5 is the baseline, t_1-t_5 give a temploral picture. Drop ideas and mentions about t_6 and t_7.
 - build a table that captures all of this including the t_i

 - Plan a table to set this out clearly

--- 

## Baseline

- t_x=5, F=A, τ_U=20, τ_S=20, ρ=0 (fixed count, R̄/R_i), m=0110, minimal power iterations (α=1, \mu=0, \chi irrelevant should be equaton (5) of the document)


## Phase 1
- Compare baseline with F=E, F=B, F=EB for sources and institutions.
- make text align with fig_2 code and insert fig_2 here
- mention that communities potentially effect spctral ranking and that the second eigenpair can be used to explore this.
- present second eigenpair 'theory' consistent with fig_4 pl and include fig_4 at this point

## Phase 2
- Compare baseline with s (1000/1111) and i(0001/1111) as in current fig_3. Make text align with fig_3 code and insert fig_3 here.
- add E and B symbols to i in fig_3

- Rewrite these points for the \alpha=1 \mu=0 0110 baseline

1. The Katz–Hubbell resolvent has the expansion (in terms of the left
   eigenpairs (λ_i, u_i) and right eigenpairs (λ_i, v_i) of H):
     π = (1−α) Σ_i [v_i^T μ / (1−α λ_i)] u_i
   The i=1 term is π itself (λ_1=1). The i=2 term is the dominant
   correction, amplified by 1/(1−α λ_2). This amplification factor is
   the same for all four modes under the √α convention.

2. The second left eigenvector φ₂ of H^T (i.e., H^T φ₂ = λ₂ φ₂) encodes
   the dominant partition of units beyond the uniform ranking. For the SS
   mode it is expected to separate Economics from Business journals. For II
   it separates institution communities (more diffuse). For bipartite, φ₂
   is the second eigenvector of M_S^T, projected back to institutions via
   φ₂^I = H_SI^T φ₂^S.

3. The spectral gap g = 1−λ₂ quantifies how rapidly community information
   decays. A small gap means community membership strongly influences
   rankings; a large gap means the ranking is relatively community-neutral.

4. Sign convention: φ₂ is normalised so that the unit with the largest a_p
   (most works) has φ₂ > 0. This is self-contained and requires no external
   field labels.

5. Computation: use scipy.sparse.linalg.eigs(H^T, k=2, which='LM'), take
   real parts (warn if |Im(λ₂)|/|λ₂| > 0.01). Do not symmetrise H — that
   discards directed structure and loses the connection to π.

## Phase 3
- Explore the nature of changes from baseline for \alpha = 0.1 and 0.5 with \chi^* and \mu consistent with the N_s/i.
- Explore effect of \rho 
- Eplore effect of \tau - both to 40

 - Explanation of \chi^* For equal expected prestige per unit across both types, the 
    flow per unit must be equalised: (1−χ)/N_s = χ/N_u Solving: χ* = N_u/(N_s+N_u). 
    With N_s ≈ 1,600 and N_u ≈ 2,600 (baseline F=A, τ_U=10, t_x=5), χ* ≈ 0.619 (numbers are computed at run time). 
    This is the dimensionally neutral calibration examined here

## Phase 4
- Time dependence
- Compare t_1 ... t_4 to t_5 baseline in all cases
- We don't know what this looks like yet. It may raise non-primitivity.
