# LaTeX

## Location
`spectral_ranking_latex/` — multi-file project, master = `main.tex`, sections in `sections/`.
Bibliography fed by Zotero (`MyLibrary.bib`).

## Current paper status (March 2026)

| Section | Title | Status |
|---|---|---|
| 1 | Introduction | Complete |
| 2 | Journal–institution citation networks | Complete |
| 3 | Corpus selection | Complete |
| 4 | Results | Placeholder — needs ranking output |
| 5 | Discussion | Placeholder |
| 6 | Prospects | Placeholder |
| Supplement | — | Placeholder |

`main.tex` compiles cleanly.

## Pending section work

### Section 2 — bipartite special case
The SI/IS-only case (block mask m = (0,1,1,0)) makes H block off-diagonal and H² block
diagonal. The bipartite structure means the undamped equation (6) need not have a unique
solution; the Katz–Hubbell resolvent (eq. 10–11) restores primitivity for any α ∈ (0,1).
A compact subsection is needed that:
- Defines the one-mode projections M_S = H_SI H_IS and M_I = H_IS H_SI
- States the Katz–Hubbell resolvent for sources (eq. 10): (I − α² M_S^T)π_S = (1−α)(μ_S + α H_IS^T μ_I)
- Shows institution recovery (eq. 11): π_I = α H_SI^T π_S + (1−α)μ_I
- Notes effective damping α² and prior transport

### Section 4 — Results
Structure (from paper):
- Baseline model (t_x=5, F=A, τ_U=10, ρ=1, m=(1,0,0,0), χ=0.5, α=0.85)
- Parameter sensitivity: α, χ, m, ρ sweeps
- Bootstraps

All content depends on spectral ranking pipeline completing first.

## Convention reminder
C_ij = attention from i (citing, row) to j (cited, column). Row sum = references given.
Prefer "reference" over "citation" in prose unless the citation direction is being
emphasised. This is the transpose of the economist's convention.
