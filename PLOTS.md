# PLOTS.md — EconomicsBusiness Project

## Conventions
- Prefer seaborn OO interface; fall back to matplotlib where seaborn is insufficient.
- Persist to disk in a LaTeX-compatible format (PDF or PGF).
- py files in spectral_results_analysis
- each py file makes one fig - work on fig_2.py

## Plot inventory

### Figure 1 — Institution retention curve (SKIP)
**Script**: `prepare_data/plot_maker.py`
**Files**: `plots/plot1_institution_elbow.pdf`, `plots/plot1_institution_elbow_latex.pdf`
**In paper**: Fig. 1, Section 3
**Description**: Long-tail elbow plot of % works retained vs. annual work-count threshold τ_U,
for the baseline corpus (F=A, t_x=5, 2020–2024). Annotations show institution counts at
key thresholds. Selected cut: τ_U=10, ~1,900 institutions, ~75% works retained.


#### Figure 2 - spectral ranking versus rank order with effects of E/B/All
-- get v_i for S and I.
-- sort v_i and allocate a unique rank to the sorted values
-- facet with top panel S and bottom panel I
-- x-axis is rank (different for top and bottom)
-- y-axis is v_i score for S or I at index=x

#### Figure 3 - spectral ranking versus rank order with effects of m=1000/0001; 1111; 0110
-- get v_i for S and I.
-- sort v_i and allocate a unique rank to the sorted values
-- facet with top panel S and bottom panel I
-- x-axis is rank (different for top and bottom)
-- y-axis is v_i score for S or I at index=x
-- plot m=0110 in black as baseline
-- plot m=1000 (for S) and m=0001 (for I) as visible red X
-- plot m=1111 as visible green X

## SKIP ALL THE FOLLOWING

### Figure 4 — Second eigenpair community analysis
**Script**: `spectral_results_analysis/community.py`
**Files**: `plots/fig3_community_eigenpair.pdf`
**In paper**: Section 4 (results), community structure subsection
**Description**: Four-panel figure showing the second eigenvector φ₂ and spectral
gap across network modes. All eigenpairs are computed from the directed ranking
operator H^T (or M_S^T for bipartite) using eigs(H.T, k=2, which='LM') with real
parts taken. Sign convention: the unit with the largest a_p (most works) has phi2 > 0.

Panel A (top left) — SS source community:
  x-axis: sources sorted by φ₂ value (rank 1 = most negative)
  y-axis: φ₂ value
  colour: F label (E=blue, B=orange, unlabelled/bridge=grey)
  Expected: clear sign split between E and B journals; finance journals near zero.

Panel B (top right) — II institution community:
  x-axis: institutions sorted by φ₂ value
  y-axis: φ₂ value
  colour: dominant-field label if available, else grey
  Expected: noisier, more institutions near zero; weaker separation than Panel A.

Panel C (bottom left) — Spectral gap comparison:
  Horizontal bar chart with one bar per case:
    SS, bipartite (M_S), II, full χ=0.5, full χ=χ*
  x-axis: spectral gap g = 1 − Re(λ₂)
  Larger gap = faster community mixing = weaker community effects on ranking.
  Annotate bars with λ₂ value and amplification factor 1/(1−α·λ₂) for all modes.
  Under the √α convention, the bipartite resolvent uses α as its round-trip
  attenuation, so the amplification formula is identical across all modes.

Panel D (bottom right) — Full joint φ₂ at χ=0.5 vs χ*:
  Scatter: φ₂ at χ=0.5 (x) vs φ₂ at χ=χ* (y), one point per source.
  colour: F label.
  Shows whether dimensional rebalancing shifts community assignments.
  Identity line for reference.

**Data sources**: edge_lists.duckdb (CSR blocks via build_csr), rankings.duckdb
(π, v, a_p for unit labels), data/ (F-label lookup by source_idx).


### Figure 5 — Parameter sensitivity
Rank correlation (Spearman) heatmap across parameter combinations (α, χ, m, ρ).

### Figure 6 — Time series
Rank stability of top sources and institutions across t_x=1–5 symmetric windows.

