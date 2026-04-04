# PLOTS.md — EconomicsBusiness Project

## Conventions
- Prefer seaborn OO interface; fall back to matplotlib where seaborn is insufficient.
- Persist to disk in a LaTeX-compatible format (PDF or PGF).
- py files in spectral_results_analysis
- each py file makes one fig - work on fig_2.py

## Plot inventory

### Figure 1 — Institution retention curve 
**Script**: `prepare_data/plot_maker.py`
**Files**: `plots/plot1_institution_elbow.pdf`, `plots/plot1_institution_elbow_latex.pdf`
**In paper**: Fig. 1, Section 3
**Status**: Complete
**Description**: Long-tail elbow plot of % works retained vs. annual work-count threshold τ_U,
for the baseline corpus (F=A, t_x=5, 2020–2024). Annotations show institution counts at
key thresholds. Selected cut: τ_U=20, ~1,734 institutions, ~85% works retained.

### Figure 2 — Spectral ranking by field (E/B/All)
**Script**: `spectral_results_analysis/fig_2.py`
**Status**: In development
**Description**: Spectral ranking versus rank order with effects of E/B/All field subsets.
Faceted plot with source and institution rankings showing sensitivity to field selection.

### Figure 3 — Spectral ranking by network mode 
**Script**: `spectral_results_analysis/fig_3.py`
**Status**: In development  
**Description**: Spectral ranking versus rank order with effects of network modes
(m=SS-only/II-only; full-joint; bipartite). Baseline bipartite in black, alternatives
as colored markers.

### Figure 4 — Community structure analysis
**Script**: `spectral_results_analysis/community.py`
**Files**: `plots/fig4_community_eigenpair.pdf`
**Status**: Complete
**In paper**: Section 4 (results), community structure subsection
**Description**: Four-panel figure showing the second eigenvector φ₂ and spectral
gap across network modes. All eigenpairs computed from the directed ranking
operator H^T using eigs(H.T, k=2, which='LM') with real parts taken.
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
**Script**: `spectral_results_analysis/fig_5.py`
**Files**: `plots/fig_5.pdf`, `plots/fig_5_latex.pdf`
**Status**: In development
**Description**: Phase 2 sensitivity: rank comparison across τ=40 and ρ=1 variants vs baseline.
Two-panel layout (sources top, institutions bottom).

### Figure 6 — Time series
**Script**: `spectral_results_analysis/fig_6.py`
**Files**: `plots/fig_6.pdf`, `plots/fig_6_latex.pdf`
**Status**: Created
**Description**: Time-series comparison of baseline (2020–24) vs t1 (2000–04), t2 (2005–09),
t3 (2010–14), t4 (2015–19). Two-panel layout (sources top, institutions bottom). x-axis
locked to baseline rank. Colours: purple/blue/green/red for t1–t4, black for baseline.
Running mean curves through scatter points (log-space rolling mean, window=max(30, n//100)).
Units absent from an earlier-period corpus are omitted from that series.

### Figure 7 — Bootstrap uncertainty
**Script**: `spectral_results_analysis/fig_7.py` (to be written; spec in `BOOTSTRAP.md`)
**Status**: Planned
**Description**: Bootstrap uncertainty bands around baseline prestige-per-work rankings.
Reads `WORKING/bootstrap/v_s_boot.npy`, `v_u_boot.npy`, `meta.json`.

