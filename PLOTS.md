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


#### Figure 2 - spectral ranking versus rank order
-- get v_i for S and I.
-- sort v_i and allocate a unique rank to the sorted values
-- facet with top panel S and bottom panel I
-- x-axis is rank (different for top and bottom)
-- y-axis is v_i score for S or I at index=x

## SKIP FOLLOWING Plots

### Figure 2 — Baseline ranking: source and institution prestige scores
Top-N sources and institutions ranked by prestige per work v(α) under the baseline
parameter set (t_x=5, F=A, τ_U=10, ρ=1, m=(1,0,0,0), χ=0.5, α=0.85).

### Figure 3 — Parameter sensitivity
Rank correlation (Spearman) heatmap across parameter combinations (α, χ, m, ρ).

### Figure 4 — Time series
Rank stability of top sources and institutions across t_x=1–5 symmetric windows.

### Figure 5 — Bipartite vs. full-joint comparison
Scatter or rank-shift plot comparing source rankings under m=(0,1,1,0) vs. m=(1,1,1,1).
