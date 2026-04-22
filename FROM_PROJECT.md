# Session summary — spectral ranking paper

## Conceptual threads resolved

**Notation alignment.** The current draft uses $\pi$ (ranking probability, Geller's stationary distribution of row-stochastic $\mathbf{H}$), $v = (A/a_p)\pi$ (influence per publication, the AIS / PHV quantity). No intermediate $w$ vector. $\pi$ is the Markov output; $v$ differs from $\pi$ by the inverse work-count scaling.

**Perplexity as a matrix-intrinsic statistic.** Row perplexity $m_i = \exp(-\sum_j H_{ij}\log H_{ij})$ is the right object, unweighted across rows for cross-kernel comparison. $\pi$-weighting imports the output back into the input description. The code already computes both; use the uniform-weighted versions as the primary architectural statistics.

**Predicting the shape of $\pi$.** Primary driver is heterogeneity of column sums $\{s_j^{\text{col}} = \sum_i H_{ij}\}$ — the row-stochastic "impact factor." High Gini → steep $\pi$. Row perplexity and modularity are second-order modifiers: high row perplexity washes out column asymmetry; asymmetric modularity amplifies it. Column sums are close kin to citing-side-normalised impact factors (Zitt-Small style) — a clean bridge to the pre-spectral literature.

**Mechanism for cross-mask differences.** $M_S = H_{SI}H_{IS}$ and $M_I = H_{IS}H_{SI}$ are row-stochastic operators on each layer. The Gini of $\{s_j^{\text{col}}(M_S)\}$ vs $\{s_j^{\text{col}}(H_{SS})\}$ predicts whether bipartite routing flattens or steepens source rankings. Mean row perplexity of the cross-layer blocks $H_{SI}$, $H_{IS}$ is the mechanism — how broadly mass fans out at each leg of the round trip.

**Bibliometric positioning.** West-Bergstrom-Jensen (author-level Eigenfactor) is the direct methodological precedent for the bipartite round-trip construction. CWTS Leiden Ranking is composite-indicator-based, non-recursive — cited as contrast rather than precedent. The institution-level spectral ranking with a bipartite decomposition is genuinely novel.

**Functional asymmetry of $S$ and $I$.** Already flagged in the introduction's first paragraph. Do not re-argue in methods or results. A single phrase at the first appearance of the four-block decomposition ("exposing within-layer and cross-layer contributions") is sufficient bridging. Let the empirical asymmetry in $\pi^S$ vs $\pi^B$ vs $\pi^I$ carry the substantive claim.

**Core-vs-bulk hypothesis.** Cross-mask differences may be driven by ~10 journals and ~50 institutions with high $v$, with bulk positions largely determined by mass-conservation residuals and volume scaling. Testable via residualise-and-re-scatter diagnostic. Not yet settled; will emerge from the analysis.

**"PageRank blurs the tail" not applicable.** The concern applies to binary undirected graphs with light-tailed degree distributions. The current setup is weighted, directed, heavy-tailed, and compositionally structured. Bulk compression is a substantive finding, not an algorithmic artefact. Time-series persistence supports this.

**Error modelling.** Moving beyond bootstrap to explicit error models. Reference-matching error rate ~0.1 (mostly within-subfield confusion). Author-level errors ~0.05 (ORCID failures) but institutional attribution is robust (name-string-based). Residual random work-to-institution misallocation is the target, probably 1–2%. Journal-country-bucket-respecting perturbation is worth running to pre-empt structured-error objections.

## Presentation sequence

| Section | Content | Baseline operator | Key indicators |
|---|---|---|---|
| 1 | Introduction — functional asymmetry flagged in prose | — | — |
| 2 | Methods — four-block decomposition, resolvent, $\pi$-to-$v$ scaling | — | — |
| 3 | Processing — corpus, filters, block assembly | — | — |
| 4.1 | Masks: $S$, $I$, $B$, then $J$ on full corpus | $B = (0110)$ | See Indicators block below |
| 4.2 | Subfields: E-only and B-only under same mask progression | $B = (0110)$ per subfield | All indicators computed per subfield |
| 4.3 | Time series: five 5-year windows under baseline mask | $B = (0110)$ at each window | Shape-stability + persistence indicators |
| 5 | Discussion — structural coupling claim, core-vs-bulk framing, bibliometric contrast | — | — |
| 6 | Error model — work-to-institution misallocation at $\epsilon \in \{0.02, 0.05, 0.1\}$ | applied to baseline | Perturbed versions of headline indicators |

$J = (1111)$ reported at end of §4.1 as the generalisation confirming $B$'s findings survive when within-layer flow is active. $S$ and $I$ introduced first as the conventional within-layer operators.

## Indicators at each step

### Per-mask architectural indicators (pure matrix properties)

| Indicator | Kernel | Meaning |
|---|---|---|
| $\text{Gini}\{s_j^{\text{col}}(H)\}$ | $H_{SS}$, $H_{II}$, $M_S$, $M_I$, $H_J$ | Primary predictor of $\pi$-shape; impact-factor-like |
| $\bar{m}^{\text{row}}(H)$ unweighted | same | Mean effective branching (fan-out) |
| $\bar{m}^{\text{row}}(H)$ stationary-weighted | same | For comparison only; shows whether $\pi$-weighting matters |
| Variance or IQR of $\{m_i^{\text{row}}\}$ | same | Heterogeneity of branching; bridges vs broadcasters |
| $\bar{m}^{\text{col}}(H)$ (column perplexity) | same | Breadth of inflow sources; second-order refinement |
| Newman-Girvan $Q$ | $H_{SS}$, $H_{II}$ | Modularity; amplifies column asymmetry when asymmetric |
| $\bar{m}(H_{SI})$, $\bar{m}(H_{IS})$ | cross-layer blocks | Bridge-width for mechanism discussion |

### Per-mask spectral outputs

| Indicator | Meaning |
|---|---|
| $\pi_p$, $v_p$ per unit | Raw ranking outputs |
| Cross-mask Spearman $\rho(v^B, v^S)$, $\rho(v^B, v^I)$ | Degree of ranking agreement |
| Cross-mask slope $d \log v^B / d \log v^{S,I}$ | Trend shift / compression-stretching |
| $N_{\text{core}}$ from residualise-and-re-scatter | Core size driving cross-mask differences |

### Subfield-specific (E, B, combined)

Same indicators as above, with the addition of:

| Indicator | Meaning |
|---|---|
| $N_{\text{core}}^E$, $N_{\text{core}}^B$ | Subfield-specific core size |
| Core overlap between E, B, combined | Whether combined core = union of subfield cores |
| $\bar{m}^I / \bar{m}^S$ ratio per subfield | Branching asymmetry across subfields (was 3.5 in aggregate) |

### Time-series (five windows)

| Indicator | Meaning |
|---|---|
| Gini of $s^{\text{col}}$ per window | Shape stability at kernel level |
| $\bar{m}$ per window | Mixing stability |
| $\rho(v^B_{t}, v^B_{t+1})$ for adjacent windows | Individual persistence |
| Fraction of units with $v > 1$ per window | Categorical stability |

### Error-model outputs

| Indicator | Meaning |
|---|---|
| Distribution of each headline indicator under $\epsilon = \{0.02, 0.05, 0.10\}$ | Uncertainty band for each claim |
| Ratio of error-band width to cross-mask differences | Robustness statement |

## Tabular layout for code refinement

### Per-subfield × per-mask × per-window indicator table

This is the master output table the analysis pipeline should produce. One row per `(subfield, mask, window)` combination.

| Column | Type | Source |
|---|---|---|
| `subfield` | str | {`ALL`, `E`, `B`} |
| `mask` | str | {`SS`, `II`, `B`, `J`} |
| `window` | str | {`2000-04`, ..., `2020-24`} |
| `n_s` | int | from `_catalog` |
| `n_u` | int | from `_catalog` |
| `lam1`, `lam2` | float | from `_catalog` |
| `gini_s_col_SS` | float | Gini of col sums of $H_{SS}$ restricted to this corpus |
| `gini_s_col_II` | float | Gini of col sums of $H_{II}$ |
| `gini_s_col_MS` | float | Gini of col sums of $M_S = H_{SI}H_{IS}$ |
| `gini_s_col_MI` | float | Gini of col sums of $M_I = H_{IS}H_{SI}$ |
| `m_bar_row_SS` | float | mean row perplexity of $H_{SS}$, unweighted |
| `m_bar_row_II` | float | mean row perplexity of $H_{II}$, unweighted |
| `m_bar_row_SI` | float | mean row perplexity of $H_{SI}$ |
| `m_bar_row_IS` | float | mean row perplexity of $H_{IS}$ |
| `m_bar_row_ratio` | float | $\bar{m}(H_{II}) / \bar{m}(H_{SS})$ (the "3.5×" generalised) |
| `modularity_SS` | float | Leiden $Q$ on $H_{SS}$ |
| `modularity_II` | float | Leiden $Q$ on $H_{II}$ |
| `rho_vB_vS` | float | Spearman $\rho(v^B, v^S)$ on common sources |
| `rho_vB_vI` | float | Spearman $\rho(v^B, v^I)$ on common institutions |
| `slope_vB_vS` | float | OLS slope of $\log v^B$ on $\log v^S$ |
| `slope_vB_vI` | float | OLS slope of $\log v^B$ on $\log v^I$ |
| `n_core_sources` | int | from residualise-and-re-scatter diagnostic |
| `n_core_insts` | int | same |
| `frac_v_above_1_s` | float | share of sources with $v > 1$ |
| `frac_v_above_1_i` | float | share of institutions with $v > 1$ |

### Code additions required

| File | Addition |
|---|---|
| `network_entropy.py` | Add $M_I$, $H_{SI}$, $H_{IS}$ to the kernels dict. Add column-sum Gini per kernel. Add row-perplexity variance alongside mean. Add unweighted-only output columns as defaults. |
| `network_entropy.py` | Add subfield restriction (E, B) — wrap the `_build_kernels` call in a loop over `fx` values. |
| new file `column_structure.py` | Compute column-sum Gini, column perplexity distribution, $\pi$-shape predictors per kernel. One row per `(subfield, mask)`. |
| new file `core_identification.py` | Residualise-and-re-scatter diagnostic. Iteratively drop top-$k$ units by $v^B$; compute residual Spearman and scatter variance; identify $N_{\text{core}}$ as the $k$ at which residuals stabilise. |
| `fig_3.py` / `fig_4.py` | Add overlay showing results restricted to non-core units as a second panel. |
| new file `subfield_summary.py` | Assemble master indicator table above. Columns as specified. One row per `(subfield, mask, window)`. |
| new file `error_model.py` | Perturb work-to-institution weights $\omega_{ix}$ at pre-matrix-assembly stage. Three rates $\epsilon \in \{0.02, 0.05, 0.10\}$, two sampling rules (uniform / journal-country-bucket), $N_{\text{reps}} \geq 50$. Output: distribution of each master-table indicator under each perturbation. |

### LaTeX changes required

| Section | Change |
|---|---|
| §2 at eq. (5) | Single phrase: "exposing within-layer and cross-layer contributions separately" |
| §2 methods | Define column-sum Gini of $H$ as pre-spectral $\pi$-shape predictor; link to fractional citing-side impact factor (Zitt-Small) |
| §2 methods | Define $\bar{m}^{\text{row}}$ (mean row perplexity, unweighted) as matrix-intrinsic mixing statistic; drop "stationary-weighted" / "NSWEB" terminology |
| §4.1 | Restructure to present $S$, $I$, $B$ as primary, $J$ as generalisation |
| §4.1 | Report column-sum Gini alongside $\bar{m}$ for each kernel; state 3.5× as ratio of $\bar{m}$'s |
| §4.1 | Preempt "PageRank blurs tail" concern in one paragraph |
| §4.2 | Add E vs B subfield table with full indicator set |
| §4.2 | Report subfield-specific $N_{\text{core}}$ and branching ratio |
| §4.3 | Cross-window Spearman for $v^B$; Gini and $\bar{m}$ stability per window |
| §5 | Frame structural coupling claim; note core concentration if diagnostic confirms |
| §5 | Cite Leiden Ranking as composite-indicator contrast, West-Bergstrom-Jensen as methodological precedent |
| §6 | Error-model section: specify $\epsilon$, sampling rules, report perturbation bands for headline indicators |

### Open items requiring your input

1. The $J$-as-baseline-or-closer question: the outline above treats $J$ as closer, consistent with your current plan.
2. Whether to run the residualise-and-re-scatter diagnostic before committing to a core-driven vs distributed-coupling narrative.
3. Whether error-model section goes before or after subfield/time analyses — current plan puts it last as a robustness envelope.
4. Terminology: drop "NSWEB" in favour of "mean row perplexity" or "mean effective branching"; the exposition needs to settle on one phrase and use it throughout.