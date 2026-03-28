# LaTeX
# EconomicsBusiness project — spectral_ranking_latex/

## Location
`spectral_ranking_latex/` — multi-file project, master = `main.tex`, sections in `sections/`.
Bibliography fed by Zotero (`MyLibrary.bib`).

## Current paper status (March 2026)

| Section | Title | Status |
|---|---|---|
| 1 | Introduction | Complete |
| 2 | Journal–institution citation networks | Complete (bipartite subsection pending — see below) |
| 3 | Corpus selection | Complete |
| 4 | Results | Placeholder — structure fully specified below |
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

## Baseline parameter correction

The placeholder in `04_results.tex` incorrectly states the baseline as
ρ=1, m=(1,0,0,0). The actual baseline in `run_rankings.py` is:

    t_x=5, F=A, τ_U=10, ρ=0 (fixed count, R̄/R_i), m=(0,1,1,0), α=0.85

The SS-only and II-only runs are comparisons, not the baseline. The parameter
table in `04_results.tex` (Tab. paramvalues) needs correcting before submission.

---

## Pending section work

### Section 2 — changes required

#### 2a. Bipartite resolvent: update to √α convention

The current paper (equations kh_source and kh_inst) uses α (and α²) as the
damping parameter throughout the bipartite resolvent. The code and analysis
now use α_step = √α as the per-step damping so that each reference — which
traverses two steps in the bipartite walk (S→I and I→S) — has the same
net attenuation α = α_step² as a single step in the SS or II walk.

**Equations to rewrite.**

Current eq:kh_source:
  (I − α² M_S^T) π_S = (1−α)(μ_S + α H_IS^T μ_I)

Replace with:
  (I − α M_S^T) π_S = (1−√α)(μ_S + √α H_IS^T μ_I)

Current eq:kh_inst:
  π_I = α H_SI^T π_S + (1−α) μ_I

Replace with:
  π_I = √α H_SI^T π_S + (1−√α) μ_I

**Prose to rewrite.** The sentence "where the effective damping is α²"
should become something like: "where α_step = √α is the per-step attenuation,
chosen so that a reference traversing two steps S→I→S is attenuated by
α_step² = α, matching the single-step attenuation in the SS and II cases."

**Motivation sentence to add.** After eq:kh_inst, add a sentence noting that
under this convention the community amplification factor 1/(1−α λ₂) has the
same form in all four modes (SS, II, bipartite, full joint), making the
spectral gap g = 1−λ₂ directly comparable across modes.

#### 2b. Dimensionally neutral mixing weight χ*

The current text presents χ only as a parameter that "preserves total citation
flow" via the identity (1−χ)² + 2χ(1−χ) + χ² = 1. This does not address the
asymmetry N_u > N_s (≈2,600 vs ≈1,600 at the baseline), which means that at
χ=0.5 each institution unit receives less prestige flow per unit than each
source unit.

**Add a paragraph** (after the χ scaling sentence, before the parameter table
or in a dedicated subsection) deriving χ*:

  At χ = 0.5 the total flow allocated to the source mode is (1−χ)² + χ(1−χ)
  = (1−χ), and to the institution mode is χ(1−χ) + χ² = χ. For equal expected
  prestige per unit across both types, the flow per unit must be equalised:
    (1−χ)/N_s = χ/N_u
  Solving: χ* = N_u/(N_s+N_u).
  With N_s ≈ 1,600 and N_u ≈ 2,600 (baseline F=A, τ_U=10, t_x=5),
  χ* ≈ 0.619. This is the dimensionally neutral calibration; χ=0.5 is used
  as the reference case throughout, with χ* examined in Part D.

Note: χ* must be computed dynamically at runtime from the units table because
N_u depends on τ_U and F. It is not a fixed constant.

**Update the parameter table entry** for χ: extend the Role column to mention
that χ* = N_u/(N_s+N_u) is the dimensionally neutral value.

#### 2c. Second eigenpair of H^T as community diagnostic

A new paragraph (or subsection "Community structure") is needed in Section 2
to introduce the spectral community analysis that appears in the results.
It should appear after the Perron–Frobenius/power-iteration paragraph.

The key points to make:

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

This paragraph motivates Figure 3 and Parts A.3, B.2, C.2, D.3 of the
results section.

---

### Section 3 — changes required

#### 3a. Institution count figure

The current text states "leaving approximately 1,900 institutions and 75% of
the works" at τ_U=10. The corpus features table (tab:corpus_features) reports
N_u=1,742 for the full 2000–2024 pooled period; neither figure matches the
session estimate of ≈2,600 institutions for the baseline window t_x=5
(2020–2024) at τ_U=10.

**Action**: once the pipeline runs on the current data, update the "1,900"
figure in the prose and the counts in tab:corpus_features to reflect the
current extraction (F=A, t_x=5, τ_U=10). These are the figures used in the
χ* derivation (§2b above) and must be consistent throughout.

Note: the session confirmed that the corpus counts in CLAUDE.md ("N_s≈1,600,
N_u≈2,600") are also approximate and pre-date the most recent data pull.
All figures should be verified against the pipeline output before submission.

#### 3b. τ_U sensitivity note

The Section 3 text presents τ_U=10 as the chosen threshold without mentioning
that τ_U=8 is examined as a robustness check. Add a brief sentence after
the τ_U=10 choice is stated:

  "The choice τ_U=10 is used throughout; the sensitivity of rankings to this
  threshold is examined in Section~4 (Part~E) using τ_U=8."

#### 3c. F-label source_idx join (implementation note, not prose change)

Section 3 defines the F-label classification (F=E: Field 14 only; F=B:
Field 20 only; F=A: either) using OpenAlex source-topic counts. In
`table_maker.py` and `community.py`, sources are stored by `source_idx`
(integer, stripped from the full OpenAlex URI `https://openalex.org/S{id}`).
The F-label join from the `data/` reference files must use the same stripped
integer key. Verify before running table_maker:

  - Check the F-label file in `data/` for its key column format.
  - If it stores the full URI, strip the prefix in the join:
      CAST(REPLACE(source_id, 'https://openalex.org/S', '') AS INTEGER)
  - If it stores the integer directly, join on source_idx = source_id directly.

This is an implementation concern that does not require a prose change in
Section 3, but it must be resolved before the F=E/F=B comparison tables
(Parts A.2, A.3 of results) can be produced. Document the resolution in
WORKPLAN.md.

---

## Section 4 — Results: full structure

The results section is built up in five parts, following the sequence in which
runs are produced and results examined. Each part produces specific figures and
tables, listed below for use in planning the `spectral_results_analysis/` scripts.

The second eigenpair (λ₂, φ₂) is computed analytically in `community.py` as
needed; it is not stored in the pipeline. φ₂ for the community partition table
(Table 3 in `table_maker.py`) is passed from `community.py` via a small CSV.

---

### Part A — Source ranking: the SS mode

**Analytical purpose**: establish the journal ranking in pure source-mode and
expose the Economics/Business community structure as the dominant spectral feature.

#### A.1 SS baseline (F=A)

Run: `rk_t5_A_tau10_rho0_m1000_chi50_alpha85`

Produce:
- **Table S1** (top sources by v, F=A, SS): `table_maker.py`, source table.
  Columns: rank_v, journal name, ISSN-L, F-label, v.
  Top 30 sources. This is the opening empirical exhibit.
- **Figure 2** (v vs rank, sources and institutions): `spectral_results_analysis/fig_2.py`.
  Already specified in PLOTS.md. Top panel: sources (SS). Bottom: placeholder
  for institutions (added when bipartite run available).

What to look for: do top journals match prior expectations from impact-factor
rankings? Note the long right tail (a few journals with v >> 1).

#### A.2 F variation in SS: Economics vs Business

Runs: `rk_t5_E_tau5_rho0_m1000_chi50_alpha85` (F=E)
      `rk_t5_B_tau5_rho0_m1000_chi50_alpha85` (F=B)

Produce:
- **Table S2** (top sources, SS F=E): source table from `table_maker.py`.
- **Table S3** (top sources, SS F=B): source table from `table_maker.py`.
- **Rank-shift exhibit**: for F=A sources, scatter rank_v(F=A) vs rank_v(F=E)
  (for sources present in both corpora), coloured by F-label. Finance journals
  (JF, JFE, RFS) expected to drop substantially under F=E restriction.

What to look for: finance journals act as community bridges in F=A — they draw
prestige from both E and B citation flows. Under F=E, the B-side flow is cut and
their rank drops. This is the key motivation for examining community structure.

#### A.3 Community structure in SS: second eigenpair

Computed by `community.py` from `C_SS` CSR blocks (F=A corpus).

Produce:
- **Figure 3, Panel A** (φ₂ sorted, SS sources, coloured E/B).
  Sign convention: unit with largest a_p has φ₂ > 0.
  Expected: clean sign split E/B; finance journals near φ₂ = 0.
- **Table S4** (community partition): sources sorted by φ₂, with
  F-label, φ₂, v, rank_v. Bridge zone |φ₂| < 0.05 annotated.
  Produced by `table_maker.py` using φ₂ CSV from `community.py`.

Record: λ₂^{SS}, spectral gap g^{SS} = 1 − λ₂^{SS},
amplification A^{SS} = 1/(1 − α λ₂^{SS}).
These are the baseline community diagnostics against which II and bipartite
are compared.

---

### Part B — Institution ranking: the II mode

**Analytical purpose**: show that the institution citation network is more
integrated than the journal network — disciplinary communities are diluted
because institutions co-locate Economics and Business.

#### B.1 II baseline (F=A)

Run: `rk_t5_A_tau10_rho0_m0001_chi50_alpha85`

Produce:
- **Table S5** (top institutions by v, F=A, II): institution table from
  `table_maker.py`. Columns: rank_v, institution name, country, v.
  Top 30 institutions.

What to look for: elite multi-faculty universities (MIT, Harvard, Chicago,
LSE) expected at the top, aggregating prestige across both E and B research
activity. Contrast with SS where E-only or B-only journals can dominate.

#### B.2 Community structure in II: second eigenpair

Produce:
- **Figure 3, Panel B** (φ₂ sorted, II institutions, coloured by dominant field
  where available, else grey).
  Expected: noisier, more institutions near φ₂ = 0 than in Panel A.

Record: λ₂^{II}, g^{II}, A^{II}.
Expected: g^{II} > g^{SS} (larger spectral gap — II mixes communities faster).
The amplification A^{II} < A^{SS}: community membership has less influence
on the II ranking than on the SS ranking.

#### B.3 SS vs II comparison

This is the central qualitative finding of Part B: moving from journals to
institutions dissolves disciplinary community separation because institutions
co-locate E and B researchers. Make this explicit in the prose with reference
to λ₂^{SS} vs λ₂^{II} and Panels A/B of Figure 3.

---

### Part C — Bipartite SI/IS mode (the paper's reference baseline)

**Analytical purpose**: show that the bipartite walk through institutions is
an intermediate case — it partially dissolves community separation via the
institution mixing step, but less completely than II alone because each source
still communicates through its specific institutional affiliates.

#### C.1 SI/IS baseline rankings (F=A)

Run: `rk_t5_A_tau10_rho0_m0110_chi50_alpha85` (paper baseline)

Produce:
- **Table 1** (top sources by v, bipartite): primary source ranking table
  for the paper. Columns: rank_v, journal name, ISSN-L, F-label, v.
- **Table 2** (top institutions by v, bipartite): primary institution ranking
  table for the paper. Columns: rank_v, institution name, country, v.
  Both top 30; these are the main empirical exhibits in the paper.
- **Figure 2** (v vs rank, updated): add institution panel (bottom) using
  bipartite institution v values.

Note: the bipartite run uses α_step = √α ≈ 0.922 per step internally,
giving α = 0.85 per reference (comparable to SS and II). Note this in the
table caption.

#### C.2 SI/IS community structure: second eigenpair of M_S

`community.py` forms M_S = H_SI @ H_IS and computes its second eigenpair.

Record: λ₂^{M_S}, g^{bip}, A^{bip} = 1/(1 − α λ₂^{M_S}).

Expected: λ₂^{M_S} < λ₂^{SS} (institution mediation blurs community
boundaries) but g^{bip} < g^{II} (the bipartite walk is not as diffuse
as direct II because sources still have specific institutional affiliates).

Derive institution community vector: φ₂^I = H_SI^T φ₂^S (normalised).
This gives the institution-side community partition induced by the source walk.

#### C.3 Compare SI/IS with SS: source rankings

Produce:
- **Rank-correlation table** (Spearman ρ between v rankings):
  SS vs SI/IS for sources. Which sources move most?
  Finance journals expected to rise in SI/IS relative to SS because their
  institution affiliates span E and B, giving them indirect access to
  prestige from both communities.

- **Scatter**: v(SS) vs v(bipartite) for sources (log scale both axes),
  coloured by F-label. Divergences highlight the institution-mediation effect.

#### C.4 Compare SI/IS with II: institution rankings

Produce:
- **Rank-correlation table**: Spearman ρ between II and SI/IS institution v.
  Large institutions with both E and B research expected to rank similarly
  in both; smaller specialist institutions may shift.

---

### Part D — Full joint mode and χ sensitivity

**Analytical purpose**: show how the direct SS and II diagonal blocks,
weighted by χ, blend the source-mode and institution-mode rankings; identify
how much community structure in the joint ranking is due to source-mode vs
institution-mode effects; and assess the dimensionally neutral calibration χ*.

#### D.1 Full joint at χ = 0.5

Run: `rk_t5_A_tau10_rho0_m1111_chi50_alpha85`

Produce:
- **Table S6** (top sources, full joint χ=0.5): source table.
- **Table S7** (top institutions, full joint χ=0.5): institution table.
- **Rank-correlation table**: full joint vs SI/IS (sources); full joint vs
  II (institutions). How much does adding the direct SS and II blocks shift
  rankings relative to the bipartite-only baseline?

Community structure in full joint: computed by `community.py` from assembled
full H. Second eigenvector φ₂ has length N = N_s + N_u; partition into
φ₂^S and φ₂^U. The second eigenvector may encode the E/B community split,
the S/I mode split, or both. Examine the loading of φ₂^S on the E/B label
and φ₂^U on institution type to determine which dominates at χ=0.5.

Record: λ₂^{full, 0.5}, g^{full, 0.5}, A^{full, 0.5}.

#### D.2 Full joint at χ* (dimensionally neutral)

Run: `rk_t5_A_tau10_rho0_m1111_chi{N}_alpha85`
where χ* = N_u/(N_s + N_u) ≈ 0.619 (computed dynamically; N is the integer
rounding used in the table name).

**Rationale for χ***: at χ = 0.5 the citation flow allocates equal mass to
the source and institution modes, but N_u > N_s (≈ 2,600 vs 1,600), so each
source unit receives more prestige per unit than each institution. χ* corrects
this: the flow to each mode is proportional to the number of units, giving
equal expected prestige per unit across types. This is the dimensionally
neutral calibration.

Produce:
- **Figure 3, Panel D**: scatter φ₂(χ=0.5) vs φ₂(χ*) for sources,
  coloured by F-label. Shows whether dimensional rebalancing shifts community
  assignments (expected: minor shifts in magnitude, stable sign partition).
- **Rank-correlation table**: v(χ=0.5) vs v(χ*) for sources and institutions.

Record: λ₂^{full, χ*}, g^{full, χ*}, A^{full, χ*}.

#### D.3 Spectral gap summary across all modes

Produce:
- **Figure 3, Panel C**: horizontal bar chart, one bar per case:
  SS F=A, II F=A, bipartite (M_S), full joint χ=0.5, full joint χ*.
  x-axis: spectral gap g = 1 − λ₂. Annotate with λ₂ and A = 1/(1−α λ₂).
  This is the quantitative spine of the community analysis.

Expected ordering: g^{SS} < g^{bip} < g^{II} (SS has tightest communities,
II most diffuse). Full joint between SS and II depending on χ. The bar chart
makes the monotone progression from journal-mode to institution-mode visible
in a single exhibit.

---

### Part E — Parameter sensitivity (Stage 1 one-at-a-time)

**Analytical purpose**: show that the main rankings are robust to parameter
choices, and identify which parameters matter most. Each comparison uses
Spearman rank correlations against the bipartite baseline.

Runs available from Stage 1 (all relative to bipartite baseline):

| Variant | Run label | Change from baseline |
|---|---|---|
| ρ=1 | rho1 | Full reference count vs fixed count |
| α=0.5 | alpha0.5 | Lower damping |
| τ_U=8 | tau8 | Relaxed institution threshold |
| SS-only | SS-only | Mode comparison (already in Part A) |
| II-only | II-only | Mode comparison (already in Part B) |
| Full joint | full-joint | Mode comparison (already in Part D) |
| Full joint χ* | full-joint-chi-star | Mode/calibration (already in Part D) |

Produce:
- **Table S8** (sensitivity summary): Spearman ρ(v) between each Stage 1
  variant and bipartite baseline, for sources and institutions separately.
  One row per variant. Shows which parameter changes matter most.

The ρ=1 vs ρ=0 comparison deserves a brief dedicated paragraph: fixed-count
normalisation (ρ=0) down-weights works with many references, which affects
review articles and handbooks disproportionately.

---

### Part F — Time series (Stage 2)

**Analytical purpose**: show ranking stability across time windows.

Runs: t_x ∈ {1,2,3,4,6} at baseline parameters (m=0110).

Produce:
- **Figure 4** (rank stability): rank of top-20 sources across t_x=1–6,
  shown as parallel coordinates or bump chart (connected line plot).
  Stable journals remain flat; volatile ones cross.
- **Table S9** (Spearman ρ between adjacent windows): N_tx × N_tx matrix
  of pairwise rank correlations for source v. Shows whether the citation
  network structure is stable over the observation window.

---

## Figure and table checklist for results section

| Item | Script | Status |
|---|---|---|
| Figure 2 (v vs rank, S+I) | fig_2.py | In progress |
| Figure 3 (four-panel community) | community.py | Specified in PLOTS.md |
| Figure 4 (time series stability) | fig_4.py | Planned (Stage 2) |
| Table 1 (top sources, bipartite) | table_maker.py | Pending pipeline |
| Table 2 (top institutions, bipartite) | table_maker.py | Pending pipeline |
| Table S1 (top sources, SS F=A) | table_maker.py | Pending pipeline |
| Table S2 (top sources, SS F=E) | table_maker.py | Pending pipeline |
| Table S3 (top sources, SS F=B) | table_maker.py | Pending pipeline |
| Table S4 (community partition, SS) | table_maker.py | Needs community.py φ₂ CSV |
| Table S5 (top institutions, II F=A) | table_maker.py | Pending pipeline |
| Table S6 (top sources, full χ=0.5) | table_maker.py | Pending pipeline |
| Table S7 (top institutions, full χ=0.5) | table_maker.py | Pending pipeline |
| Table S8 (sensitivity Spearman ρ) | diagnostics.py | Pending Stage 1 complete |
| Table S9 (time-series Spearman ρ) | diagnostics.py | Pending Stage 2 |

---

## Writing order for Section 4

Write subsections in the order they are supported by arriving results:

1. Part A (SS F=A baseline, Tables S1–S3) — available as soon as SS runs complete.
2. Part B (II F=A baseline, Table S5) — available as soon as II run completes.
3. Part C (bipartite baseline, Tables 1–2) — this is the main exhibit;
   write carefully, it anchors the paper.
4. Part A.3 / B.2 / C.2 (community analysis) — write together after
   `community.py` is complete and Figure 3 is in hand.
5. Part D (full joint, Tables S6–S7, Figure 3 Panel D) — write after full-joint
   runs complete.
6. Part E (sensitivity, Table S8) — write last before Part C, as a robustness
   check framing.
7. Part F (time series, Figure 4) — write as a closing stability exhibit.

The prose arc of Section 4: start with the simplest case (SS), introduce
community structure to motivate the bipartite walk, show the bipartite baseline
as the primary result, then introduce the full joint and χ as refinements, and
close with robustness.
