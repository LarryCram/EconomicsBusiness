# ERRORS.md — LaTeX audit against current software state

## 02_model_specification.tex (Section 2)
- **Issue**: Convention statement about C_{ij} is present (correctly states "attention from i (citing, row) to j (cited, column)"), but section title is "Journal--institution citation networks" which is dated nomenclature—now "reference networks" is preferred.
  **Suggestion**: Update section 2 title to clarify "reference networks" or keep citation/reference distinction explicit in opening paragraph.

- **Issue**: Section 2 explains "attention" and "influence" concepts clearly, but uses both "citation matrix" and "reference matrix" interchangeably without reinforcing the convention stated in Introduction.
  **Suggestion**: Standardize: cite CLAUDE.md convention in the paragraph introducing C_{ij}.

## 03_processing.tex (Corpus section)
- **Issue**: References Field 14 (Economics) and Field 20 (Business, Management) from OpenAlex but does not explain new field_eb classification (E/B/A/NULL) used elsewhere.
  **Suggestion**: After describing the OpenAlex field topic filtering, add explicit note: "Sources are classified post-hoc using field_eb: E (econ-only), B (business-only), A (mixed E+B), NULL (neither)."

- **Issue**: Table 3 caption says "Corpus features by year ($\tau_U > 20$)" but text says thresholds are $\tau_U = 20$ and $\tau_S = 20$ (not >).
  **Suggestion**: Correct caption to "$\tau_U \geq 20$" or clarify threshold in table notes.

- **Issue**: No explicit corpus size statement: the 2000--24 aggregates show ~1,610 sources and ~1,012 institutions, but CLAUDE.md states n_s=1091 and n_u=1732.
  **Suggestion**: Check whether final corpus (after all filtering) has different sizes. If 1091/1732 are correct, update Table 3 or add a note clarifying which filtering stage each count refers to.

## 04_results.tex (Results section — SHORT VERSION)
- **Issue**: DUPLICATE FILE: there are two files `04_results.tex` (215 lines) and `04_results_.tex` (289 lines) with different baselines and content. main.tex includes only `04_results`.
  **Suggestion**: Determine which is canonical. `04_results.tex` specifies baseline $\alpha=1$, $\mu=0$; `04_results_.tex` specifies $\alpha=0.85$, $\tau_U=10$. DELETE one and consolidate.

- **Issue**: Line 9 defines baseline as "$\alpha=1, \mu=0$" (pure eigenvector) but this contradicts CLAUDE.md which says baseline is $\alpha=1.0$ with Katz resolvent and mentions sensitivity studies at other $\alpha$ values. The distinction matters.
  **Suggestion**: Clarify: is pure eigenvector ($\mu=0$, $\alpha=1$) the paper baseline, or is it Katz-Hubbell with uniform $\mu$?

- **Issue**: Line 10 says baseline block mask $\mathbf{m}=0110$ but table on line 82 also lists 0110. However, line 86 in table shows "Block mask: ... 1000, 0001, (0110), 1111" without explicit link to which is baseline.
  **Suggestion**: Add parentheses or bold to clarify: "m = (0110) [baseline]" in table.

- **Issue**: Extensive [FILL] placeholders on lines 114-115, 181-193, 209: "$N_s=[FILL]$", "$g^{SS}=[FILL]$", "$\chi^*=[FILL]$", etc.
  **Suggestion**: Populate all [FILL] with actual values from data or code runs.

- **Issue**: Figure captions reference figures fig_2, fig_3, fig_4 and specify parameters ($t_x=5, \tau_U=\tau_S=20, \alpha=1$) but Section 4 outline (line 6-17) mentions "Phase 1", "Phase 2", "Phase 4" without defining them.
  **Suggestion**: Add subsection headers for Phase 1, 2, 4 or clarify what each phase comprises.

- **Issue**: Prestige-per-work metric $v$ is introduced in Section 2 (eq. 219) but in Section 4 figure captions it is called "prestige per work" (line 100, 111) without cross-reference to equation or definition.
  **Suggestion**: Add equation reference: "prestige per work $v$ (Eq. \ref{eq:vstar})".

## 04_results_.tex (Results section — LONG VERSION, NOT INCLUDED IN MAIN)
- **Issue**: This file is NOT included in main.tex but contains 289 lines of results, more detailed than 04_results.tex.
  **Suggestion**: This appears to be a draft or alternative version. Clarify status: delete, merge, or move to Supplement.

- **Issue**: Baseline differs from 04_results.tex: Table on line 15 sets $t_x \in \{1,...5,6\}$ (includes t_x=6) vs. 04_results.tex which shows 1-5. Table line 23 sets baseline $\alpha=0.85$ vs. 04_results.tex baseline $\alpha=1$.
  **Suggestion**: Resolve version conflict. CLAUDE.md mentions time-series runs t1-t4 = t_x 1-4, and baseline parameters in params.csv show 20242024 (t_x=5) with alpha=1.0. Align with params.csv.

- **Issue**: Extensive [FILL] placeholders throughout lines 21, 54-70, 83-194, 201-289.
  **Suggestion**: Fill all placeholders or delete this version in favor of consolidated section.

- **Issue**: Line 30 states baseline as "$\alpha=0.85$" but CLAUDE.md and 04_results.tex both use $\alpha=1.0$.
  **Suggestion**: Correct to match 04_results.tex and params.csv.

- **Issue**: Line 17 sets baseline $\tau_U=10$ but 03_processing.tex and 04_results.tex both use $\tau_U=\tau_S=20$. Table caption in Table 3 mentions $\tau_U > 20$.
  **Suggestion**: Standardize thresholds across all sections; align with CLAUDE.md baseline.

## 05_discussion.tex (Discussion section)
- **Issue**: PLACEHOLDER: Entire section is four lines of "[Placeholder.]" with no content under face validity, content validity, criterion validity, or construct validity subsection headers.
  **Suggestion**: CLAUDE.md explicitly notes "Sections 4–6 and Supplement: placeholder", so this is expected. Either populate or remove section heading and subsections.

## 06_prospects.tex (Prospects section)
- **Issue**: PLACEHOLDER: Section header only, one line "[Placeholder.]", no content.
  **Suggestion**: CLAUDE.md notes this is a placeholder. Remove or populate.

## 07_addenda.tex (Addenda section)
- **Issue**: EMPTY: Section header only, no content at all (4 lines total).
  **Suggestion**: CLAUDE.md notes sections 4-6 and Supplement are placeholders. Clarify: is Addenda meant to be the Supplement? Add content or remove.

## Bibliography (MyLibrary.bib, sciomtcs.bib)
- **Issue**: Not audited in detail. Spot-check: cited references like \cite{pinskiCitationInfluenceJournal1976}, \cite{vignaSpectralRanking2016}, \cite{palacios-huertaMeasurementIntellectualInfluence2004} should be verified as present.
  **Suggestion**: Run bibtex or biblatex compile to catch undefined references.

---

## Summary of Critical Issues

1. **Duplicate results file**: 04_results.tex vs 04_results_.tex with conflicting baselines and parameters. DELETE one.
2. **Baseline parameter mismatch**: 04_results.tex uses $\alpha=1, \tau_U=\tau_S=20$; 04_results_.tex uses $\alpha=0.85, \tau_U=10$. Reconcile with params.csv.
3. **Text corruption**: Lines in 01_introduction.tex contain "institutionszz" and "butz" (lines 15).
4. **Extensive [FILL] placeholders**: 04_results.tex has 9 instances; 04_results_.tex has ~40 instances. All must be populated before publication.
5. **Placeholder sections**: 05_discussion.tex and 06_prospects.tex are entirely placeholder stubs.
6. **Corpus size mismatch**: Table 3 shows ~1,610 sources / ~1,012 institutions (2000-24 aggregate) but CLAUDE.md states n_s=1091, n_u=1732. Clarify filtering stage.
7. **Field classification**: Section 3 describes OpenAlex Fields 14 and 20 but does not link to field_eb (E/B/A/NULL) classification scheme documented in CLAUDE.md.
8. **Convention clarity**: C_{ij} convention is correct but terminology inconsistently uses "citation" vs "reference" without always clarifying which.

