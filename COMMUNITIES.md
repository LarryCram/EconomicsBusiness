# Community Structure Analysis — Session Notes

## What was built

`spectral_results_analysis/community.py` — standalone script (not in pipeline).
Computes λ₂, φ₂, spectral gap, and community amplification for five modes:
SS, II, bipartite M_S, full χ=0.5, full χ*. Also runs SCC diagnostics on each matrix.

Run with: `.venv/bin/python spectral_results_analysis/community.py`

---

## Pipeline change: singleton SCC filtering

Added `filter_singletons(db, tx, fx, tau_u)` to `prepare_data/build_edge_lists.py`.
Called after `build_units` for every corpus.

**Criterion:**
- Sources: must be in giant SCC of **C_SS** (strict — excludes sources connected only through
  institutional paths but with no source-to-source cycle with the main literature)
- Institutions: must be in giant SCC of full joint **C_full**

**Effect at baseline (t5, A, τ_U=20):**
- Dropped 108 sources (first pass: publisher-omission sinks + τ_U-induced zeros)
- Then 7 further sources (OUT/IN-component in C_SS: nonzero out-row but no SS cycle)
- Dropped 2 institutions
- Final corpus: N_s=1322, N_u=1732, χ*=0.567

The 7 C_SS non-giant sources were real journals (Issues in Social and Environmental
Accounting, Scientific Bulletin of Mukachevo State University "Economics", Marketing,
VISIÓN GERENCIAL, American Review of Political Economy, J for International Business
and Entrepreneurship Development, REVISTA PROCESOS DE MERCADO). They cite the main
literature but are not cited back in source-source paths. Dropped as peripheral.

---

## Final λ₂ results (baseline t5, A, τ_U=20, α=0.85)

| Mode           | λ₂      | gap     | amplification |
|----------------|---------|---------|---------------|
| SS             | 0.77728 | 0.22272 | 2.9471        |
| II             | 0.47942 | 0.52058 | 1.6878        |
| bipartite M_S  | 0.33492 | 0.66508 | 1.3980        |
| full χ=0.5     | 0.57835 | 0.42165 | 1.9669        |
| full χ*=0.567  | 0.55950 | 0.44050 | 1.9069        |

C_II and C_full are each a single SCC (giant=N, non-giant=0) after filtering.
C_SS has one SCC of 1322 sources.

SS λ₂=0.778 is now a real community signal (previously stuck at α=0.85 due to reducibility).

---

## Key interpretations

**Ordering (strongest → weakest community structure):**
SS > full χ=0.5 ≈ full χ* > II > bipartite

**Bipartite φ₂ community split (clearest signal):**
- Positive pole (economics community): AER, Econometrica, QJE, JPE, RES, REStat,
  JPublicE, JEconometrics, AEJ:Macro, AEJ:EP, Management Science, Operations Research,
  JF, JFE, Review of Financial Studies
- Negative pole (applied business/management): Journal of Business Research, Journal of
  Environmental Management, Technological Forecasting, Finance Research Letters,
  Energy Economics, Business Strategy & Environment, Hospitality Management journals,
  Industrial Marketing Management, Annals of Operations Research, Journal of Corporate
  Finance, CSR & Environmental Management

**Finance is internally split:** top theory/empirics journals (JF, JFE, RFS) align with
economics; applied finance journals (Finance Research Letters, IRFA) align with business.
ERA/Scopus field labels (E/B) are a noisy proxy for citation-based community membership.

**E/B vs S/I as drivers of ranking differences:**
- E/B is the *content* of community structure (what φ₂ looks like)
- S/I mode controls the *strength* of that separation (amplification dial)
- SS amplifies E/B separation most (2.95×); bipartite dampens it most (1.40×)
- Journals near the E/B boundary gain relative to field-specialists as mode moves SS→bipartite
- v(F=B)/v(F=A) ratio drops are a corpus-restriction effect (distinct from mode choice):
  B-labelled journals cited heavily by E journals rank lower in F=B because E sources removed

**Full joint χ* (~0.567) design rationale:**
Operates at ~1.91× amplification — intermediate between field-separating (SS) and
field-integrating (bipartite). Robust to exact χ value in the neighbourhood of balance.

---

## What φ₂ has NOT yet been printed

- II mode φ₂ (institution names) — would show how institutions split E/B
- Full joint φ₂ (combined source+institution vector)

Comparing φ₂ positions across modes would show whether community assignments are stable
across SS/II/bipartite or whether mode reshuffles boundary journals.

To add: `print_phi2_top` calls for II and full-joint in `community.py`.
For II, need an institution name lookup (institution_idx → display name from OA parquets).

---

## Paper implications (draft points)

1. The E/B community boundary is real but soft (λ₂=0.778 < 1 in SS). Economics and
   business journals do cross-cite enough to form one giant SCC.
2. Institutional connections substantially reduce field separation: λ₂ drops from 0.778 (SS)
   to 0.335 (bipartite). Routing prestige through shared institutional authorship is a
   community mixer, not a community amplifier.
3. The citation-based field partition is finer than ERA/Scopus labels: finance splits,
   operations research/management science aligns with economics.
4. The full joint model at χ* is a defensible design choice: it operates at intermediate
   amplification, neither maximally separating fields nor washing out the signal.
5. Quantitative community diagnostic: gap and amplification columns provide a compact
   summary of how much the network structure magnifies prestige differences.
