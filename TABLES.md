# TABLES.md — EconomicsBusiness Project

## Conventions
- Produced by `prepare_data/table_maker.py` (data tables) or hand-coded in LaTeX (parameter tables).
- Final LaTeX files in `data/`; PDFs for checking in `data/`.

## Table inventory

### Table 1 — Model parameters (done, hand-coded LaTeX)
**In paper**: Table 1, Section 2
**Content**: Symbol, name, type, role for all parameters (F, t_x, τ_U, ρ, m, χ, α).

### Table 2 — Registry source matches to OpenAlex source identifiers (done)
**Script**: `prepare_data/table_maker.py`
**Files**: `data/table2_source_matching.tex`, `data/table2_source_matching.pdf`
**In paper**: Table 2, Section 3
**Content**: Journals, matched, match rate, unmatched per registry (JQL, MJL, SJL);
pairwise overlaps; OAS* and OAS totals.

### Table 3 — Corpus construction parameters (done, hand-coded LaTeX)
**In paper**: Table 3, Section 3
**Content**: Panel A: field subsets (E, B, A). Panel B: seven time windows (t_x=1–7)
with census window, target window, label.

### Table 4 — Corpus features by year (done)
**Script**: `prepare_data/table_maker.py`
**Files**: `data/table3_corpus_features.tex`, `data/table3_corpus_features.pdf`
**In paper**: Table 4, Section 3 (note: script uses `table3_` prefix, paper uses Table 4)
**Content**: Upper panel: unique-item counts (works, sources, institutions, references,
citations) for years 2000, 2024, and 2000–24. Lower panel: D1 and D9 deciles of
per-entity distributions (works/source, institutions/work, out/in-degree per work).
Institution filter: τ_U > 20.

### Table 5 — Parameters and values explored (done, hand-coded LaTeX)
**In paper**: Table 5, Section 4
**Content**: All parameters with baseline values and explored values.

## Upcoming tables (spectral ranking results available)

### Table 6 — Baseline source ranking
**Status**: Data available, table generation pending
Top sources by prestige per work under baseline parameters.

### Table 7 — Baseline institution ranking  
**Status**: Data available, table generation pending
Top institutions by prestige per work under baseline parameters.

### Table 8 — Parameter sensitivity summary
**Status**: Data available, table generation pending
Rank correlation table across parameter combinations.
