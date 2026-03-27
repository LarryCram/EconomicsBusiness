# EconomicsBusiness

Dual ordinal spectral ranking of journals and institutions (Docampo & Cram, 2026).

## Overview
Constructs joint prestige scores for economics/business journals and research institutions
from OpenAlex citation data using a Katz resolvent applied to a parameterised block
attention matrix.

## Folder structure
```
EconomicsBusiness/
  prepare_data/            # Data pipeline scripts
  spectral_ranking/        # Ranking pipeline scripts (next step)
  spectral_ranking_latex/  # LaTeX paper (multi-file, master = main.tex)
  data/                    # Small reference files and LaTeX tables
  plots/                   # Exploration and publication figures
  ZARCHIVE/                # Superseded code and notes
  params.yaml              # Model parameters (version-controlled)
  config.yaml              # Machine-specific paths (gitignored)
  CLAUDE.md                # Project conventions
```

## Pipeline

### Stage 1 — Source list construction
`prepare_data/journal_assembler_era_harzing_wos.py`
→ `comprehensive_journal_list.parquet`

### Stage 2 — OA matching and topic filtering
`prepare_data/journal_filter_match_oa.py`
→ `source_master.parquet` (final OAS: 1,659 sources)

### Stage 3 — Corpus extraction
`prepare_data/load_corpus_entities.py`
→ `corpus_works.parquet`, `corpus_authorships.parquet`, `corpus_references.parquet`

### Stage 4 — Institution retention diagnostics
`prepare_data/institution_retention.py`
→ diagnostic tables for τ_U selection

### Stage 5 — Edge list construction
`prepare_data/build_edge_lists.py`
→ `edge_lists.duckdb` with 21 tables `el_t{tx}_{fx}_tau{tau_u}`

### Stage 6 — Paper tables
`prepare_data/table_maker.py`
→ LaTeX/CSV tables in `data/`

### Stage 7 — Spectral ranking *(not yet started)*
`spectral_ranking/`
→ CSR assembly, Katz iteration, prestige scores

## Current status
- Stages 1–6: complete
- Stage 7 (spectral ranking): not yet started
- Paper: Sections 1–3 complete; Sections 4–6 and Supplement placeholder

## Key parameters
See `params.yaml` (time windows, τ_U floors) and Table 1 in the paper (ρ, m, χ, α).

## Key documents
- [CLAUDE.md](CLAUDE.md) — project conventions and C_ij convention
- [DATA_PREPARATION.md](DATA_PREPARATION.md) — pipeline documentation and known issues
- [SPECTRAL_RANKING.md](SPECTRAL_RANKING.md) — ranking pipeline specification
- [LATEX.md](LATEX.md) — paper status and pending sections
- [PLOTS.md](PLOTS.md) — plot inventory
- [TABLES.md](TABLES.md) — table inventory
