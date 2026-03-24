# CLAUDE.md — EconomicsBusiness Project

## Project root
`/home/lc/Projects/EconomicsBusiness` — VS Code workspace, synced to GitHub.

## Folder structure
```
EconomicsBusiness/
  spectral_ranking/        # Python ranking code
  spectral_ranking_latex/  # LaTeX paper (multi-file master)
  prepare_data/            # OpenAlex data pipeline
  data/                    # Small reference files (MB-scale, git-tracked)
  plots/                   # Plots for exploration and publication (git-tracked)
  config.yaml              # Machine-specific data paths — gitignored
  CLAUDE.md
  PLOTS.md
```

## Data
Large parquets live on a separate SSD. Location is machine-specific and set in `config.yaml` (gitignored). Never hardcode paths — always read from config. Small files (journal lists, corpus sources) live in `data/` and are git-tracked.

## Paper
Multi-file LaTeX in `spectral_ranking_latex/`. Master file is `main.tex`; sections are in `sections/`. Bibliography fed by Zotero (`MyLibrary.bib`). Commited to GitHub for backup.

**Convention (critical):** $C_{ij}$ = attention from $i$ (citing, row) to $j$ (cited, column). Row sum = references given out. This is the transpose of economist's convention but consistent with most non-economics bibliometrics. Prefer "reference" over "citation" unless it is precise.

## Current paper status
- `main.tex` compiles cleanly
- Section 1 (introduction): complete
- Section 2 (mathematical framework and computational summary): complete
- Section 3 (source selection, scope filtering, corpus construction): complete
- Sections 4–6 and Supplement: placeholder

## Machines
Two home Linux machines plus HPC. Code and Latex syncs via GitHub. Data moves via portable SSD.
