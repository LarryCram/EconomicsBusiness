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
  config.yaml              # Machine-specific data paths — gitignored
  CLAUDE.md
```

## Data
Large parquets live on a separate SSD. Location is machine-specific and set in `config.yaml` (gitignored). Never hardcode paths — always read from config. Small files (journal lists, corpus sources) live in `data/` and are git-tracked.

## Paper
Multi-file LaTeX in `spectral_ranking_latex/`. Master file is `main.tex`; sections are in `sections/`. Bibliography fed by Zotero (`MyLibrary.bib`). 

**Convention (critical):** $C_{ij}$ = attention from $i$ (citing, row) to $j$ (cited, column). Row sum = references given out. This is the transpose of the Pinski-Narin economist convention. Use "reference" not "citation" until Layer 3 is complete.

## Current paper status
- `main.tex` compiles cleanly
- Section 2 (mathematical framework, Layers 1–4): complete
- Section 3 (corpus construction + scope filtering, 2,341 sources): complete
- Section 1 (introduction): skeleton only
- Sections 4–6 and Supplement: placeholder

## Machines
Two home Linux machines plus HPC. Code syncs via GitHub. Data moves via portable SSD.
