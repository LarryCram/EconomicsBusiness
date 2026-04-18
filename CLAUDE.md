# CLAUDE.md — EconomicsBusiness Project

## Project root
`/home/lc/Projects/EconomicsBusiness` — VS Code workspace, synced to GitHub.

## Folder structure
```
EconomicsBusiness/
  spectral_ranking/        # Python ranking code + tests
  spectral_ranking_latex/  # LaTeX paper (multi-file master)
  spectral_results_analysis/  # Figure scripts (fig_2.py … fig_7.py)
  prepare_data/            # OpenAlex data pipeline + tests
  util/                    # Shared helpers: load_config, load_runs, Paths
  data/                    # Small reference files (MB-scale, git-tracked)
  plots/                   # Plots for exploration and publication (git-tracked)
  params.csv               # Run schedule — one row per ranking run
  config.yaml              # Machine-specific data paths — gitignored
  CLAUDE.md
  PLOTS.md
```

## Python environment
Always use `.venv/bin/python` and `.venv/bin/pip`. Never invoke bare `python` or `pip`.

## Data
Large parquets live on a separate SSD. Location is machine-specific and set in `config.yaml` (gitignored). Never hardcode paths — always read from config via `util.load_config()` which returns a `Paths` object. Small files (journal lists, corpus sources) live in `data/` and are git-tracked.

## Run schedule: params.csv
`params.csv` (project root, git-tracked) drives the entire pipeline. One row per ranking run. Columns:

```
skip, run_code, tc0, tc1, tt0, tt1, fx, tau_u, tau_s, rho, m, chi, alpha, mu_type, label, ref_units
```

- `skip`: 1 = skip this row, 0 = run
- `run_code`: 8-char string, last-2-digits of tc0/tc1/tt0/tt1 with leading zeros
  e.g. tc0=2000,tc1=2004,tt0=2000,tt1=2004 → `'00040004'`
- `chi=-1`: sentinel for χ* (resolved at runtime); appears as `chiSTAR` in table names
- `label`: human identifier ('baseline', 't1'…'t4', 'F=E', etc.)
- `ref_units`: empty for standard (vartau) runs; `{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}`
  for fixtau runs that inherit the unit set from a reference window
- Time-series runs are identified by `label in {'t1','t2','t3','t4'}`, not by a stage column
- Fixtau runs: label ends in `-fix`; `ref_units` points to baseline window units table

Load via `util.load_runs()` which filters skipped rows and coerces types.

## Table naming conventions
```
Edge lists:    el_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_{vartau|fixtau}
Units (raw):   _units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_{vartau|fixtau}
Units (mode):  _units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_{vartau|fixtau}_m{mstr}
Rankings:      rk_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_{vartau|fixtau}_rho{rho}_m{m}_chi{chi_str}_alpha{alpha_int}
```
- `_vartau`: unit set derived from τ filter applied to that window's own census
- `_fixtau`: unit set inherited from a reference window (ref_units non-empty in params.csv)
- `chi_str='STAR'` when chi==-1.0, otherwise `str(round(chi*100))` (e.g. 0.5→'50')
- The `_tau_sfx()` helper in each pipeline script computes the suffix from `ref_units`

## Field classification: field_eb
Sources in `source_master.parquet/csv` carry a `field_eb` column (replaces old `field_subset`):

- **'E'**: econ_score ≥ 2 AND bus_score < 2  (economics-dominant)
- **'B'**: bus_score ≥ 2 AND econ_score < 2  (business-dominant)
- **'A'**: econ_score ≥ 2 AND bus_score ≥ 2  (genuinely ambiguous — strong in both)
- **'X'**: econ_score < 2 AND bus_score < 2   (weak signals in both)

`field_eb` is never NULL. Scores count how many of {era_field, harzing_field, wos_categories, field_name} contain econ/business keywords. F=EB corpus filter is `field_eb IN ('E','B','A')`. F=A is the full corpus (no filter). F=X is the neither-signal residual.

## Paper
Multi-file LaTeX in `spectral_ranking_latex/`. Master file is `main.tex`; sections are in `sections/`. Bibliography fed by Zotero (`MyLibrary.bib`). Committed to GitHub for backup.

**Convention (critical):** $C_{ij}$ = attention from $i$ (citing, row) to $j$ (cited, column). Row sum = references given out. This is the transpose of economist's convention but consistent with most non-economics bibliometrics. Prefer "reference" over "citation" unless it is precise. Do not read any folder named `zarchive`, `archive`, or `ARCHIVE`.

## LaTeX audit
`ERRORS.md` (project root) lists 8 known issues in the LaTeX source. Key ones:
- `04_results_.tex` (conflicting baseline α=0.85) moved to `zarchive/` — `04_results.tex` is canonical
- Text corruption in `01_introduction.tex` ("institutionszz", "butz") — needs manual repair
- Many `[FILL]` placeholders in `04_results.tex` — to be populated from ranking runs
- Section 3 does not yet describe `field_eb` (E/B/A/NULL) classification

## Current paper status
- `main.tex` compiles cleanly
- Section 1 (introduction): complete
- Section 2 (mathematical framework and computational summary): complete
- Section 3 (source selection, scope filtering, corpus construction): complete
- Sections 4–6 and Supplement: placeholder

## Machines
Two home Linux machines plus HPC. Code and LaTeX sync via GitHub. Data moves via portable SSD.
