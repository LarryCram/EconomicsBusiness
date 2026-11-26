# EconomicsBusiness — Project Progress

_Last updated: 2026-11-26 (QSS revision status below; the sections after it are the April 2026 record)_

## QSS revision status (2026-11-26)

Branch `qss-revision`, corrected baseline `econ_bus_rev` (`ECONBUS_CONFIG=config.rev.yaml`); see CLAUDE.md and `qss_lock/README.md`.

Done on the corrected baseline:
- Error studies re-run with ω = 0 replicates (`run_error_studies.sh`): resample, source, institution, reference, and `year_swap` (one-year shifts across the window edges; replaces `year_v2`). No combined run.
- Finance/accounting rule: E or A journals with ≥ 2 of 4 finance/accounting signals become B, four economics exceptions (`data/field_rule_exceptions.csv`). Journals 525 E / 876 B / 101 A / 62 X; EBAX unaffected. §3 and §4 text updated.
- Separate E and B resample bootstraps (`--run-label F=E|F=B`) as an internal-structure probe; E+B combined stays the primary object.
- MNCS comparison (`mncs_comparison.py`): combined v against field-normalised MNCS on paired bootstrap bands (in / near / far). Journals 488 / 600 / 476, corr 0.78; institutions 346 / 632 / 1,001, corr 0.51. Gaps track citer standing (corr 0.70 / 0.76). `--counting full` (FWCI-like) is slightly worse for institutions (332 / 544 / 1,103, corr 0.48); a strict FWCI window is impossible because the corpus ends in 2024.
- AIS comparison (`benchmark_journals.py` → `data/benchmark_journals_ais.csv`): bipartite v vs JCR 2024 AIS corr 0.88, RMA slope 1.05, median offset −0.017 dex; 390 in / 412 near / 216 far of 1,018. MNCS vs AIS corr 0.77. Largest gaps are review/annual journals (AIS higher: 5-year window, citers outside the corpus) and health economics.

Next: paper restructure (supplements, model in brief, data-quality paragraph, limitations, MNCS and AIS paragraphs, benchmark/survivorship/finance/split tables), computational-cost text, response letter, Zenodo deposit.

## What this project is

A bipartite spectral ranking of economics and business journals (sources) and universities (institutions), built from OpenAlex reference data.  The ranking is the leading eigenvector of the row-normalised citation matrix $H_{SI} H_{IS}$ (sources) and $H_{IS} H_{SI}$ (institutions), normalised by each unit's activity weight.  Four models are compared: SS (source-only), II (institution-only), B (bipartite, the main model), and J (joint).

## Repository layout

```
spectral_ranking/           ranking code + tests (bipartite eigenvector, Katz-Hubbell)
spectral_ranking_bootstrap/ bootstrap code + tests
spectral_ranking_latex/     multi-file LaTeX paper (main.tex + sections/)
spectral_results_analysis/  figure scripts (fig_2.py … fig_8.py, source_communities.py, …)
prepare_data/               OpenAlex pipeline + journal classification
util/                       load_config, load_runs, Paths
data/                       small reference files (git-tracked)
plots/                      output figures (git-tracked)
params.csv                  run schedule (one row per ranking run)
config.yaml                 machine-specific data paths (gitignored)
ERRORS.md                   known LaTeX issues
PROGRESS.md                 this file
```

## Pipeline overview

```
prepare_data pipeline  →  edge_lists.duckdb  →  ranking pipeline  →  rankings.duckdb
                                                                              ↓
                                                              spectral_results_analysis/
                                                              run_results_pipeline.sh
```

Run order:
1. `prepare_data/run_prepare_data_pipeline.sh` (includes Stage 7: filter_mode_units.py)
2. `spectral_ranking/run_rankings.py`
3. `spectral_results_analysis/run_results_pipeline.sh`

## Current state of the code (as of 2026-04-22)

### Ranking runs complete
All 22 runs in `params.csv` have been executed, including baseline (2000–2024), four time windows (t1–t4, 2000–04 through 2020–24) and their SS variants.

### Figures produced
| Script | Output | Status |
|---|---|---|
| `fig_2.py` | Field subset comparison | Complete |
| `fig_3.py` | Rank curves + scatter (vSS/vII vs v_bip) + unit effects table | Complete |
| `fig_4.py` | λ₂ community separation | Complete |
| `fig_5.py` | Parameter sensitivity (τ, ρ, α, χ) | Complete |
| `fig_7.py` | Bootstrap uncertainty clouds | Complete; `--boot bootstrap_oa_errors` flag added |
| `fig_8.py` | JCR validation | Complete |
| `source_communities.py` | Leiden community detection | Complete |
| `fig_stability.py` | Stability across time windows | Complete |
| `fig_log_ratio_stability.py` | log₂(v_SS/v_bip) by community | Complete |

### Bootstrap
> **2026-11-26:** superseded. All error studies now come from `bootstrap_oa_errors.py` (filters `resample`, `source`, `institution`, `reference`, plus `year_swap`), run by `run_error_studies.sh`; see `BOOTSTRAP.md`. `bootstrap_baseline.py` and `fig_7.py` are archived in `ZARCHIVE/bootstrap_2026_11/`. The notes below are historical.

Two independent bootstrap implementations:

**`bootstrap_baseline.py`** — sampling bootstrap (work/source/institution resampling, 80% without replacement).  Tests in `tests/test_bootstrap_baseline.py`.  Output: `$WORKING/bootstrap/`.

**`bootstrap_oa_errors.py`** — OpenAlex wrong-reference error model.  With probability p (default 0.05) each reference is redirected to a randomly drawn work from the same source and publication year (its _like-set_).  C_IS is provably unchanged (same source, same year); only C_SI is rebuilt per replicate using the vectorised formula S_cw @ W_cw.  Tests in `tests/test_bootstrap_oa_errors.py` (15 tests).  Output: `$WORKING/bootstrap_oa_errors/`.  Wrong-institution mode is stubbed with `NotImplementedError`.

Plot either bootstrap with:
```bash
python spectral_results_analysis/fig_7.py                          # sampling bootstrap
python spectral_results_analysis/fig_7.py --boot bootstrap_oa_errors
```

### Key bug fixes applied (April 2026)
- **`_A_` → `_fx` audit**: six scripts had hardcoded `_A_` in DuckDB table names (old convention where A="all sources"); fixed to use `_fx = _baseline['fx']` everywhere: `fig_3.py`, `fig_4.py`, `fig_7.py`, `fig_8.py`, `source_communities.py` (including two `build_csr()` call arguments).
- **`fig3_v1_examples.py`** merged into `fig_3.py`; archived to `zarchive/`.
- **table_unit_effects.tex** was writing to `EconomicsBusiness/tables/`; fixed to `spectral_ranking_latex/tables/`.
- **V1_BAND** for scatter-cross selection changed to log-space (`|log(v)| < 0.05`) — the scatter is log-log so a multiplicative band is correct.

### Field classification (field_eb)
Sources carry `field_eb` ∈ {E, B, A, X} in `source_master.parquet/.csv`.  Institution classification uses H_SI row-normalised weights (each source contributes equally regardless of volume) stored in `institution_field_eb.parquet`.

### Key analytical findings
- **Spectral gaps**: g_SS=0.038, g_II=0.464, g_bip=0.660, g_J=0.347.  SS mode is highly compartmentalised; bipartite gap is 17× larger.
- **Community structure**: A_SS has 8 communities (Q=0.21); A_bip has 3 (Q≈0).  Psychology cluster (94% X) consistently above-baseline in v_SS/v_bip.
- **JCR validation** (April 2026, superseded by `data/benchmark_journals_ais.csv`): AIS ∝ v^0.91 (R²=0.82); JIF ∝ v^0.55 (R²=0.52).  Eigenvector-based AIS aligns most closely.
- **Stability**: source persistence (v>1 in t1 still v>1 in t5) 83.4%; institution 93.5%.  Spearman ρ(v_t1, v_t5) ≈ 0.66 for both.  Monotone-rising institutions concentrated in Sciences Po, DIW, ECB, Singapore, Australian business schools.

## Paper status (`spectral_ranking_latex/`)

| Section | Status |
|---|---|
| §1 Introduction | Complete |
| §2 Mathematical framework | Complete |
| §3 Corpus construction | Complete (field_eb paragraph needed — see ERRORS.md) |
| §4 Results | Partly drafted: mode comparison, JCR, stability written; field/λ₂ text has [FILL] blocks; parameter and bootstrap sections not yet written |
| §5–7, Supplement | Placeholder stubs |

**Known LaTeX issues** tracked in `ERRORS.md`:
- Text corruption in `01_introduction.tex` ("institutionszz", "butz") — manual repair needed
- [FILL] placeholders in `04_results.tex` (χ* value, Fig 2 N_s counts, Fig 4 φ₂ description)
- §3 missing field_eb (E/B/A/X) classification paragraph
- Corpus size in §3 table may be stale (current baseline: n_s=1,637, n_u=2,638)
- Partition subscript _I vs _U undecided (currently _I throughout; candidate: _U to harmonise with N_u)
- §4 model label paragraph (SS/II/B/J with m-codes) not yet written

## Open decisions

1. **_I vs _U** for partition subscripts in §2 — not yet resolved.
2. **Bootstrap section** in §4 — not yet written; will use both bootstrap types.
3. **Wrong-institution bootstrap** — `preload_inst_edges` / `bootstrap_step_inst` stubbed; like-sets would be institutions in the same country.

## Environment

- Python: `.venv/bin/python` (never bare `python`)
- Data: large parquets on separate SSD; paths in `config.yaml` (gitignored)
- All 90+ tests pass: `python -m pytest spectral_ranking/ spectral_ranking_bootstrap/`
