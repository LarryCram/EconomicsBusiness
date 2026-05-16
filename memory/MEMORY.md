# EconomicsBusiness Project Memory
# Last updated: 2026-05-16 (audit pass)

## Pipeline: prepare_data/

Three scripts run in order:

1. `journal_assembler_era_harzing_wos_scopus.py` → `comprehensive_journal_list.parquet`
2. `journal_filter_match_oa.py` → `oas_star.parquet`, `source_densities.parquet`, `dropped_sources.parquet`, `unmatched_journals.parquet`, `source_master.parquet`
3. `table_maker.py` → LaTeX/CSV tables in `data/`

## Key design decisions

**ERA SJL filter**: FoR 1 must start with '35' or '38'; any present FoR 2/3 must also start with '35' or '38' or be absent. Works for both 2-digit and 4-digit codes.

**WOS MJL filter**: inclusive — Category IN ('ECONOMICS', 'MANAGEMENT', 'BUSINESS', 'BUSINESS, FINANCE', 'TRANSPORTATION')

**Scopus SSL filter**: ASJC code 1400 (Business, Management and Accounting) or 2000 (Economics, Econometrics and Finance), including inactive titles.

**OA matching**: ISSN join first, then exact title match (LOWER equality) for unmatched registry journals (~60 additional). Both land in OAS*.

**Topic density filter**: SUM(Field 14 + Field 20 topic counts) / SUM(all topic counts) >= 0.4. Sources with no topic assignments (NULL density) are kept.

**OAS* → OAS**: density filter; `source_master.parquet` is the final OAS.

**Baseline thresholds**: τ_S=5 works/source/year, τ_I=10 works/institution/year (from params.csv baseline row).

## field_eb classification (current — matches code in journal_filter_match_oa.py)

Scores count how many of {era_for_codes, harzing_field, wos_categories, field_name, scopus_asjc} signal econ or business:
- **'X'**: econ_score + bus_score < 2  (weak signal in both — combined < 2)
- **'A'**: econ_score == bus_score      (tied; includes e=b=1)
- **'E'**: econ_score > bus_score       (economics-dominant)
- **'B'**: bus_score > econ_score       (business-dominant)

`field_eb` is never NULL. F=EBAX corpus uses all; F=EBA excludes X; F=E/B/A are single-field.

NOTE: Section 3 of the paper describes wrong thresholds (≥3) — the paper text does not match the code. This is an open issue in ERRORS.md.

## Parquets saved (all under WORKING/parquet/)
- `comprehensive_journal_list.parquet` — registry union (ERA+Harzing+WOS+Scopus); columns include scopus_journal_name, scopus_asjc
- `oas_star.parquet` — OA-matched long-list (pre-topic-filter)
- `source_densities.parquet` — econ_bus_density for all OAS* sources
- `dropped_sources.parquet` — sources removed by density filter, with top topic
- `unmatched_journals.parquet` — registry journals not matched to OA, with best Jaro-Winkler title candidate
- `source_master.parquet` — final OAS with top topic, density, and field_eb

## table_maker.py outputs (data/)
- `table2_source_matching.tex/.csv` — Table 2: registry matching stats + OAS* pairwise overlaps
- `table_exclusions.tex` + `unmatched_classified.csv` — combined exclusion reasons
- `table_dropped_sources.tex/.csv` — OAS* → OAS drops by field/subfield
- `sjl_dropped_sources.csv` — SJL sources dropped by topic filter with density
- `title_match_candidates.csv` — Jaro-Winkler title match candidates

## config.yaml keys
PROJECT_ROOT, DATA (relative), WORKING (SSD path), OPENALEX (SSD path to OA parquet snapshot)

## Run schedule — params.csv

Columns: `skip, run_code, tc0, tc1, tt0, tt1, fx, tau_u, tau_s, rho, omega, m, chi, alpha, mu_type, label, ref_units, epsilon`
- `run_code`: 8-char string (last-2-digits of tc0,tc1,tt0,tt1), e.g. '20242024', '00040004'
- `chi=-1` signals χ* (resolved at runtime)
- `epsilon=1` for sentinel runs (label='baseline-eps')
- `ref_units` non-empty for fixtau runs (label ends in '-fix')

Baseline: run_code=20242024, fx=EBAX, tau_u=10, tau_s=5, rho=0, omega=0, m=0110, chi=0.5, alpha=1.0, label='baseline'

Time-series runs: t1=00040004 (2000–04), t2=05090509 (2005–09), t3=10141014 (2010–14), t4=15191519 (2015–19)
Fixtau runs: t1-fix … t4-fix (inherit unit set from baseline window)

## Figure inventory (in-paper as of 2026-05-16)

Figures in `plots/`, loaded as `../plots/NAME_latex.pdf`. All listed PDFs confirmed present.

| Label | File | Section | Description |
|---|---|---|---|
| fig_1 | fig_1_latex | S3 | Source/institution retention curves |
| fig_3by | fig_3by_latex | S4 | log(v) vs baseline rank, sources + institutions |
| fig_3bz | fig_3bz_latex | S4 | log(v^S/I) vs log(v^B) scatter, coloured by field_eb |
| fig_2c | fig_2c_latex | S4 | bipartite vs within-layer v, by subfield separately |
| fig_6 | fig_6_latex | S4 | v across five time windows vs baseline rank |
| fig_stability | fig_stability_scatter_latex | S4 | v in 2000–04 vs v in 2020–24 (log-log) |
| fig_5 | fig_5_latex | S4 | Parameter sensitivity (τ40, ρ=1) |
| fig_5d | fig_5d_latex | S4 | Parameter sensitivity (α=0.85 Katz-Hubbell) |
| fig_7 | fig_7a_oa_errors_latex | S4 | Bootstrap/OA error stochastic simulation |
| fig_enclave | enclave_referer_v | S5 | HCW influence vs median referencing influence |

fig_5f_latex.pdf EXISTS in plots/ but is NOT cited in the paper (ε=1 vs baseline). Open issue in ERRORS.md.

## Paper section status (2026-05-16)
- S1 Introduction: complete
- S2 Mathematical framework: complete
- S3 Corpus selection: complete but field_eb classification paragraph wrong (see ERRORS.md)
- S4 Results: complete but ε paragraph missing, \cite{efron} broken, fig_stability caption τ values unverified
- S5 Discussion: substantial content; "exogenous sources" undefined; Critique stub only
- S6 Prospects: "[Placeholder.]"
- S7 Addenda: empty, commented out in main.tex
