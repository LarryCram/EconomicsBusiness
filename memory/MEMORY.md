# EconomicsBusiness Project Memory

## Pipeline: prepare_data/

Four scripts run in order:

1. `journal_assember_era_harzing_wos.py` → `comprehensive_journal_list.parquet`
2. `journal_filter_match_oa.py` → `oas_star.parquet`, `source_densities.parquet`, `dropped_sources.parquet`, `unmatched_journals.parquet`, `source_master.parquet`
3. `load_corpus_entities.py` (renamed from `load_works_authorships_references.py`) → `corpus_works.parquet`, `corpus_authorships.parquet`, `corpus_references.parquet`, `corpus_institutions.parquet`
4. `table_maker.py` → LaTeX/CSV/PDF tables in `data/`

## Key design decisions

**ERA SJL filter**: FoR 1 must start with '35' or '38'; any present FoR 2/3 must also start with '35' or '38' or be absent. Works for both 2-digit and 4-digit codes.

**WOS MJL filter**: inclusive — Category IN ('ECONOMICS', 'MANAGEMENT', 'BUSINESS', 'BUSINESS, FINANCE', 'TRANSPORTATION')

**OA matching**: ISSN join first, then exact title match (LOWER equality) for unmatched registry journals. Both land in OAS*.

**Topic density filter**: SUM(Field 14 + Field 20 topic counts) / SUM(all topic counts) >= 0.4. Sources with no topic assignments (NULL density) are kept. Field names: 'Economics, Econometrics and Finance' and 'Business, Management and Accounting'.

**OAS* → OAS**: density filter; `source_master.parquet` is the final OAS.

**Institution threshold τ_U**: works_per_year > 20 (hardcoded as TAU_U in table_maker.py). Sensitivity runs planned at 5 and 15. works_per_year = works_count / 25 (CORPUS_YEARS). Drops ~25% of works.

## Parquets saved (all under WORKING/parquet/)
- `comprehensive_journal_list.parquet` — registry union (ERA+Harzing+WOS)
- `oas_star.parquet` — OA-matched long-list (pre-topic-filter)
- `source_densities.parquet` — econ_bus_density for all OAS* sources
- `dropped_sources.parquet` — sources removed by density filter, with top topic
- `unmatched_journals.parquet` — registry journals not matched to OA, with best Jaro-Winkler title candidate
- `source_master.parquet` — final OAS with top topic and density
- `corpus_works.parquet` — articles/reviews in OAS, 2000-2024
- `corpus_authorships.parquet` — author-institution pairs for corpus works
- `corpus_references.parquet` — intra-corpus reference pairs (both citer and cited in corpus)
- `corpus_institutions.parquet` — institutions in corpus_authorships with works_count, works_per_year, institution_name, country_code, type

## corpus_institutions.parquet notes
- institution_idx derived from OA id by stripping 'https://openalex.org/I' prefix
- display_name aliased to institution_name
- OA institutions snapshot is a single file: `{OPENALEX}/institutions.parquet` (not a glob)

## table_maker.py outputs (data/)
- `table2_source_matching.tex/.csv/.pdf`
- `table_exclusions.tex` + `unmatched_classified.csv`
- `table_dropped_sources.tex/.csv`
- `sjl_dropped_sources.csv`
- `title_match_candidates.csv`
- `table3_corpus_features.tex/.csv/.pdf` — Table 3: corpus features

## Table 3 structure
Two panels, 7 columns (label + D1/D9 × 3 year columns: 2000, 2024, 2000-24):
- Upper panel: unique-item counts (Works, Sources, Institutions, Reference counts, Citation counts). Counts span both sub-columns with \multicolumn.
- Lower panel: $D_1$ and $D_9$ deciles (10th/90th percentiles) of: works/source, institutions/work, references/work (out-degree), citations/work (in-degree).
- 2000-24 column pools full population (not annual average).
- works/source pooled = COUNT(DISTINCT work_idx) per source across full period.
- Institution filter (τ_U > 10) applied to all rows via filtered corpus fc.

## Table 3 edge-effect semantics
- references/work: depressed in 2000 (corpus references pre-2000 not in corpus), high in 2024
- citations/work: high in 2000 (24 years of citers available), depressed in 2024 (future citers outside corpus)
- citations/work year column = in-degree of works published that year, citing from ANY corpus year (2000-2024)
- User plans to note these edge effects in the paper text

## config.yaml keys
PROJECT_ROOT, DATA (relative), WORKING (SSD path), OPENALEX (SSD path to OA parquet snapshot)

## field_eb column (replaces field_subset)

`source_master.parquet` and all related outputs (`oas_star.parquet`, `dropped_sources.parquet`,
`unmatched_journals.parquet`, `source_master.csv`) now have `field_eb` instead of `field_subset`.
Computed from multi-registry scoring: `era_field`, `harzing_field`, `wos_categories`, `field_name`.
- 'E': econ_score≥2 AND bus_score<1
- 'B': bus_score≥2 AND econ_score<1
- 'A': both signals present (econ_score≥2 AND bus_score≥1, or vice versa)
- NULL: neither signal strong
F=EB corpus filter is `field_eb IN ('E','B','A')` (previously was `IN ('E','B')`; 'A' is now included).

`era_field`, `harzing_field`, `wos_categories` are also present as columns in all those outputs.

## Run schedule — params.csv (no stage column)

The run schedule for `run_rankings.py` is driven by `params.csv` (project root).
Columns: `skip, run_code, tc0, tc1, tt0, tt1, fx, tau_u, tau_s, rho, m, chi, alpha, mu_type, label`.
There is NO `stage` column. There is NO `--stage` CLI argument.
`run_code` is an 8-char string: last-2-digits of tc0,tc1,tt0,tt1 (e.g. '20242024', '00040004').
`chi=-1` signals χ* (resolved at runtime).
`runs_from_csv()` returns all non-skipped rows; `load_runs()` applies int/float type conversions.

Time-series runs (t1–t4) are identified by label, not by a stage parameter:
- t1: run_code=00040004 (2000–04)
- t2: run_code=05090509 (2005–09)
- t3: run_code=10141014 (2010–14)
- t4: run_code=15191519 (2015–19)
Baseline: run_code=20242024 (2020–24).

## Figure inventory (current)

| Figure | Script | Status | Description |
|---|---|---|---|
| fig_2 | fig_2.py | Created | F field comparison; legend on both panels |
| fig_3 | fig_3.py | In development | Mode comparison (SS/II/bipartite) |
| fig_4 | community.py | Complete | Second eigenpair φ₂ community analysis |
| fig_5 | fig_5.py | In development | Phase 2 sensitivity (τ, ρ) |
| fig_6 | fig_6.py | Created | Time-series comparison t1–t4 vs baseline 2020–24 |
| fig_7 | fig_7.py (planned) | Planned | Bootstrap uncertainty (spec in BOOTSTRAP.md) |

fig_6 layout: two panels (sources top, institutions bottom), x-axis = baseline rank,
colours purple/blue/green/red (t1–t4) + black baseline, log-space rolling mean curves.
