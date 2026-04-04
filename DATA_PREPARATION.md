# Data Preparation

## Overview

The data preparation pipeline runs in `prepare_data/` and produces the edge lists consumed
by the spectral ranking pipeline. All intermediate parquets live under `WORKING/parquet/`
(machine-specific path from `config.yaml`). Edge lists are stored in `WORKING/edge_lists.duckdb`.

## Scripts and outputs

| Script | Inputs | Outputs | Status |
|---|---|---|---|
| `journal_assembler_era_harzing_wos.py` | Registry files | `comprehensive_journal_list.parquet` | Done |
| `journal_filter_match_oa.py` | OA snapshot, registry | `source_master.parquet`, `oas_star.parquet`, diagnostics | Done |
| `load_corpus_entities.py` | OA snapshot, `source_master.parquet` | `corpus_works.parquet`, `corpus_authorships.parquet`, `corpus_references.parquet` | Done |
| `institution_retention.py` | Corpus parquets | Diagnostic tables for τ_U selection | Done |
| `build_edge_lists.py` | Corpus parquets, `source_master.parquet` | `edge_lists.duckdb` (21 tables) | Done |
| `table_maker.py` | Parquets, DuckDB | LaTeX/CSV tables in `data/` | Done |
| `verify_edge_lists.py` | `edge_lists.duckdb` | Verification report | Done |

## Edge list schema

Each table in `edge_lists.duckdb` is named `el_t{tx}_{fx}_tau{tau_u}` and has one row
per `(citer_work, citer_institution, cited_work, cited_institution)`:

```
citer_work_idx       BIGINT   -- citing work
citer_source_idx     BIGINT   -- source of citing work
citer_inst_idx       BIGINT   -- retained institution of citing work
cited_work_idx       BIGINT   -- cited work
cited_source_idx     BIGINT   -- source of cited work
cited_inst_idx       BIGINT   -- retained institution of cited work
inst_weight          DOUBLE   -- ω_iu author-fractional weight (citing side, eq. 1)
direct_inst_weight   DOUBLE   -- 1/n_retained_institutions (citing side)
R_i                  BIGINT   -- intra-corpus reference count of citing work
a_citer_source       BIGINT   -- work count of citing source
a_cited_source       BIGINT   -- work count of cited source
a_citer_inst         DOUBLE   -- fractional work count of citing institution (Σ ω_iu)
a_cited_inst         DOUBLE   -- fractional work count of cited institution (Σ ω_jv)
```

## Known issues — fix before running spectral ranking

### 1. `params.yaml` case 6 census window is wrong
`params.yaml` has `6: {census: [2020, 2024], ...}` but the paper and DATA_PREPARATION.md
specify census=[2024, 2024] (single year, JIF analogue). Must be corrected to `[2024, 2024]`
and `build_edge_lists.py` re-run for t_x=6.

### 2. `build_edge_lists.py` does not enforce census/target window split
The `rr` CTE (reference pairs) joins both citer and cited to `fw`, which spans
`[min(cs,ts), max(ce,te)]`. For symmetric cases (t_x=1–5) this is harmless since
census=target. For asymmetric cases:
- **t_x=6**: Should require `citer_year=2024` and `cited_year ∈ [2000,2024]`. Currently
  all works in [2000,2024] are eligible as citers.
- **t_x=7**: Should require `citer_year ∈ [2020,2024]` and `cited_year=2020`. Currently
  any cited year in [2020,2024] is eligible.
Fix: add year-range predicates to the `rr` CTE using `wc.publication_year` and
`wd.publication_year`.

### 3. Institution retention uses wrong denominator for asymmetric windows
`retained_inst` filters on `COUNT(DISTINCT work_idx) / n_years >= tau_u`, where
`n_years = max_year - min_year + 1` (span of both windows). DATA_PREPARATION.md specifies
the denominator should equal the length of the *census* window. For t_x=6 (after fixing
to census=[2024,2024]), this means dividing by 1, not 25.
Fix: compute `census_years = ce - cs + 1` and use that as the denominator in
`retained_inst`, and restrict work counting to census-window works only.

### 4. `cited_inst_weight` (ω_jv) is absent from the edge list
Building C_SI and C_II requires ω_{jv} — the author-fractional institution weight for
the *cited* work/institution pair. The current schema stores only `inst_weight` = ω_{iu}
for the citing side. `a_cited_inst` is the total output (Σ_v ω_{jv}), not the per-work weight.
Fix options:
- Add `cited_inst_weight` column to the final SELECT in `build_edge_lists.py` by joining
  the `iw` CTE on the cited side.
- Persist a separate `(work_idx, inst_idx, inst_weight)` lookup table from the `iw` CTE
  for each corpus, joinable at ranking time.

Issues 1–3 require re-running `build_edge_lists.py` for at least cases 6 and 7 after
correcting `params.yaml`. Issue 4 requires either a re-run or a join at ranking time.

## Parameter design

### Field subsets (F)
`source_master.parquet` has a `field_eb` column computed from multi-registry scoring
(`era_field`, `harzing_field`, `wos_categories`, `field_name`):
- `'E'`: econ_score ≥ 2 AND bus_score < 1 (economics-only)
- `'B'`: bus_score ≥ 2 AND econ_score < 1 (business-only)
- `'A'`: both signals present (econ_score ≥ 2 AND bus_score ≥ 1, or vice versa)
- NULL: neither signal strong

The F=EB corpus filter is `field_eb IN ('E','B','A')`.
The `build_edge_lists.py` `FIELD_COND` dict translates field subset labels to SQL predicates.

### Time windows (t_x)
Seven cases defined in `params.yaml`. Cases 1–5 are symmetric 5-year windows. Cases 6–7
are asymmetric (see issues above).

### Institution threshold (τ_U)
`tau_u_floor` in `params.yaml` sets per-field-subset floors: all fields = 20 mean works/year.
The final τ_U was chosen as 20 for F=A from the institution retention diagnostic curve
(retention plot: ~1,734 institutions retained, ~85% of works before SCC filtering).

### Source threshold (τ_S)
Final τ_S = 20 mean works/year for sources.

## Corpus characteristics (baseline: t_x=5, F=A, τ_U=20, post-SCC filter)
- Sources: N_s = 1,322
- Institutions: N_u = 1,732
- χ* = 0.567 (N_u / (N_s + N_u))

Note: Final corpus numbers reflect singleton SCC filtering implemented in `build_edge_lists.py`. 
This dropped 108+7 sources and 2 institutions that were not in the giant strongly connected component.

Note: SCC filtering in `build_edge_lists.py` runs after τ_U retention and removes
sources not in the giant SCC of C_SS and institutions not in the giant SCC of C_full.
