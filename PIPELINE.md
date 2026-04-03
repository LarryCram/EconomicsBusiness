# PIPELINE.md — End-to-end data and computation pipeline

## Overview

```
OA parquet snapshot
  + registry files
        │
        ▼
[A] Source selection          → data/source_master.csv  (git-tracked)
        │
        ▼
[B] Corpus loading            → WORKING/parquet/corpus_*.parquet
        │
        ▼
[C] Edge list building        → WORKING/edge_lists.duckdb
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
[D] Spectral ranking          [E] Bootstrap (~1000 runs)
        │                              │
        ▼                              ▼
WORKING/rankings.duckdb        WORKING/bootstrap/*.npy
        │                              │
        └──────────────────────────────┘
                     │
                     ▼
              fig_2 … fig_6  →  plots/fig_x*.pdf
```

Sideline (diagnostic only, feeds τ choice but not rankings):
`prepare_data/institution_retention.py` — reads corpus parquets, produces retention curves.

---

## Stage A — Source selection

| Script | Inputs | Output |
|---|---|---|
| `journal_assembler_era_harzing_wos.py` | ERA/Harzing/WoS registry files | `data/comprehensive_journal_list.parquet` |
| `journal_filter_match_oa.py` | OA snapshot, registry | `data/source_master.csv`, `data/source_master.parquet` |

`source_master.csv` has one row per retained source with `source_idx`, `source_name`,
`field_subset` ∈ {E, B, null}.  Git-tracked; consumed by edge building and all fig scripts.
**Run once; re-run only if the source list changes.**

---

## Stage B — Corpus loading

| Script | Inputs | Output |
|---|---|---|
| `prepare_data/load_corpus_entities.py` | OA snapshot parquets, `source_master.parquet` | `WORKING/parquet/corpus_works.parquet`, `corpus_authorships.parquet`, `corpus_references.parquet` |

Loads all works published in retained sources, their author–institution affiliations,
and their intra-corpus references.  No parameters other than the source list.
**Run once; large files, machine-specific SSD.**

---

## Stage C — Edge list building

**Script**: `prepare_data/build_edge_lists.py`

**Inputs**: corpus parquets, `source_master.parquet`, `params.yaml`

**Output**: `WORKING/edge_lists.duckdb`

One edge list table per corpus configuration, named:
```
el_t{tx}_{fx}_tauU{tau_u}_tauS{tau_s}
_units_t{tx}_{fx}_tauU{tau_u}_tauS{tau_s}
```

Each row of `el_*`: one `(citer_work × citer_inst × cited_work × cited_inst)` attribution,
with ω weights, R_i reference counts, and fractional work counts.
`_units_*`: unit index (source_idx / inst_idx, type, a_p) for the corpus after τ filtering and
giant-SCC pruning.

**Corpus configurations built** (driven by `params.yaml`):

| tx | fx | tau_u | tau_s | Purpose |
|---|---|---|---|---|
| 1–5 | A, E, B, EB, NEB | `tau_u_floor[fx]` | `tau_s_floor[fx]` | All time windows × field subsets |
| 5 | A | `tau_sensitivity` (40) | `tau_sensitivity` (40) | Phase 2 τ sensitivity only |

**Re-run when**: OA snapshot updated, `params.yaml` τ values change, or new corpus
configurations are added.

---

## Stage D — Spectral ranking

**Script**: `spectral_ranking/run_rankings.py`

**Inputs**: `WORKING/edge_lists.duckdb`, `params.yaml`

**Output**: `WORKING/rankings.duckdb`

One ranking table per run, named:
```
rk_t{tx}_{fx}_tauU{tau_u}_tauS{tau_s}_rho{rho}_m{m}_chi{chi_int}_alpha{alpha_int}
```
Each table has columns `unit_idx, unit_type, pi, v, rank_pi, rank_v, a_p`.
A `_catalog` table records all run parameters and diagnostics.

**Run schedule** (all cases needed for the paper):

### Stage 1 — Phase 1 and Phase 2 runs (t_x=5, all at once)

| Label | tx | fx | tau_u | tau_s | rho | m | chi | alpha | Used by |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 5 | A | 20 | 20 | 0 | 0110 | 0.50 | 1.0 | all figs |
| `F=E` | 5 | E | 20 | 20 | 0 | 0110 | 0.50 | 1.0 | fig_2 |
| `F=B` | 5 | B | 20 | 20 | 0 | 0110 | 0.50 | 1.0 | fig_2 |
| `F=EB` | 5 | EB | 20 | 20 | 0 | 0110 | 0.50 | 1.0 | fig_2 |
| `SS-only` | 5 | A | 20 | 20 | 0 | 1000 | 0.50 | 1.0 | fig_3 |
| `II-only` | 5 | A | 20 | 20 | 0 | 0001 | 0.50 | 1.0 | fig_3 |
| `full-joint-chi-star` | 5 | A | 20 | 20 | 0 | 1111 | χ* | 1.0 | fig_3 |
| `tau40` | 5 | A | 40 | 40 | 0 | 0110 | 0.50 | 1.0 | fig_5 |
| `rho1` | 5 | A | 20 | 20 | 1 | 0110 | 0.50 | 1.0 | fig_5 |
| `alpha05` | 5 | A | 20 | 20 | 0 | 0110 | 0.50 | 0.5 | fig_5 |
| `alpha05-mublock` | 5 | A | 20 | 20 | 0 | 0110 | 0.50 | 0.5 | fig_5 |

χ* is computed at runtime as N_u/(N_s+N_u) from the baseline units table.

`alpha05-mublock` uses a type-balanced prior μ_p = 1/(2N_S) for sources,
1/(2N_U) for institutions.  **Requires extending `rank()` to accept `mu`.**

### Stage 2 — Phase 4 time-series runs

| Label | tx | fx | tau_u | tau_s | rho | m | chi | alpha |
|---|---|---|---|---|---|---|---|---|
| `t1` | 1 | A | 20 | 20 | 0 | 0110 | 0.50 | 1.0 |
| `t2` | 2 | A | 20 | 20 | 0 | 0110 | 0.50 | 1.0 |
| `t3` | 3 | A | 20 | 20 | 0 | 0110 | 0.50 | 1.0 |
| `t4` | 4 | A | 20 | 20 | 0 | 0110 | 0.50 | 1.0 |

t_x=5 is the baseline (already in Stage 1).  t_x=6 and t_x=7 are dropped.

---

## Stage E — Bootstrap

**Script**: `spectral_ranking/bootstrap_baseline.py` (to be written; spec in `BOOTSTRAP.md`)

**Inputs**: `WORKING/edge_lists.duckdb` (table `el_t5_A_tauU20_tauS20`), `params.yaml`

**Output**:
```
WORKING/bootstrap/
    v_s_boot.npy    # (B, n_s) float32
    v_u_boot.npy    # (B, n_u) float32
    meta.json       # B, seed, n_s, n_u, source_ids, inst_ids
```

B=1000 replicates of the baseline bipartite ranking on 80%-with-replacement
resamples of the deduplicated SI and IS edge arrays.  Edge arrays are loaded once
from DuckDB as numpy arrays; the loop does only: sample → COO → CSR → bipartite().

Not a `run_rankings.py` run — too many replicates for the DuckDB table pattern.

---

## Output paths — figure scripts

All figure scripts are in `spectral_results_analysis/` and write to `plots/`.
They read from `rankings.duckdb` (and `edge_lists.duckdb` for institution field
labels only).  **No computation inside figure scripts** (see integrity note below).

| Script | Primary source | Output |
|---|---|---|
| `fig_2.py` | `rankings.duckdb` | F field comparison (baseline vs F=E, F=B) |
| `fig_3.py` | `rankings.duckdb` | Mode comparison (0110 vs 1000/0001/1111) |
| `fig_4.py` | `rankings.duckdb` + eigenpairs | Community measure φ₂/√v |
| `fig_5.py` | `rankings.duckdb` | Phase 2 sensitivity (τ, ρ, α) |
| `fig_6.py` | `bootstrap/*.npy` + `rankings.duckdb` | Bootstrap uncertainty |

---

## Integrity note — computations that should move into the pipeline

Two figure scripts currently bypass `run_rankings.py`:

1. **`fig_4.py`** — calls `build_csr` + `scipy.eigs` on-the-fly to compute
   the second eigenpair (φ₂, λ₂).  These are not stored anywhere.
   Fix: add a pre-compute script (e.g., `spectral_ranking/compute_eigenpairs.py`)
   that stores φ₂ and λ₂ per run in a separate `eigenpairs.duckdb` or as
   additional tables in `rankings.duckdb`.

2. **`fig_5.py`** — currently calls `build_csr` + `bipartite()` on-the-fly.
   Fix: extend `rank()` to accept `mu`, add `alpha05-mublock` to STAGE1,
   rewrite fig_5 to read from `rankings.duckdb` only.

---

## Parameter YAML — `params.yaml`

Controls corpus construction and sensitivity values.  Does not store run
schedules (those live in `run_rankings.py` STAGE1/STAGE2).

| Key | Purpose |
|---|---|
| `time_windows` | Census/target year ranges per t_x (7 defined; 1–5 active) |
| `tau_u_floor` | Default τ_U per field subset (all 20) |
| `tau_s_floor` | Default τ_S per field subset (all 20) |
| `tau_sensitivity` | Phase 2 τ sensitivity value (40) |

**Future**: replace STAGE1/STAGE2 hardcoded lists with a run table in
`params.yaml` or a CSV, so the full run schedule is data-driven.
