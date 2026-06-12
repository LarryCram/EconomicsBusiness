# New Project: Generalised Spectral Ranking

## Purpose

Generalise the EconomicsBusiness spectral ranking to cover all fields, producing
source/institution and topic/institution rankings for world research output.

Motivated by the ERA finding that journals belong to multiple fields — the topic
projection handles this structurally by working at the work level.

## Core abstraction: seed sets

A **set** defines the left-side nodes of a bipartite ranking. Two types, fully parallel:

| type | left entities | left←work join | output |
|------|--------------|----------------|--------|
| `journal` | journals {S} in a discipline | `work.source_id` (one per work) | v_S, v_I |
| `topic` | topics {T} in an OA subfield | `work.topic_ids ∩ {T}` (multiple per work) | v_T, v_I |

The bipartite spectral machinery (build_csr, power iteration, alpha/chi) is
**identical** for both types. Only the edge-construction step differs.

## Scale

- ~2000 institutions
- ~100 journal sets (~1000 journals each, discipline-level)
- ~200 topic sets (one per OpenAlex subfield; members resolved from OA topic taxonomy)

## Data sources

- OpenAlex citation data (existing pipeline)
- `works_topics.parquet` — (work_id, topic_id) — already constructed from OA Leiden topics
- `topics_taxonomy.parquet` — (topic_id, subfield_id, ...) — from OA; clusters up to Scopus ASJC

Topic set membership is resolved at runtime: `topic_ids WHERE subfield_id = X`.
No manual enumeration needed.

## Config files

```
sets.csv   — set_id, set_type, set_name, discipline_or_subfield_id
runs.csv   — skip, set_id, window, fx, tau_u, tau_s, rho, m, chi, alpha, ...
```

`sets.csv` replaces the field-specific corpus logic. `runs.csv` replaces `params.csv`.

## Data persistence

Tiered parquets (no DuckDB required at this scale):

```
data/
  edge_lists/   el_{set_id}_{window}.parquet      # heavy; one per set×window
  units/        units_{set_id}_{window}.parquet
  rankings/     rankings_all.parquet              # small; set_id, entity_type,
                                                  # entity_id, v, rank, params
```

Rankings output is small (~1M rows total); keep as a single wide parquet for
easy cross-set queries with pandas/polars.

## What carries over from EconomicsBusiness

- `spectral_ranking/` — ranking engine (minor generalisation: `left_col` parameter)
- `prepare_data/` — OpenAlex ingestion pipeline
- `util/` — load_config, Paths, load_runs patterns

## What is jettisoned

- Field-specific corpus logic (field_eb, econ/business classification)
- `params.csv` schema
- All LaTeX / figure scripts
- vartau/fixtau distinction (may re-emerge but not a starting assumption)

## Key code change to build_csr

```python
# old: hardcoded source_id as left entity
# new: parameterised
build_csr(edges, left_col='source_id' | 'topic_id', right_col='institution_id')
```

Edge construction branches by set_type; everything downstream is unchanged.
