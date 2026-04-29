# PREPARATION.md — Project briefing for a follow-on author study

**Prepared for:** A Claude instance tasked with building a project about the
*authors* of works produced within the Economics and Business corpus studied here,
their oeuvres, and the attention those oeuvres receive from the scholarly community.

---

## 1. What this project is

This project produces a **joint ordinal spectral ranking of academic journals
(sources) and research institutions** in the fields of Economics and Business.
The ranking is derived entirely from bibliometric data: which works were published
where, who authored them, and which other works they referenced.

The central quantity is the **influence score** $v_p$, one value per source and
one per institution. A unit with $v > 1$ receives more than its proportional
share of reference attention from the corpus; a unit with $v < 1$ receives less.
The $a_p$-weighted mean of all $v_p$ is 1 by construction. Roughly 28–36% of
works are published in sources with $v > 1$, and 45–50% of works have at least
one author institution with $v > 1$.

The project is documented in a LaTeX paper (`spectral_ranking_latex/`) whose
argument runs: motivation → mathematical framework → corpus construction →
ranking results. Sections 1–3 are complete; Sections 4–6 contain partial results
and placeholders for remaining analysis.

---

## 2. Project layout

```
EconomicsBusiness/
  prepare_data/               # pipeline: registries → OA match → parquets → edge lists
  spectral_ranking/           # CSR matrix assembly + ranking algorithms
  spectral_results_analysis/  # figure and table scripts for the paper
  spectral_ranking_latex/     # multi-file LaTeX paper
    main.tex                  # master file; compiles cleanly
    sections/                 # 01_introduction … 07_addenda
    tables/                   # \input{} fragment files produced by table scripts
  util/                       # load_config(), load_runs(), Paths
  data/                       # small git-tracked reference files (CSVs, .tex fragments)
  plots/                      # all figure outputs (git-tracked PDFs/CSVs)
  params.csv                  # run schedule: one row per ranking run
  config.yaml                 # machine-specific paths (gitignored)
  CLAUDE.md                   # authoritative project conventions
```

Always use `.venv/bin/python`. Paths come from `util.load_config()` which reads
`config.yaml` and returns a `Paths` object with attributes:
- `paths.working` — SSD working directory (holds DuckDB files and `parquet/`)
- `paths.parquet` — `paths.working / 'parquet'`
- `paths.openalex` — OpenAlex snapshot directory
- `paths.plots` — `plots/` in the project root
- `paths.tables` — `spectral_ranking_latex/tables/`

---

## 3. Data sources

### 3a. External: OpenAlex snapshot (Feb 2026)
Location: `paths.openalex` (SSD, not git-tracked).
Full OpenAlex data snapshot in Parquet format, one subdirectory per entity type.
The corpus was extracted from it by `load_corpus_entities.py`.

### 3b. Journal registries (small, git-tracked in `data/`)
Three registries identify Economics/Business journals:
- **ERA 2023 SJL** — Field of Research codes; kept FoR divisions 35 and 38.
- **Harzing JQL (71st ed.)** — Peer quality assessments; used as-is.
- **Clarivate WOS JCR** — WOS categories Economics, Management, Business, etc.

Their union, matched to OpenAlex sources by ISSN then exact title, is **OAS\***.
After a topic-density filter (≥ 40% of OA topic counts in Econ/Business) the
final source list is **OAS** (`source_master.parquet`, 1,595 sources).

---

## 4. Parquet files (`paths.parquet`)

Intermediate and final pipeline outputs, stored on the SSD.

| File | Rows | Key columns | Notes |
|---|---|---|---|
| `source_master.parquet` | 1,595 | `source_idx`, `source_name`, `issn`, `field_eb`, `econ_bus_density`, `era_for_codes`, `harzing_field`, `wos_categories` | Final OAS. `field_eb` ∈ {E, B, A, X} classifies each source. |
| `corpus_works.parquet` | 1,443,647 | `work_idx`, `source_idx`, `title`, `doi`, `publication_year`, `authors_count`, `institutions_distinct_count`, `referenced_works_count`, `cited_by_count`, `type`, `volume`, `issue`, `first_page`, `last_page` | Articles + reviews, 2000–2024, in OAS sources. `work_idx` = OpenAlex integer work ID. |
| `corpus_authorships.parquet` | 3,194,949 | `work_idx`, `author_idx`, `author_name`, `institution_idx`, `institution_name`, `ror`, `country_code` | One row per (work, author, institution) triple. 862,007 distinct `author_idx`; 39,308 distinct `institution_idx`. A work with 3 authors across 2 institutions generates up to 6 rows. |
| `corpus_references.parquet` | 17,135,773 | `citer_idx`, `cited_idx` | Intra-corpus reference pairs (both values are `work_idx` in `corpus_works`). |
| `corpus_institutions.parquet` | — | institution metadata + `works_per_year` | Used for τ_I retention analysis. |
| `institution_field_eb.parquet` | — | `institution_idx`, `frac_E`, `frac_B`, `frac_A`, `frac_X`, `field_eb` | Institution field classification derived from citation-weight fractions. |
| `comprehensive_journal_list.parquet` | — | — | Registry union before OA matching. |
| `oas_star.parquet` | — | — | OA-matched long-list before topic-density filter. |
| `source_densities.parquet` | — | `source_idx`, `econ_bus_density` | Topic density per source. |

---

## 5. DuckDB files (`paths.working`)

### `edge_lists.duckdb`

One table per corpus configuration. Naming pattern:
```
el_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_{vartau|fixtau}
```

Each row is one (citing work) × (institution of citing work) × (cited work) ×
(institution of cited work) combination. A work with 3 citing institutions × 2
cited institutions generates 6 rows per reference.

**Edge list column schema:**

| Column | Type | Meaning |
|---|---|---|
| `citer_work_idx` | BIGINT | Citing work (OpenAlex integer ID) |
| `citer_source_idx` | BIGINT | Source of citing work |
| `citer_inst_idx` | BIGINT | One institution of citing work |
| `cited_work_idx` | BIGINT | Cited work |
| `cited_source_idx` | BIGINT | Source of cited work |
| `cited_inst_idx` | BIGINT | One institution of cited work |
| `inst_weight` | DOUBLE | ω_iu author-fractional weight, citing side |
| `direct_inst_weight` | DOUBLE | 1 / n_retained_institutions, citing side |
| `cited_inst_weight` | DOUBLE | ω_jv author-fractional weight, cited side |
| `direct_cited_inst_weight` | DOUBLE | 1 / n_retained_institutions, cited side |
| `R_i` | BIGINT | Intra-corpus reference count of citing work |
| `a_citer_source` | BIGINT | Total work count of citing source in corpus |
| `a_cited_source` | BIGINT | Total work count of cited source in corpus |
| `a_citer_inst` | DOUBLE | Fractional work count of citing institution |
| `a_cited_inst` | DOUBLE | Fractional work count of cited institution |

Companion unit-index tables (one per run):
```
_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_{vartau|fixtau}_m{mstr}
```
Columns: `unit_idx`, `unit_type` ('S' or 'U'), `a_p`.

### `rankings.duckdb`

One table per completed ranking run. Naming pattern:
```
rk_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_{vartau|fixtau}_rho{rho}_m{mstr}_chi{chi_str}_alpha{alpha_int}
```

**Ranking table column schema:**

| Column | Meaning |
|---|---|
| `unit_idx` | OpenAlex integer ID of source or institution |
| `unit_type` | 'S' = source, 'U' = institution |
| `pi` | Ranking probability (stationary distribution of Markov walk over citation network) |
| `v` | Influence: v_p = (A / a_p) × π_p, normalised so the a_p-weighted mean = 1 |
| `rank_pi` | Cardinal rank by π |
| `rank_v` | Cardinal rank by v |
| `a_p` | Work count for this unit (source: integer; institution: fractional) |

A `_catalog` table records run parameters and diagnostics for every completed run.

**Baseline run** (most results in the paper use this):
- `run_code = '20242024'`, `fx = 'EBAX'`, `tau_u = tau_s = 20`
- `rho = 0` (fractional reference counting), `m = '0110'` (bipartite SI/IS)
- `chi = 0.5`, `alpha = 1.0`, `mu_type = ''`
- Table: `rk_20242024_EBAX_tauU20_tauS20_vartau_rho0_m0110_chi50_alpha100`
- 1,084 sources and 1,733 institutions retained after τ and SCC filters

---

## 6. Run schedule (`params.csv`)

Each row is one spectral ranking run. Key columns:

| Column | Meaning |
|---|---|
| `skip` | 1 = skip |
| `run_code` | 8-char: last-2-digits of tc0,tc1,tt0,tt1 |
| `tc0`,`tc1` | Census window (works that make references) |
| `tt0`,`tt1` | Target window (works that receive references) |
| `fx` | Field filter: EBAX (all), E, B, A, EBA, X |
| `tau_u`,`tau_s` | Min mean annual works for institution/source retention |
| `rho` | 0 = fractional reference counting; 1 = full counting |
| `m` | Block mask: 0110 = bipartite, 1000 = SS only, 0001 = II only, 1111 = full joint |
| `chi` | Source/institution mixing: 0.5 = unit balance; −1 = χ* (resolved at runtime) |
| `alpha` | Damping (1.0 = pure Perron; 0.85 = Katz–Hubbell with prior) |
| `mu_type` | Prior: '' = zero, 'uniform', 'unit_scaled' |
| `label` | Human name: 'baseline', 't1'…'t4', 'F=E', etc. |
| `ref_units` | Non-empty → fixtau run: inherits unit set from this reference window |

Time-series runs: `t1`=2000–04, `t2`=2005–09, `t3`=2010–14, `t4`=2015–19.
`-fix` variants (t1-fix…t4-fix) use the baseline 2020–24 unit universe.

---

## 7. Analysis pipeline

### Stage 1 — Journal list assembly (`prepare_data/`)
```
journal_assembler_era_harzing_wos.py  →  comprehensive_journal_list.parquet
journal_filter_match_oa.py            →  oas_star.parquet, source_master.parquet, ...
load_corpus_entities.py               →  corpus_works, corpus_authorships,
                                         corpus_references, corpus_institutions
institution_retention.py              →  retention plots (τ_I selection)
build_institution_field_eb.py         →  institution_field_eb.parquet
build_edge_lists.py                   →  edge_lists.duckdb  (el_* + _units_* tables)
filter_mode_units.py                  →  _units_*_m{mstr} tables in edge_lists.duckdb
table_maker.py                        →  data/ table fragments + CSV summaries
```

### Stage 2 — Spectral ranking (`spectral_ranking/`)
```
run_rankings.py   reads edge_lists.duckdb
                  calls build_csr.py     →  sparse CSR matrices (C_SS, C_SI, C_IS, C_II)
                  calls katz_ranker.py   →  bipartite_resolvent() or katz() iteration
                  writes rankings.duckdb    (rk_* tables + _catalog)
```

### Stage 3 — Figures and tables (`spectral_results_analysis/`)

| Script | Output | Paper content |
|---|---|---|
| `fig_2.py` | `fig_2*.pdf` | v rank curves by field (E/B/A/X) |
| `fig_3.py` | `fig_3*.pdf` | v curves by block mode m (SS/II/bipartite/joint) |
| `fig_4.py` | `fig_4.pdf` | Community structure φ₂ vs v |
| `fig_5.py` | `fig_5*.pdf` | Sensitivity: τ, ρ, α, ω, census window |
| `fig_6.py` | `fig_6*.pdf` | Time-series: baseline vs t1–t4 fixed universe |
| `fig_7.py` | `fig_7.pdf` | Bootstrap uncertainty bands |
| `table_kernel_structure.py` | `tables/table_kernel_structure.tex` | J̄, J_col, K̄ per block |
| `source_communities.py` | CSVs + `source_communities.pdf` | SCC / spectral community analysis |

---

## 8. Mathematical essentials for the follow-on project

The **influence** $v_p$ for unit $p$ (source or institution) is:

$$v_p = \frac{A}{a_p}\,\pi_p, \qquad A = \sum_q a_q$$

where $\pi_p$ is the ranking probability (stationary distribution of the Markov
walk over the citation network) and $a_p$ is the work count of unit $p$ (integer
for sources, fractional for institutions). The normalisation ensures the
$a_p$-weighted mean of all $v_p$ equals 1. A unit with $v_p = 2$ has twice the
corpus-average influence.

The bipartite model (baseline, `m=0110`) routes attention:
- Source → Institution via $C_{SI}$
- Institution → Source via $C_{IS}$

$\chi = 0.5$ splits a work's attention equally between its source and institutions.
Institution weight $\omega_{iu}$ (author-fractional, eq. 2 in paper) distributes
a work's institution contribution among its retained authors and their institutions.

**Citation convention (critical):** $C_{ij}$ = attention *from* row $i$ (citing)
*to* column $j$ (cited). Row sum = references *given*. This is the transpose of
the economist's convention.

---

## 9. Key facts for the author project

### What is available about authors

All author data comes from OpenAlex via `corpus_authorships.parquet`:

- `author_idx` (BIGINT) — OpenAlex integer author ID; stable identifier across works
- `author_name` (VARCHAR) — OpenAlex display name at time of snapshot
- `work_idx` — links to `corpus_works.parquet`
- `institution_idx` — links to `institution_field_eb.parquet` and the ranking
- `ror` — Research Organization Registry ID for the institution
- `country_code` — country of the institution

**Scale:** 862,007 distinct `author_idx` values across 3,194,949 authorship rows
for 1,443,647 corpus works over 2000–2024.

### Connecting authors to influence scores

An author's works are linked to source and institution influence via two joins:

1. `corpus_authorships` → `corpus_works` on `work_idx` → `source_idx`
   → ranking table on (`unit_idx = source_idx`, `unit_type = 'S'`) → `v`

2. `corpus_authorships` on (`author_idx`, `work_idx`) → `institution_idx`
   → ranking table on (`unit_idx = institution_idx`, `unit_type = 'U'`) → `v`

Note: not all institutions pass the τ_I threshold and appear in the ranking.
39,308 distinct institutions are in `corpus_authorships`; only ~1,733 appear
in the baseline ranking (those averaging ≥ 20 works/year in 2020–24).

### What "oeuvre" and "attention" mean operationally

- **Oeuvre** of author $a$: all `work_idx` values in `corpus_authorships` where
  `author_idx = a`. Works outside 2000–2024 are not present.

- **Attention to an oeuvre**: two options:
  1. `corpus_works.cited_by_count` — OpenAlex total citation count
     (includes citations from all sources globally, not just the OAS corpus).
  2. Intra-corpus attention: count rows in `corpus_references` where
     `cited_idx` is in the author's oeuvre. This is the citation signal that
     directly drives the spectral ranking.

- **Influence attributable to an author**: influence scores are at the source and
  institution level, not the author level. The follow-on project must decide how
  to aggregate — e.g. average v of sources in which the author publishes, maximum
  v of their institutions, or a publication-weighted mean.

### Important data caveats

- **Author disambiguation**: OpenAlex performs its own disambiguation. `author_idx`
  identifies a disambiguated author entity, but merges and splits occur,
  particularly for common names.
- **Institutional affiliation**: reflects affiliation recorded in the OpenAlex
  snapshot per work. An author who moved institutions has different `institution_idx`
  values for works from different periods.
- **Coverage**: the corpus covers only OAS sources. Authors who also publish in
  other fields or lower-output journals have incomplete oeuvres in this data.
- **Time window**: 2000–2024 only. Earlier works exist in OpenAlex but were not
  extracted into `corpus_works`. They appear in `corpus_references.cited_idx` if
  cited by corpus works, but `corpus_authorships` has no rows for them.
- **Highly multi-authored works**: some works have 100+ institutions (max 129 in
  the baseline corpus). The authorship explosion from multi-authored works is a
  practical data-engineering consideration for the author project.

---

## 10. LaTeX paper

Master: `spectral_ranking_latex/main.tex`. Compiles with `pdflatex`.
Bibliography: `MyLibrary.bib` (Zotero export), in the same directory.
Do not read folders named `zarchive`, `archive`, or `ARCHIVE`.

| Section file | Status | Content |
|---|---|---|
| `01_introduction.tex` | Complete (minor text corruption) | Motivation, literature, road-map |
| `02_model_specification.tex` | Complete | Mathematical framework: C matrices, bipartite resolvent, v definition |
| `03_processing.tex` | Complete | Corpus construction, τ thresholds, field_eb classification |
| `04_results.tex` | Partial (many [FILL] placeholders) | Kernel structure, field effects, sensitivity, time-series |
| `05_discussion.tex` | Placeholder | Face/content/criterion/construct validity |
| `06_prospects.tex` | Placeholder | Future work — author-level analysis explicitly signalled |
| `07_addenda.tex` | Placeholder | Supplementary material |

Tables live in `spectral_ranking_latex/tables/`. Each `.tex` fragment includes
the full `\begin{table}…\end{table}` wrapper and is included via
`\input{../tables/…}`.

The paper explicitly signals the author follow-on project:
> "We intend to apply the approach to study the relations between the status of
> journals and institutions, and researcher esteem in later work."

---

## 11. Conventions that carry forward

- **Influence, not prestige**: $v$ and $\pi$ are called *influence* and *ranking
  probability*. Never use the word "prestige" — it was deliberately retired.
- **Reference vs citation**: a work *references* its reference list; it is
  *cited by* works that list it. Use "reference" for giving attention, "citation"
  only when precise.
- **Attention** = what a citing work gives. **Influence** = what a cited unit
  accumulates.
- **C_{ij} convention**: row = citing (attention giver), column = cited
  (influence receiver). Transpose of the economist's convention.
- **field_eb** classifies sources (E/B/A/X) from registry keyword counts.
  Institutions are classified by citation-weight fractions (quota rule).
- Python environment: always `.venv/bin/python`. Never bare `python`.
- No `constrained_layout` in matplotlib; use `subplots_adjust` instead.
