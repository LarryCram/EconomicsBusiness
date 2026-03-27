# ranking_spec.md — SUPERSEDED

This was the initial project specification written before the pipeline design stabilised.
It is retained for historical reference only. See SPECTRAL_RANKING.md for the current
authoritative specification and DATA_PREPARATION.md for the pipeline documentation.

---

# Summary (original)

This text describes a project about the use of ordinal spectral ranking in bibliometrics. We want to compare different ranking approaches (weighting) over networks beteen different units, namely  sources and institutions. The results will be used in a paper about dual ordinal spectral ranking over sources and institutions. The topic has not been studies very much.

# Bibliometric data.

The bibliometric data are provided as citation edge lists in a long parquet file at PARQUET_NAME. A row comprises (1) a category, citer id of a citing work, the cited id of a reference, and the weight of the reference from row i to column j. Separate lists are provided for the category units: sources s, authors a and institutions i. There are also the larger edge lists si for the citation matrix between the concatenation of sources and institutions - this coresponds to block matrices. The block edge lists are ss, ii, and si. The nodes are labelled by unique openalex entity id strings. The parquet file has a category column to pick out the various citation matrices. 

The scale of the data is roughly 500000 articles, 600 sources and 600 institutions.

# CSR format.

The edge lists of each citation matrix need to be converted in CSR format. We need dictionaries for the mapping from the compacted CSR index to the original openalex id. I want to persist these CSR matrices perhaps in parquet. The parquet columns will have long lists. The build of the CSR matrix Cij from the edge list should be the initial step and each CSR should have its category as its ID. 

# Spectral ranking and citation matrix normalisation.

I want to consider two spectral rankings, one due to Pinski and Narin (PN ranking) and one due to Geller, Bollen and West (Eigenvector - or E - ranking). The rankings differ only in the way that the citation matrix is normalised before finding its principal eigenvector. Define the row sums of Cij as Si = \sum_k Cik. Define two diagonal matrices, D1 = diag(Sj) and D2 = diag(Si). From PN, trhe nomralised matrix is M1 = D1 C. For E, the normalised matrix is M2 = D2 C. M2 is a row stochastic matrix. 

# Finding the Eigenvector. 

The eigenvector problem is w = M w where w is the principal eigenvector. Since M may not be primitive, we adopt the Katz power iteration w = \alpha Mw + (1-\alpha)v where \alpha <= 1 and v = 1/N, N is the size of M. For the PN method it is necessary to scale w so that \sum_i w_i Si = \sum_i Si. For the E method, the sum of w should be 1 sonce thery are probabilities.  

# Finding the influence per publication

The PN gives influence per reference. The influence per publication \pi_i is w_i * \sum_i S_i / a_i. For E, the influence per publication is \pi_i = w_i * \sum_j a_j /a_j. 

# Organisation.

Use a factory template for the CSR - there are cases for jj, ii, and for ji-ij with diagonal blocks and without blocks (jj=ii=0). Use a factory template for PN and E that yields bot the influence vcetore and the unfluence per article.

# Approach.

Use Python, Pandas, Numpy and duckdb as appropriate. Persist to parquet. Store working data in /home/lc/m/working/econ_bus. Decide whether to divide the project into a few separate .py files linked by persisted data or just build a workflow. Build the code in the biranking folder of the Project folder EconomicsBusiness. There is a .venv there that you will use. Use test-driven approaches if you feel that will help. I prefer minimal docstrings and compact readable code without typing or extensive try-except QA. Think carefully before coding and code like an advanced developer. If you have concerns ask me.

# Following section is provided by Calude after clarifying chat

# Journal-Institution Ordinal Spectral Ranking — Project Spec

## Overview
Construct joint ordinal spectral rankings of economics-commerce journals and institutions
using Katz iteration on citation graphs. Source list (~900 journals) combines field
coverage list with Harzing's list. Raw works downloaded from OpenAlex via openalex CLI,
processed with DuckDB, persisted to parquet. Sparse CSR matrices used throughout for
scalability to larger disciplines.

---

## data.yaml — All Constants

```yaml
# Paths
JSON_MASTER: "/home/lc/m/openalex_feb26/json"       # openalex CLI output folder
PARQUET_PATH: "/home/lc/m/openalex_feb26/parquet"   # partitioned parquet folders
                                                      #   works/, authorships/, references/
CITATION_MASTER: "/home/lc/m/openalex_feb26/citation_master.parquet"
SOURCES_FILE: "/home/lc/m/openalex_feb26/sources.parquet"

# Time windows (inclusive at both ends)
CENSUS_START: 2020
CENSUS_END: 2024
TARGET_START: 2015
TARGET_END: 2019

# Time series loop — list of [census_start, census_end, target_start, target_end]
TIME_WINDOWS:
  - [2020, 2024, 2015, 2019]
  - [2018, 2022, 2013, 2017]
  - [2016, 2020, 2011, 2015]

# Institution filtering thresholds (apply either or both)
THRESHOLD_INSTITUTION_COUNT: 5      # minimum publication count
THRESHOLD_INSTITUTION_CITES: 10.0   # minimum total citation weight

# Katz parameters
ALPHA: 0.85                          # damping factor, same for all modes
W: 0.5                               # source weight in mixed [ss,si,is,ii] mode
                                     # w=1 -> source only, w=0 -> institution only

# Iteration
KATZ_TOL: 1.0e-8
KATZ_MAX_ITER: 200
```

---

## Data Pipeline

### Step 1 — Source list
```python
import pandas as pd
sources = pd.read_parquet(config['SOURCES_FILE'])
# columns: source_id, issn, category
```

### Step 2 — Download works via openalex CLI
Download one source at a time. Remove `--fresh` to resume interrupted runs.

```bash
#!/bin/bash
# extract_sources.sh

while read -r source_id; do
  echo "Processing: $source_id"
  openalex download \
    --filter="publication_year:>1999,primary_location.source.id:$source_id" \
    --nested \
    --fresh \
    --quiet \
    --output="$JSON_MASTER"
  sleep 1
done < sources.txt
```

### Step 2b — Load JSON, filter, persist works parquet
```python
import duckdb

duckdb.sql(f"""
    COPY (
        SELECT * FROM read_json('{config["JSON_MASTER"]}/*.json', auto_detect=true)
        WHERE publication_year BETWEEN 2000 AND 2025
        AND is_paratext = false
        AND is_retracted = false
        AND type IN ('article', 'review')
    ) TO '{config["PARQUET_PATH"]}/works'
    (FORMAT PARQUET, PARTITION_BY (publication_year))
""")
```

### Step 3 — Flatten authorships, persist
Note: verify nested field paths against a sample JSON record before full run.

```python
duckdb.sql(f"""
    COPY (
        SELECT
            id AS work_id,
            primary_location.source.id AS source_id,
            publication_year AS pubyear,
            unnest(authorships).author.id AS author_id,
            unnest(authorships).institutions[1].id AS institution_id
        FROM read_parquet('{config["PARQUET_PATH"]}/works/*/*.parquet')
        WHERE author_id IS NOT NULL
        AND institution_id IS NOT NULL
    ) TO '{config["PARQUET_PATH"]}/authorships'
    (FORMAT PARQUET, PARTITION_BY (pubyear))
""")
```

### Step 4 — Flatten references, persist
```python
duckdb.sql(f"""
    COPY (
        SELECT
            id AS work_id,
            unnest(referenced_works) AS cited_work_id
        FROM read_parquet('{config["PARQUET_PATH"]}/works/*/*.parquet')
    ) TO '{config["PARQUET_PATH"]}/references'
    (FORMAT PARQUET)
""")
```

### Step 5 — Construct CITATION_MASTER
Join citer authorships to cited authorships via reference links.
Carry citer_year and cited_year so any time window can be sliced later.
Compute both institution weights. Flag journal-pair categories.

```sql
-- Run via duckdb.sql(). Schematic — adapt join keys to actual column names.
SELECT
    flag,
    ca.source_id          AS citer_source_id,
    ra.source_id          AS cited_source_id,
    ca.institution_id     AS citer_institution_id,
    ra.institution_id     AS cited_institution_id,
    ca.pubyear            AS citer_year,
    ra.pubyear            AS cited_year,
    -- weight_inst: 1 / n_institutions on the citing work
    1.0 / COUNT(ca.institution_id) OVER (PARTITION BY ca.work_id)
                          AS weight_inst,
    -- weight_frac: 1/n_authors then 1/n_institutions per author
    (1.0 / COUNT(DISTINCT ca.author_id) OVER (PARTITION BY ca.work_id))
    * (1.0 / COUNT(ca.institution_id) OVER (PARTITION BY ca.work_id, ca.author_id))
                          AS weight_frac
FROM references r
JOIN authorships ca ON r.work_id      = ca.work_id
JOIN authorships ra ON r.cited_work_id = ra.work_id
```

```python
duckdb.sql(f"""
    COPY (/* above query */)
    TO '{config["CITATION_MASTER"]}'
    (FORMAT PARQUET)
""")
```

---

## Matrix Construction

### Load and filter by time window
```python
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import yaml

with open('data.yaml') as f:
    config = yaml.safe_load(f)

df = pd.read_parquet(config['CITATION_MASTER'])

# Apply time window
df = df[
    df['citer_year'].between(config['CENSUS_START'], config['CENSUS_END']) &
    df['cited_year'].between(config['TARGET_START'], config['TARGET_END'])
]

# Apply institution thresholds (weights fixed before pruning — no readjustment)
inst_count = df.groupby('citer_institution_id')['weight_inst'].count()
inst_cites = df.groupby('citer_institution_id')['weight_inst'].sum()
keep_inst = inst_count[inst_count >= config['THRESHOLD_INSTITUTION_COUNT']].index
keep_inst = keep_inst.intersection(
    inst_cites[inst_cites >= config['THRESHOLD_INSTITUTION_CITES']].index
)
df = df[
    df['citer_institution_id'].isin(keep_inst) &
    df['cited_institution_id'].isin(keep_inst)
]
```

### Build node indexes
```python
sources = sorted(set(df['citer_source_id']) | set(df['cited_source_id']))
insts   = sorted(set(df['citer_institution_id']) | set(df['cited_institution_id']))
s_idx = {s: i for i, s in enumerate(sources)}
i_idx = {v: i for i, v in enumerate(insts)}
ns, ni = len(sources), len(insts)
```

### Compute unit_size table
unit_size is the number of works per source or institution in the citer set
for this specific flag and time window. Used for row-stochastic normalisation.

```python
unit_size_s = df.groupby('citer_source_id')['weight_inst'].sum()
unit_size_i = df.groupby('citer_institution_id')['weight_inst'].sum()
```

### Build CSR blocks from edge list
```python
def build_csr_blocks(df, weight_col):
    # SS block: source -> source
    ss_rows = df['citer_source_id'].map(s_idx).values
    ss_cols = df['cited_source_id'].map(s_idx).values
    ss_data = df[weight_col].values
    SS = csr_matrix((ss_data, (ss_rows, ss_cols)), shape=(ns, ns))

    # II block: institution -> institution
    ii_rows = df['citer_institution_id'].map(i_idx).values
    ii_cols = df['cited_institution_id'].map(i_idx).values
    ii_data = df[weight_col].values
    II = csr_matrix((ii_data, (ii_rows, ii_cols)), shape=(ni, ni))

    # SI block: source -> institution (source cites institution)
    si_rows = df['citer_source_id'].map(s_idx).values
    si_cols = df['cited_institution_id'].map(i_idx).values
    si_data = df[weight_col].values
    SI = csr_matrix((si_data, (si_rows, si_cols)), shape=(ns, ni))

    # IS block: institution -> source
    is_rows = df['citer_institution_id'].map(i_idx).values
    is_cols = df['cited_source_id'].map(s_idx).values
    is_data = df[weight_col].values
    IS = csr_matrix((is_data, (is_rows, is_cols)), shape=(ni, ns))

    return SS, II, SI, IS
```

### Row-stochastic normalisation
Divide each row by its row sum to form a proper stochastic matrix.
This is the M = D^{-1} C normalisation where D is the diagonal degree matrix.

```python
from scipy.sparse import diags

def row_stochastic(M):
    row_sums = np.array(M.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # avoid divide by zero
    D_inv = diags(1.0 / row_sums)
    return D_inv @ M
```

---

## Katz Iteration — Two Modes

### Mode 1: Bipartite SI/IS
Off-diagonal blocks only. Prestige propagates institutions -> sources -> institutions.

```python
def katz_bipartite(SI, IS, alpha, tol, max_iter):
    # normalise
    SI_n = row_stochastic(SI)
    IS_n = row_stochastic(IS)

    s_j = np.ones(SI_n.shape[0]) / SI_n.shape[0]
    s_i = np.ones(IS_n.shape[0]) / IS_n.shape[0]

    for _ in range(max_iter):
        s_j_new = alpha * (SI_n @ s_i) + (1 - alpha)
        s_i_new = alpha * (IS_n @ s_j_new) + (1 - alpha)
        s_j_new /= s_j_new.sum()
        s_i_new /= s_i_new.sum()
        if np.linalg.norm(s_j_new - s_j) < tol:
            break
        s_j, s_i = s_j_new, s_i_new

    return s_j, s_i
```

### Mode 2: Full matrix power iteration
Handles ss, ii, and mixed [ss, si, is, ii] cases via weight parameter w.
w=1 -> source only (ss), w=0 -> institution only (ii), 0<w<1 -> mixed.

Block scaling:
- SS block: w
- II block: (1-w)
- SI block: sqrt(w * (1-w))
- IS block: sqrt(w * (1-w))

```python
from scipy.sparse import bmat

def katz_full(SS, II, SI, IS, alpha, w, tol, max_iter):
    # scale blocks by w
    sqrt_w = np.sqrt(w * (1 - w))
    SS_w = w * SS
    II_w = (1 - w) * II
    SI_w = sqrt_w * SI
    IS_w = sqrt_w * IS

    # assemble full matrix and normalise
    M = bmat([[SS_w, SI_w],
              [IS_w, II_w]], format='csr')
    M_n = row_stochastic(M)

    n = M_n.shape[0]
    x = np.ones(n) / n

    for _ in range(max_iter):
        x_new = alpha * (M_n @ x) + (1 - alpha)
        x_new /= x_new.sum()
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    # split result back into source and institution scores
    s_j = x[:ns]
    s_i = x[ns:]
    return s_j, s_i
```

---

## Sensitivity Analysis — Community Effects

```python
sources_df = pd.read_parquet(config['SOURCES_FILE'])
categories = sources_df['category'].unique()

for cat in categories:
    cat_sources = sources_df[sources_df['category'] == cat]['source_id']
    df_sub = df[df['cited_source_id'].isin(cat_sources)]
    SS_sub, II_sub, SI_sub, IS_sub = build_csr_blocks(df_sub, 'weight_inst')
    s_j_sub, s_i_sub = katz_bipartite(SI_sub, IS_sub,
                                       config['ALPHA'],
                                       config['KATZ_TOL'],
                                       config['KATZ_MAX_ITER'])
    # compare institution rank order with full-matrix result via spearman

from scipy.stats import spearmanr
```

---

## Time Series Loop

```python
results = []
for window in config['TIME_WINDOWS']:
    cs, ce, ts, te = window
    # reload and filter df for this window
    # rebuild matrices
    # run katz
    # store ranks with window label
    results.append({'window': window, 's_ranks': ..., 'i_ranks': ...})
```

---

## Outputs
- `citation_master.parquet` — full unpartitioned citation edge list
- Parquet per run: source ranks, institution ranks, scores keyed by
  (flag, window, weight_col, alpha, w, mode)
- Rank correlation table across parameter combinations
- Sensitivity table: rank shifts under subdivision restriction

---

## Notes
- Use CSR throughout — scalable to larger disciplines beyond 600x600
- DuckDB reads JSON and parquet directly — no intermediate csv
- Keep code flat — two Katz functions, shared normalisation util
- Verify nested JSON field paths against sample record before full pipeline run
- Remove --fresh from bash script to make download resumable
- unit_size recomputed per flag and time window — it is downstream of all filtering
- w interpolates source/institution balance in full matrix mode;
  w=1 and w=0 reduce to pure ss and ii unipartite cases respectively
