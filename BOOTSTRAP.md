# BOOTSTRAP.md — Bootstrap Uncertainty Analysis for Baseline Ranking

## Goal - metadata error

Openalex has metadata errors at work level and at reference list level.

We have a large corpus 2000-2024 and are using 2020-2024 as the baseline.

#### Error pub_year

pub_year is out randomly in xpr% of works by +/-1 year. Impacts year-filtered edge list on citer side and cited_side. It will be easier to avoid out-of census/target year issues by confining the errors to -1 at the latest year and +1 at the ealiest year.

#### Error wrong source

Source is out randomly on xws% of works. Effects both sides of edge list.

#### Error wrong institution

Institution is out randomly on xwi% of works. Effects both sides of edge list.
Many errors are within country, and fewer cross country. Within country share ywc. 
Assume these are institution errors and do not tie them to author errors even in fractionation case.

#### Error wrong reference

A referenced work is randomly assigned to the wrong work in xwr% edge rows. This work itself can be erroneous for the above reasons. Draw from the census/target set.

#### Metadata errors could draw works from outside, but the number of outside works that would survive the filtering is small so all metadata errors occur within the large corpus. 

#### Magnitude of error

Use 10% for all x-type erors. Take ywc=0.75.

## Goal - sampling error

Quantify sampling uncertainty in the baseline spectral ranking
(m=0110, run_code=20242024, F=EBAX, τ_U=10, τ_S=5, ρ=0, α=1.0) by
repeatedly re-fitting v_s and v_u on 80%-sub-samples of the citation
edge list.  Target: B=1000 replicates; run B=5–20 during development and timing tests.

NOTE: F=A, τ_U=20, τ_S=20 in the original spec is stale — the current baseline
is F=EBAX, τ_U=10, τ_S=5 (from params.csv label='baseline'). The code reads from
load_runs() so it picks up the correct values automatically.

Two modes (--mode flag): 'work' (default, 80% citer works without replacement) or
'edge' (80% edges with replacement). The paper's OA error simulation (bootstrap_oa_errors.py)
is a separate script and is what appears in fig_7a.

Code lives in: `spectral_ranking_bootstrap/` (alongside `spectral_ranking/`).

Output:
- `$WORKING/bootstrap/` — bootstrap v_s, v_u arrays (machine-specific SSD, not git-tracked)

The baseline corpus is identified at runtime via `load_runs()` (label='baseline'):

```python
from util import load_runs
_baseline = next(r for r in load_runs() if r['label'] == 'baseline')
RUN_CODE  = _baseline['run_code']   # '20242024'
TAU_U     = _baseline['tau_u']      # 10
TAU_S     = _baseline['tau_s']      # 5
FX        = _baseline['fx']         # 'EBAX'
EL_TABLE    = f'el_{RUN_CODE}_{FX}_tauU{TAU_U}_tauS{TAU_S}_vartau'
UNITS_TABLE = f'_units_{RUN_CODE}_{FX}_tauU{TAU_U}_tauS{TAU_S}_vartau_m0110'
```

---

## 1. Statistical Design

### Resampling unit

The edge list table `el_20242024_EBAX_tauU10_tauS5_vartau` has one row per
(citer_work × citer_inst × cited_work × cited_inst) attribution.
`build_csr.py` deduplicates this before aggregating into block matrices:

- **C_SI** deduplicates on (citer_work, citer_source, cited_work, cited_inst),
  removing the citer_inst ambiguity.  Each remaining row is one *reference
  event* attributed to a (source, cited-institution) pair.
- **C_IS** deduplicates on (citer_work, citer_inst, cited_work, cited_source).

The natural bootstrap unit is one post-deduplication reference event (one row
of the inner SELECT DISTINCT used in `_build_si` / `_build_is`).  Resampling
at this level preserves the statistical meaning: uncertainty arises from which
specific references appear in the five-year window.

### Sample size

80% with replacement (m-out-of-n bootstrap, m = 0.8 N).  This is slightly
conservative vs. full-N bootstrap but avoids inflating the effective sample
size when N is large, which can matter for a corpus that is nearly exhaustive.

### Number of replicates

B = 1000 for publication.  B = 5–20 during testing.  Controlled by a `--n`
flag on the runner script.

---

## 2. Pre-computation: Load Deduplicated Edge Arrays Once

### Why bypass DuckDB inside the loop

Each call to `build_csr()` opens a DuckDB scan, materialises `_tmp_el`, runs
two SQL queries with DISTINCT + GROUP BY, and builds COO → CSR.  That overhead
(schema resolution, query planning, format conversion) is repeated B times if
we call `build_csr` naively per bootstrap.  Instead:

1. Execute the inner DISTINCT queries **once** at startup.
2. Convert results to numpy arrays.
3. The loop does only: numpy random choice → scipy COO → tocsr() → bipartite().

### SQL to pre-load

Connect to `edge_lists.duckdb` read-only.  Replicate the `_tmp_el`
construction from `build_csr.py` exactly (ρ=0 fixed-count weighting):

```python
r_bar = db.execute(
    f"SELECT AVG(rval) FROM "
    f"(SELECT DISTINCT citer_work_idx, CAST(R_i AS DOUBLE) AS rval FROM {EL_TABLE})"
).fetchone()[0]

db.execute(f"""
    CREATE OR REPLACE TEMP TABLE _tmp_el AS
    SELECT citer_work_idx, citer_source_idx, citer_inst_idx,
           cited_work_idx,  cited_source_idx, cited_inst_idx,
           inst_weight, cited_inst_weight,
           {r_bar} / CAST(R_i AS DOUBLE) AS rho_w
    FROM {EL_TABLE}
""")
```

**SI deduplicated edges** (matching `_build_si` in `build_csr.py`):
```sql
SELECT citer_source_idx, cited_inst_idx,
       rho_w * cited_inst_weight AS w
FROM (
    SELECT DISTINCT citer_work_idx, citer_source_idx,
                    cited_work_idx,  cited_inst_idx,
                    rho_w, cited_inst_weight
    FROM _tmp_el
)
```
→ pandas DataFrame, then `.to_numpy()`.  No GROUP BY — keep individual rows
so resampling can pick duplicates.

**IS deduplicated edges** (matching `_build_is` in `build_csr.py`):
```sql
SELECT citer_inst_idx, cited_source_idx,
       rho_w * inst_weight AS w
FROM (
    SELECT DISTINCT citer_work_idx, citer_inst_idx,
                    cited_work_idx,  cited_source_idx,
                    rho_w, inst_weight
    FROM _tmp_el
)
```

**Unit index** — reuse the same query as `build_csr.py`:
```python
units_df = db.execute(
    f"SELECT unit_idx, unit_type, a_p FROM {UNITS_TABLE} ORDER BY unit_type, unit_idx"
).fetchdf()
```
Extract `source_ids`, `inst_ids`, `a_s`, `a_u`, `n_s`, `n_u`.
Build `pd.Index(source_ids)` and `pd.Index(inst_ids)` for `get_indexer` lookups.

Drop `_tmp_el` after loading.

### Convert to numpy arrays

```python
# SI block
si_src_dense  = src_pd_index.get_indexer(df_si['citer_source_idx'].to_numpy())  # int32
si_inst_dense = inst_pd_index.get_indexer(df_si['cited_inst_idx'].to_numpy())   # int32
si_w = df_si['w'].to_numpy(dtype=np.float64)
N_SI = len(si_w)

# IS block
is_inst_dense = inst_pd_index.get_indexer(df_is['citer_inst_idx'].to_numpy())   # int32
is_src_dense  = src_pd_index.get_indexer(df_is['cited_source_idx'].to_numpy())  # int32
is_w = df_is['w'].to_numpy(dtype=np.float64)
N_IS = len(is_w)
```

Memory estimate: if N_SI ≈ N_IS ≈ 2M rows, storage is ~2 × 3 arrays × 2M × 8 bytes ≈ 100 MB.
Profile this at startup; if N > 10M consider the Poisson approximation (section 9).

---

## 3. The Bootstrap Loop

### Core sampling step

For each replicate b in range(B):

```python
rng = np.random.default_rng(seed + b)   # reproducible, one RNG per replicate

# Sample SI block
boot_si = rng.choice(N_SI, size=int(0.8 * N_SI), replace=True)
C_SI = sp.coo_matrix(
    (si_w[boot_si], (si_src_dense[boot_si], si_inst_dense[boot_si])),
    shape=(n_s, n_u)
).tocsr()

# Sample IS block
boot_is = rng.choice(N_IS, size=int(0.8 * N_IS), replace=True)
C_IS = sp.coo_matrix(
    (is_w[boot_is], (is_inst_dense[boot_is], is_src_dense[boot_is])),
    shape=(n_u, n_s)
).tocsr()
```

`coo_matrix.tocsr()` automatically sums duplicate (row, col) entries, so rows
that appear multiple times in `boot_si` correctly accumulate their weight.

### Ranking step

Reuse `katz_ranker.bipartite()` and `katz_ranker._row_normalise()` directly:

```python
from spectral_ranking.katz_ranker import bipartite, _row_normalise

H_SI, _ = _row_normalise(C_SI)
H_IS, _ = _row_normalise(C_IS)
pi_s, pi_u, iters, norm = bipartite(H_SI, H_IS, alpha=1.0)

# Joint-normalise and compute v (same as run_one() for m=0110)
pi_s /= 2.0
pi_u /= 2.0
A = a_s.sum() + a_u.sum()
v_s_b = A * pi_s / a_s
v_u_b = A * pi_u / a_u
```

**Important**: use the **baseline** `a_s` and `a_u` throughout (loaded once from
`_units_20242024_A_tauU20_tauS20`).  Do NOT recompute work counts per bootstrap —
we are bootstrapping reference events, not publications.

### Timing and convergence

The bipartite power iteration typically converges in 20–50 iterations at α=1.
Tighten `tol` to `1e-7` (vs baseline `1e-8`) to save 5–10 iterations per
replicate with negligible accuracy loss for bootstrap purposes.

Profile first 5 replicates, then extrapolate.  Target: < 2 s per replicate
(1000 bootstraps in under 35 minutes on a single core; easily parallelisable).

---

## 4. Storage

Results go in the machine-specific WORKING folder (read from `config.yaml` via
`util/load_config.py`; never hardcoded):

```
$WORKING/bootstrap/
    v_s_boot.npy        # shape (B, n_s), float32
    v_u_boot.npy        # shape (B, n_u), float32
    meta.json           # provenance: n, seed, tol, n_s, n_u, source_ids, inst_ids,
                        #             run_code, baseline_table
```

Float32 is sufficient (4 decimal places of precision).  At B=1000 and
N = n_s + n_u ≈ 2500 units, this is ~30 MB total.

**Checkpoint interval**: write every 50 replicates.  Allows resuming interrupted
runs via `--resume` flag.

---

## 5. Runner Script: `spectral_ranking_bootstrap/bootstrap_baseline.py`

### CLI interface

```
python spectral_ranking_bootstrap/bootstrap_baseline.py [--n 1000] [--seed 42] [--tol 1e-7] [--resume]
```

- `--n`: number of bootstrap replicates (default 1000; use 5–20 for testing)
- `--seed`: base random seed (replicate b uses seed + b)
- `--tol`: power iteration tolerance (default 1e-7)
- `--resume`: skip already-computed replicates (read checkpoint from output file)

### Structure

```
spectral_ranking_bootstrap/
    __init__.py
    bootstrap_baseline.py   # main runner (CLI entry point)
```

```python
# bootstrap_baseline.py
# 1. Load config (paths); identify baseline via load_runs() label='baseline'
# 2. Connect to edge_lists.duckdb (read-only)
# 3. Load unit index → n_s, n_u, a_s, a_u, source_ids, inst_ids, dense index maps
# 4. Load baseline ranking from rankings.duckdb for diagnostics
#    (verify first replicate is close to baseline)
# 5. Build _tmp_el, pre-load deduplicated SI and IS edge arrays (SQL → numpy),
#    drop _tmp_el
# 6. Bootstrap loop: sample → COO → CSR → bipartite → store v_s, v_u
# 7. Write checkpoint every 50 replicates
# 8. Final save: v_s_boot.npy, v_u_boot.npy, meta.json
```

---

## 6. Figure 7: Bootstrap Scatter vs. Baseline

**File**: `spectral_results_analysis/fig_7.py`
**Output**: `plots/fig_7.pdf`, `plots/fig_7_latex.pdf`

### Design

Two-panel figure mirroring fig_2/fig_3 layout (top = sources, bottom = institutions):

- **X-axis**: baseline rank (rank 1 = highest v, same orientation as fig_2)
- **Y-axis**: prestige per work v, log scale
- **Baseline**: black line (same as fig_2 baseline curve)
- **Bootstrap scatter**: for each unit, plot all B bootstrap v values as small
  points (low alpha, e.g. 0.05–0.1) — B points per unit, clustered vertically
  around baseline
- **Colour**: field_eb 'E' (red), 'B' (blue), 'A' (orange), NULL (grey) for
  sources; grey for institutions
- **Reference band** (optional): 5th–95th percentile ribbon per unit

### Data flow

```python
# 1. Load bootstrap arrays (WORKING path from config.yaml)
v_s_boot = np.load(paths.working / 'bootstrap/v_s_boot.npy')   # (B, n_s)
v_u_boot = np.load(paths.working / 'bootstrap/v_u_boot.npy')   # (B, n_u)
meta = json.load(open(paths.working / 'bootstrap/meta.json'))

# 2. Load baseline v from rankings.duckdb
#    Join on source_ids / inst_ids (from meta.json) to align dense indices

# 3. Load field labels from data/source_master.csv
#    usecols=['source_idx', 'field_eb']

# 4. For each panel: repeat baseline_rank B times (np.tile) and flatten
#    alongside the bootstrap v values; scatter plot coloured by field_eb
```

### Style notes

- `sns.set_theme(style='whitegrid', font_scale=0.95)`, `figsize=(9, 8)`, `hspace=0.44`
- Marker `'.'`, size=1, alpha=0.05 for B=1000; adjust for smaller B
- Two versions: `fig_7.pdf` (with suptitle), `fig_7_latex.pdf` (no title)

---

## 7. Speed Optimisation Notes

In rough order of impact:

1. **Load once, loop over numpy**: eliminates all DuckDB/pandas overhead inside
   the loop. Expected speedup vs. per-bootstrap `build_csr()`: 10–50×.

2. **scipy COO duplicate summation**: `coo_matrix(...).tocsr()` handles
   duplicate (row, col) aggregation internally in C; faster than pandas groupby.

3. **Tighten tol to 1e-7**: saves ~5 iterations per replicate.

4. **alpha=1 regime**: the baseline uses α=1 (pure Perron), which means
   power iteration skips prior-injection arithmetic.

5. **Parallelisation** (if needed): the bootstrap loop is embarrassingly parallel.
   Use `concurrent.futures.ProcessPoolExecutor` with `chunksize=50`.  Each worker
   takes the pre-loaded numpy arrays (passed via shared memory or re-loaded from
   file) and a seed range.

6. **Avoid M_S materialisation overhead**: `bipartite()` computes `M_S = H_SI @ H_IS`
   inside the loop.  For the bootstrap, `H_SI` and `H_IS` change every iteration
   so M_S cannot be cached.  This is unavoidable for the bipartite approach.

7. **Profile first**: before optimising further, run B=5 with timing per step
   (pre-load, sample, COO→CSR, row-norm, bipartite).  The bottleneck is likely
   `bipartite()` (power iteration) not the sampling or COO construction.

---

## 8. Implementation Sequence

1. **`spectral_ranking_bootstrap/bootstrap_baseline.py`** — write and test with `--n 5`
   - Verify: first replicate converges; v_s output shape = (n_s,)
   - Verify: mean of B=100 bootstrap v_s is close to baseline v_s (within ~5%)
   - Time 20 replicates; extrapolate to 1000

2. **`fig_7.py`** — write against B=20 test output first
   - Confirm scatter is readable; adjust alpha/marker size
   - Add field_eb colour coding for sources

3. **Scale to B=1000** — run overnight if needed; use `--resume` if interrupted

4. **Register in PLOTS.md** under Figure 7

---

## 9. Open Questions to Resolve During Implementation

- **N_SI / N_IS sizes**: query `SELECT COUNT(*) FROM (SELECT DISTINCT ...)` before
  loading arrays fully.  If > 10M rows, evaluate chunked loading or Poisson
  approximation bootstrap (see note below).

- **Poisson approximation** (fallback if N_SI is too large for in-memory COO):
  For each non-zero CSR entry C_SI[i,j] = w, draw a bootstrap weight
  w* ~ Poisson(0.8 × w).  This approximates bootstrap without storing individual
  rows but loses within-paper correlation.  Use only if memory is a genuine
  constraint.

- **Dangling rows**: if a bootstrap sample drops all references from a source
  (rare but possible with 80% sampling), that source becomes dangling in H_SI.
  The existing `_row_normalise()` handles dangling rows gracefully (leaves them
  as zero rows), so this does not require special treatment.

- **v outliers**: very small-a_p units will have high-variance v_bootstrap.
  fig_7 should use log scale on y (consistent with fig_2's log y).
