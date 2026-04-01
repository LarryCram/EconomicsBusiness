# BOOTSTRAP.md — Bootstrap Uncertainty Analysis for Baseline Ranking

## Goal

Quantify sampling uncertainty in the baseline spectral ranking (m=0110, t5, F=A,
τ_U=20, τ_S=20, ρ=0, α=1.0) by repeatedly re-fitting v_s and v_u on 80%-with-
replacement sub-samples of the citation edge list.  Target: B=1000 replicates;
run B=5–20 during development and timing tests.

Output:
- `$WORKING/bootstrap/` — bootstrap v_s, v_u arrays (machine-specific SSD, not git-tracked)
- `spectral_ranking_bootstrap/fig_5.py` → `plots/fig_5.pdf` / `plots/fig_5_latex.pdf`

---

## 1. Statistical Design

### Resampling unit

The raw edge list table `el_t5_A_tauU20_tauS20` has one row per
(citer_work × citer_inst × cited_work × cited_inst) attribution.  The block
builders deduplicate this before aggregating:

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

Connect to `edge_lists.duckdb` as read-only (or read-write to create the tmp
table once).  Run:

```python
# ρ=0 fixed-count weight (pre-compute r_bar once from full edge list)
r_bar = db.execute(
    "SELECT AVG(rval) FROM "
    "(SELECT DISTINCT citer_work_idx, CAST(R_i AS DOUBLE) AS rval FROM el_t5_A_tauU20_tauS20)"
).fetchone()[0]
```

**SI deduplicated edges (for C_SI bootstrap):**
```sql
SELECT citer_source_idx, cited_inst_idx,
       ({r_bar} / CAST(R_i AS DOUBLE)) * cited_inst_weight AS w
FROM (
    SELECT DISTINCT citer_work_idx, citer_source_idx,
                    cited_work_idx,  cited_inst_idx,
                    R_i, cited_inst_weight
    FROM el_t5_A_tauU20_tauS20
)
```
→ pandas DataFrame, then `.to_numpy()`.  No GROUP BY yet — we keep individual
rows so resampling can pick duplicates.

**IS deduplicated edges (for C_IS bootstrap):**
```sql
SELECT citer_inst_idx, cited_source_idx,
       ({r_bar} / CAST(R_i AS DOUBLE)) * inst_weight AS w
FROM (
    SELECT DISTINCT citer_work_idx, citer_inst_idx,
                    cited_work_idx,  cited_source_idx,
                    R_i, inst_weight
    FROM el_t5_A_tauU20_tauS20
)
```

**Unit index (sources and institutions, for dense-index maps and a_p):**
Reuse the same query as `build_csr()` — load once, extract `source_ids`,
`inst_ids`, `a_s`, `a_u`, `n_s`, `n_u`.  Build `pd.Index(source_ids)` and
`pd.Index(inst_ids)` for `get_indexer` lookups.

### Convert to numpy arrays

```python
# SI block
si_src_dense = src_pd_index.get_indexer(df_si['citer_source_idx'].to_numpy())  # int32
si_inst_dense = inst_pd_index.get_indexer(df_si['cited_inst_idx'].to_numpy())  # int32
si_w = df_si['w'].to_numpy(dtype=np.float64)
N_SI = len(si_w)

# IS block
is_inst_dense = inst_pd_index.get_indexer(df_is['citer_inst_idx'].to_numpy())  # int32
is_src_dense  = src_pd_index.get_indexer(df_is['cited_source_idx'].to_numpy()) # int32
is_w = df_is['w'].to_numpy(dtype=np.float64)
N_IS = len(is_w)
```

Memory estimate: if N_SI ≈ N_IS ≈ 2M rows, storage is ~2 × 3 arrays × 2M × 8 bytes ≈ 100 MB.  Profile this at startup; if N > 10M consider chunking the IS block (IS is expected to be larger since citer-institution attribution is many-to-one).

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
This is the "sample the COO" approach — we never explicitly group by (row, col);
the format conversion handles it.

### Ranking step

Reuse `katz_ranker.bipartite()` and `katz_ranker._row_normalise()` directly:

```python
from spectral_ranking.katz_ranker import bipartite, _row_normalise

H_SI, _ = _row_normalise(C_SI)
H_IS, _ = _row_normalise(C_IS)
pi_s, pi_u, iters, norm = bipartite(H_SI, H_IS, alpha=1.0)

# Joint-normalise and compute v (same as rank() for m=0110)
pi_s /= 2.0
pi_u /= 2.0
A = a_s.sum() + a_u.sum()
v_s_b = A * pi_s / a_s
v_u_b = A * pi_u / a_u
```

**Important**: use the **baseline** `a_s` and `a_u` throughout (loaded once from
`_units_t5_A_tauU20_tauS20`).  Do NOT recompute work counts per bootstrap — we
are bootstrapping reference events, not publications.

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
    meta.json           # provenance: n, seed, tol, n_s, n_u, source_ids, inst_ids
```

Float32 is sufficient (4 decimal places of precision).  At B=1000 and
N = n_s + n_u ≈ 2500 units, this is ~30 MB total.

**Checkpoint interval**: write every 50 replicates.  Allows resuming interrupted
runs via `--resume` flag.

---

## 5. Runner Script: `spectral_ranking/bootstrap_baseline.py`

### CLI interface

```
python spectral_ranking/bootstrap_baseline.py [--n 1000] [--seed 42] [--tol 1e-7] [--resume]
```

- `--n`: number of bootstrap replicates (default 1000; use 5–20 for testing)
- `--seed`: base random seed (replicate b uses seed + b)
- `--tol`: power iteration tolerance (default 1e-7)
- `--resume`: skip already-computed replicates (read checkpoint from output file)

### Structure

```python
# 1. Load config (paths), connect to edge_lists.duckdb (read-only)
# 2. Load unit index → n_s, n_u, a_s, a_u, source_ids, inst_ids, dense index maps
# 3. Load baseline ranking from rankings.duckdb for diagnostics
#    (also used to verify first replicate is close to baseline)
# 4. Pre-load deduplicated SI and IS edge arrays (SQL → numpy)
# 5. Bootstrap loop: sample → CSR → bipartite → store v_s, v_u
# 6. Write checkpoint every 50 replicates
# 7. Final save to data/bootstrap_results/v_s_boot.npy, v_u_boot.npy
#    and data/bootstrap_results/meta.json (n, seed, tol, n_s, n_u,
#    source_ids, inst_ids, baseline_table)
```

### Output files

```
$WORKING/bootstrap/          # resolved from config.yaml at runtime
    v_s_boot.npy             # shape (B, n_s), float32
    v_u_boot.npy             # shape (B, n_u), float32
    meta.json                # provenance: n, seed, tol, n_s, n_u, source_ids, inst_ids
```

The `source_ids` and `inst_ids` in meta.json map dense indices 0…n_s-1 to
original unit_idx values for joining with the baseline ranking table in fig_5.

---

## 6. Figure 5: Bootstrap Scatter vs. Baseline

**File**: `spectral_results_analysis/fig_5.py`
**Output**: `plots/fig_5.pdf`, `plots/fig_5_latex.pdf`

### Design

Two-facet figure mirroring Fig 2's layout (top = sources, bottom = institutions):

- **X-axis**: baseline v (from `rk_t5_A_tauU20_tauS20_rho0_m0110_chi50_alpha100`),
  log scale, ordered by baseline rank (highest v at left, same orientation as Fig 2)
- **Y-axis**: bootstrap v values, log scale
- **Scatter**: for each unit, plot all B bootstrap v values as small points
  (low alpha, e.g. 0.05–0.1) — B points per unit, clustered around baseline x
- **Colour**: E (red), B (blue), other (grey) — matching Fig 2 palette
- **Baseline reference**: diagonal y=x line (black, thin) — points on this line
  match the baseline exactly
- **Percentile bands** (optional, decide after seeing the scatter):
  5th–95th percentile band per unit as a thin shaded ribbon

### X-axis layout

Unlike Fig 2 (rank on x-axis), Fig 5 uses **v_baseline on x-axis** so each
unit's bootstrap scatter appears as a vertical cloud above its baseline value.
This is more informative than rank-order x since it shows whether high-v units
are more or less stable than low-v units.

If the scatter is too dense at low v, consider:
- Using rank on x-axis (same as Fig 2) with v on y — scatter appears as vertical
  smearing around the baseline curve
- Showing only quantiles (p5, p50, p95) rather than raw scatter

### Data flow

```python
# 1. Load bootstrap arrays (WORKING path from config.yaml)
v_s_boot = np.load(WORKING / 'bootstrap/v_s_boot.npy')   # (B, n_s)
v_u_boot = np.load(WORKING / 'bootstrap/v_u_boot.npy')   # (B, n_u)
meta = json.load(...)

# 2. Load baseline v from rankings.duckdb
#    Join on source_ids / inst_ids to align dense indices

# 3. Load field labels from data/source_master.csv (source_idx → E/B/'')

# 4. For sources panel:
#    x = np.tile(v_s_baseline, (B, 1))   # (B, n_s)
#    y = v_s_boot                          # (B, n_s)
#    Plot as scatter (flatten both to 1D, colour by field)

# 5. Same for institutions panel (no field labels → all grey or by type)
```

### Style notes (matching project conventions)

- Seaborn OO interface; seaborn-whitegrid style
- Figure size: ~8 × 10 inches (two stacked panels with equal height)
- Marker: `'.'`, size=1, alpha=0.05 (raw scatter) or size=2, alpha=0.3 (fewer replicates)
- Axis labels: "Baseline $v$ (m=0110)" (x), "Bootstrap $v$" (y)
- Panel labels: "(a) Sources" / "(b) Institutions"
- Two versions: `fig_5.pdf` (with suptitle "Bootstrap distribution of v, B=1000"),
  `fig_5_latex.pdf` (no title)

---

## 7. Speed Optimisation Notes

In rough order of impact:

1. **Load once, loop over numpy**: eliminates all DuckDB/pandas overhead inside
   the loop. Expected speedup vs. per-bootstrap `build_csr()`: 10–50×.

2. **scipy COO duplicate summation**: `coo_matrix(...).tocsr()` handles
   duplicate (row, col) aggregation internally in C; faster than pandas groupby.

3. **Tighten tol to 1e-7**: saves ~5 iterations per replicate.

4. **alpha=1 regime**: the baseline uses α=1 (pure Perron), which means
   `power_iteration` skips the prior-injection arithmetic.  No change needed.

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

1. **`bootstrap_baseline.py`** — write and test with `--n 5`
   - Verify: first replicate converges; v_s output shape = (n_s,)
   - Verify: mean of B=100 bootstrap v_s is close to baseline v_s (within ~5%)
   - Time 20 replicates; extrapolate to 1000

2. **`fig_5.py`** — write against B=20 test output first
   - Confirm scatter is readable; adjust alpha/marker size
   - Add field colour coding for sources

3. **Scale to B=1000** — run overnight if needed; use `--resume` if interrupted

4. **Register in PLOTS.md** under Figure 5 (replacing the skipped sensitivity
   heatmap placeholder)

---

## 9. Open Questions to Resolve During Implementation

- **N_SI / N_IS sizes**: query `SELECT COUNT(*) FROM (SELECT DISTINCT ...)` before
  loading arrays fully.  If > 10M rows, evaluate chunked loading or Poisson
  approximation bootstrap (see note below).

- **Poisson approximation** (fallback if N_SI is too large for in-memory COO):
  For each non-zero CSR entry C_SI[i,j] = w, draw a bootstrap weight
  w* ~ Poisson(0.8 × w) (or Binomial(round(w), 0.8) if w is integer-interpretable).
  This approximates bootstrap without storing individual rows but loses within-
  paper correlation.  Use only if memory is a genuine constraint.

- **Dangling rows**: if a bootstrap sample drops all references from a source
  (rare but possible with 80% sampling), that source becomes dangling in H_SI.
  The existing `_row_normalise()` handles dangling rows gracefully (leaves them
  as zero rows), so this does not require special treatment.

- **v outliers**: very small-a_p units will have high-variance v_bootstrap.
  Fig 5 should either truncate the y-axis or use log scale on y (consistent
  with Fig 2's log y).
