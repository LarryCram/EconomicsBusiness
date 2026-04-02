# WORKPLAN.md — Instructions for Claude Code
# Session 2026-03-28 — **COMPLETED 2026-04**

**STATUS: WORK COMPLETED**
This file specified all code changes and new scripts agreed in the session of
2026-03-28. All items listed below have been successfully implemented and the 
spectral ranking pipeline is now complete. Retained for historical reference
and technical documentation.

---

## Summary of decisions

1. **bipartite_resolvent** uses α_step = √α internally so the round-trip
   attenuation per reference equals α, making all four modes directly comparable.
   Caller always passes the same α (e.g. 0.85); the change is internal only.

2. **Second eigenpair (λ₂, φ₂) lives entirely in the analysis layer.**
   It is computed on-the-fly in `spectral_results_analysis/community.py` by
   calling `build_csr()` directly. It is not stored in `rankings.duckdb` and
   requires no changes to `katz_ranker.py`, `RankResult`, or the ranking table
   schema. Rebuilding CSR blocks from `edge_lists.duckdb` takes seconds; the
   eigenpair computation itself is milliseconds. The design benefit — keeping
   the pipeline schema stable while the analysis method evolves — outweighs
   the negligible rebuild cost.

3. **χ\* = N_u / (N_s + N_u)** is computed dynamically from the units table
   in `run_rankings.py` before the Stage 1 schedule is built, then added as
   an extra full-joint run. Cannot be pre-computed because N_u depends on τ_U and F.

4. **table_maker.py** joins rankings to source/institution display names from
   the parquet data files and produces LaTeX/CSV ranking tables with v.

5. τ_U = 10 is primary; τ_U = 8 is a Stage 1 sensitivity check only.

---

## File 1 — `spectral_ranking/katz_ranker.py`

### Change: rewrite `bipartite_resolvent`

The function signature is **unchanged** — callers pass `alpha` as before.
Internally use `alpha_step = sqrt(alpha)` so the per-reference attenuation
matches SS and II (both use α per reference; SI/IS previously used α² per
reference because two steps S→I→S are traversed per reference).

Mathematical change:
- Old LHS: `(I − α² M_S^T)`   Old RHS factor: `(1−α)`
- New LHS: `(I − α  M_S^T)`   New RHS factor: `(1−√α)`

The LHS now uses α = α_step², so the community amplification 1/(1−α λ₂) is
directly comparable across all modes.

Replace the function body (keep the docstring, update it to note the √α convention):

```python
def bipartite_resolvent(H_SI, H_IS, N_s, N_u, alpha):
    import math
    alpha_step = math.sqrt(alpha)      # per-step damping; alpha_step² = alpha

    N = N_s + N_u
    mu_s = np.full(N_s, 1.0 / N)
    mu_u = np.full(N_u, 1.0 / N)

    # One-mode projection M_S = H_SI @ H_IS  (N_s × N_s)
    M_S = H_SI.dot(H_IS)

    # LHS: (I − α M_S^T) in CSC for direct solver  [α = alpha_step²]
    A = eye(N_s, format='csc') - alpha * M_S.T.tocsc()

    # RHS: (1 − alpha_step)(μ_S + alpha_step H_IS^T μ_I)
    rhs = (1.0 - alpha_step) * (mu_s + alpha_step * H_IS.T.dot(mu_u))

    pi_s = spsolve(A, rhs)

    # Recover institutions: π_I = alpha_step H_SI^T π_S + (1 − alpha_step) μ_I
    pi_u = alpha_step * H_SI.T.dot(pi_s) + (1.0 - alpha_step) * mu_u

    pi_s = np.maximum(pi_s, 0.0)
    pi_u = np.maximum(pi_u, 0.0)
    total = pi_s.sum() + pi_u.sum()
    pi_s /= total
    pi_u /= total

    return pi_s, pi_u
```

No other changes to `katz_ranker.py`. `RankResult` is unchanged. No `phi2` field.

---

## File 2 — `spectral_ranking/run_rankings.py`

### Change: add χ* as a dynamic Stage 1 variant

No schema changes. The `_catalog` table, ranking table columns, and
`write_result` are all unchanged.

In `main()`, after opening `el_db` and before building the schedule, compute
χ* from the units table for the primary baseline corpus:

```python
units_row = el_db.execute(
    "SELECT unit_type, COUNT(*) AS n "
    "FROM _units_t5_A_tau20 GROUP BY unit_type"
).fetchdf()
n_s_base = int(units_row.loc[units_row.unit_type == 'S', 'n'].iloc[0])
n_u_base = int(units_row.loc[units_row.unit_type == 'U', 'n'].iloc[0])
chi_star  = n_u_base / (n_s_base + n_u_base)
print(f"  χ* = {chi_star:.4f}  (N_s={n_s_base}, N_u={n_u_base})")
```

Add to STAGE1 list (after the existing full-joint entry):

```python
RunParams(tx=5, fx='A', tau_u=20, tau_s=20, rho=0,
          m=(1, 1, 1, 1), chi=chi_star, alpha=0.85,
          label='full-joint-chi-star'),
```

χ* will appear in the table name as `chi{round(chi_star*100)}` via the existing
`table_name()` function — no changes needed there.

---

## File 3 — `spectral_results_analysis/community.py` (new file)

Purpose: compute second eigenpairs for SS, II, bipartite, and full joint runs;
produce Figure 3 (four-panel community analysis plot). Self-contained — reads
CSR blocks directly from `edge_lists.duckdb` and rankings from `rankings.duckdb`.
Makes no writes to either database.

### Eigenpair computation

Use `scipy.sparse.linalg.eigs(M.T.tocsc(), k=2, which='LM')` for all cases.
Take real parts throughout. Warn if `abs(imag(λ₂)) / abs(λ₂) > 0.01`.

**Sign convention**: the unit with the largest `a_p` in the run has `phi2 > 0`.
If that element is zero (degenerate), take the next largest `a_p` unit with
a non-zero element.

**Bipartite M_S**: form `M_S = H_SI.dot(H_IS)` locally (do not call
`bipartite_resolvent` — compute π_S separately if needed, or read it from
`rankings.duckdb`). The second eigenpair of `M_S.T` gives φ₂^S.
Institution community vector: `phi2_u = H_SI.T @ phi2_s`, then normalise
to unit Euclidean norm.

**Note on α**: in the amplification column of the summary table, use α (not
α_step = √α) in the denominator `1/(1−α·λ₂)` for all modes, because the √α
convention makes the round-trip attenuation equal to α in all cases.

### Script structure

```python
def load_field_labels(data_dir) -> dict:
    """
    Return {source_idx (int): 'E' | 'B' | 'A'} from reference file in data/.
    Warn if join to rankings is incomplete (< 80% matched).
    Note: source_idx = integer from OpenAlex URI after stripping
    'https://openalex.org/S'. Verify the label file uses the same integer form.
    """

def second_eigenpair(M, a_p) -> tuple:
    """
    Compute (lambda2_real, lambda2_imag_frac, phi2) from eigs(M.T, k=2, 'LM').
    Apply sign convention and imaginary warning.
    M: csr_matrix. a_p: ndarray of work counts for sign convention.
    """

def run_community_analysis(paths):
    """
    For each of: SS F=A, II F=A, baseline SI/IS, full-joint χ=0.5, full-joint χ*:
      1. Call build_csr() with appropriate m and corpus params.
      2. Row-normalise to H (or H_SI, H_IS for bipartite).
      3. Compute second eigenpair.
      4. Load v and a_p from rankings.duckdb for unit labels.
      5. Collect (label, lambda2, gap=1-lambda2, amplification=1/(1-alpha*lambda2))
         into a summary DataFrame.
    Print summary table.
    Call plot_figure3(results, field_labels, paths).
    """

def plot_figure3(results, field_labels, paths):
    """
    Four-panel figure — see PLOTS.md Figure 3 for full panel specification.
    Save to plots/fig3_community_eigenpair.pdf.
    """

def main():
    paths = load_config()
    run_community_analysis(paths)
```

### Runs to analyse

| Label          | m      | fx | tau_u | table name (example)                        |
|----------------|--------|----|-------|---------------------------------------------|
| SS F=A         | 1000   | A  | 10    | rk_t5_A_tau10_rho0_m1000_chi50_alpha85      |
| SS F=E         | 1000   | E  | 5     | rk_t5_E_tau5_rho0_m1000_chi50_alpha85       |
| SS F=B         | 1000   | B  | 5     | rk_t5_B_tau5_rho0_m1000_chi50_alpha85       |
| II F=A         | 0001   | A  | 10    | rk_t5_A_tau10_rho0_m0001_chi50_alpha85      |
| bipartite      | 0110   | A  | 10    | rk_t5_A_tau10_rho0_m0110_chi50_alpha85      |
| full χ=0.5     | 1111   | A  | 10    | rk_t5_A_tau10_rho0_m1111_chi50_alpha85      |
| full χ=χ*      | 1111   | A  | 10    | rk_t5_A_tau10_rho0_m1111_chi{N}_alpha85     |

Note: SS F=E and SS F=B use tau_u=5 (different edge-list table). Check whether
these runs are available in edge_lists.duckdb before building CSR blocks.

---

## File 4 — `spectral_results_analysis/table_maker.py` (new file)

Purpose: join rankings to source and institution display names from the parquet
data files; produce LaTeX and CSV ranking tables showing unit names with v.

### Data sources

Paths from `util.load_config()`. Parquet files are under `paths.data` (SSD).
Expected layout (verify filenames against actual files on the machine):

```
{paths.data}/sources.parquet          (or sources_*.parquet / partitioned)
    columns: id  (OpenAlex URI "https://openalex.org/S{int}")
             display_name
             issn_l
             type
             ...

{paths.data}/institutions.parquet     (or institutions_*.parquet)
    columns: id  (OpenAlex URI "https://openalex.org/I{int}")
             display_name
             country_code
             type
             ror
             ...
```

Use `duckdb.read_parquet(str(path))` or `duckdb.read_parquet(glob_pattern)`
if the parquet is partitioned across multiple files.

### ID join key

`source_idx` in `rankings.duckdb` is the integer extracted from the OpenAlex
source URI by stripping `"https://openalex.org/S"`. Verify this matches the
stripping logic in `prepare_data/build_edge_lists.py`.

To join in DuckDB:

```sql
-- reconstruct URI from source_idx in rankings table
SELECT r.unit_idx,
       r.v,
       r.phi2,       -- may be absent if community.py has not been run
       s.display_name,
       s.issn_l
FROM   {rk_tname} r
JOIN   read_parquet('{sources_path}') s
  ON   s.id = 'https://openalex.org/S' || CAST(r.unit_idx AS VARCHAR)
WHERE  r.unit_type = 'S'
ORDER  BY r.v DESC
LIMIT  {top_n}
```

Institution join uses `'https://openalex.org/I'` prefix and `unit_type = 'U'`.

If the parquet `id` column strips the prefix differently (e.g. already integer),
adjust the join predicate accordingly. Check one row of the parquet first.

### Tables to produce

**Table 1 — Top sources by v**

For runs: SS F=A, SS F=E, SS F=B, bipartite baseline, full-joint χ=0.5.
Filter `unit_type = 'S'`. Sort by v descending. Top N=30.
Columns: rank_v, display_name, issn_l, F_label (from field-label file), v.
Export: LaTeX (`booktabs` format) and CSV, one file per run.

**Table 2 — Top institutions by v**

For runs: II F=A, bipartite baseline, full-joint χ=0.5.
Filter `unit_type = 'U'`. Sort by v descending. Top N=30.
Columns: rank_v, display_name, country_code, v.
Export: LaTeX and CSV, one file per run.

**Table 3 — Community partition (SS baseline only)**

Sources sorted by phi2 from community.py output (not from rankings.duckdb,
since phi2 is not stored there). Pass phi2 array into this function from
`community.py` or write phi2 to a temporary CSV that `table_maker.py` reads.
Columns: display_name, F_label, phi2, v, rank_v.
Annotate bridge region: |phi2| < 0.05 (or data-driven threshold).

### Script structure

```python
def load_source_names(paths) -> pd.DataFrame:
    # Returns: source_idx (int), display_name, issn_l

def load_institution_names(paths) -> pd.DataFrame:
    # Returns: institution_idx (int), display_name, country_code

def load_field_labels(data_dir) -> dict:
    # Same logic as in community.py — consider extracting to shared util

def make_source_table(rk_db, tname, source_names, field_labels, top_n=30):
    # Returns DataFrame: rank_v, display_name, issn_l, F_label, v

def make_institution_table(rk_db, tname, inst_names, top_n=30):
    # Returns DataFrame: rank_v, display_name, country_code, v

def to_latex(df, outpath, caption, label, float_cols=None):
    # booktabs LaTeX tabular; float_cols specifies precision per column

def main():
    paths = load_config()
    source_names = load_source_names(paths)
    inst_names   = load_institution_names(paths)
    field_labels = load_field_labels(paths.data)
    rk_path      = paths.working / 'rankings.duckdb'
    with duckdb.connect(str(rk_path), read_only=True) as rk_db:
        for tname, label, unit_type in TABLE_RUNS:
            ...

TABLE_RUNS = [
    ('rk_t5_A_tau10_rho0_m1000_chi50_alpha85',  'SS-A',   'S'),
    ('rk_t5_E_tau5_rho0_m1000_chi50_alpha85',   'SS-E',   'S'),
    ('rk_t5_B_tau5_rho0_m1000_chi50_alpha85',   'SS-B',   'S'),
    ('rk_t5_A_tau10_rho0_m0001_chi50_alpha85',  'II-A',   'U'),
    ('rk_t5_A_tau10_rho0_m0110_chi50_alpha85',  'bip',    'both'),
    ('rk_t5_A_tau10_rho0_m1111_chi50_alpha85',  'full05', 'both'),
    # chi-star table name resolved at runtime from _catalog
]
```

---

## Data and ID notes for Claude Code

**config.yaml** (gitignored) provides machine-specific paths via `util.load_config()`.
Never hardcode paths. Access via `paths.working`, `paths.data`, etc.

**source_idx**: integer from `"https://openalex.org/S"` prefix strip in
`prepare_data/build_edge_lists.py`. Replicate the same logic in `table_maker.py`.

**institution_idx**: same pattern with `"https://openalex.org/I"`.

**F-label file**: mapping `source_idx → F ∈ {E, B}` should exist in `data/`
from corpus construction. If absent as a standalone file, reconstruct from the
source list used in `prepare_data/`.

**phi2 for Table 3**: since phi2 is not stored in `rankings.duckdb`, coordinate
between `community.py` and `table_maker.py`. Options: (a) `community.py` writes
a small CSV of (source_idx, phi2) to `plots/` that `table_maker.py` reads, or
(b) `table_maker.py` imports and calls the `second_eigenpair` function directly.
Option (a) is cleaner for the separation of concerns.

---

## Invariants to preserve

- `build_csr.py` and `katz_ranker.py` are pure functions with no I/O.
- `RankResult` dataclass is unchanged (no phi2, lambda2 fields).
- Ranking table schema is unchanged (no phi2 column).
- `_catalog` schema is unchanged (no lambda2 or community_amplification columns).
- All DuckDB writes stay in `run_rankings.py`; analysis scripts are read-only.
- `bipartite_resolvent` signature unchanged; √α is internal.
- Table name scheme unchanged; χ* runs use `chi{round(chi_star*100)}`.
