"""
bootstrap_oa_errors.py — 4-stage bootstrap for OpenAlex metadata errors.

Stage 1 — build_base_works
    Persist all OAS corpus works 2020-2024 with institution fractionation weights.
    Source: corpus_works.parquet + corpus_authorships.parquet.

Stage 2 — build_work_replicates
    B perturbed work tables.  For each replicate, three independent draws:
      pub_year    — 2% of in-window works are replaced by a random pre-window work
                    (2016–2019) drawn WITH REPLACEMENT from stage1_prewindow_pool.
                    The replacement inherits the source/institution of the drawn work
                    but keeps the original (in-window) pub_year.
      source      — 0.03% of remaining works get a new source (uniform OAS draw).
      institution — 1% of remaining works get a new institution (75% within-country).
    Error types are drawn independently; pub_year takes precedence on conflicts.
    Only changed rows persisted (delta format).

Stage 3 — build_ref_replicates
    B perturbed reference tables: ~2% wrong references (1.5% same-source,
    0.5% cross-source replacement).  Only changed rows persisted.

Stage 4 — build_rankings
    B bipartite (m=0110) spectral rankings per error filter, each isolating one
    source of uncertainty: resample (sampling error), source, institution,
    reference (and the superseded year model).  Error types are never combined.
    Uses the fixed baseline unit set; no tau recomputation.

Year-swap — build_year_swap_inputs / _replicates / _rankings
    The publication-year model used in the paper (needs stages 1 and 3 first):
    works dated 2019, 2020 or 2024 are shifted across the nearest window edge
    with probability 2% (2019→2020 in; 2020→2019 and 2024→2025 out), and the
    references are rebuilt from corpus_references for the shifted years.

Year-v2 — build_year_v2_pool / _replicates / _rankings
    Superseded year model (in effect a 2% deletion of window works).

A replicate with no errors reproduces the stored baseline ranking exactly
(institution weights ω and reference counts R_i are taken as in the pipeline).

Error rates (empirically estimated):
    pub_year    — 2%   of works dated 2019, 2020 or 2024 shifted across the window
                        edge (year-swap model; the stage-2 'year' and year-v2
                        models are superseded).
    source      — 0.03% uniform draw from OAS source pool
    institution — 1%   75% within-country, 25% global uniform
    references  — 2%   1.5% same-source cited-work, 0.5% cross-source cited-work

Storage: $WORKING/bootstrap_oa_errors/
    stage1_base_works.parquet
    stage2_work_errors.parquet    (replicate_id, work_idx, pub_year, source_idx,
                                   inst_idx, inst_weight, country_code, inst_idx_old)
    stage3_base_refs.parquet      (ref_idx, citer_work_idx, cited_work_idx, cited_source_idx, R_i)
    stage3_ref_errors.parquet     (replicate_id, ref_idx, new_cited_work_idx)
    stage1_meta.json, stage1_year_v2_meta.json   ω used for the stage-1 tables
    stage4_{filter}/v_s_boot.npy       (B, n_s) float32
    stage4_{filter}/v_u_boot.npy       (B, n_u) float32
    stage4_{filter}/lam_ratio_boot.npy (B,) float32
    stage4_{filter}/meta.json          error_filter, omega, unit ids
    stage1_year_swap_pool.parquet, stage1_year_swap_refs.parquet, stage2_year_swap.parquet
    stage4_year_v2/, stage4_year_swap/  same layout

CLI
---
python spectral_ranking_bootstrap/bootstrap_oa_errors.py --stage STAGES
    [--filters resample,source,institution,reference]
    [--n 1000]  [--seed 42]  [--omega {0,1}]

STAGES is a comma list from {1,2,3,4,year_v2,year_swap}, e.g. --stage 1,2,3
then --stage 4 --filters resample (one process per filter can run in parallel)
and --stage year_swap.

--omega sets the institution weighting of the replicates (default 0, the
author-fractional weighting of the baseline).  Stage-4 meta.json records it.
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs
from spectral_ranking.katz_ranker import bipartite, _row_normalise

YEAR_LO      = 2020
YEAR_HI      = 2024
YEAR_PRE_LO  = 2016    # start of pre-window replacement pool (original model)
YEAR_PRE_HI  = 2019    # end   of pre-window replacement pool
YEAR_PRE_LO_V2 = 2015  # start of pre-window pool for year-v2 model
# Per-type OA error rates (empirically estimated)
P_ERROR_YEAR = 0.020   # 2%   in-window works replaced by pre-window draws
P_ERROR_SRC  = 0.0003  # 0.03% work-to-journal misassignment
P_ERROR_INST = 0.010   # 1%   institution errors
P_ERROR_REF  = 0.020   # 2%   reference errors (total; 75% same-source, 25% cross-source)
P_WITHIN     = 0.75    # fraction of institution errors within same country
STAGE_DIR = 'bootstrap_oa_errors'
MAX_FILTER_ITERS = 200

# Stage-4 error filters.  Each isolates one source of uncertainty; they are
# deliberately not combined.
ERROR_FILTERS   = ('resample', 'source', 'institution', 'reference', 'year')
PAPER_FILTERS   = ('resample', 'source', 'institution', 'reference')  # + year_swap

# Institution weighting ω, as in params.csv: 0 = author-fractional (baseline,
# paper eq. 1; edge-list column inst_weight), 1 = direct 1/N_inst (edge-list
# column direct_inst_weight).  Replicates must use the baseline's ω so that a
# replicate with no errors reproduces the baseline ranking exactly.
DEFAULT_OMEGA = 0


def _institution_weights(paths, work_ids: np.ndarray, omega: int) -> pd.DataFrame:
    """
    One row per (work_idx, inst_idx) with the institution weight ω_iu of the work.

    Mirrors prepare_data/build_edge_lists.py (iw_raw) exactly: authorships are
    DISTINCT (work, author, institution) with a non-null institution;
      ω=0: Σ_authors 1 / (n_authors · n_inst_per_author)
      ω=1: 1 / n_institutions
    Weights are over all of the work's institutions (retained or not), as in the
    edge list.  country_code is the institution's country on that work.

    Columns: work_idx int64, inst_idx int64, inst_weight float64, country_code object.
    """
    import duckdb as _duckdb
    if omega not in (0, 1):
        raise ValueError(f'omega must be 0 or 1, got {omega!r}')
    w_expr = ('SUM(1.0 / wac.n_authors / aic.n_inst_per_author)' if omega == 0
              else 'ANY_VALUE(1.0 / wac.n_institutions)')
    con = _duckdb.connect(':memory:')
    con.register('_ids', pd.DataFrame({'work_idx': np.asarray(work_ids, dtype=np.int64)}))
    df = con.execute(f"""
        WITH auths AS (
            SELECT DISTINCT work_idx, author_idx, institution_idx
            FROM '{paths.parquet}/corpus_authorships.parquet'
            WHERE institution_idx IS NOT NULL
              AND work_idx IN (SELECT work_idx FROM _ids)
        ),
        wac AS (
            SELECT work_idx,
                   COUNT(DISTINCT author_idx)      AS n_authors,
                   COUNT(DISTINCT institution_idx) AS n_institutions
            FROM auths GROUP BY work_idx
        ),
        aic AS (
            SELECT work_idx, author_idx,
                   COUNT(DISTINCT institution_idx) AS n_inst_per_author
            FROM auths GROUP BY work_idx, author_idx
        ),
        iw AS (
            SELECT a.work_idx, a.institution_idx AS inst_idx, {w_expr} AS inst_weight
            FROM auths a
            JOIN wac ON a.work_idx = wac.work_idx
            JOIN aic ON a.work_idx = aic.work_idx AND a.author_idx = aic.author_idx
            GROUP BY a.work_idx, a.institution_idx
        ),
        cc AS (
            SELECT work_idx, institution_idx AS inst_idx, MIN(country_code) AS country_code
            FROM '{paths.parquet}/corpus_authorships.parquet'
            WHERE institution_idx IS NOT NULL
              AND work_idx IN (SELECT work_idx FROM _ids)
            GROUP BY work_idx, institution_idx
        )
        SELECT iw.work_idx, iw.inst_idx, iw.inst_weight, cc.country_code
        FROM iw LEFT JOIN cc USING (work_idx, inst_idx)
        ORDER BY iw.work_idx, iw.inst_idx
    """).fetchdf()
    con.close()
    df['work_idx']    = df['work_idx'].astype(np.int64)
    df['inst_idx']    = df['inst_idx'].astype(np.int64)
    df['inst_weight'] = df['inst_weight'].astype(np.float64)
    return df


def _attach_institutions(works: pd.DataFrame, paths, omega: int) -> pd.DataFrame:
    """
    Left-join works (work_idx, pub_year, source_idx) to their institution rows.
    Works with no institution appear once with inst_idx=NA, inst_weight=1.0.
    """
    iw = _institution_weights(paths, works['work_idx'].unique(), omega)
    out = works.merge(iw, on='work_idx', how='left')
    out.loc[out['inst_idx'].isna(), 'inst_weight'] = 1.0
    out['inst_weight'] = out['inst_weight'].astype(np.float64)
    out['inst_idx'] = out['inst_idx'].astype('Int64')
    return out


def _write_stage_meta(paths, name: str, omega: int) -> None:
    out_dir = paths.working / STAGE_DIR
    (out_dir / f'{name}_meta.json').write_text(json.dumps({'omega': omega}, indent=2))


def _read_stage_meta(paths, name: str) -> dict:
    """ω used to build a stage-1 table; files written before ω was recorded used ω=1."""
    p = paths.working / STAGE_DIR / f'{name}_meta.json'
    return json.loads(p.read_text()) if p.exists() else {'omega': 1}


# ── Stage 1 ────────────────────────────────────────────────────────────────────

def build_base_works(paths, omega: int = DEFAULT_OMEGA) -> pd.DataFrame:
    """
    Load baseline edge-list works for YEAR_LO..YEAR_HI with institution fractionation.

    Restricted to works that appear as citer or cited in the baseline edge list,
    so base_works exactly matches the baseline corpus (same set as build_edge_lists).

    Returns and saves a DataFrame with columns:
        work_idx     int64
        pub_year     int16
        source_idx   int64
        inst_idx     Int64   (nullable; pd.NA for works with no institution)
        inst_weight  float64 (ω_iu per `omega`, see _institution_weights;
                              1.0 if no institution)
        country_code object  (nullable str)

    One row per (work_idx, inst_idx).  Works with no authorship record appear
    once with inst_idx=pd.NA, inst_weight=1.0.
    """
    import duckdb as _duckdb

    out_dir = paths.working / STAGE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Stage 1: loading corpus_works ...', flush=True)
    works = pd.read_parquet(
        str(paths.parquet / 'corpus_works.parquet'),
        columns=['work_idx', 'publication_year', 'source_idx'],
    )
    works = works[works['publication_year'].between(YEAR_LO, YEAR_HI)].copy()
    works = works.rename(columns={'publication_year': 'pub_year'})
    works['pub_year'] = works['pub_year'].astype(np.int16)
    print(f'  Works {YEAR_LO}-{YEAR_HI} (all corpus): {len(works):,}', flush=True)

    # Restrict to works in the baseline edge list (citer ∪ cited)
    bl       = next(r for r in load_runs() if r['label'] == 'baseline')
    el_table = f'el_{bl["run_code"]}_{bl["fx"]}_tauU{bl["tau_u"]}_tauS{bl["tau_s"]}_vartau'
    print(f'Stage 1: restricting to baseline edge list ({el_table}) ...', flush=True)
    db = _duckdb.connect(str(paths.working / 'edge_lists.duckdb'), read_only=True)
    el_works_df = db.execute(f"""
        SELECT work_idx FROM (
            SELECT citer_work_idx AS work_idx FROM {el_table}
            UNION
            SELECT cited_work_idx  AS work_idx FROM {el_table}
        )
    """).fetchdf()
    db.close()
    baseline_work_set = set(el_works_df['work_idx'].tolist())
    works = works[works['work_idx'].isin(baseline_work_set)]
    print(f'  Works after edge-list filter: {len(works):,}', flush=True)

    print(f'Stage 1: institution weights from corpus_authorships (ω={omega}) ...', flush=True)
    base = _attach_institutions(works, paths, omega)

    # Verify containment: every work_idx in base must be in the baseline edge list
    base_work_ids = set(base['work_idx'].unique().tolist())
    assert base_work_ids <= baseline_work_set, (
        f"base_works contains {len(base_work_ids - baseline_work_set)} works "
        f"not in baseline edge list"
    )
    print(f'  Containment check passed: {len(base_work_ids):,} unique works ⊆ edge list', flush=True)

    out_path = out_dir / 'stage1_base_works.parquet'
    base.to_parquet(str(out_path), index=False)
    print(f'  Stage 1 saved: {out_path}  ({len(base):,} rows)', flush=True)

    # ── Pre-window replacement pool: all OAS corpus works 2016–2019 ───────────
    print(f'Stage 1: building pre-window pool ({YEAR_PRE_LO}–{YEAR_PRE_HI}) ...', flush=True)
    pre_works = pd.read_parquet(
        str(paths.parquet / 'corpus_works.parquet'),
        columns=['work_idx', 'publication_year', 'source_idx'],
    )
    pre_works = pre_works[pre_works['publication_year'].between(YEAR_PRE_LO, YEAR_PRE_HI)].copy()
    pre_works = pre_works.dropna(subset=['source_idx'])
    pre_works = pre_works.rename(columns={'publication_year': 'pub_year'})
    pre_works['pub_year']    = pre_works['pub_year'].astype(np.int16)
    pre_works['source_idx']  = pre_works['source_idx'].astype(np.int64)
    print(f'  Pre-window OAS works: {len(pre_works):,}', flush=True)

    pre_base = _attach_institutions(pre_works, paths, omega)

    pre_path = out_dir / 'stage1_prewindow_pool.parquet'
    pre_base.to_parquet(str(pre_path), index=False)
    _write_stage_meta(paths, 'stage1', omega)
    print(f'  Pre-window pool saved: {pre_path}  '
          f'({pre_base["work_idx"].nunique():,} unique works, {len(pre_base):,} rows)',
          flush=True)

    return base


# ── Stage 2 helpers ────────────────────────────────────────────────────────────

def _build_inst_pools(inst_pool: pd.DataFrame) -> tuple:
    """
    Pre-compute per-country institution arrays for fast vectorised draws.
    Call once before the B-replicate loop; pass the result to perturb_works_one.

    Returns (cc_to_insts, all_inst_arr, all_cc_arr) where:
        cc_to_insts  dict[str → (inst_arr int64, cc_arr object)]
        all_inst_arr ndarray int64
        all_cc_arr   ndarray object
    """
    inst_pool    = inst_pool.reset_index(drop=True)
    all_inst_arr = inst_pool['institution_idx'].to_numpy(dtype=np.int64)
    all_cc_arr   = inst_pool['country_code'].to_numpy(dtype=object)
    cc_to_insts: dict = {}
    for cc, grp in inst_pool.groupby('country_code', sort=False):
        arr = grp['institution_idx'].to_numpy(dtype=np.int64)
        cc_to_insts[str(cc)] = (arr, np.full(len(arr), cc, dtype=object))
    return cc_to_insts, all_inst_arr, all_cc_arr


def resample_works_one(in_win_ids: np.ndarray, rng: np.random.Generator) -> dict:
    """
    100%-with-replacement resample of in-window works (standard bootstrap).

    Draws len(in_win_ids) indices with replacement.
    Returns {work_idx: count} for works drawn ≥1 time; absent works are not stored.
    ~63.2% unique coverage expected per replicate (1 - 1/e).
    """
    n = len(in_win_ids)
    drawn = in_win_ids[rng.integers(0, n, size=n)]
    unique, counts = np.unique(drawn, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def perturb_works_one(base: pd.DataFrame,
                      pre_window: pd.DataFrame,
                      source_pool: np.ndarray,
                      inst_pool: pd.DataFrame,
                      rng: np.random.Generator,
                      _inst_pools: tuple | None = None) -> pd.DataFrame:
    """
    Apply one round of independent OA errors to base_works.

    Error types (all independent; pub_year takes precedence on the same work):

      pub_year (2%) — replace a randomly selected in-window work with a work drawn
          WITH REPLACEMENT from pre_window (2016-2019).  The replacement inherits
          the drawn work's source_idx, inst_idx, inst_weight, and country_code, but
          keeps the original in-window pub_year so the work remains in the corpus.
          ALL (work, inst) rows for the replaced work are overwritten.

      source (0.03%) — for remaining works, change source_idx to a uniform draw
          from source_pool.  ALL rows for the work are updated.

      institution (1%) — for remaining works, swap one institution per work
          (75% within-country, 25% global uniform).

    Returns only changed rows (delta), with extra columns:
        inst_idx_old  Int64 — original inst_idx for institution errors; pd.NA otherwise
        error_type    str

    pre_window: DataFrame from stage1_prewindow_pool.parquet (columns identical to base).
    _inst_pools: pre-built result of _build_inst_pools(inst_pool); pass from B-loop.
    """
    in_window   = base['pub_year'].between(YEAR_LO, YEAR_HI)
    in_win_ids  = base[in_window]['work_idx'].unique()
    n_in        = len(in_win_ids)

    # Independent draws for each error type
    u_py  = rng.random(n_in)
    u_src = rng.random(n_in)
    u_ins = rng.random(n_in)

    py_sel  = in_win_ids[u_py  < P_ERROR_YEAR]
    src_sel = in_win_ids[u_src < P_ERROR_SRC]
    ins_sel = in_win_ids[u_ins < P_ERROR_INST]

    # pub_year replacement takes precedence; remove those works from source/institution
    py_set  = set(py_sel.tolist())
    src_sel = src_sel[~np.isin(src_sel, list(py_set))]
    ins_set = set(ins_sel.tolist()) - py_set
    ins_sel = ins_sel[np.isin(ins_sel, list(ins_set))]
    # institution further takes precedence over source
    src_sel = src_sel[~np.isin(src_sel, list(ins_set))]

    parts: list[pd.DataFrame] = []

    # ── pub_year: replace in-window work with pre-window draw ─────────────────
    if len(py_sel) > 0 and len(pre_window) > 0:
        prewin_unique = pre_window['work_idx'].unique()
        draw_idx      = rng.integers(0, len(prewin_unique), size=len(py_sel))
        drawn_works   = prewin_unique[draw_idx]

        orig_year_ser = (base[base['work_idx'].isin(py_set)]
                         .drop_duplicates('work_idx')
                         .set_index('work_idx')['pub_year'])

        drawn_map = pd.DataFrame({
            'drawn_work': drawn_works,
            'work_idx':   py_sel,
            'pub_year':   orig_year_ser.loc[py_sel].values.astype(np.int16),
        })

        # Bring in source/inst/country from the drawn pre-window work (drop its year)
        pre_renamed = (pre_window
                       .rename(columns={'work_idx': 'drawn_work'})
                       .drop(columns=['pub_year']))
        merged = drawn_map.merge(pre_renamed, on='drawn_work', how='left')
        merged = merged.drop(columns=['drawn_work'])
        merged['pub_year']     = merged['pub_year'].astype(np.int16)
        merged['inst_idx_old'] = pd.array([pd.NA] * len(merged), dtype='Int64')
        merged['error_type']   = 'year'
        parts.append(merged)

    # ── source — vectorised ───────────────────────────────────────────────────
    if len(src_sel) > 0:
        new_srcs   = rng.choice(source_pool, size=len(src_sel), replace=True)
        src_map    = dict(zip(src_sel.tolist(), new_srcs.tolist()))
        src_subset = base[base['work_idx'].isin(set(src_sel.tolist()))].copy()
        src_subset['source_idx']   = src_subset['work_idx'].map(src_map).astype(np.int64)
        src_subset['inst_idx_old'] = pd.array([pd.NA] * len(src_subset), dtype='Int64')
        src_subset['error_type']   = 'source'
        parts.append(src_subset)

    # ── institution — vectorised, no per-work Python loop ─────────────────────
    if len(ins_sel) > 0:
        if _inst_pools is None:
            _inst_pools = _build_inst_pools(inst_pool)
        cc_to_insts, all_inst_arr, all_cc_arr = _inst_pools

        ins_subset = base[base['work_idx'].isin(ins_set)].copy().reset_index(drop=True)
        valid_rows = ins_subset[ins_subset['inst_idx'].notna()].copy()
        if len(valid_rows) > 0:
            valid_rows['_rk'] = rng.random(len(valid_rows))
            pick_pos = valid_rows.groupby('work_idx', sort=False)['_rk'].idxmin()
            picked   = valid_rows.loc[pick_pos].copy()
            n_p      = len(picked)

            countries   = picked['country_code'].to_numpy(dtype=object)
            within_flag = rng.random(n_p) < P_WITHIN
            new_inst_ids = np.empty(n_p, dtype=np.int64)
            new_inst_ccs = np.empty(n_p, dtype=object)

            cross = ~within_flag
            if cross.any():
                ci = rng.integers(0, len(all_inst_arr), size=int(cross.sum()))
                new_inst_ids[cross] = all_inst_arr[ci]
                new_inst_ccs[cross] = all_cc_arr[ci]

            for cc, (pool_insts, pool_ccs) in cc_to_insts.items():
                mask = within_flag & (countries == cc)
                if mask.any():
                    ci = rng.integers(0, len(pool_insts), size=int(mask.sum()))
                    new_inst_ids[mask] = pool_insts[ci]
                    new_inst_ccs[mask] = cc

            no_pool = within_flag & ~np.isin(countries, list(cc_to_insts.keys()))
            if no_pool.any():
                ci = rng.integers(0, len(all_inst_arr), size=int(no_pool.sum()))
                new_inst_ids[no_pool] = all_inst_arr[ci]
                new_inst_ccs[no_pool] = all_cc_arr[ci]

            old_by_work    = dict(zip(picked['work_idx'].tolist(),
                                      picked['inst_idx'].astype(np.int64).tolist()))
            new_id_by_work = dict(zip(picked['work_idx'].tolist(), new_inst_ids.tolist()))
            new_cc_by_work = dict(zip(picked['work_idx'].tolist(), new_inst_ccs.tolist()))

            ins_delta = ins_subset[
                ins_subset['work_idx'].isin(set(picked['work_idx'].tolist()))
            ].copy()

            old_arr   = ins_delta['work_idx'].map(old_by_work).to_numpy()
            cur_arr   = ins_delta['inst_idx'].to_numpy(dtype=object)
            is_target = cur_arr == old_arr

            new_id_arr  = ins_delta['work_idx'].map(new_id_by_work).to_numpy()
            new_cc_arr2 = ins_delta['work_idx'].map(new_cc_by_work).to_numpy(dtype=object)

            cur_int = np.where(is_target, new_id_arr, old_arr.astype(float)).astype(object)
            ins_delta['inst_idx'] = pd.array(
                [int(v) if v is not None and not (isinstance(v, float) and np.isnan(v))
                 else pd.NA for v in cur_int], dtype='Int64')
            ins_delta['country_code'] = np.where(is_target, new_cc_arr2,
                                                  ins_delta['country_code'].to_numpy())
            ins_delta['inst_idx_old'] = pd.array(
                [int(old_arr[i]) if is_target[i] else pd.NA
                 for i in range(len(ins_delta))], dtype='Int64')
            ins_delta['error_type'] = 'institution'
            parts.append(ins_delta)

    if not parts:
        empty = base.head(0).copy()
        empty['inst_idx_old'] = pd.array([], dtype='Int64')
        empty['error_type']   = pd.array([], dtype=str)
        return empty

    result = pd.concat(parts, ignore_index=True)
    if 'inst_idx_old' not in result.columns:
        result['inst_idx_old'] = pd.array([pd.NA] * len(result), dtype='Int64')
    return result


def build_work_replicates(base: pd.DataFrame, paths,
                          B: int = 1000, seed: int = 42) -> None:
    """
    Stage 2: generate B work replicates and save deltas.

    Saves: $WORKING/bootstrap_oa_errors/stage2_work_errors.parquet
    """
    out_dir  = paths.working / STAGE_DIR
    out_path = out_dir / 'stage2_work_errors.parquet'
    print(f'Stage 2: generating {B} work replicates ...', flush=True)

    # Pre-window pool for pub_year replacement (built in Stage 1)
    pre_path = out_dir / 'stage1_prewindow_pool.parquet'
    if not pre_path.exists():
        raise FileNotFoundError(
            f'{pre_path} not found — run Stage 1 first to build the pre-window pool.'
        )
    pre_window = pd.read_parquet(str(pre_path))
    print(f'  Pre-window pool: {pre_window["work_idx"].nunique():,} unique works '
          f'({YEAR_PRE_LO}–{YEAR_PRE_HI})', flush=True)

    source_pool = base['source_idx'].dropna().unique().astype(np.int64)
    inst_pool   = pd.read_parquet(
        str(paths.parquet / 'corpus_institutions.parquet'),
        columns=['institution_idx', 'country_code'],
    ).dropna(subset=['institution_idx'])
    inst_pool['institution_idx'] = inst_pool['institution_idx'].astype(np.int64)
    inst_pool = inst_pool.reset_index(drop=True)

    # Pre-build institution pools once; passed into every perturb call
    inst_pools = _build_inst_pools(inst_pool)

    all_parts: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    for b in range(B):
        rng   = np.random.default_rng(seed + b)
        delta = perturb_works_one(base, pre_window, source_pool, inst_pool, rng,
                                  _inst_pools=inst_pools)
        if len(delta) > 0:
            delta.insert(0, 'replicate_id', np.int32(b))
            all_parts.append(delta)
        if (b + 1) % 100 == 0:
            print(f'  {b+1}/{B}  {time.perf_counter()-t0:.1f}s', flush=True)

    out_df = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    out_df.to_parquet(str(out_path), index=False)
    print(f'  Stage 2 saved: {out_path}  ({len(out_df):,} delta rows)', flush=True)


# ── Stage 3 ────────────────────────────────────────────────────────────────────

def load_base_refs(db, el_table: str) -> pd.DataFrame:
    """
    Return DISTINCT (citer_work_idx, cited_work_idx, cited_source_idx, R_i) from
    the edge list, with a sequential ref_idx column.

    R_i is the pipeline's reference count of the citing work.  It also counts
    references to corpus works in sources below τ_S, which are not edges, so it
    can exceed the number of the citer's rows here; see _r_extra.
    """
    df = db.execute(f"""
        SELECT DISTINCT citer_work_idx, cited_work_idx, cited_source_idx, R_i
        FROM {el_table}
        ORDER BY citer_work_idx, cited_work_idx
    """).fetchdf()
    df.insert(0, 'ref_idx', np.arange(len(df), dtype=np.int64))
    return df


def build_ref_replicates(db, el_table: str, base_works: pd.DataFrame,
                         paths, B: int = 1000, seed: int = 42) -> None:
    """
    Stage 3: generate B reference replicates and save deltas.

    Saves:
      stage3_base_refs.parquet   — (ref_idx, citer_work_idx, cited_work_idx, cited_source_idx, R_i)
      stage3_ref_errors.parquet  — (replicate_id, ref_idx, new_cited_work_idx)
    """
    out_dir = paths.working / STAGE_DIR
    print('Stage 3: loading base references from edge list ...', flush=True)
    base_refs = load_base_refs(db, el_table)
    refs_path = out_dir / 'stage3_base_refs.parquet'
    base_refs.to_parquet(str(refs_path), index=False)
    print(f'  Base refs: {len(base_refs):,} rows  saved: {refs_path}', flush=True)

    # Replacement pool: cited works in the baseline edge list only (bug fix —
    # previously used all base_works, which includes non-cited works)
    work_by_src: dict[int, np.ndarray] = {}
    cited_pairs = base_refs[['cited_work_idx', 'cited_source_idx']].drop_duplicates()
    for src, grp in cited_pairs.groupby('cited_source_idx'):
        work_by_src[int(src)] = grp['cited_work_idx'].to_numpy(dtype=np.int64)
    all_work_ids = base_refs['cited_work_idx'].unique().astype(np.int64)

    cited_src = base_refs['cited_source_idx'].to_numpy(dtype=np.int64)
    N_refs    = len(base_refs)
    p_same    = 0.75 * P_ERROR_REF
    p_diff    = P_ERROR_REF

    print(f'Stage 3: generating {B} ref replicates ...', flush=True)
    all_parts: list[pd.DataFrame] = []
    t0 = time.perf_counter()

    for b in range(B):
        rng  = np.random.default_rng(seed + 10_000 + b)
        u    = rng.random(N_refs)
        same = u < p_same
        diff = (u >= p_same) & (u < p_diff)
        err_idx = np.where(same | diff)[0]
        if len(err_idx) == 0:
            continue

        new_cited  = np.empty(len(err_idx), dtype=np.int64)
        err_types  = np.empty(len(err_idx), dtype=object)
        for k, idx in enumerate(err_idx):
            src = int(cited_src[idx])
            if same[idx]:
                pool = work_by_src.get(src, all_work_ids)
                err_types[k] = 'ref_same'
            else:
                pool = all_work_ids
                err_types[k] = 'ref_cross'
            new_cited[k] = pool[int(rng.integers(0, len(pool)))]

        # All replacement targets must be in the baseline cited-work pool
        assert np.isin(new_cited, all_work_ids).all(), (
            f"Stage 3 replicate {b}: "
            f"{(~np.isin(new_cited, all_work_ids)).sum()} new_cited values "
            f"outside baseline cited-work pool"
        )

        part = pd.DataFrame({
            'replicate_id':       np.full(len(err_idx), b, dtype=np.int32),
            'ref_idx':            err_idx.astype(np.int64),
            'new_cited_work_idx': new_cited,
            'error_type':         err_types.astype(str),
        })
        all_parts.append(part)

        if (b + 1) % 100 == 0:
            print(f'  {b+1}/{B}  {time.perf_counter()-t0:.1f}s', flush=True)

    errors_df = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    err_path  = out_dir / 'stage3_ref_errors.parquet'
    errors_df.to_parquet(str(err_path), index=False)
    print(f'  Stage 3 saved: {err_path}  ({len(errors_df):,} delta rows)', flush=True)


# ── Stage 4 helpers ────────────────────────────────────────────────────────────

def apply_work_delta(base: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    """
    Overlay delta rows onto base_works for one replicate.

    For Y/S errors: delta contains ALL (work, inst) rows for affected works, so
    we drop those works from base and append delta (without bookkeeping columns).
    For I errors: same treatment — delta contains the full updated row set for
    the affected work.

    delta must have a 'replicate_id' column (dropped here) and optionally
    'inst_idx_old' (dropped here).
    """
    if delta is None or len(delta) == 0:
        return base
    drop_cols = [c for c in ('replicate_id', 'inst_idx_old') if c in delta.columns]
    delta_clean = delta.drop(columns=drop_cols)
    changed = delta_clean['work_idx'].unique()
    base_reduced = base[~base['work_idx'].isin(changed)]
    return pd.concat([base_reduced, delta_clean], ignore_index=True)


def _r_extra(base_refs_df: pd.DataFrame) -> pd.Series | None:
    """
    Per citing work, the references counted in the pipeline's R_i that are not
    edge-list references (references to corpus works in sources below τ_S).

    ρ_i = R̄ / R_i in the pipeline (build_csr).  A replicate recounts the citer's
    kept edge-list references and adds this fixed extra, so with no errors the
    replicate's ρ_i equals the pipeline's exactly.  Returns None for stage-3
    files written before R_i was stored (legacy: ρ from edge-list refs only).
    """
    if 'R_i' not in base_refs_df.columns:
        print('  WARNING: stage3_base_refs has no R_i — ρ uses edge-list references '
              'only and will not match the pipeline; re-run stage 3.', flush=True)
        return None
    g = base_refs_df.groupby('citer_work_idx')
    extra = g['R_i'].first().astype(np.int64) - g.size()
    if (extra < 0).any():
        raise ValueError('R_i smaller than the number of edge-list references')
    return extra[extra > 0]


def apply_ref_delta(base_refs_arr: np.ndarray, err_idx: np.ndarray,
                    new_cited: np.ndarray) -> np.ndarray:
    """
    Return a modified copy of base_refs_arr with cited_work column patched.
    base_refs_arr: (N, 3) int64 — columns [citer_work, cited_work, cited_source]
    """
    refs = base_refs_arr.copy()
    refs[err_idx, 1] = new_cited
    return refs


def _build_matrices(refs_arr: np.ndarray, works_yr: pd.DataFrame,
                    src_index: pd.Index, inst_index: pd.Index,
                    r_bar: float, n_s: int, n_u: int,
                    work_mult: dict | None = None,
                    r_extra: pd.Series | None = None,
                    ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Build C_SI (n_s × n_u) and C_IS (n_u × n_s) from reference and work arrays.

    refs_arr: (N, 3) int64 — [citer_work_idx, cited_work_idx, cited_source_idx]
              (cited_source_idx column is ignored; we re-derive from works_yr)
    works_yr: DataFrame filtered to pub_year window, columns:
              work_idx, pub_year, source_idx, inst_idx, inst_weight

    Uses fixed baseline unit indices (src_index, inst_index).
    Rows with source/institution not in units are silently dropped.

    r_extra: per-citer count of references outside the edge list (from _r_extra),
             added to the recounted R_i so that ρ_i matches the pipeline.
    """
    # ── Filter references to works in the year window ─────────────────────────
    valid_set = set(works_yr['work_idx'].to_numpy().tolist())
    citer_ok  = np.isin(refs_arr[:, 0], list(valid_set))
    cited_ok  = np.isin(refs_arr[:, 1], list(valid_set))
    if work_mult is not None:
        # Citers absent from the resample (multiplicity 0) contribute no references
        citer_ok = citer_ok & np.array(
            [work_mult.get(int(w), 0) > 0 for w in refs_arr[:, 0]], dtype=bool)
    keep      = citer_ok & cited_ok
    refs      = refs_arr[keep]
    if len(refs) == 0:
        return (sp.csr_matrix((n_s, n_u)), sp.csr_matrix((n_u, n_s)))

    # ── Compute R_i and rho_w ──────────────────────────────────────────────────
    uniq, inv, counts = np.unique(refs[:, 0], return_inverse=True, return_counts=True)
    if r_extra is not None:
        counts = counts + r_extra.reindex(uniq, fill_value=0).to_numpy(dtype=np.int64)
    rho_w = r_bar / counts[inv]   # (N_keep,)
    if work_mult is not None:
        # Scale each reference by the citer's resample count (k appearances = k× weight)
        rho_w = rho_w * np.array(
            [work_mult.get(int(cw), 1) for cw in refs[:, 0]], dtype=np.float64)

    # ── Work-property lookup: work_idx → source_idx (int64) ───────────────────
    # A work has exactly one source; multiple rows (one per inst).
    src_lookup = (works_yr.drop_duplicates('work_idx')
                           .set_index('work_idx')['source_idx'])

    # Work-property for institutions: (work_idx, inst_idx, inst_weight)
    inst_lookup = works_yr[works_yr['inst_idx'].notna()].copy()
    inst_lookup['inst_idx'] = inst_lookup['inst_idx'].astype(np.int64)

    # ── C_SI: citer_source → cited_inst ───────────────────────────────────────
    # One row per (citer_work, citer_source, cited_work, cited_inst) unique.
    # Join refs with cited inst assignments on cited_work_idx.
    refs_df = pd.DataFrame({
        'citer_work': refs[:, 0],
        'cited_work': refs[:, 1],
        'rho_w':      rho_w,
    })
    refs_df['citer_source'] = refs_df['citer_work'].map(src_lookup)
    cited_insts = inst_lookup[['work_idx', 'inst_idx', 'inst_weight']].rename(
        columns={'work_idx': 'cited_work', 'inst_idx': 'cited_inst',
                 'inst_weight': 'cited_iw'})
    si_df = refs_df.merge(cited_insts, on='cited_work', how='inner')
    si_df = si_df.dropna(subset=['citer_source'])
    si_df = si_df.drop_duplicates(['citer_work', 'citer_source', 'cited_work', 'cited_inst'])
    si_df['w'] = si_df['rho_w'] * si_df['cited_iw']

    cs_d = src_index.get_indexer(si_df['citer_source'].to_numpy(dtype=np.int64))
    ci_d = inst_index.get_indexer(si_df['cited_inst'].to_numpy(dtype=np.int64))
    ok   = (cs_d >= 0) & (ci_d >= 0)
    C_SI = sp.coo_matrix(
        (si_df['w'].to_numpy(dtype=np.float64)[ok], (cs_d[ok], ci_d[ok])),
        shape=(n_s, n_u),
    ).tocsr()

    # ── C_IS: citer_inst → cited_source ───────────────────────────────────────
    citer_insts = inst_lookup[['work_idx', 'inst_idx', 'inst_weight']].rename(
        columns={'work_idx': 'citer_work', 'inst_idx': 'citer_inst',
                 'inst_weight': 'citer_iw'})
    is_df = refs_df.merge(citer_insts, on='citer_work', how='inner')
    is_df['cited_source'] = is_df['cited_work'].map(src_lookup)
    is_df = is_df.dropna(subset=['cited_source'])
    is_df = is_df.drop_duplicates(['citer_work', 'citer_inst', 'cited_work', 'cited_source'])
    is_df['w'] = is_df['rho_w'] * is_df['citer_iw']

    ci2_d = inst_index.get_indexer(is_df['citer_inst'].to_numpy(dtype=np.int64))
    cs2_d = src_index.get_indexer(is_df['cited_source'].to_numpy(dtype=np.int64))
    ok2   = (ci2_d >= 0) & (cs2_d >= 0)
    C_IS  = sp.coo_matrix(
        (is_df['w'].to_numpy(dtype=np.float64)[ok2], (ci2_d[ok2], cs2_d[ok2])),
        shape=(n_u, n_s),
    ).tocsr()

    return C_SI, C_IS


def _rank_one(C_SI: sp.csr_matrix, C_IS: sp.csr_matrix,
              n_s: int, n_u: int,
              a_s: np.ndarray, a_u: np.ndarray,
              A_full: float) -> tuple:
    """
    Filter to bipartite core, run spectral ranking, return full-length v arrays
    (NaN for units absent from core) and diagnostics.
    """
    s_idx, u_idx, n_iter = _bipartite_core(C_SI, C_IS)
    sub_SI = C_SI[s_idx, :][:, u_idx]
    sub_IS = C_IS[u_idx, :][:, s_idx]
    H_SI, _ = _row_normalise(sub_SI)
    H_IS, _ = _row_normalise(sub_IS)
    pi_s, pi_u, lam1, lam2, _iters, _norm = bipartite(H_SI, H_IS, alpha=1.0)

    v_s = np.full(n_s, np.nan, dtype=np.float32)
    v_u = np.full(n_u, np.nan, dtype=np.float32)
    v_s[s_idx] = (A_full * pi_s / 2.0 / a_s[s_idx]).astype(np.float32)
    v_u[u_idx] = (A_full * pi_u / 2.0 / a_u[u_idx]).astype(np.float32)
    return v_s, v_u, float(lam1), float(lam2), len(s_idx), len(u_idx), n_iter


def _bipartite_core(C_SI: sp.csr_matrix, C_IS: sp.csr_matrix) -> tuple:
    """
    Iteratively remove sources/institutions with zero indegree or outdegree
    until convergence (bipartite analogue of SCC).

    A source is retained iff it has:
      - at least one outgoing SI edge (row of C_SI non-zero)
      - at least one incoming IS edge (column of C_IS non-zero)

    An institution is retained iff it has:
      - at least one outgoing IS edge (row of C_IS non-zero)
      - at least one incoming SI edge (column of C_SI non-zero)

    Returns
    -------
    s_idx  : int array — global source indices retained
    u_idx  : int array — global institution indices retained
    n_iter : int — number of filtering passes to convergence
    """
    s_idx = np.arange(C_SI.shape[0], dtype=np.int32)
    u_idx = np.arange(C_IS.shape[0], dtype=np.int32)

    for n_iter in range(1, MAX_FILTER_ITERS + 1):
        sub_SI = C_SI[s_idx, :][:, u_idx]
        sub_IS = C_IS[u_idx, :][:, s_idx]

        s_out = np.diff(sub_SI.indptr) > 0
        s_in  = np.asarray(sub_IS.sum(axis=0)).ravel() > 0
        u_out = np.diff(sub_IS.indptr) > 0
        u_in  = np.asarray(sub_SI.sum(axis=0)).ravel() > 0

        s_keep = s_out & s_in
        u_keep = u_out & u_in

        if s_keep.all() and u_keep.all():
            break

        s_idx = s_idx[s_keep]
        u_idx = u_idx[u_keep]

    return s_idx, u_idx, n_iter


# ── Stage 4 ────────────────────────────────────────────────────────────────────

def build_rankings(paths, error_filter: str, B: int = 1000, seed: int = 42) -> None:
    """
    Stage 4: build B bipartite (m=0110) spectral rankings, each isolating one
    source of uncertainty.

    error_filter : one of ERROR_FILTERS
        'resample'    — 100%-with-replacement resample of works only (sampling error)
        'source'      — source-error work deltas only
        'institution' — institution-error work deltas only
        'reference'   — reference deltas only
        'year'        — original year-error model (superseded by year_v2; not in paper)
    Output goes to stage4_{filter}/.  The year-v2 model is run by build_year_v2_rankings.
    """
    import duckdb

    out_dir    = paths.working / STAGE_DIR
    if error_filter not in ERROR_FILTERS:
        raise ValueError(f'error_filter must be one of {ERROR_FILTERS}, got {error_filter!r}')
    stage4_dir = out_dir / f'stage4_{error_filter}'
    stage4_dir.mkdir(parents=True, exist_ok=True)

    # ── Load baseline units ────────────────────────────────────────────────────
    bl          = next(r for r in load_runs() if r['label'] == 'baseline')
    run_code    = bl['run_code'];  tau_u = bl['tau_u'];  tau_s = bl['tau_s']
    fx          = bl['fx']
    el_table    = f'el_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_vartau'
    units_table = f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_vartau_m0110'

    db       = duckdb.connect(str(paths.working / 'edge_lists.duckdb'), read_only=True)
    units_df = db.execute(
        f'SELECT unit_idx, unit_type, a_p FROM "{units_table}" ORDER BY unit_type, unit_idx'
    ).fetchdf()
    r_bar = db.execute(
        f'SELECT AVG(rval) FROM '
        f'(SELECT DISTINCT citer_work_idx, CAST(R_i AS DOUBLE) AS rval FROM {el_table})'
    ).fetchone()[0]
    db.close()

    src_df  = units_df[units_df['unit_type'] == 'S'].reset_index(drop=True)
    inst_df = units_df[units_df['unit_type'] == 'U'].reset_index(drop=True)
    source_ids = src_df['unit_idx'].to_numpy(dtype=np.int64)
    inst_ids   = inst_df['unit_idx'].to_numpy(dtype=np.int64)
    a_s        = src_df['a_p'].to_numpy(dtype=np.float64)
    a_u        = inst_df['a_p'].to_numpy(dtype=np.float64)
    n_s        = len(source_ids);  n_u = len(inst_ids)
    A_full     = float(a_s.sum() + a_u.sum())
    src_index  = pd.Index(source_ids)
    inst_index = pd.Index(inst_ids)
    print(f'Units: n_s={n_s}, n_u={n_u},  r_bar={r_bar:.4f}', flush=True)

    # ── Load stage inputs ──────────────────────────────────────────────────────
    print('Stage 4: loading stage 1-3 outputs ...', flush=True)
    base_works = pd.read_parquet(str(out_dir / 'stage1_base_works.parquet'))
    in_win_ids = (base_works[base_works['pub_year'].between(YEAR_LO, YEAR_HI)]
                  ['work_idx'].unique().astype(np.int64))
    print(f'  In-window works available for resample: {len(in_win_ids):,}', flush=True)

    work_errors = pd.read_parquet(str(out_dir / 'stage2_work_errors.parquet'))
    work_deltas: dict[int, pd.DataFrame] = {
        int(rid): grp for rid, grp in work_errors.groupby('replicate_id')
    }

    base_refs_df  = pd.read_parquet(str(out_dir / 'stage3_base_refs.parquet'))
    base_refs_arr = base_refs_df[['citer_work_idx', 'cited_work_idx',
                                   'cited_source_idx']].to_numpy(dtype=np.int64)
    r_extra = _r_extra(base_refs_df)

    ref_errors = pd.read_parquet(str(out_dir / 'stage3_ref_errors.parquet'))
    ref_deltas_all: dict[int, tuple[np.ndarray, np.ndarray]] = {
        int(rid): (grp['ref_idx'].to_numpy(dtype=np.int64),
                   grp['new_cited_work_idx'].to_numpy(dtype=np.int64))
        for rid, grp in ref_errors.groupby('replicate_id')
    }

    # ── Apply error_filter ────────────────────────────────────────────────────
    if error_filter in ('year', 'source', 'institution'):
        filtered = work_errors[work_errors['error_type'] == error_filter]
        work_deltas = {int(rid): grp
                       for rid, grp in filtered.groupby('replicate_id')}
        ref_deltas: dict = {}
        print(f'  error_filter={error_filter!r}: '
              f'{len(work_deltas)} work replicates with this error type, no ref deltas',
              flush=True)
    elif error_filter == 'reference':
        work_deltas = {}
        ref_deltas  = ref_deltas_all
        print(f'  error_filter=reference: no work deltas, '
              f'{len(ref_deltas)} ref replicates, no resample', flush=True)
    else:  # 'resample'
        work_deltas = {}
        ref_deltas  = {}
        print(f'  error_filter=resample: sampling error only (100% w/ replacement)',
              flush=True)

    max_b2 = max(work_deltas) + 1 if work_deltas else 1
    max_b3 = max(ref_deltas)  + 1 if ref_deltas  else 1

    # ── Draw B pairings ────────────────────────────────────────────────────────
    rng_pair = np.random.default_rng(seed + 99_999)
    b2_arr   = rng_pair.integers(0, max_b2, size=B)
    b3_arr   = rng_pair.integers(0, max_b3, size=B)

    # ── Allocate outputs ───────────────────────────────────────────────────────
    v_s_boot       = np.full((B, n_s), np.nan, dtype=np.float32)
    v_u_boot       = np.full((B, n_u), np.nan, dtype=np.float32)
    lam_ratio_boot = np.full(B, np.nan, dtype=np.float32)
    rep_times: list[float] = []

    print(f'Stage 4: running {B} replicates ...', flush=True)
    t0 = time.perf_counter()

    for b in range(B):
        t_rep = time.perf_counter()
        b2, b3 = int(b2_arr[b]), int(b3_arr[b])

        works_b = apply_work_delta(base_works, work_deltas.get(b2))
        works_yr = works_b[works_b['pub_year'].between(YEAR_LO, YEAR_HI)]

        ref_err = ref_deltas.get(b3)
        refs_b  = (apply_ref_delta(base_refs_arr, ref_err[0], ref_err[1])
                   if ref_err is not None else base_refs_arr)

        if error_filter == 'resample':
            rng_rs    = np.random.default_rng(seed + 30_000 + b)
            work_mult = resample_works_one(in_win_ids, rng_rs)
        else:
            work_mult = None

        C_SI, C_IS = _build_matrices(refs_b, works_yr,
                                     src_index, inst_index, r_bar, n_s, n_u,
                                     work_mult=work_mult, r_extra=r_extra)

        if C_SI.nnz == 0 or C_IS.nnz == 0:
            continue

        v_s_b, v_u_b, lam1, lam2, n_sc, n_uc, n_it = \
            _rank_one(C_SI, C_IS, n_s, n_u, a_s, a_u, A_full)

        v_s_boot[b]       = v_s_b
        v_u_boot[b]       = v_u_b
        lam_ratio_boot[b] = float(lam2 / lam1) if lam1 > 0 else np.nan
        elapsed = time.perf_counter() - t_rep
        rep_times.append(elapsed)

        if (b + 1) % 10 == 0 or b == 0:
            avg = np.mean(rep_times[-10:])
            eta = (B - b - 1) * avg
            print(f'  rep {b+1:4d}/{B}  {elapsed:.2f}s  '
                  f'core=({n_sc},{n_uc})  ETA {eta/60:.1f}min', flush=True)

    # ── Save ───────────────────────────────────────────────────────────────────
    np.save(str(stage4_dir / 'v_s_boot.npy'),     v_s_boot)
    np.save(str(stage4_dir / 'v_u_boot.npy'),     v_u_boot)
    np.save(str(stage4_dir / 'lam_ratio_boot.npy'), lam_ratio_boot)

    meta = dict(
        n=B, seed=seed, error_filter=error_filter, n_s=n_s, n_u=n_u,
        source_ids=source_ids.tolist(),
        inst_ids=inst_ids.tolist(),
        run_code=run_code, units_table=units_table,
        omega=_read_stage_meta(paths, "stage1")["omega"],
        b2_arr=b2_arr.tolist(), b3_arr=b3_arr.tolist(),
    )
    (stage4_dir / 'meta.json').write_text(json.dumps(meta, indent=2))

    total = time.perf_counter() - t0
    avg   = float(np.mean(rep_times)) if rep_times else 0.0
    print(f'\nDone. {B} replicates  {total/60:.1f}min  avg={avg:.2f}s/rep', flush=True)
    print(f'Output: {stage4_dir}', flush=True)


# ── Year-v2 model ─────────────────────────────────────────────────────────────
#
# Pool: all corpus works with YEAR_PRE_LO_V2 ≤ py ≤ YEAR_HI (2015-2024).
# Each replicate draws 2% of the pool at random.
#   In-window draw  (2020-2024): DELETE the work from the corpus.
#   Pre-window draw (2015-2019): INSERT it with pub_year reassigned to 2020,
#                                keeping original source_idx, inst_idx, etc.
# Saved to stage2_year_v2.parquet; rankings in stage4_year_v2/.
#
# NOTE: replicate references are the baseline's (stage3_base_refs), and no
# inserted work appears in them (checked 2026-11-26: 0 of ~418k).  Inserted
# works therefore add no edges, and the model is in effect a 2% deletion of
# window works (~6,000 of 299,810 ranked-corpus works per replicate).  The paper
# (§4.4, Table 3) describes it that way; misdating *into* the window is not modelled.


def build_year_v2_pool(paths, omega: int = DEFAULT_OMEGA) -> pd.DataFrame:
    """
    Build the 2015-2024 work pool for the year-v2 model.

    Saves stage1_year_v2_pool.parquet and returns it.
    Columns: work_idx, pub_year, source_idx, inst_idx (Int64), inst_weight,
             country_code.  One row per (work, institution); works with no
             institution appear once with inst_idx=NA, inst_weight=1.0.
    """
    out_dir  = paths.working / STAGE_DIR
    out_path = out_dir / 'stage1_year_v2_pool.parquet'

    print(f'Year-v2: building pool {YEAR_PRE_LO_V2}–{YEAR_HI} ...', flush=True)
    works = pd.read_parquet(
        str(paths.parquet / 'corpus_works.parquet'),
        columns=['work_idx', 'publication_year', 'source_idx'],
    )
    works = works[works['publication_year'].between(YEAR_PRE_LO_V2, YEAR_HI)].copy()
    works = works.rename(columns={'publication_year': 'pub_year'})
    works['pub_year']    = works['pub_year'].astype(np.int16)
    works['source_idx']  = works['source_idx'].astype(np.int64)

    pool = _attach_institutions(works, paths, omega)

    pool.to_parquet(str(out_path), index=False)
    _write_stage_meta(paths, 'stage1_year_v2', omega)
    n_works = pool['work_idx'].nunique()
    in_win  = pool[pool['pub_year'].between(YEAR_LO, YEAR_HI)]['work_idx'].nunique()
    pre_win = n_works - in_win
    print(f'  Pool: {n_works:,} unique works  '
          f'(in-window {in_win:,}, pre-window {pre_win:,})', flush=True)
    print(f'  Saved: {out_path}', flush=True)
    return pool


def build_year_v2_replicates(pool: pd.DataFrame, paths,
                              B: int = 1000, seed: int = 42) -> None:
    """
    Generate B year-v2 replicates and save deltas.

    For each replicate, selects 2% of unique works in the 2015-2024 pool.
      In-window  (2020-2024): mark for deletion  (error_type='del').
      Pre-window (2015-2019): insert into corpus with pub_year=2020
                              keeping original source/inst (error_type='ins').

    Saves stage2_year_v2.parquet with columns:
        replicate_id  int32
        work_idx      int64
        pub_year      int16
        source_idx    int64
        inst_idx      Int64
        inst_weight   float64
        country_code  object
        error_type    str     ('del' | 'ins')
    """
    out_dir  = paths.working / STAGE_DIR
    out_path = out_dir / 'stage2_year_v2.parquet'

    all_work_ids = pool['work_idx'].unique().astype(np.int64)
    n_pool       = len(all_work_ids)
    py_ser       = pool.drop_duplicates('work_idx').set_index('work_idx')['pub_year']

    print(f'Year-v2: generating {B} replicates '
          f'(pool={n_pool:,}, rate={P_ERROR_YEAR:.1%}) ...', flush=True)

    all_parts: list[pd.DataFrame] = []
    t0 = time.perf_counter()

    for b in range(B):
        rng = np.random.default_rng(seed + b)
        u   = rng.random(n_pool)
        sel = all_work_ids[u < P_ERROR_YEAR]
        if len(sel) == 0:
            continue

        sel_py = py_ser.loc[sel].values.astype(np.int16)
        in_win_mask  = (sel_py >= YEAR_LO) & (sel_py <= YEAR_HI)
        pre_win_mask = (sel_py >= YEAR_PRE_LO_V2) & (sel_py < YEAR_LO)

        parts = []

        # Deletions — just the work_idx is needed; other columns are placeholders
        if in_win_mask.any():
            del_ids = sel[in_win_mask]
            del_df  = pool[pool['work_idx'].isin(del_ids)][
                ['work_idx', 'pub_year', 'source_idx', 'inst_idx',
                 'inst_weight', 'country_code']
            ].copy()
            del_df['error_type'] = 'del'
            parts.append(del_df)

        # Insertions — use first (work, inst) row per pre-window work; set py=2020
        if pre_win_mask.any():
            ins_ids = sel[pre_win_mask]
            ins_df  = pool[pool['work_idx'].isin(ins_ids)][
                ['work_idx', 'pub_year', 'source_idx', 'inst_idx',
                 'inst_weight', 'country_code']
            ].copy()
            ins_df['pub_year']   = np.int16(YEAR_LO)
            ins_df['error_type'] = 'ins'
            parts.append(ins_df)

        if not parts:
            continue

        delta = pd.concat(parts, ignore_index=True)
        delta.insert(0, 'replicate_id', np.int32(b))
        all_parts.append(delta)

        if (b + 1) % 100 == 0:
            print(f'  {b+1}/{B}  {time.perf_counter()-t0:.1f}s', flush=True)

    out_df = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    out_df.to_parquet(str(out_path), index=False)
    print(f'  Saved: {out_path}  ({len(out_df):,} delta rows)', flush=True)


def apply_year_v2_delta(base: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    """
    Apply one year-v2 replicate delta to base_works.

    Rows with error_type='del' are removed from base (work_idx dropped).
    Rows with error_type='ins' are appended (pub_year already set to 2020).
    """
    if delta is None or len(delta) == 0:
        return base
    delta = delta.drop(columns=['replicate_id'], errors='ignore')
    del_ids = set(delta.loc[delta['error_type'] == 'del', 'work_idx'].tolist())
    ins_df  = delta[delta['error_type'] == 'ins'].drop(columns=['error_type'])
    result  = base[~base['work_idx'].isin(del_ids)]
    if len(ins_df):
        result = pd.concat([result, ins_df], ignore_index=True)
    return result


def build_year_v2_rankings(paths, B: int = 1000, seed: int = 42) -> None:
    """
    Stage 4 for the year-v2 model.

    Loads stage2_year_v2.parquet, applies each replicate's delta via
    apply_year_v2_delta, runs the bipartite spectral ranking, and saves
    outputs to stage4_year_v2/.
    """
    import duckdb

    out_dir    = paths.working / STAGE_DIR
    stage4_dir = out_dir / 'stage4_year_v2'
    stage4_dir.mkdir(parents=True, exist_ok=True)

    bl          = next(r for r in load_runs() if r['label'] == 'baseline')
    run_code    = bl['run_code'];  tau_u = bl['tau_u'];  tau_s = bl['tau_s']
    fx          = bl['fx']
    el_table    = f'el_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_vartau'
    units_table = f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_vartau_m0110'

    db = duckdb.connect(str(paths.working / 'edge_lists.duckdb'), read_only=True)
    units_df = db.execute(
        f'SELECT unit_idx, unit_type, a_p FROM "{units_table}" ORDER BY unit_type, unit_idx'
    ).fetchdf()
    r_bar = db.execute(
        f'SELECT AVG(rval) FROM '
        f'(SELECT DISTINCT citer_work_idx, CAST(R_i AS DOUBLE) AS rval FROM {el_table})'
    ).fetchone()[0]
    db.close()

    src_df  = units_df[units_df['unit_type'] == 'S'].reset_index(drop=True)
    inst_df = units_df[units_df['unit_type'] == 'U'].reset_index(drop=True)
    source_ids = src_df['unit_idx'].to_numpy(dtype=np.int64)
    inst_ids   = inst_df['unit_idx'].to_numpy(dtype=np.int64)
    a_s        = src_df['a_p'].to_numpy(dtype=np.float64)
    a_u        = inst_df['a_p'].to_numpy(dtype=np.float64)
    n_s        = len(source_ids);  n_u = len(inst_ids)
    A_full     = float(a_s.sum() + a_u.sum())
    src_index  = pd.Index(source_ids)
    inst_index = pd.Index(inst_ids)
    print(f'Units: n_s={n_s}, n_u={n_u},  r_bar={r_bar:.4f}', flush=True)

    base_works = pd.read_parquet(str(out_dir / 'stage1_base_works.parquet'))
    base_refs_df  = pd.read_parquet(str(out_dir / 'stage3_base_refs.parquet'))
    base_refs_arr = base_refs_df[
        ['citer_work_idx', 'cited_work_idx', 'cited_source_idx']
    ].to_numpy(dtype=np.int64)
    r_extra = _r_extra(base_refs_df)

    year_v2_deltas_df = pd.read_parquet(str(out_dir / 'stage2_year_v2.parquet'))
    year_v2_deltas = {
        int(rid): grp for rid, grp in year_v2_deltas_df.groupby('replicate_id')
    }

    v_s_boot       = np.full((B, n_s), np.nan, dtype=np.float32)
    v_u_boot       = np.full((B, n_u), np.nan, dtype=np.float32)
    lam_ratio_boot = np.full(B, np.nan, dtype=np.float32)
    rep_times: list[float] = []

    print(f'Stage 4 year-v2: running {B} replicates ...', flush=True)
    t0 = time.perf_counter()

    for b in range(B):
        t_rep   = time.perf_counter()
        works_b = apply_year_v2_delta(base_works, year_v2_deltas.get(b))
        works_yr = works_b[works_b['pub_year'].between(YEAR_LO, YEAR_HI)]

        C_SI, C_IS = _build_matrices(
            base_refs_arr, works_yr, src_index, inst_index, r_bar, n_s, n_u,
            r_extra=r_extra,
        )
        if C_SI.nnz == 0 or C_IS.nnz == 0:
            continue

        v_s_b, v_u_b, lam1, lam2, n_sc, n_uc, n_it = \
            _rank_one(C_SI, C_IS, n_s, n_u, a_s, a_u, A_full)

        v_s_boot[b]       = v_s_b
        v_u_boot[b]       = v_u_b
        lam_ratio_boot[b] = float(lam2 / lam1) if lam1 > 0 else np.nan

        elapsed = time.perf_counter() - t_rep
        rep_times.append(elapsed)
        if (b + 1) % 10 == 0 or b == 0:
            avg = np.mean(rep_times[-10:])
            eta = (B - b - 1) * avg
            print(f'  rep {b+1:4d}/{B}  {elapsed:.2f}s  '
                  f'core=({n_sc},{n_uc})  ETA {eta/60:.1f}min', flush=True)

    np.save(str(stage4_dir / 'v_s_boot.npy'),       v_s_boot)
    np.save(str(stage4_dir / 'v_u_boot.npy'),       v_u_boot)
    np.save(str(stage4_dir / 'lam_ratio_boot.npy'), lam_ratio_boot)

    meta = dict(
        n=B, seed=seed, error_filter='year_v2', n_s=n_s, n_u=n_u,
        source_ids=source_ids.tolist(), inst_ids=inst_ids.tolist(),
        run_code=run_code, units_table=units_table,
        omega=_read_stage_meta(paths, "stage1_year_v2")["omega"],
        completed=B,
    )
    (stage4_dir / 'meta.json').write_text(json.dumps(meta, indent=2))

    total = time.perf_counter() - t0
    avg   = float(np.mean(rep_times)) if rep_times else 0.0
    print(f'\nDone. {B} replicates  {total/60:.1f}min  avg={avg:.2f}s/rep', flush=True)
    print(f'Output: {stage4_dir}', flush=True)


# ── Year-swap model (misdating across the window edges) ───────────────────────
#
# A publication-year error is usually a one-year shift (online-first vs issue
# year, or a working-paper date carried by a merged version).  It changes the
# ranking only when it moves a work across an edge of the 2020-2024 window:
#   2019 → 2020  into the window   (the work becomes a reference target, and
#                                    its references to window works count)
#   2020 → 2019  out of the window
#   2024 → 2025  out of the window
# Each work published in a boundary year is shifted with probability
# P_ERROR_YEAR.  2025 → 2024 is not modelled: the frozen corpus ends in 2024.
#
# References are rebuilt from corpus_references exactly as build_edge_lists
# would for the shifted years (both ends in the window and holding a retained
# institution, citer ≠ cited, cited year ≤ citer year + 1); an edge also needs
# both sources and an institution at each end to survive the edge list's
# singleton/SCC pruning.  R_i counts every such reference of the citer, edge or
# not, so with no shifts the replicate reproduces the baseline (checked before
# the replicates run).  Unit sets and a_p stay at the baseline, as in the other
# error studies.

YEAR_SWAP = {YEAR_LO - 1: YEAR_LO, YEAR_LO: YEAR_LO - 1, YEAR_HI: YEAR_HI + 1}


def _baseline_run() -> dict:
    bl = next(r for r in load_runs() if r['label'] == 'baseline')
    if (bl['tc0'], bl['tc1'], bl['tt0'], bl['tt1']) != (YEAR_LO, YEAR_HI, YEAR_LO, YEAR_HI):
        raise ValueError('year-swap model assumes census = target = '
                         f'[{YEAR_LO}, {YEAR_HI}]; baseline is {bl}')
    if set(bl['fx']) != set('EBAX'):
        raise ValueError(f"year-swap model assumes fx='EBAX', got {bl['fx']!r}")
    return bl


def build_year_swap_inputs(paths, omega: int = DEFAULT_OMEGA) -> None:
    """
    Stage 1 for the year-swap model (needs stage1_base_works).  Saves
      stage1_year_swap_pool.parquet  reference-table works missing from
                                     stage1_base_works (all those dated YEAR_LO-1,
                                     and window works with no baseline edge), with
                                     institution rows (same columns)
      stage1_year_swap_refs.parquet  citer_work_idx, cited_work_idx, edge (bool):
          every corpus reference between works dated YEAR_LO-1..YEAR_HI that
          both hold a retained institution (citer ≠ cited); edge = both sources,
          and an institution at each end, survive in the baseline edge list.
          Retained sets follow build_edge_lists for the baseline.
    """
    import duckdb as _duckdb
    bl = _baseline_run()
    pq = paths.parquet
    out_dir = paths.working / STAGE_DIR
    lo, hi = YEAR_LO - 1, YEAR_HI
    n_years = YEAR_HI - YEAR_LO + 1

    print(f'Year-swap: building {lo}–{hi} reference table and work pool ...', flush=True)
    # Edge-list units: the τ-retained sets after build_edge_lists' singleton/SCC
    # pruning, which deletes edge rows but leaves R_i as counted before it.
    run_code, fx = bl['run_code'], bl['fx']
    el_table = f"el_{run_code}_{fx}_tauU{bl['tau_u']}_tauS{bl['tau_s']}_vartau"
    db = _duckdb.connect(str(paths.working / 'edge_lists.duckdb'), read_only=True)
    el_src = db.execute(f'SELECT citer_source_idx AS idx FROM {el_table} UNION '
                        f'SELECT cited_source_idx FROM {el_table}').fetchdf()
    el_inst = db.execute(f'SELECT citer_inst_idx AS idx FROM {el_table} UNION '
                         f'SELECT cited_inst_idx FROM {el_table}').fetchdf()
    db.close()

    con = _duckdb.connect(':memory:')
    con.register('el_src', el_src)
    con.register('el_inst', el_inst)
    refs = con.execute(f"""
        WITH fw AS (
            SELECT w.work_idx, w.source_idx, w.publication_year AS py
            FROM '{pq}/corpus_works.parquet' w
            JOIN '{pq}/source_master.parquet' sm ON w.source_idx = sm.source_idx
            WHERE w.publication_year BETWEEN {lo} AND {hi}
        ),
        census AS (SELECT * FROM fw WHERE py BETWEEN {YEAR_LO} AND {YEAR_HI}),
        auths AS (
            SELECT DISTINCT work_idx, institution_idx
            FROM '{pq}/corpus_authorships.parquet'
            WHERE institution_idx IS NOT NULL
              AND work_idx IN (SELECT work_idx FROM fw)
        ),
        retained_inst AS (
            SELECT institution_idx FROM auths
            WHERE work_idx IN (SELECT work_idx FROM census)
              AND institution_idx IN (
                  SELECT institution_idx FROM '{pq}/corpus_institutions.parquet')
            GROUP BY institution_idx
            HAVING COUNT(DISTINCT work_idx) / {n_years}.0 >= {bl['tau_u']}
        ),
        retained_source AS (
            SELECT source_idx FROM census GROUP BY source_idx
            HAVING COUNT(DISTINCT work_idx) / {n_years}.0 >= {bl['tau_s']}
        ),
        rw AS (
            SELECT f.work_idx,
                   ANY_VALUE(f.source_idx IN (SELECT source_idx FROM retained_source)
                             AND f.source_idx IN (SELECT idx FROM el_src)) AS src_ok,
                   BOOL_OR(a.institution_idx IN (SELECT idx FROM el_inst))  AS inst_ok
            FROM auths a JOIN fw f ON a.work_idx = f.work_idx
            WHERE a.institution_idx IN (SELECT institution_idx FROM retained_inst)
            GROUP BY f.work_idx
        )
        SELECT r.citer_idx AS citer_work_idx, r.cited_idx AS cited_work_idx,
               (c.src_ok AND d.src_ok AND c.inst_ok AND d.inst_ok) AS edge
        FROM '{pq}/corpus_references.parquet' r
        JOIN rw c ON r.citer_idx = c.work_idx
        JOIN rw d ON r.cited_idx = d.work_idx
        WHERE r.citer_idx != r.cited_idx
        ORDER BY 1, 2
    """).fetchdf()
    con.close()
    refs['citer_work_idx'] = refs['citer_work_idx'].astype(np.int64)
    refs['cited_work_idx'] = refs['cited_work_idx'].astype(np.int64)
    refs['edge'] = refs['edge'].astype(bool)
    refs.to_parquet(str(out_dir / 'stage1_year_swap_refs.parquet'), index=False)
    print(f'  References: {len(refs):,}  (edges {int(refs["edge"].sum()):,})', flush=True)

    # Pool: every work in the reference table that stage1_base_works lacks — the
    # 2019 works, and window works with no baseline edge (a shift can give them one).
    base_ids = pd.read_parquet(str(out_dir / 'stage1_base_works.parquet'),
                               columns=['work_idx'])['work_idx'].unique()
    ref_ids = np.unique(np.concatenate([refs['citer_work_idx'].to_numpy(),
                                        refs['cited_work_idx'].to_numpy()]))
    works = pd.read_parquet(str(pq / 'corpus_works.parquet'),
                            columns=['work_idx', 'publication_year', 'source_idx'])
    works = works[works['work_idx'].isin(np.setdiff1d(ref_ids, base_ids))].rename(
        columns={'publication_year': 'pub_year'})
    works['pub_year']   = works['pub_year'].astype(np.int16)
    works['source_idx'] = works['source_idx'].astype(np.int64)
    pool = _attach_institutions(works, paths, omega)
    pool.to_parquet(str(out_dir / 'stage1_year_swap_pool.parquet'), index=False)
    _write_stage_meta(paths, 'stage1_year_swap', omega)
    n_lo = works['pub_year'].eq(lo).sum()
    print(f'  Pool: {len(works):,} works ({n_lo:,} dated {lo}, '
          f'{len(works) - n_lo:,} window works without a baseline edge)', flush=True)


def _draw_year_swaps(cand_ids: np.ndarray, cand_year: np.ndarray,
                     rng: np.random.Generator, p: float = P_ERROR_YEAR):
    """Select each boundary-year work with probability p; return (work_idx, new year)."""
    sel = rng.random(len(cand_ids)) < p
    new = np.array([YEAR_SWAP[int(y)] for y in cand_year[sel]], dtype=np.int16)
    return cand_ids[sel], new


def _year_swap_keep(ci: np.ndarray, cd: np.ndarray, year: np.ndarray) -> np.ndarray:
    """
    References the pipeline would count, given recorded years: both ends in the
    window and the cited work dated no later than the citer's year + 1.
    ci, cd index into year (positions of citer and cited works).
    """
    yc, yd = year[ci], year[cd]
    return ((yc >= YEAR_LO) & (yc <= YEAR_HI) & (yd >= YEAR_LO) & (yd <= YEAR_HI)
            & (yd <= yc + 1))


def _year_swap_refs(W: np.ndarray, ci: np.ndarray, cd: np.ndarray, edge: np.ndarray,
                    year: np.ndarray):
    """
    Edge references and the non-edge part of R_i for one set of recorded years.

    Returns (refs_arr (N, 3) int64 [citer, cited, 0], r_extra Series indexed by
    citer work_idx) in the form _build_matrices takes.
    """
    keep = _year_swap_keep(ci, cd, year)
    ke = keep & edge
    n_all  = np.bincount(ci[keep], minlength=len(W))
    n_edge = np.bincount(ci[ke],   minlength=len(W))
    extra = n_all - n_edge
    pos = np.nonzero(extra > 0)[0]
    r_extra = pd.Series(extra[pos].astype(np.int64), index=W[pos])
    refs_arr = np.column_stack([W[ci[ke]], W[cd[ke]], np.zeros(int(ke.sum()), np.int64)])
    return refs_arr, r_extra


def _swap_works_yr(base_yr: pd.DataFrame, pool: pd.DataFrame,
                   swap_ids: np.ndarray, new_year: np.ndarray) -> pd.DataFrame:
    """Window works after the shifts: shifted-out works dropped, shifted-in added."""
    in_ok  = (new_year >= YEAR_LO) & (new_year <= YEAR_HI)
    out_ids = swap_ids[~in_ok]
    ins_ids = swap_ids[in_ok]
    works = base_yr[~base_yr['work_idx'].isin(out_ids)]
    if len(ins_ids):
        ins = pool[pool['work_idx'].isin(ins_ids)].copy()
        ins['pub_year'] = np.int16(YEAR_LO)
        works = pd.concat([works, ins], ignore_index=True)
    return works


def build_year_swap_replicates(paths, B: int = 1000, seed: int = 42) -> None:
    """Stage 2: B sets of boundary shifts over all corpus works dated 2019, 2020
    or 2024 → stage2_year_swap.parquet (replicate_id, work_idx, pub_year, new_year)."""
    out_dir = paths.working / STAGE_DIR
    cand = pd.read_parquet(str(paths.parquet / 'corpus_works.parquet'),
                           columns=['work_idx', 'publication_year'])
    cand = cand[cand['publication_year'].isin(list(YEAR_SWAP))].sort_values('work_idx')
    cand_ids  = cand['work_idx'].to_numpy(np.int64)
    cand_year = cand['publication_year'].to_numpy(np.int16)
    print(f'Year-swap: {B} replicates over {len(cand_ids):,} boundary-year works '
          f'({", ".join(f"{y}:{(cand_year == y).sum():,}" for y in YEAR_SWAP)}), '
          f'rate {P_ERROR_YEAR:.1%}', flush=True)
    parts = []
    for b in range(B):
        rng = np.random.default_rng(seed + 20_000 + b)
        ids, new = _draw_year_swaps(cand_ids, cand_year, rng)
        old = cand_year[np.searchsorted(cand_ids, ids)]
        parts.append(pd.DataFrame({'replicate_id': np.int32(b), 'work_idx': ids,
                                   'pub_year': old, 'new_year': new}))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(str(out_dir / 'stage2_year_swap.parquet'), index=False)
    print(f'  Saved stage2_year_swap.parquet ({len(df):,} rows, '
          f'{len(df) / B:,.0f} per replicate)', flush=True)


def build_year_swap_rankings(paths, B: int = 1000, seed: int = 42) -> None:
    """Stage 4 for the year-swap model → stage4_year_swap/ (same layout as the others)."""
    import duckdb

    out_dir    = paths.working / STAGE_DIR
    stage4_dir = out_dir / 'stage4_year_swap'
    stage4_dir.mkdir(parents=True, exist_ok=True)

    bl = _baseline_run()
    run_code, tau_u, tau_s, fx = bl['run_code'], bl['tau_u'], bl['tau_s'], bl['fx']
    el_table    = f'el_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_vartau'
    units_table = f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_vartau_m0110'
    db = duckdb.connect(str(paths.working / 'edge_lists.duckdb'), read_only=True)
    units_df = db.execute(
        f'SELECT unit_idx, unit_type, a_p FROM "{units_table}" ORDER BY unit_type, unit_idx'
    ).fetchdf()
    r_bar = db.execute(
        f'SELECT AVG(rval) FROM '
        f'(SELECT DISTINCT citer_work_idx, CAST(R_i AS DOUBLE) AS rval FROM {el_table})'
    ).fetchone()[0]
    db.close()

    src_df  = units_df[units_df['unit_type'] == 'S'].reset_index(drop=True)
    inst_df = units_df[units_df['unit_type'] == 'U'].reset_index(drop=True)
    source_ids = src_df['unit_idx'].to_numpy(dtype=np.int64)
    inst_ids   = inst_df['unit_idx'].to_numpy(dtype=np.int64)
    a_s = src_df['a_p'].to_numpy(dtype=np.float64)
    a_u = inst_df['a_p'].to_numpy(dtype=np.float64)
    n_s, n_u = len(source_ids), len(inst_ids)
    A_full = float(a_s.sum() + a_u.sum())
    src_index, inst_index = pd.Index(source_ids), pd.Index(inst_ids)

    base_works = pd.read_parquet(str(out_dir / 'stage1_base_works.parquet'))
    pool = pd.read_parquet(str(out_dir / 'stage1_year_swap_pool.parquet'))
    base_yr = pd.concat([base_works, pool], ignore_index=True)
    base_yr = base_yr[base_yr['pub_year'].between(YEAR_LO, YEAR_HI)]
    refs = pd.read_parquet(str(out_dir / 'stage1_year_swap_refs.parquet'))
    swaps = pd.read_parquet(str(out_dir / 'stage2_year_swap.parquet'))
    swaps_by_rep = {int(r): g for r, g in swaps.groupby('replicate_id')}

    W = np.unique(np.concatenate([refs['citer_work_idx'].to_numpy(np.int64),
                                  refs['cited_work_idx'].to_numpy(np.int64)]))
    ci = np.searchsorted(W, refs['citer_work_idx'].to_numpy(np.int64))
    cd = np.searchsorted(W, refs['cited_work_idx'].to_numpy(np.int64))
    edge = refs['edge'].to_numpy(bool)
    yr_ser = (pd.concat([base_works[['work_idx', 'pub_year']], pool[['work_idx', 'pub_year']]])
                .drop_duplicates('work_idx').set_index('work_idx')['pub_year'])
    year0 = yr_ser.reindex(W).to_numpy()
    if np.isnan(year0.astype(float)).any():
        raise ValueError('reference works missing from stage1 base works / pool')
    year0 = year0.astype(np.int16)
    print(f'Units: n_s={n_s}, n_u={n_u}, r_bar={r_bar:.4f};  '
          f'references {len(refs):,} over {len(W):,} works', flush=True)

    # Zero-shift identity: the rebuilt references must be the baseline's.
    refs0, extra0 = _year_swap_refs(W, ci, cd, edge, year0)
    base_refs_df = pd.read_parquet(str(out_dir / 'stage3_base_refs.parquet'))
    bpairs = base_refs_df[['citer_work_idx', 'cited_work_idx']].to_numpy(np.int64)
    p0 = refs0[:, :2]
    same_edges = (len(p0) == len(bpairs) and
                  (p0[np.lexsort((p0[:, 1], p0[:, 0]))] ==
                   bpairs[np.lexsort((bpairs[:, 1], bpairs[:, 0]))]).all())
    bx = _r_extra(base_refs_df)
    # Citers with no edge carry no weight; the baseline lists extras for edge citers only.
    extra0_e = extra0[extra0.index.isin(np.unique(p0[:, 0]))]
    same_r = bx is not None and extra0_e.sort_index().equals(bx.astype(np.int64).sort_index())
    print(f'  zero-shift check: edges {len(p0):,} vs baseline {len(bpairs):,} '
          f'{"match" if same_edges else "DIFFER"};  R_i {"match" if same_r else "DIFFER"}',
          flush=True)
    if not (same_edges and same_r):
        raise RuntimeError('year-swap references do not reproduce the baseline edge list')

    v_s_boot = np.full((B, n_s), np.nan, dtype=np.float32)
    v_u_boot = np.full((B, n_u), np.nan, dtype=np.float32)
    lam_ratio_boot = np.full(B, np.nan, dtype=np.float32)
    n_in = np.zeros(B, np.int32); n_out = np.zeros(B, np.int32)
    d_edges = np.zeros(B, np.int64)
    rep_times: list[float] = []
    print(f'Stage 4 year-swap: running {B} replicates ...', flush=True)
    t0 = time.perf_counter()
    empty = pd.DataFrame({'work_idx': np.array([], np.int64),
                          'new_year': np.array([], np.int16)})

    for b in range(B):
        t_rep = time.perf_counter()
        sw = swaps_by_rep.get(b, empty)
        ids = sw['work_idx'].to_numpy(np.int64)
        new = sw['new_year'].to_numpy(np.int16)
        year = year0.copy()
        pos = np.searchsorted(W, ids)
        hit = (pos < len(W)) & (W[np.minimum(pos, len(W) - 1)] == ids)
        year[pos[hit]] = new[hit]
        refs_b, extra_b = _year_swap_refs(W, ci, cd, edge, year)
        works_yr = _swap_works_yr(base_yr, pool, ids, new)
        in_ok = (new >= YEAR_LO) & (new <= YEAR_HI)
        n_in[b], n_out[b] = int(in_ok.sum()), int((~in_ok).sum())
        d_edges[b] = len(refs_b) - len(p0)

        C_SI, C_IS = _build_matrices(refs_b, works_yr, src_index, inst_index,
                                     r_bar, n_s, n_u, r_extra=extra_b)
        if C_SI.nnz == 0 or C_IS.nnz == 0:
            continue
        v_s_b, v_u_b, lam1, lam2, n_sc, n_uc, _ = \
            _rank_one(C_SI, C_IS, n_s, n_u, a_s, a_u, A_full)
        v_s_boot[b], v_u_boot[b] = v_s_b, v_u_b
        lam_ratio_boot[b] = float(lam2 / lam1) if lam1 > 0 else np.nan

        elapsed = time.perf_counter() - t_rep
        rep_times.append(elapsed)
        if (b + 1) % 10 == 0 or b == 0:
            eta = (B - b - 1) * np.mean(rep_times[-10:])
            print(f'  rep {b+1:4d}/{B}  {elapsed:.2f}s  in={n_in[b]} out={n_out[b]} '
                  f'Δedges={d_edges[b]:+d}  core=({n_sc},{n_uc})  ETA {eta/60:.1f}min',
                  flush=True)

    np.save(str(stage4_dir / 'v_s_boot.npy'), v_s_boot)
    np.save(str(stage4_dir / 'v_u_boot.npy'), v_u_boot)
    np.save(str(stage4_dir / 'lam_ratio_boot.npy'), lam_ratio_boot)
    meta = dict(
        n=B, seed=seed, error_filter='year_swap', n_s=n_s, n_u=n_u,
        source_ids=source_ids.tolist(), inst_ids=inst_ids.tolist(),
        run_code=run_code, units_table=units_table,
        omega=_read_stage_meta(paths, 'stage1_year_swap')['omega'],
        rate=P_ERROR_YEAR, swaps={str(k): v for k, v in YEAR_SWAP.items()},
        mean_shifted_in=float(n_in.mean()), mean_shifted_out=float(n_out.mean()),
        mean_edge_change=float(d_edges.mean()),
        completed=B,
    )
    (stage4_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    total = time.perf_counter() - t0
    print(f'\nDone. {B} replicates  {total/60:.1f}min  '
          f'avg={np.mean(rep_times) if rep_times else 0:.2f}s/rep  '
          f'shifted in {n_in.mean():.0f}, out {n_out.mean():.0f}, '
          f'Δedges {d_edges.mean():+.0f} per replicate', flush=True)
    print(f'Output: {stage4_dir}', flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='4-stage metadata error bootstrap for the bipartite spectral ranking.'
    )
    parser.add_argument('--stage', required=True,
                        help="Comma list of stages from {1,2,3,4,year_v2,year_swap}")
    parser.add_argument('--filters', default=','.join(PAPER_FILTERS),
                        help=f'Stage-4 error filters, comma list from {ERROR_FILTERS} '
                             f'(default: {",".join(PAPER_FILTERS)})')
    parser.add_argument('--n',    type=int, default=1000, help='Replicates (stages 2-4)')
    parser.add_argument('--seed', type=int, default=42,  help='Base random seed')
    parser.add_argument('--omega', type=int, default=DEFAULT_OMEGA, choices=[0, 1],
                        help='Institution weighting for stage 1 / year-v2 pool: '
                             '0 = author-fractional (baseline), 1 = 1/N_inst')
    args = parser.parse_args()

    import duckdb

    paths = load_config()
    B     = args.n
    seed  = args.seed
    run_stages = {x.strip() for x in args.stage.split(',') if x.strip()}
    bad = run_stages - {'1', '2', '3', '4', 'year_v2', 'year_swap'}
    if bad:
        parser.error(f'unknown stage(s): {sorted(bad)}')
    filters = [f.strip() for f in args.filters.split(',') if f.strip()]
    bad = set(filters) - set(ERROR_FILTERS)
    if bad:
        parser.error(f'unknown filter(s): {sorted(bad)}')

    bl          = next(r for r in load_runs() if r['label'] == 'baseline')
    el_table    = (f"el_{bl['run_code']}_{bl['fx']}"
                   f"_tauU{bl['tau_u']}_tauS{bl['tau_s']}_vartau")

    out_dir = paths.working / STAGE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    base_works = None
    if '1' in run_stages:
        base_works = build_base_works(paths, omega=args.omega)

    if '2' in run_stages:
        if base_works is None:
            base_works = pd.read_parquet(str(out_dir / 'stage1_base_works.parquet'))
        build_work_replicates(base_works, paths, B=B, seed=seed)

    if '3' in run_stages:
        if base_works is None:
            base_works = pd.read_parquet(str(out_dir / 'stage1_base_works.parquet'))
        db = duckdb.connect(str(paths.working / 'edge_lists.duckdb'), read_only=True)
        build_ref_replicates(db, el_table, base_works, paths, B=B, seed=seed)
        db.close()

    if '4' in run_stages:
        for ef in filters:
            print(f'\n── Stage 4: error_filter={ef!r} ──', flush=True)
            build_rankings(paths, ef, B=B, seed=seed)

    if 'year_v2' in run_stages:
        pool = build_year_v2_pool(paths, omega=args.omega)
        build_year_v2_replicates(pool, paths, B=B, seed=seed)
        build_year_v2_rankings(paths, B=B, seed=seed)

    if 'year_swap' in run_stages:
        build_year_swap_inputs(paths, omega=args.omega)
        build_year_swap_replicates(paths, B=B, seed=seed)
        build_year_swap_rankings(paths, B=B, seed=seed)


if __name__ == '__main__':
    main()
