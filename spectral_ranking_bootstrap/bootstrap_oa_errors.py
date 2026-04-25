"""
bootstrap_oa_errors.py — 4-stage bootstrap for OpenAlex metadata errors.

Stage 1 — build_base_works
    Persist all OAS corpus works 2020-2024 with institution fractionation weights.
    Source: corpus_works.parquet + corpus_authorships.parquet.

Stage 2 — build_work_replicates
    B perturbed work tables: disjoint ~10% pub_year / ~10% source / ~10% institution
    errors per replicate.  Only changed rows persisted (delta format).

Stage 3 — build_ref_replicates
    B perturbed reference tables: ~10% wrong references (5% same-source,
    5% cross-source replacement).  Only changed rows persisted.

Stage 4 — build_rankings
    B bipartite (m=0110) spectral rankings from random pairings of Stage 2 × Stage 3.
    Uses the fixed baseline unit set; no tau recomputation.

Error rates (empirically estimated):
    pub_year    — 2%   ±1 year, boundary-clipped: 2020→2021, 2024→2023 only
    source      — 0.03% uniform draw from OAS source pool
    institution — 3%   75% within-country, 25% global uniform (Tübingen study: ~100/2800)
    references  — 5%   2.5% same-source cited-work, 2.5% cross-source cited-work

Storage: $WORKING/bootstrap_oa_errors/
    stage1_base_works.parquet
    stage2_work_errors.parquet    (replicate_id, work_idx, pub_year, source_idx,
                                   inst_idx, inst_weight, country_code, inst_idx_old)
    stage3_base_refs.parquet      (ref_idx, citer_work_idx, cited_work_idx, cited_source_idx)
    stage3_ref_errors.parquet     (replicate_id, ref_idx, new_cited_work_idx)
    stage4/v_s_boot.npy                (B, n_s) float32  — all errors combined
    stage4/v_u_boot.npy                (B, n_u) float32
    stage4/lam_ratio_boot.npy          (B,) float32
    stage4/meta.json                   includes error_filter='all'
    stage4_{type}/...                  same layout for --error-type year|source|institution|reference

CLI
---
python spectral_ranking_bootstrap/bootstrap_oa_errors.py
    [--stage {1,2,3,4,all}]
    [--n 1000]  [--seed 42]  [--tol 1e-7]
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
from spectral_ranking_bootstrap.bootstrap_baseline import _bipartite_core

YEAR_LO   = 2020
YEAR_HI   = 2024
# Per-type OA error rates (empirically estimated)
P_ERROR_YEAR = 0.020   # 2%   publication-year errors
P_ERROR_SRC  = 0.0003  # 0.03% work-to-journal misassignment
P_ERROR_INST = 0.030   # 3%   institution errors (Tübingen study ~100/2800)
P_ERROR_REF  = 0.050   # 5%   reference errors (total; split equally same/cross)
P_WITHIN     = 0.75    # fraction of institution errors within same country
STAGE_DIR = 'bootstrap_oa_errors'


# ── Stage 1 ────────────────────────────────────────────────────────────────────

def build_base_works(paths) -> pd.DataFrame:
    """
    Load baseline edge-list works for YEAR_LO..YEAR_HI with institution fractionation.

    Restricted to works that appear as citer or cited in the baseline edge list,
    so base_works exactly matches the baseline corpus (same set as build_edge_lists).

    Returns and saves a DataFrame with columns:
        work_idx     int64
        pub_year     int16
        source_idx   int64
        inst_idx     Int64   (nullable; pd.NA for works with no institution)
        inst_weight  float32 (1/N_i; 1.0 if no institution)
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

    print('Stage 1: loading corpus_authorships ...', flush=True)
    auth = pd.read_parquet(
        str(paths.parquet / 'corpus_authorships.parquet'),
        columns=['work_idx', 'institution_idx', 'country_code'],
    )
    auth = auth[auth['work_idx'].isin(works['work_idx'])]
    auth = auth.drop_duplicates(['work_idx', 'institution_idx'])
    auth = auth.dropna(subset=['institution_idx'])

    n_inst = auth.groupby('work_idx').size().rename('n_inst')
    auth   = auth.join(n_inst, on='work_idx')
    auth['inst_weight'] = (1.0 / auth['n_inst']).astype(np.float32)
    auth = auth.drop(columns='n_inst').rename(columns={'institution_idx': 'inst_idx'})
    auth['inst_idx'] = auth['inst_idx'].astype(np.int64)

    base = works.merge(
        auth[['work_idx', 'inst_idx', 'inst_weight', 'country_code']],
        on='work_idx', how='left',
    )
    no_inst = base['inst_idx'].isna()
    base.loc[no_inst, 'inst_weight'] = np.float32(1.0)
    base['inst_idx'] = base['inst_idx'].astype('Int64')

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


def perturb_works_one(base: pd.DataFrame,
                      source_pool: np.ndarray,
                      inst_pool: pd.DataFrame,
                      rng: np.random.Generator,
                      _inst_pools: tuple | None = None) -> pd.DataFrame:
    """
    Apply one round of disjoint errors to base_works — fully vectorised.

    Returns only the changed rows (delta), with an extra column:
        inst_idx_old  Int64   — original inst_idx for I-type errors; pd.NA otherwise

    For Y- and S-type errors, ALL (work, inst) rows for that work are returned
    (since pub_year and source_idx are work-level properties shared by all rows).
    For I-type errors, ALL rows for affected works are returned so that
    apply_work_delta can replace them cleanly.

    _inst_pools: pre-built result of _build_inst_pools(inst_pool).  Pass from the
    outer B-loop to avoid rebuilding on every call.
    """
    work_ids = base['work_idx'].unique()
    n_works  = len(work_ids)
    u        = rng.random(n_works)
    _cut_src  = P_ERROR_YEAR + P_ERROR_SRC
    _cut_inst = _cut_src + P_ERROR_INST
    py_mask  = u < P_ERROR_YEAR
    src_mask = (u >= P_ERROR_YEAR) & (u < _cut_src)
    ins_mask = (u >= _cut_src)     & (u < _cut_inst)

    parts: list[pd.DataFrame] = []

    # ── pub_year — fully vectorised ───────────────────────────────────────────
    if py_mask.any():
        py_works  = work_ids[py_mask]
        py_subset = base[base['work_idx'].isin(set(py_works.tolist()))].copy()
        # One old-year per unique work; vectorised ±1 with boundary clipping
        uniq  = py_subset.drop_duplicates('work_idx').set_index('work_idx')['pub_year']
        delta = rng.integers(0, 2, size=len(uniq), dtype=np.int16) * 2 - 1
        new_y = uniq.values.astype(np.int16) + delta
        new_y = np.where(new_y < YEAR_LO, np.int16(YEAR_LO + 1), new_y)
        new_y = np.where(new_y > YEAR_HI, np.int16(YEAR_HI - 1), new_y)
        py_map = dict(zip(uniq.index.tolist(), new_y.tolist()))
        py_subset['pub_year'] = py_subset['work_idx'].map(py_map).astype(np.int16)
        py_subset['inst_idx_old'] = pd.array([pd.NA] * len(py_subset), dtype='Int64')
        py_subset['error_type'] = 'year'
        parts.append(py_subset)

    # ── source — vectorised ───────────────────────────────────────────────────
    if src_mask.any():
        src_works  = work_ids[src_mask]
        new_srcs   = rng.choice(source_pool, size=len(src_works), replace=True)
        src_map    = dict(zip(src_works.tolist(), new_srcs.tolist()))
        src_subset = base[base['work_idx'].isin(set(src_works.tolist()))].copy()
        src_subset['source_idx'] = src_subset['work_idx'].map(src_map).astype(np.int64)
        src_subset['inst_idx_old'] = pd.array([pd.NA] * len(src_subset), dtype='Int64')
        src_subset['error_type'] = 'source'
        parts.append(src_subset)

    # ── institution — vectorised, no per-work Python loop ─────────────────────
    if ins_mask.any():
        if _inst_pools is None:
            _inst_pools = _build_inst_pools(inst_pool)
        cc_to_insts, all_inst_arr, all_cc_arr = _inst_pools

        ins_works  = work_ids[ins_mask]
        ins_subset = base[base['work_idx'].isin(set(ins_works.tolist()))].copy()
        ins_subset  = ins_subset.reset_index(drop=True)

        valid_rows = ins_subset[ins_subset['inst_idx'].notna()].copy()
        if len(valid_rows) > 0:
            # Random row selection per work via sort-key trick
            valid_rows['_rk'] = rng.random(len(valid_rows))
            pick_pos = valid_rows.groupby('work_idx', sort=False)['_rk'].idxmin()
            picked   = valid_rows.loc[pick_pos].copy()
            n_p      = len(picked)

            countries   = picked['country_code'].to_numpy(dtype=object)
            within_flag = rng.random(n_p) < P_WITHIN
            new_inst_ids = np.empty(n_p, dtype=np.int64)
            new_inst_ccs = np.empty(n_p, dtype=object)

            # Cross-country draws: one vectorised call
            cross = ~within_flag
            if cross.any():
                ci = rng.integers(0, len(all_inst_arr), size=int(cross.sum()))
                new_inst_ids[cross] = all_inst_arr[ci]
                new_inst_ccs[cross] = all_cc_arr[ci]

            # Within-country draws: one vectorised call per unique country (~100 iters)
            for cc, (pool_insts, pool_ccs) in cc_to_insts.items():
                mask = within_flag & (countries == cc)
                if mask.any():
                    ci = rng.integers(0, len(pool_insts), size=int(mask.sum()))
                    new_inst_ids[mask] = pool_insts[ci]
                    new_inst_ccs[mask] = cc

            # Fallback: within_flag=True but country not in pool → cross draw
            no_pool = within_flag & ~np.isin(countries, list(cc_to_insts.keys()))
            if no_pool.any():
                ci = rng.integers(0, len(all_inst_arr), size=int(no_pool.sum()))
                new_inst_ids[no_pool] = all_inst_arr[ci]
                new_inst_ccs[no_pool] = all_cc_arr[ci]

            # Build lookup dicts: work_idx → old/new inst
            old_by_work = dict(zip(
                picked['work_idx'].tolist(),
                picked['inst_idx'].astype(np.int64).tolist(),
            ))
            new_id_by_work = dict(zip(picked['work_idx'].tolist(), new_inst_ids.tolist()))
            new_cc_by_work = dict(zip(picked['work_idx'].tolist(), new_inst_ccs.tolist()))

            # Apply: in ins_subset rows for picked works, swap the target institution
            ins_delta = ins_subset[
                ins_subset['work_idx'].isin(set(picked['work_idx'].tolist()))
            ].copy()

            old_arr     = ins_delta['work_idx'].map(old_by_work).to_numpy()
            cur_arr     = ins_delta['inst_idx'].to_numpy(dtype=object)
            is_target   = cur_arr == old_arr   # row-wise comparison

            new_id_arr  = ins_delta['work_idx'].map(new_id_by_work).to_numpy()
            new_cc_arr2 = ins_delta['work_idx'].map(new_cc_by_work).to_numpy(dtype=object)

            cur_int = np.where(is_target, new_id_arr, old_arr.astype(float)).astype(object)
            ins_delta['inst_idx']    = pd.array(
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
        delta = perturb_works_one(base, source_pool, inst_pool, rng,
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
    Return DISTINCT (citer_work_idx, cited_work_idx, cited_source_idx) from the
    edge list, with a sequential ref_idx column.
    """
    df = db.execute(f"""
        SELECT DISTINCT citer_work_idx, cited_work_idx, cited_source_idx
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
      stage3_base_refs.parquet   — (ref_idx, citer_work_idx, cited_work_idx, cited_source_idx)
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
    p_same    = 0.5 * P_ERROR_REF
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
                    r_bar: float, n_s: int, n_u: int
                    ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Build C_SI (n_s × n_u) and C_IS (n_u × n_s) from reference and work arrays.

    refs_arr: (N, 3) int64 — [citer_work_idx, cited_work_idx, cited_source_idx]
              (cited_source_idx column is ignored; we re-derive from works_yr)
    works_yr: DataFrame filtered to pub_year window, columns:
              work_idx, pub_year, source_idx, inst_idx, inst_weight

    Uses fixed baseline unit indices (src_index, inst_index).
    Rows with source/institution not in units are silently dropped.
    """
    # ── Filter references to works in the year window ─────────────────────────
    valid_set = set(works_yr['work_idx'].to_numpy().tolist())
    citer_ok  = np.isin(refs_arr[:, 0], list(valid_set))
    cited_ok  = np.isin(refs_arr[:, 1], list(valid_set))
    keep      = citer_ok & cited_ok
    refs      = refs_arr[keep]
    if len(refs) == 0:
        return (sp.csr_matrix((n_s, n_u)), sp.csr_matrix((n_u, n_s)))

    # ── Compute R_i and rho_w ──────────────────────────────────────────────────
    _, inv, counts = np.unique(refs[:, 0], return_inverse=True, return_counts=True)
    rho_w = r_bar / counts[inv]   # (N_keep,)

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


# ── Stage 4 ────────────────────────────────────────────────────────────────────

def build_rankings(paths, B: int = 32, seed: int = 42,
                   error_filter: str = 'all') -> None:
    """
    Stage 4: build B bipartite (m=0110) spectral rankings from random
    Stage 2 × Stage 3 pairings.

    error_filter : 'all' | 'year' | 'source' | 'institution' | 'reference'
        'all'         — apply all work deltas + all ref deltas (default)
        'year'        — apply only year-error work deltas; no ref deltas
        'source'      — apply only source-error work deltas; no ref deltas
        'institution' — apply only institution-error work deltas; no ref deltas
        'reference'   — no work deltas; apply all ref deltas
        Output goes to stage4/ for 'all', stage4_{filter}/ for specific types.
    """
    import duckdb

    out_dir    = paths.working / STAGE_DIR
    suffix     = '' if error_filter == 'all' else f'_{error_filter}'
    stage4_dir = out_dir / f'stage4{suffix}'
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

    work_errors = pd.read_parquet(str(out_dir / 'stage2_work_errors.parquet'))
    work_deltas: dict[int, pd.DataFrame] = {
        int(rid): grp for rid, grp in work_errors.groupby('replicate_id')
    }

    base_refs_df  = pd.read_parquet(str(out_dir / 'stage3_base_refs.parquet'))
    base_refs_arr = base_refs_df[['citer_work_idx', 'cited_work_idx',
                                   'cited_source_idx']].to_numpy(dtype=np.int64)

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
              f'{len(ref_deltas)} ref replicates', flush=True)
    else:  # 'all'
        ref_deltas = ref_deltas_all
        print(f'  error_filter=all: {len(work_deltas)} work replicates, '
              f'{len(ref_deltas)} ref replicates', flush=True)

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

        C_SI, C_IS = _build_matrices(refs_b, works_yr,
                                     src_index, inst_index, r_bar, n_s, n_u)

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
        b2_arr=b2_arr.tolist(), b3_arr=b3_arr.tolist(),
    )
    (stage4_dir / 'meta.json').write_text(json.dumps(meta, indent=2))

    total = time.perf_counter() - t0
    avg   = float(np.mean(rep_times)) if rep_times else 0.0
    print(f'\nDone. {B} replicates  {total/60:.1f}min  avg={avg:.2f}s/rep', flush=True)
    print(f'Output: {stage4_dir}', flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='4-stage metadata error bootstrap for the bipartite spectral ranking.'
    )
    parser.add_argument('--stage', default='all',
                        choices=['1', '2', '3', '4', 'all'],
                        help='Which stage(s) to run (default: all)')
    parser.add_argument('--n',    type=int, default=32,  help='Replicates (stages 2-4)')
    parser.add_argument('--seed', type=int, default=0,  help='Base random seed')
    args = parser.parse_args()

    import duckdb

    paths = load_config()
    B     = args.n
    seed  = args.seed
    run_stages = ({'1', '2', '3', '4'} if args.stage == 'all'
                  else {args.stage})

    bl          = next(r for r in load_runs() if r['label'] == 'baseline')
    el_table    = (f"el_{bl['run_code']}_{bl['fx']}"
                   f"_tauU{bl['tau_u']}_tauS{bl['tau_s']}_vartau")

    out_dir = paths.working / STAGE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    base_works = None
    if '1' in run_stages:
        base_works = build_base_works(paths)

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
        for ef in ('all', 'year', 'source', 'institution', 'reference'):
            print(f'\n── Stage 4: error_filter={ef!r} ──', flush=True)
            build_rankings(paths, B=B, seed=seed, error_filter=ef)


if __name__ == '__main__':
    main()
