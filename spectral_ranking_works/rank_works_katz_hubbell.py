"""
rank_works_katz_hubbell.py — Katz-Hubbell ranking of individual works from the baseline
citation graph.

Uses the same work-work directed citation matrix H_ww as rank_works.py, but applies
Katz-Hubbell with damping factor α=0.85 and a uniform prior instead of eigenvector
decomposition.  Katz-Hubbell is well-defined even when H_ww is reducible or has
periodic SCCs (which make the raw spectral approach ill-posed).

Google matrix:  G = α(H + d·1ᵀ/N) + (1−α)·1·1ᵀ/N
  where d[i]=1 if work i is dangling (zero out-degree), H is row-stochastic.
Power iteration converges at rate α per step; tolerance = 1e-8 in L1 norm.

Source and institution rankings derived from the Katz-Hubbell vector by three schemes:
  (a) median:              robust to special-issue / single-paper outliers (preferred)
  (b) uniform mean:        v_s = mean(v_kh_w  for w in source/institution s)
  (c) out-degree weighted: v_s = sum(out_w * v_kh_w) / sum(out_w)

The KH distribution has a heavy right tail (min≈0.84, median=1, max≈3400×median),
so median is the most informative single aggregation.

Output ($WORKING/work_rankings/):
  work_rankings_katz_hubbell.parquet        — work_idx, v_kh, out_weight
  source_rankings_katz_hubbell.parquet      — source_idx, n_works,
                                               v_median, v_uniform, v_degree
  institution_rankings_katz_hubbell.parquet — institution_idx, n_works,
                                               v_median, v_uniform, v_degree
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs
from spectral_ranking_bootstrap.bootstrap_baseline import compute_r_bar, create_tmp_el
from spectral_ranking_works.rank_works import (
    _find_el_table, _load_intersection_edges, _build_matrix, _aggregate_sources,
)
import duckdb


def katz_hubbell(H: sp.csr_matrix, alpha: float = 0.85,
                 tol: float = 1e-8, maxiter: int = 500) -> tuple[np.ndarray, int, float]:
    """
    Katz-Hubbell power iteration on a row-stochastic matrix H (dangling rows allowed).

    The Google matrix is: G = α(H + d·1ᵀ/N) + (1−α)·1·1ᵀ/N
    where d[i]=1 for zero-out-degree (dangling) rows.  Gᵀ is column-stochastic,
    so the Katz-Hubbell vector v = Gᵀv is the stationary distribution.

    Returns
    -------
    v        : (N,) float64  Katz-Hubbell vector (sums to 1, all positive)
    n_iter   : int           iterations to convergence
    residual : float         final L1 residual
    """
    N = H.shape[0]
    Ht = H.T.tocsr()

    dangling = (np.array(H.sum(axis=1)).ravel() == 0).astype(np.float64)

    v = np.full(N, 1.0 / N, dtype=np.float64)

    for it in range(1, maxiter + 1):
        dangling_contrib = alpha * float(dangling @ v) / N
        v_new = alpha * Ht.dot(v) + dangling_contrib + (1.0 - alpha) / N
        v_new /= v_new.sum()          # renormalise for numerical safety
        residual = float(np.abs(v_new - v).sum())
        v = v_new
        if residual < tol:
            return v, it, residual

    return v, maxiter, residual


def main():
    alpha = 0.85
    tol   = 1e-8

    paths   = load_config()
    out_dir = paths.working / 'work_rankings'
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = next(r for r in load_runs() if r['label'] == 'baseline')
    run_code = baseline['run_code']
    tau_u    = int(baseline['tau_u'])
    tau_s    = int(baseline['tau_s'])

    el_db_path = paths.working / 'edge_lists.duckdb'

    db = duckdb.connect(str(el_db_path), read_only=True)
    el_table = _find_el_table(db, run_code, tau_u, tau_s)
    print(f'Edge list: {el_table}', flush=True)
    r_bar = compute_r_bar(db, el_table)
    print(f'r_bar = {r_bar:.6f}', flush=True)
    db.close()

    db = duckdb.connect(str(el_db_path), read_only=False)
    create_tmp_el(db, el_table, r_bar)

    print('Loading intersection edges ...', flush=True)
    df_edges, df_src = _load_intersection_edges(db)
    db.execute('DROP TABLE IF EXISTS _tmp_el')
    db.close()

    n_citer = df_edges['citer_work_idx'].nunique()
    n_cited = df_edges['cited_work_idx'].nunique()
    print(f'  Edges: {len(df_edges):,}   '
          f'citer works: {n_citer:,}   cited works: {n_cited:,}')

    print('Building H_ww ...', flush=True)
    H, work_ids, out_weight, n_zero = _build_matrix(df_edges)
    N = len(work_ids)
    print(f'  N={N:,}   nnz={H.nnz:,}   density={H.nnz/N**2:.2e}   '
          f'dangling={n_zero:,} ({100*n_zero/N:.1f}%)')

    print(f'\nRunning Katz-Hubbell (α={alpha}, tol={tol}) ...', flush=True)
    t0 = time.perf_counter()
    v_kh, n_iter, residual = katz_hubbell(H, alpha=alpha, tol=tol)
    elapsed = time.perf_counter() - t0
    print(f'  Converged in {n_iter} iterations, {elapsed:.1f}s   '
          f'L1 residual = {residual:.2e}')

    # Normalise to median = 1
    pos = v_kh[v_kh > 0]
    med = float(np.median(pos)) if len(pos) > 0 else 1.0
    v_kh_norm = v_kh / med

    df_works = pd.DataFrame({
        'work_idx':   work_ids,
        'v_kh':       v_kh_norm.astype(np.float32),
        'out_weight': out_weight.astype(np.float32),
    })
    df_works.to_parquet(out_dir / 'work_rankings_katz_hubbell.parquet', index=False)
    print(f'\nSaved work_rankings_katz_hubbell.parquet  ({N:,} works)')
    print(f'  v_kh:  min={df_works.v_kh.min():.4f}  '
          f'p25={df_works.v_kh.quantile(0.25):.4f}  '
          f'median={df_works.v_kh.median():.4f}  '
          f'p75={df_works.v_kh.quantile(0.75):.4f}  '
          f'p95={df_works.v_kh.quantile(0.95):.4f}  '
          f'max={df_works.v_kh.max():.4f}')

    # ── Aggregate to sources ──────────────────────────────────────────────────
    print('\nAggregating to sources ...', flush=True)
    df_agg = _aggregate_sources(work_ids, v_kh_norm, out_weight, df_src)

    # Pull works→source mapping for median and degree-weighted aggregation
    pq_db = duckdb.connect(':memory:')
    df_work_src = pq_db.execute(f"""
        SELECT DISTINCT work_idx, source_idx
        FROM '{paths.working / "parquet/corpus_works.parquet"}'
    """).fetchdf()
    pq_db.close()

    df_kh_src = pd.DataFrame({'work_idx': work_ids, 'v_kh': v_kh_norm,
                               'out_weight': out_weight})
    df_src_merged = df_work_src.merge(df_kh_src, on='work_idx', how='inner')
    df_src_merged['out_weight'] = df_src_merged['out_weight'].clip(lower=1e-9)

    def _wtd_mean(grp, df_ref):
        w = df_ref.loc[grp.index, 'out_weight'].values
        return np.average(grp.values, weights=w) if w.sum() > 0 else np.nan

    src_agg = df_src_merged.groupby('source_idx')['v_kh'].agg(
        n_works='count', v_median='median', v_uniform='mean'
    ).reset_index()
    src_agg.columns = ['source_idx', 'n_works', 'v_median', 'v_uniform']
    v_deg_src = (df_src_merged.groupby('source_idx')['v_kh']
                              .apply(lambda g: _wtd_mean(g, df_src_merged))
                              .reset_index(name='v_degree'))
    src_agg = src_agg.merge(v_deg_src, on='source_idx', how='left')
    src_agg.to_parquet(out_dir / 'source_rankings_katz_hubbell.parquet', index=False)
    print(f'Saved source_rankings_katz_hubbell.parquet  ({len(src_agg):,} sources)')

    # ── Aggregate to institutions ─────────────────────────────────────────────
    print('\nAggregating to institutions ...', flush=True)
    inst_path = paths.working / 'parquet/corpus_institutions.parquet'
    auth_path = paths.working / 'parquet/corpus_authorships.parquet'

    db2 = duckdb.connect(':memory:')
    df_work_inst = db2.execute(f"""
        SELECT DISTINCT a.work_idx, a.institution_idx
        FROM '{auth_path}' a
        INNER JOIN '{inst_path}' i USING (institution_idx)
        WHERE a.institution_idx IS NOT NULL
    """).fetchdf()
    db2.close()

    df_inst_merged = df_work_inst.merge(df_kh_src, on='work_idx', how='inner')
    df_inst_merged['out_weight'] = df_inst_merged['out_weight'].clip(lower=1e-9)

    inst_agg = df_inst_merged.groupby('institution_idx')['v_kh'].agg(
        n_works='count', v_median='median', v_uniform='mean'
    ).reset_index()
    inst_agg.columns = ['institution_idx', 'n_works', 'v_median', 'v_uniform']
    v_deg_inst = (df_inst_merged.groupby('institution_idx')['v_kh']
                                .apply(lambda g: _wtd_mean(g, df_inst_merged))
                                .reset_index(name='v_degree'))
    inst_agg = inst_agg.merge(v_deg_inst, on='institution_idx', how='left')
    inst_agg.to_parquet(out_dir / 'institution_rankings_katz_hubbell.parquet', index=False)
    print(f'Saved institution_rankings_katz_hubbell.parquet  ({len(inst_agg):,} institutions)')

    # ── Print source comparison table ─────────────────────────────────────────
    rk_path = paths.working / 'rankings.duckdb'
    rk_db2  = duckdb.connect(str(rk_path), read_only=True)
    rk_tname = (f"rk_{run_code}_EBAX_tauU{tau_u}_tauS{tau_s}"
                f"_vartau_rho0_m0110_chi50_alpha100")
    df_spec_src = rk_db2.execute(f"""
        SELECT unit_idx AS source_idx, v AS v_spec, rank_v AS spectral_rank
        FROM {rk_tname}
        WHERE unit_type = 'S' ORDER BY rank_v
    """).fetchdf()
    df_spec_inst = rk_db2.execute(f"""
        SELECT unit_idx AS institution_idx, v AS v_spec, rank_v AS spectral_rank
        FROM {rk_tname}
        WHERE unit_type = 'U' ORDER BY rank_v
    """).fetchdf()
    rk_db2.close()

    df_src_meta  = pd.read_parquet(paths.working / 'parquet/source_master.parquet')[
        ['source_idx', 'source_name', 'field_eb']]
    df_inst_meta = pd.read_parquet(inst_path)[
        ['institution_idx', 'institution_name', 'type', 'country_code', 'works_per_year']]

    for col in ['v_median', 'v_uniform', 'v_degree']:
        src_agg[f'rk_{col}']  = src_agg[col].rank(ascending=False, method='min',
                                                    na_option='keep')
        inst_agg[f'rk_{col}'] = inst_agg[col].rank(ascending=False, method='min',
                                                     na_option='keep')

    print('\n── Top-60 spectral sources vs Katz-Hubbell ──────────────────────────────')
    top_src = (df_spec_src.head(60)
                          .merge(df_src_meta, on='source_idx', how='left')
                          .merge(src_agg, on='source_idx', how='left'))
    print(f"{'Sp':>3}  {'src_idx':>12}  {'Source':<44}  {'F':>2}  {'nw':>5}  "
          f"{'v_spec':>7}  {'v_med':>6}  {'rk_med':>7}  {'v_mn':>6}  {'rk_mn':>6}")
    print('-' * 120)
    for _, row in top_src.iterrows():
        def _rk(c): return int(row[c]) if pd.notna(row[c]) else -1
        def _vl(c): return row[c]      if pd.notna(row[c]) else float('nan')
        print(f"{int(row['spectral_rank']):>3}  {int(row['source_idx']):>12}  "
              f"{str(row['source_name'])[:43]:<44}  {str(row['field_eb']):>2}  "
              f"{int(row['n_works']) if pd.notna(row['n_works']) else -1:>5}  "
              f"{_vl('v_spec'):>7.3f}  {_vl('v_median'):>6.3f}  {_rk('rk_v_median'):>7}  "
              f"{_vl('v_uniform'):>6.3f}  {_rk('rk_v_uniform'):>6}")

    print('\n── Top-50 spectral institutions vs Katz-Hubbell ─────────────────────────')
    top_inst = (df_spec_inst.head(50)
                            .merge(df_inst_meta, on='institution_idx', how='left')
                            .merge(inst_agg, on='institution_idx', how='left'))
    print(f"{'Sp':>3}  {'inst_idx':>12}  {'Institution':<44}  {'Type':<12}  {'CC':>3}  "
          f"{'nw':>5}  {'v_spec':>7}  {'v_med':>6}  {'rk_med':>7}  {'v_mn':>6}  {'rk_mn':>6}")
    print('-' * 135)
    for _, row in top_inst.iterrows():
        def _rk(c): return int(row[c]) if pd.notna(row[c]) else -1
        def _vl(c): return row[c]      if pd.notna(row[c]) else float('nan')
        print(f"{int(row['spectral_rank']):>3}  {int(row['institution_idx']):>12}  "
              f"{str(row['institution_name'])[:43]:<44}  {str(row['type']):<12}  "
              f"{str(row['country_code']):>3}  "
              f"{int(row['n_works']) if pd.notna(row['n_works']) else -1:>5}  "
              f"{_vl('v_spec'):>7.3f}  {_vl('v_median'):>6.3f}  {_rk('rk_v_median'):>7}  "
              f"{_vl('v_uniform'):>6.3f}  {_rk('rk_v_uniform'):>6}")

    print(f'\nOutput: {out_dir}')


if __name__ == '__main__':
    main()
