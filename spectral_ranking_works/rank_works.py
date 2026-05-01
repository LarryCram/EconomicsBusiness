"""
rank_works.py — spectral ranking of individual works from the baseline citation graph.

Builds the work-work directed citation matrix H_ww restricted to works that appear
on both the citer and cited sides after tau-filtering (the intersection).  Computes
6 leading right eigenpairs of H_ww (hub scores: work w scores high when it cites
high-scoring works) and 6 leading left eigenpairs of H_ww (authority scores: work w
scores high when cited by high-scoring works).

Pairwise eigenvalue ratios lambda_k / lambda_1 are reported for both sides.

Source rankings are derived from the leading eigenvectors by aggregation in two ways:
  (a) uniform average:        v_s = mean(v_w  for w in source s)
  (b) out-degree weighted:    v_s = sum(out_w * v_w) / sum(out_w)
                              where out_w = sum of rho_w for edges leaving w

applied to both hub (right) and authority (left) vectors giving four source columns.

Output ($WORKING/work_rankings/):
  work_rankings.parquet     — work_idx, v_hub, v_auth, out_weight
  source_rankings.parquet   — source_idx, n_works,
                               v_hub_uniform, v_hub_degree,
                               v_auth_uniform, v_auth_degree
  eigenvalues.json          — all 6 right and left eigenvalues + pairwise ratios
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, ArpackNoConvergence

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs
from spectral_ranking_bootstrap.bootstrap_baseline import compute_r_bar, create_tmp_el
import duckdb


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_el_table(db, run_code: str, tau_u: int, tau_s: int) -> str:
    """Find baseline vartau el table; prefers ALL fx."""
    suffix = f'_tauU{tau_u}_tauS{tau_s}_vartau'
    prefix = f'el_{run_code}_'
    tables = [r[0] for r in db.execute('SHOW TABLES').fetchall()]
    candidates = [t for t in tables if t.startswith(prefix) and t.endswith(suffix)]
    if not candidates:
        raise FileNotFoundError(
            f'No vartau el table for run_code={run_code}, tau_u={tau_u}, tau_s={tau_s}. '
            f'Available: {[t for t in tables if t.startswith(prefix)]}'
        )
    for t in candidates:
        if '_EBAX_' in t:
            return t
    return candidates[0]


def _load_intersection_edges(db) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load (citer_work, cited_work, rho_w) restricted to works that are both
    citers and cited (the intersection).  Also returns the citer→source map.

    _tmp_el must already exist in db.
    """
    df_edges = db.execute("""
        SELECT DISTINCT
            citer_work_idx,
            cited_work_idx,
            MAX(rho_w) AS rho_w
        FROM _tmp_el
        WHERE cited_work_idx IN (SELECT DISTINCT citer_work_idx FROM _tmp_el)
        GROUP BY citer_work_idx, cited_work_idx
    """).fetchdf()

    df_src = db.execute("""
        SELECT DISTINCT citer_work_idx AS work_idx, citer_source_idx AS source_idx
        FROM _tmp_el
    """).fetchdf()

    return df_edges, df_src


def _build_matrix(df_edges: pd.DataFrame) -> tuple:
    """
    Build row-normalised work-work sparse matrix H_ww.

    Returns
    -------
    H          : (N, N) CSR row-stochastic (zero rows left as zero)
    work_ids   : (N,) int64 — original work_idx values
    out_weight : (N,) float64 — unnormalised row sum (sum of rho_w per citer)
    n_zero_rows: int — number of works with no outgoing edges in the intersection
    """
    work_ids = np.union1d(
        df_edges['citer_work_idx'].to_numpy(dtype=np.int64),
        df_edges['cited_work_idx'].to_numpy(dtype=np.int64),
    )
    N = len(work_ids)
    idx = pd.Index(work_ids)

    row  = idx.get_indexer(df_edges['citer_work_idx'].to_numpy()).astype(np.int32)
    col  = idx.get_indexer(df_edges['cited_work_idx'].to_numpy()).astype(np.int32)
    data = df_edges['rho_w'].to_numpy(dtype=np.float64)

    W = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()

    out_weight  = np.array(W.sum(axis=1)).ravel()
    n_zero_rows = int((out_weight == 0).sum())

    # Row-normalise; zero rows stay zero (dangling nodes)
    row_sums = out_weight.copy()
    row_sums[row_sums == 0] = 1.0
    H = sp.diags(1.0 / row_sums) @ W

    return H, work_ids, out_weight, n_zero_rows


def _compute_eigenpairs(H: sp.csr_matrix, k: int = 6) -> dict:
    """
    Compute k leading right and left eigenpairs of H.
    Returns a dict with keys 'right' and 'left', each containing
    {'vals': ndarray(k complex), 'vecs': ndarray(N, k complex)} or None on failure.
    Prints eigenvalues and all pairwise lambda_i / lambda_1 ratios.
    """
    results = {}
    for side, mat, label in [('right', H, 'H_ww'), ('left', H.T, 'H_ww^T')]:
        print(f'\n{label} eigenpairs (k={k}) ...', flush=True)
        t0 = time.perf_counter()
        try:
            vals, vecs = eigs(mat, k=k, which='LM', tol=1e-7, maxiter=10000)
            order = np.argsort(-np.abs(vals))
            vals  = vals[order]
            vecs  = vecs[:, order]
            elapsed = time.perf_counter() - t0
            print(f'  Converged in {elapsed:.1f}s')
            lam1 = abs(vals[0])
            print(f'  {"k":>3}  {"Re(λ)":>10}  {"Im(λ)":>10}  '
                  f'{"|λ|":>10}  {"|λ_k|/|λ_1|":>12}')
            for i, lam in enumerate(vals):
                ratio = abs(lam) / lam1 if lam1 > 0 else np.nan
                print(f'  {i+1:>3}  {lam.real:>10.6f}  {lam.imag:>10.6f}  '
                      f'{abs(lam):>10.6f}  {ratio:>12.6f}')
            results[side] = {'vals': vals, 'vecs': vecs}
        except ArpackNoConvergence as e:
            print(f'  ArpackNoConvergence: {e}')
            results[side] = None
        except Exception as e:
            print(f'  Failed: {e}')
            results[side] = None

    return results


def _leading_vector(ep: dict | None, side: str) -> np.ndarray | None:
    """Extract the eigenvector whose eigenvalue is closest to +1 on the real axis."""
    if ep is None:
        return None
    vals = ep['vals']
    # Prefer eigenvalue nearest +1 (Perron root for row-stochastic);
    # avoids picking a complex rotation when a periodic SCC dominates by modulus.
    best = int(np.argmin(np.abs(vals - 1.0)))
    v = np.real(ep['vecs'][:, best])
    if v.mean() < 0:
        v = -v
    return v


def _aggregate_sources(work_ids, v, out_weight, df_src) -> pd.DataFrame:
    """
    Aggregate work score v to source level with uniform and degree-weighted averages.
    Returns DataFrame with source_idx, n_works, v_uniform, v_degree.
    """
    idx = pd.Index(work_ids)
    df  = df_src.copy()
    df['dense']      = idx.get_indexer(df['work_idx'].to_numpy())
    df = df[df['dense'] >= 0].copy()
    df['v']          = v[df['dense'].to_numpy()]
    df['out_weight'] = out_weight[df['dense'].to_numpy()]

    rows = []
    for src, grp in df.groupby('source_idx'):
        ow = grp['out_weight'].to_numpy()
        vv = grp['v'].to_numpy()
        tot_ow = ow.sum()
        rows.append(dict(
            source_idx = int(src),
            n_works    = len(grp),
            v_uniform  = float(vv.mean()),
            v_degree   = float((ow * vv).sum() / tot_ow) if tot_ow > 0 else np.nan,
        ))
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    paths   = load_config()
    out_dir = paths.working / 'work_rankings'
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = next(r for r in load_runs() if r['label'] == 'baseline')
    run_code = baseline['run_code']
    tau_u    = int(baseline['tau_u'])
    tau_s    = int(baseline['tau_s'])

    el_db_path = paths.working / 'edge_lists.duckdb'

    # ── Edge list ──────────────────────────────────────────────────────────────
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

    # ── Build matrix ───────────────────────────────────────────────────────────
    print('Building H_ww ...', flush=True)
    H, work_ids, out_weight, n_zero = _build_matrix(df_edges)
    N = len(work_ids)
    print(f'  N={N:,}   nnz={H.nnz:,}   density={H.nnz/N**2:.2e}   '
          f'zero-out-degree (dangling) works={n_zero:,} ({100*n_zero/N:.1f}%)')

    # ── Eigenpairs ─────────────────────────────────────────────────────────────
    ep = _compute_eigenpairs(H, k=6)

    # Persist eigenvalue diagnostics
    eig_diag = {}
    for side in ('right', 'left'):
        if ep[side] is not None:
            vals  = ep[side]['vals']
            lam1  = abs(vals[0])
            eig_diag[side] = [
                dict(k=i+1,
                     real=float(v.real), imag=float(v.imag),
                     abs=float(abs(v)),
                     ratio=float(abs(v)/lam1) if lam1 > 0 else None)
                for i, v in enumerate(vals)
            ]
    with open(out_dir / 'eigenvalues.json', 'w') as f:
        json.dump(eig_diag, f, indent=2)
    print(f'\nSaved eigenvalues.json')

    # ── Work rankings ──────────────────────────────────────────────────────────
    v_hub  = _leading_vector(ep['right'], 'right')
    v_auth = _leading_vector(ep['left'],  'left')

    if v_hub is None and v_auth is None:
        print('Both eigenpair computations failed; no rankings produced.')
        return

    # Normalise to median=1 over positive-valued works
    def norm_median(v):
        pos = v[v > 0]
        return v / np.median(pos) if len(pos) > 0 else v

    if v_hub  is not None: v_hub  = norm_median(v_hub)
    if v_auth is not None: v_auth = norm_median(v_auth)

    df_works = pd.DataFrame({
        'work_idx':   work_ids,
        'v_hub':      (v_hub  if v_hub  is not None else np.full(N, np.nan)).astype(np.float32),
        'v_auth':     (v_auth if v_auth is not None else np.full(N, np.nan)).astype(np.float32),
        'out_weight': out_weight.astype(np.float32),
    })
    df_works.to_parquet(out_dir / 'work_rankings.parquet', index=False)
    print(f'Saved work_rankings.parquet  ({N:,} works)')
    print(f'  v_hub:  min={df_works.v_hub.min():.4f}  '
          f'median={df_works.v_hub.median():.4f}  '
          f'max={df_works.v_hub.max():.4f}  '
          f'zero={int((df_works.v_hub==0).sum())}')
    print(f'  v_auth: min={df_works.v_auth.min():.4f}  '
          f'median={df_works.v_auth.median():.4f}  '
          f'max={df_works.v_auth.max():.4f}  '
          f'zero={int((df_works.v_auth==0).sum())}')

    # ── Source aggregation ─────────────────────────────────────────────────────
    print('\nAggregating to sources ...', flush=True)
    agg_parts = []
    for vec, name in [(v_hub, 'hub'), (v_auth, 'auth')]:
        if vec is None:
            continue
        df_agg = _aggregate_sources(work_ids, vec, out_weight, df_src)
        df_agg = df_agg.rename(columns={
            'v_uniform': f'v_{name}_uniform',
            'v_degree':  f'v_{name}_degree',
        })
        if not agg_parts:
            agg_parts.append(df_agg)
        else:
            agg_parts.append(df_agg.drop(columns=['n_works']))

    df_src_agg = agg_parts[0]
    for part in agg_parts[1:]:
        df_src_agg = df_src_agg.merge(part, on='source_idx')

    df_src_agg.to_parquet(out_dir / 'source_rankings.parquet', index=False)
    print(f'Saved source_rankings.parquet  ({len(df_src_agg):,} sources)')

    print(f'\nOutput: {out_dir}')


if __name__ == '__main__':
    main()
