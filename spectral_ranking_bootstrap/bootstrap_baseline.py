"""
bootstrap_baseline.py — Bootstrap uncertainty analysis for the baseline spectral ranking.

Two sampling modes (select with --mode):

  work  (default)
        Sample 80% of unique citer works (without replacement).  Keep only edge
        rows in the work-level SI and IS tables where the citer work was drawn.
        Both SI and IS are filtered by the same work selection, so dropping a
        work removes its contribution consistently from both matrices.
        Measures rank sensitivity to work inclusion (sampling variability),
        not errors in how works are recorded.

  edge
        Sample 80% (with replacement) of the deduplicated SI and IS unit edge
        lists independently.  Measures sensitivity to individual edge-level
        citation links.

Each replicate is filtered to its bipartite core (iterative removal of
sources/institutions with zero indegree or zero outdegree), matching the SCC
criterion applied in the full pipeline.

Results stored as (B, n_s) and (B, n_u) float32 arrays; NaN indicates a unit
that was absent from the bipartite core in that replicate.

CLI
---
python spectral_ranking_bootstrap/bootstrap_baseline.py [--n 1000] [--seed 42]
                                                        [--tol 1e-7] [--resume]
                                                        [--mode work|edge]
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs
from spectral_ranking.katz_ranker import bipartite, _row_normalise

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_INTERVAL = 50
SAMPLE_FRACTION     = 0.8
MAX_FILTER_ITERS    = 200


# ── Core functions (importable by tests) ──────────────────────────────────────

def compute_r_bar(db, el_table: str) -> float:
    """
    Compute r_bar = mean of distinct (citer_work_idx, R_i) pairs.
    Matches the ρ=0 fixed-count weighting in build_csr.py exactly.
    """
    r_bar = db.execute(
        f"SELECT AVG(rval) FROM "
        f"(SELECT DISTINCT citer_work_idx, CAST(R_i AS DOUBLE) AS rval FROM {el_table})"
    ).fetchone()[0]
    return float(r_bar)


def create_tmp_el(db, el_table: str, r_bar: float) -> None:
    """Materialise _tmp_el with rho_w column, exactly as build_csr.py does for rho=0."""
    db.execute(f"""
        CREATE OR REPLACE TEMP TABLE _tmp_el AS
        SELECT citer_work_idx, citer_source_idx, citer_inst_idx,
               cited_work_idx,  cited_source_idx, cited_inst_idx,
               inst_weight, cited_inst_weight,
               {r_bar} / CAST(R_i AS DOUBLE) AS rho_w
        FROM {el_table}
    """)


def preload_edges(db, units_table: str) -> dict:
    """
    Pre-load deduplicated SI and IS unit edge lists from _tmp_el.

    Applies the same DISTINCT as build_csr.py, producing one entry per
    (citer_source, cited_inst) pair for SI and one per
    (citer_inst, cited_source) pair for IS.

    Parameters
    ----------
    db : duckdb connection with _tmp_el already created.
    units_table : name of the _units_... table.

    Returns
    -------
    dict with keys:
        si_src_dense, si_inst_dense, si_w   — SI edge arrays  (N_SI,)
        is_inst_dense, is_src_dense, is_w   — IS edge arrays  (N_IS,)
        source_ids, inst_ids                — original ids (int64)
        a_s, a_u                            — work counts (float64)
        n_s, n_u                            — dimensions
        N_SI, N_IS                          — edge counts
        A_full                              — a_s.sum() + a_u.sum()
    """
    # ── Unit index ────────────────────────────────────────────────────────────
    units_df = db.execute(
        f"SELECT unit_idx, unit_type, a_p FROM {units_table} ORDER BY unit_type, unit_idx"
    ).fetchdf()

    src_df  = units_df[units_df['unit_type'] == 'S'].reset_index(drop=True)
    inst_df = units_df[units_df['unit_type'] == 'U'].reset_index(drop=True)

    source_ids = src_df['unit_idx'].to_numpy(dtype=np.int64)
    inst_ids   = inst_df['unit_idx'].to_numpy(dtype=np.int64)
    a_s = src_df['a_p'].to_numpy(dtype=np.float64)
    a_u = inst_df['a_p'].to_numpy(dtype=np.float64)
    n_s = len(source_ids)
    n_u = len(inst_ids)

    src_pd_index  = pd.Index(source_ids)
    inst_pd_index = pd.Index(inst_ids)

    # ── SI deduplicated edges ─────────────────────────────────────────────────
    df_si = db.execute("""
        SELECT citer_source_idx, cited_inst_idx,
               SUM(rho_w * cited_inst_weight) AS w
        FROM (
            SELECT DISTINCT citer_work_idx, citer_source_idx,
                            cited_work_idx,  cited_inst_idx,
                            rho_w, cited_inst_weight
            FROM _tmp_el
        )
        GROUP BY citer_source_idx, cited_inst_idx
    """).fetchdf()

    si_src_dense  = src_pd_index.get_indexer(df_si['citer_source_idx'].to_numpy()).astype(np.int32)
    si_inst_dense = inst_pd_index.get_indexer(df_si['cited_inst_idx'].to_numpy()).astype(np.int32)
    si_w          = df_si['w'].to_numpy(dtype=np.float64)
    # Drop any edges involving units outside the units table
    valid_si      = (si_src_dense >= 0) & (si_inst_dense >= 0)
    si_src_dense  = si_src_dense[valid_si]
    si_inst_dense = si_inst_dense[valid_si]
    si_w          = si_w[valid_si]
    N_SI = len(si_w)

    # ── IS deduplicated edges ─────────────────────────────────────────────────
    df_is = db.execute("""
        SELECT citer_inst_idx, cited_source_idx,
               SUM(rho_w * inst_weight) AS w
        FROM (
            SELECT DISTINCT citer_work_idx, citer_inst_idx,
                            cited_work_idx,  cited_source_idx,
                            rho_w, inst_weight
            FROM _tmp_el
        )
        GROUP BY citer_inst_idx, cited_source_idx
    """).fetchdf()

    is_inst_dense = inst_pd_index.get_indexer(df_is['citer_inst_idx'].to_numpy()).astype(np.int32)
    is_src_dense  = src_pd_index.get_indexer(df_is['cited_source_idx'].to_numpy()).astype(np.int32)
    is_w          = df_is['w'].to_numpy(dtype=np.float64)
    valid_is      = (is_inst_dense >= 0) & (is_src_dense >= 0)
    is_inst_dense = is_inst_dense[valid_is]
    is_src_dense  = is_src_dense[valid_is]
    is_w          = is_w[valid_is]
    N_IS = len(is_w)

    return dict(
        si_src_dense=si_src_dense,
        si_inst_dense=si_inst_dense,
        si_w=si_w,
        is_inst_dense=is_inst_dense,
        is_src_dense=is_src_dense,
        is_w=is_w,
        source_ids=source_ids,
        inst_ids=inst_ids,
        a_s=a_s,
        a_u=a_u,
        n_s=n_s,
        n_u=n_u,
        N_SI=N_SI,
        N_IS=N_IS,
        A_full=float(a_s.sum() + a_u.sum()),
    )


def preload_work_edges(db, units_table: str) -> dict:
    """
    Pre-load work-level SI and IS edge lists from _tmp_el.

    SI: one row per distinct (citer_work_idx, citer_source_idx, cited_inst_idx),
        weight = SUM(rho_w * cited_inst_weight) over distinct cited_work_idx.
    IS: one row per distinct (citer_work_idx, citer_inst_idx, cited_source_idx),
        weight = SUM(rho_w * inst_weight) over distinct cited_work_idx.

    Dropping a citer work removes its contribution from both SI and IS, modelling
    the scenario where a work is missing from the corpus entirely.

    Returns
    -------
    dict with keys:
        si_work_dense, si_src_dense, si_inst_dense, si_w  — SI rows  (M_SI,)
        is_work_dense, is_inst_dense, is_src_dense, is_w  — IS rows  (M_IS,)
        unique_works                                       — (n_works,) int64
        n_works                                            — int
        source_ids, inst_ids                               — original ids (int64)
        a_s, a_u                                           — work counts (float64)
        n_s, n_u                                           — dimensions
        A_full                                             — a_s.sum() + a_u.sum()
    """
    # ── Unit index ────────────────────────────────────────────────────────────
    units_df = db.execute(
        f"SELECT unit_idx, unit_type, a_p FROM {units_table} ORDER BY unit_type, unit_idx"
    ).fetchdf()

    src_df  = units_df[units_df['unit_type'] == 'S'].reset_index(drop=True)
    inst_df = units_df[units_df['unit_type'] == 'U'].reset_index(drop=True)

    source_ids = src_df['unit_idx'].to_numpy(dtype=np.int64)
    inst_ids   = inst_df['unit_idx'].to_numpy(dtype=np.int64)
    a_s = src_df['a_p'].to_numpy(dtype=np.float64)
    a_u = inst_df['a_p'].to_numpy(dtype=np.float64)
    n_s = len(source_ids)
    n_u = len(inst_ids)

    src_pd_index  = pd.Index(source_ids)
    inst_pd_index = pd.Index(inst_ids)

    # ── SI work-level edges ───────────────────────────────────────────────────
    df_si = db.execute("""
        SELECT citer_work_idx, citer_source_idx, cited_inst_idx,
               SUM(rho_w * cited_inst_weight) AS w
        FROM (
            SELECT DISTINCT citer_work_idx, citer_source_idx,
                            cited_work_idx,  cited_inst_idx,
                            rho_w, cited_inst_weight
            FROM _tmp_el
        )
        GROUP BY citer_work_idx, citer_source_idx, cited_inst_idx
    """).fetchdf()

    si_src_dense  = src_pd_index.get_indexer(df_si['citer_source_idx'].to_numpy()).astype(np.int32)
    si_inst_dense = inst_pd_index.get_indexer(df_si['cited_inst_idx'].to_numpy()).astype(np.int32)
    valid_si      = (si_src_dense >= 0) & (si_inst_dense >= 0)
    si_src_dense  = si_src_dense[valid_si]
    si_inst_dense = si_inst_dense[valid_si]
    si_w          = df_si['w'].to_numpy(dtype=np.float64)[valid_si]
    si_work       = df_si['citer_work_idx'].to_numpy(dtype=np.int64)[valid_si]

    # ── IS work-level edges ───────────────────────────────────────────────────
    df_is = db.execute("""
        SELECT citer_work_idx, citer_inst_idx, cited_source_idx,
               SUM(rho_w * inst_weight) AS w
        FROM (
            SELECT DISTINCT citer_work_idx, citer_inst_idx,
                            cited_work_idx,  cited_source_idx,
                            rho_w, inst_weight
            FROM _tmp_el
        )
        GROUP BY citer_work_idx, citer_inst_idx, cited_source_idx
    """).fetchdf()

    is_inst_dense = inst_pd_index.get_indexer(df_is['citer_inst_idx'].to_numpy()).astype(np.int32)
    is_src_dense  = src_pd_index.get_indexer(df_is['cited_source_idx'].to_numpy()).astype(np.int32)
    valid_is      = (is_inst_dense >= 0) & (is_src_dense >= 0)
    is_inst_dense = is_inst_dense[valid_is]
    is_src_dense  = is_src_dense[valid_is]
    is_w          = df_is['w'].to_numpy(dtype=np.float64)[valid_is]
    is_work       = df_is['citer_work_idx'].to_numpy(dtype=np.int64)[valid_is]

    # ── Work index ────────────────────────────────────────────────────────────
    unique_works  = np.unique(np.concatenate([si_work, is_work]))
    n_works       = len(unique_works)
    work_pd_index = pd.Index(unique_works)

    si_work_dense = work_pd_index.get_indexer(si_work).astype(np.int32)
    is_work_dense = work_pd_index.get_indexer(is_work).astype(np.int32)

    return dict(
        si_work_dense=si_work_dense,
        si_src_dense=si_src_dense,
        si_inst_dense=si_inst_dense,
        si_w=si_w,
        is_work_dense=is_work_dense,
        is_inst_dense=is_inst_dense,
        is_src_dense=is_src_dense,
        is_w=is_w,
        unique_works=unique_works,
        n_works=n_works,
        source_ids=source_ids,
        inst_ids=inst_ids,
        a_s=a_s,
        a_u=a_u,
        n_s=n_s,
        n_u=n_u,
        A_full=float(a_s.sum() + a_u.sum()),
    )


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


def bootstrap_step(b: int, seed: int, edges: dict, tol: float) -> tuple:
    """
    Run one bootstrap replicate.

    Samples 80% (with replacement) of the deduplicated SI and IS unit edge
    lists independently, filters to the bipartite core, and runs the spectral
    ranking on the core submatrix.

    Parameters
    ----------
    b     : replicate index (per-replicate seed = seed + b)
    seed  : base seed
    edges : dict returned by preload_edges()
    tol   : ARPACK tolerance

    Returns
    -------
    v_s_b    : float32 (n_s,) — NaN for units absent from bipartite core
    v_u_b    : float32 (n_u,) — NaN for units absent from bipartite core
    lam1     : float — dominant eigenvalue of M_S
    lam2     : float — second eigenvalue of M_S
    n_s_core : int — sources in bipartite core
    n_u_core : int — institutions in bipartite core
    n_iter   : int — filtering passes to reach bipartite core
    """
    rng  = np.random.default_rng(seed + b)
    n_s  = edges['n_s']
    n_u  = edges['n_u']

    # ── Sample SI and IS unit edge lists independently ────────────────────────
    boot_si = rng.choice(edges['N_SI'], size=int(SAMPLE_FRACTION * edges['N_SI']), replace=True)
    C_SI = sp.coo_matrix(
        (edges['si_w'][boot_si],
         (edges['si_src_dense'][boot_si], edges['si_inst_dense'][boot_si])),
        shape=(n_s, n_u),
    ).tocsr()

    boot_is = rng.choice(edges['N_IS'], size=int(SAMPLE_FRACTION * edges['N_IS']), replace=True)
    C_IS = sp.coo_matrix(
        (edges['is_w'][boot_is],
         (edges['is_inst_dense'][boot_is], edges['is_src_dense'][boot_is])),
        shape=(n_u, n_s),
    ).tocsr()

    # ── Filter to bipartite core ───────────────────────────────────────────────
    s_idx, u_idx, n_iter = _bipartite_core(C_SI, C_IS)

    sub_SI = C_SI[s_idx, :][:, u_idx]
    sub_IS = C_IS[u_idx, :][:, s_idx]

    # ── Spectral ranking on core ───────────────────────────────────────────────
    H_SI, _ = _row_normalise(sub_SI)
    H_IS, _ = _row_normalise(sub_IS)
    pi_s_sub, pi_u_sub, lam1, lam2, _iters, _norm = bipartite(
        H_SI, H_IS, alpha=1.0, tol=tol,
    )

    # ── Map back to full arrays (NaN for absent units) ────────────────────────
    A_full = edges['A_full']
    v_s_b  = np.full(n_s, np.nan, dtype=np.float32)
    v_u_b  = np.full(n_u, np.nan, dtype=np.float32)
    v_s_b[s_idx] = (A_full * pi_s_sub / 2.0 / edges['a_s'][s_idx]).astype(np.float32)
    v_u_b[u_idx] = (A_full * pi_u_sub / 2.0 / edges['a_u'][u_idx]).astype(np.float32)

    return v_s_b, v_u_b, lam1, lam2, len(s_idx), len(u_idx), n_iter


def bootstrap_step_work(b: int, seed: int, edges: dict, tol: float) -> tuple:
    """
    Run one bootstrap replicate using citer-work sampling.

    Samples 80% of unique citer works (without replacement).  All SI and IS
    edge rows whose citer_work_idx was not selected are dropped before building
    the matrices.  Dropping a work removes its contribution from both SI and IS
    consistently — equivalent to removing the work from the corpus entirely.

    Parameters
    ----------
    b     : replicate index (per-replicate seed = seed + b)
    seed  : base seed
    edges : dict returned by preload_work_edges()
    tol   : ARPACK tolerance

    Returns
    -------
    Same 7-tuple as bootstrap_step.
    """
    rng     = np.random.default_rng(seed + b)
    n_s     = edges['n_s']
    n_u     = edges['n_u']
    n_works = edges['n_works']

    # ── Sample 80% of citer works (without replacement) ───────────────────────
    n_sel    = int(SAMPLE_FRACTION * n_works)
    sel      = rng.choice(n_works, size=n_sel, replace=False)
    work_sel = np.zeros(n_works, dtype=bool)
    work_sel[sel] = True

    # ── Filter SI and IS to selected works ────────────────────────────────────
    mask_si = work_sel[edges['si_work_dense']]
    C_SI = sp.coo_matrix(
        (edges['si_w'][mask_si],
         (edges['si_src_dense'][mask_si], edges['si_inst_dense'][mask_si])),
        shape=(n_s, n_u),
    ).tocsr()

    mask_is = work_sel[edges['is_work_dense']]
    C_IS = sp.coo_matrix(
        (edges['is_w'][mask_is],
         (edges['is_inst_dense'][mask_is], edges['is_src_dense'][mask_is])),
        shape=(n_u, n_s),
    ).tocsr()

    # ── Filter to bipartite core ───────────────────────────────────────────────
    s_idx, u_idx, n_iter = _bipartite_core(C_SI, C_IS)

    sub_SI = C_SI[s_idx, :][:, u_idx]
    sub_IS = C_IS[u_idx, :][:, s_idx]

    # ── Spectral ranking on core ───────────────────────────────────────────────
    H_SI, _ = _row_normalise(sub_SI)
    H_IS, _ = _row_normalise(sub_IS)
    pi_s_sub, pi_u_sub, lam1, lam2, _iters, _norm = bipartite(
        H_SI, H_IS, alpha=1.0, tol=tol,
    )

    # ── Map back to full arrays (NaN for absent units) ────────────────────────
    A_full = edges['A_full']
    v_s_b  = np.full(n_s, np.nan, dtype=np.float32)
    v_u_b  = np.full(n_u, np.nan, dtype=np.float32)
    v_s_b[s_idx] = (A_full * pi_s_sub / 2.0 / edges['a_s'][s_idx]).astype(np.float32)
    v_u_b[u_idx] = (A_full * pi_u_sub / 2.0 / edges['a_u'][u_idx]).astype(np.float32)

    return v_s_b, v_u_b, lam1, lam2, len(s_idx), len(u_idx), n_iter


def save_checkpoint(out_dir: Path, v_s_boot: np.ndarray, v_u_boot: np.ndarray,
                    lam_ratio_boot: np.ndarray, meta: dict) -> None:
    """Write current arrays and meta.json to out_dir."""
    np.save(out_dir / 'v_s_boot.npy', v_s_boot)
    np.save(out_dir / 'v_u_boot.npy', v_u_boot)
    np.save(out_dir / 'lam_ratio_boot.npy', lam_ratio_boot)
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(out_dir: Path) -> tuple:
    """
    Load existing arrays and meta from out_dir.
    Returns (v_s_boot, v_u_boot, lam_ratio_boot, meta).
    """
    v_s_boot = np.load(out_dir / 'v_s_boot.npy')
    v_u_boot = np.load(out_dir / 'v_u_boot.npy')
    lam_ratio_path = out_dir / 'lam_ratio_boot.npy'
    if lam_ratio_path.exists():
        lam_ratio_boot = np.load(lam_ratio_path)
    else:
        lam_ratio_boot = np.full(v_s_boot.shape[0], np.nan, dtype=np.float32)
    with open(out_dir / 'meta.json') as f:
        meta = json.load(f)
    return v_s_boot, v_u_boot, lam_ratio_boot, meta


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap uncertainty analysis for the baseline spectral ranking.'
    )
    parser.add_argument('--n',      type=int,   default=1000,   help='Number of bootstrap replicates')
    parser.add_argument('--seed',   type=int,   default=42,     help='Base random seed')
    parser.add_argument('--tol',    type=float, default=1e-7,   help='ARPACK tolerance')
    parser.add_argument('--resume', action='store_true',        help='Resume from existing checkpoint')
    parser.add_argument('--mode',   default='work',
                        choices=['work', 'edge'],
                        help='Sampling mode: work=citer-work (default), edge=unit-edge-list')
    args = parser.parse_args()

    B    = args.n
    seed = args.seed
    tol  = args.tol
    mode = args.mode

    # ── Identify baseline corpus ───────────────────────────────────────────────
    _baseline   = next(r for r in load_runs() if r['label'] == 'baseline')
    run_code    = _baseline['run_code']
    tau_u       = _baseline['tau_u']
    tau_s       = _baseline['tau_s']
    fx          = _baseline['fx']
    el_table    = f'el_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_vartau'
    units_table = f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_vartau_m0110'

    # ── Output directory ───────────────────────────────────────────────────────
    paths   = load_config()
    out_dir = paths.working / 'bootstrap'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {out_dir}  mode={mode}', flush=True)

    # ── Connect and pre-load ───────────────────────────────────────────────────
    import duckdb
    el_db_path = paths.working / 'edge_lists.duckdb'

    t_preload = time.perf_counter()
    print('Computing r_bar ...', flush=True)
    db = duckdb.connect(str(el_db_path), read_only=True)
    r_bar = compute_r_bar(db, el_table)
    db.close()
    print(f'  r_bar = {r_bar:.6f}', flush=True)

    db = duckdb.connect(str(el_db_path), read_only=False)
    create_tmp_el(db, el_table, r_bar)
    print('  _tmp_el created', flush=True)

    if mode == 'work':
        print('Pre-loading work-level SI and IS edge lists (citer-work mode) ...', flush=True)
        edges = preload_work_edges(db, units_table)
        db.execute('DROP TABLE IF EXISTS _tmp_el')
        db.close()
        t_preload = time.perf_counter() - t_preload
        print(
            f'  Pre-load done in {t_preload:.1f}s  '
            f'n_s={edges["n_s"]}  n_u={edges["n_u"]}  '
            f'n_works={edges["n_works"]}  '
            f'M_SI={len(edges["si_w"])}  M_IS={len(edges["is_w"])}',
            flush=True,
        )
        step_fn = bootstrap_step_work
    else:
        print('Pre-loading deduplicated SI and IS unit edge lists (edge mode) ...', flush=True)
        edges = preload_edges(db, units_table)
        db.execute('DROP TABLE IF EXISTS _tmp_el')
        db.close()
        t_preload = time.perf_counter() - t_preload
        print(
            f'  Pre-load done in {t_preload:.1f}s  '
            f'n_s={edges["n_s"]}  n_u={edges["n_u"]}  '
            f'N_SI={edges["N_SI"]}  N_IS={edges["N_IS"]}',
            flush=True,
        )
        step_fn = bootstrap_step

    # ── Resume or initialise ───────────────────────────────────────────────────
    n_s = edges['n_s']
    n_u = edges['n_u']
    start_b = 0

    if args.resume and (out_dir / 'v_s_boot.npy').exists():
        v_s_boot, v_u_boot, lam_ratio_boot, meta = load_checkpoint(out_dir)
        start_b = meta.get('completed', 0)
        print(f'Resuming from replicate {start_b}', flush=True)
        if v_s_boot.shape[0] < B:
            extra_s   = np.full((B - v_s_boot.shape[0], n_s), np.nan, dtype=np.float32)
            extra_u   = np.full((B - v_u_boot.shape[0], n_u), np.nan, dtype=np.float32)
            extra_lam = np.full(B - lam_ratio_boot.shape[0], np.nan, dtype=np.float32)
            v_s_boot       = np.vstack([v_s_boot, extra_s])
            v_u_boot       = np.vstack([v_u_boot, extra_u])
            lam_ratio_boot = np.concatenate([lam_ratio_boot, extra_lam])
    else:
        v_s_boot       = np.full((B, n_s), np.nan, dtype=np.float32)
        v_u_boot       = np.full((B, n_u), np.nan, dtype=np.float32)
        lam_ratio_boot = np.full(B, np.nan, dtype=np.float32)
        meta = dict(
            n=B, seed=seed, tol=tol, mode=mode,
            n_s=n_s, n_u=n_u,
            source_ids=edges['source_ids'].tolist(),
            inst_ids=edges['inst_ids'].tolist(),
            run_code=run_code,
            baseline_table=el_table,
            completed=0,
        )

    meta['n'] = B

    # ── Bootstrap loop ─────────────────────────────────────────────────────────
    rep_times  = []
    core_s_log = []
    core_u_log = []

    for b in range(start_b, B):
        t0 = time.perf_counter()
        v_s_b, v_u_b, lam1, lam2, n_s_core, n_u_core, n_iter = \
            step_fn(b, seed, edges, tol)

        v_s_boot[b]       = v_s_b
        v_u_boot[b]       = v_u_b
        lam_ratio_boot[b] = float(lam2 / lam1) if lam1 > 0 else np.nan
        elapsed = time.perf_counter() - t0
        rep_times.append(elapsed)
        core_s_log.append(n_s_core)
        core_u_log.append(n_u_core)

        if (b + 1) % 10 == 0 or b == start_b:
            avg       = np.mean(rep_times[-10:])
            remaining = (B - b - 1) * avg
            print(
                f'  replicate {b+1:4d}/{B}  {elapsed:.2f}s/rep  '
                f'core=({n_s_core},{n_u_core})  filt={n_iter}  '
                f'lam2/lam1={lam_ratio_boot[b]:.4f}  '
                f'ETA {remaining/60:.1f}min',
                flush=True,
            )

        if (b + 1) % CHECKPOINT_INTERVAL == 0:
            meta['completed'] = b + 1
            save_checkpoint(out_dir, v_s_boot, v_u_boot, lam_ratio_boot, meta)
            print(f'  checkpoint at replicate {b+1}', flush=True)

    # ── Final save ─────────────────────────────────────────────────────────────
    meta['completed'] = B
    save_checkpoint(out_dir, v_s_boot, v_u_boot, lam_ratio_boot, meta)

    avg_rep = np.mean(rep_times) if rep_times else 0.0
    print(
        f'\nDone. {B} replicates  '
        f'time={sum(rep_times)/60:.1f}min  avg={avg_rep:.2f}s/rep',
        flush=True,
    )
    print(
        f'Bipartite core: '
        f'sources  min={min(core_s_log)}  mean={np.mean(core_s_log):.0f}  '
        f'max={max(core_s_log)}  (full={n_s})',
        flush=True,
    )
    print(
        f'               '
        f'insts    min={min(core_u_log)}  mean={np.mean(core_u_log):.0f}  '
        f'max={max(core_u_log)}  (full={n_u})',
        flush=True,
    )
    print(f'Output: {out_dir}', flush=True)


if __name__ == '__main__':
    main()
