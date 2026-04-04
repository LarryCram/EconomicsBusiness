"""
bootstrap_baseline.py — Bootstrap uncertainty analysis for the baseline spectral ranking.

Resamples 80% (with replacement) of the deduplicated SI and IS reference events
B times, fits bipartite spectral rankings, and stores v_s and v_u arrays.

CLI
---
python spectral_ranking_bootstrap/bootstrap_baseline.py [--n 1000] [--seed 42]
                                                        [--tol 1e-7] [--resume]
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
from spectral_ranking.katz_ranker import bipartite, _row_normalise, NotPrimitiveError

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_INTERVAL = 50
SAMPLE_FRACTION = 0.8


# ── Core functions (importable by tests) ─────────────────────────────────────

def compute_r_bar(db, el_table: str) -> float:
    """
    Compute r_bar = mean of distinct (citer_work_idx, R_i) pairs.

    This matches the ρ=0 fixed-count weighting in build_csr.py exactly.
    """
    r_bar = db.execute(
        f"SELECT AVG(rval) FROM "
        f"(SELECT DISTINCT citer_work_idx, CAST(R_i AS DOUBLE) AS rval FROM {el_table})"
    ).fetchone()[0]
    return float(r_bar)


def create_tmp_el(db, el_table: str, r_bar: float) -> None:
    """
    Materialise _tmp_el with rho_w column, exactly as build_csr.py does for rho=0.
    """
    db.execute(f"""
        CREATE OR REPLACE TEMP TABLE _tmp_el AS
        SELECT citer_work_idx, citer_source_idx, citer_inst_idx,
               cited_work_idx,  cited_source_idx, cited_inst_idx,
               inst_weight, cited_inst_weight,
               {r_bar} / CAST(R_i AS DOUBLE) AS rho_w
        FROM {el_table}
    """)


def preload_edges(db, units_table: str):
    """
    Pre-load deduplicated SI and IS edge arrays from _tmp_el (must already exist).

    Also loads unit index (source_ids, inst_ids, a_s, a_u).

    Parameters
    ----------
    db : duckdb connection with _tmp_el already created.
    units_table : name of the _units_... table.

    Returns
    -------
    dict with keys:
        si_src_dense, si_inst_dense, si_w   — SI edge arrays
        is_inst_dense, is_src_dense, is_w   — IS edge arrays
        source_ids, inst_ids                — original ids (int64)
        a_s, a_u                            — work counts (float64)
        n_s, n_u                            — dimensions
        N_SI, N_IS                          — edge counts
    """
    # Load unit index
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

    # SI deduplicated edges (inner DISTINCT from _build_si in build_csr.py)
    df_si = db.execute("""
        SELECT citer_source_idx, cited_inst_idx,
               rho_w * cited_inst_weight AS w
        FROM (
            SELECT DISTINCT citer_work_idx, citer_source_idx,
                            cited_work_idx,  cited_inst_idx,
                            rho_w, cited_inst_weight
            FROM _tmp_el
        )
    """).fetchdf()

    si_src_dense  = src_pd_index.get_indexer(df_si['citer_source_idx'].to_numpy()).astype(np.int32)
    si_inst_dense = inst_pd_index.get_indexer(df_si['cited_inst_idx'].to_numpy()).astype(np.int32)
    si_w          = df_si['w'].to_numpy(dtype=np.float64)
    N_SI = len(si_w)

    # IS deduplicated edges (inner DISTINCT from _build_is in build_csr.py)
    df_is = db.execute("""
        SELECT citer_inst_idx, cited_source_idx,
               rho_w * inst_weight AS w
        FROM (
            SELECT DISTINCT citer_work_idx, citer_inst_idx,
                            cited_work_idx,  cited_source_idx,
                            rho_w, inst_weight
            FROM _tmp_el
        )
    """).fetchdf()

    is_inst_dense = inst_pd_index.get_indexer(df_is['citer_inst_idx'].to_numpy()).astype(np.int32)
    is_src_dense  = src_pd_index.get_indexer(df_is['cited_source_idx'].to_numpy()).astype(np.int32)
    is_w          = df_is['w'].to_numpy(dtype=np.float64)
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
    )


def bootstrap_step(b: int, seed: int, edges: dict, tol: float) -> tuple:
    """
    Run one bootstrap replicate.

    Parameters
    ----------
    b : replicate index (used to derive per-replicate seed)
    seed : base seed
    edges : dict returned by preload_edges()
    tol : power iteration tolerance

    Returns
    -------
    (pi_s, pi_u, iters, final_norm)  — individually L1-normalised, float64
    """
    rng = np.random.default_rng(seed + b)

    n_s    = edges['n_s']
    n_u    = edges['n_u']
    N_SI   = edges['N_SI']
    N_IS   = edges['N_IS']

    # Sample SI block
    boot_si = rng.choice(N_SI, size=int(SAMPLE_FRACTION * N_SI), replace=True)
    C_SI = sp.coo_matrix(
        (edges['si_w'][boot_si],
         (edges['si_src_dense'][boot_si], edges['si_inst_dense'][boot_si])),
        shape=(n_s, n_u),
    ).tocsr()

    # Sample IS block
    boot_is = rng.choice(N_IS, size=int(SAMPLE_FRACTION * N_IS), replace=True)
    C_IS = sp.coo_matrix(
        (edges['is_w'][boot_is],
         (edges['is_inst_dense'][boot_is], edges['is_src_dense'][boot_is])),
        shape=(n_u, n_s),
    ).tocsr()

    H_SI, _ = _row_normalise(C_SI)
    H_IS, _ = _row_normalise(C_IS)
    pi_s, pi_u, iters, final_norm = bipartite(H_SI, H_IS, alpha=1.0, tol=tol,
                                              skip_primitive_check=True)

    return pi_s, pi_u, iters, final_norm


def compute_v(pi_s: np.ndarray, pi_u: np.ndarray,
              a_s: np.ndarray, a_u: np.ndarray) -> tuple:
    """
    Compute prestige-per-work v from individually-normalised pi_s, pi_u.

    Applies joint normalisation (divide by 2) before computing v, consistent
    with the m=0110 formula in katz_ranker.rank().

    Returns
    -------
    v_s_b : float32 array, shape (n_s,)
    v_u_b : float32 array, shape (n_u,)
    """
    pi_s = pi_s / 2.0
    pi_u = pi_u / 2.0
    A = a_s.sum() + a_u.sum()
    v_s_b = (A * pi_s / a_s).astype(np.float32)
    v_u_b = (A * pi_u / a_u).astype(np.float32)
    return v_s_b, v_u_b


def save_checkpoint(out_dir: Path, v_s_boot: np.ndarray, v_u_boot: np.ndarray,
                    meta: dict) -> None:
    """Write current arrays and meta.json to out_dir."""
    np.save(out_dir / 'v_s_boot.npy', v_s_boot)
    np.save(out_dir / 'v_u_boot.npy', v_u_boot)
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(out_dir: Path) -> tuple:
    """
    Load existing arrays and meta from out_dir.

    Returns
    -------
    (v_s_boot, v_u_boot, meta)
    """
    v_s_boot = np.load(out_dir / 'v_s_boot.npy')
    v_u_boot = np.load(out_dir / 'v_u_boot.npy')
    with open(out_dir / 'meta.json') as f:
        meta = json.load(f)
    return v_s_boot, v_u_boot, meta


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap uncertainty analysis for the baseline spectral ranking.'
    )
    parser.add_argument('--n',      type=int,   default=1000, help='Number of bootstrap replicates')
    parser.add_argument('--seed',   type=int,   default=42,   help='Base random seed')
    parser.add_argument('--tol',    type=float, default=1e-7, help='Power iteration tolerance')
    parser.add_argument('--resume', action='store_true',      help='Resume from existing checkpoint')
    args = parser.parse_args()

    B    = args.n
    seed = args.seed
    tol  = args.tol

    # ── Identify baseline corpus ───────────────────────────────────────────
    _baseline = next(r for r in load_runs() if r['label'] == 'baseline')
    run_code  = _baseline['run_code']    # '20242024'
    tau_u     = _baseline['tau_u']       # 20
    tau_s     = _baseline['tau_s']       # 20
    fx        = _baseline['fx']          # 'A'
    el_table  = f'el_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}'
    units_table = f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}'
    rk_table  = f'rk_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_rho0_m0110_chi50_alpha100'

    # ── Output directory ───────────────────────────────────────────────────
    paths = load_config()
    out_dir = paths.working / 'bootstrap'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {out_dir}', flush=True)

    # ── Connect to edge_lists.duckdb (read-only) ───────────────────────────
    import duckdb
    el_db_path = paths.working / 'edge_lists.duckdb'
    db = duckdb.connect(str(el_db_path), read_only=True)

    # ── Pre-load: r_bar and _tmp_el ────────────────────────────────────────
    t_preload = time.perf_counter()
    print('Computing r_bar ...', flush=True)
    r_bar = compute_r_bar(db, el_table)
    print(f'  r_bar = {r_bar:.6f}', flush=True)

    # Need a writable connection to create temp tables
    db.close()
    db = duckdb.connect(str(el_db_path), read_only=False)
    create_tmp_el(db, el_table, r_bar)
    print('  _tmp_el created', flush=True)

    print('Pre-loading SI and IS edges ...', flush=True)
    edges = preload_edges(db, units_table)
    db.execute('DROP TABLE IF EXISTS _tmp_el')
    db.close()

    t_preload = time.perf_counter() - t_preload
    n_s   = edges['n_s']
    n_u   = edges['n_u']
    N_SI  = edges['N_SI']
    N_IS  = edges['N_IS']
    a_s   = edges['a_s']
    a_u   = edges['a_u']
    print(
        f'  Pre-load done in {t_preload:.1f}s  '
        f'n_s={n_s}  n_u={n_u}  N_SI={N_SI}  N_IS={N_IS}',
        flush=True,
    )

    # ── Resume or initialise ───────────────────────────────────────────────
    start_b = 0
    if args.resume and (out_dir / 'v_s_boot.npy').exists():
        v_s_boot, v_u_boot, meta = load_checkpoint(out_dir)
        start_b = meta.get('completed', 0)
        print(f'Resuming from replicate {start_b}', flush=True)
        # Expand arrays if B increased
        if v_s_boot.shape[0] < B:
            extra_s = np.zeros((B - v_s_boot.shape[0], n_s), dtype=np.float32)
            extra_u = np.zeros((B - v_u_boot.shape[0], n_u), dtype=np.float32)
            v_s_boot = np.vstack([v_s_boot, extra_s])
            v_u_boot = np.vstack([v_u_boot, extra_u])
    else:
        v_s_boot = np.zeros((B, n_s), dtype=np.float32)
        v_u_boot = np.zeros((B, n_u), dtype=np.float32)
        meta = dict(
            n=B, seed=seed, tol=tol,
            n_s=n_s, n_u=n_u,
            source_ids=edges['source_ids'].tolist(),
            inst_ids=edges['inst_ids'].tolist(),
            run_code=run_code,
            baseline_table=el_table,
            completed=0,
        )

    # Update n in meta in case --n changed
    meta['n'] = B

    # ── Bootstrap loop ─────────────────────────────────────────────────────
    rep_times = []
    skipped   = 0
    for b in range(start_b, B):
        t0 = time.perf_counter()
        try:
            pi_s, pi_u, iters, final_norm = bootstrap_step(b, seed, edges, tol)
        except NotPrimitiveError as e:
            skipped += 1
            print(f'  replicate {b+1:4d}/{B}  SKIPPED (non-primitive): {e}', flush=True)
            continue
        v_s_b, v_u_b = compute_v(pi_s, pi_u, a_s, a_u)
        v_s_boot[b] = v_s_b
        v_u_boot[b] = v_u_b
        elapsed = time.perf_counter() - t0
        rep_times.append(elapsed)

        if (b + 1) % 10 == 0 or b == start_b:
            avg = np.mean(rep_times[-10:])
            remaining = (B - b - 1) * avg
            print(
                f'  replicate {b+1:4d}/{B}  iters={iters:3d}  '
                f'norm={final_norm:.2e}  {elapsed:.2f}s/rep  '
                f'ETA {remaining/60:.1f}min'
                + (f'  skipped={skipped}' if skipped else ''),
                flush=True,
            )

        # Checkpoint every CHECKPOINT_INTERVAL replicates
        if (b + 1) % CHECKPOINT_INTERVAL == 0:
            meta['completed'] = b + 1
            meta['skipped']   = skipped
            save_checkpoint(out_dir, v_s_boot, v_u_boot, meta)
            print(f'  checkpoint at replicate {b+1}', flush=True)

    # ── Final save ─────────────────────────────────────────────────────────
    meta['completed'] = B
    meta['skipped']   = skipped
    save_checkpoint(out_dir, v_s_boot, v_u_boot, meta)

    avg_rep = np.mean(rep_times) if rep_times else 0.0
    print(
        f'\nDone. {B} replicates  skipped={skipped}  '
        f'time={sum(rep_times)/60:.1f}min  '
        f'avg={avg_rep:.2f}s/rep\n'
        f'Output: {out_dir}',
        flush=True,
    )


if __name__ == '__main__':
    main()
