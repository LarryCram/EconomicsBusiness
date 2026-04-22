"""
bootstrap_oa_errors.py — Bootstrap modelling OpenAlex data errors in the bipartite
spectral ranking.

Modes (--mode):

  ref  (default)
        For each reference link (citer_work → cited_work), with probability
        p_ref the cited work is replaced by a randomly drawn work from the
        same source and publication year (with replacement).  Models errors
        in reference assignment, e.g. a citation landing on the wrong paper
        in the same journal due to a misread page number.
        C_IS is unchanged (cited source identical); only C_SI is perturbed.

  inst  [NOT YET IMPLEMENTED]
        For each institution assignment in the corpus, with probability
        p_inst the institution is replaced by a randomly drawn institution
        from the same country.  Models author-affiliation disambiguation
        errors.  Perturbs both C_SI (cited institution) and C_IS (citing
        institution).

Results stored in $WORKING/bootstrap_oa_errors/ in the same format as
bootstrap_baseline.py (v_s_boot.npy, v_u_boot.npy, lam_ratio_boot.npy,
meta.json) so that fig_7.py can load either bootstrap without modification.

CLI
---
python spectral_ranking_bootstrap/bootstrap_oa_errors.py [--n 1000] [--seed 42]
                                                         [--tol 1e-7] [--resume]
                                                         [--mode ref]
                                                         [--p 0.05]
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
from spectral_ranking_bootstrap.bootstrap_baseline import (
    compute_r_bar,
    create_tmp_el,
    _bipartite_core,
    save_checkpoint,
    load_checkpoint,
)

CHECKPOINT_INTERVAL = 50
DEFAULT_P_REF       = 0.05


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_unit_index(db, units_table: str) -> dict:
    """Return unit arrays and dense-index maps from a _units_... table."""
    units_df = db.execute(
        f"SELECT unit_idx, unit_type, a_p FROM {units_table} ORDER BY unit_type, unit_idx"
    ).fetchdf()
    src_df  = units_df[units_df['unit_type'] == 'S'].reset_index(drop=True)
    inst_df = units_df[units_df['unit_type'] == 'U'].reset_index(drop=True)
    source_ids = src_df['unit_idx'].to_numpy(dtype=np.int64)
    inst_ids   = inst_df['unit_idx'].to_numpy(dtype=np.int64)
    return dict(
        source_ids    = source_ids,
        inst_ids      = inst_ids,
        a_s           = src_df['a_p'].to_numpy(dtype=np.float64),
        a_u           = inst_df['a_p'].to_numpy(dtype=np.float64),
        n_s           = len(source_ids),
        n_u           = len(inst_ids),
        src_pd_index  = pd.Index(source_ids),
        inst_pd_index = pd.Index(inst_ids),
    )


def _run_core_and_rank(C_SI: sp.csr_matrix, C_IS: sp.csr_matrix,
                       n_s: int, n_u: int, a_s: np.ndarray, a_u: np.ndarray,
                       A_full: float, tol: float) -> tuple:
    """
    Filter to bipartite core, run spectral ranking, return full-length v arrays
    (NaN for units absent from core) plus diagnostics.
    """
    s_idx, u_idx, n_iter = _bipartite_core(C_SI, C_IS)

    sub_SI = C_SI[s_idx, :][:, u_idx]
    sub_IS = C_IS[u_idx, :][:, s_idx]

    H_SI, _ = _row_normalise(sub_SI)
    H_IS, _ = _row_normalise(sub_IS)
    pi_s_sub, pi_u_sub, lam1, lam2, _iters, _norm = bipartite(
        H_SI, H_IS, alpha=1.0, tol=tol,
    )

    v_s_b = np.full(n_s, np.nan, dtype=np.float32)
    v_u_b = np.full(n_u, np.nan, dtype=np.float32)
    v_s_b[s_idx] = (A_full * pi_s_sub / 2.0 / a_s[s_idx]).astype(np.float32)
    v_u_b[u_idx] = (A_full * pi_u_sub / 2.0 / a_u[u_idx]).astype(np.float32)

    return v_s_b, v_u_b, lam1, lam2, len(s_idx), len(u_idx), n_iter


# ── Wrong-reference mode ───────────────────────────────────────────────────────

def preload_ref_edges(db, units_table: str, works_df: pd.DataFrame) -> dict:
    """
    Pre-load reference-level data for the wrong-reference bootstrap mode.

    works_df must have columns [work_idx, publication_year] covering at
    least all cited_work_idx values in the edge list.  Rows without a
    matched year are excluded from like-sets (their references cannot be
    replaced, which is conservative).

    Returns a dict with keys:

      ref_citer_src_dense  (N_refs,) int32  — dense source index for citer
      ref_cited_work_dense (N_refs,) int32  — dense cited-work index
      ref_rho_w            (N_refs,) float64 — rho_w for the reference
      like_sets            dict[int → ndarray(int32)]
                             cw_dense → candidate cw_dense values
                             (only for works with ≥2 candidates)
      W_cw                 (N_cw, n_u) CSR sparse
                             row w: cited_inst_weight for work w's institutions
      C_IS                 (n_u, n_s) CSR sparse — unnormalised baseline C_IS
      source_ids, inst_ids, a_s, a_u, n_s, n_u, A_full
      cited_work_ids       (N_cw,) int64 — original cited_work_idx values
    """
    units = _load_unit_index(db, units_table)
    n_s   = units['n_s']
    n_u   = units['n_u']
    src_pd_index  = units['src_pd_index']
    inst_pd_index = units['inst_pd_index']

    # ── Reference-level: one row per distinct (citer_work, cited_work) ────────
    df_refs = db.execute("""
        SELECT DISTINCT citer_work_idx, citer_source_idx, cited_work_idx,
                        MAX(rho_w) AS rho_w
        FROM _tmp_el
        GROUP BY citer_work_idx, citer_source_idx, cited_work_idx
    """).fetchdf()
    # rho_w is constant per citer_work_idx; MAX is just a safe aggregator.

    # ── Work-institution matrix: one row per distinct (cited_work, cited_inst) ─
    df_wi = db.execute("""
        SELECT DISTINCT cited_work_idx, cited_inst_idx, cited_inst_weight
        FROM _tmp_el
    """).fetchdf()
    # cited_inst_weight is a property of (cited_work, institution); take MAX in
    # case of floating-point duplicates introduced by DISTINCT.
    df_wi = (df_wi
             .groupby(['cited_work_idx', 'cited_inst_idx'], as_index=False)['cited_inst_weight']
             .max())

    # ── Cited-source for each cited_work (for like-set grouping) ──────────────
    df_csrc = db.execute("""
        SELECT DISTINCT cited_work_idx, cited_source_idx
        FROM _tmp_el
    """).fetchdf()

    # ── Baseline C_IS (fixed across all replicates) ───────────────────────────
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
    valid_is      = (is_inst_dense >= 0) & (is_src_dense >= 0)
    C_IS = sp.coo_matrix(
        (df_is['w'].to_numpy(dtype=np.float64)[valid_is],
         (is_inst_dense[valid_is], is_src_dense[valid_is])),
        shape=(n_u, n_s),
    ).tocsr()

    # ── Build dense cited-work index ──────────────────────────────────────────
    cited_work_ids = np.unique(df_refs['cited_work_idx'].to_numpy(dtype=np.int64))
    N_cw           = len(cited_work_ids)
    cw_pd_index    = pd.Index(cited_work_ids)

    # ── W_cw: (N_cw, n_u) sparse ──────────────────────────────────────────────
    cw_dense   = cw_pd_index.get_indexer(df_wi['cited_work_idx'].to_numpy()).astype(np.int32)
    inst_dense = inst_pd_index.get_indexer(df_wi['cited_inst_idx'].to_numpy()).astype(np.int32)
    valid_wi   = (cw_dense >= 0) & (inst_dense >= 0)
    W_cw = sp.coo_matrix(
        (df_wi['cited_inst_weight'].to_numpy(dtype=np.float64)[valid_wi],
         (cw_dense[valid_wi], inst_dense[valid_wi])),
        shape=(N_cw, n_u),
    ).tocsr()

    # ── Reference arrays ──────────────────────────────────────────────────────
    citer_src_dense   = src_pd_index.get_indexer(
        df_refs['citer_source_idx'].to_numpy()
    ).astype(np.int32)
    cited_work_dense  = cw_pd_index.get_indexer(
        df_refs['cited_work_idx'].to_numpy()
    ).astype(np.int32)
    rho_w             = df_refs['rho_w'].to_numpy(dtype=np.float64)

    valid_refs        = (citer_src_dense >= 0) & (cited_work_dense >= 0)
    citer_src_dense   = citer_src_dense[valid_refs]
    cited_work_dense  = cited_work_dense[valid_refs]
    rho_w             = rho_w[valid_refs]

    # ── Like-sets: group cited_work by (cited_source, year) ───────────────────
    df_csrc_year = (df_csrc
                    .merge(works_df[['work_idx', 'publication_year']],
                           left_on='cited_work_idx', right_on='work_idx',
                           how='inner')
                    .drop(columns='work_idx'))
    df_csrc_year['cw_dense'] = cw_pd_index.get_indexer(
        df_csrc_year['cited_work_idx'].to_numpy()
    ).astype(np.int32)
    df_csrc_year = df_csrc_year[df_csrc_year['cw_dense'] >= 0]

    like_sets: dict[int, np.ndarray] = {}
    for (src_id, year), grp in df_csrc_year.groupby(
            ['cited_source_idx', 'publication_year'], sort=False):
        candidates = grp['cw_dense'].to_numpy(dtype=np.int32)
        if len(candidates) >= 2:
            for cw in candidates:
                like_sets[int(cw)] = candidates

    A_full = float(units['a_s'].sum() + units['a_u'].sum())

    # Fraction of REFERENCES (not just works) whose cited_work is in a like-set
    refs_in_ls = np.sum([1 for cw in cited_work_dense if int(cw) in like_sets])
    ref_ls_frac = refs_in_ls / max(len(cited_work_dense), 1)
    # Average like-set size (for works that are in one)
    ls_sizes = [len(v) for v in like_sets.values()]
    avg_ls_size = float(np.mean(ls_sizes)) if ls_sizes else 0.0

    print(
        f'  Ref preload: N_refs={len(rho_w):,}  N_cw={N_cw:,}  n_s={n_s}  n_u={n_u}',
        flush=True,
    )
    print(
        f'  Like-sets: work_coverage={len(like_sets)/N_cw:.1%}  '
        f'ref_coverage={ref_ls_frac:.1%}  avg_set_size={avg_ls_size:.1f}',
        flush=True,
    )
    print(
        f'  At p=0.05: ~{0.05 * ref_ls_frac:.2%} of refs effectively replaced per replicate '
        f'(p × ref_coverage × (1 − 1/avg_k) ≈ '
        f'{0.05 * ref_ls_frac * (1 - 1/avg_ls_size) if avg_ls_size > 1 else 0:.2%} different work)',
        flush=True,
    )

    return dict(
        ref_citer_src_dense  = citer_src_dense,
        ref_cited_work_dense = cited_work_dense,
        ref_rho_w            = rho_w,
        like_sets            = like_sets,
        W_cw                 = W_cw,
        C_IS                 = C_IS,
        cited_work_ids       = cited_work_ids,
        source_ids           = units['source_ids'],
        inst_ids             = units['inst_ids'],
        a_s                  = units['a_s'],
        a_u                  = units['a_u'],
        n_s                  = n_s,
        n_u                  = n_u,
        A_full               = A_full,
    )


def build_like_sets(cited_work_dense: np.ndarray,
                    cited_source_idx: np.ndarray,
                    cited_year: np.ndarray) -> dict[int, np.ndarray]:
    """
    Build like-sets from parallel arrays of (cw_dense, source_idx, year).

    A like-set for work w is all works with the same (source_idx, year) as w,
    including w itself.  Only works with ≥2 candidates in their group are
    entered; the rest cannot be meaningfully replaced.

    Returns dict[cw_dense → ndarray(int32) of cw_dense candidates].
    """
    df = pd.DataFrame({
        'cw_dense':   cited_work_dense,
        'source_idx': cited_source_idx,
        'year':       cited_year,
    })
    like_sets: dict[int, np.ndarray] = {}
    for _, grp in df.groupby(['source_idx', 'year'], sort=False):
        candidates = grp['cw_dense'].to_numpy(dtype=np.int32)
        if len(candidates) >= 2:
            for cw in candidates:
                like_sets[int(cw)] = candidates
    return like_sets


def build_csi_perturbed(ref_data: dict, p: float, rng: np.random.Generator) -> sp.csr_matrix:
    """
    Build a perturbed C_SI matrix for one bootstrap replicate.

    For each reference, with probability p the cited work is replaced by a
    random draw (with replacement) from its like-set.  References whose cited
    work has no like-set entry (singleton in its source×year group) are never
    replaced.

    Uses the vectorised formulation  C_SI = S_cw @ W_cw  where S_cw is a
    (n_s, N_cw) accumulation matrix: S_cw[s, w] = sum of rho_w for all
    references from source s that now point to cited_work w.

    Parameters
    ----------
    ref_data : dict returned by preload_ref_edges
    p        : per-reference error probability
    rng      : numpy Generator (per-replicate seeded)

    Returns
    -------
    C_SI : (n_s, N_cw) @ (N_cw, n_u)  →  (n_s, n_u) CSR, unnormalised
    """
    citer_src   = ref_data['ref_citer_src_dense']  # (N_refs,)
    cited_work  = ref_data['ref_cited_work_dense'].copy()  # (N_refs,) — mutable copy
    rho_w       = ref_data['ref_rho_w']
    like_sets   = ref_data['like_sets']
    W_cw        = ref_data['W_cw']
    n_s         = ref_data['n_s']
    N_cw        = W_cw.shape[0]

    # ── Apply perturbations ───────────────────────────────────────────────────
    error_mask = rng.random(len(cited_work)) < p
    for i in np.where(error_mask)[0]:
        cw = int(cited_work[i])
        ls = like_sets.get(cw)
        if ls is not None:
            cited_work[i] = rng.choice(ls)

    # ── S_cw accumulation matrix ──────────────────────────────────────────────
    S_cw = sp.coo_matrix(
        (rho_w, (citer_src, cited_work)),
        shape=(n_s, N_cw),
    ).tocsr()

    return S_cw @ W_cw  # (n_s, n_u)


def bootstrap_step_ref(b: int, seed: int, ref_data: dict, p: float,
                       tol: float) -> tuple:
    """
    Run one wrong-reference bootstrap replicate.

    Parameters
    ----------
    b        : replicate index (per-replicate seed = seed + b)
    seed     : base seed
    ref_data : dict returned by preload_ref_edges
    p        : per-reference error probability
    tol      : ARPACK tolerance

    Returns
    -------
    Same 7-tuple as bootstrap_baseline.bootstrap_step:
    (v_s_b, v_u_b, lam1, lam2, n_s_core, n_u_core, n_iter)
    """
    rng   = np.random.default_rng(seed + b)
    C_SI  = build_csi_perturbed(ref_data, p, rng)
    C_IS  = ref_data['C_IS']

    return _run_core_and_rank(
        C_SI, C_IS,
        ref_data['n_s'], ref_data['n_u'],
        ref_data['a_s'], ref_data['a_u'],
        ref_data['A_full'], tol,
    )


# ── Wrong-institution mode (placeholder) ──────────────────────────────────────

def preload_inst_edges(db, units_table: str, el_table: str,
                       institutions_df: pd.DataFrame) -> dict:
    """
    NOT YET IMPLEMENTED.

    Will pre-load institution-level data for the wrong-institution bootstrap.

    institutions_df must have columns [institution_idx, country_code] for
    all corpus institutions; like-sets will be institutions in the same country.

    Unlike the ref mode, both C_SI and C_IS are perturbed: a misidentified
    citing institution changes C_IS, and a misidentified cited institution
    changes C_SI.
    """
    raise NotImplementedError(
        'Wrong-institution bootstrap mode is not yet implemented. '
        'Use --mode ref.'
    )


def bootstrap_step_inst(b: int, seed: int, inst_data: dict, p: float,
                        tol: float) -> tuple:
    """NOT YET IMPLEMENTED.  See preload_inst_edges."""
    raise NotImplementedError(
        'Wrong-institution bootstrap mode is not yet implemented. '
        'Use --mode ref.'
    )


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap modelling OpenAlex data errors in the spectral ranking.'
    )
    parser.add_argument('--n',      type=int,   default=1000,       help='Number of replicates')
    parser.add_argument('--seed',   type=int,   default=42,         help='Base random seed')
    parser.add_argument('--tol',    type=float, default=1e-7,       help='ARPACK tolerance')
    parser.add_argument('--resume', action='store_true',            help='Resume from checkpoint')
    parser.add_argument('--mode',   default='ref', choices=['ref', 'inst'],
                        help='Error model: ref=wrong-reference (default), inst=wrong-institution')
    parser.add_argument('--p',      type=float, default=DEFAULT_P_REF,
                        help=f'Per-unit error probability (default {DEFAULT_P_REF})')
    args = parser.parse_args()

    if args.mode == 'inst':
        raise NotImplementedError(
            'Wrong-institution bootstrap mode is not yet implemented. '
            'Use --mode ref.'
        )

    B    = args.n
    seed = args.seed
    tol  = args.tol
    p    = args.p

    import duckdb

    _baseline   = next(r for r in load_runs() if r['label'] == 'baseline')
    run_code    = _baseline['run_code']
    tau_u       = _baseline['tau_u']
    tau_s       = _baseline['tau_s']
    fx          = _baseline['fx']
    el_table    = f'el_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_vartau'
    units_table = f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_vartau_m0110'

    paths       = load_config()
    out_dir     = paths.working / 'bootstrap_oa_errors'
    out_dir.mkdir(parents=True, exist_ok=True)
    el_db_path  = paths.working / 'edge_lists.duckdb'
    print(f'Output: {out_dir}  mode={args.mode}  p={p}', flush=True)

    # ── Load works for year lookup ─────────────────────────────────────────────
    print('Loading corpus_works for publication years ...', flush=True)
    works_df = pd.read_parquet(
        str(paths.parquet / 'corpus_works.parquet'),
        columns=['work_idx', 'publication_year'],
    )

    # ── Pre-load ───────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print('Computing r_bar ...', flush=True)
    db = duckdb.connect(str(el_db_path), read_only=True)
    r_bar = compute_r_bar(db, el_table)
    db.close()
    print(f'  r_bar = {r_bar:.6f}', flush=True)

    db = duckdb.connect(str(el_db_path), read_only=False)
    create_tmp_el(db, el_table, r_bar)
    print('Pre-loading reference edges ...', flush=True)
    ref_data = preload_ref_edges(db, units_table, works_df)
    db.execute('DROP TABLE IF EXISTS _tmp_el')
    db.close()
    print(f'  Pre-load done in {time.perf_counter() - t0:.1f}s', flush=True)

    n_s = ref_data['n_s']
    n_u = ref_data['n_u']

    # ── Resume or initialise ───────────────────────────────────────────────────
    start_b = 0
    if args.resume and (out_dir / 'v_s_boot.npy').exists():
        v_s_boot, v_u_boot, lam_ratio_boot, meta = load_checkpoint(out_dir)
        start_b = meta.get('completed', 0)
        print(f'Resuming from replicate {start_b}', flush=True)
        if v_s_boot.shape[0] < B:
            v_s_boot       = np.vstack([v_s_boot,
                np.full((B - v_s_boot.shape[0], n_s), np.nan, dtype=np.float32)])
            v_u_boot       = np.vstack([v_u_boot,
                np.full((B - v_u_boot.shape[0], n_u), np.nan, dtype=np.float32)])
            lam_ratio_boot = np.concatenate([lam_ratio_boot,
                np.full(B - lam_ratio_boot.shape[0], np.nan, dtype=np.float32)])
    else:
        v_s_boot       = np.full((B, n_s), np.nan, dtype=np.float32)
        v_u_boot       = np.full((B, n_u), np.nan, dtype=np.float32)
        lam_ratio_boot = np.full(B, np.nan, dtype=np.float32)
        meta = dict(
            n=B, seed=seed, tol=tol, mode=args.mode, p=p,
            n_s=n_s, n_u=n_u,
            source_ids=ref_data['source_ids'].tolist(),
            inst_ids=ref_data['inst_ids'].tolist(),
            run_code=run_code,
            el_table=el_table,
            completed=0,
        )
    meta['n'] = B

    # ── Bootstrap loop ─────────────────────────────────────────────────────────
    rep_times  = []
    core_s_log = []
    core_u_log = []

    for b in range(start_b, B):
        t_rep = time.perf_counter()
        v_s_b, v_u_b, lam1, lam2, n_s_core, n_u_core, n_iter = \
            bootstrap_step_ref(b, seed, ref_data, p, tol)

        v_s_boot[b]       = v_s_b
        v_u_boot[b]       = v_u_b
        lam_ratio_boot[b] = float(lam2 / lam1) if lam1 > 0 else np.nan
        elapsed = time.perf_counter() - t_rep
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
