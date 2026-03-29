"""
run_rankings.py — Parameter driver for spectral ranking pipeline.

Reads: WORKING/edge_lists.duckdb (edge lists + unit index tables)
Writes: WORKING/rankings.duckdb  (ranking tables + _catalog)

Usage
-----
  python spectral_ranking/run_rankings.py          # baseline + Stage 1
  python spectral_ranking/run_rankings.py --stage 2  # time-series sweep

Output table naming
-------------------
  rk_t{tx}_{fx}_tau{tau_u}_rho{rho}_m{mstr}_chi{chi_int}_alpha{alpha_int}
  e.g. rk_t5_A_tau20_rho0_m0110_chi50_alpha85   (baseline)

Each table has columns: unit_idx, unit_type, pi, v, rank_pi, rank_v, a_p.
A _catalog table records all run parameters and diagnostics.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import numpy as np
import duckdb

sys.path.insert(0, str(Path(__file__).parent))        # spectral_ranking/
sys.path.insert(0, str(Path(__file__).parent.parent)) # project root for util

from util import load_config, load_params
from build_csr import build_csr
from katz_ranker import rank

_params     = load_params()
TAU_U_FLOOR = _params['tau_u_floor']   # keyed by fx; all currently 20


# ─── Parameter set definition ─────────────────────────────────────────────────

@dataclass
class RunParams:
    tx:    int
    fx:    str
    tau_u: int
    rho:   int     # 0 = fixed count (R̄/R_i); 1 = full count
    m:     tuple   # (m_SS, m_SI, m_IS, m_II)
    chi:   float   # mixing weight (only material for m=(1,1,1,1))
    alpha: float
    label: str = ''


def table_name(p: RunParams) -> str:
    mstr     = ''.join(str(x) for x in p.m)
    chi_int  = round(p.chi * 100)
    alpha_int = round(p.alpha * 100)
    return (f'rk_t{p.tx}_{p.fx}_tau{p.tau_u}'
            f'_rho{p.rho}_m{mstr}_chi{chi_int}_alpha{alpha_int}')


# ─── Parameter schedules ──────────────────────────────────────────────────────

# Baseline parameters
BASELINE = RunParams(
    tx=5, fx='A', tau_u=TAU_U_FLOOR['A'], rho=0,
    m=(0, 1, 1, 0), chi=0.5, alpha=0.85,
    label='baseline',
)

# Stage 1: one-at-a-time deviations from baseline
STAGE1 = [
    BASELINE,
    RunParams(tx=5, fx='A', tau_u=TAU_U_FLOOR['A'], rho=1,
              m=(0, 1, 1, 0), chi=0.5, alpha=0.85,   label='rho1'),
    RunParams(tx=5, fx='A', tau_u=TAU_U_FLOOR['A'], rho=0,
              m=(0, 1, 1, 0), chi=0.5, alpha=0.50,   label='alpha0.5'),
    RunParams(tx=5, fx='E',   tau_u=TAU_U_FLOOR['E'], rho=0,
              m=(0, 1, 1, 0), chi=0.5, alpha=0.85,   label='F=E'),
    RunParams(tx=5, fx='B',   tau_u=TAU_U_FLOOR['B'], rho=0,
              m=(0, 1, 1, 0), chi=0.5, alpha=0.85,   label='F=B'),
    RunParams(tx=5, fx='EB',  tau_u=TAU_U_FLOOR['EB'], rho=0,
              m=(0, 1, 1, 0), chi=0.5, alpha=0.85,   label='F=EB'),
    RunParams(tx=5, fx='NEB', tau_u=TAU_U_FLOOR['NEB'], rho=0,
              m=(0, 1, 1, 0), chi=0.5, alpha=0.85,   label='F=~EB'),
    RunParams(tx=5, fx='A', tau_u=TAU_U_FLOOR['A'], rho=0,
              m=(1, 0, 0, 0), chi=0.5, alpha=0.85,   label='SS-only'),
    RunParams(tx=5, fx='A', tau_u=TAU_U_FLOOR['A'], rho=0,
              m=(0, 0, 0, 1), chi=0.5, alpha=0.85,   label='II-only'),
    RunParams(tx=5, fx='A', tau_u=TAU_U_FLOOR['A'], rho=0,
              m=(1, 1, 1, 1), chi=0.5, alpha=0.85,   label='full-joint'),
    # τ_U sensitivity variant (τ_U=10) — requires el_t5_A_tau10 edge list
    RunParams(tx=5, fx='A', tau_u=10, rho=0,
              m=(0, 1, 1, 0), chi=0.5, alpha=0.85,   label='tau10'),
]

# Stage 2: time-series sweep (baseline parameters, vary tx)
STAGE2 = [
    RunParams(tx=tx, fx='A', tau_u=TAU_U_FLOOR['A'], rho=0,
              m=(0, 1, 1, 0), chi=0.5, alpha=0.85,
              label=f't{tx}')
    for tx in [1, 2, 3, 4, 6]
]


# ─── Database helpers ─────────────────────────────────────────────────────────

def ensure_catalog(db) -> None:
    db.execute("""
        CREATE TABLE IF NOT EXISTS _catalog (
            table_name  VARCHAR PRIMARY KEY,
            tx          INTEGER,
            fx          VARCHAR,
            tau_u       INTEGER,
            rho         INTEGER,
            m_SS        INTEGER,
            m_SI        INTEGER,
            m_IS        INTEGER,
            m_II        INTEGER,
            chi         DOUBLE,
            alpha       DOUBLE,
            n_s         INTEGER,
            n_u         INTEGER,
            iters       INTEGER,
            final_norm  DOUBLE,
            label       VARCHAR,
            created_at  VARCHAR
        )
    """)


def write_result(db, tname: str, p: RunParams, result, csr_data) -> None:
    """Write ranking table and update _catalog."""
    import pandas as pd

    parts = []
    if result.pi_s is not None:
        parts.append(pd.DataFrame({
            'unit_idx':  csr_data.source_ids.astype(np.int64),
            'unit_type': 'S',
            'pi':        result.pi_s.astype(np.float64),
            'v':         result.v_s.astype(np.float64),
            'a_p':       csr_data.a_s.astype(np.float64),
        }))
    if result.pi_u is not None:
        parts.append(pd.DataFrame({
            'unit_idx':  csr_data.inst_ids.astype(np.int64),
            'unit_type': 'U',
            'pi':        result.pi_u.astype(np.float64),
            'v':         result.v_u.astype(np.float64),
            'a_p':       csr_data.a_u.astype(np.float64),
        }))

    if not parts:
        raise RuntimeError(f"No results to write for {tname}")

    df = pd.concat(parts, ignore_index=True)

    # Compute ranks
    df['rank_pi'] = _dense_rank_desc(df['pi'].to_numpy()).astype(np.int32)
    df['rank_v']  = _dense_rank_desc(df['v'].to_numpy()).astype(np.int32)
    df = df[['unit_idx', 'unit_type', 'pi', 'v', 'rank_pi', 'rank_v', 'a_p']]

    # Bulk insert via register — single vectorised copy, no row-by-row overhead
    db.execute(f"DROP TABLE IF EXISTS {tname}")
    db.register('_write_df', df)
    db.execute(f"CREATE TABLE {tname} AS SELECT * FROM _write_df")
    db.unregister('_write_df')

    # Update catalog
    n_s = csr_data.n_s if result.pi_s is not None else 0
    n_u = csr_data.n_u if result.pi_u is not None else 0
    db.execute(
        "INSERT OR REPLACE INTO _catalog VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [tname, p.tx, p.fx, p.tau_u, p.rho,
         p.m[0], p.m[1], p.m[2], p.m[3],
         p.chi, p.alpha,
         n_s, n_u,
         result.iters, result.final_norm,
         p.label,
         datetime.now().isoformat(timespec='seconds')]
    )


def _dense_rank_desc(values: np.ndarray) -> np.ndarray:
    """Return dense rank (1 = highest value). Ties share the same rank."""
    order = np.argsort(-values, kind='stable')
    ranks = np.empty(len(values), dtype=np.int32)
    ranks[order[0]] = 1
    current_rank = 1
    for i in range(1, len(order)):
        if values[order[i]] < values[order[i - 1]]:
            current_rank = i + 1
        ranks[order[i]] = current_rank
    return ranks


# ─── Main driver ──────────────────────────────────────────────────────────────

def run_one(el_db, rk_db, p: RunParams, verbose: bool = True) -> bool:
    """
    Run one parameter combination.  Returns True on success, False if the
    required edge list table is absent (soft skip).
    """
    tname = table_name(p)
    el_tname = f'el_t{p.tx}_{p.fx}_tau{p.tau_u}'
    units_tname = f'_units_t{p.tx}_{p.fx}_tau{p.tau_u}'

    # Check tables exist before committing compute time
    existing = {row[0] for row in el_db.execute("SHOW TABLES").fetchall()}
    if el_tname not in existing:
        if verbose:
            print(f"  SKIP  {tname}  (edge list {el_tname} not found)")
        return False
    if units_tname not in existing:
        if verbose:
            print(f"  SKIP  {tname}  (units table {units_tname} not found)")
        return False

    if verbose:
        print(f"  {tname} [{p.label}] ...", end='  ', flush=True)

    t0 = datetime.now()
    data = build_csr(el_db, p.tx, p.fx, p.tau_u, p.rho, p.m)
    t_csr = (datetime.now() - t0).total_seconds()

    t1 = datetime.now()
    result = rank(data, p.m, p.chi, p.alpha)
    t_rank = (datetime.now() - t1).total_seconds()

    t2 = datetime.now()
    write_result(rk_db, tname, p, result, data)
    t_write = (datetime.now() - t2).total_seconds()

    if verbose:
        iters_str = f"{result.iters} iters" if result.iters else "direct"
        print(f"n_s={data.n_s}, n_u={data.n_u}, {iters_str}, "
              f"norm={result.final_norm:.2e}, "
              f"csr={t_csr:.1f}s  rank={t_rank:.1f}s  write={t_write:.1f}s  total={t_csr+t_rank+t_write:.1f}s")
    return True


def compute_chi_star(el_db, p: RunParams) -> float:
    """
    Compute χ* = N_u / (N_s + N_u) from the units table for RunParams p.
    Provides dimensional balance: equal expected prestige per unit across
    sources and institutions.
    """
    uname = f'_units_t{p.tx}_{p.fx}_tau{p.tau_u}'
    rows = el_db.execute(
        f"SELECT unit_type, COUNT(*) AS n FROM {uname} GROUP BY unit_type"
    ).fetchall()
    counts = {r[0]: r[1] for r in rows}
    n_s = counts.get('S', 0)
    n_u = counts.get('U', 0)
    return n_u / (n_s + n_u)


def clean_stale(rk_db, schedule: list) -> None:
    """Drop ranking tables not in the current schedule."""
    expected = {table_name(p) for p in schedule}
    all_tables = {row[0] for row in rk_db.execute('SHOW TABLES').fetchall()}
    stale = [t for t in all_tables
             if t.startswith('rk_') and t not in expected]

    for t in sorted(stale):
        rk_db.execute(f'DROP TABLE IF EXISTS {t}')
        rk_db.execute("DELETE FROM _catalog WHERE table_name = ?", [t])
        print(f'  Dropped stale table: {t}')

    # Also purge catalog entries with no corresponding table
    orphans = [row[0] for row in rk_db.execute(
        "SELECT table_name FROM _catalog WHERE table_name NOT IN (SELECT name FROM (SHOW TABLES))"
    ).fetchall()]
    for t in sorted(orphans):
        rk_db.execute("DELETE FROM _catalog WHERE table_name = ?", [t])
        print(f'  Removed orphaned catalog entry: {t}')

    if not stale and not orphans:
        print('  No stale tables found.')


def main():
    parser = argparse.ArgumentParser(description='Spectral ranking pipeline')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=1,
                        help='Parameter schedule: 1=baseline+one-at-a-time, 2=time-series')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Run baseline only (fast sanity check)')
    args = parser.parse_args()

    paths = load_config()
    el_path = paths.working / 'edge_lists.duckdb'
    rk_path = paths.working / 'rankings.duckdb'

    if not el_path.exists():
        raise FileNotFoundError(f"edge_lists.duckdb not found at {el_path}. "
                                f"Run prepare_data/build_edge_lists.py first.")

    import time as _time
    _T = {}

    t0 = _time.perf_counter()
    with duckdb.connect(str(el_path), read_only=True) as el_db, \
         duckdb.connect(str(rk_path)) as rk_db:
        _T['open_db'] = _time.perf_counter() - t0

        # Build schedule — χ* requires N_s, N_u from the units table
        t0 = _time.perf_counter()
        if args.baseline_only:
            schedule = [BASELINE]
        elif args.stage == 2:
            schedule = list(STAGE2)
        else:
            chi_star = compute_chi_star(el_db, BASELINE)
            chi_star_run = RunParams(
                tx=BASELINE.tx, fx=BASELINE.fx, tau_u=BASELINE.tau_u, rho=0,
                m=(1, 1, 1, 1), chi=chi_star, alpha=0.85,
                label='full-joint-chi*',
            )
            print(f"χ* = {chi_star:.4f}  →  {table_name(chi_star_run)}")
            schedule = list(STAGE1) + [chi_star_run]
        _T['schedule'] = _time.perf_counter() - t0

        t0 = _time.perf_counter()
        rk_db.execute(f"SET temp_directory = '{paths.working}/.tmp'")
        ensure_catalog(rk_db)
        _T['ensure_catalog'] = _time.perf_counter() - t0

        t0 = _time.perf_counter()
        print('=== Cleaning stale tables ===')
        clean_stale(rk_db, schedule)
        _T['clean_stale'] = _time.perf_counter() - t0

        n_ok = n_skip = 0
        for p in schedule:
            ok = run_one(el_db, rk_db, p)
            if ok:
                n_ok += 1
            else:
                n_skip += 1


    print(f"\nDone: {n_ok} completed, {n_skip} skipped.")
    print("main() overhead (s): " + "  ".join(f"{k}={v:.2f}" for k, v in _T.items()))

    # Summary
    t0 = _time.perf_counter()
    with duckdb.connect(str(rk_path), read_only=True) as db:
        print("\n=== Catalog ===")
        db.sql("SELECT table_name, n_s, n_u, iters, final_norm, label, created_at "
               "FROM _catalog ORDER BY created_at").show()
    print(f"catalog_query={_time.perf_counter()-t0:.2f}s")


if __name__ == '__main__':
    main()
