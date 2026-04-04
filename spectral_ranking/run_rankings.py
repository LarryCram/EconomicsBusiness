"""
run_rankings.py — Parameter driver for spectral ranking pipeline.

Reads: WORKING/edge_lists.duckdb (edge lists + unit index tables)
Writes: WORKING/rankings.duckdb  (ranking tables + _catalog)

Usage
-----
  python spectral_ranking/run_rankings.py              # all non-skipped rows
  python spectral_ranking/run_rankings.py --baseline-only

Run schedule
------------
Driven by params.csv (project root).  Each non-skipped row is one run.
--baseline-only selects label='baseline' for a fast sanity check.

Output table naming
-------------------
  rk_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_rho{rho}_m{mstr}_chi{chi_str}_alpha{alpha_int}

  run_code  8-char string: last-2-digits of tc0,tc1,tt0,tt1  e.g. '20242024'
  chi_str   integer percentage for fixed chi, 'STAR' for chi=-1 (resolved at runtime)

  e.g. rk_20242024_A_tauU20_tauS20_rho0_m0110_chi50_alpha100   (baseline)
       rk_20242024_A_tauU20_tauS20_rho0_m1111_chiSTAR_alpha100 (chi_star)

Each table has columns: unit_idx, unit_type, pi, v, rank_pi, rank_v, a_p.
A _catalog table records all run parameters and diagnostics.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import duckdb

sys.path.insert(0, str(Path(__file__).parent))        # spectral_ranking/
sys.path.insert(0, str(Path(__file__).parent.parent)) # project root for util

from util import load_config, load_runs
from build_csr import build_csr
from katz_ranker import rank


# ─── Parameter set definition ─────────────────────────────────────────────────

@dataclass
class RunParams:
    run_code: str    # 8-char time window key, e.g. '20242024'
    fx:       str
    tau_u:    int
    tau_s:    int
    rho:      int    # 0 = fixed count (R̄/R_i); 1 = full count
    m:        tuple  # (m_SS, m_SI, m_IS, m_II)
    chi:      float  # -1.0 = chi_star (resolved at runtime)
    alpha:    float
    label:    str = ''


def table_name(p: RunParams) -> str:
    mstr      = ''.join(str(x) for x in p.m)
    chi_str   = 'STAR' if p.chi == -1.0 else str(round(p.chi * 100))
    alpha_int = round(p.alpha * 100)
    return (f'rk_{p.run_code}_{p.fx}_tauU{p.tau_u}_tauS{p.tau_s}'
            f'_rho{p.rho}_m{mstr}_chi{chi_str}_alpha{alpha_int}')


# ─── Load run schedule from CSV ───────────────────────────────────────────────

def runs_from_csv() -> list:
    """Return list of RunParams from all non-skipped rows in params.csv."""
    rows = load_runs()
    result = []
    for r in rows:
        m = tuple(int(c) for c in r['m'])   # '0110' → (0,1,1,0)
        result.append(RunParams(
            run_code=r['run_code'],
            fx=r['fx'],
            tau_u=r['tau_u'],
            tau_s=r['tau_s'],
            rho=r['rho'],
            m=m,
            chi=r['chi'],
            alpha=r['alpha'],
            label=r['label'],
        ))
    return result


# ─── Database helpers ─────────────────────────────────────────────────────────

def ensure_catalog(db) -> None:
    db.execute("""
        CREATE TABLE IF NOT EXISTS _catalog (
            table_name  VARCHAR PRIMARY KEY,
            run_code    VARCHAR,
            fx          VARCHAR,
            tau_u       INTEGER,
            tau_s       INTEGER,
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
    # Migrate pre-run_code schema
    cols = {row[0] for row in db.execute("DESCRIBE _catalog").fetchall()}
    if 'run_code' not in cols:
        db.execute("ALTER TABLE _catalog ADD COLUMN run_code VARCHAR")
    if 'tau_s' not in cols:
        db.execute("ALTER TABLE _catalog ADD COLUMN tau_s INTEGER")
        db.execute("UPDATE _catalog SET tau_s = 0")


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
    df['rank_pi'] = _dense_rank_desc(df['pi'].to_numpy()).astype(np.int32)
    df['rank_v']  = _dense_rank_desc(df['v'].to_numpy()).astype(np.int32)
    df = df[['unit_idx', 'unit_type', 'pi', 'v', 'rank_pi', 'rank_v', 'a_p']]

    db.execute(f"DROP TABLE IF EXISTS {tname}")
    db.register('_write_df', df)
    db.execute(f"CREATE TABLE {tname} AS SELECT * FROM _write_df")
    db.unregister('_write_df')

    n_s = csr_data.n_s if result.pi_s is not None else 0
    n_u = csr_data.n_u if result.pi_u is not None else 0
    db.execute(
        """INSERT OR REPLACE INTO _catalog
           (table_name, run_code, fx, tau_u, tau_s, rho,
            m_SS, m_SI, m_IS, m_II, chi, alpha,
            n_s, n_u, iters, final_norm, label, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [tname, p.run_code, p.fx, p.tau_u, p.tau_s, p.rho,
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
            current_rank += 1
        ranks[order[i]] = current_rank
    return ranks


# ─── Main driver ──────────────────────────────────────────────────────────────

def compute_chi_star(el_db, p: RunParams) -> float:
    """Compute χ* = N_u / (N_s + N_u) from the units table for RunParams p."""
    uname = f'_units_{p.run_code}_{p.fx}_tauU{p.tau_u}_tauS{p.tau_s}'
    rows = el_db.execute(
        f"SELECT unit_type, COUNT(*) AS n FROM {uname} GROUP BY unit_type"
    ).fetchall()
    counts = {r[0]: r[1] for r in rows}
    n_s = counts.get('S', 0)
    n_u = counts.get('U', 0)
    return n_u / (n_s + n_u)


def run_one(el_db, rk_db, p: RunParams, verbose: bool = True) -> bool:
    """
    Run one parameter combination.  Returns True on success, False if the
    required edge list table is absent (soft skip).
    """
    el_tname    = f'el_{p.run_code}_{p.fx}_tauU{p.tau_u}_tauS{p.tau_s}'
    units_tname = f'_units_{p.run_code}_{p.fx}_tauU{p.tau_u}_tauS{p.tau_s}'
    tname       = table_name(p)

    existing = {row[0] for row in el_db.execute("SHOW TABLES").fetchall()}
    if el_tname not in existing:
        if verbose:
            print(f"  SKIP  {tname}  (edge list {el_tname} not found)")
        return False
    if units_tname not in existing:
        if verbose:
            print(f"  SKIP  {tname}  (units table {units_tname} not found)")
        return False

    # Resolve chi_star at runtime; p.chi stays -1.0 for table naming
    chi_resolved = compute_chi_star(el_db, p) if p.chi == -1.0 else p.chi

    if verbose:
        chi_disp = f'{chi_resolved:.4f}' if p.chi == -1.0 else str(p.chi)
        print(f"  {tname} [{p.label}] chi={chi_disp} ...", end='  ', flush=True)

    t0 = datetime.now()
    data = build_csr(el_db, p.run_code, p.fx, p.tau_u, p.tau_s, p.rho, p.m)
    t_csr = (datetime.now() - t0).total_seconds()

    t1 = datetime.now()
    result = rank(data, p.m, chi_resolved, p.alpha)
    t_rank = (datetime.now() - t1).total_seconds()

    t2 = datetime.now()
    write_result(rk_db, tname, p, result, data)
    t_write = (datetime.now() - t2).total_seconds()

    if verbose:
        iters_str = f"{result.iters} iters" if result.iters else "direct"
        print(f"n_s={data.n_s}, n_u={data.n_u}, {iters_str}, "
              f"norm={result.final_norm:.2e}, "
              f"csr={t_csr:.1f}s  rank={t_rank:.1f}s  write={t_write:.1f}s  "
              f"total={t_csr+t_rank+t_write:.1f}s")
    return True


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

    orphans = [row[0] for row in rk_db.execute(
        "SELECT table_name FROM _catalog "
        "WHERE table_name NOT IN (SELECT name FROM (SHOW TABLES))"
    ).fetchall()]
    for t in sorted(orphans):
        rk_db.execute("DELETE FROM _catalog WHERE table_name = ?", [t])
        print(f'  Removed orphaned catalog entry: {t}')

    if not stale and not orphans:
        print('  No stale tables found.')


def main():
    parser = argparse.ArgumentParser(description='Spectral ranking pipeline')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Run baseline only (fast sanity check)')
    args = parser.parse_args()

    paths = load_config()
    el_path = paths.working / 'edge_lists.duckdb'
    rk_path = paths.working / 'rankings.duckdb'

    if not el_path.exists():
        raise FileNotFoundError(
            f"edge_lists.duckdb not found at {el_path}. "
            f"Run prepare_data/build_edge_lists.py first."
        )

    import time as _time
    _T = {}

    t0 = _time.perf_counter()
    with duckdb.connect(str(el_path), read_only=True) as el_db, \
         duckdb.connect(str(rk_path)) as rk_db:
        _T['open_db'] = _time.perf_counter() - t0

        t0 = _time.perf_counter()
        if args.baseline_only:
            schedule = [p for p in runs_from_csv() if p.label == 'baseline']
        else:
            schedule = runs_from_csv()
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

    t0 = _time.perf_counter()
    with duckdb.connect(str(rk_path), read_only=True) as db:
        print("\n=== Catalog ===")
        db.sql("SELECT table_name, n_s, n_u, iters, final_norm, label, created_at "
               "FROM _catalog ORDER BY created_at").show()
    print(f"catalog_query={_time.perf_counter()-t0:.2f}s")


if __name__ == '__main__':
    main()
