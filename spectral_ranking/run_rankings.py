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
    run_code:  str    # 8-char time window key, e.g. '20242024'
    fx:        str
    tau_u:     int
    tau_s:     int
    rho:         int    # 0 = fixed count (R̄/R_i); 1 = full count
    m:           tuple  # (m_SS, m_SI, m_IS, m_II)
    chi:         float  # -1.0 = chi_star (resolved at runtime)
    alpha:       float
    mu_type:     str  = ''    # '' = Perron (alpha=1); 'uniform' or 'unit_scaled'
    label:       str  = ''
    ref_units:   str  = ''    # non-empty → fixtau corpus
    direct_inst: bool = False  # False = author-fractional ω; True = 1/N_inst ω


_MU_SUFFIX = {
    '':           '',
    'uniform':    '_muUniform',
    'unit_scaled': '_muUnitScaled',
}


def table_name(p: RunParams) -> str:
    tau_sfx   = '_fixtau' if p.ref_units else '_vartau'
    mstr      = ''.join(str(x) for x in p.m)
    chi_str   = 'STAR' if p.chi == -1.0 else str(round(p.chi * 100))
    alpha_int = round(p.alpha * 100)
    mu_sfx    = _MU_SUFFIX.get(p.mu_type, f'_mu{p.mu_type}')
    omega_sfx = '_omega1' if p.direct_inst else ''
    return (f'rk_{p.run_code}_{p.fx}_tauU{p.tau_u}_tauS{p.tau_s}{tau_sfx}'
            f'_rho{p.rho}_m{mstr}_chi{chi_str}_alpha{alpha_int}{mu_sfx}{omega_sfx}')


def _make_mu(mu_type: str, n_s: int, n_u: int) -> 'np.ndarray | None':
    """
    Construct prior vector for Katz–Hubbell iteration.

    'uniform'     — μ_p = 1/(N_S+N_U) for all units  (Katz uniform prior)
    'unit_scaled' — μ_p = 1/N_S for sources, 1/N_U for institutions
    ''            — None (Perron / alpha=1 path)
    """
    if not mu_type:
        return None
    N = n_s + n_u
    if mu_type == 'uniform':
        return np.full(N, 1.0 / N, dtype=np.float64)
    elif mu_type == 'unit_scaled':
        return np.concatenate([
            np.full(n_s, 1.0 / n_s, dtype=np.float64),
            np.full(n_u, 1.0 / n_u, dtype=np.float64),
        ])
    else:
        raise ValueError(
            f"Unknown mu_type {mu_type!r}. Expected 'uniform', 'unit_scaled', or ''."
        )


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
            mu_type=r.get('mu_type', ''),
            label=r['label'],
            ref_units=r.get('ref_units', ''),
            direct_inst=bool(int(r.get('omega', 0))),
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
            mu_type     VARCHAR,
            n_s         INTEGER,
            n_u         INTEGER,
            lam1        DOUBLE,
            lam2        DOUBLE,
            iters       INTEGER,
            final_norm  DOUBLE,
            label       VARCHAR,
            created_at  VARCHAR
        )
    """)
    # Migrate older schemas
    cols = {row[0] for row in db.execute("DESCRIBE _catalog").fetchall()}
    if 'run_code' not in cols:
        db.execute("ALTER TABLE _catalog ADD COLUMN run_code VARCHAR")
    if 'tau_s' not in cols:
        db.execute("ALTER TABLE _catalog ADD COLUMN tau_s INTEGER")
        db.execute("UPDATE _catalog SET tau_s = 0")
    if 'lam1' not in cols:
        db.execute("ALTER TABLE _catalog ADD COLUMN lam1 DOUBLE")
        db.execute("UPDATE _catalog SET lam1 = 0.0")
    if 'lam2' not in cols:
        db.execute("ALTER TABLE _catalog ADD COLUMN lam2 DOUBLE")
        db.execute("UPDATE _catalog SET lam2 = 0.0")
    if 'mu_type' not in cols:
        db.execute("ALTER TABLE _catalog ADD COLUMN mu_type VARCHAR")
        db.execute("UPDATE _catalog SET mu_type = ''")
    if 'omega' not in cols:
        db.execute("ALTER TABLE _catalog ADD COLUMN omega INTEGER")
        db.execute("UPDATE _catalog SET omega = 0")


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
           (table_name, run_code, fx, tau_u, tau_s, rho, omega,
            m_SS, m_SI, m_IS, m_II, chi, alpha, mu_type,
            n_s, n_u, lam1, lam2, iters, final_norm, label, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [tname, p.run_code, p.fx, p.tau_u, p.tau_s, p.rho, int(p.direct_inst),
         p.m[0], p.m[1], p.m[2], p.m[3],
         p.chi, p.alpha, p.mu_type,
         n_s, n_u,
         result.lam1, result.lam2,
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
    """Compute χ* = N_u / (N_s + N_u) from the mode-specific units table."""
    tau_sfx = '_fixtau' if p.ref_units else '_vartau'
    mstr  = ''.join(str(x) for x in p.m)
    uname = f'_units_{p.run_code}_{p.fx}_tauU{p.tau_u}_tauS{p.tau_s}{tau_sfx}_m{mstr}'
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
    tau_sfx     = '_fixtau' if p.ref_units else '_vartau'
    mstr        = ''.join(str(x) for x in p.m)
    el_tname    = f'el_{p.run_code}_{p.fx}_tauU{p.tau_u}_tauS{p.tau_s}{tau_sfx}'
    units_tname = f'_units_{p.run_code}_{p.fx}_tauU{p.tau_u}_tauS{p.tau_s}{tau_sfx}_m{mstr}'
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
    data = build_csr(el_db, p.run_code, p.fx, p.tau_u, p.tau_s, p.rho, p.m, p.ref_units,
                     direct_inst=p.direct_inst)
    t_csr = (datetime.now() - t0).total_seconds()

    mu = _make_mu(p.mu_type, data.n_s, data.n_u)

    t1 = datetime.now()
    result = rank(data, p.m, chi_resolved, p.alpha, mu=mu)
    t_rank = (datetime.now() - t1).total_seconds()

    t2 = datetime.now()
    write_result(rk_db, tname, p, result, data)
    t_write = (datetime.now() - t2).total_seconds()

    if verbose:
        if result.lam1 != 0.0:
            algo_str = (f"lam1={result.lam1:.6f}  lam2={result.lam2:.6f}  "
                        f"gap={1.0 - abs(result.lam2):.2e}")
        else:
            algo_str = f"{result.iters} iters  norm={result.final_norm:.2e}"
        print(f"n_s={data.n_s}, n_u={data.n_u},  {algo_str},  "
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
        db.sql("SELECT table_name, n_s, n_u, lam1, lam2, iters, final_norm, label, created_at "
               "FROM _catalog ORDER BY created_at").show()
    print(f"catalog_query={_time.perf_counter()-t0:.2f}s")


if __name__ == '__main__':
    main()
