"""
filter_mode_units.py — Compute mode-specific SCC unit tables.

For each (run_code, fx, tau_u, tau_s) corpus and each mode m appearing
in params.csv, finds the giant SCC of the mode-relevant subgraph and writes

  _units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_m{mstr}

to edge_lists.duckdb.  Must be run after build_edge_lists.py.

The mode-specific SCC is always a subset of the C_full SCC already stored in
_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}.

Mode → blocks used for SCC
---------------------------
  m=1000  (SS only)     : C_SS  on sources only
  m=0001  (II only)     : C_II  on institutions only
  m=0110  (bipartite)   : C_SI, C_IS  on sources + institutions
  m=1111  (full joint)  : all four blocks on sources + institutions

The resulting _units_..._m{mstr} tables are what build_csr.py and
bootstrap_baseline.py read in order to guarantee a mode-primitive unit set.

Usage
-----
  python prepare_data/filter_mode_units.py
  python prepare_data/filter_mode_units.py --dry-run   # report only, no writes
"""

import sys
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import duckdb
from scipy.sparse import csr_matrix, bmat as sp_bmat
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

paths   = load_config()
DB_PATH = paths.working / 'edge_lists.duckdb'


# ─── Naming ───────────────────────────────────────────────────────────────────

def mode_str(m: tuple) -> str:
    """(0,1,1,0) → '0110'"""
    return ''.join(str(x) for x in m)


def raw_units_table(run_code: str, fx: str, tau_u: int, tau_s: int) -> str:
    """Name of the C_full SCC-filtered units table from build_edge_lists.py."""
    return f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}'


def mode_units_table(run_code: str, fx: str, tau_u: int, tau_s: int,
                     m: tuple) -> str:
    """Name of the mode-specific SCC units table written by this script."""
    return f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_m{mode_str(m)}'


# ─── Block builder ────────────────────────────────────────────────────────────

def _build_connectivity_block(db, sql: str,
                               row_ix: pd.Index, col_ix: pd.Index,
                               shape: tuple) -> csr_matrix:
    """
    Build a boolean (0/1) CSR matrix from a SQL query that returns two
    columns (row_id, col_id).  IDs not in row_ix or col_ix are ignored.
    """
    df = db.execute(sql).fetchdf()
    if df.empty:
        return csr_matrix(shape)
    r = row_ix.get_indexer(df.iloc[:, 0].to_numpy(dtype=np.int64))
    c = col_ix.get_indexer(df.iloc[:, 1].to_numpy(dtype=np.int64))
    mask = (r >= 0) & (c >= 0)
    if not mask.any():
        return csr_matrix(shape)
    ones = np.ones(mask.sum(), dtype=np.float64)
    return csr_matrix((ones, (r[mask], c[mask])), shape=shape)


# ─── Core SCC computation ─────────────────────────────────────────────────────

def compute_mode_scc(db, run_code: str, fx: str, tau_u: int, tau_s: int,
                     m: tuple) -> tuple:
    """
    Iteratively find the giant SCC of the mode-specific graph starting from
    the C_full SCC unit set already stored in the raw _units_ table.

    Parameters
    ----------
    db        : open duckdb connection (read-only is fine).
    run_code, fx, tau_u, tau_s : corpus identifiers.
    m         : (m_SS, m_SI, m_IS, m_II) block mask.

    Returns
    -------
    src_ids  : int64 array of surviving source_idx values.
    inst_ids : int64 array of surviving institution_idx values.
               Empty for m=1000 (SS-only) and m=0001 (II-only).
    a_s      : float64 work counts for src_ids (from raw _units_ table).
    a_u      : float64 work counts for inst_ids.
    n_dropped_s  : total sources dropped.
    n_dropped_u  : total institutions dropped.
    """
    m_SS, m_SI, m_IS, m_II = m
    tname = f'el_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}'
    uname = raw_units_table(run_code, fx, tau_u, tau_s)

    # Load initial unit set from the C_full SCC table
    units = db.execute(
        f"SELECT unit_idx, unit_type, a_p FROM {uname} ORDER BY unit_type, unit_idx"
    ).fetchdf()

    src_df  = units[units['unit_type'] == 'S'].reset_index(drop=True)
    inst_df = units[units['unit_type'] == 'U'].reset_index(drop=True)

    src_ids  = src_df['unit_idx'].to_numpy(dtype=np.int64)
    inst_ids = inst_df['unit_idx'].to_numpy(dtype=np.int64)
    a_s_map  = dict(zip(src_df['unit_idx'].astype(int), src_df['a_p']))
    a_u_map  = dict(zip(inst_df['unit_idx'].astype(int), inst_df['a_p']))

    # For institution-only mode, discard sources from initial set
    if m == (0, 0, 0, 1):
        src_ids = np.array([], dtype=np.int64)

    # For source-only mode, discard institutions from initial set
    if m == (1, 0, 0, 0):
        inst_ids = np.array([], dtype=np.int64)

    needs_inst = bool(m_SI or m_IS or m_II)

    total_dropped_s = 0
    total_dropped_u = 0

    for iteration in range(20):
        n_s = len(src_ids)
        n_u = len(inst_ids)

        if n_s == 0 and not needs_inst:
            break
        if needs_inst and n_s == 0 and n_u == 0:
            break

        src_index  = pd.Index(src_ids)
        inst_index = pd.Index(inst_ids)

        # Build mode-relevant blocks (connectivity only — no rho weighting)
        C_SS = _build_connectivity_block(
            db,
            f"SELECT DISTINCT citer_source_idx, cited_source_idx FROM {tname}",
            src_index, src_index, (n_s, n_s),
        ) if m_SS and n_s > 0 else csr_matrix((n_s, n_s))

        C_SI = _build_connectivity_block(
            db,
            f"SELECT DISTINCT citer_source_idx, cited_inst_idx FROM {tname}",
            src_index, inst_index, (n_s, n_u),
        ) if m_SI and n_s > 0 and n_u > 0 else csr_matrix((n_s, n_u))

        C_IS = _build_connectivity_block(
            db,
            f"SELECT DISTINCT citer_inst_idx, cited_source_idx FROM {tname}",
            inst_index, src_index, (n_u, n_s),
        ) if m_IS and n_u > 0 and n_s > 0 else csr_matrix((n_u, n_s))

        C_II = _build_connectivity_block(
            db,
            f"SELECT DISTINCT citer_inst_idx, cited_inst_idx FROM {tname}",
            inst_index, inst_index, (n_u, n_u),
        ) if m_II and n_u > 0 else csr_matrix((n_u, n_u))

        # Assemble combined matrix for SCC
        if needs_inst and n_u > 0:
            C_combined = sp_bmat([[C_SS, C_SI], [C_IS, C_II]], format='csr')
            _, labels = connected_components(C_combined, directed=True, connection='strong')
            giant = Counter(labels).most_common(1)[0][0]
            drop_src  = src_ids[labels[:n_s] != giant]
            drop_inst = inst_ids[labels[n_s:] != giant]
        else:
            # Source-only or institution-only
            C_combined = C_SS if n_s > 0 else C_II
            if C_combined.shape[0] == 0:
                break
            _, labels = connected_components(C_combined, directed=True, connection='strong')
            giant = Counter(labels).most_common(1)[0][0]
            if n_s > 0:
                drop_src  = src_ids[labels != giant]
                drop_inst = np.array([], dtype=np.int64)
            else:
                drop_src  = np.array([], dtype=np.int64)
                drop_inst = inst_ids[labels != giant]

        if len(drop_src) == 0 and len(drop_inst) == 0:
            print(f'      stable after {iteration} pass(es)', flush=True)
            break

        print(f'      pass {iteration + 1}: '
              f'drop {len(drop_src)} sources, {len(drop_inst)} insts',
              flush=True)
        total_dropped_s += len(drop_src)
        total_dropped_u += len(drop_inst)

        src_ids  = np.setdiff1d(src_ids,  drop_src)
        inst_ids = np.setdiff1d(inst_ids, drop_inst)

    a_s = np.array([a_s_map[int(i)] for i in src_ids],  dtype=np.float64)
    a_u = np.array([a_u_map[int(i)] for i in inst_ids], dtype=np.float64)

    return src_ids, inst_ids, a_s, a_u, total_dropped_s, total_dropped_u


# ─── Write helper ─────────────────────────────────────────────────────────────

def write_mode_units(db, run_code: str, fx: str, tau_u: int, tau_s: int,
                     m: tuple, src_ids: np.ndarray, inst_ids: np.ndarray,
                     a_s: np.ndarray, a_u: np.ndarray) -> None:
    """
    Write surviving units to _units_..._m{mstr} table in edge_lists.duckdb.
    Overwrites any existing table of the same name.
    """
    tname = mode_units_table(run_code, fx, tau_u, tau_s, m)

    rows = (
        [{'unit_idx': int(idx), 'unit_type': 'S', 'a_p': float(ap)}
         for idx, ap in zip(src_ids, a_s)]
        +
        [{'unit_idx': int(idx), 'unit_type': 'U', 'a_p': float(ap)}
         for idx, ap in zip(inst_ids, a_u)]
    )
    df = (pd.DataFrame(rows) if rows
          else pd.DataFrame(columns=['unit_idx', 'unit_type', 'a_p']))
    df['unit_idx'] = df['unit_idx'].astype(np.int64)
    df['a_p']      = df['a_p'].astype(np.float64)

    db.execute(f'DROP TABLE IF EXISTS {tname}')
    db.register('_fmu_write', df)
    db.execute(f'CREATE TABLE {tname} AS SELECT * FROM _fmu_write')
    db.unregister('_fmu_write')


# ─── Per-corpus driver ────────────────────────────────────────────────────────

def process_corpus(db, run_code: str, fx: str, tau_u: int, tau_s: int,
                   modes: list, dry_run: bool = False) -> None:
    """
    Compute and write mode-specific unit tables for all modes in `modes`
    for one corpus (run_code, fx, tau_u, tau_s).
    """
    uname  = raw_units_table(run_code, fx, tau_u, tau_s)
    tname  = f'el_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}'
    tables = {r[0] for r in db.execute('SHOW TABLES').fetchall()}

    if tname not in tables:
        print(f'  SKIP {tname} — edge list not found', flush=True)
        return
    if uname not in tables:
        print(f'  SKIP {uname} — raw units table not found '
              f'(run build_edge_lists.py first)', flush=True)
        return

    n_raw = db.execute(f"SELECT COUNT(*) FROM {uname}").fetchone()[0]

    for m in modes:
        mstr   = mode_str(m)
        out_tn = mode_units_table(run_code, fx, tau_u, tau_s, m)
        print(f'  {out_tn} ...', end='  ', flush=True)

        src_ids, inst_ids, a_s, a_u, n_ds, n_du = compute_mode_scc(
            db, run_code, fx, tau_u, tau_s, m
        )
        n_out = len(src_ids) + len(inst_ids)
        print(f'n_s={len(src_ids):,}  n_u={len(inst_ids):,}  '
              f'(dropped {n_ds} src, {n_du} inst from {n_raw} raw)',
              flush=True)

        if not dry_run:
            write_mode_units(db, run_code, fx, tau_u, tau_s, m,
                             src_ids, inst_ids, a_s, a_u)


# ─── Stale-table cleanup ──────────────────────────────────────────────────────

def clean_stale_mode_units(db, expected_tables: set) -> None:
    """
    Drop _units_..._m{mstr} tables that are no longer in the expected set.
    Does not touch _units_ tables without a _m suffix (those belong to
    build_edge_lists.py).
    """
    import re
    pattern = re.compile(r'^_units_.*_m[01]{4}$')
    all_tables = {r[0] for r in db.execute('SHOW TABLES').fetchall()}
    stale = [t for t in all_tables
             if pattern.match(t) and t not in expected_tables]
    for t in sorted(stale):
        db.execute(f'DROP TABLE IF EXISTS {t}')
        print(f'  Dropped stale mode-units table: {t}', flush=True)
    if not stale:
        print('  No stale mode-units tables.', flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute mode-specific SCC unit tables for all runs in params.csv.'
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Report results without writing to the database.')
    args = parser.parse_args()

    rows = load_runs()

    # Collect unique (run_code, fx, tau_u, tau_s) → set of modes
    corpus_modes: dict = {}
    for r in rows:
        m = tuple(int(c) for c in r['m'])
        key = (r['run_code'], r['fx'], r['tau_u'], r['tau_s'])
        corpus_modes.setdefault(key, set()).add(m)

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f'edge_lists.duckdb not found at {DB_PATH}. '
            'Run prepare_data/build_edge_lists.py first.'
        )

    expected_tables: set = set()
    for (run_code, fx, tau_u, tau_s), modes in corpus_modes.items():
        for m in modes:
            expected_tables.add(mode_units_table(run_code, fx, tau_u, tau_s, m))

    with duckdb.connect(str(DB_PATH), read_only=args.dry_run) as db:
        print('=== Cleaning stale mode-units tables ===')
        if not args.dry_run:
            clean_stale_mode_units(db, expected_tables)
        else:
            print('  (dry-run — skipping cleanup)')

        print('=== Building mode-specific unit tables ===')
        for (run_code, fx, tau_u, tau_s), modes in sorted(corpus_modes.items()):
            print(f'\n  corpus: run_code={run_code}  fx={fx}  '
                  f'tau_u={tau_u}  tau_s={tau_s}', flush=True)
            process_corpus(db, run_code, fx, tau_u, tau_s,
                           sorted(modes), dry_run=args.dry_run)

    print('\nDone.')


if __name__ == '__main__':
    main()
    print('FINISHED!')
