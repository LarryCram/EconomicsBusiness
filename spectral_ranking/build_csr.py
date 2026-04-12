"""
build_csr.py — Build raw CSR blocks from a pre-built edge list.

Reads one edge list table (el_t{tx}_{fx}_tauU{tau_u}_tauS{tau_s}) and the
corresponding unit index table (_units_t{tx}_{fx}_tauU{tau_u}_tauS{tau_s})
from edge_lists.duckdb.
Applies ρ weighting, builds scipy CSR matrices for the requested blocks,
and returns a CSRData object.

No matrix assembly, no row-normalisation, no algorithm logic — all of that
lives in katz_ranker.py.

The edge list and _units tables must already exist (run
prepare_data/build_edge_lists.py first).

Public API
----------
build_csr(db, run_code, fx, tau_u, tau_s, rho, m) -> CSRData
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from dataclasses import dataclass
from typing import Optional


@dataclass
class CSRData:
    """
    Raw (un-normalised) CSR block matrices and unit metadata for one corpus.

    Blocks that were not requested (corresponding m bit = 0) are None.
    Matrices use dense indices 0…n_s-1 (sources) and 0…n_u-1 (institutions).
    """
    C_SS: Optional[sp.csr_matrix]   # (n_s × n_s) source–source
    C_SI: Optional[sp.csr_matrix]   # (n_s × n_u) source–institution
    C_IS: Optional[sp.csr_matrix]   # (n_u × n_s) institution–source
    C_II: Optional[sp.csr_matrix]   # (n_u × n_u) institution–institution

    source_ids: np.ndarray   # original source_idx values, length n_s
    inst_ids:   np.ndarray   # original institution_idx values, length n_u
    a_s: np.ndarray          # integer work count per source,       length n_s
    a_u: np.ndarray          # fractional work count per institution, length n_u

    n_s: int
    n_u: int


def build_csr(db, run_code: str, fx: str, tau_u: int, tau_s: int, rho: int, m: tuple) -> CSRData:
    """
    Build raw CSR blocks for corpus (run_code, fx, tau_u, tau_s) with ρ weighting.

    Parameters
    ----------
    db : duckdb connection (open, writeable not required).
    run_code : 8-char time window key, e.g. '20242024'.
    fx : field subset ('E', 'B', 'A', 'EB', 'NEB').
    tau_u : institution retention threshold (mean annual census works).
    tau_s : source retention threshold (mean annual census works).
    rho : 0 → fixed count (ρ_i = R̄/R_i); 1 → full count (ρ_i = 1).
    m : (m_SS, m_SI, m_IS, m_II) ∈ {0,1}^4 — which blocks to build.

    Returns
    -------
    CSRData with requested blocks populated; others None.
    """
    import time
    _t = {}   # timing dict — remove when no longer needed

    mstr   = ''.join(str(x) for x in m)
    tname  = f'el_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}'
    uname  = f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_m{mstr}'

    # ── Load unit index ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        units_df = db.execute(
            f"SELECT unit_idx, unit_type, a_p FROM {uname} ORDER BY unit_type, unit_idx"
        ).fetchdf()
    except Exception as e:
        raise RuntimeError(
            f"Mode-specific unit table '{uname}' not found in edge_lists.duckdb. "
            f"Run prepare_data/filter_mode_units.py first."
        ) from e

    src_df  = units_df[units_df['unit_type'] == 'S'].reset_index(drop=True)
    inst_df = units_df[units_df['unit_type'] == 'U'].reset_index(drop=True)

    source_ids = src_df['unit_idx'].to_numpy(dtype=np.int64)
    inst_ids   = inst_df['unit_idx'].to_numpy(dtype=np.int64)
    a_s = src_df['a_p'].to_numpy(dtype=np.float64)
    a_u = inst_df['a_p'].to_numpy(dtype=np.float64)
    n_s = len(source_ids)
    n_u = len(inst_ids)

    # Vectorised index maps — pd.Index.get_indexer is a single hash-table
    # pass over the array; avoids the intermediate Series allocation of .loc[]
    src_idx  = pd.Index(source_ids)
    inst_idx = pd.Index(inst_ids)
    _t['units'] = time.perf_counter() - t0

    # ── Materialise slim, ρ-weighted temp table (one scan of {tname}) ──────
    # Drops the five a_* / direct_inst_weight columns not needed here.
    # All block queries below read _tmp_el from memory rather than
    # re-scanning the on-disk edge-list table each time.
    t0 = time.perf_counter()
    if rho == 0:
        r_bar = db.execute(
            f"SELECT AVG(rval) FROM "
            f"(SELECT DISTINCT citer_work_idx, CAST(R_i AS DOUBLE) AS rval FROM {tname})"
        ).fetchone()[0]
        if r_bar is None:
            raise RuntimeError(
                f"Edge list table '{tname}' is empty — cannot compute R̄ for ρ=0 weighting."
            )
        rho_col = f"{r_bar} / CAST(R_i AS DOUBLE)"
    else:
        rho_col = "1.0"

    db.execute(f"""
        CREATE OR REPLACE TEMP TABLE _tmp_el AS
        SELECT citer_work_idx, citer_source_idx, citer_inst_idx,
               cited_work_idx,  cited_source_idx, cited_inst_idx,
               inst_weight, cited_inst_weight,
               {rho_col} AS rho_w
        FROM {tname}
    """)
    _t['tmp_el'] = time.perf_counter() - t0

    # ── Block builders ─────────────────────────────────────────────────────

    def _build_ss() -> sp.csr_matrix:
        """
        C_SS: de-duplicate on (citer_work, cited_work) to count each
        reference once regardless of how many institution combinations exist.
        """
        df = db.execute("""
            SELECT citer_source_idx, cited_source_idx, SUM(rho_w) AS weight
            FROM (
                SELECT DISTINCT citer_work_idx, citer_source_idx,
                                cited_work_idx,  cited_source_idx, rho_w
                FROM _tmp_el
            )
            GROUP BY citer_source_idx, cited_source_idx
        """).fetchdf()
        rows = src_idx.get_indexer(df['citer_source_idx'].to_numpy())
        cols = src_idx.get_indexer(df['cited_source_idx'].to_numpy())
        mask = (rows >= 0) & (cols >= 0)
        return sp.coo_matrix(
            (df['weight'].to_numpy(dtype=np.float64)[mask], (rows[mask], cols[mask])),
            shape=(n_s, n_s)
        ).tocsr()

    def _build_si() -> sp.csr_matrix:
        """
        C_SI: de-duplicate over citer_inst so each (citer_work, cited_work,
        cited_inst) contributes ρ_i × ω_jv exactly once.
        """
        df = db.execute("""
            SELECT citer_source_idx, cited_inst_idx,
                   SUM(rho_w * cited_inst_weight) AS weight
            FROM (
                SELECT DISTINCT citer_work_idx, citer_source_idx,
                                cited_work_idx,  cited_inst_idx,
                                rho_w, cited_inst_weight
                FROM _tmp_el
            )
            GROUP BY citer_source_idx, cited_inst_idx
        """).fetchdf()
        rows = src_idx.get_indexer(df['citer_source_idx'].to_numpy())
        cols = inst_idx.get_indexer(df['cited_inst_idx'].to_numpy())
        mask = (rows >= 0) & (cols >= 0)
        return sp.coo_matrix(
            (df['weight'].to_numpy(dtype=np.float64)[mask], (rows[mask], cols[mask])),
            shape=(n_s, n_u)
        ).tocsr()

    def _build_is() -> sp.csr_matrix:
        """
        C_IS: de-duplicate over cited_inst so each (citer_work, citer_inst,
        cited_work) contributes ρ_i × ω_iu exactly once.
        """
        df = db.execute("""
            SELECT citer_inst_idx, cited_source_idx,
                   SUM(rho_w * inst_weight) AS weight
            FROM (
                SELECT DISTINCT citer_work_idx, citer_inst_idx,
                                cited_work_idx,  cited_source_idx,
                                rho_w, inst_weight
                FROM _tmp_el
            )
            GROUP BY citer_inst_idx, cited_source_idx
        """).fetchdf()
        rows = inst_idx.get_indexer(df['citer_inst_idx'].to_numpy())
        cols = src_idx.get_indexer(df['cited_source_idx'].to_numpy())
        mask = (rows >= 0) & (cols >= 0)
        return sp.coo_matrix(
            (df['weight'].to_numpy(dtype=np.float64)[mask], (rows[mask], cols[mask])),
            shape=(n_u, n_s)
        ).tocsr()

    def _build_ii() -> sp.csr_matrix:
        """
        C_II: no de-duplication — every (citer_inst, cited_inst) combination
        is a genuine cross-product contribution ρ_i × ω_iu × ω_jv.
        """
        df = db.execute("""
            SELECT citer_inst_idx, cited_inst_idx,
                   SUM(rho_w * inst_weight * cited_inst_weight) AS weight
            FROM _tmp_el
            GROUP BY citer_inst_idx, cited_inst_idx
        """).fetchdf()
        rows = inst_idx.get_indexer(df['citer_inst_idx'].to_numpy())
        cols = inst_idx.get_indexer(df['cited_inst_idx'].to_numpy())
        mask = (rows >= 0) & (cols >= 0)
        return sp.coo_matrix(
            (df['weight'].to_numpy(dtype=np.float64)[mask], (rows[mask], cols[mask])),
            shape=(n_u, n_u)
        ).tocsr()

    # ── Assemble requested blocks ──────────────────────────────────────────
    m_SS, m_SI, m_IS, m_II = m
    t0 = time.perf_counter(); C_SS = _build_ss() if m_SS else None; _t['SS'] = time.perf_counter() - t0
    t0 = time.perf_counter(); C_SI = _build_si() if m_SI else None; _t['SI'] = time.perf_counter() - t0
    t0 = time.perf_counter(); C_IS = _build_is() if m_IS else None; _t['IS'] = time.perf_counter() - t0
    t0 = time.perf_counter(); C_II = _build_ii() if m_II else None; _t['II'] = time.perf_counter() - t0

    db.execute("DROP TABLE IF EXISTS _tmp_el")

    total = sum(_t.values())
    print(f"    build_csr timing (s): "
          + "  ".join(f"{k}={v:.2f}" for k, v in _t.items())
          + f"  TOTAL={total:.2f}", flush=True)

    return CSRData(
        C_SS=C_SS, C_SI=C_SI, C_IS=C_IS, C_II=C_II,
        source_ids=source_ids, inst_ids=inst_ids,
        a_s=a_s, a_u=a_u,
        n_s=n_s, n_u=n_u,
    )
