"""
build_csr.py — Build raw CSR blocks from a pre-built edge list.

Reads one edge list table (el_t{tx}_{fx}_tau{tau_u}) and the corresponding
unit index table (_units_t{tx}_{fx}_tau{tau_u}) from edge_lists.duckdb.
Applies ρ weighting, builds scipy CSR matrices for the requested blocks,
and returns a CSRData object.

No matrix assembly, no row-normalisation, no algorithm logic — all of that
lives in katz_ranker.py.

The edge list and _units tables must already exist (run
prepare_data/build_edge_lists.py first).

Public API
----------
build_csr(db, tx, fx, tau_u, rho, m) -> CSRData
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


def build_csr(db, tx: int, fx: str, tau_u: int, rho: int, m: tuple) -> CSRData:
    """
    Build raw CSR blocks for corpus (tx, fx, tau_u) with ρ weighting.

    Parameters
    ----------
    db : duckdb connection (open, writeable not required).
    tx : time window index (1–7).
    fx : field subset ('E', 'B', 'A').
    tau_u : institution retention threshold.
    rho : 0 → fixed count (ρ_i = R̄/R_i); 1 → full count (ρ_i = 1).
    m : (m_SS, m_SI, m_IS, m_II) ∈ {0,1}^4 — which blocks to build.

    Returns
    -------
    CSRData with requested blocks populated; others None.
    """
    tname  = f'el_t{tx}_{fx}_tau{tau_u}'
    uname  = f'_units_t{tx}_{fx}_tau{tau_u}'

    # ── Load unit index ────────────────────────────────────────────────────
    try:
        units_df = db.execute(
            f"SELECT unit_idx, unit_type, a_p FROM {uname} ORDER BY unit_type, unit_idx"
        ).fetchdf()
    except Exception as e:
        raise RuntimeError(
            f"Unit index table '{uname}' not found in edge_lists.duckdb. "
            f"Run prepare_data/build_edge_lists.py first."
        ) from e

    src_df  = units_df[units_df['unit_type'] == 'S'].reset_index(drop=True)
    inst_df = units_df[units_df['unit_type'] == 'U'].reset_index(drop=True)

    source_ids = src_df['unit_idx'].to_numpy(dtype=np.int64)
    inst_ids   = inst_df['unit_idx'].to_numpy(dtype=np.int64)
    a_s = src_df['a_p'].to_numpy(dtype=np.float64)
    a_u = inst_df['a_p'].to_numpy(dtype=np.float64)
    n_s = len(source_ids)
    n_u = len(inst_ids)

    # Dense index maps: original idx → 0-based position
    src_map  = pd.Series(np.arange(n_s, dtype=np.int32), index=source_ids)
    inst_map = pd.Series(np.arange(n_u, dtype=np.int32), index=inst_ids)

    # ── ρ weighting ────────────────────────────────────────────────────────
    if rho == 0:
        # Fixed count: ρ_i = R̄ / R_i  where R̄ = mean over distinct citer works
        r_bar = db.execute(
            f"SELECT AVG(R_i) FROM (SELECT DISTINCT citer_work_idx, R_i FROM {tname})"
        ).fetchone()[0]
        rho_expr = f"CAST({r_bar} AS DOUBLE) / CAST(R_i AS DOUBLE)"
    else:
        rho_expr = "1.0"

    # ── Block builders ─────────────────────────────────────────────────────

    def _build_ss() -> sp.csr_matrix:
        """
        C_SS: de-duplicate on (citer_work, cited_work) to count each
        reference once regardless of how many institution combinations exist.
        """
        sql = f"""
            SELECT citer_source_idx, cited_source_idx, SUM(rho_w) AS weight
            FROM (
                SELECT DISTINCT citer_work_idx, citer_source_idx,
                                cited_work_idx,  cited_source_idx,
                                {rho_expr} AS rho_w
                FROM {tname}
            )
            GROUP BY citer_source_idx, cited_source_idx
        """
        df = db.execute(sql).fetchdf()
        rows = src_map.loc[df['citer_source_idx'].values].values
        cols = src_map.loc[df['cited_source_idx'].values].values
        return sp.coo_matrix(
            (df['weight'].to_numpy(dtype=np.float64), (rows, cols)),
            shape=(n_s, n_s)
        ).tocsr()

    def _build_si() -> sp.csr_matrix:
        """
        C_SI: de-duplicate over citer_inst so each (citer_work, cited_work,
        cited_inst) contributes ρ_i × ω_jv exactly once.
        """
        sql = f"""
            SELECT citer_source_idx, cited_inst_idx,
                   SUM(rho_w * cited_inst_weight) AS weight
            FROM (
                SELECT DISTINCT citer_work_idx, citer_source_idx,
                                cited_work_idx,  cited_inst_idx,
                                {rho_expr} AS rho_w, cited_inst_weight
                FROM {tname}
            )
            GROUP BY citer_source_idx, cited_inst_idx
        """
        df = db.execute(sql).fetchdf()
        rows = src_map.loc[df['citer_source_idx'].values].values
        cols = inst_map.loc[df['cited_inst_idx'].values].values
        return sp.coo_matrix(
            (df['weight'].to_numpy(dtype=np.float64), (rows, cols)),
            shape=(n_s, n_u)
        ).tocsr()

    def _build_is() -> sp.csr_matrix:
        """
        C_IS: de-duplicate over cited_inst so each (citer_work, citer_inst,
        cited_work) contributes ρ_i × ω_iu exactly once.
        """
        sql = f"""
            SELECT citer_inst_idx, cited_source_idx,
                   SUM(rho_w * inst_weight) AS weight
            FROM (
                SELECT DISTINCT citer_work_idx, citer_inst_idx,
                                cited_work_idx,  cited_source_idx,
                                {rho_expr} AS rho_w, inst_weight
                FROM {tname}
            )
            GROUP BY citer_inst_idx, cited_source_idx
        """
        df = db.execute(sql).fetchdf()
        rows = inst_map.loc[df['citer_inst_idx'].values].values
        cols = src_map.loc[df['cited_source_idx'].values].values
        return sp.coo_matrix(
            (df['weight'].to_numpy(dtype=np.float64), (rows, cols)),
            shape=(n_u, n_s)
        ).tocsr()

    def _build_ii() -> sp.csr_matrix:
        """
        C_II: no de-duplication — every (citer_inst, cited_inst) combination
        is a genuine cross-product contribution ρ_i × ω_iu × ω_jv.
        """
        sql = f"""
            SELECT citer_inst_idx, cited_inst_idx,
                   SUM(({rho_expr}) * inst_weight * cited_inst_weight) AS weight
            FROM {tname}
            GROUP BY citer_inst_idx, cited_inst_idx
        """
        df = db.execute(sql).fetchdf()
        rows = inst_map.loc[df['citer_inst_idx'].values].values
        cols = inst_map.loc[df['cited_inst_idx'].values].values
        return sp.coo_matrix(
            (df['weight'].to_numpy(dtype=np.float64), (rows, cols)),
            shape=(n_u, n_u)
        ).tocsr()

    # ── Assemble requested blocks ──────────────────────────────────────────
    m_SS, m_SI, m_IS, m_II = m
    C_SS = _build_ss() if m_SS else None
    C_SI = _build_si() if m_SI else None
    C_IS = _build_is() if m_IS else None
    C_II = _build_ii() if m_II else None

    return CSRData(
        C_SS=C_SS, C_SI=C_SI, C_IS=C_IS, C_II=C_II,
        source_ids=source_ids, inst_ids=inst_ids,
        a_s=a_s, a_u=a_u,
        n_s=n_s, n_u=n_u,
    )
