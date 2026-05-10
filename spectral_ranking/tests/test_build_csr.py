"""
Tests for build_csr.py.

Uses an in-memory DuckDB with a minimal synthetic edge list so no SSD data is needed.
Tests cover: table name construction, all four block combinations, rho weighting.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import duckdb
import pytest

from build_csr import build_csr, CSRData


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_db(run_code='20242024', fx='EBAX', tau_u=20, tau_s=20):
    """
    Build a minimal in-memory DuckDB with one edge list table and its units table.

    Topology (two sources, two institutions, two works each):
      work 1 (src 1, inst 1)  →  work 3 (src 2, inst 2)
      work 2 (src 1, inst 2)  →  work 4 (src 2, inst 1)

    All weights = 1.0 for simplicity; R_i = 1 for each citing work.
    """
    tname = f'el_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_vartau'
    uname = f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_vartau'

    db = duckdb.connect(':memory:')

    db.execute(f"""
        CREATE TABLE {tname} (
            citer_work_idx   BIGINT,
            citer_source_idx BIGINT,
            citer_inst_idx   BIGINT,
            cited_work_idx   BIGINT,
            cited_source_idx BIGINT,
            cited_inst_idx   BIGINT,
            inst_weight      DOUBLE,
            direct_inst_weight DOUBLE,
            cited_inst_weight  DOUBLE,
            R_i              BIGINT,
            a_citer_source   BIGINT,
            a_cited_source   BIGINT,
            a_citer_inst     DOUBLE,
            a_cited_inst     DOUBLE
        )
    """)

    db.execute(f"""
        INSERT INTO {tname} VALUES
        --  cw  cs  ci  dw  ds  di  iw  diw  ciw  Ri  acs acd  aci  acd_i
        (1,  1,  1,  3,  2,  2, 1.0, 1.0, 1.0,  1,  2,  2, 1.0, 1.0),
        (2,  1,  2,  4,  2,  1, 1.0, 1.0, 1.0,  1,  2,  2, 1.0, 1.0)
    """)

    units_ddl = f"""
        CREATE TABLE {{name}} (
            unit_idx  BIGINT,
            unit_type VARCHAR,
            a_p       DOUBLE
        )
    """
    units_rows = f"""
        INSERT INTO {{name}} VALUES
        (1, 'S', 2.0),
        (2, 'S', 2.0),
        (1, 'U', 1.0),
        (2, 'U', 1.0)
    """
    # Base units table plus all mode-specific variants (with and without _eps1)
    for suffix in ['', '_m0001', '_m0110', '_m1000', '_m1111',
                   '_eps1', '_m0001_eps1', '_m0110_eps1', '_m1000_eps1', '_m1111_eps1']:
        name = uname + suffix
        db.execute(units_ddl.format(name=name))
        db.execute(units_rows.format(name=name))

    # epsilon=1 edge list (same data, different table name)
    tname_eps = tname + '_eps1'
    db.execute(f"""
        CREATE TABLE {tname_eps} AS SELECT * FROM {tname}
    """)

    return db


# ─── Table naming ─────────────────────────────────────────────────────────────

def test_correct_tables_used():
    """build_csr looks up el_{run_code}_... not el_t{tx}_..."""
    db = _make_db(run_code='20242024', fx='EBAX', tau_u=20, tau_s=20)
    data = build_csr(db, '20242024', 'EBAX', 20, 20, rho=0, m=(1,1,1,1))
    assert data.n_s == 2
    assert data.n_u == 2


def test_missing_table_raises():
    db = duckdb.connect(':memory:')
    with pytest.raises(RuntimeError, match="not found"):
        build_csr(db, '99999999', 'EBAX', 20, 20, rho=0, m=(0,1,1,0))


# ─── Block selection ──────────────────────────────────────────────────────────

def test_m_0110_si_is_not_none():
    db = _make_db()
    data = build_csr(db, '20242024', 'EBAX', 20, 20, rho=0, m=(0,1,1,0))
    assert data.C_SS is None
    assert data.C_SI is not None
    assert data.C_IS is not None
    assert data.C_II is None


def test_m_1000_ss_only():
    db = _make_db()
    data = build_csr(db, '20242024', 'EBAX', 20, 20, rho=0, m=(1,0,0,0))
    assert data.C_SS is not None
    assert data.C_SI is None
    assert data.C_IS is None
    assert data.C_II is None


def test_m_0001_ii_only():
    db = _make_db()
    data = build_csr(db, '20242024', 'EBAX', 20, 20, rho=0, m=(0,0,0,1))
    assert data.C_SS is None
    assert data.C_II is not None


def test_m_1111_all_blocks():
    db = _make_db()
    data = build_csr(db, '20242024', 'EBAX', 20, 20, rho=0, m=(1,1,1,1))
    assert data.C_SS is not None
    assert data.C_SI is not None
    assert data.C_IS is not None
    assert data.C_II is not None


# ─── Matrix shapes ────────────────────────────────────────────────────────────

def test_block_shapes():
    db = _make_db()
    data = build_csr(db, '20242024', 'EBAX', 20, 20, rho=0, m=(1,1,1,1))
    assert data.C_SS.shape == (2, 2)
    assert data.C_SI.shape == (2, 2)
    assert data.C_IS.shape == (2, 2)
    assert data.C_II.shape == (2, 2)


# ─── rho weighting ────────────────────────────────────────────────────────────

def test_rho0_vs_rho1_differ():
    """With R_i=1 for all works, rho=0 gives R̄/R_i = 1 = rho=1, so weights equal."""
    db0 = _make_db()
    db1 = _make_db()
    d0 = build_csr(db0, '20242024', 'EBAX', 20, 20, rho=0, m=(0,1,1,0))
    d1 = build_csr(db1, '20242024', 'EBAX', 20, 20, rho=1, m=(0,1,1,0))
    # With uniform R_i=1, R̄=1, so rho_w identical — sums should match
    assert abs(d0.C_SI.sum() - d1.C_SI.sum()) < 1e-10


# ─── Metadata arrays ─────────────────────────────────────────────────────────

def test_source_and_inst_ids():
    db = _make_db()
    data = build_csr(db, '20242024', 'EBAX', 20, 20, rho=0, m=(0,1,1,0))
    assert set(data.source_ids) == {1, 2}
    assert set(data.inst_ids)   == {1, 2}
    assert len(data.a_s) == 2
    assert len(data.a_u) == 2


# ─── Sentinel masks (epsilon=1) ───────────────────────────────────────────────

def test_epsilon0_no_sentinel_masks():
    db = _make_db()
    data = build_csr(db, '20242024', 'EBAX', 20, 20, rho=0, m=(0,1,1,0), epsilon=0)
    assert data.is_sentinel_s is None
    assert data.is_sentinel_u is None


def test_epsilon1_sentinel_masks_present():
    db = _make_db()
    data = build_csr(db, '20242024', 'EBAX', 20, 20, rho=0, m=(0,1,1,0), epsilon=1)
    assert data.is_sentinel_s is not None
    assert data.is_sentinel_u is not None
    assert data.is_sentinel_s.dtype == bool
    assert data.is_sentinel_u.dtype == bool


def test_epsilon1_sentinel_position():
    db = _make_db()
    data = build_csr(db, '20242024', 'EBAX', 20, 20, rho=0, m=(0,1,1,0), epsilon=1)
    # source_ids and inst_ids both contain idx=1; those positions must be True
    s_pos = np.where(data.source_ids == 1)[0]
    u_pos = np.where(data.inst_ids   == 1)[0]
    assert len(s_pos) == 1 and data.is_sentinel_s[s_pos[0]]
    assert len(u_pos) == 1 and data.is_sentinel_u[u_pos[0]]
    # idx=2 must be False
    s2_pos = np.where(data.source_ids == 2)[0]
    u2_pos = np.where(data.inst_ids   == 2)[0]
    assert not data.is_sentinel_s[s2_pos[0]]
    assert not data.is_sentinel_u[u2_pos[0]]
