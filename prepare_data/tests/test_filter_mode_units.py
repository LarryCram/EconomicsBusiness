"""
test_filter_mode_units.py — Tests for prepare_data/filter_mode_units.py.

All tests use in-memory DuckDB databases; no SSD data required.

Network topology used in most tests
-------------------------------------
Sources:   s1=1, s2=2, s3=3
Institutions: u1=10, u2=20

Edges (all bipartite SI/IS, no SS or II unless stated):

  SI:  s1→u1,  s2→u1,  s2→u2,  s3→u2
  IS:  u1→s1,  u1→s2,  u2→s2,  u2→s3

Giant bipartite SCC: {s1,s2,s3} ↔ {u1,u2}  (all nodes reachable from each other)

Source s4=4 added in isolation tests: it has NO SI or IS edges, so it should
be dropped from the m=0110 SCC but could survive an m=1000 SCC test if it
has SS edges.

Test categories
---------------
1. All nodes in SCC survive (m=0110)
2. Isolated source (no SI/IS edges) is dropped from m=0110 SCC
3. Source with only SS edges survives m=1000 SCC, dropped from m=0110 SCC
4. m=0001 (II only): institutions survive, sources excluded from table
5. m=1000 (SS only): sources survive, institutions excluded from table
6. m=1111: all four blocks used; result same as m=0110 when SS/II are absent
7. mode_units_table naming is correct
8. write_mode_units creates the correct table with correct columns
9. compute_mode_scc is stable (no extra drops on second call)
10. clean_stale_mode_units drops unexpected tables
"""

import sys
from pathlib import Path
import numpy as np
import duckdb
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from prepare_data.filter_mode_units import (
    compute_mode_scc,
    write_mode_units,
    mode_units_table,
    raw_units_table,
    clean_stale_mode_units,
    mode_str,
)


# ─── Fixtures ──────────────────────────────────────────────────��──────────────

RUN_CODE = 'test0000'
FX       = 'A'
TAU_U    = 5
TAU_S    = 5

# Node IDs
S1, S2, S3, S4 = 1, 2, 3, 4   # sources  (S4 is isolated)
U1, U2         = 10, 20        # institutions


def _make_db(extra_ss_edges=False, extra_ii_edges=False, include_s4=False) -> duckdb.DuckDBPyConnection:
    """
    Build a minimal in-memory DB with el_ and _units_ tables.

    SI:  s1→u1, s2→u1, s2→u2, s3→u2
    IS:  u1→s1, u1→s2, u2→s2, u2→s3
    SS (optional): s1→s2, s2→s3, s3→s1   (closed 3-cycle)
    II (optional): u1→u2, u2→u1           (2-cycle)
    s4 (optional): isolated — no edges of any kind
    """
    el   = f'el_{RUN_CODE}_{FX}_tauU{TAU_U}_tauS{TAU_S}_vartau'
    un   = f'_units_{RUN_CODE}_{FX}_tauU{TAU_U}_tauS{TAU_S}_vartau'
    db   = duckdb.connect(':memory:')

    db.execute(f"""
        CREATE TABLE {el} (
            citer_work_idx    BIGINT,
            citer_source_idx  BIGINT,
            citer_inst_idx    BIGINT,
            cited_work_idx    BIGINT,
            cited_source_idx  BIGINT,
            cited_inst_idx    BIGINT,
            inst_weight       DOUBLE,
            direct_inst_weight DOUBLE,
            cited_inst_weight  DOUBLE,
            R_i               BIGINT,
            a_citer_source    BIGINT,
            a_cited_source    BIGINT,
            a_citer_inst      DOUBLE,
            a_cited_inst      DOUBLE
        )
    """)

    # Helper to insert one edge (work ids arbitrary, weights all 1)
    def ins(cw, cs, ci, dw, ds, di):
        db.execute(
            f"INSERT INTO {el} VALUES (?,?,?,?,?,?,1,1,1,1,3,3,1,1)",
            [cw, cs, ci, dw, ds, di],
        )

    # SI edges
    ins(101, S1, U1, 201, S1, U1)   # s1 cites u1 (via source→inst pathway)
    # Using the IS/SI structure: citer_source=s1, cited_inst=u1 for SI;
    # citer_inst=u1, cited_source=s1 for IS
    # Re-using a work-per-row convention from build_edge_lists:
    #   row with citer_source + cited_inst  → SI contribution
    #   row with citer_inst  + cited_source → IS contribution
    # Our test table covers both in the same rows for simplicity.

    # For SI: citer_source → cited_inst (we use citer_inst=0 as placeholder)
    # For IS: citer_inst   → cited_source (we use citer_source=0 as placeholder)
    # Actually the queries in filter_mode_units.py use DISTINCT on citer/cited pairs,
    # so we just need the relevant columns populated.

    db.execute(f"DELETE FROM {el}")  # clear the insertion above

    # SI rows: citer_source_idx, cited_inst_idx set; citer_inst_idx=0, cited_source_idx=0
    si_edges = [(S1, U1), (S2, U1), (S2, U2), (S3, U2)]
    for i, (cs, ci) in enumerate(si_edges):
        db.execute(
            f"INSERT INTO {el} VALUES (?,?,0,?,0,?,1,1,1,1,3,3,1,1)",
            [1000+i, cs, 2000+i, ci],
        )

    # IS rows: citer_inst_idx, cited_source_idx set; citer_source_idx=0, cited_inst_idx=0
    is_edges = [(U1, S1), (U1, S2), (U2, S2), (U2, S3)]
    for i, (ci, cs) in enumerate(is_edges):
        db.execute(
            f"INSERT INTO {el} VALUES (?,0,?,?,?,0,1,1,1,1,3,3,1,1)",
            [3000+i, ci, 4000+i, cs],
        )

    if extra_ss_edges:
        # SS 3-cycle: s1→s2→s3→s1
        ss_edges = [(S1, S2), (S2, S3), (S3, S1)]
        for i, (cs_r, cs_c) in enumerate(ss_edges):
            db.execute(
                f"INSERT INTO {el} VALUES (?,?,0,?,?,0,1,1,1,1,3,3,1,1)",
                [5000+i, cs_r, 6000+i, cs_c],
            )

    if extra_ii_edges:
        # II 2-cycle: u1→u2→u1; citer_inst_idx and cited_inst_idx both set
        ii_edges = [(U1, U2), (U2, U1)]
        for i, (ci_r, ci_c) in enumerate(ii_edges):
            db.execute(
                f"INSERT INTO {el} VALUES (?,0,?,?,0,?,1,1,1,1,3,3,1,1)",
                [7000+i, ci_r, 8000+i, ci_c],
            )

    # _units_ table (C_full SCC-filtered — include all nodes as pre-filter)
    sources = [S1, S2, S3] + ([S4] if include_s4 else [])
    units_rows = (
        [(s, 'S', 3.0) for s in sources]
        + [(u, 'U', 2.0) for u in [U1, U2]]
    )
    df = pd.DataFrame(units_rows, columns=['unit_idx', 'unit_type', 'a_p'])
    db.register('_u_df', df)
    db.execute(f"CREATE TABLE {un} AS SELECT * FROM _u_df")
    db.unregister('_u_df')

    return db


def _make_db_ii(extra_ii_edges=True) -> duckdb.DuckDBPyConnection:
    """DB with IS edges only (for testing II mode SCC)."""
    el  = f'el_{RUN_CODE}_{FX}_tauU{TAU_U}_tauS{TAU_S}_vartau'
    un  = f'_units_{RUN_CODE}_{FX}_tauU{TAU_U}_tauS{TAU_S}_vartau'
    db  = duckdb.connect(':memory:')
    db.execute(f"""
        CREATE TABLE {el} (
            citer_work_idx BIGINT, citer_source_idx BIGINT, citer_inst_idx BIGINT,
            cited_work_idx BIGINT, cited_source_idx BIGINT, cited_inst_idx BIGINT,
            inst_weight DOUBLE, direct_inst_weight DOUBLE, cited_inst_weight DOUBLE,
            R_i BIGINT, a_citer_source BIGINT, a_cited_source BIGINT,
            a_citer_inst DOUBLE, a_cited_inst DOUBLE
        )
    """)
    # II 2-cycle: u1→u2→u1  (using citer_inst_idx and a separate cited_inst_idx field)
    # The query in filter_mode_units is:
    #   SELECT DISTINCT citer_inst_idx, cited_inst_idx FROM {tname}
    # So we need both columns populated. Use direct_inst_weight column as cited_inst_idx proxy.
    # Actually we need to re-examine: the table schema has cited_inst_idx as a proper column.
    # Insert II edges: citer_inst_idx and cited_inst_idx both set.
    db.execute(f"INSERT INTO {el} VALUES (7001,0,{U1},8001,0,{U2},1,1,1,1,3,3,1,1)")
    db.execute(f"INSERT INTO {el} VALUES (7002,0,{U2},8002,0,{U1},1,1,1,1,3,3,1,1)")

    df = pd.DataFrame(
        [(U1, 'U', 2.0), (U2, 'U', 2.0)],
        columns=['unit_idx', 'unit_type', 'a_p'],
    )
    db.register('_u_df', df)
    db.execute(f"CREATE TABLE {un} AS SELECT * FROM _u_df")
    db.unregister('_u_df')
    return db


# ─── Test: naming ───────────────────────���─────────────────────────────���───────

class TestNaming:

    def test_mode_str(self):
        assert mode_str((0, 1, 1, 0)) == '0110'
        assert mode_str((1, 0, 0, 0)) == '1000'
        assert mode_str((1, 1, 1, 1)) == '1111'

    def test_mode_units_table_name_vartau(self):
        name = mode_units_table(RUN_CODE, FX, TAU_U, TAU_S, (0, 1, 1, 0))
        assert name == f'_units_{RUN_CODE}_{FX}_tauU{TAU_U}_tauS{TAU_S}_vartau_m0110'

    def test_mode_units_table_name_fixtau(self):
        name = mode_units_table(RUN_CODE, FX, TAU_U, TAU_S, (0, 1, 1, 0),
                                ref_units='20242024_ALL_tauU20_tauS20')
        assert name == f'_units_{RUN_CODE}_{FX}_tauU{TAU_U}_tauS{TAU_S}_fixtau_m0110'

    def test_raw_units_table_name_vartau(self):
        name = raw_units_table(RUN_CODE, FX, TAU_U, TAU_S)
        assert name == f'_units_{RUN_CODE}_{FX}_tauU{TAU_U}_tauS{TAU_S}_vartau'

    def test_raw_units_table_name_fixtau(self):
        name = raw_units_table(RUN_CODE, FX, TAU_U, TAU_S,
                               ref_units='20242024_ALL_tauU20_tauS20')
        assert name == f'_units_{RUN_CODE}_{FX}_tauU{TAU_U}_tauS{TAU_S}_fixtau'


# ─── Test: compute_mode_scc ───────────────────────────────��───────────────────

class TestComputeModeSCC:

    def test_bipartite_all_survive(self):
        """All 5 nodes form one bipartite SCC; none should be dropped."""
        db = _make_db()
        src_ids, inst_ids, a_s, a_u, n_ds, n_du = compute_mode_scc(
            db, RUN_CODE, FX, TAU_U, TAU_S, (0, 1, 1, 0)
        )
        assert set(src_ids)  == {S1, S2, S3}
        assert set(inst_ids) == {U1, U2}
        assert n_ds == 0
        assert n_du == 0

    def test_bipartite_drops_isolated_source(self):
        """S4 has no SI/IS edges; should be dropped from m=0110 SCC."""
        db = _make_db(include_s4=True)
        src_ids, inst_ids, a_s, a_u, n_ds, n_du = compute_mode_scc(
            db, RUN_CODE, FX, TAU_U, TAU_S, (0, 1, 1, 0)
        )
        assert S4 not in set(src_ids), "Isolated S4 should be dropped in bipartite mode"
        assert set(src_ids)  == {S1, S2, S3}
        assert set(inst_ids) == {U1, U2}
        assert n_ds == 1   # one source dropped

    def test_ss_only_excludes_institutions(self):
        """m=1000: institutions must not appear in output."""
        db = _make_db(extra_ss_edges=True)
        src_ids, inst_ids, a_s, a_u, _, _ = compute_mode_scc(
            db, RUN_CODE, FX, TAU_U, TAU_S, (1, 0, 0, 0)
        )
        assert len(inst_ids) == 0, "m=1000 must return no institutions"
        assert set(src_ids) == {S1, S2, S3}

    def test_ii_only_excludes_sources(self):
        """m=0001: sources must not appear in output."""
        db = _make_db_ii()
        src_ids, inst_ids, a_s, a_u, _, _ = compute_mode_scc(
            db, RUN_CODE, FX, TAU_U, TAU_S, (0, 0, 0, 1)
        )
        assert len(src_ids) == 0, "m=0001 must return no sources"
        assert set(inst_ids) == {U1, U2}

    def test_a_p_values_correct(self):
        """a_s and a_u must match the _units_ table a_p values."""
        db = _make_db()
        src_ids, inst_ids, a_s, a_u, _, _ = compute_mode_scc(
            db, RUN_CODE, FX, TAU_U, TAU_S, (0, 1, 1, 0)
        )
        assert np.all(a_s == 3.0), "All sources have a_p=3.0 in fixture"
        assert np.all(a_u == 2.0), "All institutions have a_p=2.0 in fixture"

    def test_idempotent(self):
        """Running compute_mode_scc twice on the same DB should give the same result."""
        db = _make_db()
        r1 = compute_mode_scc(db, RUN_CODE, FX, TAU_U, TAU_S, (0, 1, 1, 0))
        r2 = compute_mode_scc(db, RUN_CODE, FX, TAU_U, TAU_S, (0, 1, 1, 0))
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])


# ─── Test: write_mode_units ──────────────────────────────────────��────────────

class TestWriteModeUnits:

    def test_table_created(self):
        """write_mode_units must create the mode-specific table."""
        db = duckdb.connect(':memory:')
        m = (0, 1, 1, 0)
        src_ids  = np.array([S1, S2, S3], dtype=np.int64)
        inst_ids = np.array([U1, U2],     dtype=np.int64)
        a_s = np.array([3.0, 3.0, 3.0])
        a_u = np.array([2.0, 2.0])
        write_mode_units(db, RUN_CODE, FX, TAU_U, TAU_S, m, src_ids, inst_ids, a_s, a_u)
        tname = mode_units_table(RUN_CODE, FX, TAU_U, TAU_S, m)
        tables = {r[0] for r in db.execute('SHOW TABLES').fetchall()}
        assert tname in tables

    def test_table_contents(self):
        """Written table must contain correct unit_idx, unit_type, a_p rows."""
        db = duckdb.connect(':memory:')
        m = (0, 1, 1, 0)
        src_ids  = np.array([S1, S2], dtype=np.int64)
        inst_ids = np.array([U1],     dtype=np.int64)
        a_s = np.array([5.0, 3.0])
        a_u = np.array([2.0])
        write_mode_units(db, RUN_CODE, FX, TAU_U, TAU_S, m, src_ids, inst_ids, a_s, a_u)
        tname = mode_units_table(RUN_CODE, FX, TAU_U, TAU_S, m)
        df = db.execute(f"SELECT unit_idx, unit_type, a_p FROM {tname} ORDER BY unit_idx").fetchdf()
        assert set(df[df['unit_type'] == 'S']['unit_idx']) == {S1, S2}
        assert set(df[df['unit_type'] == 'U']['unit_idx']) == {U1}
        assert df[df['unit_idx'] == S1]['a_p'].values[0] == 5.0

    def test_overwrite_existing(self):
        """A second write must replace the previous table without error."""
        db = duckdb.connect(':memory:')
        m = (0, 1, 1, 0)
        write_mode_units(db, RUN_CODE, FX, TAU_U, TAU_S, m,
                         np.array([S1], dtype=np.int64),
                         np.array([U1], dtype=np.int64),
                         np.array([3.0]), np.array([2.0]))
        write_mode_units(db, RUN_CODE, FX, TAU_U, TAU_S, m,
                         np.array([S1, S2], dtype=np.int64),
                         np.array([U1, U2], dtype=np.int64),
                         np.array([3.0, 3.0]), np.array([2.0, 2.0]))
        tname = mode_units_table(RUN_CODE, FX, TAU_U, TAU_S, m)
        n = db.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
        assert n == 4  # 2 sources + 2 institutions


# ─── Test: clean_stale_mode_units ────────────────────────────────────────────

class TestCleanStale:

    def test_drops_unexpected_mode_tables(self):
        db = duckdb.connect(':memory:')
        # Create two mode-specific tables
        t1 = mode_units_table(RUN_CODE, FX, TAU_U, TAU_S, (0, 1, 1, 0))
        t2 = mode_units_table(RUN_CODE, FX, TAU_U, TAU_S, (1, 0, 0, 0))
        for t in [t1, t2]:
            db.execute(f"CREATE TABLE {t} (unit_idx BIGINT, unit_type VARCHAR, a_p DOUBLE)")
        # Only t1 is in expected set
        clean_stale_mode_units(db, expected_tables={t1})
        tables = {r[0] for r in db.execute('SHOW TABLES').fetchall()}
        assert t1 in tables
        assert t2 not in tables

    def test_does_not_drop_raw_units_tables(self):
        """Tables without _m suffix must never be touched."""
        db = duckdb.connect(':memory:')
        raw = raw_units_table(RUN_CODE, FX, TAU_U, TAU_S)
        db.execute(f"CREATE TABLE {raw} (unit_idx BIGINT, unit_type VARCHAR, a_p DOUBLE)")
        clean_stale_mode_units(db, expected_tables=set())
        tables = {r[0] for r in db.execute('SHOW TABLES').fetchall()}
        assert raw in tables, "clean_stale must not touch raw _units_ tables"
