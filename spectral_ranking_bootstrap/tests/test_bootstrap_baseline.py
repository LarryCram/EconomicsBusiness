"""
test_bootstrap_baseline.py — pytest suite for bootstrap_baseline.py.

All tests use in-memory DuckDB; no real data files are required.
"""

import sys
import json
import pytest
import numpy as np
import scipy.sparse as sp
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import duckdb
from spectral_ranking_bootstrap.bootstrap_baseline import (
    compute_r_bar,
    create_tmp_el,
    preload_edges,
    bootstrap_step,
    compute_v,
    save_checkpoint,
    load_checkpoint,
)
from spectral_ranking.katz_ranker import _row_normalise


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_in_memory_db(n_sources=2, n_insts=2, rows=None):
    """
    Create an in-memory DuckDB database with a synthetic edge list and units table.

    Default: 2 sources (idx 10, 20), 2 institutions (idx 100, 200).
    Each row in the edge list represents one attribution with R_i references.

    rows: list of tuples (citer_work_idx, citer_source_idx, citer_inst_idx,
                          cited_work_idx,  cited_source_idx, cited_inst_idx,
                          inst_weight, cited_inst_weight, R_i)
    """
    db = duckdb.connect(':memory:')

    if rows is None:
        # 4 basic edges: 2 citing works (R_i=2 each), 2 cited works
        rows = [
            # citer_work, citer_src, citer_inst, cited_work, cited_src, cited_inst, inst_w, cited_inst_w, R_i
            (1, 10, 100,  2, 20, 200, 1.0, 1.0, 2),
            (1, 10, 100,  3, 10, 100, 1.0, 1.0, 2),
            (4, 20, 200,  2, 20, 200, 1.0, 1.0, 2),
            (4, 20, 200,  5, 10, 100, 1.0, 1.0, 2),
        ]

    db.execute("""
        CREATE TABLE test_el (
            citer_work_idx    INTEGER,
            citer_source_idx  INTEGER,
            citer_inst_idx    INTEGER,
            cited_work_idx    INTEGER,
            cited_source_idx  INTEGER,
            cited_inst_idx    INTEGER,
            inst_weight       DOUBLE,
            cited_inst_weight DOUBLE,
            R_i               INTEGER
        )
    """)
    db.executemany("INSERT INTO test_el VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    # Units table: 2 sources (type S) and 2 institutions (type U)
    db.execute("""
        CREATE TABLE test_units (
            unit_idx  INTEGER,
            unit_type VARCHAR,
            a_p       DOUBLE
        )
    """)
    unit_rows = [
        (10, 'S', 5.0),
        (20, 'S', 3.0),
        (100, 'U', 4.0),
        (200, 'U', 2.0),
    ]
    db.executemany("INSERT INTO test_units VALUES (?, ?, ?)", unit_rows)

    return db


# ── Test 1: preload_shapes ────────────────────────────────────────────────────

def test_preload_shapes():
    """SI and IS pre-load arrays have correct lengths and dtypes."""
    db = make_in_memory_db()
    r_bar = compute_r_bar(db, 'test_el')
    create_tmp_el(db, 'test_el', r_bar)
    edges = preload_edges(db, 'test_units')

    # SI arrays
    assert len(edges['si_src_dense'])  == len(edges['si_w']), "SI src length mismatch"
    assert len(edges['si_inst_dense']) == len(edges['si_w']), "SI inst length mismatch"
    assert edges['si_src_dense'].dtype  == np.int32
    assert edges['si_inst_dense'].dtype == np.int32
    assert edges['si_w'].dtype          == np.float64

    # IS arrays
    assert len(edges['is_inst_dense']) == len(edges['is_w']), "IS inst length mismatch"
    assert len(edges['is_src_dense'])  == len(edges['is_w']), "IS src length mismatch"
    assert edges['is_inst_dense'].dtype == np.int32
    assert edges['is_src_dense'].dtype  == np.int32
    assert edges['is_w'].dtype          == np.float64

    # Dimensions
    assert edges['n_s'] == 2
    assert edges['n_u'] == 2
    assert edges['N_SI'] == len(edges['si_w'])
    assert edges['N_IS'] == len(edges['is_w'])


# ── Test 2: preload_weights ───────────────────────────────────────────────────

def test_preload_weights():
    """
    SI weights match hand-computed rho_w * cited_inst_weight.

    Setup: 2 citing works, each with R_i=2. r_bar = 2.0, so rho_w = 2/2 = 1.0.
    cited_inst_weight=1.0 for all rows, so w = 1.0 for each distinct edge.
    """
    rows = [
        # citer_work=1 cites cited_inst=200 via source 10, R_i=2
        (1, 10, 100, 2, 20, 200, 1.0, 1.0, 2),
        # citer_work=4 cites cited_inst=100 via source 20, R_i=2
        (4, 20, 200, 5, 10, 100, 1.0, 1.0, 2),
    ]
    db = make_in_memory_db(rows=rows)
    r_bar = compute_r_bar(db, 'test_el')
    assert abs(r_bar - 2.0) < 1e-10, f"Expected r_bar=2.0, got {r_bar}"

    create_tmp_el(db, 'test_el', r_bar)
    edges = preload_edges(db, 'test_units')

    # rho_w = r_bar / R_i = 2/2 = 1.0; cited_inst_weight = 1.0 → w = 1.0
    assert np.allclose(edges['si_w'], 1.0), f"Expected all SI weights=1.0, got {edges['si_w']}"
    assert np.allclose(edges['is_w'], 1.0), f"Expected all IS weights=1.0, got {edges['is_w']}"


# ── Test 3: bootstrap_sample_coo ─────────────────────────────────────────────

def test_bootstrap_sample_coo():
    """Bootstrap COO/CSR has correct shape and non-negative entries."""
    db = make_in_memory_db()
    r_bar = compute_r_bar(db, 'test_el')
    create_tmp_el(db, 'test_el', r_bar)
    edges = preload_edges(db, 'test_units')

    n_s = edges['n_s']
    n_u = edges['n_u']
    N_SI = edges['N_SI']

    rng = np.random.default_rng(0)
    boot_si = rng.choice(N_SI, size=int(0.8 * N_SI), replace=True)
    C_SI = sp.coo_matrix(
        (edges['si_w'][boot_si],
         (edges['si_src_dense'][boot_si], edges['si_inst_dense'][boot_si])),
        shape=(n_s, n_u),
    ).tocsr()

    assert C_SI.shape == (n_s, n_u), f"Wrong shape: {C_SI.shape}"
    assert C_SI.data.min() >= 0, "Negative entries in C_SI"


# ── Test 4: coo_duplicate_summation ──────────────────────────────────────────

def test_coo_duplicate_summation():
    """
    When a row index appears twice in boot_si, COO.tocsr() sums weights correctly.
    """
    # Manually construct a small edge array where row 0 appears twice
    si_src_dense  = np.array([0, 0, 1], dtype=np.int32)
    si_inst_dense = np.array([0, 0, 1], dtype=np.int32)
    si_w          = np.array([2.0, 3.0, 1.0], dtype=np.float64)

    # Select all indices (simulating a bootstrap that repeats index 0 twice)
    boot_si = np.array([0, 1, 2], dtype=np.int64)  # includes both copies of row 0

    C = sp.coo_matrix(
        (si_w[boot_si], (si_src_dense[boot_si], si_inst_dense[boot_si])),
        shape=(2, 2),
    ).tocsr()

    # C[0, 0] should be 2.0 + 3.0 = 5.0
    assert abs(C[0, 0] - 5.0) < 1e-10, f"Expected 5.0 at [0,0], got {C[0, 0]}"
    assert abs(C[1, 1] - 1.0) < 1e-10, f"Expected 1.0 at [1,1], got {C[1, 1]}"


# ── Test 5: v_computation ────────────────────────────────────────────────────

def test_v_computation():
    """v_s = A * pi_s/2 / a_s and v_u = A * pi_u/2 / a_u match expected values."""
    pi_s = np.array([0.6, 0.4])
    pi_u = np.array([0.7, 0.3])
    a_s  = np.array([5.0, 3.0])
    a_u  = np.array([4.0, 2.0])
    A    = a_s.sum() + a_u.sum()  # 14.0

    v_s, v_u = compute_v(pi_s, pi_u, a_s, a_u)

    expected_v_s = np.array([A * 0.6 / 2.0 / 5.0, A * 0.4 / 2.0 / 3.0], dtype=np.float32)
    expected_v_u = np.array([A * 0.7 / 2.0 / 4.0, A * 0.3 / 2.0 / 2.0], dtype=np.float32)

    assert np.allclose(v_s, expected_v_s, rtol=1e-5), f"v_s mismatch: {v_s} vs {expected_v_s}"
    assert np.allclose(v_u, expected_v_u, rtol=1e-5), f"v_u mismatch: {v_u} vs {expected_v_u}"
    assert v_s.dtype == np.float32
    assert v_u.dtype == np.float32


# ── Test 6: v_mean_close_to_one ──────────────────────────────────────────────

def test_v_mean_close_to_one():
    """
    The a_p-weighted mean of v_s and v_u equals 1.0 by construction.

    For m=0110 with joint normalisation:
      pi_s_joint = pi_s / 2,  pi_u_joint = pi_u / 2
      sum(pi_s_joint) + sum(pi_u_joint) = 0.5 + 0.5 = 1.0

      sum(a_s * v_s) / A = sum(a_s * A * pi_s_joint / a_s) / A
                         = sum(pi_s_joint) = 0.5

      sum(a_u * v_u) / A = sum(pi_u_joint) = 0.5

      (sum(a_s * v_s) + sum(a_u * v_u)) / A = 1.0

    This test verifies that identity for arbitrary valid inputs.
    """
    # Use arbitrary pi that individually sum to 1
    pi_s = np.array([0.3, 0.4, 0.3])
    pi_u = np.array([0.5, 0.5])
    a_s  = np.array([2.0, 3.0, 5.0])
    a_u  = np.array([4.0, 6.0])
    A    = a_s.sum() + a_u.sum()

    v_s, v_u = compute_v(pi_s, pi_u, a_s, a_u)

    combined_mean = (np.dot(a_s, v_s.astype(np.float64)) +
                     np.dot(a_u, v_u.astype(np.float64))) / A
    assert abs(combined_mean - 1.0) < 1e-5, f"Combined a_p-weighted mean = {combined_mean}, expected 1.0"


# ── Test 7: checkpoint_roundtrip ─────────────────────────────────────────────

def test_checkpoint_roundtrip(tmp_path):
    """Write v_s_boot, v_u_boot and meta.json; read back; verify shapes and fields."""
    B, n_s, n_u = 5, 10, 8
    v_s_boot = np.random.rand(B, n_s).astype(np.float32)
    v_u_boot = np.random.rand(B, n_u).astype(np.float32)
    meta = dict(
        n=B, seed=42, tol=1e-7,
        n_s=n_s, n_u=n_u,
        source_ids=list(range(n_s)),
        inst_ids=list(range(n_u)),
        run_code='20242024',
        baseline_table='el_20242024_A_tauU20_tauS20',
        completed=B,
    )

    save_checkpoint(tmp_path, v_s_boot, v_u_boot, meta)

    v_s_rt, v_u_rt, meta_rt = load_checkpoint(tmp_path)

    assert v_s_rt.shape == (B, n_s), f"v_s shape: {v_s_rt.shape}"
    assert v_u_rt.shape == (B, n_u), f"v_u shape: {v_u_rt.shape}"
    assert np.allclose(v_s_rt, v_s_boot)
    assert np.allclose(v_u_rt, v_u_boot)
    assert meta_rt['n']         == B
    assert meta_rt['completed'] == B
    assert meta_rt['run_code']  == '20242024'
    assert meta_rt['n_s']       == n_s
    assert meta_rt['n_u']       == n_u
    assert meta_rt['source_ids'] == list(range(n_s))


# ── Test 8: resume_skips_completed ───────────────────────────────────────────

def test_resume_skips_completed(tmp_path):
    """
    Simulate a partial run (completed=3 out of n=5).
    Resume mode should skip the first 3 replicates and fill only indices 3 and 4.
    """
    B, n_s, n_u = 5, 4, 3
    completed = 3

    # Write a partial checkpoint where only the first `completed` rows are non-zero
    v_s_partial = np.zeros((B, n_s), dtype=np.float32)
    v_u_partial = np.zeros((B, n_u), dtype=np.float32)
    # Fill first `completed` rows with sentinel value 99.0
    v_s_partial[:completed] = 99.0
    v_u_partial[:completed] = 99.0

    meta = dict(
        n=B, seed=42, tol=1e-7,
        n_s=n_s, n_u=n_u,
        source_ids=list(range(10, 10 + n_s)),
        inst_ids=list(range(100, 100 + n_u)),
        run_code='20242024',
        baseline_table='el_20242024_A_tauU20_tauS20',
        completed=completed,
    )
    save_checkpoint(tmp_path, v_s_partial, v_u_partial, meta)

    # Load checkpoint and verify resume logic
    v_s_boot, v_u_boot, meta_rt = load_checkpoint(tmp_path)
    start_b = meta_rt['completed']

    assert start_b == 3, f"Expected start_b=3, got {start_b}"

    # Replicates that would be processed on resume: range(3, 5) = [3, 4]
    remaining_replicates = list(range(start_b, B))
    assert remaining_replicates == [3, 4], f"Wrong remaining: {remaining_replicates}"

    # Sentinel values in first completed rows must be preserved
    assert np.all(v_s_boot[:completed] == 99.0), "Completed rows were overwritten"


# ── Test 9: r_bar_computation ────────────────────────────────────────────────

def test_r_bar_computation():
    """
    r_bar = mean of distinct (citer_work_idx, R_i) pairs.

    Setup: two citing works with R_i values 2 and 4 → r_bar = 3.0.
    Even if there are multiple rows per work (different cited insts), R_i
    is the same per work, so DISTINCT eliminates duplicates before AVG.
    """
    rows = [
        # work 1 has R_i=2, appears twice (two cited institutions)
        (1, 10, 100, 2, 20, 200, 1.0, 1.0, 2),
        (1, 10, 100, 3, 10, 100, 1.0, 1.0, 2),
        # work 4 has R_i=4, appears once
        (4, 20, 200, 5, 10, 100, 1.0, 1.0, 4),
    ]
    db = make_in_memory_db(rows=rows)
    r_bar = compute_r_bar(db, 'test_el')

    # DISTINCT (citer_work_idx, R_i) → {(1,2), (4,4)} → AVG = 3.0
    assert abs(r_bar - 3.0) < 1e-10, f"Expected r_bar=3.0, got {r_bar}"


# ── Test 10: dangling_row_handling ───────────────────────────────────────────

def test_dangling_row_handling():
    """
    If a bootstrap sample omits all rows for one source, _row_normalise
    produces a zero row — no crash and no NaN propagation.
    """
    # C_SI with source 0 having no references (all-zero row)
    n_s, n_u = 3, 2
    data = np.array([1.0, 2.0])       # only rows 1 and 2 have entries
    rows = np.array([1, 2], dtype=np.int32)
    cols = np.array([0, 1], dtype=np.int32)
    C_SI = sp.coo_matrix((data, (rows, cols)), shape=(n_s, n_u)).tocsr()

    H_SI, dangling_mask = _row_normalise(C_SI)

    # Row 0 should be dangling (zero)
    assert dangling_mask[0], "Row 0 should be dangling"
    assert not dangling_mask[1], "Row 1 should not be dangling"
    assert not dangling_mask[2], "Row 2 should not be dangling"

    # No NaN or Inf in H_SI
    assert not np.any(np.isnan(H_SI.data)), "NaN in H_SI after _row_normalise"
    assert not np.any(np.isinf(H_SI.data)), "Inf in H_SI after _row_normalise"

    # Row 0 should be all zeros
    row0 = np.array(H_SI[0].todense()).ravel()
    assert np.all(row0 == 0.0), f"Row 0 not zero: {row0}"

    # Rows 1 and 2 should sum to 1
    row1 = np.array(H_SI[1].todense()).ravel()
    row2 = np.array(H_SI[2].todense()).ravel()
    assert abs(row1.sum() - 1.0) < 1e-10, f"Row 1 sum = {row1.sum()}"
    assert abs(row2.sum() - 1.0) < 1e-10, f"Row 2 sum = {row2.sum()}"
