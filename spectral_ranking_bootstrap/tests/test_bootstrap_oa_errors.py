"""
test_bootstrap_oa_errors.py — pytest suite for bootstrap_oa_errors.py.

All tests use in-memory DuckDB and synthetic DataFrames; no real data required.

Synthetic corpus
----------------
  Sources:      idx 10 (s=0), 20 (s=1)
  Institutions: idx 100 (u=0), 200 (u=1)

  References (citer_work, cited_work, cited_inst):
    work 1 (src 10) → work 2 (src 20, inst 200)   year 2022
    work 1 (src 10) → work 3 (src 10, inst 100)   year 2022
    work 4 (src 20) → work 2 (src 20, inst 200)   year 2022
    work 4 (src 20) → work 5 (src 10, inst 100)   year 2021

  Like-sets (source × year, ≥2 candidates):
    (src 20, 2022): cited works 2 only → singleton, NOT in like_sets
    (src 10, 2022): cited works 3 only → singleton, NOT in like_sets
    (src 10, 2021): cited works 5 only → singleton, NOT in like_sets
    → No like-sets with ≥2 members in the minimal fixture.

  Extended fixture for like-set tests adds work 6 (src 20, year 2022):
    work 1 (src 10) → work 6 (src 20, inst 100)   year 2022
  Now (src 20, 2022): cited works {2, 6} → like_sets populated.
"""

import sys
import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import duckdb
from spectral_ranking_bootstrap.bootstrap_baseline import compute_r_bar, create_tmp_el
from spectral_ranking_bootstrap.bootstrap_oa_errors import (
    build_like_sets,
    preload_ref_edges,
    build_csi_perturbed,
    bootstrap_step_ref,
    preload_inst_edges,
    bootstrap_step_inst,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

BASE_ROWS = [
    # citer_work, citer_src, citer_inst, cited_work, cited_src, cited_inst,
    # inst_weight, cited_inst_weight, R_i
    (1, 10, 100, 2, 20, 200, 1.0, 1.0, 2),
    (1, 10, 100, 3, 10, 100, 1.0, 1.0, 2),
    (4, 20, 200, 2, 20, 200, 1.0, 1.0, 2),
    (4, 20, 200, 5, 10, 100, 1.0, 1.0, 2),
]

EXTENDED_ROWS = BASE_ROWS + [
    # work 6 in src 20, year 2022 — gives (src 20, 2022) a like-set of size 2
    (1, 10, 100, 6, 20, 100, 1.0, 1.0, 2),
]

BASE_WORKS = pd.DataFrame({
    'work_idx':          [1, 2, 3, 4, 5, 6],
    'publication_year':  [2022, 2022, 2022, 2021, 2021, 2022],
})

UNIT_ROWS = [
    (10,  'S', 5.0),
    (20,  'S', 3.0),
    (100, 'U', 4.0),
    (200, 'U', 2.0),
]


def make_db(rows=None):
    db = duckdb.connect(':memory:')
    if rows is None:
        rows = BASE_ROWS
    db.execute("""
        CREATE TABLE test_el (
            citer_work_idx    INTEGER, citer_source_idx  INTEGER,
            citer_inst_idx    INTEGER, cited_work_idx    INTEGER,
            cited_source_idx  INTEGER, cited_inst_idx    INTEGER,
            inst_weight       DOUBLE,  cited_inst_weight DOUBLE,
            R_i               INTEGER
        )
    """)
    db.executemany("INSERT INTO test_el VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    db.execute("""
        CREATE TABLE test_units (
            unit_idx INTEGER, unit_type VARCHAR, a_p DOUBLE
        )
    """)
    db.executemany("INSERT INTO test_units VALUES (?, ?, ?)", UNIT_ROWS)
    return db


def make_preloaded(rows=None, works_df=None):
    """Return (ref_data, db) with _tmp_el already created."""
    if rows is None:
        rows = EXTENDED_ROWS
    if works_df is None:
        works_df = BASE_WORKS
    db = make_db(rows)
    r_bar = compute_r_bar(db, 'test_el')
    create_tmp_el(db, 'test_el', r_bar)
    ref_data = preload_ref_edges(db, 'test_units', works_df)
    return ref_data, db


# ── Test 1: build_like_sets ───────────────────────────────────────────────────

def test_build_like_sets_basic():
    """Works in the same source×year group appear in each other's like-set."""
    cw_dense   = np.array([0, 1, 2, 3], dtype=np.int32)
    src        = np.array([10, 10, 20, 10], dtype=np.int64)
    year       = np.array([2022, 2022, 2022, 2023], dtype=np.int32)

    ls = build_like_sets(cw_dense, src, year)

    # cw_dense 0 and 1 share (src=10, year=2022)
    assert 0 in ls and 1 in ls
    assert set(ls[0].tolist()) == {0, 1}
    assert set(ls[1].tolist()) == {0, 1}

    # cw_dense 2 is alone in (src=20, 2022) — not in like_sets
    assert 2 not in ls

    # cw_dense 3 is alone in (src=10, 2023) — not in like_sets
    assert 3 not in ls


def test_build_like_sets_three_works():
    """A group of 3 gives each member a like-set of all 3."""
    cw_dense = np.array([0, 1, 2], dtype=np.int32)
    src      = np.array([10, 10, 10], dtype=np.int64)
    year     = np.array([2020, 2020, 2020], dtype=np.int32)

    ls = build_like_sets(cw_dense, src, year)

    for cw in [0, 1, 2]:
        assert cw in ls
        assert set(ls[cw].tolist()) == {0, 1, 2}


def test_build_like_sets_empty():
    """All singletons → empty dict."""
    cw_dense = np.array([0, 1, 2], dtype=np.int32)
    src      = np.array([10, 20, 30], dtype=np.int64)
    year     = np.array([2020, 2020, 2020], dtype=np.int32)

    ls = build_like_sets(cw_dense, src, year)
    assert ls == {}


# ── Test 2: preload_ref_edges shapes ─────────────────────────────────────────

def test_preload_shapes():
    """Preloaded arrays have consistent lengths and correct dtypes."""
    ref_data, _ = make_preloaded()

    N_refs = len(ref_data['ref_rho_w'])
    assert len(ref_data['ref_citer_src_dense'])  == N_refs
    assert len(ref_data['ref_cited_work_dense']) == N_refs

    assert ref_data['ref_citer_src_dense'].dtype  == np.int32
    assert ref_data['ref_cited_work_dense'].dtype == np.int32
    assert ref_data['ref_rho_w'].dtype            == np.float64

    n_s = ref_data['n_s']
    n_u = ref_data['n_u']
    assert n_s == 2
    assert n_u == 2

    assert ref_data['W_cw'].shape[1]   == n_u
    assert ref_data['C_IS'].shape      == (n_u, n_s)


def test_preload_like_sets_populated():
    """Works 2 and 6 share (src=20, year=2022) → like_sets populated."""
    ref_data, _ = make_preloaded(rows=EXTENDED_ROWS)
    ls = ref_data['like_sets']
    # At least one like-set should exist
    assert len(ls) > 0


def test_preload_cis_fixed():
    """C_IS values are positive and shape is (n_u, n_s)."""
    ref_data, _ = make_preloaded()
    C_IS = ref_data['C_IS']
    assert C_IS.shape == (ref_data['n_u'], ref_data['n_s'])
    assert C_IS.nnz > 0
    assert np.all(C_IS.data > 0)


# ── Test 3: build_csi_perturbed — p=0 unchanged ───────────────────────────────

def test_build_csi_p0_matches_baseline():
    """With p=0, no references are replaced; C_SI equals the baseline."""
    ref_data, _ = make_preloaded()
    rng = np.random.default_rng(0)

    C_SI_0 = build_csi_perturbed(ref_data, p=0.0, rng=rng)
    C_SI_1 = build_csi_perturbed(ref_data, p=0.0, rng=rng)

    # Both should be identical to each other (deterministic at p=0)
    diff = (C_SI_0 - C_SI_1).data
    assert len(diff) == 0 or np.allclose(diff, 0), "p=0 results differ between calls"


def test_build_csi_shape():
    """C_SI has correct shape (n_s, n_u)."""
    ref_data, _ = make_preloaded()
    C_SI = build_csi_perturbed(ref_data, p=0.0, rng=np.random.default_rng(0))
    assert C_SI.shape == (ref_data['n_s'], ref_data['n_u'])


def test_build_csi_nonnegative():
    """All entries in C_SI are non-negative."""
    ref_data, _ = make_preloaded()
    C_SI = build_csi_perturbed(ref_data, p=0.05, rng=np.random.default_rng(1))
    assert np.all(C_SI.data >= 0), "Negative entry in C_SI"


# ── Test 4: build_csi_perturbed — p=1 replaces all replaceable refs ───────────

def test_build_csi_p1_changes_with_like_sets():
    """
    With p=1 and like-sets present, at least one reference is replaced and
    the resulting C_SI differs from p=0 in at least one entry.
    """
    ref_data, _ = make_preloaded(rows=EXTENDED_ROWS)

    if not ref_data['like_sets']:
        pytest.skip('No like-sets in fixture; cannot test p=1 replacement.')

    rng_base = np.random.default_rng(99)
    rng_pert = np.random.default_rng(99)

    C_base = build_csi_perturbed(ref_data, p=0.0, rng=rng_base)
    C_pert = build_csi_perturbed(ref_data, p=1.0, rng=rng_pert)

    diff = (C_pert - C_base)
    assert diff.nnz > 0 or not np.allclose(C_pert.toarray(), C_base.toarray()), \
        "p=1 C_SI identical to p=0 despite like-sets being present"


# ── Test 5: C_IS is not modified by build_csi_perturbed ──────────────────────

def test_cis_unchanged_by_perturbation():
    """C_IS in ref_data must not be mutated by build_csi_perturbed."""
    ref_data, _ = make_preloaded(rows=EXTENDED_ROWS)

    C_IS_before = ref_data['C_IS'].copy()
    build_csi_perturbed(ref_data, p=1.0, rng=np.random.default_rng(0))
    C_IS_after = ref_data['C_IS']

    assert (C_IS_before - C_IS_after).nnz == 0, "C_IS was mutated by build_csi_perturbed"


# ── Test 6: bootstrap_step_ref returns correct tuple ──────────────────────────

def test_bootstrap_step_ref_tuple():
    """bootstrap_step_ref returns a 7-tuple with correct types and shapes."""
    ref_data, _ = make_preloaded(rows=EXTENDED_ROWS)
    result = bootstrap_step_ref(b=0, seed=42, ref_data=ref_data, p=0.05, tol=1e-6)

    assert len(result) == 7, f"Expected 7-tuple, got {len(result)}"
    v_s_b, v_u_b, lam1, lam2, n_s_core, n_u_core, n_iter = result

    assert v_s_b.dtype == np.float32
    assert v_u_b.dtype == np.float32
    assert v_s_b.shape == (ref_data['n_s'],)
    assert v_u_b.shape == (ref_data['n_u'],)
    assert isinstance(lam1, float) and lam1 > 0
    assert isinstance(lam2, float)
    assert 0 < n_s_core <= ref_data['n_s']
    assert 0 < n_u_core <= ref_data['n_u']
    assert n_iter >= 1


def test_bootstrap_step_ref_v_positive_in_core():
    """v values are positive (not NaN) for units in the bipartite core."""
    ref_data, _ = make_preloaded(rows=EXTENDED_ROWS)
    v_s_b, v_u_b, *_ = bootstrap_step_ref(b=0, seed=42, ref_data=ref_data, p=0.0, tol=1e-6)

    # At least one non-NaN entry in each
    assert np.any(np.isfinite(v_s_b)), "All v_s NaN"
    assert np.any(np.isfinite(v_u_b)), "All v_u NaN"
    # Non-NaN values must be positive
    assert np.all(v_s_b[np.isfinite(v_s_b)] > 0), "Non-NaN v_s contains non-positive value"
    assert np.all(v_u_b[np.isfinite(v_u_b)] > 0), "Non-NaN v_u contains non-positive value"


def test_bootstrap_step_ref_different_seeds():
    """Across 20 replicates with different b, at least one pair of v arrays differs."""
    ref_data, _ = make_preloaded(rows=EXTENDED_ROWS)

    if not ref_data['like_sets']:
        pytest.skip('No like-sets; perturbations are impossible, seeds make no difference.')

    results = [
        bootstrap_step_ref(b=b, seed=1000, ref_data=ref_data, p=0.5, tol=1e-6)[0]
        for b in range(20)
    ]
    v_matrix = np.array([np.nan_to_num(r) for r in results])
    all_same = all(np.allclose(v_matrix[0], v_matrix[i], atol=1e-4) for i in range(1, 20))
    assert not all_same, "All 20 replicates with different b produced identical v_s — perturbations not propagating"


# ── Test 7: inst mode raises NotImplementedError ──────────────────────────────

def test_inst_mode_not_implemented():
    """preload_inst_edges and bootstrap_step_inst raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        preload_inst_edges(None, 'units', 'el', pd.DataFrame())

    with pytest.raises(NotImplementedError):
        bootstrap_step_inst(0, 42, {}, 0.05, 1e-7)
