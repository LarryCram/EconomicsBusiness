"""
test_rank_works.py — pytest suite for rank_works.py

No real data required; all tests use synthetic DataFrames and in-memory DuckDB.

Synthetic fixture
-----------------
Works {1,2,3,4,6}, sources {10,20}, plus phantom work 99 (never a citer).
Citation edges (rho_w = 1.0 throughout):

  1 (src10) → 2, 3
  2 (src20) → 1, 3
  3 (src10) → 4, 6
  4 (src20) → 2
  6 (src20) → 99          ← 99 is not a citer work

After intersection filter (cited_work must also be a citer_work):
  Retained: 1→2, 1→3, 2→1, 2→3, 3→4, 3→6, 4→2   (7 edges)
  work_ids: {1, 2, 3, 4, 6}   N=5
  Work 6: enters via 3→6 (cited), but 6→99 excluded → dangling (zero out-degree)
  Work 99: never retained

H_ww (row-normalised, rho_w=1, zero row for work 6):
  d0(w1): [0,  .5, .5, 0,  0 ]
  d1(w2): [.5, 0,  .5, 0,  0 ]
  d2(w3): [0,  0,  0,  .5, .5]
  d3(w4): [0,  1,  0,  0,  0 ]
  d4(w6): [0,  0,  0,  0,  0 ]  ← dangling

Irreducible sub-matrix (works {1,2,3,4}, strip work 6):
  row-stochastic, primitive → leading eigenvalue = 1 exactly.
"""

import sys
import pytest
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spectral_ranking_works.rank_works import (
    _build_matrix,
    _compute_eigenpairs,
    _leading_vector,
    _aggregate_sources,
    _load_intersection_edges,
    _find_el_table,
)


# ── Synthetic fixture ──────────────────────────────────────────────────────────

# (citer_work, citer_src, cited_work, cited_src, rho_w)
_RAW_EDGES = [
    (1, 10, 2, 20, 1.0),
    (1, 10, 3, 10, 1.0),
    (2, 20, 1, 10, 1.0),
    (2, 20, 3, 10, 1.0),
    (3, 10, 4, 20, 1.0),
    (3, 10, 6, 20, 1.0),
    (4, 20, 2, 20, 1.0),
    (6, 20, 99, 10, 1.0),   # 99 is not a citer → filtered out
]

_CITER_WORKS = {row[0] for row in _RAW_EDGES}   # {1,2,3,4,6}


def _filtered_edges() -> pd.DataFrame:
    """Edge list after intersection filter (cited must be a citer)."""
    rows = [
        {'citer_work_idx': cw, 'cited_work_idx': cd, 'rho_w': rw}
        for cw, _, cd, _, rw in _RAW_EDGES
        if cd in _CITER_WORKS
    ]
    return pd.DataFrame(rows)


def _src_map() -> pd.DataFrame:
    return pd.DataFrame({
        'work_idx':   [1,  2,  3,  4,  6],
        'source_idx': [10, 20, 10, 20, 20],
    })


FILTERED = _filtered_edges()


def _irreducible_H():
    """H and work_ids for works {1,2,3,4} only — row-stochastic, primitive."""
    sub = FILTERED[(FILTERED['cited_work_idx'] != 6) &
                   (FILTERED['citer_work_idx'] != 6)].copy()
    H, work_ids, out_weight, n_zero = _build_matrix(sub)
    return H, work_ids


# ── _build_matrix ──────────────────────────────────────────────────────────────

class TestBuildMatrix:

    def test_shape(self):
        H, work_ids, out_weight, n_zero = _build_matrix(FILTERED)
        assert H.shape == (5, 5)
        assert len(work_ids) == 5
        assert len(out_weight) == 5

    def test_work_ids_sorted(self):
        _, work_ids, _, _ = _build_matrix(FILTERED)
        assert np.all(work_ids[:-1] <= work_ids[1:])

    def test_correct_work_ids(self):
        _, work_ids, _, _ = _build_matrix(FILTERED)
        assert set(work_ids.tolist()) == {1, 2, 3, 4, 6}

    def test_non_dangling_rows_sum_to_one(self):
        H, work_ids, _, _ = _build_matrix(FILTERED)
        row_sums = np.array(H.sum(axis=1)).ravel()
        idx6 = int(np.where(work_ids == 6)[0][0])
        for i in range(5):
            if i == idx6:
                assert row_sums[i] == pytest.approx(0.0)
            else:
                assert row_sums[i] == pytest.approx(1.0), f'row {i} sums to {row_sums[i]}'

    def test_dangling_count(self):
        _, _, _, n_zero = _build_matrix(FILTERED)
        assert n_zero == 1   # only work 6

    def test_out_weight_dangling_is_zero(self):
        _, work_ids, out_weight, _ = _build_matrix(FILTERED)
        idx6 = int(np.where(work_ids == 6)[0][0])
        assert out_weight[idx6] == pytest.approx(0.0)

    def test_out_weight_two_citations(self):
        """Work 1 makes 2 citations (rho_w=1 each) → out_weight = 2."""
        _, work_ids, out_weight, _ = _build_matrix(FILTERED)
        idx1 = int(np.where(work_ids == 1)[0][0])
        assert out_weight[idx1] == pytest.approx(2.0)

    def test_out_weight_one_citation(self):
        """Work 4 makes 1 citation → out_weight = 1."""
        _, work_ids, out_weight, _ = _build_matrix(FILTERED)
        idx4 = int(np.where(work_ids == 4)[0][0])
        assert out_weight[idx4] == pytest.approx(1.0)

    def test_nonnegative_entries(self):
        H, _, _, _ = _build_matrix(FILTERED)
        assert H.nnz > 0
        assert H.data.min() >= 0.0

    def test_irreducible_sub_no_dangling(self):
        H, _, _, n_zero = _build_matrix(
            FILTERED[(FILTERED['citer_work_idx'] != 6) &
                     (FILTERED['cited_work_idx'] != 6)])
        assert n_zero == 0


# ── _compute_eigenpairs ────────────────────────────────────────────────────────

class TestComputeEigenpairs:

    def test_returns_right_and_left_keys(self):
        H, _ = _irreducible_H()
        ep = _compute_eigenpairs(H, k=2)
        assert 'right' in ep and 'left' in ep

    def test_right_converges_for_irreducible(self):
        H, _ = _irreducible_H()
        ep = _compute_eigenpairs(H, k=2)
        assert ep['right'] is not None

    def test_left_converges_for_irreducible(self):
        H, _ = _irreducible_H()
        ep = _compute_eigenpairs(H, k=2)
        assert ep['left'] is not None

    def test_leading_right_eigenvalue_is_one(self):
        """Row-stochastic irreducible matrix: Perron root = 1."""
        H, _ = _irreducible_H()
        ep = _compute_eigenpairs(H, k=2)
        assert abs(ep['right']['vals'][0]) == pytest.approx(1.0, abs=1e-5)

    def test_leading_left_eigenvalue_is_one(self):
        H, _ = _irreducible_H()
        ep = _compute_eigenpairs(H, k=2)
        assert abs(ep['left']['vals'][0]) == pytest.approx(1.0, abs=1e-5)

    def test_k_eigenpairs_returned(self):
        H, _ = _irreducible_H()
        ep = _compute_eigenpairs(H, k=2)
        assert len(ep['right']['vals']) == 2
        assert ep['right']['vecs'].shape[1] == 2

    def test_ratios_all_le_one(self):
        """All |λ_k| / |λ_1| must be ≤ 1."""
        H, _ = _irreducible_H()
        ep = _compute_eigenpairs(H, k=2)
        vals = ep['right']['vals']
        lam1 = abs(vals[0])
        for lam in vals:
            assert abs(lam) / lam1 <= 1.0 + 1e-6

    def test_second_ratio_lt_one_for_primitive(self):
        """Primitive matrix: |λ_2| / |λ_1| strictly < 1."""
        H, _ = _irreducible_H()
        ep = _compute_eigenpairs(H, k=2)
        vals = ep['right']['vals']
        ratio = abs(vals[1]) / abs(vals[0])
        assert ratio < 1.0 - 1e-4

    def test_full_matrix_right_converges(self):
        """eigs on the 5-node matrix with dangling work 6 should converge."""
        H, _, _, _ = _build_matrix(FILTERED)
        ep = _compute_eigenpairs(H, k=2)
        assert ep['right'] is not None

    def test_dangling_hub_score_is_zero(self):
        """Work 6 (zero out-degree) must have zero hub (right eigenvector) score."""
        H, work_ids, _, _ = _build_matrix(FILTERED)
        ep = _compute_eigenpairs(H, k=2)
        if ep['right'] is None:
            pytest.skip('right eigenpairs did not converge')
        v = _leading_vector(ep['right'], 'right')
        idx6 = int(np.where(work_ids == 6)[0][0])
        assert abs(v[idx6]) == pytest.approx(0.0, abs=1e-4)


# ── _leading_vector ────────────────────────────────────────────────────────────

class TestLeadingVector:

    def test_none_input_returns_none(self):
        assert _leading_vector(None, 'right') is None

    def test_returns_real_array(self):
        H, _ = _irreducible_H()
        ep = _compute_eigenpairs(H, k=2)
        v = _leading_vector(ep['right'], 'right')
        assert v.ndim == 1
        assert np.issubdtype(v.dtype, np.floating)

    def test_positive_mean(self):
        H, _ = _irreducible_H()
        ep = _compute_eigenpairs(H, k=2)
        v = _leading_vector(ep['right'], 'right')
        assert v.mean() > 0

    def test_negative_mean_flipped(self):
        """Vector with negative mean is sign-flipped to positive mean."""
        ep = {
            'vals': np.array([1.0 + 0j, 0.5 + 0j]),
            'vecs': np.array([[-1.0 + 0j, 0.3 + 0j],
                              [-2.0 + 0j, 0.1 + 0j]]),
        }
        v = _leading_vector(ep, 'right')
        assert v.mean() > 0
        assert v[0] == pytest.approx(1.0)
        assert v[1] == pytest.approx(2.0)

    def test_positive_mean_unchanged(self):
        """Vector already positive is returned as-is."""
        ep = {
            'vals': np.array([1.0 + 0j]),
            'vecs': np.array([[3.0 + 0j], [1.0 + 0j]]),
        }
        v = _leading_vector(ep, 'right')
        assert v[0] == pytest.approx(3.0)


# ── _aggregate_sources ─────────────────────────────────────────────────────────

class TestAggregateSources:
    """
    work_ids sorted: [1,2,3,4,6] → dense [0,1,2,3,4]
    source 10: works 1(d0), 3(d2)        out_weights 2, 2
    source 20: works 2(d1), 4(d3), 6(d4) out_weights 2, 1, 0
    """

    def _run(self, v):
        _, work_ids, out_weight, _ = _build_matrix(FILTERED)
        return _aggregate_sources(work_ids, v, out_weight, _src_map())

    def test_returns_two_sources(self):
        df = self._run(np.ones(5))
        assert len(df) == 2
        assert set(df['source_idx'].tolist()) == {10, 20}

    def test_n_works_src10(self):
        df = self._run(np.ones(5))
        assert df[df['source_idx'] == 10].iloc[0]['n_works'] == 2

    def test_n_works_src20(self):
        df = self._run(np.ones(5))
        assert df[df['source_idx'] == 20].iloc[0]['n_works'] == 3

    def test_uniform_mean_src10(self):
        """v = [1,_,3,_,_] for source 10 → uniform mean = 2."""
        v = np.array([1.0, 0.0, 3.0, 0.0, 0.0])
        df = self._run(v)
        assert df[df['source_idx'] == 10].iloc[0]['v_uniform'] == pytest.approx(2.0)

    def test_degree_weighted_src10(self):
        """out_weights both 2; v=[1,3] → degree-weighted = (2*1+2*3)/4 = 2."""
        v = np.array([1.0, 0.0, 3.0, 0.0, 0.0])
        df = self._run(v)
        assert df[df['source_idx'] == 10].iloc[0]['v_degree'] == pytest.approx(2.0)

    def test_degree_weighted_src20(self):
        """works 2(ow=2,v=4), 4(ow=1,v=2), 6(ow=0,v=5) → (2*4+1*2+0*5)/(2+1+0)=10/3."""
        v = np.array([0.0, 4.0, 0.0, 2.0, 5.0])
        df = self._run(v)
        val = df[df['source_idx'] == 20].iloc[0]['v_degree']
        assert val == pytest.approx(10.0 / 3.0, rel=1e-5)

    def test_dangling_excluded_from_degree_weight(self):
        """Work 6 (out_weight=0) contributes nothing to degree-weighted average."""
        # Give work 6 a huge v; it must not move the degree-weighted score
        v_no6 = np.array([0.0, 4.0, 0.0, 2.0, 0.0])
        v_with6 = np.array([0.0, 4.0, 0.0, 2.0, 1e9])
        df1 = self._run(v_no6)
        df2 = self._run(v_with6)
        v1 = df1[df1['source_idx'] == 20].iloc[0]['v_degree']
        v2 = df2[df2['source_idx'] == 20].iloc[0]['v_degree']
        assert v1 == pytest.approx(v2)

    def test_has_required_columns(self):
        df = self._run(np.ones(5))
        for col in ('source_idx', 'n_works', 'v_uniform', 'v_degree'):
            assert col in df.columns

    def test_works_absent_from_src_ignored(self):
        """If work 6 absent from df_src, source 20 has only 2 works."""
        _, work_ids, out_weight, _ = _build_matrix(FILTERED)
        partial_src = _src_map()[_src_map()['work_idx'] != 6]
        df = _aggregate_sources(work_ids, np.ones(5), out_weight, partial_src)
        assert df[df['source_idx'] == 20].iloc[0]['n_works'] == 2


# ── _load_intersection_edges ───────────────────────────────────────────────────

def _make_tmp_el_db():
    db = duckdb.connect(':memory:')
    db.execute("""
        CREATE TABLE _tmp_el (
            citer_work_idx  INTEGER,
            citer_source_idx INTEGER,
            cited_work_idx  INTEGER,
            rho_w           DOUBLE
        )
    """)
    for cw, cs, cd, _, rw in _RAW_EDGES:
        db.execute('INSERT INTO _tmp_el VALUES (?, ?, ?, ?)', [cw, cs, cd, rw])
    return db


class TestLoadIntersectionEdges:

    def test_phantom_work_excluded_from_cited(self):
        """Work 99 (never a citer) must not appear as cited_work_idx."""
        db = _make_tmp_el_db()
        df_edges, _ = _load_intersection_edges(db)
        db.close()
        assert 99 not in df_edges['cited_work_idx'].values

    def test_work6_appears_as_cited(self):
        """Work 6 is a citer (6→99) so 3→6 is retained."""
        db = _make_tmp_el_db()
        df_edges, _ = _load_intersection_edges(db)
        db.close()
        assert 6 in df_edges['cited_work_idx'].values

    def test_edge_count(self):
        """7 distinct (citer, cited) pairs survive the filter."""
        db = _make_tmp_el_db()
        df_edges, _ = _load_intersection_edges(db)
        db.close()
        assert len(df_edges) == 7

    def test_rho_w_positive(self):
        db = _make_tmp_el_db()
        df_edges, _ = _load_intersection_edges(db)
        db.close()
        assert (df_edges['rho_w'] > 0).all()

    def test_src_map_covers_all_citers(self):
        """df_src must include every citer_work_idx value."""
        db = _make_tmp_el_db()
        _, df_src = _load_intersection_edges(db)
        db.close()
        assert _CITER_WORKS.issubset(set(df_src['work_idx'].tolist()))


# ── _find_el_table ─────────────────────────────────────────────────────────────

def _make_table_db():
    db = duckdb.connect(':memory:')
    for name in [
        'el_20242024_EBAX_tauU20_tauS20_vartau',
        'el_20242024_EB_tauU20_tauS20_vartau',
        'el_20242024_EBAX_tauU20_tauS20_fixtau',   # fixtau — still vartau preferred
        'rk_unrelated',
    ]:
        db.execute(f'CREATE TABLE "{name}" (x INTEGER)')
    return db


class TestFindElTable:

    def test_prefers_ebax_fx(self):
        db = _make_table_db()
        t = _find_el_table(db, '20242024', 20, 20)
        db.close()
        assert '_EBAX_' in t

    def test_returns_vartau(self):
        db = _make_table_db()
        t = _find_el_table(db, '20242024', 20, 20)
        db.close()
        assert t.endswith('_vartau')

    def test_raises_if_no_match(self):
        db = _make_table_db()
        with pytest.raises(FileNotFoundError):
            _find_el_table(db, '00000000', 20, 20)
        db.close()

    def test_falls_back_to_non_all(self):
        """When no ALL table exists, returns the only vartau candidate."""
        db = duckdb.connect(':memory:')
        db.execute('CREATE TABLE "el_20242024_EB_tauU20_tauS20_vartau" (x INTEGER)')
        t = _find_el_table(db, '20242024', 20, 20)
        db.close()
        assert 'EB' in t
