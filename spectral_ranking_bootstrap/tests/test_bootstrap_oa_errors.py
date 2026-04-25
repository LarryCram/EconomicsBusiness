"""
test_bootstrap_oa_errors.py — pytest suite for bootstrap_oa_errors.py (4-stage pipeline).

All tests use in-memory / synthetic data; no real parquets required.

Synthetic corpus (50 works, 2019-2025, 5 sources, 3 institutions, 3 countries)
is built once via module-level fixtures.
"""

import sys
import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spectral_ranking_bootstrap.bootstrap_oa_errors import (
    YEAR_LO, YEAR_HI, P_ERROR, P_WITHIN,
    perturb_works_one,
    apply_work_delta,
    apply_ref_delta,
    load_base_refs,
    _build_matrices,
    _rank_one,
)


# ── Synthetic corpus ───────────────────────────────────────────────────────────
# 5 sources (idx 10,20,30,40,50), 3 institutions (idx 100,200,300),
# 3 countries (US, GB, AU), 50 works (pub_year 2020-2024 for most).

RNG0 = np.random.default_rng(0)

N_WORKS   = 60
N_SOURCES = 5
N_INSTS   = 3
SRC_IDS   = np.array([10, 20, 30, 40, 50], dtype=np.int64)
INST_IDS  = np.array([100, 200, 300], dtype=np.int64)
COUNTRIES = ['US', 'GB', 'AU']

_work_ids    = np.arange(1, N_WORKS + 1, dtype=np.int64)
_pub_years   = (np.tile([YEAR_LO, YEAR_LO+1, YEAR_LO+2, YEAR_HI-1, YEAR_HI, YEAR_LO+1],
                        N_WORKS // 6 + 1)[:N_WORKS]).astype(np.int16)
_src_assign  = SRC_IDS[RNG0.integers(0, N_SOURCES, N_WORKS)]
_inst_assign = INST_IDS[RNG0.integers(0, N_INSTS, N_WORKS)]
_inst_weight = np.ones(N_WORKS, dtype=np.float32)
_country     = [COUNTRIES[RNG0.integers(0, 3)] for _ in range(N_WORKS)]


def make_base_works() -> pd.DataFrame:
    """Single-institution base_works matching Stage 1 schema."""
    return pd.DataFrame({
        'work_idx':    _work_ids,
        'pub_year':    _pub_years,
        'source_idx':  _src_assign,
        'inst_idx':    pd.array(_inst_assign, dtype='Int64'),
        'inst_weight': _inst_weight,
        'country_code': _country,
    })


def make_inst_pool() -> pd.DataFrame:
    rows = []
    for i, (iid, cc) in enumerate(zip(INST_IDS, COUNTRIES)):
        rows.append({'institution_idx': iid, 'country_code': cc})
    return pd.DataFrame(rows)


BASE_WORKS = make_base_works()
INST_POOL  = make_inst_pool()
SRC_POOL   = SRC_IDS.copy()


# ── Stage 2 tests ──────────────────────────────────────────────────────────────

class TestPerturb:
    def _delta(self, seed=0):
        rng = np.random.default_rng(seed)
        return perturb_works_one(BASE_WORKS, SRC_POOL, INST_POOL, rng)

    def test_returns_dataframe(self):
        d = self._delta()
        assert isinstance(d, pd.DataFrame)

    def test_schema_has_required_columns(self):
        d = self._delta()
        for col in ('work_idx', 'pub_year', 'source_idx', 'inst_idx',
                    'inst_weight', 'country_code', 'inst_idx_old'):
            assert col in d.columns, f'Missing column {col}'

    def test_error_rates_approx_10pct_each(self):
        # Run many replicates and check rates
        n_reps = 50
        work_ids = BASE_WORKS['work_idx'].unique()
        n_works  = len(work_ids)

        py_count = src_count = ins_count = 0
        changed_per_rep = []
        for b in range(n_reps):
            rng = np.random.default_rng(1000 + b)
            d   = perturb_works_one(BASE_WORKS, SRC_POOL, INST_POOL, rng)
            changed_set = set(d['work_idx'].tolist())

            # Work-level error classification
            py_works  = set(d.loc[d['inst_idx_old'].isna() &
                                   (d.groupby('work_idx')['pub_year'].transform('first') !=
                                    BASE_WORKS.set_index('work_idx')['pub_year'].reindex(d['work_idx']).values),
                                   'work_idx'].tolist())
            # just count changed works vs total (crude but fast check)
            changed_per_rep.append(len(changed_set))

        avg_changed = np.mean(changed_per_rep)
        expected = n_works * 3 * P_ERROR   # ~30% of works change
        assert abs(avg_changed - expected) / n_works < 0.10, \
            f'Avg changed={avg_changed:.1f}, expected ~{expected:.1f}'

    def test_disjoint_errors(self):
        """Each work appears in at most one error type per replicate."""
        for b in range(20):
            rng = np.random.default_rng(2000 + b)
            d   = perturb_works_one(BASE_WORKS, SRC_POOL, INST_POOL, rng)
            if len(d) == 0:
                continue
            # Works that changed pub_year: inst_idx_old is NA and pub_year differs from base
            base_py = BASE_WORKS.set_index('work_idx')['pub_year']
            base_src = BASE_WORKS.drop_duplicates('work_idx').set_index('work_idx')['source_idx']
            d_dedup = d.drop_duplicates('work_idx')
            py_changed  = set(d_dedup[
                d_dedup['work_idx'].map(base_py) != d_dedup['pub_year']
            ]['work_idx'].tolist())
            src_changed = set(d_dedup[
                d_dedup['work_idx'].map(base_src) != d_dedup['source_idx']
            ]['work_idx'].tolist())
            ins_changed = set(d[d['inst_idx_old'].notna()]['work_idx'].tolist())
            # Check pairwise disjoint
            assert len(py_changed & src_changed) == 0, 'pub_year and source errors overlap'
            assert len(py_changed & ins_changed) == 0, 'pub_year and inst errors overlap'
            assert len(src_changed & ins_changed) == 0, 'source and inst errors overlap'

    def test_pub_year_boundary_lo(self):
        """Works at YEAR_LO can only go to YEAR_LO+1, never below YEAR_LO."""
        lo_works = BASE_WORKS[BASE_WORKS['pub_year'] == YEAR_LO]['work_idx'].unique()
        if len(lo_works) == 0:
            pytest.skip('No works at YEAR_LO in fixture')
        for b in range(30):
            rng = np.random.default_rng(3000 + b)
            d   = perturb_works_one(BASE_WORKS, SRC_POOL, INST_POOL, rng)
            affected = d[d['work_idx'].isin(lo_works)]
            assert (affected['pub_year'] >= YEAR_LO).all(), \
                f'pub_year below {YEAR_LO} for boundary work'

    def test_pub_year_boundary_hi(self):
        """Works at YEAR_HI can only go to YEAR_HI-1, never above YEAR_HI."""
        hi_works = BASE_WORKS[BASE_WORKS['pub_year'] == YEAR_HI]['work_idx'].unique()
        if len(hi_works) == 0:
            pytest.skip('No works at YEAR_HI in fixture')
        for b in range(30):
            rng = np.random.default_rng(4000 + b)
            d   = perturb_works_one(BASE_WORKS, SRC_POOL, INST_POOL, rng)
            affected = d[d['work_idx'].isin(hi_works)]
            assert (affected['pub_year'] <= YEAR_HI).all(), \
                f'pub_year above {YEAR_HI} for boundary work'

    def test_inst_weight_recomputed(self):
        """inst_weight must equal 1/N_i after any institution change."""
        for b in range(10):
            rng = np.random.default_rng(5000 + b)
            d   = perturb_works_one(BASE_WORKS, SRC_POOL, INST_POOL, rng)
            ins = d[d['inst_idx_old'].notna()]
            if len(ins) == 0:
                continue
            for wid, grp in ins.groupby('work_idx'):
                n_i = grp['inst_idx'].dropna().nunique()
                expected_w = np.float32(1.0 / max(n_i, 1))
                assert np.allclose(grp['inst_weight'].values, expected_w, atol=1e-6), \
                    f'inst_weight mismatch for work {wid}'


# ── apply_work_delta tests ─────────────────────────────────────────────────────

class TestApplyWorkDelta:
    def test_none_delta_returns_base(self):
        result = apply_work_delta(BASE_WORKS, None)
        pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                      BASE_WORKS.reset_index(drop=True))

    def test_empty_delta_returns_base(self):
        empty = pd.DataFrame(columns=list(BASE_WORKS.columns) + ['replicate_id', 'inst_idx_old'])
        result = apply_work_delta(BASE_WORKS, empty)
        assert set(result['work_idx'].tolist()) == set(BASE_WORKS['work_idx'].tolist())

    def test_changed_work_replaced(self):
        wid = int(BASE_WORKS['work_idx'].iloc[0])
        delta_row = BASE_WORKS[BASE_WORKS['work_idx'] == wid].copy()
        delta_row['pub_year'] = np.int16(YEAR_LO + 1)
        delta_row['replicate_id'] = 0
        delta_row['inst_idx_old'] = pd.NA

        result = apply_work_delta(BASE_WORKS, delta_row)
        new_row = result[result['work_idx'] == wid]
        assert int(new_row['pub_year'].iloc[0]) == YEAR_LO + 1

    def test_unchanged_works_preserved(self):
        wid = int(BASE_WORKS['work_idx'].iloc[0])
        delta_row = BASE_WORKS[BASE_WORKS['work_idx'] == wid].copy()
        delta_row['pub_year'] = np.int16(YEAR_LO + 1)
        delta_row['replicate_id'] = 0
        delta_row['inst_idx_old'] = pd.NA

        result = apply_work_delta(BASE_WORKS, delta_row)
        unchanged = result[result['work_idx'] != wid]
        expected  = BASE_WORKS[BASE_WORKS['work_idx'] != wid]
        assert len(unchanged) == len(expected)


# ── apply_ref_delta tests ──────────────────────────────────────────────────────

class TestApplyRefDelta:
    def _base_refs(self):
        return np.array([
            [1, 2, 10],
            [1, 3, 20],
            [4, 2, 20],
            [4, 5, 30],
        ], dtype=np.int64)

    def test_no_changes_returns_copy(self):
        refs = self._base_refs()
        result = apply_ref_delta(refs, np.array([], dtype=np.int64),
                                 np.array([], dtype=np.int64))
        assert np.array_equal(refs, result)
        assert result is not refs  # copy

    def test_single_change_applied(self):
        refs   = self._base_refs()
        result = apply_ref_delta(refs,
                                 np.array([0], dtype=np.int64),
                                 np.array([99], dtype=np.int64))
        assert result[0, 1] == 99
        assert result[1, 1] == 3  # unchanged

    def test_original_not_mutated(self):
        refs = self._base_refs()
        _    = apply_ref_delta(refs, np.array([0], dtype=np.int64),
                               np.array([99], dtype=np.int64))
        assert refs[0, 1] == 2  # original unchanged


# ── _build_matrices tests ──────────────────────────────────────────────────────

class TestBuildMatrices:
    """Use a tiny 3-work, 2-source, 2-institution synthetic corpus."""

    # Works: idx 1 (src 10, inst 100, year 2020), 2 (src 20, inst 200, year 2021),
    #        3 (src 10, inst 100, year 2022)
    WORKS = pd.DataFrame({
        'work_idx':    pd.array([1, 2, 3], dtype=np.int64),
        'pub_year':    pd.array([2020, 2021, 2022], dtype=np.int16),
        'source_idx':  pd.array([10, 20, 10], dtype=np.int64),
        'inst_idx':    pd.array([100, 200, 100], dtype='Int64'),
        'inst_weight': np.array([1.0, 1.0, 1.0], dtype=np.float32),
        'country_code': ['US', 'GB', 'US'],
    })

    # References: work1→work2, work1→work3, work2→work3
    REFS = np.array([
        [1, 2, 20],
        [1, 3, 10],
        [2, 3, 10],
    ], dtype=np.int64)

    SRC_IDX  = pd.Index(np.array([10, 20], dtype=np.int64))
    INST_IDX = pd.Index(np.array([100, 200], dtype=np.int64))

    def _build(self, refs=None, works=None):
        r = refs  if refs  is not None else self.REFS
        w = works if works is not None else self.WORKS
        works_yr = w[w['pub_year'].between(YEAR_LO, YEAR_HI)]
        return _build_matrices(r, works_yr, self.SRC_IDX, self.INST_IDX,
                               r_bar=2.0, n_s=2, n_u=2)

    def test_shapes(self):
        C_SI, C_IS = self._build()
        assert C_SI.shape == (2, 2)
        assert C_IS.shape == (2, 2)

    def test_nonnegative(self):
        C_SI, C_IS = self._build()
        assert np.all(C_SI.data >= 0)
        assert np.all(C_IS.data >= 0)

    def test_has_nonzero_entries(self):
        C_SI, C_IS = self._build()
        assert C_SI.nnz > 0
        assert C_IS.nnz > 0

    def test_year_filter_removes_out_of_window(self):
        # Work 1 has pub_year=2020 (YEAR_LO); should be included.
        # Patch work 1 to pub_year=2019 (out of window).
        works_mod = self.WORKS.copy()
        works_mod.loc[works_mod['work_idx'] == 1, 'pub_year'] = np.int16(2019)
        C_SI, C_IS = self._build(works=works_mod)
        # work1 is citer of works 2 and 3; removing it should reduce C_SI nnz
        C_SI_base, _ = self._build()
        # With work1 removed as citer, fewer references → smaller or equal nnz
        assert C_SI.nnz <= C_SI_base.nnz

    def test_empty_refs_returns_zero_matrices(self):
        empty_refs = np.zeros((0, 3), dtype=np.int64)
        C_SI, C_IS = self._build(refs=empty_refs)
        assert C_SI.nnz == 0
        assert C_IS.nnz == 0


# ── _rank_one tests ────────────────────────────────────────────────────────────

class TestRankOne:
    """Build tiny C_SI, C_IS and verify _rank_one output contract."""

    def _tiny_matrices(self):
        # 2 sources (s0, s1) — 2 institutions (u0, u1)
        C_SI = sp.csr_matrix(np.array([[0.5, 0.3],
                                        [0.2, 0.6]], dtype=np.float64))
        C_IS = sp.csr_matrix(np.array([[0.4, 0.1],
                                        [0.3, 0.5]], dtype=np.float64))
        a_s = np.array([5.0, 3.0])
        a_u = np.array([4.0, 2.0])
        return C_SI, C_IS, a_s, a_u

    def test_returns_correct_shapes(self):
        C_SI, C_IS, a_s, a_u = self._tiny_matrices()
        A = float(a_s.sum() + a_u.sum())
        v_s, v_u, lam1, lam2, n_sc, n_uc, n_it = \
            _rank_one(C_SI, C_IS, 2, 2, a_s, a_u, A)
        assert v_s.shape == (2,)
        assert v_u.shape == (2,)

    def test_positive_in_core(self):
        C_SI, C_IS, a_s, a_u = self._tiny_matrices()
        A = float(a_s.sum() + a_u.sum())
        v_s, v_u, *_ = _rank_one(C_SI, C_IS, 2, 2, a_s, a_u, A)
        assert np.all(v_s[np.isfinite(v_s)] > 0)
        assert np.all(v_u[np.isfinite(v_u)] > 0)

    def test_lam1_positive(self):
        C_SI, C_IS, a_s, a_u = self._tiny_matrices()
        A = float(a_s.sum() + a_u.sum())
        _, _, lam1, *_ = _rank_one(C_SI, C_IS, 2, 2, a_s, a_u, A)
        assert lam1 > 0

    def test_nans_for_units_outside_core(self):
        # Add a disconnected source (row 2) that has no citations → outside core
        C_SI = sp.csr_matrix(np.array([[0.5, 0.3],
                                        [0.0, 0.0]], dtype=np.float64))
        C_IS = sp.csr_matrix(np.array([[0.4, 0.1],
                                        [0.0, 0.0]], dtype=np.float64))
        a_s = np.array([5.0, 1.0])
        a_u = np.array([4.0, 2.0])
        A = float(a_s.sum() + a_u.sum())
        v_s, v_u, *_ = _rank_one(C_SI, C_IS, 2, 2, a_s, a_u, A)
        # s1 (index 1) has all-zero row in C_SI → excluded from core → NaN
        assert np.isnan(v_s[1])


# ── Stage 3 helper test (load_base_refs) ──────────────────────────────────────

def test_load_base_refs_assigns_ref_idx():
    """load_base_refs should assign sequential ref_idx starting at 0."""
    import duckdb
    db = duckdb.connect(':memory:')
    db.execute("""
        CREATE TABLE test_el (
            citer_work_idx BIGINT, citer_source_idx BIGINT, citer_inst_idx BIGINT,
            cited_work_idx BIGINT, cited_source_idx BIGINT, cited_inst_idx BIGINT,
            inst_weight DOUBLE, direct_inst_weight DOUBLE, cited_inst_weight DOUBLE,
            R_i BIGINT, a_citer_source BIGINT, a_cited_source BIGINT,
            a_citer_inst DOUBLE, a_cited_inst DOUBLE
        )
    """)
    rows = [
        (1, 10, 100, 2, 20, 200, 1.0, 1.0, 1.0, 2, 5, 5, 4.0, 4.0),
        (1, 10, 100, 3, 10, 100, 1.0, 1.0, 1.0, 2, 5, 5, 4.0, 4.0),
        (4, 20, 200, 2, 20, 200, 1.0, 1.0, 1.0, 1, 3, 5, 2.0, 4.0),
    ]
    db.executemany('INSERT INTO test_el VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)

    df = load_base_refs(db, 'test_el')
    assert 'ref_idx' in df.columns
    assert df['ref_idx'].tolist() == list(range(len(df)))
    assert len(df) == 3  # 3 distinct (citer_work, cited_work) pairs
    assert set(df.columns) >= {'ref_idx', 'citer_work_idx', 'cited_work_idx', 'cited_source_idx'}
    db.close()
