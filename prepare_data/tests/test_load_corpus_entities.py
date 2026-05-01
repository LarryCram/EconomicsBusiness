"""
Tests for load_corpus_entities.py.

Uses in-memory DuckDB with synthetic parquet-like tables so no real data is needed.
Covers: institution type filtering, LEFT→INNER join change, works/authorships loading.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import duckdb
import tempfile
import os
from load_corpus_entities import INSTITUTION_TYPES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_db_with_tables(tmp_path):
    """
    Create an in-memory DuckDB with synthetic authorships and institutions
    tables that mimic the real parquet schema.
    """
    db = duckdb.connect(':memory:')

    # Synthetic corpus_authorships
    db.execute("""
        CREATE TABLE authorships (
            work_idx         BIGINT,
            author_idx       BIGINT,
            author_name      VARCHAR,
            institution_idx  BIGINT,
            institution_name VARCHAR,
            ror              VARCHAR,
            country_code     VARCHAR
        )
    """)
    db.execute("""
        INSERT INTO authorships VALUES
        -- education institution (should be retained)
        (101, 1, 'Alice',  10, 'State University',  NULL, 'US'),
        (102, 2, 'Bob',    10, 'State University',  NULL, 'US'),
        (103, 3, 'Carol',  20, 'Think Tank',        NULL, 'US'),
        -- nonprofit (should be retained)
        (104, 4, 'Dave',   20, 'Think Tank',        NULL, 'US'),
        -- government (should be retained)
        (105, 5, 'Eve',    30, 'Central Bank',      NULL, 'US'),
        -- company (should be excluded)
        (106, 6, 'Frank',  40, 'Pharma Corp',       NULL, 'US'),
        (107, 7, 'Grace',  40, 'Pharma Corp',       NULL, 'US'),
        -- healthcare (should be excluded)
        (108, 8, 'Heidi',  50, 'City Hospital',     NULL, 'US'),
        -- other (should be retained — e.g. Federal Reserve Banks)
        (109, 9, 'Ivan',   70, 'Federal Reserve',   NULL, 'US'),
        -- archive (should be excluded)
        (110, 10, 'Judy',  60, 'National Library',  NULL, 'US'),
        -- NULL institution_idx (should be excluded at authorship level)
        (111, 11, 'Karl',  NULL, NULL,              NULL, 'US')
    """)

    # Synthetic OpenAlex institutions
    db.execute("""
        CREATE TABLE oa_institutions (
            id           VARCHAR,
            display_name VARCHAR,
            country_code VARCHAR,
            type         VARCHAR
        )
    """)
    db.execute("""
        INSERT INTO oa_institutions VALUES
        ('https://openalex.org/I10', 'State University', 'US', 'education'),
        ('https://openalex.org/I20', 'Think Tank',       'US', 'nonprofit'),
        ('https://openalex.org/I30', 'Central Bank',     'US', 'government'),
        ('https://openalex.org/I40', 'Pharma Corp',      'US', 'company'),
        ('https://openalex.org/I50', 'City Hospital',    'US', 'healthcare'),
        ('https://openalex.org/I60', 'National Library', 'US', 'archive'),
        ('https://openalex.org/I70', 'Federal Reserve',  'US', 'other')
    """)

    return db


def _run_institution_query(db, corpus_years=25):
    """Run the load_institutions logic against the synthetic tables."""
    type_list = ', '.join(f"'{t}'" for t in INSTITUTION_TYPES)
    return db.execute(f"""
        WITH inst_works AS (
            SELECT institution_idx,
                   COUNT(DISTINCT work_idx) AS works_count
            FROM authorships
            WHERE institution_idx IS NOT NULL
            GROUP BY institution_idx
        ),
        inst_meta AS (
            SELECT CAST(REGEXP_REPLACE(id, 'https://openalex.org/I', '') AS BIGINT)
                       AS institution_idx,
                   display_name AS institution_name,
                   country_code,
                   type
            FROM oa_institutions
            WHERE type IN ({type_list})
        )
        SELECT iw.institution_idx,
               im.institution_name,
               im.country_code,
               im.type,
               iw.works_count,
               ROUND(iw.works_count / {corpus_years}.0, 4) AS works_per_year
        FROM inst_works iw
        INNER JOIN inst_meta im USING (institution_idx)
        ORDER BY iw.works_count DESC
    """).fetchdf()


# ── INSTITUTION_TYPES constant ────────────────────────────────────────────────

def test_institution_types_contains_education():
    assert 'education' in INSTITUTION_TYPES

def test_institution_types_contains_nonprofit():
    assert 'nonprofit' in INSTITUTION_TYPES

def test_institution_types_contains_government():
    assert 'government' in INSTITUTION_TYPES

def test_institution_types_excludes_company():
    assert 'company' not in INSTITUTION_TYPES

def test_institution_types_excludes_healthcare():
    assert 'healthcare' not in INSTITUTION_TYPES

def test_institution_types_contains_other():
    assert 'other' in INSTITUTION_TYPES

def test_institution_types_excludes_archive():
    assert 'archive' not in INSTITUTION_TYPES


# ── Type filtering ─────────────────────────────────────────────────────────────

def test_only_allowed_types_retained():
    db = _make_db_with_tables(None)
    result = _run_institution_query(db)
    assert set(result['type'].unique()) <= set(INSTITUTION_TYPES)

def test_company_excluded():
    db = _make_db_with_tables(None)
    result = _run_institution_query(db)
    assert 40 not in result['institution_idx'].values   # Pharma Corp

def test_healthcare_excluded():
    db = _make_db_with_tables(None)
    result = _run_institution_query(db)
    assert 50 not in result['institution_idx'].values   # City Hospital

def test_archive_excluded():
    db = _make_db_with_tables(None)
    result = _run_institution_query(db)
    assert 60 not in result['institution_idx'].values   # National Library

def test_other_retained():
    db = _make_db_with_tables(None)
    result = _run_institution_query(db)
    assert 70 in result['institution_idx'].values       # Federal Reserve

def test_education_retained():
    db = _make_db_with_tables(None)
    result = _run_institution_query(db)
    assert 10 in result['institution_idx'].values       # State University

def test_nonprofit_retained():
    db = _make_db_with_tables(None)
    result = _run_institution_query(db)
    assert 20 in result['institution_idx'].values       # Think Tank

def test_government_retained():
    db = _make_db_with_tables(None)
    result = _run_institution_query(db)
    assert 30 in result['institution_idx'].values       # Central Bank

def test_retained_count():
    """Exactly 4 institutions should survive: education, nonprofit, government, other."""
    db = _make_db_with_tables(None)
    result = _run_institution_query(db)
    assert len(result) == 4


# ── INNER JOIN: no orphan rows ────────────────────────────────────────────────

def test_no_null_institution_names():
    """INNER JOIN means institutions with no OA metadata are dropped, not kept with NULLs."""
    db = _make_db_with_tables(None)
    # Add an authorship for an institution not in oa_institutions
    db.execute("INSERT INTO authorships VALUES (200, 99, 'Zara', 999, 'Ghost Inst', NULL, 'XX')")
    result = _run_institution_query(db)
    assert result['institution_name'].isna().sum() == 0
    assert 999 not in result['institution_idx'].values


# ── works_count and works_per_year ────────────────────────────────────────────

def test_works_count_correct():
    """State University (idx=10) appears in works 101 and 102 → works_count=2."""
    db = _make_db_with_tables(None)
    result = _run_institution_query(db)
    row = result[result['institution_idx'] == 10].iloc[0]
    assert int(row['works_count']) == 2

def test_works_per_year_correct():
    db = _make_db_with_tables(None)
    result = _run_institution_query(db, corpus_years=25)
    row = result[result['institution_idx'] == 10].iloc[0]
    assert abs(row['works_per_year'] - 2 / 25.0) < 1e-4

def test_null_institution_idx_not_counted():
    """The authorship with institution_idx=NULL (work 110) must not inflate any count."""
    db = _make_db_with_tables(None)
    result = _run_institution_query(db)
    assert result['works_count'].sum() < 10   # 9 valid authorships, 3 retained institutions


# ── build_edge_lists retained_inst gate ──────────────────────────────────────

def _run_retained_inst_query(db, corpus_institutions_idxs, tau_u=1, census_years=25):
    """
    Simulate the retained_inst CTE from build_edge_lists.py:
    counts works per institution in the census window, gates on corpus_institutions,
    and filters by tau_u.  corpus_institutions_idxs is the set of allowed idx values
    (the output of load_institutions after type filtering).
    """
    id_list = ', '.join(str(i) for i in corpus_institutions_idxs)
    return db.execute(f"""
        SELECT institution_idx, COUNT(DISTINCT work_idx) AS n_works
        FROM authorships
        WHERE institution_idx IS NOT NULL
          AND institution_idx IN ({id_list})
        GROUP BY institution_idx
        HAVING COUNT(DISTINCT work_idx) / {census_years}.0 >= {tau_u}
    """).fetchdf()


def test_edge_list_gate_excludes_company():
    """Company institutions absent from corpus_institutions must not enter retained_inst."""
    db = _make_db_with_tables(None)
    allowed = [10, 20, 30, 70]   # education, nonprofit, government, other
    result = _run_retained_inst_query(db, allowed, tau_u=0)
    assert 40 not in result['institution_idx'].values   # Pharma Corp (company)

def test_edge_list_gate_excludes_healthcare():
    db = _make_db_with_tables(None)
    allowed = [10, 20, 30, 70]
    result = _run_retained_inst_query(db, allowed, tau_u=0)
    assert 50 not in result['institution_idx'].values   # City Hospital (healthcare)

def test_edge_list_gate_retains_education():
    db = _make_db_with_tables(None)
    allowed = [10, 20, 30, 70]
    result = _run_retained_inst_query(db, allowed, tau_u=0)
    assert 10 in result['institution_idx'].values       # State University

def test_edge_list_gate_retains_other():
    db = _make_db_with_tables(None)
    allowed = [10, 20, 30, 70]
    result = _run_retained_inst_query(db, allowed, tau_u=0)
    assert 70 in result['institution_idx'].values       # Federal Reserve

def test_edge_list_gate_corpus_institutions_is_sole_type_gate():
    """
    retained_inst inherits the type filter solely from the corpus_institutions
    allow-list — no separate type logic needed in build_edge_lists.
    """
    db = _make_db_with_tables(None)
    allowed = [10, 20, 30, 70]
    result = _run_retained_inst_query(db, allowed, tau_u=0)
    assert set(result['institution_idx'].tolist()) <= set(allowed)


# ── Ordering ──────────────────────────────────────────────────────────────────

def test_ordered_by_works_count_desc():
    db = _make_db_with_tables(None)
    result = _run_institution_query(db)
    counts = result['works_count'].tolist()
    assert counts == sorted(counts, reverse=True)
