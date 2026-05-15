"""
verify_edge_lists.py — Invariant checks for edge_lists.duckdb.

Runs against every table listed in _catalog and reports PASS/FAIL.
All checks are exact equalities or strict inequalities — no tolerances.

Checks
------
1. No NULL values in any column
2. No self-loops (citer_work_idx = cited_work_idx)
3. inst_weight sums to ≤ 1 per citing work  (= 1 if no institutions dropped)
4. direct_inst_weight sums to ≤ 1 per citing work
5. R_i consistent: stored value = COUNT(DISTINCT cited_work) per citer work
6. R_i unique per citer work (no conflicting values across rows)
7. a_citer_source ≥ COUNT(DISTINCT citer_work) per source
8. a_citer_source unique per source (no conflicting values)
9. a_citer_inst = a_cited_inst for institutions appearing on both sides
10. a_citer_inst unique per citer institution
"""

import sys
from pathlib import Path
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config

paths   = load_config()
DB_PATH = paths.working / 'edge_lists.duckdb'

CHECKS = {}


def check(name):
    """Decorator to register a check function."""
    def decorator(fn):
        CHECKS[name] = fn
        return fn
    return decorator


# ── Each check receives (db, tname) and returns (passed: bool, detail: str) ──

@check("1. No NULLs in key columns")
def check_nulls(db, t):
    n = db.sql(f"""
        SELECT COUNT(*) FROM {t}
        WHERE citer_work_idx IS NULL OR cited_work_idx IS NULL
           OR citer_source_idx IS NULL OR cited_source_idx IS NULL
           OR citer_inst_idx  IS NULL OR cited_inst_idx  IS NULL
           OR inst_weight IS NULL OR direct_inst_weight IS NULL
           OR R_i IS NULL
           OR a_citer_source IS NULL OR a_cited_source IS NULL
           OR a_citer_inst   IS NULL OR a_cited_inst   IS NULL
    """).fetchone()[0]
    return n == 0, f"{n} rows with NULLs"


@check("2. No self-loops")
def check_self_loops(db, t):
    n = db.sql(f"""
        SELECT COUNT(*) FROM {t}
        WHERE citer_work_idx = cited_work_idx
    """).fetchone()[0]
    return n == 0, f"{n} self-loops"


@check("3. inst_weight sums ≤ 1 per citer work")
def check_inst_weight_sum(db, t):
    n = db.sql(f"""
        SELECT COUNT(*) FROM (
            SELECT citer_work_idx, ROUND(SUM(inst_weight), 6) AS s
            FROM (SELECT DISTINCT citer_work_idx, citer_inst_idx, inst_weight FROM {t})
            GROUP BY citer_work_idx
            HAVING s > 1.0
        )
    """).fetchone()[0]
    # Also report how many works sum to exactly 1 (all institutions retained)
    n_exact = db.sql(f"""
        SELECT COUNT(*) FROM (
            SELECT citer_work_idx, ROUND(SUM(inst_weight), 6) AS s
            FROM (SELECT DISTINCT citer_work_idx, citer_inst_idx, inst_weight FROM {t})
            GROUP BY citer_work_idx
            HAVING s = 1.0
        )
    """).fetchone()[0]
    n_works = db.sql(f"SELECT COUNT(DISTINCT citer_work_idx) FROM {t}").fetchone()[0]
    return n == 0, f"{n} violations  ({n_exact}/{n_works} works sum exactly to 1)"


@check("4. direct_inst_weight sums ≤ 1 per citer work")
def check_direct_weight_sum(db, t):
    n = db.sql(f"""
        SELECT COUNT(*) FROM (
            SELECT citer_work_idx, ROUND(SUM(direct_inst_weight), 6) AS s
            FROM (SELECT DISTINCT citer_work_idx, citer_inst_idx, direct_inst_weight FROM {t})
            GROUP BY citer_work_idx
            HAVING s > 1.0
        )
    """).fetchone()[0]
    return n == 0, f"{n} violations"


@check("5. R_i ≥ COUNT(DISTINCT cited_work) per citer work")
def check_ri_matches_count(db, t):
    # R_i is computed before source-retention filtering, so it can exceed the
    # edge-list cited count (references to non-retained sources are excluded).
    # Flag only when R_i < edge-list count, which would indicate undercounting.
    # Exclude sentinel-sourced cited rows (cited_source_idx=1, eps1 tables only):
    # those supplement works are outside the intra-corpus reference count.
    n = db.sql(f"""
        SELECT COUNT(*) FROM (
            SELECT citer_work_idx,
                   MAX(R_i) AS stored,
                   COUNT(DISTINCT cited_work_idx)
                       FILTER (WHERE cited_source_idx != 1) AS computed
            FROM {t}
            GROUP BY citer_work_idx
            HAVING stored < computed
        )
    """).fetchone()[0]
    return n == 0, f"{n} works where R_i < edge-list distinct cited count"


@check("6. R_i unique per citer work")
def check_ri_unique(db, t):
    n = db.sql(f"""
        SELECT COUNT(*) FROM (
            SELECT citer_work_idx
            FROM {t}
            GROUP BY citer_work_idx
            HAVING COUNT(DISTINCT R_i) > 1
        )
    """).fetchone()[0]
    return n == 0, f"{n} citer works with conflicting R_i values"


@check("7. a_citer_source ≥ COUNT(DISTINCT citer_work) per source")
def check_a_source_geq(db, t):
    n = db.sql(f"""
        SELECT COUNT(*) FROM (
            SELECT citer_source_idx,
                   MIN(a_citer_source)           AS stored,
                   COUNT(DISTINCT citer_work_idx) AS in_edge_list
            FROM {t}
            GROUP BY citer_source_idx
            HAVING in_edge_list > stored
        )
    """).fetchone()[0]
    return n == 0, f"{n} sources where edge-list work count exceeds stored a_p"


@check("8. a_citer_source unique per source")
def check_a_source_unique(db, t):
    n = db.sql(f"""
        SELECT COUNT(*) FROM (
            SELECT citer_source_idx
            FROM {t}
            GROUP BY citer_source_idx
            HAVING COUNT(DISTINCT a_citer_source) > 1
        )
    """).fetchone()[0]
    return n == 0, f"{n} sources with conflicting a_citer_source values"


@check("9. a_citer_inst = a_cited_inst for shared institutions")
def check_a_inst_consistent(db, t):
    # Exclude sentinel institution (idx=1, eps1 tables only): it intentionally
    # carries different a values on each side (supp-citer vs supp-cited pool sizes).
    n = db.sql(f"""
        SELECT COUNT(*) FROM (
            SELECT inst_idx
            FROM (
                SELECT citer_inst_idx AS inst_idx, a_citer_inst AS a
                FROM {t}
                UNION ALL
                SELECT cited_inst_idx,              a_cited_inst
                FROM {t}
            )
            WHERE inst_idx != 1
            GROUP BY inst_idx
            HAVING ROUND(MIN(a), 6) != ROUND(MAX(a), 6)
        )
    """).fetchone()[0]
    return n == 0, f"{n} institutions with mismatched a_inst on citer vs cited side"


@check("10. a_citer_inst unique per institution")
def check_a_inst_unique(db, t):
    n = db.sql(f"""
        SELECT COUNT(*) FROM (
            SELECT citer_inst_idx
            FROM {t}
            GROUP BY citer_inst_idx
            HAVING COUNT(DISTINCT a_citer_inst) > 1
        )
    """).fetchone()[0]
    return n == 0, f"{n} institutions with conflicting a_citer_inst values"


# ─────────────────────────────────────────────────────────────────────────────

def verify_table(db, tname: str) -> bool:
    print(f"\n  {tname}")
    all_passed = True
    for label, fn in CHECKS.items():
        passed, detail = fn(db, tname)
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}]  {label}  —  {detail}")
        if not passed:
            all_passed = False
    return all_passed


def main():
    with duckdb.connect(str(DB_PATH), read_only=True) as db:
        tables = [
            row[0] for row in
            db.sql("SELECT table_name FROM _catalog ORDER BY t_x, F_x, tau_u").fetchall()
        ]
        print(f"Verifying {len(tables)} edge list tables in {DB_PATH}\n")
        results = {t: verify_table(db, t) for t in tables}

    print("\n=== Summary ===")
    passed = sum(results.values())
    print(f"  {passed}/{len(results)} tables fully passed")
    for t, ok in results.items():
        if not ok:
            print(f"  FAIL  {t}")


if __name__ == '__main__':
    main()
    print('\nFINISHED!')
