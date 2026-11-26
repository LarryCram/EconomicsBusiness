"""
el_helpers.py — edge-list helpers shared by the work-level rankers
(rank_works.py, rank_works_katz_hubbell.py).

Relocated unchanged from spectral_ranking_bootstrap/bootstrap_baseline.py
(archived 2026-11-26).
"""


def compute_r_bar(db, el_table: str) -> float:
    """
    Compute r_bar = mean of distinct (citer_work_idx, R_i) pairs.
    Matches the ρ=0 fixed-count weighting in build_csr.py exactly.
    """
    r_bar = db.execute(
        f"SELECT AVG(rval) FROM "
        f"(SELECT DISTINCT citer_work_idx, CAST(R_i AS DOUBLE) AS rval FROM {el_table})"
    ).fetchone()[0]
    return float(r_bar)


def create_tmp_el(db, el_table: str, r_bar: float) -> None:
    """Materialise _tmp_el with rho_w column, exactly as build_csr.py does for rho=0.

    Uses direct 1/N_inst institution weights (omega=1).
    """
    db.execute(f"""
        CREATE OR REPLACE TEMP TABLE _tmp_el AS
        SELECT citer_work_idx, citer_source_idx, citer_inst_idx,
               cited_work_idx,  cited_source_idx, cited_inst_idx,
               direct_inst_weight       AS inst_weight,
               direct_cited_inst_weight AS cited_inst_weight,
               {r_bar} / CAST(R_i AS DOUBLE) AS rho_w
        FROM {el_table}
    """)
