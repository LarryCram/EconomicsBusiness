"""
exogenous_source_count.py — Exogenous reference fraction per OAS source.

For each source in OAS, counts what fraction of references emitted by its
works (published 2010–2024) point to works outside OAS (any pub year).

"Outside OAS" means the cited work's source is not in source_master.parquet.
Works whose source cannot be found in the snapshot (rare) are counted as
exogenous.

Output: printed table, sorted by exogenous fraction ASC (highest at bottom).

Data sources
------------
  WORKING/parquet/corpus_works.parquet      — citer works (already OAS-filtered)
  WORKING/parquet/source_master.parquet     — OAS source set
  OPENALEX/references/*.parquet             — all snapshot references
  OPENALEX/works/*.parquet                  — source lookup for cited works
"""

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config

CITER_YEAR_MIN = 2010
CITER_YEAR_MAX = 2024


def main():
    paths = load_config()
    PARQUET  = paths.parquet
    OA       = paths.openalex

    with duckdb.connect() as db:
        db.execute(f"SET temp_directory = '{paths.working}/.tmp'")
        db.execute("SET memory_limit = '56GB'")
        db.execute("SET preserve_insertion_order = false")

        print("Step 1: materialise citer works (OAS, 2010–2024) ...")
        db.execute(f"""
            CREATE TEMP TABLE _citers AS
            SELECT work_idx, source_idx
            FROM '{PARQUET}/corpus_works.parquet'
            WHERE publication_year BETWEEN {CITER_YEAR_MIN} AND {CITER_YEAR_MAX}
        """)
        n_citers = db.execute("SELECT COUNT(*) FROM _citers").fetchone()[0]
        print(f"  {n_citers:,} citer works")

        print("Step 2: pull all references from those citers (raw snapshot) ...")
        db.execute(f"""
            CREATE TEMP TABLE _refs AS
            SELECT r.citer_idx, r.cited_idx
            FROM '{OA}/references/*.parquet' r
            JOIN _citers c ON r.citer_idx = c.work_idx
        """)
        n_refs = db.execute("SELECT COUNT(*) FROM _refs").fetchone()[0]
        print(f"  {n_refs:,} references")

        print("Step 3: look up source of each distinct cited work (raw snapshot) ...")
        # Materialise distinct cited_idx first so we scan works/*.parquet
        # with a small probe set rather than a full cross join.
        db.execute("""
            CREATE TEMP TABLE _cited_ids AS
            SELECT DISTINCT cited_idx FROM _refs
        """)
        n_cited = db.execute("SELECT COUNT(*) FROM _cited_ids").fetchone()[0]
        print(f"  {n_cited:,} distinct cited works")

        # Probe the snapshot schema to find the source column name
        sample = db.execute(f"SELECT * FROM '{OA}/works/*.parquet' LIMIT 0").description
        snap_cols = {d[0] for d in sample}
        if 'source_idx' in snap_cols:
            src_col = 'source_idx'
        else:
            # Fallback: string source_id needs stripping
            src_col = "CAST(regexp_replace(source_id, 'https://openalex.org/S', '') AS BIGINT)"

        db.execute(f"""
            CREATE TEMP TABLE _cited_sources AS
            SELECT work_idx, {src_col} AS source_idx
            FROM '{OA}/works/*.parquet'
            WHERE work_idx IN (SELECT cited_idx FROM _cited_ids)
              AND {src_col} IS NOT NULL
        """)

        print("Step 4: compute exogenous fraction per citer source ...")
        result = db.execute(f"""
            WITH oas AS (SELECT source_idx FROM '{PARQUET}/source_master.parquet')
            SELECT
                c.source_idx                                          AS source_idx,
                sm.source_name,
                sm.era_journal_name,
                sm.harzing_journal_name,
                sm.wos_journal_name,
                COUNT(*)                                              AS n_refs,
                COUNT(*) FILTER (
                    WHERE cs.source_idx IS NULL
                       OR cs.source_idx NOT IN (SELECT source_idx FROM oas)
                )                                                     AS n_exogenous,
                ROUND(
                    COUNT(*) FILTER (
                        WHERE cs.source_idx IS NULL
                           OR cs.source_idx NOT IN (SELECT source_idx FROM oas)
                    ) * 1.0 / COUNT(*), 4
                )                                                     AS frac_exogenous
            FROM _refs r
            JOIN _citers c ON r.citer_idx = c.work_idx
            LEFT JOIN _cited_sources cs ON r.cited_idx = cs.work_idx
            LEFT JOIN '{PARQUET}/source_master.parquet' sm ON c.source_idx = sm.source_idx
            GROUP BY c.source_idx, sm.source_name,
                     sm.era_journal_name, sm.harzing_journal_name, sm.wos_journal_name
            ORDER BY frac_exogenous ASC
        """).df()

    print(f"\n{'source_idx':>12}  {'n_refs':>8}  {'n_exog':>8}  {'frac_exog':>10}  "
          f"{'source_name':<40}  era / harzing / wos")
    print("-" * 130)
    for _, row in result.iterrows():
        names = "  /  ".join(
            str(row[c]) if row[c] and str(row[c]) != 'None' else '-'
            for c in ('era_journal_name', 'harzing_journal_name', 'wos_journal_name')
        )
        print(f"{int(row['source_idx']):>12}  {int(row['n_refs']):>8,}  "
              f"{int(row['n_exogenous']):>8,}  {row['frac_exogenous']:>10.4f}  "
              f"{str(row['source_name']):<40}  {names}")

    print(f"\n{len(result)} sources.  Sorted ASC by exogenous fraction "
          f"(highest exogenous at bottom).")

    out = paths.data / 'exogenous_source_count.csv'
    result.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
    print("FINISHED!")
