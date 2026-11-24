"""
apply_institution_corrections.py — Apply the institution (and journal) correction tables to frozen corpus parquets.

QSS revision. Replaces the role of INSTITUTION_EXCLUSIONS in load_corpus_entities.py
without touching OpenAlex: it reads the frozen corpus parquets (the locked
WORKING/parquet folder) and writes corrected copies into the parquet folder of the
active config (normally config.rev.yaml → econ_bus_rev). Downstream stages
(build_edge_lists, verify_edge_lists, build_institution_field_eb,
filter_mode_units, run_rankings) then run unchanged against the corrected copies.

Correction table: data/institution_corrections.csv
  old_idx, action (merge|drop), new_idx, ... , status
Only rows with status 'accepted' are applied.

merge  : authorships of old_idx are relabelled to new_idx (name, ror, country taken
         from new_idx) in corpus_authorships and corpus_authorships_supp. A relabelled
         row is dropped if the same (work, author, new_idx) authorship already exists,
         so author-fractional weights stay correct. old_idx is removed from
         corpus_institutions and works_count / works_per_year of new_idx recomputed.
drop   : old_idx is removed from corpus_institutions (its authorships stay, exactly as
         with the INSTITUTION_EXCLUSIONS filter; they are dropped at edge-list build).

Journal merges (data/source_corrections.csv): works of old_idx are relabelled to new_idx
(and new_name) in corpus_works / corpus_works_supp; in source_master the old row is
removed and the target renamed, its works/citation counts summed, and its registry
signals (ISSN, registry names and categories, scores, field_eb) taken from the old
record when metadata_from == 'old'. Registry-stage parquets (oas_star,
source_densities, dropped_sources) describe journal matching and are left unchanged.

--empty copies every parquet unchanged (byte-identical): the reproduction check that
the rerun pipeline reproduces the locked baseline before any correction is applied.

Usage:
  ECONBUS_CONFIG=config.rev.yaml .venv/bin/python prepare_data/apply_institution_corrections.py \
      --source /home/lc/m/working/econ_bus/parquet [--empty]
"""

import argparse
import shutil
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config

CORPUS_YEARS = 25   # 2000–2024, as in load_corpus_entities.py
FORBIDDEN = ('econ_bus_qss_submitted',)
CHANGED = ('corpus_authorships.parquet', 'corpus_authorships_supp.parquet',
           'corpus_institutions.parquet', 'corpus_works.parquet',
           'corpus_works_supp.parquet', 'source_master.parquet')
# source_master columns taken from the merged (old) record when metadata_from == 'old'
REGISTRY_COLS = ('issn', 'era_journal_name', 'era_for_codes', 'harzing_journal_name',
                 'harzing_field', 'wos_journal_name', 'wos_categories', 'scopus_journal_name',
                 'scopus_asjc', 'econ_score', 'bus_score', 'field_eb')


def load_table(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d['status'] == 'accepted'].copy()
    d['old_idx'] = d['old_idx'].astype('int64')
    d['new_idx'] = pd.to_numeric(d['new_idx'], errors='coerce').astype('Int64')
    assert d['old_idx'].is_unique, 'old_idx must be unique'
    m = d[d.action == 'merge']
    assert m['new_idx'].notna().all(), 'every merge needs new_idx'
    assert not set(m['new_idx'].astype('int64')) & set(d['old_idx']), \
        'a merge target is itself corrected (chain) — resolve the chain in the table'
    assert set(d.action) <= {'merge', 'drop'}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='frozen parquet folder (read only)')
    ap.add_argument('--corrections', default=None)
    ap.add_argument('--source-corrections', default=None,
                    help='journal merges (default data/source_corrections.csv)')
    ap.add_argument('--empty', action='store_true', help='copy unchanged (reproduction check)')
    args = ap.parse_args()

    paths = load_config()
    src = Path(args.source).resolve()
    dst = paths.parquet.resolve()
    assert src != dst, 'target parquet folder equals the source'
    assert not any(f in str(dst) for f in FORBIDDEN), f'refusing to write into {dst}'
    assert dst.parent.name != 'econ_bus', 'refusing to write into the locked econ_bus folder'
    dst.mkdir(parents=True, exist_ok=True)

    # 1. copy every parquet (unchanged files byte-identical)
    for f in sorted(src.glob('*.parquet')):
        if args.empty or f.name not in CHANGED:
            shutil.copy2(f, dst / f.name)
    if args.empty:
        print(f'--empty: copied {len(list(src.glob("*.parquet")))} parquets unchanged to {dst}')
        return

    corr = load_table(Path(args.corrections) if args.corrections
                      else paths.data / 'institution_corrections.csv')
    merges = corr[corr.action == 'merge'][['old_idx', 'new_idx']].astype('int64')
    drops = corr[corr.action == 'drop'][['old_idx']]
    con = duckdb.connect()
    con.register('m', merges)
    con.register('dr', drops)
    ci = f"'{src}/corpus_institutions.parquet'"

    missing = con.execute(f"SELECT new_idx FROM m WHERE new_idx NOT IN "
                          f"(SELECT institution_idx FROM {ci})").fetchall()
    assert not missing, f'merge targets not in corpus_institutions: {missing}'

    # 2. corpus_authorships: relabel merges, drop duplicate relabelled authorships
    con.execute(f"""
        COPY (
            WITH a AS (
                SELECT *, row_number() OVER () AS rn
                FROM '{src}/corpus_authorships.parquet'),
            r AS (
                SELECT a.rn, a.work_idx, a.author_idx, a.author_name,
                       COALESCE(m.new_idx, a.institution_idx)                  AS institution_idx,
                       CASE WHEN m.new_idx IS NULL THEN a.institution_name
                            ELSE t.institution_name END                       AS institution_name,
                       CASE WHEN m.new_idx IS NULL THEN a.ror ELSE t.ror END   AS ror,
                       CASE WHEN m.new_idx IS NULL THEN a.country_code
                            ELSE t.country_code END                           AS country_code,
                       m.new_idx IS NOT NULL                                  AS relabelled
                FROM a
                LEFT JOIN m ON a.institution_idx = m.old_idx
                LEFT JOIN {ci} t ON t.institution_idx = m.new_idx),
            keep AS (
                SELECT * FROM r
                WHERE NOT relabelled
                   OR (NOT EXISTS (SELECT 1 FROM r o
                                   WHERE NOT o.relabelled AND o.work_idx = r.work_idx
                                     AND o.author_idx = r.author_idx
                                     AND o.institution_idx = r.institution_idx)
                       AND rn = (SELECT MIN(r2.rn) FROM r r2
                                 WHERE r2.relabelled AND r2.work_idx = r.work_idx
                                   AND r2.author_idx = r.author_idx
                                   AND r2.institution_idx = r.institution_idx)))
            SELECT work_idx, author_idx, author_name, institution_idx,
                   institution_name, ror, country_code
            FROM keep ORDER BY rn
        ) TO '{dst}/corpus_authorships.parquet' (FORMAT PARQUET)
    """)

    # 3. corpus_authorships_supp: same relabel / dedupe
    con.execute(f"""
        COPY (
            WITH a AS (
                SELECT *, row_number() OVER () AS rn
                FROM '{src}/corpus_authorships_supp.parquet'),
            r AS (
                SELECT a.rn, a.work_idx, a.author_idx,
                       COALESCE(m.new_idx, a.institution_idx) AS institution_idx,
                       a.supp_role, m.new_idx IS NOT NULL AS relabelled
                FROM a LEFT JOIN m ON a.institution_idx = m.old_idx),
            keep AS (
                SELECT * FROM r
                WHERE NOT relabelled
                   OR (NOT EXISTS (SELECT 1 FROM r o
                                   WHERE NOT o.relabelled AND o.work_idx = r.work_idx
                                     AND o.author_idx = r.author_idx
                                     AND o.institution_idx = r.institution_idx
                                     AND o.supp_role IS NOT DISTINCT FROM r.supp_role)
                       AND rn = (SELECT MIN(r2.rn) FROM r r2
                                 WHERE r2.relabelled AND r2.work_idx = r.work_idx
                                   AND r2.author_idx = r.author_idx
                                   AND r2.institution_idx = r.institution_idx
                                   AND r2.supp_role IS NOT DISTINCT FROM r.supp_role)))
            SELECT work_idx, author_idx, institution_idx, supp_role
            FROM keep ORDER BY rn
        ) TO '{dst}/corpus_authorships_supp.parquet' (FORMAT PARQUET)
    """)

    # 4. corpus_institutions: remove merged and dropped ids, recount merge targets
    con.execute(f"""
        COPY (
            WITH cnt AS (
                SELECT institution_idx, COUNT(DISTINCT work_idx) AS works_count
                FROM '{dst}/corpus_authorships.parquet'
                WHERE institution_idx IN (SELECT new_idx FROM m)
                GROUP BY 1)
            SELECT ci.institution_idx, ci.institution_name, ci.country_code, ci."type", ci.ror,
                   COALESCE(cnt.works_count, ci.works_count)                       AS works_count,
                   CASE WHEN cnt.works_count IS NULL THEN ci.works_per_year
                        ELSE ROUND(cnt.works_count / {CORPUS_YEARS}.0, 4) END     AS works_per_year
            FROM {ci} ci
            LEFT JOIN cnt USING (institution_idx)
            WHERE ci.institution_idx NOT IN (SELECT old_idx FROM m)
              AND ci.institution_idx NOT IN (SELECT old_idx FROM dr)
            ORDER BY works_count DESC
        ) TO '{dst}/corpus_institutions.parquet' (FORMAT PARQUET)
    """)

    # 5. journal (source) merges: relabel works, merge source_master rows
    sc = pd.read_csv(Path(args.source_corrections) if args.source_corrections
                     else paths.data / 'source_corrections.csv')
    sc = sc[sc.status == 'accepted']
    assert set(sc.action) <= {'merge'} and sc.old_idx.is_unique
    assert not set(sc.new_idx) & set(sc.old_idx), 'chained source merges'
    con.register('sm_m', sc[['old_idx', 'new_idx', 'new_name', 'metadata_from']]
                 .astype({'old_idx': 'int64', 'new_idx': 'int64'}))
    con.execute(f"""
        COPY (
            SELECT w.* REPLACE (COALESCE(m.new_idx, w.source_idx) AS source_idx,
                                COALESCE(m.new_name, tg.new_name, w.source_name) AS source_name)
            FROM (SELECT *, row_number() OVER () AS _rn
                  FROM '{src}/corpus_works.parquet') w
            LEFT JOIN sm_m m ON w.source_idx = m.old_idx
            LEFT JOIN (SELECT DISTINCT new_idx, new_name FROM sm_m) tg
                   ON w.source_idx = tg.new_idx
            ORDER BY w._rn
        ) TO '{dst}/_tmp_works.parquet' (FORMAT PARQUET)
    """)
    con.execute(f"""COPY (SELECT * EXCLUDE (_rn) FROM '{dst}/_tmp_works.parquet')
                    TO '{dst}/corpus_works.parquet' (FORMAT PARQUET)""")
    (dst / '_tmp_works.parquet').unlink()
    con.execute(f"""
        COPY (
            SELECT w.* REPLACE (COALESCE(m.new_idx, w.source_idx) AS source_idx)
            FROM (SELECT *, row_number() OVER () AS _rn
                  FROM '{src}/corpus_works_supp.parquet') w
            LEFT JOIN sm_m m ON w.source_idx = m.old_idx
            ORDER BY w._rn
        ) TO '{dst}/_tmp_supp.parquet' (FORMAT PARQUET)
    """)
    con.execute(f"""COPY (SELECT * EXCLUDE (_rn) FROM '{dst}/_tmp_supp.parquet')
                    TO '{dst}/corpus_works_supp.parquet' (FORMAT PARQUET)""")
    (dst / '_tmp_supp.parquet').unlink()
    sm_cols = [r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM '{src}/source_master.parquet'").fetchall()]
    repl = [f"CASE WHEN m.metadata_from = 'old' THEN o.{c} ELSE t.{c} END AS {c}"
            for c in REGISTRY_COLS if c in sm_cols]
    repl.append("COALESCE(m.new_name, t.source_name) AS source_name")
    for c in ('works_count', 'cited_by_count'):
        if c in sm_cols:
            repl.append(f"t.{c} + COALESCE(o.{c}, 0) AS {c}")
    con.execute(f"""
        COPY (
            SELECT t.* REPLACE ({', '.join(repl)})
            FROM '{src}/source_master.parquet' t
            LEFT JOIN sm_m m ON t.source_idx = m.new_idx
            LEFT JOIN '{src}/source_master.parquet' o ON o.source_idx = m.old_idx
            WHERE t.source_idx NOT IN (SELECT old_idx FROM sm_m)
        ) TO '{dst}/source_master.parquet' (FORMAT PARQUET)
    """)

    # 6. report
    def n(path, q='COUNT(*)'):
        return con.execute(f"SELECT {q} FROM '{path}'").fetchone()[0]
    for f in CHANGED:
        print(f'{f}: {n(src / f):,} -> {n(dst / f):,} rows')
    print(f'applied {len(merges)} institution merges, {len(drops)} drops and '
          f'{len(sc)} journal merges; corrected parquets in {dst}')


if __name__ == '__main__':
    main()
