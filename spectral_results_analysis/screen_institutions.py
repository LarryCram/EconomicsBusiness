"""
screen_institutions.py — Screen baseline institutions for OpenAlex identity problems.

QSS revision. Three screens on the frozen corpus (no OpenAlex access):

1. Home screen: give every author a home institution (the institution they
   appear under most often, 2015–2024, authors with >= 3 authorships). Flag a
   baseline institution X when >= 25% of its 2020–24 authorships come from
   authors whose home is one particular other institution Y. Catches
   misattributions (Harvard University Press <- Harvard), duplicate ids
   (system vs campus) and schools filed separately from their university.

2. Name screen: pairs (X, Y) where X is a baseline institution and Y another
   corpus institution with >= 20 corpus works, and either the normalised names
   are equal (Centre/Center, '&'/'and', stop words removed), one normalised
   name contains the other (>= 2 words), or Jaro–Winkler >= 0.93. Catches
   affiliation bodies with near-identical names (CEPR GB vs CEPR US), which
   the home screen misses because their authors' home is a university.

3. Country screen: flag X when >= 50% of its 2020–24 authorships come from
   authors whose home institution is in another country.

Output: data/institution_screen.csv (one row per flagged institution, with all
flags and the best partner), printed summaries.

Usage:
  .venv/bin/python spectral_results_analysis/screen_institutions.py
"""

import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

HOME_SHARE = 0.25
COUNTRY_SHARE = 0.50
JW_MIN = 0.93
MIN_PARTNER_WORKS = 20
STOP = {'of', 'the', 'and', 'for', 'de', 'des', 'la', 'le', 'du', 'di', 'at', 'in'}


def norm(name: str) -> str:
    s = name.lower().replace('&', ' and ').replace('centre', 'center')
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    return ' '.join(w for w in s.split() if w not in STOP)


def main():
    paths = load_config()
    P = paths.parquet
    bl = next(r for r in load_runs() if r['label'] == 'baseline')
    t = (f"rk_{bl['run_code']}_{bl['fx']}_tauU{bl['tau_u']}_tauS{bl['tau_s']}"
         f"_vartau_rho0_m0110_chi50_alpha100")
    with duckdb.connect(str(paths.working / 'rankings.duckdb'), read_only=True) as rk:
        b = rk.execute(f"SELECT unit_idx, v, a_p FROM {t} WHERE unit_type='U'").df()
    b = b.sort_values('v', ascending=False).reset_index(drop=True)
    b['irank'] = b.index + 1

    con = duckdb.connect()
    con.register('b', b)
    ci = con.execute(f"""
        SELECT institution_idx, institution_name, country_code, "type" AS itype,
               works_count FROM '{P}/corpus_institutions.parquet'""").df()
    con.register('ci', ci)

    con.execute(f"""
        CREATE TEMP TABLE a AS
        SELECT DISTINCT au.work_idx, au.author_idx, au.institution_idx, w.publication_year AS y
        FROM '{P}/corpus_authorships.parquet' au
        JOIN '{P}/corpus_works.parquet' w USING (work_idx)
        WHERE w.publication_year BETWEEN 2015 AND 2024
          AND au.institution_idx IN (SELECT institution_idx FROM ci)
    """)
    con.execute("""
        CREATE TEMP TABLE home AS
        WITH ai AS (SELECT author_idx, institution_idx, COUNT(*) n FROM a GROUP BY 1, 2)
        SELECT author_idx, arg_max(institution_idx, n) AS home_inst, SUM(n) AS tot
        FROM ai GROUP BY 1 HAVING SUM(n) >= 3
    """)

    # ── 1. home screen ────────────────────────────────────────────────────────
    s1 = con.execute("""
        WITH x AS (
            SELECT a.institution_idx AS inst, h.home_inst, COUNT(*) AS n
            FROM a JOIN home h USING (author_idx)
            WHERE a.y BETWEEN 2020 AND 2024
              AND a.institution_idx IN (SELECT unit_idx FROM b)
            GROUP BY 1, 2)
        SELECT inst AS institution_idx,
               SUM(n) AS n_auth,
               1 - SUM(CASE WHEN home_inst = inst THEN n ELSE 0 END) / SUM(n) AS away,
               arg_max(home_inst, CASE WHEN home_inst <> inst THEN n ELSE -1 END) AS home_partner,
               MAX(CASE WHEN home_inst <> inst THEN n END) / SUM(n) AS home_share
        FROM x GROUP BY 1
    """).df()

    # ── 3. country screen ─────────────────────────────────────────────────────
    s3 = con.execute("""
        SELECT a.institution_idx,
               AVG(CASE WHEN hc.country_code IS DISTINCT FROM xc.country_code
                        THEN 1.0 ELSE 0.0 END) AS foreign_share,
               mode(hc.country_code) AS main_home_country
        FROM a
        JOIN home h USING (author_idx)
        JOIN ci hc ON hc.institution_idx = h.home_inst
        JOIN ci xc ON xc.institution_idx = a.institution_idx
        WHERE a.y BETWEEN 2020 AND 2024
          AND a.institution_idx IN (SELECT unit_idx FROM b)
          AND h.home_inst <> a.institution_idx
        GROUP BY 1
    """).df()
    n_all = con.execute("""
        SELECT a.institution_idx, COUNT(*) AS n FROM a JOIN home h USING (author_idx)
        WHERE a.y BETWEEN 2020 AND 2024 GROUP BY 1""").df()
    s3 = s3.merge(n_all, on='institution_idx')
    s3 = s3.merge(s1[['institution_idx', 'away']], on='institution_idx')
    # foreign_share above is among away authorships; convert to share of all
    s3['foreign_share'] = s3['foreign_share'] * s3['away']

    # ── 2. name screen ────────────────────────────────────────────────────────
    ci['nname'] = ci['institution_name'].fillna('').map(norm)
    X = ci[ci.institution_idx.isin(b.unit_idx)][['institution_idx', 'institution_name', 'nname']]
    Y = ci[ci.works_count >= MIN_PARTNER_WORKS][['institution_idx', 'institution_name',
                                                 'nname', 'works_count']]
    con.register('X', X)
    con.register('Y', Y)
    pairs = con.execute(f"""
        SELECT X.institution_idx, Y.institution_idx AS name_partner,
               jaro_winkler_similarity(X.nname, Y.nname) AS jw,
               X.nname = Y.nname AS same_norm,
               (length(Y.nname) - length(replace(Y.nname, ' ', ''))) >= 1
                 AND contains(' ' || X.nname || ' ', ' ' || Y.nname || ' ') AS x_contains_y,
               (length(X.nname) - length(replace(X.nname, ' ', ''))) >= 1
                 AND contains(' ' || Y.nname || ' ', ' ' || X.nname || ' ') AS y_contains_x,
               Y.works_count AS partner_works
        FROM X, Y
        WHERE X.institution_idx <> Y.institution_idx
          AND (X.nname = Y.nname
               OR jaro_winkler_similarity(X.nname, Y.nname) >= {JW_MIN}
               OR ((length(Y.nname) - length(replace(Y.nname, ' ', ''))) >= 1
                   AND contains(' ' || X.nname || ' ', ' ' || Y.nname || ' '))
               OR ((length(X.nname) - length(replace(X.nname, ' ', ''))) >= 1
                   AND contains(' ' || Y.nname || ' ', ' ' || X.nname || ' ')))
    """).df()
    pairs['score'] = (pairs.same_norm * 3 + pairs.x_contains_y * 2 + pairs.y_contains_x * 1
                      + pairs.jw)
    best_pair = (pairs.sort_values(['institution_idx', 'score', 'partner_works'],
                                   ascending=[True, False, False])
                 .groupby('institution_idx').head(1))
    n_pairs = pairs.groupby('institution_idx').size().rename('n_name_partners')

    # ── combine ───────────────────────────────────────────────────────────────
    out = b.rename(columns={'unit_idx': 'institution_idx'})
    out = out.merge(ci[['institution_idx', 'institution_name', 'country_code', 'itype',
                        'works_count']], on='institution_idx', how='left')
    out = out.merge(s1, on='institution_idx', how='left')
    out = out.merge(s3[['institution_idx', 'foreign_share', 'main_home_country']],
                    on='institution_idx', how='left')
    out = out.merge(best_pair[['institution_idx', 'name_partner', 'jw', 'same_norm',
                               'x_contains_y', 'y_contains_x']], on='institution_idx',
                    how='left')
    out = out.merge(n_pairs, on='institution_idx', how='left')
    name_of = dict(zip(ci.institution_idx, ci.institution_name))
    cc_of = dict(zip(ci.institution_idx, ci.country_code))
    out['home_partner_name'] = out.home_partner.map(name_of)
    out['home_partner_cc'] = out.home_partner.map(cc_of)
    out['name_partner_name'] = out.name_partner.map(name_of)
    out['name_partner_cc'] = out.name_partner.map(cc_of)
    out['flag_home'] = out.home_share >= HOME_SHARE
    out['flag_country'] = out.foreign_share >= COUNTRY_SHARE
    out['flag_name'] = out.name_partner.notna()
    flagged = out[out.flag_home | out.flag_country | out.flag_name].copy()
    flagged['n_flags'] = flagged[['flag_home', 'flag_country', 'flag_name']].sum(axis=1)

    cols = ['irank', 'institution_idx', 'institution_name', 'country_code', 'itype', 'v',
            'a_p', 'flag_home', 'flag_name', 'flag_country', 'n_flags', 'away',
            'home_share', 'home_partner', 'home_partner_name', 'home_partner_cc',
            'name_partner', 'name_partner_name', 'name_partner_cc', 'jw', 'same_norm',
            'x_contains_y', 'y_contains_x', 'n_name_partners', 'foreign_share',
            'main_home_country']
    flagged = flagged[cols].sort_values('irank')
    dest = paths.data / 'institution_screen.csv'
    flagged.to_csv(dest, index=False)

    print(f"baseline institutions: {len(b)}")
    print(f"flagged: {len(flagged)}  (home {flagged.flag_home.sum()}, "
          f"name {flagged.flag_name.sum()}, country {flagged.flag_country.sum()})")
    print(f"written to {dest}")


if __name__ == '__main__':
    main()
