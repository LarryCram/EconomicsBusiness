"""
analyse_meta_outliers.py — Diagnose why Meta A+ journals rank lower than expected.

The 10 sources that are rated A+ (top human tier) but rank furthest below the
top-25 in the EBAX spectral ranking are examined across four diagnostic dimensions:

  1. Intra-corpus citation rate  — fraction of OA cited_by_count that is captured
                                   within the OAS corpus (2020-24 census works).
  2. Citation breadth            — number of distinct OAS sources that cite the journal.
  3. Citer quality               — work-count-weighted mean v of citing sources.
  4. Field insularity            — fraction of citations from E-field vs B/A/X sources.

For each journal, the script prints a structured diagnostic and proposes a
primary mechanism for the spectral rank gap.

Console output only; no files written.
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

paths = load_config()

_bl = next(r for r in load_runs() if r['label'] == 'baseline')
RK_TNAME = f"rk_{_bl['run_code']}_{_bl['fx']}_tauU{_bl['tau_u']}_tauS{_bl['tau_s']}_vartau_rho0_m0110_chi50_alpha100"
EL_TNAME = f"el_{_bl['run_code']}_{_bl['fx']}_tauU{_bl['tau_u']}_tauS{_bl['tau_s']}_vartau"

# 10 Meta A+ journals furthest outside top-25 in EBAX spectral ranking
OUTLIERS = [
    (202812398, 'Information Systems Research',          'B', 217),
    (142306484, 'Journal of Operations Management',      'B', 201),
    (166002381, 'Journal of Applied Psychology',         'B', 157),
    (119950638, 'Journal of Marketing Research',         'B', 119),
    (38024979,  'Journal of International Business Studies', 'B', 112),
    (145429826, 'Journal of Consumer Research',          'B', 109),
    (160506855, 'The Accounting Review',                 'A',  95),
    (122767448, 'Journal of Management',                 'B',  83),
    (33323087,  'Management Science',                    'B',  82),
    (206124708, 'Organization Science',                  'B',  79),
]
IDS     = [x[0] for x in OUTLIERS]
ID_NAME = {x[0]: x[1] for x in OUTLIERS}
ID_RANK = {x[0]: x[3] for x in OUTLIERS}
IDS_SQL = ','.join(str(i) for i in IDS)

SM_PATH = f"'{paths.parquet}/source_master.parquet'"
WK_PATH = f"'{paths.parquet}/corpus_works.parquet'"


def load_data() -> dict:
    rk_path = str(paths.working / 'rankings.duckdb')
    el_path = str(paths.working / 'edge_lists.duckdb')

    with duckdb.connect(rk_path, read_only=True) as db_rk, \
         duckdb.connect(el_path, read_only=True) as db_el:

        # Citer field distribution
        df_citer_field = db_el.execute(f"""
            SELECT cited_source_idx,
                   sm.field_eb                    AS citer_field,
                   COUNT(DISTINCT citer_work_idx) AS n_citing_works
            FROM {EL_TNAME} el
            JOIN {SM_PATH} sm ON sm.source_idx = el.citer_source_idx
            WHERE cited_source_idx IN ({IDS_SQL})
            GROUP BY cited_source_idx, sm.field_eb
        """).df()

        # Distinct citing sources + work-count-weighted mean v of citers
        df_rk_src = db_rk.execute(
            f"SELECT unit_idx AS citer_source_idx, v FROM {RK_TNAME} WHERE unit_type='S'"
        ).df()

        df_citers_raw = db_el.execute(f"""
            SELECT cited_source_idx, citer_source_idx,
                   COUNT(DISTINCT citer_work_idx) AS n_citing_works
            FROM {EL_TNAME}
            WHERE cited_source_idx IN ({IDS_SQL})
            GROUP BY cited_source_idx, citer_source_idx
        """).df().merge(df_rk_src, on='citer_source_idx', how='left')

        # Out-citation field distribution
        df_out_field = db_el.execute(f"""
            SELECT citer_source_idx,
                   sm.field_eb                    AS cited_field,
                   COUNT(DISTINCT cited_work_idx) AS n_cited_works
            FROM {EL_TNAME} el
            JOIN {SM_PATH} sm ON sm.source_idx = el.cited_source_idx
            WHERE citer_source_idx IN ({IDS_SQL})
            GROUP BY citer_source_idx, sm.field_eb
        """).df()

        # OA cited_by_count vs intra-corpus (census works 2020-24)
        df_oa = db_el.execute(f"""
            SELECT source_idx,
                   COUNT(*)          AS n_works_2024,
                   SUM(cited_by_count) AS oa_cited_by
            FROM {WK_PATH}
            WHERE source_idx IN ({IDS_SQL})
              AND publication_year BETWEEN 2020 AND 2024
            GROUP BY source_idx
        """).df()

        df_intra = db_el.execute(f"""
            SELECT cited_source_idx,
                   COUNT(DISTINCT citer_work_idx) AS intra_citers
            FROM {EL_TNAME}
            WHERE cited_source_idx IN ({IDS_SQL})
            GROUP BY cited_source_idx
        """).df()

        # a_p from the ranking table
        df_ap = db_rk.execute(f"""
            SELECT unit_idx, a_p, v
            FROM {RK_TNAME}
            WHERE unit_type='S' AND unit_idx IN ({IDS_SQL})
        """).df()

    return dict(
        citer_field=df_citer_field,
        citers_raw=df_citers_raw,
        out_field=df_out_field,
        oa=df_oa,
        intra=df_intra,
        ap=df_ap,
    )


def build_summary(data: dict) -> pd.DataFrame:
    rows = []
    for src_id, name, field_eb, rank in OUTLIERS:
        cf = data['citer_field']
        cf_src = cf[cf['cited_source_idx'] == src_id]
        total_in_cites = cf_src['n_citing_works'].sum()
        e_frac  = cf_src.loc[cf_src['citer_field'] == 'E', 'n_citing_works'].sum() / max(total_in_cites, 1)
        b_frac  = cf_src.loc[cf_src['citer_field'] == 'B', 'n_citing_works'].sum() / max(total_in_cites, 1)

        cr = data['citers_raw']
        cr_src = cr[cr['cited_source_idx'] == src_id].dropna(subset=['v'])
        n_distinct_citers = cr_src['citer_source_idx'].nunique()
        tot_wt = cr_src['n_citing_works'].sum()
        wmean_v = (cr_src['v'] * cr_src['n_citing_works']).sum() / tot_wt if tot_wt else float('nan')
        frac_high_v = (cr_src.loc[cr_src['v'] >= 3, 'n_citing_works'].sum() / tot_wt
                       if tot_wt else float('nan'))

        of = data['out_field']
        of_src = of[of['citer_source_idx'] == src_id]
        total_out = of_src['n_cited_works'].sum()

        oa_row  = data['oa'][data['oa']['source_idx'] == src_id]
        intra_r = data['intra'][data['intra']['cited_source_idx'] == src_id]
        n_works_2024 = int(oa_row['n_works_2024'].iloc[0]) if len(oa_row) else 0
        oa_total     = float(oa_row['oa_cited_by'].iloc[0]) if len(oa_row) else 0
        intra_n      = int(intra_r['intra_citers'].iloc[0]) if len(intra_r) else 0
        intra_pct    = 100 * intra_n / oa_total if oa_total else float('nan')

        ap_row = data['ap'][data['ap']['unit_idx'] == src_id]
        a_p = float(ap_row['a_p'].iloc[0]) if len(ap_row) else float('nan')
        v   = float(ap_row['v'].iloc[0])   if len(ap_row) else float('nan')

        rows.append(dict(
            src_id=src_id, name=name, field_eb=field_eb,
            rank=rank, v=v, a_p=a_p,
            n_works_2024=n_works_2024,
            oa_cited_by=oa_total, intra_pct=intra_pct,
            n_distinct_citers=n_distinct_citers,
            total_in_corpus_cites=total_in_cites,
            total_out_corpus_cites=total_out,
            wmean_v_citer=wmean_v,
            frac_high_v_citer=frac_high_v,
            e_frac=e_frac, b_frac=b_frac,
        ))
    return pd.DataFrame(rows)


MECHANISMS = {
    # primary reason and supporting data to quote
    202812398: (
        "External citation sink",
        "Only 10.6% of OA citations are intra-corpus. IS research is cited "
        "heavily by CS and engineering journals outside OAS. With 93 distinct "
        "citing sources (fewest of the 10) and few out-citations in the corpus, "
        "the journal is peripheral to the OAS citation network regardless of "
        "its global standing."
    ),
    142306484: (
        "Narrow, low-v citation community",
        "Operations management is a tight specialist subfield. Only 240 distinct "
        "citing sources, weighted mean v = 1.20 (lowest of the 10), and only 3.2% "
        "of citing volume comes from high-v (v ≥ 3) sources. OM also cites "
        "engineering journals outside OAS, so its out-citations within the corpus "
        "are limited, weakening the network reciprocity that drives spectral rank."
    ),
    166002381: (
        "Organisational behaviour / psychology field straddle",
        "JAP straddles business and I/O psychology. Only 9% of its OA citations "
        "are captured within OAS. Its out-citation profile in the corpus is "
        "strikingly thin (140 cited works), confirming that its reference lists "
        "point heavily outside OAS into psychology. The result is that the journal "
        "participates minimally in the OAS citation network as both giver and "
        "receiver of reference attention."
    ),
    119950638: (
        "Extreme external citation leakage + narrow footprint",
        "Only 2.8% of OA citations are intra-corpus — the lowest of all 10 — "
        "and only 39 works in the corpus are cited by JMR. Its methodological "
        "contributions are cited across marketing, psychology, and statistics "
        "journals outside OAS. Within OAS, only 93 sources cite it, almost all "
        "from B-field, and the weighted mean v of those citers is 1.74 — moderate "
        "but unremarkable."
    ),
    38024979: (
        "Interdisciplinary field with split citation communities",
        "International business draws from economics (21% E-field citers by works), "
        "sociology, geography, and political science. The E-field fraction is the "
        "highest among B-classified outliers, yet weighted mean citer v is only "
        "1.32. JIBS cites moderately within the corpus (1333 out-cited works) "
        "but the field's dispersion across disciplines outside OAS limits intra-"
        "corpus citation density."
    ),
    145429826: (
        "Consumer behaviour field straddle with psychology",
        "JCR publishes qualitative and sociological consumer research alongside "
        "quantitative work. Only 12.3% of OA citations are intra-corpus. Citing "
        "volume is almost entirely B-field (97%), and citer breadth is limited to "
        "168 distinct sources. Qualitative and interpretive papers attract citations "
        "from sociology and anthropology journals outside OAS."
    ),
    160506855: (
        "Accounting sub-community isolation",
        "The three top accounting journals (JAE rank 29, JAR rank 46, TAR rank 95) "
        "form a relatively self-contained citation cluster. TAR is cited mainly by "
        "A-field journals (73% of citing volume), almost all within accounting. "
        "Despite a respectable citer quality (wmean_v = 1.73, 23% high-v), "
        "accounting's citation community does not connect strongly to the high-v "
        "economics core, capping the influence that can flow through the network."
    ),
    122767448: (
        "Broad management field with moderate-v citation base",
        "Journal of Management is a generalist outlet cited by 507 distinct sources "
        "(high breadth) but the citing community is B-dominated (88%) with wmean_v = "
        "1.41. The journal publishes across strategy, OB, HR, and entrepreneurship, "
        "whose citation communities are all moderately ranked B-field sources. "
        "Breadth without high-v anchor journals limits spectral rank."
    ),
    33323087: (
        "Large work count dilutes per-work influence despite high citer quality",
        "Management Science has the highest citer quality among the 10 (wmean_v = "
        "1.84, 22% from v ≥ 3 sources, 657 distinct citing sources). Its moderate "
        "spectral rank reflects the influence formula: v = (A/a_p)×π. With 2,123 "
        "works in 2020-24 (the largest denominator of the 10), ranking probability "
        "π is diluted per work even as the journal's total network presence is large. "
        "Its B-only rank (104) is substantially worse than its EBAX rank (82), "
        "reflecting that Economics journals also cite it but it earns less from the "
        "B-only network where it competes with more focused journals."
    ),
    206124708: (
        "High-v citers but sociological reference network outside OAS",
        "Organization Science has the highest wmean_v_citer (2.15) and the largest "
        "fraction of high-v citing volume (28%) of all 10 — it is cited by quality "
        "sources. Yet it ranks only 79 because organizational theory draws heavily "
        "on sociology, institutional theory, and political science journals not in "
        "OAS, so OS's reference lists direct attention outside the corpus. With 21% "
        "intra-corpus citation rate and out-citations confined almost entirely to "
        "B-field (90%), the journal contributes and receives within a network "
        "sub-community that caps its spectral influence."
    ),
}


def print_analysis(df: pd.DataFrame) -> None:
    print('=' * 80)
    print('SPECTRAL RANK GAP ANALYSIS — Meta A+ journals outside the spectral top-25')
    print('Journals ordered by spectral rank (furthest first)')
    print('=' * 80)

    # Summary table
    cols = ['name', 'rank', 'v', 'a_p', 'n_works_2024',
            'intra_pct', 'n_distinct_citers', 'wmean_v_citer',
            'frac_high_v_citer', 'e_frac', 'b_frac']
    labels = ['Name', 'Rank', 'v', 'a_p', 'n_wks\n2024',
              'intra\n%', 'n_citers', 'wmean\nv_citer',
              'frac\nv≥3', 'E\nfrac', 'B\nfrac']
    print()
    hdr = f"{'Name':<42} {'Rank':>5} {'v':>6} {'a_p':>6} "
    hdr += f"{'n_wks':>6} {'intra%':>7} {'n_cit':>6} {'wm_v':>5} {'f≥3':>5} {'E%':>4} {'B%':>4}"
    print(hdr)
    print('-' * len(hdr))
    for _, row in df.iterrows():
        print(
            f"{row['name']:<42} {row['rank']:>5} {row['v']:>6.2f} {row['a_p']:>6.0f} "
            f"{row['n_works_2024']:>6} {row['intra_pct']:>6.1f}% "
            f"{row['n_distinct_citers']:>6} {row['wmean_v_citer']:>5.2f} "
            f"{100*row['frac_high_v_citer']:>4.0f}% "
            f"{100*row['e_frac']:>3.0f}% {100*row['b_frac']:>3.0f}%"
        )

    print()
    print('Columns: Rank=EBAX spectral rank; v=influence score; a_p=corpus work count;')
    print('n_wks=works published 2020-24; intra%=% of OA cited_by_count captured')
    print('within OAS corpus; n_cit=distinct citing sources; wm_v=work-weighted mean')
    print('v of citing sources; f≥3=fraction of citing volume from v≥3 sources;')
    print('E%/B%=fraction of citing volume from E/B-field sources.')
    print()

    # Per-journal narrative
    print('=' * 80)
    print('PER-JOURNAL DIAGNOSES')
    print('=' * 80)
    for _, row in df.iterrows():
        sid = row['src_id']
        mechanism, explanation = MECHANISMS[sid]
        print(f"\n{row['name']}  (rank {row['rank']}, field {row['field_eb']}, v={row['v']:.2f})")
        print(f"Primary mechanism: {mechanism}")
        print(f"  {explanation}")

    print()
    print('=' * 80)
    print('CROSS-CUTTING MECHANISMS')
    print('=' * 80)
    print("""
1. External citation leakage — OAS covers Economics and Business journals only.
   Journals that routinely attract citations from psychology, engineering, CS, or
   sociology (JAP, JMR, ISR, JCR, OS) have a large fraction of their OA citations
   outside the corpus. The spectral ranking only sees intra-OAS flows; a journal
   whose influence is global but whose citers are outside OAS will be systematically
   underranked. The most extreme case is JMR (2.8% intra-corpus).

2. B-field network discount — The Economics (E) subnetwork contains the very
   high-v journals (AER v=15.3, Econometrica v=12.5, JPE v=8.5) whose attention
   amplifies any journal they cite. The Business (B) subnetwork lacks comparable
   anchor journals. B-only journals are cited by other B-journals whose v is
   moderate, limiting the maximum influence that can propagate to them.

3. Large work-count denominator — Influence v = (A/a_p)×π. High-output journals
   (Management Science: 2,123 works in 2020-24) accumulate large π but the
   per-work normalisation lowers v. This does not reflect lower quality — it
   reflects that the journal is a broad, high-volume outlet rather than a focused
   one where every paper is a high-profile contribution.

4. Tight within-field citation clusters — The accounting trio (JAE, JAR, TAR)
   and the IS/OM specialisms form citation communities that cite each other but
   connect weakly to the high-v economics core. Within-community influence
   recirculates but does not radiate out to gain reflected influence from
   high-v neighbours.

5. Low network participation (out-citations within OAS) — Journals whose
   reference lists point heavily outside OAS (JAP, JMR) contribute little to
   the network's citation flow. In the Markov walk, they are destinations but
   not conduits, and a unit that does not propagate attention through the network
   earns less reflected influence from what it cites.
""")


def main():
    print('Loading data…')
    data = load_data()
    df = build_summary(data)
    print_analysis(df)


if __name__ == '__main__':
    main()
