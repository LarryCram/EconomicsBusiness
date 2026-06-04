"""
table_maker.py — Generate publishable tables for the paper.

Table 1: Registry overview with field classification.
    Inputs:
        comprehensive_journal_list.parquet  -- pre-OA registry union
        oas_star.parquet                    -- OA-matched long-list (OAS*)
        source_master.parquet               -- OAS* with field_eb
    Outputs (written to tables/ and data/):
        table1_registry_overview.tex
        table1_registry_overview.csv

Table 2: Source registry matching statistics.
    Inputs:
        comprehensive_journal_list.parquet  -- pre-OA registry union
        oas_star.parquet                    -- OA-matched long-list (OAS*)
    Outputs (written to data/):
        table2_source_matching.tex          -- LaTeX fragment, \\input{} into paper
        table2_source_matching.csv          -- for inspection / sanity check

Table 3: Corpus features.
    Inputs:
        corpus_works.parquet
        corpus_authorships.parquet
        corpus_references.parquet
    Intermediate:
        corpus_institutions.parquet         -- institution-level summary with works_per_year
    Outputs (written to data/):
        table3_corpus_features.tex
        table3_corpus_features.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

import duckdb
import pandas as pd
import subprocess
import tempfile

paths  = load_config()
_baseline = next(r for r in load_runs() if r['label'] == 'baseline')

PROJECT_FOLDER = paths.project_root
DATA           = paths.data
TABLES         = paths.tables
WORKING        = paths.working
PARQUET        = paths.parquet

TAU_U        = _baseline['tau_u']
TAU_S        = _baseline['tau_s']
TC0          = _baseline['tc0']   # census window start (baseline = 2020)
TC1          = _baseline['tc1']   # census window end   (baseline = 2024)
CENSUS_YEARS = TC1 - TC0 + 1     # 5 for baseline
CORPUS_YEARS = 25  # 2000-2024 inclusive


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_table1_data(db):
    """
    Return data for Table 1: registry overview with field classification.

    Rows: JQL (Harzing), MJL (WOS), SJL (ERA), SSL (Scopus),
          OAS** (raw union), OAS* (density-filtered), OAS (τ_S-filtered).
    Columns per row: Sources, OA Match, Match %, E, B, A, X.

    Source counts include duplicates: a source in both JQL and SSL is
    counted once in each row.  E/B/A/X are taken from source_master (the
    combined five-signal scoring) for OA-matched sources from that registry.

    OAS** field_eb uses source_master's 5-signal score where available; for
    sources dropped by the topic density filter a 4-signal fallback is computed
    from registry fields (era_for_codes, harzing_field, wos_categories, scopus_asjc)
    without the OpenAlex topic field signal.
    """
    # Registry sizes — COUNT(*) so cross-registry duplicates count per registry
    sizes = db.sql(f"""
        SELECT
            COUNT(*) FILTER (WHERE harzing_journal_name IS NOT NULL) AS jql_total,
            COUNT(*) FILTER (WHERE wos_journal_name     IS NOT NULL) AS mql_total,
            COUNT(*) FILTER (WHERE era_journal_name     IS NOT NULL) AS sjl_total,
            COUNT(*) FILTER (WHERE scopus_journal_name  IS NOT NULL) AS ssl_total
        FROM '{PARQUET}/comprehensive_journal_list.parquet'
    """).fetchone()
    jql_total, mql_total, sjl_total, ssl_total = sizes

    # OA matches per registry (OAS*)
    matched = db.sql(f"""
        SELECT
            COUNT(*) FILTER (WHERE harzing_journal_name IS NOT NULL) AS jql_matched,
            COUNT(*) FILTER (WHERE wos_journal_name     IS NOT NULL) AS mql_matched,
            COUNT(*) FILTER (WHERE era_journal_name     IS NOT NULL) AS sjl_matched,
            COUNT(*) FILTER (WHERE scopus_journal_name  IS NOT NULL) AS ssl_matched
        FROM '{PARQUET}/oas_star.parquet'
    """).fetchone()
    jql_matched, mql_matched, sjl_matched, ssl_matched = matched

    # E/B/A/X for OAS** and each registry: all oas_star sources with field_eb from
    # source_master where available; 4-signal registry fallback for density-dropped sources.
    # This makes registry E/B/A/X consistent with the OA Match count (both from oas_star).
    eb_df = db.sql(f"""
        WITH scored AS (
            SELECT os.source_idx,
                   os.harzing_journal_name,
                   os.wos_journal_name,
                   os.era_journal_name,
                   os.scopus_journal_name,
                   (COALESCE(len(list_filter(os.era_for_codes, x -> LEFT(x,2)='38')), 0)
                    + CASE WHEN len(list_filter(string_split(COALESCE(os.harzing_field,''), ', '),
                                   x -> x IN ('Economics','F&A'))) > 0 THEN 1 ELSE 0 END
                    + CASE WHEN regexp_matches(LOWER(array_to_string(os.wos_categories,' ')),
                                              'econom|financ|banking') THEN 1 ELSE 0 END
                    + CASE WHEN list_contains(os.scopus_asjc, '2000') THEN 1 ELSE 0 END
                   ) AS econ4,
                   (COALESCE(len(list_filter(os.era_for_codes, x -> LEFT(x,2)='35')), 0)
                    + CASE WHEN len(list_filter(string_split(COALESCE(os.harzing_field,''), ', '),
                                   x -> x IN ('Bus Hist','Comm','Entrep','Gen & Strat','IB',
                                              'Innovation','Marketing','MIS,KM','OS/OB,HRM/IR',
                                              'OR,MS,POM','PSM','Tourism'))) > 0 THEN 1 ELSE 0 END
                    + CASE WHEN regexp_matches(LOWER(array_to_string(os.wos_categories,' ')),
                                              'business|commerc|management|tourism|transport') THEN 1 ELSE 0 END
                    + CASE WHEN list_contains(os.scopus_asjc, '1400') THEN 1 ELSE 0 END
                   ) AS bus4,
                   sm.field_eb AS sm_field_eb
            FROM '{PARQUET}/oas_star.parquet' os
            LEFT JOIN '{PARQUET}/source_master.parquet' sm USING (source_idx)
        )
        SELECT
            COALESCE(sm_field_eb,
                CASE
                    WHEN econ4 + bus4 < 2 THEN 'X'
                    WHEN econ4 = bus4     THEN 'A'
                    WHEN econ4 > bus4     THEN 'E'
                    ELSE                       'B'
                END
            ) AS field_eb,
            COUNT(*) FILTER (WHERE harzing_journal_name IS NOT NULL) AS jql_n,
            COUNT(*) FILTER (WHERE wos_journal_name     IS NOT NULL) AS mql_n,
            COUNT(*) FILTER (WHERE era_journal_name     IS NOT NULL) AS sjl_n,
            COUNT(*) FILTER (WHERE scopus_journal_name  IS NOT NULL) AS ssl_n,
            COUNT(*)                                                  AS oas2_n
        FROM scored
        GROUP BY 1
    """).df()

    eb = {row['field_eb']: row for _, row in eb_df.iterrows()}

    def _eb(col, label):
        return int(eb.get(label, {}).get(col, 0))

    oas2_total = int(eb_df['oas2_n'].sum())

    # OAS* (density-filtered): source_master counts
    oas_star_df = db.sql(f"""
        SELECT field_eb, COUNT(*) AS n
        FROM '{PARQUET}/source_master.parquet'
        GROUP BY field_eb
    """).df()
    oas_star_eb    = {row['field_eb']: int(row['n']) for _, row in oas_star_df.iterrows()}
    oas_star_total = int(oas_star_df['n'].sum())

    # OAS (τ_S-filtered): source_master sources that pass the census filter
    oas_df = db.sql(f"""
        SELECT sm.source_idx, sm.field_eb
        FROM '{PARQUET}/source_master.parquet' sm
        JOIN '{PARQUET}/corpus_works.parquet' cw
            ON sm.source_idx = cw.source_idx
        WHERE cw.publication_year BETWEEN {TC0} AND {TC1}
        GROUP BY sm.source_idx, sm.field_eb
        HAVING COUNT(DISTINCT cw.work_idx) / {CENSUS_YEARS}.0 >= {TAU_S}
    """).df()
    oas_eb    = oas_df.groupby('field_eb').size().to_dict()
    oas_total = len(oas_df)

    data = {
        'JQL':   {'total': jql_total, 'matched': jql_matched,
                  'E': _eb('jql_n', 'E'), 'B': _eb('jql_n', 'B'),
                  'A': _eb('jql_n', 'A'), 'X': _eb('jql_n', 'X')},
        'MJL':   {'total': mql_total, 'matched': mql_matched,
                  'E': _eb('mql_n', 'E'), 'B': _eb('mql_n', 'B'),
                  'A': _eb('mql_n', 'A'), 'X': _eb('mql_n', 'X')},
        'SJL':   {'total': sjl_total, 'matched': sjl_matched,
                  'E': _eb('sjl_n', 'E'), 'B': _eb('sjl_n', 'B'),
                  'A': _eb('sjl_n', 'A'), 'X': _eb('sjl_n', 'X')},
        'SSL':   {'total': ssl_total, 'matched': ssl_matched,
                  'E': _eb('ssl_n', 'E'), 'B': _eb('ssl_n', 'B'),
                  'A': _eb('ssl_n', 'A'), 'X': _eb('ssl_n', 'X')},
        'OAS**': {'total': oas2_total, 'matched': None,
                  'E': _eb('oas2_n', 'E'), 'B': _eb('oas2_n', 'B'),
                  'A': _eb('oas2_n', 'A'), 'X': _eb('oas2_n', 'X')},
        'OAS*':  {'total': oas_star_total, 'matched': None,
                  'E': oas_star_eb.get('E', 0), 'B': oas_star_eb.get('B', 0),
                  'A': oas_star_eb.get('A', 0), 'X': oas_star_eb.get('X', 0)},
        'OAS':   {'total': oas_total, 'matched': None,
                  'E': int(oas_eb.get('E', 0)), 'B': int(oas_eb.get('B', 0)),
                  'A': int(oas_eb.get('A', 0)), 'X': int(oas_eb.get('X', 0))},
    }

    print("\n=== TABLE 1: REGISTRY OVERVIEW ===")
    print(f"{'Register':<8} {'Sources':>9} {'OA Match':>9} {'Match %':>8} {'E':>6} {'B':>6} {'A':>6} {'X':>6}")
    for reg, d in data.items():
        pct = f"{d['matched']/d['total']*100:.1f}%" if d['matched'] is not None else "—"
        matched_str = _i(d['matched']) if d['matched'] is not None else "—"
        print(f"{reg:<8} {_i(d['total']):>9} {matched_str:>9} {pct:>8} "
              f"{_i(d['E']):>6} {_i(d['B']):>6} {_i(d['A']):>6} {_i(d['X']):>6}")
    return data


def write_latex_table1(data, out_path):
    REGS = ['JQL', 'MJL', 'SJL', 'SSL']

    L = []
    L.append(r"\begin{table}[htbp]")
    L.append(r"\centering")
    L.append(
        r"""\caption{Registry and OpenAlex sources classified by field. JQL:
        Harzing Journal Quality List; MJL: Web of Science Master Journal List; SJL: ERA Subject Journal List; SSL: Scopus Source List. A source appearing in multiple registries is counted in each. E/B/A/X are assigned by the combined five-signal scoring. OAS$^{**}$ is the union of all registry sources matched to OpenAlex. OAS$^{*}$ drops sources whose OpenAlex topic content is less than 40\% economics or business. OAS drops sources publishing below $\tau_S$.}"""
    )
    L.append(r"\label{tab:registry_overview}")
    L.append(r"\begin{tabular}{lrrrrrrrr}")
    L.append(r"\toprule")
    L.append(r"Registry & Sources & OA Match & Match\% & E & B & A & X \\")
    L.append(r"\midrule")
    for reg in REGS:
        d = data[reg]
        L.append(
            f"{reg} & {_i(d['total'])} & {_i(d['matched'])} & {_pct(d['matched'], d['total'])}"
            f" & {_i(d['E'])} & {_i(d['B'])} & {_i(d['A'])} & {_i(d['X'])} \\\\"
        )
    L.append(r"\midrule")
    d = data['OAS**']
    L.append(
        f"OAS$^{{**}}$ & {_i(d['total'])} & --- & ---"
        f" & {_i(d['E'])} & {_i(d['B'])} & {_i(d['A'])} & {_i(d['X'])} \\\\"
    )
    d = data['OAS*']
    L.append(
        f"OAS$^{{*}}$ & {_i(d['total'])} & --- & ---"
        f" & {_i(d['E'])} & {_i(d['B'])} & {_i(d['A'])} & {_i(d['X'])} \\\\"
    )
    d = data['OAS']
    L.append(
        f"OAS & {_i(d['total'])} & --- & ---"
        f" & {_i(d['E'])} & {_i(d['B'])} & {_i(d['A'])} & {_i(d['X'])} \\\\"
    )
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")

    out_path.write_text("\n".join(L) + "\n")
    print(f"LaTeX written to {out_path}")


def write_csv_table1(data, out_path):
    rows = []
    for reg, d in data.items():
        rows.append({
            'Register': reg,
            'Sources': d['total'],
            'OA_Match': d['matched'] if d['matched'] is not None else '',
            'Match_pct': f"{d['matched']/d['total']*100:.1f}" if d['matched'] is not None else '',
            'E': d['E'], 'B': d['B'], 'A': d['A'], 'X': d['X'],
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"CSV written to {out_path}")


def build_table2_data(db):
    """Return (rows, overlaps) where:
        rows     = [(label, total, matched, unmatched), ...]  one per registry
        overlaps = (mql_sjl, mql_jql, sjl_jql, all_three,
                    ssl_jql, ssl_mql, ssl_sjl, all_four, total_oas_star)
    """
    # Pre-match registry sizes: DISTINCT names to collapse join-induced duplicates
    sizes = db.sql(f"""
        SELECT
            COUNT(DISTINCT harzing_journal_name) FILTER (WHERE harzing_journal_name IS NOT NULL) AS jql_total,
            COUNT(DISTINCT wos_journal_name)     FILTER (WHERE wos_journal_name     IS NOT NULL) AS mql_total,
            COUNT(DISTINCT era_journal_name)     FILTER (WHERE era_journal_name     IS NOT NULL) AS sjl_total,
            COUNT(DISTINCT scopus_journal_name)  FILTER (WHERE scopus_journal_name  IS NOT NULL) AS ssl_total
        FROM '{PARQUET}/comprehensive_journal_list.parquet'
    """).fetchone()
    jql_total, mql_total, sjl_total, ssl_total = sizes

    # Matched: distinct registry journal names that found >=1 OA source
    matched = db.sql(f"""
        SELECT
            COUNT(DISTINCT harzing_journal_name) FILTER (WHERE harzing_journal_name IS NOT NULL) AS jql_matched,
            COUNT(DISTINCT wos_journal_name)     FILTER (WHERE wos_journal_name     IS NOT NULL) AS mql_matched,
            COUNT(DISTINCT era_journal_name)     FILTER (WHERE era_journal_name     IS NOT NULL) AS sjl_matched,
            COUNT(DISTINCT scopus_journal_name)  FILTER (WHERE scopus_journal_name  IS NOT NULL) AS ssl_matched
        FROM '{PARQUET}/oas_star.parquet'
    """).fetchone()
    jql_matched, mql_matched, sjl_matched, ssl_matched = matched

    # Pairwise overlaps and union: count OA sources (rows), not distinct registry names
    overlaps_raw = db.sql(f"""
        SELECT
            COUNT(*) FILTER (WHERE wos_journal_name    IS NOT NULL AND era_journal_name     IS NOT NULL) AS mql_sjl,
            COUNT(*) FILTER (WHERE wos_journal_name    IS NOT NULL AND harzing_journal_name IS NOT NULL) AS mql_jql,
            COUNT(*) FILTER (WHERE era_journal_name    IS NOT NULL AND harzing_journal_name IS NOT NULL) AS sjl_jql,
            COUNT(*) FILTER (WHERE wos_journal_name    IS NOT NULL AND era_journal_name     IS NOT NULL
                                                                   AND harzing_journal_name IS NOT NULL) AS all_three,
            COUNT(*) FILTER (WHERE scopus_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL) AS ssl_jql,
            COUNT(*) FILTER (WHERE scopus_journal_name IS NOT NULL AND wos_journal_name     IS NOT NULL) AS ssl_mql,
            COUNT(*) FILTER (WHERE scopus_journal_name IS NOT NULL AND era_journal_name     IS NOT NULL) AS ssl_sjl,
            COUNT(*) FILTER (WHERE scopus_journal_name IS NOT NULL AND wos_journal_name     IS NOT NULL
                                                                   AND era_journal_name     IS NOT NULL
                                                                   AND harzing_journal_name IS NOT NULL) AS all_four,
            COUNT(*) AS total_oas_star
        FROM '{PARQUET}/oas_star.parquet'
    """).fetchone()
    mql_sjl, mql_jql, sjl_jql, all_three, ssl_jql, ssl_mql, ssl_sjl, all_four, total_oas_star = overlaps_raw

    # Post-filter OAS counts from source_master (after topic density filter)
    oas = db.sql(f"""
        SELECT
            COUNT(DISTINCT harzing_journal_name) FILTER (WHERE harzing_journal_name IS NOT NULL) AS jql_oas,
            COUNT(DISTINCT wos_journal_name)     FILTER (WHERE wos_journal_name     IS NOT NULL) AS mql_oas,
            COUNT(DISTINCT era_journal_name)     FILTER (WHERE era_journal_name     IS NOT NULL) AS sjl_oas,
            COUNT(DISTINCT scopus_journal_name)  FILTER (WHERE scopus_journal_name  IS NOT NULL) AS ssl_oas,
            COUNT(*) FILTER (WHERE wos_journal_name IS NOT NULL AND era_journal_name     IS NOT NULL) AS mql_sjl_oas,
            COUNT(*) FILTER (WHERE wos_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL) AS mql_jql_oas,
            COUNT(*) FILTER (WHERE era_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL) AS sjl_jql_oas,
            COUNT(*) FILTER (WHERE wos_journal_name IS NOT NULL AND era_journal_name IS NOT NULL
                                                                AND harzing_journal_name IS NOT NULL) AS all_three_oas,
            COUNT(*) AS total_oas
        FROM '{PARQUET}/source_master.parquet'
    """).fetchone()
    jql_oas, mql_oas, sjl_oas, ssl_oas, mql_sjl_oas, mql_jql_oas, sjl_jql_oas, all_three_oas, total_oas = oas

    rows = [
        ('JQL', jql_total, jql_matched, jql_total - jql_matched),
        ('MJL', mql_total, mql_matched, mql_total - mql_matched),
        ('SJL', sjl_total, sjl_matched, sjl_total - sjl_matched),
        ('SSL', ssl_total, ssl_matched, ssl_total - ssl_matched),
    ]
    overlaps     = (mql_sjl, mql_jql, sjl_jql, all_three,
                    ssl_jql, ssl_mql, ssl_sjl, all_four, total_oas_star)
    overlaps_oas = (mql_sjl_oas, mql_jql_oas, sjl_jql_oas, all_three_oas, total_oas)
    oas_provenance = (jql_oas, mql_oas, sjl_oas, ssl_oas)

    # Console summary
    print("\n=== TABLE 2: SOURCE REGISTRY MATCHING ===")
    print(f"{'Register':<8} {'Sources':>9} {'Matched':>9} {'Rate':>7} {'Unmatched':>10}")
    for reg, total, matched, unmatched in rows:
        print(f"{reg:<8} {total:>9,} {matched:>9,} {matched/total*100:>6.1f}% {unmatched:>10,}")
    print(f"\nPairwise overlaps of matched OAS** (raw union, pre-density filter):")
    print(f"  MJL \u2229 SJL          {mql_sjl:>6,} sources")
    print(f"  MJL \u2229 JQL          {mql_jql:>6,} sources")
    print(f"  SJL \u2229 JQL          {sjl_jql:>6,} sources")
    print(f"  MJL \u2229 SJL \u2229 JQL    {all_three:>6,} sources")
    print(f"  SSL \u2229 JQL          {ssl_jql:>6,} sources")
    print(f"  SSL \u2229 MJL          {ssl_mql:>6,} sources")
    print(f"  SSL \u2229 SJL          {ssl_sjl:>6,} sources")
    print(f"  All four         {all_four:>6,} sources")
    print(f"  Union (OAS**)    {total_oas_star:>6,} unique sources")
    print(f"\nPost-topic-filter OAS ({total_oas:,} sources):")
    print(f"  {'Register':<8} {'OAS sources':>12}")
    for reg, n in zip(('JQL', 'MJL', 'SJL', 'SSL'), oas_provenance):
        print(f"  {reg:<8} {n:>12,}")
    print(f"  MJL \u2229 SJL          {mql_sjl_oas:>6,} sources")
    print(f"  MJL \u2229 JQL          {mql_jql_oas:>6,} sources")
    print(f"  SJL \u2229 JQL          {sjl_jql_oas:>6,} sources")
    print(f"  MJL \u2229 SJL \u2229 JQL    {all_three_oas:>6,} sources")

    return rows, overlaps, overlaps_oas, oas_provenance


# ---------------------------------------------------------------------------
# PDF compilation
# ---------------------------------------------------------------------------

def compile_pdf(tex_fragment_path):
    """
    Wrap a .tex table fragment in a minimal standalone document and compile
    to PDF with pdflatex.  Output is written alongside the .tex file.
    """
    fragment = tex_fragment_path.read_text()
    doc = "\n".join([
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}",
        r"\pagestyle{empty}",
        fragment,
        r"\end{document}",
    ])
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "table.tex"
        src.write_text(doc)
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", str(src)],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"pdflatex failed for {tex_fragment_path.name}:")
            print(result.stdout[-2000:])
            return
        pdf_src = Path(tmpdir) / "table.pdf"
        pdf_dst = tex_fragment_path.with_suffix(".pdf")
        pdf_dst.write_bytes(pdf_src.read_bytes())
    print(f"PDF written to {pdf_dst}")


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _i(n):
    """Integer with thousands separator."""
    return f"{n:,}"

def _pct(matched, total):
    """Match rate as LaTeX-safe percentage string."""
    return f"{matched / total * 100:.1f}\\%"


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def write_latex_table2(rows, overlaps, out_path):
    mql_sjl, mql_jql, sjl_jql, all_three, ssl_jql, ssl_mql, ssl_sjl, all_four, total_oas_star = overlaps

    L = []
    L.append(r"\begin{table}[htbp]")
    L.append(r"\centering")
    L.append(
        r"\caption{Matching of the four register-based sources to OpenAlex source identifiers."
        r" Overlaps count shared matched entries; OAS$^{**}$ is the union across all four registries.}"
    )
    L.append(r"\label{tab:source_matching}")
    L.append(r"\begin{tabular}{lrrrr}")
    L.append(r"\toprule")
    L.append(r"Register & Sources & Matched & Match rate & Unmatched \\")
    L.append(r"\midrule")
    for reg, total, matched, unmatched in rows:
        L.append(f"{reg} & {_i(total)} & {_i(matched)} & {_pct(matched, total)} & {_i(unmatched)} \\\\")
    L.append(r"\midrule")
    pairs = [
        (r"Overlap: $\mathrm{MJL} \cap \mathrm{SJL}$",                                                                    mql_sjl),
        (r"Overlap: $\mathrm{MJL} \cap \mathrm{JQL}$",                                                                    mql_jql),
        (r"Overlap: $\mathrm{SJL} \cap \mathrm{JQL}$",                                                                    sjl_jql),
        (r"Overlap: $\mathrm{MJL} \cap \mathrm{SJL} \cap \mathrm{JQL}$",                                                  all_three),
        (r"Overlap: $\mathrm{SSL} \cap \mathrm{JQL}$",                                                                    ssl_jql),
        (r"Overlap: $\mathrm{SSL} \cap \mathrm{MJL}$",                                                                    ssl_mql),
        (r"Overlap: $\mathrm{SSL} \cap \mathrm{SJL}$",                                                                    ssl_sjl),
        (r"Overlap: all four registries",                                                                                   all_four),
        (r"$(\mathrm{JQL} \cup \mathrm{MJL} \cup \mathrm{SJL} \cup \mathrm{SSL}) \cap \mathrm{OA}$\quad (OAS$^{**}$)", total_oas_star),
    ]
    for label, count in pairs:
        L.append(
            f"\\multicolumn{{2}}{{l}}{{{label}}} & "
            f"\\multicolumn{{3}}{{r}}{{{_i(count)} sources}} \\\\"
        )
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")

    out_path.write_text("\n".join(L) + "\n")
    print(f"LaTeX written to {out_path}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv_table2(rows, overlaps, out_path):
    mql_sjl, mql_jql, sjl_jql, all_three, ssl_jql, ssl_mql, ssl_sjl, all_four, total_oas_star = overlaps

    top = pd.DataFrame(rows, columns=['Register', 'Sources', 'Matched', 'Unmatched'])
    top.insert(3, 'Match_rate', top['Matched'] / top['Sources'])

    bottom = pd.DataFrame([
        ['MJL \u2229 SJL',                    mql_sjl],
        ['MJL \u2229 JQL',                    mql_jql],
        ['SJL \u2229 JQL',                    sjl_jql],
        ['MJL \u2229 SJL \u2229 JQL',         all_three],
        ['SSL \u2229 JQL',                    ssl_jql],
        ['SSL \u2229 MJL',                    ssl_mql],
        ['SSL \u2229 SJL',                    ssl_sjl],
        ['All four registries',               all_four],
        ['Union (long-list)',                 total_oas_star],
    ], columns=['Overlap', 'Sources'])

    with open(out_path, 'w') as f:
        f.write("# Registry matching\n")
        top.to_csv(f, index=False)
        f.write("\n# Pairwise overlaps\n")
        bottom.to_csv(f, index=False)

    print(f"CSV written to {out_path}")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _categorise_by_name(name):
    """Classify an unmatched journal title by name patterns."""
    if not name:
        return 'Unknown'
    n = name.lower()
    if any(ord(c) > 127 for c in name) or any(w in n for w in [
        'revue', 'revista', 'rivista', 'zeitschrift', 'cahiers', 'cahier',
        'annales', 'annali', 'ekonomi', 'ekonomisk', 'tijdschrift', 'ekonomika',
        'ekonom', 'zhurnal', 'wirtschaft', 'ragion', 'contab',
        'estudios', 'cuadernos', 'hacienda', 'politica economica',
    ]):
        return 'Non-English / regional language'
    if any(w in n for w in ['newsletter', ' news', 'bulletin', 'today', 'update']):
        return 'Newsletter / trade publication'
    if any(w in n for w in ['working paper', 'discussion paper']):
        return 'Working paper / discussion paper series'
    if any(w in n for w in ['annual', 'yearbook', 'year book']):
        return 'Annual edition / book series'
    if any(w in n for w in ['actuari', 'insurance']):
        return 'Actuarial / insurance'
    if any(w in n for w in ['tax', 'taxation', 'fiscal', ' law ', 'legal', 'juridic']):
        return 'Tax / legal specialist'
    if any(w in n for w in ['educat', 'teaching', 'learning', 'pedagog', 'higher ed']):
        return 'Education / pedagogical'
    return 'Small / specialist source'


def export_title_match_candidates(db):
    """CSV of unmatched registry journals with a likely OA title match for manual review."""
    rel = db.sql(f"""
        SELECT
            candidate_source_idx                             AS oa_source_idx,
            candidate_name                                  AS oa_title,
            similarity                                      AS jaro_winkler,
            ref_name                                        AS registry_title,
            era_journal_name, harzing_journal_name, wos_journal_name, scopus_journal_name
        FROM '{PARQUET}/unmatched_journals.parquet'
        WHERE likely_title_match = true
        ORDER BY similarity DESC
    """)
    rel.show()
    out = DATA / 'title_match_candidates.csv'
    rel.df().to_csv(out, index=False)
    print(f"Title match candidates saved to {out}")


def build_table_exclusions(db):
    """Combined table: all reasons for exclusion from OAS (both stages)."""
    unmatched_df = db.sql(f"""
        SELECT ref_name, era_journal_name, harzing_journal_name, wos_journal_name, scopus_journal_name,
               candidate_name, similarity
        FROM '{PARQUET}/unmatched_journals.parquet'
    """).df()

    unmatched_df = unmatched_df.copy()
    unmatched_df['category'] = unmatched_df['ref_name'].apply(_categorise_by_name)
    name_cats = (unmatched_df.groupby('category').size()
                             .reset_index(name='sources')
                             .sort_values('sources', ascending=False))

    topic_df = db.sql(f"""
        SELECT COALESCE(field_name, 'No topic assigned') AS category,
               COUNT(*) AS sources
        FROM '{PARQUET}/dropped_sources.parquet'
        GROUP BY category
        ORDER BY sources DESC
    """).df()

    # Console print
    print("\n=== EXCLUSION SUMMARY ===")
    for _, r in name_cats.iterrows():
        print(f"  {r['category']:<40} {r['sources']:>6,}")
    print(f"  --- topic density filter ---")
    for _, r in topic_df.iterrows():
        print(f"  {r['category']:<40} {r['sources']:>6,}")

    # LaTeX
    L = []
    L.append(r"\begin{table}[htbp]")
    L.append(r"\centering")
    L.append(r"\caption{Reasons for exclusion from OAS. Stage~1: registry sources with no"
             r" OpenAlex ISSN match. Stage~2: matched sources excluded by topic density filter.}")
    L.append(r"\label{tab:oas_exclusions}")
    L.append(r"\begin{tabular}{lr}")
    L.append(r"\toprule")
    L.append(r"\textbf{Reason} & \textbf{Sources} \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{2}{l}{\textit{Stage 1 — no OpenAlex ISSN or title match}} \\")
    for _, r in name_cats.iterrows():
        L.append(f"\\quad {r['category']} & {_i(r['sources'])} \\\\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{2}{l}{\textit{Stage 2 — excluded by topic density filter}} \\")
    for _, r in topic_df.iterrows():
        L.append(f"\\quad {r['category']} & {_i(r['sources'])} \\\\")
    total = name_cats['sources'].sum() + topic_df['sources'].sum()
    L.append(r"\midrule")
    L.append(f"\\textbf{{Total}} & \\textbf{{{_i(total)}}} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")

    out_tex = TABLES / 'table_exclusions.tex'
    out_tex.write_text("\n".join(L) + "\n")
    print(f"LaTeX written to {out_tex}")

    # CSV with full detail for stage 1 no-match
    unmatched_df[['ref_name', 'era_journal_name', 'wos_journal_name', 'harzing_journal_name',
                  'scopus_journal_name', 'candidate_name', 'similarity', 'category']].to_csv(
        DATA / 'unmatched_classified.csv', index=False)
    print(f"Classified unmatched journals saved to data/unmatched_classified.csv")


def build_table_dropped_sources(db):
    """Table: categorisation of OAS* sources excluded by topic density filter."""
    rel = db.sql(f"""
        SELECT field_name AS top_field,
               COUNT(*) AS sources
        FROM '{PARQUET}/dropped_sources.parquet'
        GROUP BY field_name
        ORDER BY sources DESC
    """)
    rel.show()

    df = rel.df()
    total = df['sources'].sum()

    # Detailed CSV: field + subfield counts
    db.sql(f"""
        SELECT field_name AS top_field, subfield_name AS top_subfield,
               COUNT(*) AS sources
        FROM '{PARQUET}/dropped_sources.parquet'
        GROUP BY field_name, subfield_name
        ORDER BY sources DESC
    """).df().to_csv(DATA / 'table_dropped_sources.csv', index=False)

    # LaTeX
    L = []
    L.append(r"\begin{table}[htbp]")
    L.append(r"\centering")
    L.append(r"\caption{Top field of OAS$^*$ sources excluded by topic density filter.}")
    L.append(r"\label{tab:oas_dropped}")
    L.append(r"\begin{tabular}{lr}")
    L.append(r"\toprule")
    L.append(r"\textbf{Top field} & \textbf{Sources} \\")
    L.append(r"\midrule")
    for _, row in df.iterrows():
        field = row['top_field'] if row['top_field'] else 'No topic assigned'
        L.append(f"{field} & {_i(row['sources'])} \\\\")
    L.append(r"\midrule")
    L.append(f"\\textbf{{Total}} & \\textbf{{{_i(total)}}} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")

    out_tex = TABLES / 'table_dropped_sources.tex'
    out_tex.write_text("\n".join(L) + "\n")
    print(f"LaTeX written to {out_tex}")


def sjl_dropped_by_topic_filter(db):
    """Print SJL sources in OAS* that were removed by the topic density filter."""
    rel = db.sql(f"""
        SELECT o.source_name, o.era_journal_name, o.works_count,
               ROUND(d.econ_bus_density, 3) AS density,
               dr.field_name AS top_field, dr.subfield_name AS top_subfield
        FROM '{PARQUET}/oas_star.parquet' o
        JOIN '{PARQUET}/source_densities.parquet' d USING (source_idx)
        LEFT JOIN '{PARQUET}/dropped_sources.parquet' dr USING (source_idx)
        WHERE o.era_journal_name IS NOT NULL
          AND o.source_idx NOT IN (SELECT source_idx FROM '{PARQUET}/source_master.parquet')
        ORDER BY o.works_count DESC
    """)
    print(f"\n=== SJL SOURCES DROPPED BY TOPIC FILTER ===")
    rel.show()
    rel.df().to_csv(DATA / 'sjl_dropped_sources.csv', index=False)
    print(f"Saved to data/sjl_dropped_sources.csv")


# ---------------------------------------------------------------------------
# Table 3: Corpus features
# ---------------------------------------------------------------------------


def build_table3_data(db):
    """
    Compute Table 3 corpus feature counts for years 2000, 2024, and 2000-24.

    Institution filter: works_per_year > TAU_U, applied globally.
    The filtered corpus is the set of works that have at least one authorship
    from a retained institution.

    Returns a dict:
        {row_label: {'y2000': int, 'y2024': int, 'all': int}, ...}
    """
    row = db.sql(f"""
        WITH retained_inst AS (
            SELECT institution_idx
            FROM '{PARQUET}/corpus_institutions.parquet'
            WHERE works_per_year > {TAU_U}
        ),
        retained_work_idx AS (
            SELECT DISTINCT work_idx
            FROM '{PARQUET}/corpus_authorships.parquet'
            WHERE institution_idx IN (SELECT institution_idx FROM retained_inst)
        ),
        fc AS (
            -- filtered corpus: works with at least one retained-institution authorship
            SELECT w.work_idx, w.source_idx, w.publication_year
            FROM '{PARQUET}/corpus_works.parquet' w
            JOIN retained_work_idx USING (work_idx)
        ),
        -- Works
        w2000 AS (SELECT COUNT(DISTINCT work_idx) AS n FROM fc WHERE publication_year = 2000),
        w2024 AS (SELECT COUNT(DISTINCT work_idx) AS n FROM fc WHERE publication_year = 2024),
        wall  AS (SELECT COUNT(DISTINCT work_idx) AS n FROM fc),
        -- Sources
        s2000 AS (SELECT COUNT(DISTINCT source_idx) AS n FROM fc WHERE publication_year = 2000),
        s2024 AS (SELECT COUNT(DISTINCT source_idx) AS n FROM fc WHERE publication_year = 2024),
        sall  AS (SELECT COUNT(DISTINCT source_idx) AS n FROM fc),
        -- Institutions (distinct institutions active in that year/range)
        i2000 AS (
            SELECT COUNT(DISTINCT a.institution_idx) AS n
            FROM '{PARQUET}/corpus_authorships.parquet' a
            JOIN retained_inst USING (institution_idx)
            WHERE a.work_idx IN (SELECT work_idx FROM fc WHERE publication_year = 2000)
        ),
        i2024 AS (
            SELECT COUNT(DISTINCT a.institution_idx) AS n
            FROM '{PARQUET}/corpus_authorships.parquet' a
            JOIN retained_inst USING (institution_idx)
            WHERE a.work_idx IN (SELECT work_idx FROM fc WHERE publication_year = 2024)
        ),
        iall  AS (SELECT COUNT(*) AS n FROM retained_inst),
        -- Reference counts (out-degree: citer in filtered corpus, grouped by citer year)
        r2000 AS (
            SELECT COUNT(*) AS n
            FROM '{PARQUET}/corpus_references.parquet' r
            JOIN fc ON r.citer_idx = fc.work_idx AND fc.publication_year = 2000
        ),
        r2024 AS (
            SELECT COUNT(*) AS n
            FROM '{PARQUET}/corpus_references.parquet' r
            JOIN fc ON r.citer_idx = fc.work_idx AND fc.publication_year = 2024
        ),
        rall  AS (
            SELECT COUNT(*) AS n
            FROM '{PARQUET}/corpus_references.parquet' r
            JOIN fc ON r.citer_idx = fc.work_idx
        ),
        -- Citation counts (in-degree: cited in filtered corpus, grouped by cited year)
        c2000 AS (
            SELECT COUNT(*) AS n
            FROM '{PARQUET}/corpus_references.parquet' r
            JOIN fc ON r.cited_idx = fc.work_idx AND fc.publication_year = 2000
        ),
        c2024 AS (
            SELECT COUNT(*) AS n
            FROM '{PARQUET}/corpus_references.parquet' r
            JOIN fc ON r.cited_idx = fc.work_idx AND fc.publication_year = 2024
        ),
        call_ AS (
            SELECT COUNT(*) AS n
            FROM '{PARQUET}/corpus_references.parquet' r
            JOIN fc ON r.cited_idx = fc.work_idx
        )
        SELECT
            (SELECT n FROM w2000) AS works_2000,
            (SELECT n FROM w2024) AS works_2024,
            (SELECT n FROM wall)  AS works_all,
            (SELECT n FROM s2000) AS sources_2000,
            (SELECT n FROM s2024) AS sources_2024,
            (SELECT n FROM sall)  AS sources_all,
            (SELECT n FROM i2000) AS inst_2000,
            (SELECT n FROM i2024) AS inst_2024,
            (SELECT n FROM iall)  AS inst_all,
            (SELECT n FROM r2000) AS ref_2000,
            (SELECT n FROM r2024) AS ref_2024,
            (SELECT n FROM rall)  AS ref_all,
            (SELECT n FROM c2000) AS cit_2000,
            (SELECT n FROM c2024) AS cit_2024,
            (SELECT n FROM call_) AS cit_all
    """).fetchone()

    (works_2000, works_2024, works_all,
     src_2000,   src_2024,   src_all,
     inst_2000,  inst_2024,  inst_all,
     ref_2000,   ref_2024,   ref_all,
     cit_2000,   cit_2024,   cit_all) = row

    data = {
        'Works':                    {'y2000': works_2000, 'y2024': works_2024, 'all': works_all},
        'Sources':                  {'y2000': src_2000,   'y2024': src_2024,   'all': src_all},
        'Institutions':             {'y2000': inst_2000,  'y2024': inst_2024,  'all': inst_all},
        'Reference counts (out-degree)': {'y2000': ref_2000, 'y2024': ref_2024, 'all': ref_all},
        'Citation counts (in-degree)':   {'y2000': cit_2000, 'y2024': cit_2024, 'all': cit_all},
    }

    print(f"\n=== TABLE 3: CORPUS FEATURES (τ_U > {TAU_U}) τ_S > {TAU_S}) ===")
    print(f"{'Quantity':<35} {'2000':>10} {'2024':>10} {'2000-24':>12}")
    for label, vals in data.items():
        print(f"{label:<35} {vals['y2000']:>10,} {vals['y2024']:>10,} {vals['all']:>12,}")

    return data


def build_table3_distributions(db):
    """
    Compute Q1 (25th) and Q3 (75th) percentiles for four per-entity metrics,
    for years 2000, 2024, and 2000-24 (pooled).

        works/source          -- distinct works per source (per-year and pooled)
        institutions/work     -- distinct institutions per work (all institutions)
        references/work       -- outgoing intra-corpus references per work
        citations/work        -- incoming intra-corpus references per work

    Returns a dict:
        {row_label: {'q1_2000': float, 'q3_2000': float,
                     'q1_2024': float, 'q3_2024': float,
                     'q1_all':  float, 'q3_all':  float}, ...}
    """
    def _q(cte_sql, year_col='publication_year'):
        return db.sql(f"""
            WITH retained_inst AS (
                SELECT institution_idx FROM '{PARQUET}/corpus_institutions.parquet'
                WHERE works_per_year > {TAU_U}
            ),
            retained_work_idx AS (
                SELECT DISTINCT work_idx FROM '{PARQUET}/corpus_authorships.parquet'
                WHERE institution_idx IN (SELECT institution_idx FROM retained_inst)
            ),
            fc AS (
                SELECT w.work_idx, w.source_idx, w.publication_year
                FROM '{PARQUET}/corpus_works.parquet' w
                JOIN retained_work_idx USING (work_idx)
            ),
            {cte_sql}
            SELECT
                QUANTILE_CONT(val, 0.10) FILTER (WHERE {year_col} = 2000) AS q1_2000,
                QUANTILE_CONT(val, 0.90) FILTER (WHERE {year_col} = 2000) AS q3_2000,
                QUANTILE_CONT(val, 0.10) FILTER (WHERE {year_col} = 2024) AS q1_2024,
                QUANTILE_CONT(val, 0.90) FILTER (WHERE {year_col} = 2024) AS q3_2024,
                QUANTILE_CONT(val, 0.10)                                   AS q1_all,
                QUANTILE_CONT(val, 0.90)                                   AS q3_all
            FROM metric
        """).fetchone()

    wps = _q("""
        wps_yr AS (
            SELECT source_idx, publication_year, COUNT(DISTINCT work_idx) AS val
            FROM fc GROUP BY source_idx, publication_year
        ),
        wps_pool AS (
            SELECT source_idx, COUNT(DISTINCT work_idx) AS val FROM fc GROUP BY source_idx
        ),
        metric AS (
            -- per-year rows for 2000/2024 quantiles; pooled rows for all-years quantile
            SELECT source_idx, publication_year, val FROM wps_yr
            UNION ALL
            -- tag pooled rows with a sentinel year so FILTER can distinguish them,
            -- then override q1_all/q3_all from wps_pool via scalar subquery
            SELECT source_idx, -1 AS publication_year, val FROM wps_pool
        )
    """)
    # wps_pool gives pooled counts; extract q1_all/q3_all directly
    wps_pool_row = db.sql(f"""
        WITH retained_inst AS (
            SELECT institution_idx FROM '{PARQUET}/corpus_institutions.parquet'
            WHERE works_per_year > {TAU_U}
        ),
        retained_work_idx AS (
            SELECT DISTINCT work_idx FROM '{PARQUET}/corpus_authorships.parquet'
            WHERE institution_idx IN (SELECT institution_idx FROM retained_inst)
        ),
        fc AS (
            SELECT w.work_idx, w.source_idx, w.publication_year
            FROM '{PARQUET}/corpus_works.parquet' w JOIN retained_work_idx USING (work_idx)
        ),
        wps_pool AS (
            SELECT COUNT(DISTINCT work_idx) AS val FROM fc GROUP BY source_idx
        )
        SELECT QUANTILE_CONT(val, 0.10) AS q1_all, QUANTILE_CONT(val, 0.90) AS q3_all
        FROM wps_pool
    """).fetchone()

    ipw = _q(f"""
        metric AS (
            SELECT a.work_idx, fc.publication_year, COUNT(DISTINCT a.institution_idx) AS val
            FROM '{PARQUET}/corpus_authorships.parquet' a
            JOIN fc USING (work_idx)
            GROUP BY a.work_idx, fc.publication_year
        )
    """)

    rpw = _q(f"""
        metric AS (
            SELECT r.citer_idx AS work_idx, fc.publication_year, COUNT(*) AS val
            FROM '{PARQUET}/corpus_references.parquet' r
            JOIN fc ON r.citer_idx = fc.work_idx
            GROUP BY r.citer_idx, fc.publication_year
        )
    """)

    cpw = _q(f"""
        metric AS (
            SELECT r.cited_idx AS work_idx, fc.publication_year, COUNT(*) AS val
            FROM '{PARQUET}/corpus_references.parquet' r
            JOIN fc ON r.cited_idx = fc.work_idx
            GROUP BY r.cited_idx, fc.publication_year
        )
    """)

    def _row(r, all_override=None):
        q1_2000, q3_2000, q1_2024, q3_2024, q1_all, q3_all = r
        if all_override:
            q1_all, q3_all = all_override
        return {'d1_2000': q1_2000, 'd9_2000': q3_2000,
                'd1_2024': q1_2024, 'd9_2024': q3_2024,
                'd1_all':  q1_all,  'd9_all':  q3_all}

    dists = {
        'Works per source':              _row(wps, all_override=wps_pool_row),
        'Institutions per work':         _row(ipw),
        'References per work (out-degree)': _row(rpw),
        'Citations per work (in-degree)':   _row(cpw),
    }

    print(f"\n=== TABLE 3: DISTRIBUTIONS (τ_U > {TAU_U}) τ_S > {TAU_S}) ===")
    print(f"{'Quantity':<40} {'2000 D_1':>8} {'D_9':>8} {'2024 D_1':>8} {'D_9':>8} {'All D_1':>8} {'D_9':>8}")
    for label, d in dists.items():
        print(f"{label:<40} {d['d1_2000']:>8.1f} {d['d9_2000']:>8.1f}"
              f" {d['d1_2024']:>8.1f} {d['d9_2024']:>8.1f}"
              f" {d['d1_all']:>8.1f} {d['d9_all']:>8.1f}")

    return dists


def _f(x):
    """Float to 1 decimal place."""
    return f"{x:.1f}"


def write_latex_table3(counts, dists, out_path):
    L = []
    L.append(r"\begin{table}[htbp]")
    L.append(r"\centering")
    L.append(
        r"\caption{Corpus features by year ($\tau_U > "
        rf"{TAU_U}$). "
        r"Upper panel: unique-item counts. "
        r"Lower panel: $D_1$ and $D_9$ deciles of per-entity distributions. "
        r"The 2000--24 column pools the full period.}"
    )
    L.append(r"\label{tab:corpus_features}")
    L.append(r"\begin{tabular}{lrrrrrr}")
    L.append(r"\toprule")
    L.append(
        r" & \multicolumn{2}{c}{2000}"
        r" & \multicolumn{2}{c}{2024}"
        r" & \multicolumn{2}{c}{2000--24} \\"
    )
    L.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    L.append(
        r"Quantity"
        r" & \multicolumn{2}{c}{count}"
        r" & \multicolumn{2}{c}{count}"
        r" & \multicolumn{2}{c}{count} \\"
    )
    L.append(r"\midrule")
    # Count rows: single value spans both sub-columns
    for label, vals in counts.items():
        L.append(
            f"{label}"
            f" & \\multicolumn{{2}}{{r}}{{{_i(vals['y2000'])}}}"
            f" & \\multicolumn{{2}}{{r}}{{{_i(vals['y2024'])}}}"
            f" & \\multicolumn{{2}}{{r}}{{{_i(vals['all'])}}} \\\\"
        )
    L.append(r"\midrule")
    L.append(r"Quantity & $D_1$ & $D_9$ & $D_1$ & $D_9$ & $D_1$ & $D_9$ \\")
    L.append(r"\midrule")
    # Distribution rows
    for label, d in dists.items():
        L.append(
            f"{label}"
            f" & {_f(d['d1_2000'])} & {_f(d['d9_2000'])}"
            f" & {_f(d['d1_2024'])} & {_f(d['d9_2024'])}"
            f" & {_f(d['d1_all'])} & {_f(d['d9_all'])} \\\\"
        )
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")

    out_path.write_text("\n".join(L) + "\n")
    print(f"LaTeX written to {out_path}")


def write_csv_table3(counts, dists, out_path):
    rows = []
    for label, vals in counts.items():
        rows.append({
            'Quantity': label, 'type': 'count',
            '2000_D1': vals['y2000'], '2000_D9': vals['y2000'],
            '2024_D1': vals['y2024'], '2024_D9': vals['y2024'],
            '2000-24_D1': vals['all'], '2000-24_D9': vals['all'],
        })
    for label, d in dists.items():
        rows.append({
            'Quantity': label, 'type': 'distribution',
            '2000_D1': d['d1_2000'], '2000_D9': d['d9_2000'],
            '2024_D1': d['d1_2024'], '2024_D9': d['d9_2024'],
            '2000-24_D1': d['d1_all'], '2000-24_D9': d['d9_all'],
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"CSV written to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    with duckdb.connect() as db:
        t1_data = build_table1_data(db)
        rows, overlaps, overlaps_oas, oas_provenance = build_table2_data(db)
        export_title_match_candidates(db)
        build_table_exclusions(db)
        build_table_dropped_sources(db)
        sjl_dropped_by_topic_filter(db)
        t3_counts = build_table3_data(db)
        t3_dists  = build_table3_distributions(db)
    write_latex_table1(t1_data, TABLES / 'table1_registry_overview.tex')
    write_csv_table1(t1_data,   DATA   / 'table1_registry_overview.csv')
    compile_pdf(TABLES / 'table1_registry_overview.tex')
    write_latex_table2(rows, overlaps, TABLES / 'table2_source_matching.tex')
    write_csv_table2(rows, overlaps, DATA   / 'table2_source_matching.csv')
    compile_pdf(TABLES / 'table2_source_matching.tex')
    write_latex_table3(t3_counts, t3_dists, TABLES / 'table3_corpus_features.tex')
    write_csv_table3(t3_counts, t3_dists, DATA     / 'table3_corpus_features.csv')
    compile_pdf(TABLES / 'table3_corpus_features.tex')


if __name__ == "__main__":
    main()
    print("FINISHED!")
