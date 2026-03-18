"""
table_maker.py — Generate publishable tables for the paper.

Table 2: Source registry matching statistics.
    Inputs:
        comprehensive_journal_list.parquet  -- pre-OA registry union
        oas_star.parquet                    -- OA-matched long-list (OAS*)
    Outputs (written to data/):
        table2_source_matching.tex          -- LaTeX fragment, \\input{} into paper
        table2_source_matching.csv          -- for inspection / sanity check
"""

from pathlib import Path
import duckdb
import yaml
import pandas as pd

# Load config
config_path = Path('./config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)
    PROJECT_FOLDER = Path(config['PROJECT_ROOT'])
    DATA = PROJECT_FOLDER / Path(config.get('DATA'))
    WORKING = Path(config.get('WORKING'))

PARQUET = WORKING / 'parquet'


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_table2_data(db):
    """Return (rows, overlaps) where:
        rows     = [(label, total, matched, unmatched), ...]  one per registry
        overlaps = (mql_sjl, mql_jql, sjl_jql, all_three, total_oas_star)
    """
    # Pre-match registry sizes: DISTINCT names to collapse join-induced duplicates
    sizes = db.sql(f"""
        SELECT
            COUNT(DISTINCT harzing_journal_name) FILTER (WHERE harzing_journal_name IS NOT NULL) AS jql_total,
            COUNT(DISTINCT wos_journal_name)     FILTER (WHERE wos_journal_name     IS NOT NULL) AS mql_total,
            COUNT(DISTINCT era_journal_name)     FILTER (WHERE era_journal_name     IS NOT NULL) AS sjl_total
        FROM '{PARQUET}/comprehensive_journal_list.parquet'
    """).fetchone()
    jql_total, mql_total, sjl_total = sizes

    # Matched: distinct registry journal names that found >=1 OA source
    matched = db.sql(f"""
        SELECT
            COUNT(DISTINCT harzing_journal_name) FILTER (WHERE harzing_journal_name IS NOT NULL) AS jql_matched,
            COUNT(DISTINCT wos_journal_name)     FILTER (WHERE wos_journal_name     IS NOT NULL) AS mql_matched,
            COUNT(DISTINCT era_journal_name)     FILTER (WHERE era_journal_name     IS NOT NULL) AS sjl_matched
        FROM '{PARQUET}/oas_star.parquet'
    """).fetchone()
    jql_matched, mql_matched, sjl_matched = matched

    # Pairwise overlaps and union: count OA sources (rows), not distinct registry names
    overlaps_raw = db.sql(f"""
        SELECT
            COUNT(*) FILTER (WHERE wos_journal_name IS NOT NULL AND era_journal_name     IS NOT NULL)                       AS mql_sjl,
            COUNT(*) FILTER (WHERE wos_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL)                       AS mql_jql,
            COUNT(*) FILTER (WHERE era_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL)                       AS sjl_jql,
            COUNT(*) FILTER (WHERE wos_journal_name IS NOT NULL AND era_journal_name IS NOT NULL
                                                                AND harzing_journal_name IS NOT NULL)                       AS all_three,
            COUNT(*)                                                                                                         AS total_oas_star
        FROM '{PARQUET}/oas_star.parquet'
    """).fetchone()
    mql_sjl, mql_jql, sjl_jql, all_three, total_oas_star = overlaps_raw

    # Post-filter OAS counts from source_master (after topic density filter)
    oas = db.sql(f"""
        SELECT
            COUNT(DISTINCT harzing_journal_name) FILTER (WHERE harzing_journal_name IS NOT NULL) AS jql_oas,
            COUNT(DISTINCT wos_journal_name)     FILTER (WHERE wos_journal_name     IS NOT NULL) AS mql_oas,
            COUNT(DISTINCT era_journal_name)     FILTER (WHERE era_journal_name     IS NOT NULL) AS sjl_oas,
            COUNT(*) FILTER (WHERE wos_journal_name IS NOT NULL AND era_journal_name     IS NOT NULL) AS mql_sjl_oas,
            COUNT(*) FILTER (WHERE wos_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL) AS mql_jql_oas,
            COUNT(*) FILTER (WHERE era_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL) AS sjl_jql_oas,
            COUNT(*) FILTER (WHERE wos_journal_name IS NOT NULL AND era_journal_name IS NOT NULL
                                                                AND harzing_journal_name IS NOT NULL) AS all_three_oas,
            COUNT(*) AS total_oas
        FROM '{PARQUET}/source_master.parquet'
    """).fetchone()
    jql_oas, mql_oas, sjl_oas, mql_sjl_oas, mql_jql_oas, sjl_jql_oas, all_three_oas, total_oas = oas

    rows = [
        ('JQL', jql_total, jql_matched, jql_total - jql_matched),
        ('MJL', mql_total, mql_matched, mql_total - mql_matched),
        ('SJL', sjl_total, sjl_matched, sjl_total - sjl_matched),
    ]
    overlaps      = (mql_sjl,     mql_jql,     sjl_jql,     all_three,     total_oas_star)
    overlaps_oas  = (mql_sjl_oas, mql_jql_oas, sjl_jql_oas, all_three_oas, total_oas)
    oas_provenance = (jql_oas, mql_oas, sjl_oas)

    # Console summary
    print("\n=== TABLE 2: SOURCE REGISTRY MATCHING ===")
    print(f"{'Register':<8} {'Journals':>9} {'Matched':>9} {'Rate':>7} {'Unmatched':>10}")
    for reg, total, matched, unmatched in rows:
        print(f"{reg:<8} {total:>9,} {matched:>9,} {matched/total*100:>6.1f}% {unmatched:>10,}")
    print(f"\nPairwise overlaps of matched OAS* (long-list, pre-topic filter):")
    print(f"  MJL \u2229 SJL          {mql_sjl:>6,} sources")
    print(f"  MJL \u2229 JQL          {mql_jql:>6,} sources")
    print(f"  SJL \u2229 JQL          {sjl_jql:>6,} sources")
    print(f"  MJL \u2229 SJL \u2229 JQL    {all_three:>6,} sources")
    print(f"  Union (OAS*)     {total_oas_star:>6,} unique sources")
    print(f"\nPost-topic-filter OAS ({total_oas:,} sources):")
    print(f"  {'Register':<8} {'OAS sources':>12}")
    for reg, n in zip(('JQL', 'MJL', 'SJL'), oas_provenance):
        print(f"  {reg:<8} {n:>12,}")
    print(f"  MJL \u2229 SJL          {mql_sjl_oas:>6,} sources")
    print(f"  MJL \u2229 JQL          {mql_jql_oas:>6,} sources")
    print(f"  SJL \u2229 JQL          {sjl_jql_oas:>6,} sources")
    print(f"  MJL \u2229 SJL \u2229 JQL    {all_three_oas:>6,} sources")

    return rows, overlaps, overlaps_oas, oas_provenance


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
    mql_sjl, mql_jql, sjl_jql, all_three, total_oas_star = overlaps

    L = []
    L.append(r"\begin{table}[htbp]")
    L.append(r"\centering")
    L.append(
        r"\caption{Matching of the three register-based sources to OpenAlex source identifiers."
        r" Pairwise overlaps are based on shared matched entries.}"
    )
    L.append(r"\label{tab:source_matching}")
    L.append(r"\begin{tabular}{lrrrr}")
    L.append(r"\toprule")
    L.append(r"Register & Journals & Matched & Match rate & Unmatched \\")
    L.append(r"\midrule")
    for reg, total, matched, unmatched in rows:
        L.append(f"{reg} & {_i(total)} & {_i(matched)} & {_pct(matched, total)} & {_i(unmatched)} \\\\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{5}{l}{\textit{Pairwise overlap of matched OAS$^*$}} \\")
    pairs = [
        (r"MJL $\cap$ SJL",              mql_sjl),
        (r"MJL $\cap$ JQL",              mql_jql),
        (r"SJL $\cap$ JQL",              sjl_jql),
        (r"MJL $\cap$ SJL $\cap$ JQL",  all_three),
    ]
    for label, count in pairs:
        L.append(
            f"\\multicolumn{{2}}{{l}}{{{label}}} & "
            f"\\multicolumn{{3}}{{r}}{{{_i(count)} sources}} \\\\"
        )
    L.append(r"\midrule")
    L.append(
        f"\\multicolumn{{2}}{{l}}{{\\textbf{{Union (long-list)}}}} & "
        f"\\multicolumn{{3}}{{r}}{{\\textbf{{{_i(total_oas_star)} unique OAS}}}} \\\\"
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
    mql_sjl, mql_jql, sjl_jql, all_three, total_oas_star = overlaps

    top = pd.DataFrame(rows, columns=['Register', 'Journals', 'Matched', 'Unmatched'])
    top.insert(3, 'Match_rate', top['Matched'] / top['Journals'])

    bottom = pd.DataFrame([
        ['MJL \u2229 SJL',         mql_sjl],
        ['MJL \u2229 JQL',         mql_jql],
        ['SJL \u2229 JQL',         sjl_jql],
        ['MJL \u2229 SJL \u2229 JQL', all_three],
        ['Union (long-list)',      total_oas_star],
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
    return 'Small / specialist journal'


def export_title_match_candidates(db):
    """CSV of unmatched registry journals with a likely OA title match for manual review."""
    rel = db.sql(f"""
        SELECT
            candidate_source_id                             AS oa_source_id,
            candidate_name                                  AS oa_title,
            similarity                                      AS jaro_winkler,
            ref_name                                        AS registry_title,
            era_journal_name, harzing_journal_name, wos_journal_name
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
        SELECT ref_name, era_journal_name, harzing_journal_name, wos_journal_name,
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
    L.append(r"\caption{Reasons for exclusion from OAS. Stage~1: registry journals with no"
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

    out_tex = DATA / 'table_exclusions.tex'
    out_tex.write_text("\n".join(L) + "\n")
    print(f"LaTeX written to {out_tex}")

    # CSV with full detail for stage 1 no-match
    unmatched_df[['ref_name', 'era_journal_name', 'wos_journal_name', 'harzing_journal_name',
                  'candidate_name', 'similarity', 'category']].to_csv(
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

    out_tex = DATA / 'table_dropped_sources.tex'
    out_tex.write_text("\n".join(L) + "\n")
    print(f"LaTeX written to {out_tex}")


def sjl_dropped_by_topic_filter(db):
    """Print SJL sources in OAS* that were removed by the topic density filter."""
    rel = db.sql(f"""
        SELECT o.source_name, o.era_journal_name, o.works_count,
               ROUND(d.econ_bus_density, 3) AS density,
               dr.field_name AS top_field, dr.subfield_name AS top_subfield
        FROM '{PARQUET}/oas_star.parquet' o
        JOIN '{PARQUET}/source_densities.parquet' d USING (source_id)
        LEFT JOIN '{PARQUET}/dropped_sources.parquet' dr USING (source_id)
        WHERE o.era_journal_name IS NOT NULL
          AND o.source_id NOT IN (SELECT source_id FROM '{PARQUET}/source_master.parquet')
        ORDER BY o.works_count DESC
    """)
    print(f"\n=== SJL SOURCES DROPPED BY TOPIC FILTER ===")
    rel.show()
    rel.df().to_csv(DATA / 'sjl_dropped_sources.csv', index=False)
    print(f"Saved to data/sjl_dropped_sources.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    with duckdb.connect() as db:
        rows, overlaps, overlaps_oas, oas_provenance = build_table2_data(db)
        export_title_match_candidates(db)
        build_table_exclusions(db)
        build_table_dropped_sources(db)
        sjl_dropped_by_topic_filter(db)
    write_latex_table2(rows, overlaps, DATA / 'table2_source_matching.tex')
    write_csv_table2(rows, overlaps, DATA / 'table2_source_matching.csv')


if __name__ == "__main__":
    main()
    print("FINISHED!")
