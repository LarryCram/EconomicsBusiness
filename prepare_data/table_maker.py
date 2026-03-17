"""
table_maker.py — Generate publishable tables for the paper.

Table 2: Source registry matching statistics.
    Inputs:
        comprehensive_journal_list.parquet  — pre-OA registry union
        oas_star.parquet                    — OA-matched long-list (OAS*)
    Outputs (written to data/):
        table2_source_matching.tex          — LaTeX fragment, \input{} into paper
        table2_source_matching.csv          — for inspection / sanity check
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
    # Pre-match registry sizes from the combined registry file
    sizes = db.sql(f"""
        SELECT
            COUNT(*) FILTER (WHERE harzing_journal_name IS NOT NULL) AS jql_total,
            COUNT(*) FILTER (WHERE wos_journal_name     IS NOT NULL) AS mql_total,
            COUNT(*) FILTER (WHERE era_journal_name     IS NOT NULL) AS sjl_total
        FROM '{PARQUET}/comprehensive_journal_list.parquet'
    """).fetchone()
    jql_total, mql_total, sjl_total = sizes

    # OA-matched counts and pairwise overlaps from OAS*
    oas = db.sql(f"""
        SELECT
            COUNT(*) FILTER (WHERE harzing_journal_name IS NOT NULL)                                                           AS jql_matched,
            COUNT(*) FILTER (WHERE wos_journal_name     IS NOT NULL)                                                           AS mql_matched,
            COUNT(*) FILTER (WHERE era_journal_name     IS NOT NULL)                                                           AS sjl_matched,
            COUNT(*) FILTER (WHERE wos_journal_name IS NOT NULL AND era_journal_name     IS NOT NULL)                          AS mql_sjl,
            COUNT(*) FILTER (WHERE wos_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL)                          AS mql_jql,
            COUNT(*) FILTER (WHERE era_journal_name IS NOT NULL AND harzing_journal_name IS NOT NULL)                          AS sjl_jql,
            COUNT(*) FILTER (WHERE wos_journal_name IS NOT NULL AND era_journal_name IS NOT NULL
                                                                AND harzing_journal_name IS NOT NULL)                          AS all_three,
            COUNT(*)                                                                                                            AS total_oas_star
        FROM '{PARQUET}/oas_star.parquet'
    """).fetchone()
    jql_matched, mql_matched, sjl_matched, mql_sjl, mql_jql, sjl_jql, all_three, total_oas_star = oas

    rows = [
        ('JQL', jql_total, jql_matched, jql_total - jql_matched),
        ('MJL', mql_total, mql_matched, mql_total - mql_matched),
        ('SJL', sjl_total, sjl_matched, sjl_total - sjl_matched),
    ]
    overlaps = (mql_sjl, mql_jql, sjl_jql, all_three, total_oas_star)

    # Console summary
    print("\n=== TABLE 2: SOURCE REGISTRY MATCHING ===")
    print(f"{'Register':<8} {'Journals':>9} {'Matched':>9} {'Rate':>7} {'Unmatched':>10}")
    for reg, total, matched, unmatched in rows:
        print(f"{reg:<8} {total:>9,} {matched:>9,} {matched/total*100:>6.1f}% {unmatched:>10,}")
    print(f"\nPairwise overlaps of matched OAS*:")
    print(f"  MJL \u2229 SJL          {mql_sjl:>6,} sources")
    print(f"  MJL \u2229 JQL          {mql_jql:>6,} sources")
    print(f"  SJL \u2229 JQL          {sjl_jql:>6,} sources")
    print(f"  MJL \u2229 SJL \u2229 JQL    {all_three:>6,} sources")
    print(f"  Union (OAS*)     {total_oas_star:>6,} unique sources")

    return rows, overlaps


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
# Entry point
# ---------------------------------------------------------------------------

def main():
    with duckdb.connect() as db:
        rows, overlaps = build_table2_data(db)
    write_latex_table2(rows, overlaps, DATA / 'table2_source_matching.tex')
    write_csv_table2(rows, overlaps, DATA / 'table2_source_matching.csv')


if __name__ == "__main__":
    main()
    print("FINISHED!")
