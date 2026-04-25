"""
check_misplacements.py — Two-stage OA metadata quality check.

Stage 1 (years):
  Build a source quality table (one row per source) and a continuity table
  (one row per anomalous issue).

  Source quality flags are assessed on the most recent RECENT_VOLS volumes
  so that a journal's current policy governs, not its 2000-era behaviour.

    no_page_numbers  — recent volumes have < PAGE_THRESHOLD fraction of works
                       with numeric first_page
    early_access     — recent volumes show > EA_THRESHOLD fraction of issues
                       with pub_year spread > 1  (online-first publishing)

  Sources carrying either flag are excluded from the continuity table.
  The continuity table flags issues whose yr_spread > 1 in clean sources.

Stage 2 (gaps):
  Pagination gap check within (source, volume, issue) groups.
  Top N sources by paginated-work count.
"""

import re
import sys
import argparse
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config

# ── Tuneable constants ─────────────────────────────────────────────────────────
MIN_GAP        = 5     # pages; report gaps >= this value
N_SOURCES      = 16    # default sources for gap check
MAX_VOL        = 800   # volumes above this treated as misencoded years
RECENT_VOLS    = 5     # number of most-recent volumes used to assess source policy
EA_THRESHOLD   = 0.25  # fraction of recent issues with yr_spread > 1 → early_access flag
PAGE_THRESHOLD = 0.50  # fraction of recent works needing numeric first_page to be "paginated"


# ── Data loader (shared) ───────────────────────────────────────────────────────

def _load_all_vol_data(paths) -> pd.DataFrame:
    """
    Load all corpus works that have a parseable integer volume (1–MAX_VOL)
    and non-null issue.  first_page included but may be NULL.

    Columns: source_idx, source_name, work_idx, title,
             publication_year, volume (int), issue (str), first_page (int or NaN)
    """
    corpus = str(paths.parquet / 'corpus_works.parquet')
    db = duckdb.connect(':memory:')
    df = db.execute(f"""
        SELECT
            source_idx,
            source_name,
            work_idx,
            title,
            publication_year,
            TRY_CAST(volume     AS INTEGER) AS volume,
            issue,
            TRY_CAST(first_page AS INTEGER) AS first_page
        FROM '{corpus}'
        WHERE TRY_CAST(volume AS INTEGER) BETWEEN 1 AND {MAX_VOL}
    """).fetchdf()
    db.close()
    return df


# ── Source quality table ───────────────────────────────────────────────────────

def build_source_quality_table(df: pd.DataFrame,
                                recent_vols:    int   = RECENT_VOLS,
                                ea_threshold:   float = EA_THRESHOLD,
                                page_threshold: float = PAGE_THRESHOLD,
                                ) -> pd.DataFrame:
    """
    One row per source.  Flags assessed on the most recent `recent_vols`
    distinct volume numbers to reflect current publishing policy.

    Columns
    -------
    source_idx, source_name,
    n_works_total, vol_min, vol_max,
    -- recent-volume stats --
    n_recent_works, frac_with_page,
    n_recent_issues, n_issues_ea, frac_issues_ea,
    -- flags --
    no_page_numbers  (bool)
    early_access     (bool)
    vol1_catchall    (bool)  — source has vol=1/iss='1' works alongside vol_max > 1
    """
    rows = []

    for (src_idx, src_name), src in df.groupby(['source_idx', 'source_name'],
                                                sort=False):
        vol_min = int(src['volume'].min())
        vol_max = int(src['volume'].max())
        n_total = len(src)

        # ── restrict to most recent volumes ───────────────────────────────────
        recent_vol_nums = sorted(src['volume'].unique())[-recent_vols:]
        recent = src[src['volume'].isin(recent_vol_nums)]

        # no_page_numbers: fraction of recent works with numeric first_page
        n_recent = len(recent)
        n_paged  = int((recent['first_page'].notna() &
                        (recent['first_page'] > 0)).sum())
        frac_page = n_paged / n_recent if n_recent else 0.0

        # early_access: per recent issue, compute yr_spread; flag if > 1
        recent_iss = recent.dropna(subset=['issue'])
        recent_iss = recent_iss[recent_iss['issue'] != '']
        issue_rows = []
        for (vol, iss), grp in recent_iss.groupby(['volume', 'issue'],
                                                   sort=False):
            if len(grp) < 3:
                continue
            spread = int(grp['publication_year'].max() -
                         grp['publication_year'].min())
            issue_rows.append(spread)

        n_recent_issues = len(issue_rows)
        n_ea = sum(1 for s in issue_rows if s > 1)
        frac_ea = n_ea / n_recent_issues if n_recent_issues else 0.0

        has_v1i1 = (
            vol_max > 1 and
            ((src['volume'] == 1) & (src['issue'] == '1')).any()
        )

        rows.append({
            'source_idx':       int(src_idx),
            'source_name':      src_name,
            'n_works_total':    n_total,
            'vol_min':          vol_min,
            'vol_max':          vol_max,
            'n_recent_works':   n_recent,
            'frac_with_page':   round(frac_page, 3),
            'n_recent_issues':  n_recent_issues,
            'n_issues_ea':      n_ea,
            'frac_issues_ea':   round(frac_ea, 3),
            'no_page_numbers':  frac_page < page_threshold,
            'early_access':     frac_ea   > ea_threshold,
            'vol1_catchall':    has_v1i1,
        })

    result = (pd.DataFrame(rows)
                .sort_values(['no_page_numbers', 'early_access', 'vol1_catchall',
                              'n_works_total'],
                             ascending=[False, False, False, False])
                .reset_index(drop=True))
    return result


# ── Issue year-continuity table ────────────────────────────────────────────────

def check_issue_year_clusters(df: pd.DataFrame,
                               good_source_ids: set,
                               min_works: int = 3) -> pd.DataFrame:
    """
    For every (source, volume, issue) group in `good_source_ids` with
    >= min_works works that have numeric first_page, flag issues where
    pub_year spread > 1.

    Returns one row per flagged issue.
    """
    sub = df[
        df['source_idx'].isin(good_source_ids) &
        df['first_page'].notna() &
        (df['first_page'] > 0) &
        df['issue'].notna() &
        (df['issue'] != '')
    ].copy()

    # Modal publication year for every (source, volume) across all its works
    vol_modal = (
        df[df['source_idx'].isin(good_source_ids)]
        .groupby(['source_idx', 'volume'])['publication_year']
        .agg(lambda s: int(s.mode().iloc[0]))
    )

    rows = []
    for (src_idx, src_name, vol, iss), grp in sub.groupby(
            ['source_idx', 'source_name', 'volume', 'issue'], sort=False):
        if len(grp) < min_works:
            continue
        yr_counts  = grp['publication_year'].value_counts().sort_index()
        yr_min     = int(yr_counts.index.min())
        yr_max     = int(yr_counts.index.max())
        yr_spread  = yr_max - yr_min
        if yr_spread <= 1:
            continue

        vol_yr     = int(vol_modal.get((src_idx, vol), yr_min))
        yr_excess  = max(yr_max - vol_yr, vol_yr - yr_min)  # max deviation in either direction

        minority_yr = int(yr_counts.idxmin())
        examples = (grp[grp['publication_year'] == minority_yr]['title']
                    .dropna().head(2).tolist())

        rows.append({
            'source_idx':       int(src_idx),
            'source_name':      src_name,
            'volume':           int(vol),
            'issue':            iss,
            'n_works':          len(grp),
            'yr_min':           yr_min,
            'yr_max':           yr_max,
            'yr_spread':        yr_spread,
            'vol_typical_yr':   vol_yr,
            'yr_excess':        yr_excess,
            'years_and_counts': '{' + ', '.join(
                f'{int(y)}: {int(c)}' for y, c in yr_counts.items()) + '}',
            'example_titles':   ' | '.join(str(t) for t in examples),
        })

    return (pd.DataFrame(rows)
              .sort_values(['yr_excess', 'yr_spread'], ascending=False)
              .reset_index(drop=True))


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_quality_report(qt: pd.DataFrame) -> None:
    n_no_page = qt['no_page_numbers'].sum()
    n_ea      = qt['early_access'].sum()
    n_v1      = qt['vol1_catchall'].sum()
    any_flag  = qt['no_page_numbers'] | qt['early_access'] | qt['vol1_catchall']
    n_clean   = (~any_flag).sum()

    print(f'\nSource quality table — {len(qt):,} sources with numeric volumes')
    print(f'  no_page_numbers : {n_no_page:>5}')
    print(f'  early_access    : {n_ea:>5}')
    print(f'  vol1_catchall   : {n_v1:>5}')
    print(f'  clean (no flags): {n_clean:>5}')

    flagged = qt[any_flag]
    if flagged.empty:
        return
    print(f'\nFlagged sources (showing up to 40):')
    print(f'  {"Source":<50}  {"idx":>12}  {"pg_ok":>5}  {"ea_frac":>7}  flags')
    print('  ' + '─' * 85)
    for _, r in flagged.head(40).iterrows():
        flags = ('NO_PAGE '  if r['no_page_numbers'] else '        ') + \
                ('EA '       if r['early_access']    else '   ')     + \
                ('V1CATCH'   if r['vol1_catchall']   else '')
        print(f'  {r["source_name"][:49]:<50}  {r["source_idx"]:>12}'
              f'  {r["frac_with_page"]:>5.2f}  {r["frac_issues_ea"]:>7.2f}  {flags}')
    if len(flagged) > 40:
        print(f'  ... {len(flagged)-40} more (saved to CSV)')


def print_cluster_report(ct: pd.DataFrame, max_rows: int = 60) -> None:
    if ct.empty:
        print('\nNo year-continuity anomalies found in clean sources.')
        return
    n_src = ct['source_idx'].nunique()
    print(f'\n{len(ct)} issue(s) with yr_spread > 1  across {n_src} clean source(s)\n')
    print(f'  {"Source":<48}  {"vol":>4}  {"iss":>4}  '
          f'{"n":>5}  {"vtyr":>4}  {"xcs":>4}  years(count)')
    print('  ' + '─' * 98)
    for _, r in ct.head(max_rows).iterrows():
        print(f'  {r["source_name"][:47]:<48}  {r["volume"]:>4}  '
              f'{str(r["issue"])[:4]:>4}  {r["n_works"]:>5}  '
              f'{r["vol_typical_yr"]:>4}  {r["yr_excess"]:>4}  {r["years_and_counts"]}')
        if r['example_titles']:
            for t in r['example_titles'].split(' | '):
                print(f'      minority: {t[:80]}')
    if len(ct) > max_rows:
        print(f'  ... {len(ct)-max_rows} more rows (saved to CSV)')


# ── DOI-consistency check ─────────────────────────────────────────────────────

MAX_OUTLIERS      = 2     # works with a foreign token at or below this count are flagged
MIN_DOI_FRAC      = 0.80  # skip sources where fewer than this fraction have DOIs
MIN_DOMINANT_FRAC = 0.90  # skip sources where the modal token covers less than this fraction


def _doi_journal_token(doi: str | None) -> str | None:
    """
    Extract the journal-specific token from a DOI.
    Strategy: strip URL prefix, drop the '10.XXXX/' publisher prefix,
    then take the first alphanumeric run from the suffix.
    e.g.  10.1287/orsc.9.5.600  →  'orsc'
          10.1002/hec.638        →  'hec'
          10.1080/01603477.2005  →  '01603477'
          10.1093/wber/lhae037   →  'wber'
    Returns None for missing or malformed DOIs.
    """
    if not doi:
        return None
    doi = doi.lower()
    doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '')
    parts = doi.split('/', 1)
    if len(parts) < 2 or not parts[1]:
        return None
    token = re.split(r'[./_\-:]', parts[1])[0]
    return token if token else None


def check_doi_consistency(paths,
                           good_source_ids: set,
                           max_outliers:      int   = MAX_OUTLIERS,
                           min_doi_frac:      float = MIN_DOI_FRAC,
                           min_dominant_frac: float = MIN_DOMINANT_FRAC,
                           ) -> tuple[pd.DataFrame, dict]:
    """
    For each clean source, find the dominant DOI journal token and flag
    works whose token is a singleton or small block (≤ max_outliers) that
    differs from the dominant pattern.  Also flags works with no DOI in
    well-DOI'd sources.

    Skips sources with:
      - low DOI coverage (< min_doi_frac)
      - ambiguous dominant token (Elsevier 'j', or coverage < min_dominant_frac)

    Returns (flagged_df, stats_dict).
    """
    corpus = str(paths.parquet / 'corpus_works.parquet')
    ids_str = ','.join(str(x) for x in good_source_ids)
    db = duckdb.connect(':memory:')
    df = db.execute(f"""
        SELECT source_idx, source_name, work_idx, doi, title, publication_year
        FROM '{corpus}'
        WHERE source_idx IN ({ids_str})
    """).fetchdf()
    db.close()

    df['token'] = df['doi'].apply(_doi_journal_token)

    rows = []
    n_skipped = n_checked = 0

    for (src_idx, src_name), grp in df.groupby(['source_idx', 'source_name'],
                                                sort=False):
        n_total    = len(grp)
        n_with_doi = int(grp['doi'].notna().sum())
        doi_frac   = n_with_doi / n_total if n_total else 0.0

        if doi_frac < min_doi_frac:
            n_skipped += 1
            continue

        tok_counts = grp['token'].value_counts(dropna=True)
        if tok_counts.empty:
            n_skipped += 1
            continue

        dominant     = tok_counts.index[0]
        dom_count    = int(tok_counts.iloc[0])
        dom_frac     = dom_count / n_with_doi

        # Skip Elsevier ('j' token) and sources without a clear dominant pattern
        if dominant == 'j' or dom_frac < min_dominant_frac:
            n_skipped += 1
            continue

        n_checked += 1

        # Flag works with a foreign token that is a singleton or small block
        for tok, cnt in tok_counts.items():
            if tok == dominant:
                continue
            # Old DOI schemes that embed the journal token as a prefix (e.g. old Springer)
            if tok.startswith(dominant):
                continue
            # Article-level identifiers (long numeric strings) are not journal codes
            if len(tok) > 12:
                continue
            if cnt <= max_outliers:
                for _, w in grp[grp['token'] == tok].iterrows():
                    rows.append({
                        'source_idx':        int(src_idx),
                        'source_name':       src_name,
                        'work_idx':          int(w['work_idx']),
                        'publication_year':  int(w['publication_year']),
                        'doi':               w['doi'],
                        'doi_token':         tok,
                        'dominant_token':    dominant,
                        'n_token_in_source': int(cnt),
                        'flag_type':         'foreign_token',
                        'title':             w['title'],
                    })

        # Flag no-DOI works (token=None) in well-DOI'd sources
        for _, w in grp[grp['doi'].isna()].iterrows():
            rows.append({
                'source_idx':        int(src_idx),
                'source_name':       src_name,
                'work_idx':          int(w['work_idx']),
                'publication_year':  int(w['publication_year']),
                'doi':               None,
                'doi_token':         None,
                'dominant_token':    dominant,
                'n_token_in_source': 0,
                'flag_type':         'no_doi',
                'title':             w['title'],
            })

    stats = {'n_checked': n_checked, 'n_skipped': n_skipped, 'n_flagged': len(rows)}
    result = (pd.DataFrame(rows)
                .sort_values(['source_name', 'n_token_in_source', 'publication_year'])
                .reset_index(drop=True))
    return result, stats


def print_doi_report(flagged: pd.DataFrame, stats: dict, max_rows: int = 80) -> None:
    print(f"\nDOI-consistency check")
    print(f"  Sources checked : {stats['n_checked']:>5}")
    print(f"  Sources skipped : {stats['n_skipped']:>5}  "
          f"(low DOI coverage, Elsevier, or ambiguous pattern)")
    n_no_doi  = (flagged['flag_type'] == 'no_doi').sum()        if not flagged.empty else 0
    n_foreign = (flagged['flag_type'] == 'foreign_token').sum() if not flagged.empty else 0
    print(f"  Foreign-token works : {n_foreign:>5}")
    print(f"  No-DOI works        : {n_no_doi:>5}  (not shown or saved)")
    if flagged.empty:
        return
    foreign = flagged[flagged['flag_type'] == 'foreign_token'].copy()
    if foreign.empty:
        print('\nNo foreign-token works found.')
        return
    print()
    print(f"  {'Source':<48}  {'n':>3}  {'dom':>8}  {'tok':>10}  yr    title")
    print('  ' + '─' * 105)
    for _, r in foreign.head(max_rows).iterrows():
        print(f"  {r['source_name'][:47]:<48}  {r['n_token_in_source']:>3}  "
              f"{str(r['dominant_token']):>8}  {str(r['doi_token']):>10}  "
              f"{r['publication_year']}  {str(r['title'])[:42]}")
    if len(foreign) > max_rows:
        print(f'  ... {len(foreign)-max_rows} more rows (saved to CSV)')


# ── Pagination gap check ───────────────────────────────────────────────────────

def load_paginated_works(paths, n_sources: int) -> pd.DataFrame:
    corpus = str(paths.parquet / 'corpus_works.parquet')
    db = duckdb.connect(':memory:')
    df = db.execute(f"""
        WITH parsed AS (
            SELECT
                source_idx, source_name, work_idx, title,
                publication_year, issue,
                TRY_CAST(volume     AS INTEGER) AS volume,
                TRY_CAST(first_page AS INTEGER) AS first_page,
                TRY_CAST(last_page  AS INTEGER) AS last_page
            FROM '{corpus}'
            WHERE volume IS NOT NULL AND first_page IS NOT NULL
        ),
        clean AS (
            SELECT * FROM parsed
            WHERE volume BETWEEN 1 AND {MAX_VOL} AND first_page > 0
        ),
        top_sources AS (
            SELECT source_idx FROM clean
            GROUP BY source_idx ORDER BY COUNT(*) DESC LIMIT {n_sources}
        )
        SELECT c.*
        FROM clean c JOIN top_sources t USING (source_idx)
        ORDER BY source_idx, volume, TRY_CAST(issue AS INTEGER), first_page
    """).fetchdf()
    db.close()
    return df


def check_gaps(df: pd.DataFrame, min_gap: int) -> pd.DataFrame:
    records = []
    for (src_idx, vol, iss), grp in df.groupby(
            ['source_idx', 'volume', 'issue'], sort=False, dropna=False):
        if len(grp) < 2:
            continue
        grp = grp.sort_values('first_page').reset_index(drop=True)
        for i in range(len(grp) - 1):
            cur = grp.iloc[i]
            nxt = grp.iloc[i + 1]
            prev_end = (int(cur['last_page']) if pd.notna(cur['last_page'])
                        else int(cur['first_page']))
            gap = int(nxt['first_page']) - prev_end - 1
            if gap >= min_gap:
                records.append({
                    'source_idx':      int(src_idx),
                    'source_name':     cur['source_name'],
                    'volume':          int(vol),
                    'issue':           iss,
                    'prev_work_idx':   int(cur['work_idx']),
                    'prev_year':       int(cur['publication_year']),
                    'prev_title':      cur['title'],
                    'prev_last_page':  prev_end,
                    'next_work_idx':   int(nxt['work_idx']),
                    'next_year':       int(nxt['publication_year']),
                    'next_title':      nxt['title'],
                    'next_first_page': int(nxt['first_page']),
                    'gap_pages':       gap,
                })
    return pd.DataFrame(records)


def print_gap_report(df_works: pd.DataFrame, gaps: pd.DataFrame) -> None:
    src_stats = (df_works.groupby(['source_idx', 'source_name'])
                 .agg(n_works=('work_idx','count'), n_vols=('volume','nunique'))
                 .reset_index().sort_values('n_works', ascending=False))
    gap_counts = (gaps.groupby('source_idx').size()
                  if not gaps.empty else pd.Series(dtype=int))
    print(f'\n{"Source":<55}  {"idx":>12}  {"works":>6}  {"vols":>5}  {"gaps":>5}')
    print('─' * 90)
    for _, row in src_stats.iterrows():
        print(f'{row["source_name"][:54]:<55}  {row["source_idx"]:>12}  '
              f'{row["n_works"]:>6}  {row["n_vols"]:>5}  '
              f'{gap_counts.get(row["source_idx"], 0):>5}')
    if gaps.empty:
        print('\nNo gaps found.')
        return
    print(f'\n{"─"*90}\nGap detail:')
    for src_idx, src_gaps in gaps.groupby('source_idx'):
        print(f'\n  {src_gaps["source_name"].iloc[0]}  (source_idx={src_idx})')
        for _, g in src_gaps.sort_values(['volume','issue','prev_last_page']).iterrows():
            iss = f' iss {g["issue"]}' if pd.notna(g['issue']) else ''
            print(f'    vol {g["volume"]:3d}{iss}  '
                  f'pp.{g["prev_last_page"]}–{g["next_first_page"]}  '
                  f'gap={g["gap_pages"]}pp')
            print(f'      before ({g["prev_year"]}): {str(g["prev_title"])[:72]}')
            print(f'      after  ({g["next_year"]}): {str(g["next_title"])[:72]}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode')

    py = sub.add_parser('years', help='Source quality + issue year-continuity check')
    py.add_argument('--min-works',  type=int,   default=3,
                    help='Min works per issue (default 3)')
    py.add_argument('--recent-vols', type=int,  default=RECENT_VOLS,
                    help=f'Recent volumes used for policy assessment (default {RECENT_VOLS})')
    py.add_argument('--ea',         type=float, default=EA_THRESHOLD,
                    help=f'Early-access flag threshold (default {EA_THRESHOLD})')
    py.add_argument('--page',       type=float, default=PAGE_THRESHOLD,
                    help=f'Min paginated fraction to keep (default {PAGE_THRESHOLD})')
    py.add_argument('--out', default='',
                    help='CSV stem; saves <stem>_quality.csv and <stem>_continuity.csv')

    di = sub.add_parser('dois', help='DOI-consistency check across clean sources')
    di.add_argument('--max-outliers',  type=int,   default=MAX_OUTLIERS,
                    help=f'Flag foreign tokens with at most this many works (default {MAX_OUTLIERS})')
    di.add_argument('--min-doi-frac',  type=float, default=MIN_DOI_FRAC,
                    help=f'Min DOI coverage to check a source (default {MIN_DOI_FRAC})')
    di.add_argument('--min-dom-frac',  type=float, default=MIN_DOMINANT_FRAC,
                    help=f'Min dominant-token coverage to check a source (default {MIN_DOMINANT_FRAC})')
    di.add_argument('--quality-csv',   default='',
                    help='Path to existing quality CSV (skips rebuilding it)')
    di.add_argument('--out', default='',
                    help='Output CSV path for flagged works')

    pg = sub.add_parser('gaps', help='Pagination gap check (top N sources)')
    pg.add_argument('--n',   type=int, default=N_SOURCES)
    pg.add_argument('--gap', type=int, default=MIN_GAP)
    pg.add_argument('--out', default='')

    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        return

    paths = load_config()

    if args.mode == 'years':
        print('Loading volume/issue data for all corpus sources ...', flush=True)
        df = _load_all_vol_data(paths)
        print(f'  {len(df):,} works  ·  {df["source_idx"].nunique():,} sources',
              flush=True)

        print(f'Building source quality table '
              f'(recent_vols={args.recent_vols}) ...', flush=True)
        qt = build_source_quality_table(df,
                                        recent_vols=args.recent_vols,
                                        ea_threshold=args.ea,
                                        page_threshold=args.page)
        print_quality_report(qt)

        good_ids = set(
            qt.loc[~qt['no_page_numbers'] & ~qt['early_access'] & ~qt['vol1_catchall'],
                   'source_idx'])
        print(f'\nChecking year continuity in {len(good_ids)} clean sources ...',
              flush=True)
        ct = check_issue_year_clusters(df, good_ids,
                                       min_works=args.min_works)
        print_cluster_report(ct)

        if args.out:
            stem = args.out.rstrip('.csv')
            qp = Path(f'{stem}_quality.csv')
            cp = Path(f'{stem}_continuity.csv')
            qt.to_csv(qp, index=False)
            ct.to_csv(cp, index=False)
            print(f'\nSaved {qp}  and  {cp}')

    elif args.mode == 'dois':
        if args.quality_csv:
            qt = pd.read_csv(args.quality_csv)
            print(f'Loaded quality table from {args.quality_csv}', flush=True)
        else:
            print('Building source quality table to identify clean sources ...',
                  flush=True)
            df = _load_all_vol_data(paths)
            qt = build_source_quality_table(df)

        good_ids = set(
            qt.loc[~qt['no_page_numbers'] & ~qt['early_access'] & ~qt['vol1_catchall'],
                   'source_idx'])
        print(f'Checking DOI consistency in {len(good_ids)} clean sources ...',
              flush=True)
        flagged, stats = check_doi_consistency(
            paths, good_ids,
            max_outliers=args.max_outliers,
            min_doi_frac=args.min_doi_frac,
            min_dominant_frac=args.min_dom_frac,
        )
        print_doi_report(flagged, stats)
        foreign = flagged[flagged['flag_type'] == 'foreign_token'] if not flagged.empty else flagged
        if args.out and not foreign.empty:
            foreign.to_csv(args.out, index=False)
            print(f'\nSaved {args.out}  ({len(foreign)} foreign-token works)')

    elif args.mode == 'gaps':
        print(f'Loading top {args.n} sources ...', flush=True)
        df = load_paginated_works(paths, args.n)
        gaps = check_gaps(df, args.gap)
        print_gap_report(df, gaps)
        if args.out:
            gaps.to_csv(args.out, index=False)
            print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
    print('\nFINISHED!')
