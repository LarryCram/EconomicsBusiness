"""
benchmark_journals.py — Compare cardinal journal influence v with external journal lists, using error bands.

QSS revision, referee 3 ("UT-Dallas, FT50, AJG ... SJR, Eigenfactor or RePEc").
Rank correlations would discard the cardinality of v and its error bands, and the
external indices differ in construction, so every benchmark is treated as an ordered
tier list or a membership list and compared on the log v scale:

  tiers      AJG 2024 (4*..1; Harzing JQL 71), ABDC 2025 (A*..C), JCR 2024 AIS quartile
             (best quartile over a journal's categories), SJR best quartile (if a
             SCImago download is present in data/source_masters/scimagojr*.csv), RePEc
             simple impact factor (quartiles among matched corpus journals)
             -> each tier's median and IQR of log v, adjacent-tier gaps in band widths,
                journals significantly misplaced (whole band beyond the median of an
                adjacent tier)
  membership FT50 (Harzing JQL 'FT 2016' flag), UTD24 (UT Dallas list)
             -> best v threshold (Youden) and members / non-members whose whole band
                lies on the wrong side of it

No cardinal fit to AIS or SJR values is made: agreement is assessed at the tier level
and disagreement is not interpreted as error in either index.

Band: 2.5–97.5% interval of log v over the 1,000 work-resampling bootstrap replicates
(stage4_resample, ω=1 replicates, carried over to the corrected baseline as agreed);
half-width in dex = (log p97.5 − log p2.5)/2, applied around the corrected baseline v.

Models: baseline bipartite (m=0110) and journal-only (m=1000), F=EBAX, from the active
config's rankings.duckdb (config.rev.yaml = corrected baseline).

Outputs:
  data/benchmark_journals_summary.csv          summary statistics only (git-tracked)
  WORKING/benchmarks/benchmark_journals.csv    per-journal detail incl. licensed
                                               benchmark values (kept out of git)

Usage:
  ECONBUS_CONFIG=config.rev.yaml .venv/bin/python spectral_results_analysis/benchmark_journals.py \
      --boot /home/lc/m/working/econ_bus/bootstrap_oa_errors/stage4_resample
"""

import argparse
import json
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

ISSN_RE = re.compile(r'\b(\d{4})-?(\d{3}[\dXx])\b')
# Review of Finance's current ISSNs (its works are under source 49857371, merged in the
# corrected corpus; kept so the script also works on uncorrected data)
ISSN_ALIASES = {'1572-3097': 49857371, '1573-692X': 49857371}

TIERS = {
    'AJG 2024': ['4*', '4', '3', '2', '1'],
    'ABDC 2025': ['A*', 'A', 'B', 'C'],
    'AIS quartile': ['Q1', 'Q2', 'Q3', 'Q4'],
    'SJR quartile': ['Q1', 'Q2', 'Q3', 'Q4'],
    'RePEc IF quartile': ['Q1', 'Q2', 'Q3', 'Q4'],
}

# UT Dallas Top 100 Business School Research Rankings journal list (UTD24).
# The UTD page could not be fetched (TLS error); list as published by UTD — verify.
UTD24 = [
    'The Accounting Review', 'Journal of Accounting and Economics',
    'Journal of Accounting Research', 'Journal of Finance', 'Journal of Financial Economics',
    'Review of Financial Studies', 'Information Systems Research',
    'INFORMS Journal on Computing', 'MIS Quarterly', 'Journal of Consumer Research',
    'Journal of Marketing', 'Journal of Marketing Research', 'Marketing Science',
    'Management Science', 'Operations Research', 'Journal of Operations Management',
    'Manufacturing and Service Operations Management', 'Production and Operations Management',
    'Academy of Management Journal', 'Academy of Management Review',
    'Administrative Science Quarterly', 'Organization Science',
    'Journal of International Business Studies', 'Strategic Management Journal',
]


PUBLISHER_RE = re.compile(
    r'(?i)\b(ltd|inc|universit|department|faculty|school|institut|academ|akad|press|verlag|'
    r'editore|society|association|cent(er|re)|college|bank|ministry|council|foundation|'
    r'publish|publication|wiley|sons|siebeck|springer|elsevier|emerald|sage|taylor|'
    r'routledge|gruyter|brill|kiad)')


def norm(s: str) -> str:
    s = str(s).lower().replace('&', ' and ')
    s = re.sub(r'[^\w ]+', ' ', s)      # keep non-Latin letters (avoid false matches)
    w = s.split()
    if w and w[0] == 'the':
        w = w[1:]
    return ' '.join(w)


def issns(*vals):
    out = set()
    for v in vals:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        out.update(f'{a}-{b}'.upper() for a, b in ISSN_RE.findall(str(v)))
    return out


def issn_map(P: Path) -> dict:
    """ISSN -> source_idx using the registry ISSN lists matched in the pipeline."""
    d = duckdb.connect().execute(f"""
        SELECT sm.source_idx, list_distinct(flatten(list(cjl.unique_issn_list))) AS il,
               any_value(sm.issn) AS issn_l
        FROM '{P}/source_master.parquet' sm
        LEFT JOIN '{P}/comprehensive_journal_list.parquet' cjl
               ON list_contains(cjl.unique_issn_list, sm.issn)
        GROUP BY sm.source_idx
    """).df()
    m = {}
    for r in d.itertuples():
        for i in issns(r.issn_l, *(r.il if r.il is not None else [])):
            m.setdefault(i, r.source_idx)
    m.update(ISSN_ALIASES)
    return m


def match(df: pd.DataFrame, cols, imap) -> pd.Series:
    return df.apply(lambda r: next((imap[i] for i in issns(*[r[c] for c in cols])
                                    if i in imap), np.nan), axis=1)


def name_map(sm: pd.DataFrame) -> dict:
    """normalised journal name -> source_idx, from OpenAlex and registry names."""
    m = {}
    for c in ('source_name', 'harzing_journal_name', 'wos_journal_name',
              'era_journal_name', 'scopus_journal_name'):
        if c in sm.columns:
            for i, n in zip(sm.index, sm[c]):
                if isinstance(n, str) and n.strip():
                    m.setdefault(norm(n), i)
    return m


def repec_match(rp: pd.DataFrame, nmap: dict) -> pd.DataFrame:
    """RePEc titles are 'Journal, Publisher'. Match the title with the last one or two
    comma-separated parts removed (publisher); prefer the one-part split. When several
    RePEc series match one journal, keep the best-supported match (one-part split), then
    the highest impact factor."""
    def f(t):
        parts = t.split(', ')
        for gap in (1, 2):
            k = len(parts) - gap
            # a two-part split is accepted only if the removed part looks like a
            # publisher (titles such as 'Organization, Technology and Management in
            # Construction' must not match 'Organization')
            if gap == 2 and not PUBLISHER_RE.search(parts[-2]):
                continue
            if k >= 1:
                sid = nmap.get(norm(', '.join(parts[:k])))
                if sid is not None:
                    return pd.Series({'source_idx': sid, 'gap': gap})
        return pd.Series({'source_idx': np.nan, 'gap': np.nan})
    m = pd.concat([rp, rp['title_publisher'].apply(f)], axis=1).dropna(subset=['source_idx'])
    return (m.sort_values(['source_idx', 'gap', 'simple_if'], ascending=[True, True, False])
             .groupby('source_idx').head(1))


def load_benchmarks(paths, sm: pd.DataFrame) -> pd.DataFrame:
    src = paths.data / 'source_masters'
    imap = issn_map(paths.parquet)
    nmap = name_map(sm)
    parts = []

    h = pd.read_excel(src / 'ecobus_journal_harzing_era.xlsx', sheet_name='Harzing')
    h.columns = [c.split('\n')[0] + ('_' + c.split('\n')[1] if '\n' in c else '')
                 for c in h.columns]
    h['source_idx'] = match(h, ['ISSN'], imap)
    parts.append(h.dropna(subset=['source_idx'])
                  .assign(**{'AJG 2024': h['AJG_2024'].astype(str).str.strip()
                             .replace({'nan': np.nan}),
                             'FT50': h['FT_2016'].eq(1)})
                  .groupby('source_idx')[['AJG 2024', 'FT50']].first())

    w = pd.read_excel(src / 'ecobus_journal_harzing_era.xlsx', sheet_name='wos_jcr')
    w['source_idx'] = match(w, ['ISSN', 'eISSN'], imap)
    w = w.dropna(subset=['source_idx', 'AIS Quartile'])
    w = w[w['AIS Quartile'].astype(str).str.match(r'Q[1-4]')]
    parts.append(w.groupby('source_idx')['AIS Quartile'].min().rename('AIS quartile'))

    a = pd.read_excel(src / 'ABDC-JQL-2025-v1-260326.xlsx', sheet_name='2025 JQL')
    a['source_idx'] = match(a, ['ISSN', 'ISSNOnline'], imap)
    parts.append(a.dropna(subset=['source_idx'])
                  .assign(abdc=lambda x: x['2025 rating'].astype(str).str.strip())
                  .groupby('source_idx')['abdc'].first().rename('ABDC 2025'))

    sjr_files = sorted(src.glob('scimagojr*.csv'))
    if sjr_files:
        s = pd.read_csv(sjr_files[-1], sep=';')
        s['source_idx'] = match(s, ['Issn'], imap)
        s = s.dropna(subset=['source_idx'])
        s = s[s['SJR Best Quartile'].astype(str).str.match(r'Q[1-4]')]
        parts.append(s.groupby('source_idx')['SJR Best Quartile'].min().rename('SJR quartile'))
        print(f'SJR: {sjr_files[-1].name}')
    else:
        print('SJR: no data/source_masters/scimagojr*.csv — SJR quartiles skipped')

    rp_files = sorted(src.glob('repec_simple_if_*.csv'))
    if rp_files:
        rp = repec_match(pd.read_csv(rp_files[-1]), nmap)
        parts.append(rp.set_index('source_idx')['simple_if'].rename('repec_if'))

    b = pd.concat(parts, axis=1)
    b.index = b.index.astype('int64')
    utd = {nmap.get(norm(n)) for n in UTD24} - {None}
    b = b.reindex(b.index.union(pd.Index(sorted(utd), dtype='int64')))
    b['UTD24'] = b.index.isin(utd)
    missing = [n for n in UTD24 if nmap.get(norm(n)) is None]
    print(f'UTD24: matched {len(utd)} of 24; not in corpus or unmatched: {missing}')
    return b


def bands(boot: Path) -> pd.Series:
    meta = json.loads((boot / 'meta.json').read_text())
    v = np.load(boot / 'v_s_boot.npy')
    with np.errstate(divide='ignore', invalid='ignore'):
        lv = np.log10(np.where(v > 0, v, np.nan))
    lo, hi = np.nanpercentile(lv, 2.5, axis=0), np.nanpercentile(lv, 97.5, axis=0)
    return pd.Series((hi - lo) / 2, index=np.asarray(meta['source_ids'], dtype='int64'),
                     name='hw')


def tier_rows(d, col, tiers, model, field):
    g = d[d[col].isin(tiers)]
    if len(g) < 5:
        return []
    med = g.groupby(col)['lv'].median().reindex(tiers)
    bw = 2 * g['hw'].median()
    rows = []
    for k, t in enumerate(tiers):
        x = g[g[col] == t]
        if x.empty:
            continue
        up = med.iloc[k - 1] if k > 0 else np.nan
        dn = med.iloc[k + 1] if k + 1 < len(tiers) else np.nan
        rows.append(dict(
            model=model, field=field, benchmark=col, tier=t, n=len(x),
            median_log_v=x.lv.median(), q25=x.lv.quantile(.25), q75=x.lv.quantile(.75),
            gap_to_better_tier_in_bandwidths=(up - med.iloc[k]) / bw if k > 0 else np.nan,
            sig_above_better_tier_median=int(((x.lv - x.hw) > up).sum()) if k > 0 else 0,
            sig_below_worse_tier_median=int(((x.lv + x.hw) < dn).sum())
            if not np.isnan(dn) else 0))
    return rows


def membership(d, col, universe_mask, model):
    x = d[universe_mask].copy()
    x[col] = x[col].fillna(False).astype(bool)
    if x[col].sum() == 0:
        return None, None, None
    cand = np.sort(x.lv.unique())
    best = max(cand, key=lambda c: ((x.lv >= c) & x[col]).sum() / x[col].sum()
               + ((x.lv < c) & ~x[col]).sum() / (~x[col]).sum())
    below = x[x[col] & ((x.lv + x.hw) < best)]
    above = x[~x[col] & ((x.lv - x.hw) > best)]
    row = dict(model=model, field='all', benchmark=col, tier='threshold v*', n=int(x[col].sum()),
               median_log_v=best, sig_below_worse_tier_median=len(below),
               sig_above_better_tier_median=len(above))
    return row, below, above


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', required=True)
    args = ap.parse_args()
    paths = load_config()
    bl = next(r for r in load_runs() if r['label'] == 'baseline')
    sm = pd.read_parquet(paths.parquet / 'source_master.parquet')
    sm['field_eb'] = sm['field_eb'].fillna('X')
    sm = sm.set_index('source_idx')
    bench = load_benchmarks(paths, sm)
    hw = bands(Path(args.boot))
    rk = duckdb.connect(str(paths.working / 'rankings.duckdb'), read_only=True)

    summary, detail = [], []
    for m, model in [('0110', 'bipartite'), ('1000', 'journal-only')]:
        t = (f"rk_{bl['run_code']}_{bl['fx']}_tauU{bl['tau_u']}_tauS{bl['tau_s']}"
             f"_vartau_rho0_m{m}_chi50_alpha100")
        v = rk.execute(f"SELECT unit_idx, v FROM {t} WHERE unit_type='S'").df() \
              .set_index('unit_idx')['v']
        d = pd.DataFrame({'v': v}).join(sm[['source_name', 'field_eb']]).join(bench).join(hw)
        d['lv'] = np.log10(d['v'])
        d['hw'] = d['hw'].fillna(d['hw'].median())
        # RePEc: quartiles of simple IF among matched journals in this ranking
        r = d['repec_if'].dropna()
        d['RePEc IF quartile'] = pd.qcut(r.rank(method='first', ascending=False), 4,
                                         labels=['Q1', 'Q2', 'Q3', 'Q4']).astype(str) \
            .reindex(d.index)
        for c in TIERS:
            if c not in d.columns:
                d[c] = np.nan
        d['model'] = model
        detail.append(d.reset_index().rename(columns={'index': 'source_idx'}))
        print(f"\n######## {model}: {len(d)} journals; median band half-width {d.hw.median():.3f} dex")
        print('  matched: ' + ', '.join(f"{c} {d[c].isin(TIERS[c]).sum()}" for c in TIERS)
              + f", FT50 {int(d.FT50.fillna(False).sum())}, UTD24 {int(d.UTD24.fillna(False).sum())}")

        rows = []
        for f in ['all', 'E', 'B', 'A']:
            x = d if f == 'all' else d[d.field_eb == f]
            for c, tiers in TIERS.items():
                rows += tier_rows(x, c, tiers, model, f)
        tt = pd.DataFrame(rows)
        summary.extend(tt.to_dict('records'))
        show = tt[tt.field == 'all'][['benchmark', 'tier', 'n', 'median_log_v',
                                      'gap_to_better_tier_in_bandwidths',
                                      'sig_above_better_tier_median',
                                      'sig_below_worse_tier_median']]
        print(show.round(2).to_string(index=False))

        jql = d['AJG 2024'].notna() | d['FT50'].fillna(False).astype(bool)
        for col, universe in [('FT50', jql), ('UTD24', jql | d['UTD24'].fillna(False).astype(bool))]:
            row, below, above = membership(d, col, universe, model)
            if row is None:
                continue
            summary.append(row)
            print(f"  {col}: threshold v*={10 ** row['median_log_v']:.2f}; members {row['n']}, "
                  f"significantly below {len(below)}, non-members significantly above {len(above)}")
            print('    members below    :', '; '.join(below.sort_values('lv').source_name.str[:34]))
            print('    non-members above:', '; '.join(above.sort_values('lv', ascending=False)
                                                     .source_name.str[:34].head(20)))

    pd.DataFrame(summary).to_csv(paths.data / 'benchmark_journals_summary.csv', index=False)
    out = paths.working / 'benchmarks'
    out.mkdir(exist_ok=True)
    pd.concat(detail).to_csv(out / 'benchmark_journals.csv', index=False)
    print(f"\nwritten: {paths.data / 'benchmark_journals_summary.csv'} and "
          f"{out / 'benchmark_journals.csv'}")


if __name__ == '__main__':
    main()
