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

  cardinal   JCR 2024 Article Influence Score (AIS) values. AIS is also an eigenvector
             measure per article with 1 = average article, so log v and log AIS are
             compared directly: correlation, RMA slope and offset, then AIS is mapped
             onto the v scale with the RMA line (fitted on all matched journals of the
             model) and each journal is classed in band / near / far against its v
             band (as in mncs_comparison.py). Reported by field and by v-rank band,
             with corr(log MNCS, log AIS) for reference when the MNCS comparison has
             been run. AIS differs from v in its citing population (all of WoS), cited
             window (2019-23 cited in 2024) and exclusion of journal self-citations.
             No fit to SJR values is made.
Disagreement is not interpreted as error in either index.

Band: 2.5–97.5% interval of log v over the 1,000 work-resampling bootstrap replicates
(stage4_resample, run on the corrected corpus with ω=0 replicates, 2026-11-26);
half-width in dex = (log p97.5 − log p2.5)/2, applied around the corrected baseline v.

Models: baseline bipartite (m=0110) and journal-only (m=1000), F=EBAX, from the active
config's rankings.duckdb (config.rev.yaml = corrected baseline).

Outputs:
  data/benchmark_journals_summary.csv          summary statistics only (git-tracked)
  data/benchmark_journals_ais.csv              cardinal AIS comparison, summary only
  WORKING/benchmarks/benchmark_journals.csv    per-journal detail incl. licensed
                                               benchmark values (kept out of git)

Usage:
  ECONBUS_CONFIG=config.rev.yaml .venv/bin/python spectral_results_analysis/benchmark_journals.py
      [--boot <stage4_resample folder>]
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
sys.path.insert(0, str(Path(__file__).parent))
from util import load_config, load_runs
from mncs_comparison import classify

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
# Checked 2026-11-24 against the UTD list supplied by the author (24 titles, identical).
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
               any_value(sm.issn) AS issn_l, any_value(sm.works_count) AS works
        FROM '{P}/source_master.parquet' sm
        LEFT JOIN '{P}/comprehensive_journal_list.parquet' cjl
               ON list_contains(cjl.unique_issn_list, sm.issn)
        GROUP BY sm.source_idx
    """).df()
    # an ISSN can reach several sources through shared registry rows; resolve
    # deterministically: the source whose own ISSN-L it is, then most works, then lowest id
    cand = {}
    for r in d.itertuples():
        own = issns(r.issn_l)
        for i in issns(r.issn_l, *(r.il if r.il is not None else [])):
            key = (i not in own, -(r.works if r.works == r.works else 0), r.source_idx)
            if i not in cand or key < cand[i][0]:
                cand[i] = (key, r.source_idx)
    m = {i: sid for i, (_, sid) in cand.items()}
    m.update(ISSN_ALIASES)
    return m


def match(df: pd.DataFrame, cols, imap) -> pd.Series:
    # sorted: set iteration order depends on string hashing and differs between runs
    return df.apply(lambda r: next((imap[i] for i in sorted(issns(*[r[c] for c in cols]))
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
    w['ais'] = pd.to_numeric(w['Article Influence Score'], errors='coerce')
    # AIS is a journal value; the rare differences between category rows come from
    # several JCR titles sharing an ISSN match: keep the largest
    parts.append(w[w['ais'] > 0].dropna(subset=['source_idx']).groupby('source_idx')['ais'].max())
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


def rma_line(x: np.ndarray, y: np.ndarray):
    """Reduced-major-axis fit y = a + b x; returns (corr, slope b, intercept a)."""
    r = np.corrcoef(x, y)[0, 1]
    b = np.sign(r) * np.std(y) / np.std(x)
    return r, b, np.mean(y) - b * np.mean(x)


def ais_summary(d: pd.DataFrame, model: str) -> pd.DataFrame:
    """Cardinal comparison of log v with log AIS. d needs lv, hw, ais, field_eb and
    optionally mncs; adds columns ais_gap and ais_class (in place) and returns one row
    per field and per v-rank band."""
    m = d['ais'].notna() & (d['ais'] > 0)
    x = d[m]
    la = np.log10(x['ais'])
    _, b, a = rma_line(la.to_numpy(), x['lv'].to_numpy())
    gap = x['lv'] - (a + b * la)
    d['ais_gap'] = gap.reindex(d.index)
    d['ais_class'] = pd.Series(classify((gap - x['hw']).to_numpy(), (gap + x['hw']).to_numpy()),
                               index=x.index).reindex(d.index)
    rank = d['lv'].rank(ascending=False)
    bands = pd.cut(rank, [0, 10, 100, 500, np.inf], labels=['1-10', '11-100', '101-500', '501+'])
    groups = ([('field', 'all', m)] + [('field', f, m & (d.field_eb == f)) for f in ['E', 'B', 'A', 'X']]
              + [('v_rank', str(lab), m & (bands == lab)) for lab in bands.cat.categories])
    rows = []
    for gt, g, mask in groups:
        y = d[mask]
        if len(y) < 5:
            continue
        ly, lay = y['lv'].to_numpy(), np.log10(y['ais'].to_numpy())
        r, slope, icpt = rma_line(lay, ly)
        cls = y['ais_class']
        row = dict(model=model, group_type=gt, group=g, n=len(y),
                   corr_logv_logais=r, spearman=y['lv'].corr(y['ais'], method='spearman'),
                   rma_slope=slope, rma_intercept=icpt,
                   median_logv_minus_logais=np.median(ly - lay),
                   sd_logv_minus_logais=np.std(ly - lay, ddof=1),
                   median_gap_to_line=y['ais_gap'].median(),
                   median_abs_gap_to_line=y['ais_gap'].abs().median(),
                   median_band_halfwidth=y['hw'].median(),
                   in_band=int((cls == 'in').sum()), near=int((cls == 'near').sum()),
                   far=int((cls == 'far').sum()))
        if 'mncs' in y.columns and y['mncs'].gt(0).sum() > 2:
            z = y[y['mncs'] > 0]
            row['corr_logmncs_logais'] = np.corrcoef(np.log10(z['mncs']), np.log10(z['ais']))[0, 1]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', default=None,
                    help='stage4_resample folder (default: WORKING/bootstrap_oa_errors/stage4_resample)')
    args = ap.parse_args()
    paths = load_config()
    boot = Path(args.boot) if args.boot else paths.working / 'bootstrap_oa_errors' / 'stage4_resample'
    bl = next(r for r in load_runs() if r['label'] == 'baseline')
    sm = pd.read_parquet(paths.parquet / 'source_master.parquet')
    sm['field_eb'] = sm['field_eb'].fillna('X')
    sm = sm.set_index('source_idx')
    bench = load_benchmarks(paths, sm)
    hw = bands(boot)
    rk = duckdb.connect(str(paths.working / 'rankings.duckdb'), read_only=True)
    mncs_file = paths.working / 'benchmarks' / 'mncs_comparison_units.csv'
    mncs = None
    if mncs_file.exists():
        mn = pd.read_csv(mncs_file)
        mncs = mn[mn.unit_type == 'S'].set_index('unit_idx')['mncs']
    else:
        print(f'MNCS: {mncs_file} not found — run mncs_comparison.py for the MNCS reference column')

    summary, detail, ais_rows = [], [], []
    for m, model in [('0110', 'bipartite'), ('1000', 'journal-only')]:
        t = (f"rk_{bl['run_code']}_{bl['fx']}_tauU{bl['tau_u']}_tauS{bl['tau_s']}"
             f"_vartau_rho0_m{m}_chi50_alpha100")
        v = rk.execute(f"SELECT unit_idx, v FROM {t} WHERE unit_type='S'").df() \
              .set_index('unit_idx')['v']
        d = pd.DataFrame({'v': v}).join(sm[['source_name', 'field_eb']]).join(bench).join(hw)
        d['lv'] = np.log10(d['v'])
        d['hw'] = d['hw'].fillna(d['hw'].median())
        if mncs is not None:
            d['mncs'] = mncs.reindex(d.index)
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

        at = ais_summary(d, model)
        ais_rows.append(at)
        print('  AIS (cardinal):')
        print(at[['group_type', 'group', 'n', 'corr_logv_logais', 'rma_slope',
                  'median_logv_minus_logais', 'median_abs_gap_to_line', 'median_band_halfwidth',
                  'in_band', 'near', 'far'] + (['corr_logmncs_logais'] if 'corr_logmncs_logais' in at else [])]
              .round(3).to_string(index=False))

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
    pd.concat(ais_rows).round(4).to_csv(paths.data / 'benchmark_journals_ais.csv', index=False)
    out = paths.working / 'benchmarks'
    out.mkdir(exist_ok=True)
    pd.concat(detail).to_csv(out / 'benchmark_journals.csv', index=False)
    print(f"\nwritten: {paths.data / 'benchmark_journals_summary.csv'}, "
          f"{paths.data / 'benchmark_journals_ais.csv'} and {out / 'benchmark_journals.csv'}")


if __name__ == '__main__':
    main()
