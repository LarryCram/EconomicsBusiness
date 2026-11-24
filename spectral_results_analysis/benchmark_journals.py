"""
benchmark_journals.py — Compare cardinal journal influence v with external rankings, using error bands.

QSS revision, referee 3 (and 1). Rank correlations would discard the cardinality of v
and its error bands, so each benchmark is compared at its own level of measurement:

  cardinal   JCR Article Influence Score (AIS; mean article = 1, like v):
             log v on log AIS, by field; residuals judged against each journal's band;
             E-vs-B offset at fixed AIS (the calibration prediction)
  tiers      AJG 2024 (4*,4,3,2,1; from Harzing JQL 71) and ABDC 2025 (A*,A,B,C):
             each tier's position on the log v scale, adjacent-tier gaps in band
             widths, journals significantly misplaced (whole band beyond the median of
             an adjacent tier)
  membership FT50 (Harzing JQL 'FT 2016' flag): best v threshold (Youden) and the
             members / non-members whose whole band lies on the wrong side of it

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

ISSN_RE = re.compile(r'\d{4}-\d{3}[\dXx]')
AJG_TIERS = ['4*', '4', '3', '2', '1']
# OpenAlex files Review of Finance's current works under the old-name source
# 'European Finance Review' (49857371); its current ISSNs point at a near-empty source.
ISSN_ALIASES = {'1572-3097': 49857371, '1573-692X': 49857371}
ABDC_TIERS = ['A*', 'A', 'B', 'C']


def issns(*vals):
    out = set()
    for v in vals:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        out.update(m.upper() for m in ISSN_RE.findall(str(v)))
    return out


def issn_map(P: Path) -> dict:
    """ISSN -> source_idx using the registry ISSN lists matched in the pipeline."""
    con = duckdb.connect()
    d = con.execute(f"""
        SELECT sm.source_idx, list_distinct(flatten(list(cjl.unique_issn_list))) AS issns,
               any_value(sm.issn) AS issn_l
        FROM '{P}/source_master.parquet' sm
        LEFT JOIN '{P}/comprehensive_journal_list.parquet' cjl
               ON list_contains(cjl.unique_issn_list, sm.issn)
        GROUP BY sm.source_idx
    """).df()
    m = {}
    for r in d.itertuples():
        for i in issns(r.issn_l, *(r.issns if r.issns is not None else [])):
            m.setdefault(i, r.source_idx)
    m.update(ISSN_ALIASES)
    return m


def match(df: pd.DataFrame, cols, imap) -> pd.Series:
    return df.apply(lambda r: next((imap[i] for i in issns(*[r[c] for c in cols])
                                    if i in imap), np.nan), axis=1)


def load_benchmarks(data: Path, imap) -> pd.DataFrame:
    src = data / 'source_masters'
    h = pd.read_excel(src / 'ecobus_journal_harzing_era.xlsx', sheet_name='Harzing')
    h.columns = [c.split('\n')[0] + ('_' + c.split('\n')[1] if '\n' in c else '')
                 for c in h.columns]
    h['source_idx'] = match(h, ['ISSN'], imap)
    h = h.dropna(subset=['source_idx'])
    h = h.assign(ajg=h['AJG_2024'].astype(str).str.strip().replace({'nan': np.nan}),
                 ft50=h['FT_2016'].eq(1)).groupby('source_idx')[['ajg', 'ft50']].first()

    w = pd.read_excel(src / 'ecobus_journal_harzing_era.xlsx', sheet_name='wos_jcr')
    w['source_idx'] = match(w, ['ISSN', 'eISSN'], imap)
    w = (w.dropna(subset=['source_idx', 'Article Influence Score'])
          .groupby('source_idx')['Article Influence Score'].max().rename('ais'))
    w = w[w > 0]

    a = pd.read_excel(src / 'ABDC-JQL-2025-v1-260326.xlsx', sheet_name='2025 JQL')
    a['source_idx'] = match(a, ['ISSN', 'ISSNOnline'], imap)
    a = (a.dropna(subset=['source_idx'])
          .assign(abdc=lambda x: x['2025 rating'].astype(str).str.strip())
          .groupby('source_idx')['abdc'].first())

    b = pd.concat([h, w, a], axis=1)
    b.index = b.index.astype('int64')
    return b


def bands(boot: Path) -> pd.Series:
    meta = json.loads((boot / 'meta.json').read_text())
    v = np.load(boot / 'v_s_boot.npy')
    with np.errstate(divide='ignore', invalid='ignore'):
        lv = np.log10(np.where(v > 0, v, np.nan))
    lo, hi = np.nanpercentile(lv, 2.5, axis=0), np.nanpercentile(lv, 97.5, axis=0)
    return pd.Series((hi - lo) / 2, index=np.asarray(meta['source_ids'], dtype='int64'),
                     name='hw')


def tier_table(d, col, tiers, model, field, rows):
    g = d.dropna(subset=[col])
    g = g[g[col].isin(tiers)]
    if len(g) < 5:
        return
    med = g.groupby(col)['lv'].median().reindex(tiers)
    hw_med = g['hw'].median()
    for k, t in enumerate(tiers):
        x = g[g[col] == t]
        if x.empty:
            continue
        up = med.iloc[k - 1] if k > 0 else np.nan          # next better tier
        dn = med.iloc[k + 1] if k + 1 < len(tiers) else np.nan
        sig_up = int(((x.lv - x.hw) > up).sum()) if not np.isnan(up) else 0
        sig_dn = int(((x.lv + x.hw) < dn).sum()) if not np.isnan(dn) else 0
        gap = (med.iloc[k - 1] - med.iloc[k]) / (2 * hw_med) if k > 0 else np.nan
        rows.append(dict(model=model, field=field, benchmark=col, tier=t, n=len(x),
                         median_log_v=x.lv.median(), q25=x.lv.quantile(.25),
                         q75=x.lv.quantile(.75), median_halfwidth=x.hw.median(),
                         gap_to_better_tier_in_bandwidths=gap,
                         sig_above_better_tier_median=sig_up,
                         sig_below_worse_tier_median=sig_dn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', required=True)
    args = ap.parse_args()
    paths = load_config()
    bl = next(r for r in load_runs() if r['label'] == 'baseline')
    imap = issn_map(paths.parquet)
    bench = load_benchmarks(paths.data, imap)
    hw = bands(Path(args.boot))
    sm = pd.read_parquet(paths.parquet / 'source_master.parquet',
                         columns=['source_idx', 'source_name', 'field_eb'])
    sm['field_eb'] = sm['field_eb'].fillna('X')
    sm = sm.set_index('source_idx')
    rk = duckdb.connect(str(paths.working / 'rankings.duckdb'), read_only=True)

    summary, detail = [], []
    for m, model in [('0110', 'bipartite'), ('1000', 'journal-only')]:
        t = (f"rk_{bl['run_code']}_{bl['fx']}_tauU{bl['tau_u']}_tauS{bl['tau_s']}"
             f"_vartau_rho0_m{m}_chi50_alpha100")
        v = rk.execute(f"SELECT unit_idx, v FROM {t} WHERE unit_type='S'").df() \
              .set_index('unit_idx')['v']
        d = pd.DataFrame({'v': v}).join(sm).join(bench).join(hw)
        d['lv'] = np.log10(d['v'])
        d['hw'] = d['hw'].fillna(d['hw'].median())
        d['model'] = model
        detail.append(d.reset_index())
        print(f"\n######## {model}: {len(d)} journals; matched AIS {d.ais.notna().sum()}, "
              f"AJG {d.ajg.notna().sum()}, ABDC {d.abdc.notna().sum()}, "
              f"FT50 {int(d.ft50.fillna(False).sum())}; median band half-width "
              f"{d.hw.median():.3f} dex")

        # cardinal: AIS
        print('  AIS (fit log v = a + b log AIS):')
        for f in ['E', 'B', 'A', 'all']:
            x = d.dropna(subset=['ais'])
            x = x if f == 'all' else x[x.field_eb == f]
            if len(x) < 10:
                continue
            la = np.log10(x.ais)
            b, a = np.polyfit(la, x.lv, 1)
            res = x.lv - (a + b * la)
            sig = int((res.abs() > x.hw).sum())
            print(f"    {f:3s} n={len(x):4d} slope={b:.2f} intercept={a:+.2f} "
                  f"scatter={res.std():.3f} dex  |resid|>band: {sig} ({sig / len(x):.0%})")
            summary.append(dict(model=model, field=f, benchmark='AIS', tier='', n=len(x),
                                slope=b, intercept=a, scatter=res.std(), n_sig=sig))
        x = d.dropna(subset=['ais'])
        x = x[x.field_eb.isin(['E', 'B'])]
        X = np.column_stack([np.ones(len(x)), np.log10(x.ais), (x.field_eb == 'B').astype(float)])
        coef = np.linalg.lstsq(X, x.lv, rcond=None)[0]
        print(f"    B offset relative to E at equal AIS: {coef[2]:+.3f} dex (x{10 ** coef[2]:.2f})")
        summary.append(dict(model=model, field='B vs E', benchmark='AIS', tier='offset',
                            n=len(x), intercept=coef[2]))

        # tiers
        rows = []
        for f in ['E', 'B', 'A', 'all']:
            x = d if f == 'all' else d[d.field_eb == f]
            tier_table(x, 'ajg', AJG_TIERS, model, f, rows)
            tier_table(x, 'abdc', ABDC_TIERS, model, f, rows)
        tt = pd.DataFrame(rows)
        summary.extend(tt.to_dict('records'))
        show = tt[tt.field == 'all'][['benchmark', 'tier', 'n', 'median_log_v', 'q25', 'q75',
                                      'gap_to_better_tier_in_bandwidths',
                                      'sig_above_better_tier_median',
                                      'sig_below_worse_tier_median']]
        print('  tiers (all fields):')
        print(show.round(2).to_string(index=False))

        # membership: FT50 among JQL journals
        x = d[d.ajg.notna() | d.ft50.fillna(False)].copy()
        x['ft50'] = x.ft50.fillna(False)
        cand = np.sort(x.lv.unique())
        best = max(cand, key=lambda c: ((x.lv >= c) & x.ft50).sum() / x.ft50.sum()
                   + ((x.lv < c) & ~x.ft50).sum() / (~x.ft50).sum())
        below = x[x.ft50 & ((x.lv + x.hw) < best)]
        above = x[~x.ft50 & ((x.lv - x.hw) > best)]
        print(f"  FT50: threshold v*={10 ** best:.2f}; members {int(x.ft50.sum())}, "
              f"significantly below: {len(below)}; non-members significantly above: {len(above)}")
        print('    members below :', '; '.join(below.sort_values('lv').source_name.str[:32]))
        print('    non-members above:', '; '.join(above.sort_values('lv', ascending=False)
                                                .source_name.str[:32].head(25)))
        summary.append(dict(model=model, field='JQL', benchmark='FT50', tier='threshold',
                            n=len(x), intercept=10 ** best, n_sig=len(below) + len(above),
                            sig_below_worse_tier_median=len(below),
                            sig_above_better_tier_median=len(above)))

    pd.DataFrame(summary).to_csv(paths.data / 'benchmark_journals_summary.csv', index=False)
    out = paths.working / 'benchmarks'
    out.mkdir(exist_ok=True)
    pd.concat(detail).to_csv(out / 'benchmark_journals.csv', index=False)
    print(f"\nwritten: {paths.data / 'benchmark_journals_summary.csv'} and "
          f"{out / 'benchmark_journals.csv'}")


if __name__ == '__main__':
    main()
