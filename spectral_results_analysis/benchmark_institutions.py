"""
benchmark_institutions.py — Compare institutional influence with external institution rankings, using error bands.

QSS revision, referee 1 ("institutional rankings are never benchmarked") and referee 3.
As for journals, external rankings are treated as ordered tier lists and compared on
the log scale of our measures, judged against the bootstrap bands; no rank correlations.

External rankings (public; CSVs in data/source_masters):
  ShanghaiRanking GRAS 2025: Economics, Business Administration, Finance, Management
      (ranks 1–50 exact, then bands 51–75 … 401–500)
  RePEc top institutions (ideas.repec.org, 2026-11-24): rank of the best-ranked
      RePEc entity of each institution (universities appear by department)

Both external rankings grow with institution size, while v is influence per work.
Each comparison is therefore made twice: per-work influence v and total influence
pi (pi ∝ v × works). Tiers:
  GRAS : 1–25, 26–50, 51–100, 101–200, 201–300, 301–500
  RePEc: 1–25, 26–50, 51–100, 101–200, 201–469

Bands: 2.5–97.5% interval of log v over the work-resampling bootstrap replicates
(stage4_resample); the same half-width applies to log pi. With the default bootstrap
(the active WORKING, i.e. run on the corrected corpus) every institution has its own band.
With a bootstrap from another WORKING (e.g. the submitted run), institutions changed by
the identity corrections have no carried-over band (flagged; median half-width used).

Matching: normalised names (department suffix after '→' and acronyms in parentheses
removed; Centre/Center unified) within the same country; exact match first, then a
unique prefix match, then a manual alias list. Unmatched entries are listed.

Outputs:
  data/benchmark_institutions_summary.csv     tier statistics
  data/benchmark_institutions_matches.csv     external name -> institution_idx (for review)

Usage:
  ECONBUS_CONFIG=config.rev.yaml .venv/bin/python spectral_results_analysis/benchmark_institutions.py
      [--boot <stage4_resample folder>]
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs
from benchmark_journals import tier_rows

COUNTRY = {
    'Argentina': 'AR', 'Australia': 'AU', 'Austria': 'AT', 'Belgium': 'BE', 'Brazil': 'BR',
    'Canada': 'CA', 'Chile': 'CL', 'China': 'CN', 'China-Hong Kong': 'HK', 'Hong Kong': 'HK',
    'China-Macau': 'MO', 'Macao': 'MO', 'China-Taiwan': 'TW', 'Taiwan': 'TW',
    'Colombia': 'CO', 'Croatia': 'HR', 'Cyprus': 'CY', 'Czech Republic': 'CZ',
    'Denmark': 'DK', 'Finland': 'FI', 'France': 'FR', 'Germany': 'DE', 'Greece': 'GR',
    'Hungary': 'HU', 'India': 'IN', 'Iran': 'IR', 'Ireland': 'IE', 'Israel': 'IL',
    'Italy': 'IT', 'Japan': 'JP', 'Lebanon': 'LB', 'Lithuania': 'LT', 'Luxembourg': 'LU',
    'Malaysia': 'MY', 'Mexico': 'MX', 'Netherlands': 'NL', 'New Zealand': 'NZ',
    'Norway': 'NO', 'Pakistan': 'PK', 'Philippines': 'PH', 'Poland': 'PL', 'Portugal': 'PT',
    'Qatar': 'QA', 'Romania': 'RO', 'Russia': 'RU', 'Saudi Arabia': 'SA', 'Singapore': 'SG',
    'Slovenia': 'SI', 'South Africa': 'ZA', 'South Korea': 'KR', 'Spain': 'ES', 'Sweden': 'SE',
    'Switzerland': 'CH', 'Türkiye': 'TR', 'Turkey': 'TR', 'United Arab Emirates': 'AE',
    'United Kingdom': 'GB', 'United States': 'US', 'Uruguay': 'UY', 'Vietnam': 'VN',
}

# external name (normalised) -> OpenAlex institution name, where the names differ
ALIASES = {
    'board of governors of federal reserve system': 'Federal Reserve Board of Governors',
    'federal reserve board': 'Federal Reserve Board of Governors',
    'london school of economics': 'London School of Economics and Political Science',
    'centre for economic policy research': 'Centre for Economic Policy Research',
    'center for economic policy research': 'Centre for Economic Policy Research',
    'paris school of economics': 'Paris School of Economics',
    'toulouse school of economics': 'Toulouse School of Economics',
    'university of california berkeley': 'University of California, Berkeley',
    'university of california los angeles': 'University of California, Los Angeles',
    'university of california san diego': 'University of California San Diego',
    'ucla': 'University of California, Los Angeles',
    # native-language or variant names in RePEc / GRAS (checked 2026-11-24)
    'world bank group': 'World Bank',
    'oxford university': 'University of Oxford',
    'universita commerciale luigi bocconi': 'Bocconi University',
    'universitat zurich': 'University of Zurich',
    'universiteit van tilburg': 'Tilburg University',
    'rijksuniversiteit groningen': 'University of Groningen',
    'aarhus universitet': 'Aarhus University',
    'alma mater studiorum universita bologna': 'University of Bologna',
    'universiteit van amsterdam': 'University of Amsterdam',
    'kobenhavns universitet': 'University of Copenhagen',
    'erasmus universiteit rotterdam': 'Erasmus University Rotterdam',
    'universite catholique louvain': 'UCLouvain',
    'rheinische friedrich wilhelms universitat bonn': 'University of Bonn',
    'university maryland': 'University of Maryland, College Park',
    'university western ontario': 'Western University',
    'european commission': 'Joint Research Centre',
    'diw berlin': 'German Institute for Economic Research',
    'peter g peterson institute for international economics': 'Peterson Institute for International Economics',
    'osterreichisches institut fur wirtschaftsforschung': 'Austrian Institute of Economic Research',
    'toulouse capitole university': 'Toulouse School of Economics',
    'pompeu fabra university': 'Universitat Pompeu Fabra',
    'university montreal': 'Université de Montréal',
    'university new south wales': 'UNSW Sydney',
    'banca d italia': 'Bank of Italy',
    'sciences po': "Institut d'Etudes Politiques de Paris",
}

RANK_TIERS = {
    'GRAS': [(1, 25, '1-25'), (26, 50, '26-50'), (51, 100, '51-100'), (101, 200, '101-200'),
             (201, 300, '201-300'), (301, 500, '301-500')],
    'RePEc': [(1, 25, '1-25'), (26, 50, '26-50'), (51, 100, '51-100'), (101, 200, '101-200'),
              (201, 469, '201-469')],
}


def norm(s: str) -> str:
    s = unicodedata.normalize('NFKD', str(s))
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.split('→')[0]
    s = re.sub(r'\([^)]*\)', ' ', s)
    s = s.replace('&', ' and ').replace('centre', 'center')
    s = re.sub(r'[^\w ]+', ' ', s)
    w = [x for x in s.split() if x not in {'the', 'of', 'and', 'at', 'de', 'di', 'la', 'le'}]
    return ' '.join(w)


def rank_low(r) -> int:
    return int(str(r).split('-')[0])


def tier_of(rank, kind):
    for lo, hi, lab in RANK_TIERS[kind]:
        if lo <= rank <= hi:
            return lab
    return np.nan


def matcher(ci: pd.DataFrame):
    ci = ci.assign(n=ci.institution_name.map(norm))
    by_cc = {cc: g for cc, g in ci.groupby('country_code')}
    # several ids can share a name (e.g. Joint Research Centre campuses): take the largest
    by_name = (ci.sort_values('works_count').drop_duplicates('institution_name', keep='last')
                 .set_index('institution_name').institution_idx.to_dict())

    def f(name, country):
        cc = COUNTRY.get(country)
        n = norm(name)
        if n in ALIASES and ALIASES[n] in by_name:
            return by_name[ALIASES[n]], 'alias'
        g = by_cc.get(cc)
        if g is None:
            return np.nan, ''
        x = g[g.n == n]
        if len(x):
            return x.sort_values('works_count', ascending=False).institution_idx.iloc[0], 'exact'
        x = g[g.n.str.startswith(n + ' ')
              | (g.n.map(lambda o: n.startswith(o + ' ')) & (g.n.str.len() > 12))]
        if len(x) == 1:
            return x.institution_idx.iloc[0], 'prefix'
        return np.nan, ''
    return f


def bands(boot: Path) -> pd.Series:
    meta = json.loads((boot / 'meta.json').read_text())
    v = np.load(boot / 'v_u_boot.npy')
    with np.errstate(divide='ignore', invalid='ignore'):
        lv = np.log10(np.where(v > 0, v, np.nan))
    lo, hi = np.nanpercentile(lv, 2.5, axis=0), np.nanpercentile(lv, 97.5, axis=0)
    return pd.Series((hi - lo) / 2, index=np.asarray(meta['inst_ids'], dtype='int64'), name='hw')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', default=None,
                    help='stage4_resample folder (default: WORKING/bootstrap_oa_errors/stage4_resample)')
    args = ap.parse_args()
    paths = load_config()
    boot = Path(args.boot) if args.boot else paths.working / 'bootstrap_oa_errors' / 'stage4_resample'
    src = paths.data / 'source_masters'
    bl = next(r for r in load_runs() if r['label'] == 'baseline')
    t = (f"rk_{bl['run_code']}_{bl['fx']}_tauU{bl['tau_u']}_tauS{bl['tau_s']}"
         f"_vartau_rho0_m0110_chi50_alpha100")
    rk = duckdb.connect(str(paths.working / 'rankings.duckdb'), read_only=True)
    d = rk.execute(f"SELECT unit_idx, v, pi, a_p FROM {t} WHERE unit_type='U'").df() \
          .set_index('unit_idx')
    ci = pd.read_parquet(paths.parquet / 'corpus_institutions.parquet')
    ci = ci[ci.institution_idx.isin(d.index)]
    d = d.join(ci.set_index('institution_idx')[['institution_name', 'country_code']])
    corr = pd.read_csv(paths.data / 'institution_corrections.csv')
    corrected = set(corr[corr.status == 'accepted'].new_idx.dropna().astype('int64'))
    if boot.resolve().is_relative_to(paths.working.resolve()):
        corrected = set()        # bootstrap run on this (corrected) corpus: bands are native
    hw = bands(boot)
    d = d.join(hw)
    d['band_carried'] = d.hw.notna() & ~d.index.isin(corrected)
    d.loc[~d.band_carried, 'hw'] = np.nan
    d['hw'] = d['hw'].fillna(d['hw'].median())
    d['lv'] = np.log10(d['v'])
    d['lpi'] = np.log10(d['pi'] / d['pi'].sum())
    match = matcher(ci)

    ext = []
    g = pd.read_csv(src / 'shanghai_gras_2025_econ_business.csv')
    for r in g.itertuples():
        idx, how = match(r.univ_name, r.region)
        ext.append(dict(list=f'GRAS {r.subject}', ext_name=r.univ_name, country=r.region,
                        ext_rank=rank_low(r.ranking), institution_idx=idx, how=how))
    rp = pd.read_csv(src / sorted(p.name for p in src.glob('repec_top_institutions_*.csv'))[-1])
    for r in rp.itertuples():
        idx, how = match(r.name, r.country)
        ext.append(dict(list='RePEc', ext_name=r.name, country=r.country,
                        ext_rank=int(r.repec_rank), institution_idx=idx, how=how))
    e = pd.DataFrame(ext)
    e['oa_name'] = e.institution_idx.map(d.institution_name)
    e.to_csv(paths.data / 'benchmark_institutions_matches.csv', index=False)

    rows = []
    for lst, x in e.dropna(subset=['institution_idx']).groupby('list'):
        kind = 'RePEc' if lst == 'RePEc' else 'GRAS'
        best = x.groupby('institution_idx').ext_rank.min()          # best entity per institution
        tiers = [lab for _, _, lab in RANK_TIERS[kind]]
        y = d.join(best.rename('ext_rank'), how='inner')
        y['tier'] = y.ext_rank.map(lambda r: tier_of(r, kind))
        n_ext = x.ext_name.nunique() if kind == 'GRAS' else len(x)
        n_all = e[e.list == lst].ext_name.nunique()
        print(f"\n=== {lst}: {len(best)} institutions matched ({n_ext} of {n_all} external entries; "
              f"{int((~y.band_carried).sum())} without carried-over band)")
        for meas, col in [('v (per work)', 'lv'), ('pi (total)', 'lpi')]:
            z = y.assign(lv=y[col])
            tt = pd.DataFrame(tier_rows(z, 'tier', tiers, meas, 'all'))
            tt['benchmark'] = lst
            rows += tt.to_dict('records')
            print(f'  {meas}:')
            print(tt[['tier', 'n', 'median_log_v', 'gap_to_better_tier_in_bandwidths',
                      'sig_above_better_tier_median', 'sig_below_worse_tier_median']]
                  .round(2).to_string(index=False))
    pd.DataFrame(rows).rename(columns={'model': 'measure', 'median_log_v': 'median_log'}) \
      .to_csv(paths.data / 'benchmark_institutions_summary.csv', index=False)
    un = e[e.institution_idx.isna()]
    print(f"\nunmatched external entries: {len(un)} of {len(e)} "
          f"(top-50 unmatched: {', '.join((un[un.ext_rank <= 50].ext_name.str[:30] + ' [' + un[un.ext_rank <= 50].list + ']').head(40))})")


if __name__ == '__main__':
    main()
