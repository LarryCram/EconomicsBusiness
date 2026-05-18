"""
analyse_enclaves.py — For each HCW, compare source v vs mean institution v.

For every Highly Cited Work (intra-corpus citations >= 99th pctile by year,
publication years YEAR_LO–YEAR_HI), compute:
  source_v    : spectral v of the publishing source
  mean_inst_v : mean spectral v across all affiliated institutions

Gap = source_v - mean_inst_v.
  Positive  → source ranks higher than its authors' institutions
  Negative  → institutions rank higher than the source  (enclave signal)

Also plots, for each HCW:
  left panel  : log10(source_v of HCW)    vs log10(median source_v of citing works)
  right panel : log10(mean_inst_v of HCW) vs log10(median mean_inst_v of citing works)
"""

from __future__ import annotations
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

_paths = load_config()
_bl    = next(r for r in load_runs() if r['label'] == 'baseline')

RK_TABLE = (f"rk_{_bl['run_code']}_{_bl['fx']}"
            f"_tauU{_bl['tau_u']}_tauS{_bl['tau_s']}"
            f"_vartau_rho0_m0110_chi50_alpha100")

WK_PATH  = f"'{_paths.parquet}/corpus_works.parquet'"
REF_PATH = f"'{_paths.parquet}/corpus_references.parquet'"
SM_PATH  = f"'{_paths.parquet}/source_master.parquet'"
AU_PATH  = f"'{_paths.parquet}/corpus_authorships.parquet'"

YEAR_LO  = 2014
YEAR_HI  = 2023
HCW_PCT  = 99
TOP_N    = 10

FIELD_COLOURS = {'E': '#e41a1c', 'B': '#377eb8', 'A': '#ff7f0e', 'X': '#888888'}

NMF_K    = 10
NMF_SEED = 42

# Topic→cluster mapping derived from NMF run on combined-enclave (source_v<1 AND mean_inst_v<1) HCW.
# Topics 0,2,6,8,9 collapsed into Sustainability (ESG/green/carbon/environment).
_TOPIC_CLUSTERS = {
    0: 'Sustainability',
    1: 'Resource curse & development',
    2: 'Sustainability',
    3: 'Digital transformation & China',
    4: 'COVID-19 & financial markets',
    5: 'Bitcoin & policy uncertainty',
    6: 'Sustainability',
    7: 'Systematic literature reviews',
    8: 'Sustainability',
    9: 'Sustainability',
}

_STOPWORDS = {
    'a', 'an', 'the', 'of', 'in', 'and', 'to', 'for', 'with', 'on', 'by',
    'from', 'is', 'are', 'its', 'it', 'that', 'this', 'as', 'at', 'or',
    'not', 'be', 'we', 'our', 'their', 'using', 'based', 'under', 'into',
    'over', 'between', 'across', 'than', 'more', 'less', 'within', 'through',
    'among', 'but', 'about', 'also', 'both', 'each', 'which', 'when',
    'where', 'how', 'what', 'who', 'been', 'have', 'has', 'had', 'will',
    'can', 'do', 'does', 'did', 'may', 'might', 'would', 'could', 'should',
    'all', 'any', 'no', 'new', 'two', 'one', 'three', 'four', 'five',
    'evidence', 'study', 'effect', 'effects', 'approach', 'paper', 'model',
    'models', 'analysis', 'data', 'empirical', 'theoretical', 'theory',
    'research', 'review', 'role', 'impact', 'social', 'international',
    'firm', 'firms', 'market', 'markets', 'use',
    'stock', 'connectedness', 'volatility', 'management', 'business', 'corporate',
}

import string as _string

def _tokenise(title: str) -> list[str]:
    t = title.lower().translate(str.maketrans('', '', _string.punctuation + '–—'))
    return [w for w in t.split() if w not in _STOPWORDS and len(w) > 2]


def _tfidf_terms(fg_titles: list[str], bg_titles: list[str],
                 top_n: int = 10) -> tuple[list, list]:
    """Returns (top unigrams, top bigrams) each as list of (term, score)."""
    from collections import Counter

    def _ngrams(titles, n):
        result = Counter()
        for t in titles:
            tokens = _tokenise(t)
            if n == 1:
                grams = tokens
            else:
                grams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            result.update(set(grams))
        return result

    fg_n = max(len(fg_titles), 1)
    bg_n = max(len(bg_titles), 1)

    results = []
    for n in (1, 2):
        fg_df = _ngrams(fg_titles, n)
        bg_df = _ngrams(bg_titles, n)
        scores = {
            w: (cnt / fg_n) * (np.log10((bg_n + 1) / (bg_df.get(w, 0) + 1)) + 1)
            for w, cnt in fg_df.items()
        }
        results.append(sorted(scores.items(), key=lambda x: -x[1])[:top_n])

    return results[0], results[1]


def load_v() -> tuple[pd.DataFrame, pd.DataFrame]:
    with duckdb.connect(str(_paths.working / 'rankings.duckdb'), read_only=True) as db:
        src_v  = db.execute(
            f"SELECT unit_idx AS source_idx, v FROM {RK_TABLE} WHERE unit_type='S'"
        ).df()
        inst_v = db.execute(
            f"SELECT unit_idx AS institution_idx, v FROM {RK_TABLE} WHERE unit_type='U'"
        ).df()
    return src_v, inst_v


def compute_hcw(src_v: pd.DataFrame) -> pd.DataFrame:
    """HCW flags for OAS works in YEAR_LO–YEAR_HI with a source v."""
    con = duckdb.connect(':memory:')
    df = con.execute(f"""
        WITH intra AS (
            SELECT cited_idx AS work_idx, COUNT(*) AS n_intra
            FROM {REF_PATH}
            GROUP BY cited_idx
        )
        SELECT
            w.work_idx,
            w.source_idx,
            w.publication_year,
            COALESCE(w.title, '') AS title,
            COALESCE(i.n_intra, 0) AS n_intra
        FROM {WK_PATH} w
        JOIN {SM_PATH} sm ON sm.source_idx = w.source_idx
        LEFT JOIN intra i ON i.work_idx = w.work_idx
        WHERE w.publication_year BETWEEN {YEAR_LO} AND {YEAR_HI}
    """).df()
    con.close()

    thresh = (
        df.groupby('publication_year')['n_intra']
        .quantile(HCW_PCT / 100.0)
        .rename('threshold')
    )
    df = df.merge(thresh, on='publication_year', how='left')
    df['is_hcw'] = df['n_intra'] >= df['threshold']
    df = df[df['is_hcw']].copy()

    df = df.merge(src_v, on='source_idx', how='inner').rename(columns={'v': 'source_v'})
    return df


def mean_inst_v(hcw: pd.DataFrame, inst_v: pd.DataFrame) -> pd.DataFrame:
    """For each HCW, compute mean institution v across all affiliated institutions."""
    hcw_set = hcw[['work_idx']].copy()
    con = duckdb.connect(':memory:')
    con.register('_hcw_set', hcw_set)
    auth = con.execute(f"""
        SELECT a.work_idx, a.institution_idx
        FROM {AU_PATH} a
        JOIN _hcw_set h ON h.work_idx = a.work_idx
        WHERE a.institution_idx IS NOT NULL
    """).df()
    con.close()

    auth = auth.merge(inst_v, on='institution_idx', how='inner')
    mv = (
        auth.groupby('work_idx')['v']
        .mean()
        .rename('mean_inst_v')
        .reset_index()
    )
    return mv


def referer_v_stats(hcw: pd.DataFrame,
                    src_v: pd.DataFrame,
                    inst_v: pd.DataFrame) -> pd.DataFrame:
    """
    For each HCW, compute:
      median_citing_source_v : median source v of corpus works that cite this HCW
      median_citing_inst_v   : median mean-institution v of citing works
    Returns DataFrame with work_idx plus the two median columns.
    """
    hcw_set = hcw[['work_idx']].copy()

    # ── source panel ──────────────────────────────────────────────────────────
    print('  Computing median citing source v…')
    con = duckdb.connect(':memory:')
    con.register('_hcw_set', hcw_set)
    con.register('_src_v', src_v)
    cite_src = con.execute(f"""
        SELECT r.cited_idx AS work_idx, MEDIAN(sv.v) AS median_citing_source_v
        FROM {REF_PATH} r
        JOIN _hcw_set   h  ON h.work_idx      = r.cited_idx
        JOIN {WK_PATH}  w  ON w.work_idx       = r.citer_idx
        JOIN _src_v     sv ON sv.source_idx    = w.source_idx
        GROUP BY r.cited_idx
    """).df()
    con.close()
    print(f'    {len(cite_src):,} HCW with ≥1 ranked citing source')

    # ── institution panel ─────────────────────────────────────────────────────
    # Step 1: all (citer → HCW) pairs
    print('  Fetching citer→HCW pairs…')
    con = duckdb.connect(':memory:')
    con.register('_hcw_set', hcw_set)
    pairs = con.execute(f"""
        SELECT r.citer_idx AS citer, r.cited_idx AS hcw_idx
        FROM {REF_PATH} r
        JOIN _hcw_set h ON h.work_idx = r.cited_idx
    """).df()
    con.close()
    print(f'    {len(pairs):,} citing edges, {pairs["citer"].nunique():,} unique citers')

    # Step 2: mean institution v for each unique citing work
    print('  Computing mean institution v per citing work…')
    unique_citers = pd.DataFrame({'work_idx': pairs['citer'].unique()})
    con = duckdb.connect(':memory:')
    con.register('_citers', unique_citers)
    con.register('_inst_v', inst_v)
    citer_iv = con.execute(f"""
        SELECT a.work_idx, AVG(iv.v) AS mean_inst_v
        FROM {AU_PATH} a
        JOIN _citers  c  ON c.work_idx         = a.work_idx
        JOIN _inst_v  iv ON iv.institution_idx  = a.institution_idx
        WHERE a.institution_idx IS NOT NULL
        GROUP BY a.work_idx
    """).df()
    con.close()
    print(f'    {len(citer_iv):,} citers with ≥1 ranked institution')

    # Step 3: median per HCW
    merged = pairs.merge(citer_iv, left_on='citer', right_on='work_idx', how='inner')
    median_inst = (
        merged.groupby('hcw_idx')['mean_inst_v']
        .median()
        .rename('median_citing_inst_v')
        .reset_index()
        .rename(columns={'hcw_idx': 'work_idx'})
    )

    return cite_src.merge(median_inst, on='work_idx', how='outer')


def plot_referer_v(df: pd.DataFrame, out_path: Path) -> None:
    """
    Two-panel scatter for all HCW.
    Left:  x = log(source_v),    y = log(median source_v of citing works)
    Right: x = log(inst_v),      y = log(median inst_v of citing works)
    Reference lines at log(v)=0 (v=1).
    """
    sns.set_theme(style='whitegrid', font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    panels = [
        ('source_v',    'median_citing_source_v', 'Sources',      axes[0]),
        ('mean_inst_v', 'median_citing_inst_v',   'Institutions', axes[1]),
    ]

    # compute common x range across both panels
    all_lx = []
    for x_col, y_col, _, _ in panels:
        sub = df.dropna(subset=[x_col, y_col])
        sub = sub[(sub[x_col] > 0) & (sub[y_col] > 0)]
        all_lx.append(np.log10(sub[x_col].values))
    lx_all = np.concatenate(all_lx)
    pad = (lx_all.max() - lx_all.min()) * 0.03
    xlim = (lx_all.min() - pad, lx_all.max() + pad)

    for i, (x_col, y_col, panel_title, ax) in enumerate(panels):
        sub = df.dropna(subset=[x_col, y_col]).copy()
        sub = sub[(sub[x_col] > 0) & (sub[y_col] > 0)]

        lx = np.log10(sub[x_col].values)
        ly = np.log10(sub[y_col].values)

        for field, color in FIELD_COLOURS.items():
            mask = sub['field_eb'].values == field
            ax.scatter(lx[mask], ly[mask],
                       c=color, s=6, alpha=0.5, label=field, linewidths=0)

        ax.axhline(0, color='k', lw=0.8, alpha=0.4)
        ax.axvline(0, color='k', lw=0.8, alpha=0.4)
        ax.set_xlim(xlim)
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['-1', '0', '1'])
        ax.yaxis.set_minor_locator(mticker.FixedLocator([-0.5, 0.5]))

        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):d}'))
        ax.xaxis.set_minor_locator(mticker.FixedLocator([-0.5, 0.5]))

        ax.grid(which='minor', axis='both', color='lightgray', linewidth=0.5, linestyle='-')
        ax.tick_params(which='minor', length=0)

        ax.set_xlabel(r'$\log(v)$ HCW', fontsize=9)
        ax.set_title(panel_title, fontsize=11, pad=6)

        if i == 0:
            ax.set_ylabel(r'$\log(v)$ median refs', fontsize=9)
        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = axes[1].legend(handles, labels, title='Field', fontsize=8,
                         title_fontsize=8, markerscale=3, loc='lower right',
                         framealpha=0.85, handletextpad=0.4, borderpad=0.5)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.12, wspace=0.12)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


def print_2x2(df: pd.DataFrame, x_col: str, y_col: str, label: str) -> None:
    """
    2x2 contingency table: HCW v vs 1  ×  median citing v vs 1.
    Prints counts and fractions of total, with row and column marginals.
    """
    sub = df.dropna(subset=[x_col, y_col]).copy()
    sub = sub[(sub[x_col] > 0) & (sub[y_col] > 0)]
    n = len(sub)

    hcw_hi  = sub[x_col] >= 1
    cite_hi = sub[y_col] >= 1

    c = {
        (False, False): ((~hcw_hi) & (~cite_hi)).sum(),
        (False, True ): ((~hcw_hi) &   cite_hi ).sum(),
        (True,  False): (  hcw_hi  & (~cite_hi)).sum(),
        (True,  True ): (  hcw_hi  &   cite_hi ).sum(),
    }

    def cell(v): return f'{v:>6,} ({v/n:5.1%})'

    col_lo = c[False, False] + c[True, False]
    col_hi = c[False, True ] + c[True, True ]
    row_lo = c[False, False] + c[False, True]
    row_hi = c[True,  False] + c[True,  True]

    W = 16
    print(f'\n  2×2 — {label}   (n={n:,})')
    print(f'  {"":20s}  {"citing v < 1":>{W}}  {"citing v ≥ 1":>{W}}  {"| row total":>{W}}')
    print(f'  {"─"*20}  {"─"*W}  {"─"*W}  {"─"*W}')
    for hcw_above, row_label, row_tot in [
        (False, 'HCW v < 1', row_lo),
        (True,  'HCW v ≥ 1', row_hi),
    ]:
        lo = c[(hcw_above, False)]
        hi = c[(hcw_above, True)]
        print(f'  {row_label:20s}  {cell(lo):>{W}}  {cell(hi):>{W}}  | {cell(row_tot):>{W}}')
    print(f'  {"─"*20}  {"─"*W}  {"─"*W}  {"─"*W}')
    print(f'  {"col total":20s}  {cell(col_lo):>{W}}  {cell(col_hi):>{W}}  | {cell(n):>{W}}')


def report_enclave_cell(df: pd.DataFrame, inst_v: pd.DataFrame,
                        mask: pd.Series, label: str) -> None:
    """
    Report for HCW matching mask: top-10 sources, institutions, TF-IDF terms.
    """
    cell = df[mask].copy()
    print(f'\n{"═" * 80}')
    print(f'  {label}   ({len(cell):,} HCW)')
    print(f'{"═" * 80}')

    # total OAS works per source in window (denominator for propensity)
    con = duckdb.connect(':memory:')
    src_totals = con.execute(f"""
        SELECT w.source_idx, COUNT(*) AS n_works
        FROM {WK_PATH} w
        JOIN {SM_PATH} sm ON sm.source_idx = w.source_idx
        WHERE w.publication_year BETWEEN {YEAR_LO} AND {YEAR_HI}
        GROUP BY w.source_idx
    """).df()
    con.close()

    # top sources
    src_counts = (
        cell.groupby(['source_idx', 'source_name'])['work_idx']
        .count().rename('n_hcw')
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    src_counts = src_counts.merge(src_totals, on='source_idx', how='left')
    src_counts['pct'] = src_counts['n_hcw'] / src_counts['n_works']
    print(f'\n  Top 10 sources:')
    print(f'    {"n_hcw":>6}  {"n_works":>7}  {"pct":>6}  source')
    for _, r in src_counts.iterrows():
        print(f'    {r["n_hcw"]:>6}  {r["n_works"]:>7}  {r["pct"]:>5.1%}  {r["source_name"]}')

    # total OAS works per institution in window
    cell_set = cell[['work_idx']].copy()
    con = duckdb.connect(':memory:')
    con.register('_inst_v', inst_v)
    inst_totals = con.execute(f"""
        SELECT a.institution_idx, a.institution_name, COUNT(DISTINCT a.work_idx) AS n_works
        FROM {AU_PATH} a
        JOIN _inst_v iv ON iv.institution_idx = a.institution_idx
        JOIN {WK_PATH} w ON w.work_idx = a.work_idx
        JOIN {SM_PATH} sm ON sm.source_idx = w.source_idx
        WHERE w.publication_year BETWEEN {YEAR_LO} AND {YEAR_HI}
        GROUP BY a.institution_idx, a.institution_name
    """).df()

    con.register('_cell', cell_set)
    auth = con.execute(f"""
        SELECT a.work_idx, a.institution_idx, a.institution_name
        FROM {AU_PATH} a
        JOIN _cell c ON c.work_idx = a.work_idx
        JOIN _inst_v iv ON iv.institution_idx = a.institution_idx
    """).df()
    con.close()

    inst_counts = (
        auth.drop_duplicates(subset=['work_idx', 'institution_idx'])
        .groupby(['institution_idx', 'institution_name'])['work_idx']
        .count().rename('n_hcw')
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    inst_counts = inst_counts.merge(inst_totals[['institution_idx', 'n_works']],
                                    on='institution_idx', how='left')
    inst_counts['pct'] = inst_counts['n_hcw'] / inst_counts['n_works']
    print(f'\n  Top 10 institutions:')
    print(f'    {"n_hcw":>6}  {"n_works":>7}  {"pct":>6}  institution')
    for _, r in inst_counts.iterrows():
        print(f'    {r["n_hcw"]:>6}  {r["n_works"]:>7}  {r["pct"]:>5.1%}  {r["institution_name"]}')

    # TF-IDF terms
    fg_titles = cell['title'].dropna().tolist()
    bg_titles = df['title'].dropna().tolist()
    unigrams, bigrams = _tfidf_terms(fg_titles, bg_titles, top_n=50)
    top_bigrams = bigrams[:10]
    bigram_words = {w for term, _ in top_bigrams for w in term.split()}
    unigrams_filtered = [(t, s) for t, s in unigrams if t not in bigram_words][:10]
    print(f'\n  Top 10 TF-IDF bigrams:')
    for i, (term, score) in enumerate(top_bigrams, 1):
        print(f'    {i:>2}. {term:<30} {score:.4f}')
    print(f'\n  Top 10 TF-IDF unigrams (excluding words in top bigrams):')
    for i, (term, score) in enumerate(unigrams_filtered, 1):
        print(f'    {i:>2}. {term:<25} {score:.4f}')


def report_crypto(df: pd.DataFrame) -> None:
    """Crypto/bitcoin/blockchain HCW: source distribution and enclave membership."""
    kw = r'crypto|bitcoin|blockchain|cryptocurrency|\bdefi\b|\bnft\b'
    crypto = df[df['title'].str.lower().str.contains(kw, na=False, regex=True)].copy()

    print(f'\n{"═" * 80}')
    print(f'  Crypto / Bitcoin / Blockchain HCW   ({len(crypto):,} works)')
    print(f'{"═" * 80}')

    print(f'\n  source_v:  median={crypto["source_v"].median():.3f}'
          f'  mean={crypto["source_v"].mean():.3f}'
          f'  v<1: {(crypto["source_v"]<1).sum()} ({(crypto["source_v"]<1).mean():.1%})')
    print(f'  inst_v:    median={crypto["mean_inst_v"].median():.3f}'
          f'  mean={crypto["mean_inst_v"].mean():.3f}'
          f'  v<1: {(crypto["mean_inst_v"]<1).sum()} ({(crypto["mean_inst_v"]<1).mean():.1%})')

    src_enc  = (crypto['source_v'] < 1) & (crypto['median_citing_source_v'] < 1)
    inst_enc = (crypto['mean_inst_v'] < 1) & (crypto['median_citing_inst_v'] < 1)
    both_enc = (crypto['source_v'] < 1) & (crypto['mean_inst_v'] < 1)
    print(f'\n  In source enclave cell  (source_v<1, citing_src_v<1): {src_enc.sum()}')
    print(f'  In inst enclave cell    (inst_v<1,  citing_inst_v<1):  {inst_enc.sum()}')
    print(f'  In both-v<1 cell        (source_v<1, inst_v<1):        {both_enc.sum()}')

    print(f'\n  Top sources:')
    print(f'    {"n_hcw":>6}  {"med_v":>6}  source')
    src_grp = (crypto.groupby('source_name')
               .agg(n_hcw=('work_idx','count'), med_v=('source_v','median'))
               .sort_values('n_hcw', ascending=False).head(10))
    for name, r in src_grp.iterrows():
        print(f'    {r["n_hcw"]:>6}  {r["med_v"]:>6.3f}  {name}')

    print(f'\n  Top 10 by n_intra:')
    print(f'    {"n_intra":>7}  {"src_v":>6}  {"inst_v":>7}  title')
    for _, r in crypto.nlargest(10, 'n_intra').iterrows():
        print(f'    {r["n_intra"]:>7}  {r["source_v"]:>6.3f}'
              f'  {r["mean_inst_v"]:>7.3f}  {r["title"][:65]}')


def cluster_echo_chambers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run NMF on combined-enclave (source_v<1 AND mean_inst_v<1) HCW titles.
    Returns df with 'topic' (int) and 'cluster' (str) columns; NaN for non-enclave rows.
    Prints top terms per topic so the _TOPIC_CLUSTERS mapping can be verified.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF

    enc_mask = (df['source_v'] < 1) & (df['mean_inst_v'] < 1)
    enc = df[enc_mask].copy()
    print(f'  Clustering {len(enc):,} combined-enclave HCW with NMF (k={NMF_K})…')

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.6,
                          sublinear_tf=True, stop_words='english')
    X   = vec.fit_transform(enc['title'].fillna(''))
    nmf = NMF(n_components=NMF_K, random_state=NMF_SEED)
    W   = nmf.fit_transform(X)

    enc = enc.copy()
    enc['topic']   = W.argmax(axis=1)
    enc['cluster'] = enc['topic'].map(_TOPIC_CLUSTERS)

    terms = vec.get_feature_names_out()
    print('\n  NMF topic top terms:')
    for t in range(NMF_K):
        top_idx  = nmf.components_[t].argsort()[-10:][::-1]
        top_str  = ', '.join(terms[i] for i in top_idx)
        label    = _TOPIC_CLUSTERS.get(t, '?')
        print(f'    Topic {t:>2} → {label:<35} {top_str}')

    return df.merge(enc[['work_idx', 'topic', 'cluster']], on='work_idx', how='left')


def report_echo_chambers(df: pd.DataFrame,
                         src_v: pd.DataFrame,
                         inst_v: pd.DataFrame) -> None:
    """Print per-cluster summaries: n_HCW, total works, sources, institutions, lead journals/institutions."""
    if 'cluster' not in df.columns:
        print('  cluster_echo_chambers() must be called first.')
        return

    con = duckdb.connect(':memory:')
    src_totals = con.execute(f"""
        SELECT w.source_idx, COUNT(*) AS n_works
        FROM {WK_PATH} w
        JOIN {SM_PATH} sm ON sm.source_idx = w.source_idx
        WHERE w.publication_year BETWEEN {YEAR_LO} AND {YEAR_HI}
        GROUP BY w.source_idx
    """).df()
    con.close()

    enc      = df.dropna(subset=['cluster'])
    clusters = sorted(enc['cluster'].unique())

    print(f'\n{"═"*80}')
    print(f'  ECHO CHAMBER SUMMARIES   ({len(enc):,} combined-enclave HCW, {len(clusters)} clusters)')
    print(f'{"═"*80}')

    for cl in clusters:
        cell  = enc[enc['cluster'] == cl]
        n_hcw = len(cell)

        src_hcw = (
            cell.groupby(['source_idx', 'source_name'])['work_idx']
            .count().rename('n_hcw')
            .sort_values(ascending=False)
            .reset_index()
        )
        src_hcw = src_hcw.merge(src_totals, on='source_idx', how='left')
        src_hcw = src_hcw.merge(src_v, on='source_idx', how='left')
        src_hcw['pct'] = src_hcw['n_hcw'] / src_hcw['n_works']
        n_sources   = len(src_hcw)
        total_works = int(src_hcw['n_works'].sum())

        cell_set = cell[['work_idx']].copy()
        con = duckdb.connect(':memory:')
        con.register('_cell', cell_set)
        con.register('_inst_v', inst_v)
        auth = con.execute(f"""
            SELECT a.work_idx, a.institution_idx, a.institution_name
            FROM {AU_PATH} a
            JOIN _cell   c  ON c.work_idx          = a.work_idx
            JOIN _inst_v iv ON iv.institution_idx   = a.institution_idx
        """).df()
        con.close()

        inst_counts = (
            auth.drop_duplicates(subset=['work_idx', 'institution_idx'])
            .groupby(['institution_idx', 'institution_name'])['work_idx']
            .count().rename('n_hcw')
            .sort_values(ascending=False)
            .reset_index()
        )
        inst_counts = inst_counts.merge(inst_v, on='institution_idx', how='left')
        n_institutions = len(inst_counts)

        print(f'\n  ── {cl}')
        print(f'     HCW: {n_hcw:,}  |  sources: {n_sources:,}  |  '
              f'institutions: {n_institutions:,}  |  total works: {total_works:,}')

        print(f'     Lead journals (top 5):')
        for _, r in src_hcw.head(5).iterrows():
            v_str = f'v={r["v"]:.3f}' if pd.notna(r.get('v')) else 'v=n/a'
            print(f'       {int(r["n_hcw"]):>4} HCW / {int(r["n_works"]):>6} works '
                  f'({r["pct"]:.1%})  {r["source_name"]} ({v_str})')

        print(f'     Lead institutions (top 5):')
        for _, r in inst_counts.head(5).iterrows():
            v_str = f'v={r["v"]:.3f}' if pd.notna(r.get('v')) else 'v=n/a'
            print(f'       {int(r["n_hcw"]):>4} HCW  {r["institution_name"]} ({v_str})')


def check_enclave_connectivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each enclave cluster, build the local citation network:
      nodes = cluster HCW + all corpus works that cite those HCW
      edges = all references where both endpoints are in the node set

    Reports and returns: node count, directed edge count, density, connected components.
    Returns DataFrame with columns: cluster, n_nodes, n_edges, density, n_comp.
    """
    enc = df.dropna(subset=['cluster']).copy()
    clusters = sorted(enc['cluster'].unique())

    def _union_find(nodes: set, edges) -> int:
        parent = {n: n for n in nodes}

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        for a, b in edges:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        return len({find(n) for n in nodes})

    print(f'\n{"═"*80}')
    print(f'  ENCLAVE LOCAL NETWORK')
    print(f'  Nodes = cluster HCW + all corpus works citing those HCW')
    print(f'  Edges = all references between any two nodes in the set')
    print(f'{"═"*80}')
    print(f'\n  {"Cluster":<40}  {"Nodes":>7}  {"Edges":>8}  {"Density":>9}  {"Comp.":>6}')
    print(f'  {"─"*40}  {"─"*7}  {"─"*8}  {"─"*9}  {"─"*6}')

    records = []
    for cl in clusters:
        hcw_set = set(enc[enc['cluster'] == cl]['work_idx'].tolist())
        hcw_df  = pd.DataFrame({'work_idx': list(hcw_set)})

        con = duckdb.connect(':memory:')
        con.register('_hcw', hcw_df)
        citers_df = con.execute(f"""
            SELECT DISTINCT r.citer_idx AS work_idx
            FROM {REF_PATH} r
            JOIN _hcw h ON h.work_idx = r.cited_idx
        """).df()
        con.close()

        node_set = hcw_set | set(citers_df['work_idx'].tolist())
        node_df  = pd.DataFrame({'work_idx': list(node_set)})

        con = duckdb.connect(':memory:')
        con.register('_nodes', node_df)
        edges_df = con.execute(f"""
            SELECT r.citer_idx, r.cited_idx
            FROM {REF_PATH} r
            JOIN _nodes n1 ON n1.work_idx = r.citer_idx
            JOIN _nodes n2 ON n2.work_idx = r.cited_idx
        """).df()
        con.close()

        n_nodes   = len(node_set)
        n_edges   = len(edges_df)
        max_edges = n_nodes * (n_nodes - 1)
        density   = n_edges / max_edges if max_edges > 0 else 0.0
        n_comp    = _union_find(node_set, zip(edges_df['citer_idx'], edges_df['cited_idx']))

        print(f'  {cl:<40}  {n_nodes:>7,}  {n_edges:>8,}  {density:>9.2e}  {n_comp:>6,}')
        records.append(dict(cluster=cl, n_nodes=n_nodes, n_edges=n_edges,
                            density=density, n_comp=n_comp))

    return pd.DataFrame(records)


def make_enclave_table(df: pd.DataFrame, conn_stats: pd.DataFrame,
                       out_path: Path) -> None:
    """
    Write a LaTeX booktabs table of enclave clusters to out_path.
    Requires df to have 'cluster', 'source_v', 'median_citing_source_v' columns
    and conn_stats (from check_enclave_connectivity) to have 'cluster', 'n_nodes',
    'n_comp'. Rows sorted by n_hcw descending.
    """
    enc = df.dropna(subset=['cluster']).copy()
    cs  = conn_stats.set_index('cluster')

    rows = []
    for cl in sorted(enc['cluster'].unique()):
        cell         = enc[enc['cluster'] == cl]
        n_hcw        = len(cell)
        med_v_hcw    = cell[['source_v', 'mean_inst_v']].mean(axis=1).median()
        med_v_citing = cell['median_citing_source_v'].dropna().median()
        n_nodes      = int(cs.loc[cl, 'n_nodes']) if cl in cs.index else 0
        n_comp       = int(cs.loc[cl, 'n_comp'])  if cl in cs.index else 0
        rows.append(dict(cluster=cl, n_hcw=n_hcw, n_nodes=n_nodes, n_comp=n_comp,
                         med_v_hcw=med_v_hcw, med_v_citing=med_v_citing))

    rows.sort(key=lambda r: -r['n_hcw'])

    def _esc(s: str) -> str:
        return s.replace('&', r'\&')

    header = (
        r'    Enclave'
        r' & $n_{\text{HCW}}$'
        r' & Works'
        r' & CC'
        r' & $\langle v \rangle_{\text{HCW}}$'
        r' & $\langle v \rangle_{\text{citing}}$ \\'
    )
    caption = (
        r'  \caption{Anti-ranking enclaves: topic clusters of highly cited works (HCW) '
        r'with source $v < 1$ in low-influence units, identified by NMF clustering on '
        r'HCW titles. Works is the count of HCW plus all corpus works citing those HCW '
        r'(1-hop neighbourhood); CC is the number of weakly connected components '
        r'in the induced citation subgraph. '
        r'$\langle v \rangle_{\text{HCW}}$ and $\langle v \rangle_{\text{citing}}$ '
        r'are the median source influence of the HCW and of their citing works respectively.}'
    )

    lines = [
        r'\begin{table}[!htbp]',
        r'  \centering',
        caption,
        r'  \label{tab:enclaves}',
        r'  \small',
        r'  \begin{tabular}{lrrrrr}',
        r'    \toprule',
        header,
        r'    \midrule',
    ]
    for r in rows:
        lines.append(
            f"    {_esc(r['cluster'])}"
            f" & {r['n_hcw']:,}"
            f" & {r['n_nodes']:,}"
            f" & {r['n_comp']:,}"
            f" & {r['med_v_hcw']:.3f}"
            f" & {r['med_v_citing']:.3f} \\\\"
        )
    lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
        '',
    ]

    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Saved enclave table → {out_path}')


def print_table(df: pd.DataFrame, title: str) -> None:
    print(f'\n{"─" * 100}')
    print(f'  {title}')
    print(f'{"─" * 100}')
    print(f'{"#":>3}  {"source_v":>9}  {"mean_inst_v":>11}  {"gap":>8}  '
          f'{"n_intra":>7}  {"year":>4}  {"source / title"}')
    print('─' * 100)
    for rank, row in enumerate(df.itertuples(), 1):
        src_label = str(row.source_name)[:28]
        title_str = str(row.title)[:45]
        print(f'{rank:>3}  {row.source_v:>9.3f}  {row.mean_inst_v:>11.3f}  '
              f'{row.gap:>+8.3f}  {row.n_intra:>7}  {row.publication_year:>4}  '
              f'{src_label} | {title_str}')


def main() -> None:
    print('Loading v scores…')
    src_v, inst_v = load_v()
    print(f'  Sources with v: {len(src_v):,}   Institutions with v: {len(inst_v):,}')

    print('Computing HCW flags (joins corpus_references)…')
    hcw = compute_hcw(src_v)
    print(f'  HCW in {YEAR_LO}–{YEAR_HI}: {len(hcw):,}')

    print('Computing mean institution v per HCW…')
    mv = mean_inst_v(hcw, inst_v)
    print(f'  HCW with ≥1 ranked institution: {len(mv):,}')

    con = duckdb.connect(':memory:')
    sm = con.execute(
        f"SELECT source_idx, source_name, field_eb FROM {SM_PATH}"
    ).df()
    con.close()

    hcw = hcw.merge(mv, on='work_idx', how='inner')
    print(f'  HCW with both source and institution v: {len(hcw):,}')

    df = hcw.merge(sm, on='source_idx', how='left')
    df['gap'] = df['source_v'] - df['mean_inst_v']

    top_src  = df.nlargest(TOP_N, 'gap')
    top_inst = df.nsmallest(TOP_N, 'gap')

    print_table(top_src,  f'Top {TOP_N} HCW: source v >> mean institution v  (gap = source_v − mean_inst_v)')
    print_table(top_inst, f'Top {TOP_N} HCW: mean institution v >> source v  (enclave signal)')

    print(f'\nOverall gap stats across {len(df):,} HCW:')
    print(f'  mean   gap = {df["gap"].mean():+.3f}')
    print(f'  median gap = {df["gap"].median():+.3f}')
    print(f'  std    gap = {df["gap"].std():.3f}')

    print('\nComputing referer v stats…')
    rv = referer_v_stats(df, src_v, inst_v)
    df = df.merge(rv, on='work_idx', how='left')

    print_2x2(df, 'source_v',    'median_citing_source_v', 'source')
    print_2x2(df, 'mean_inst_v', 'median_citing_inst_v',   'institution')

    src_enc  = (df['source_v'] < 1) & (df['median_citing_source_v'] < 1)
    inst_enc = (df['mean_inst_v'] < 1) & (df['median_citing_inst_v'] < 1)
    both_enc = (df['source_v'] < 1) & (df['mean_inst_v'] < 1)

    report_enclave_cell(df, inst_v, src_enc,
                        'Source enclave: source_v < 1  AND  median citing source_v < 1')
    report_enclave_cell(df, inst_v, inst_enc,
                        'Institution enclave: mean_inst_v < 1  AND  median citing inst_v < 1')
    report_enclave_cell(df, inst_v, both_enc,
                        'Combined: source_v < 1  AND  mean_inst_v < 1')

    report_crypto(df)

    print('\nClustering enclave HCW by topic (NMF)…')
    df = cluster_echo_chambers(df)
    report_echo_chambers(df, src_v, inst_v)
    conn_stats = check_enclave_connectivity(df)
    make_enclave_table(df, conn_stats, _paths.tables / 'table_enclaves.tex')

    plot_referer_v(df, _paths.plots / 'enclave_referer_v.pdf')


if __name__ == '__main__':
    main()
