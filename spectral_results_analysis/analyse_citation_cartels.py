"""

analyse_citation_cartels.py — Detect citation-cartel-like patterns in OAS.

For each OAS source, computes the fraction of its works that are year-adjusted
top-centile by intra-corpus citation count (HCW = Highly Cited Work). Flags
sources where this propensity is anomalously high relative to spectral influence v.

Steps:
  1. Year-specific HCW thresholds (99th pctile of n_intra per publication year).
  2. Source-level n_works, n_hcw, propensity.
  3. LOWESS residuals on propensity ~ log10(v) to identify anomalies.
  4. Two-panel scatter figure: v vs n_hcw count (left), v vs propensity (right).
  5. Console: threshold table, residual table, flagged sources.
  6. TF-IDF title terms for flagged low-v (cartel candidate) sources.
  7. Mutual citation table among cartel candidates.

Outputs:
  plots/cartel_v_vs_hcw.png   — rank figure
  Console: all diagnostics
"""

import sys
import string
from pathlib import Path
from collections import Counter

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config

_paths = load_config()

# ── Constants ──────────────────────────────────────────────────────────────────

RK_TABLE = 'rk_20242024_EBAX_tauU20_tauS20_vartau_rho0_m0110_chi50_alpha100'

WK_PATH  = f"'{_paths.parquet}/corpus_works.parquet'"
REF_PATH = f"'{_paths.parquet}/corpus_references.parquet'"
SM_PATH  = f"'{_paths.parquet}/source_master.parquet'"
AU_PATH  = f"'{_paths.parquet}/corpus_authorships.parquet'"

MIN_WORKS       = 30    # minimum works per source for reliable propensity
YEAR_LO         = 2015
YEAR_HI         = 2024
HCW_PCT         = 99    # top-centile definition
RESID_SD_THRESH = 2.0   # SD threshold for anomaly flag
LOW_V_THRESH    = 1.5   # v cutoff for cartel candidate sub-filter
TOP_N_TERMS     = 15    # TF-IDF terms per flagged source
TOP_N_ANNOTATE  = 20    # max labels on propensity panel

FIELD_COLOURS = {
    'E': '#e41a1c',
    'B': '#377eb8',
    'A': '#ff7f0e',
    'X': '#888888',
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
    'firm', 'firms', 'market', 'markets', 'does', 'are', 'use',
}


# ── 1. HCW computation ────────────────────────────────────────────────────────

def compute_hcw() -> pd.DataFrame:
    """
    Returns work-level DataFrame:
        work_idx, source_idx, publication_year, title, n_intra, threshold, is_hcw
    Restricted to OAS works (joined with source_master), YEAR_LO–YEAR_HI.
    n_intra = count of citations received from any other corpus work.
    is_hcw  = n_intra >= year-specific 99th percentile threshold.
    """
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

    # Year-specific threshold (over all OAS works in that year, including 0-cited)
    thresh = (
        df.groupby('publication_year')['n_intra']
        .quantile(HCW_PCT / 100.0)
        .rename('threshold')
    )
    df = df.merge(thresh, on='publication_year', how='left')
    df['is_hcw'] = df['n_intra'] >= df['threshold']
    return df


# ── 2. Source-level aggregation ───────────────────────────────────────────────

def source_stats(df_works: pd.DataFrame, df_src_v: pd.DataFrame,
                 df_sm: pd.DataFrame) -> pd.DataFrame:
    """
    Per-source HCW counts for all ranked sources, joined with v and source_master.
    No MIN_WORKS filter applied here — callers filter for propensity analysis.
    """
    agg = (
        df_works.groupby('source_idx')
        .agg(n_works=('work_idx', 'count'), n_hcw=('is_hcw', 'sum'))
        .reset_index()
    )
    agg['n_hcw']      = agg['n_hcw'].astype(int)
    agg['propensity'] = agg['n_hcw'] / agg['n_works']
    agg = agg.merge(df_src_v, on='source_idx', how='inner')
    agg = agg.merge(
        df_sm[['source_idx', 'source_name', 'field_eb', 'subfield_name']],
        on='source_idx', how='left',
    )
    return agg.reset_index(drop=True)


def add_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """LOWESS residuals of propensity ~ log10(v). Adds residual and residual_sd."""
    log_v  = np.log10(np.clip(df['v'].values, 1e-4, None))
    fitted = lowess(df['propensity'].values, log_v, frac=0.30, return_sorted=False)
    df = df.copy()
    df['propensity_fitted'] = fitted
    df['residual']          = df['propensity'] - fitted
    sd                      = df['residual'].std()
    df['residual_sd']       = df['residual'] / sd if sd > 0 else 0.0
    return df


def institution_hcw_stats(df_works: pd.DataFrame,
                          df_inst_v: pd.DataFrame) -> pd.DataFrame:
    """
    Per-institution n_works (OAS works co-authored in YEAR_LO–YEAR_HI) and n_hcw.
    Joined with institution v from baseline ranking. Sorted by v descending.
    """
    all_works = df_works[['work_idx']].copy()
    hcw_works = df_works.loc[df_works['is_hcw'], ['work_idx']].copy()

    con = duckdb.connect(':memory:')
    con.register('_all_works', all_works)
    con.register('_hcw_works', hcw_works)
    df = con.execute(f"""
        SELECT
            a.institution_idx,
            a.institution_name,
            COUNT(DISTINCT a.work_idx)  AS n_works,
            COUNT(DISTINCT h.work_idx)  AS n_hcw
        FROM {AU_PATH} a
        JOIN _all_works aw ON aw.work_idx = a.work_idx
        LEFT JOIN _hcw_works h  ON h.work_idx  = a.work_idx
        WHERE a.institution_idx IS NOT NULL
        GROUP BY a.institution_idx, a.institution_name
    """).df()
    con.close()

    df['n_hcw']      = df['n_hcw'].astype(int)
    df['propensity'] = df['n_hcw'] / df['n_works']
    df = df[df['n_works'] >= MIN_WORKS].copy()
    df = df.merge(df_inst_v, on='institution_idx', how='inner')
    df = df.sort_values('v', ascending=False).reset_index(drop=True)
    df['rank_v'] = np.arange(1, len(df) + 1)
    return df


# ── 2b. HCW referer v-distribution ───────────────────────────────────────────

def hcw_referer_stats(df_works: pd.DataFrame,
                      df_src_v: pd.DataFrame,
                      df_sm: pd.DataFrame) -> pd.DataFrame:
    """
    For every HCW, find all intra-corpus works that reference it and compute
    percentiles of the v-distribution of their host sources.

    Columns returned:
        work_idx, source_idx, source_name, publication_year, title,
        n_intra, v_host,
        n_referencers, v_median, v_p80, v_p90, v_p95, v_p99,
        ratio_p95  (v_p95 / v_host — low values flag suspicious HCW)
    Sorted by ratio_p95 ascending (most suspicious first).
    """
    hcw = df_works.loc[df_works['is_hcw'],
                       ['work_idx', 'source_idx', 'publication_year',
                        'title', 'n_intra']].copy()

    con = duckdb.connect(':memory:')
    con.register('_hcw',   hcw)
    con.register('_src_v', df_src_v)

    df = con.execute(f"""
        WITH referencers AS (
            SELECT
                r.cited_idx                                         AS work_idx,
                sv.v                                                AS referer_v
            FROM {REF_PATH} r
            JOIN _hcw    h  ON h.work_idx   = r.cited_idx
            JOIN {WK_PATH} cw ON cw.work_idx = r.citer_idx
            JOIN _src_v  sv ON sv.source_idx = cw.source_idx
        )
        SELECT
            work_idx,
            COUNT(*)                                                    AS n_referencers,
            MEDIAN(referer_v)                                           AS v_median,
            PERCENTILE_CONT(0.80) WITHIN GROUP (ORDER BY referer_v)    AS v_p80,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY referer_v)    AS v_p90,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY referer_v)    AS v_p95,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY referer_v)    AS v_p99
        FROM referencers
        GROUP BY work_idx
    """).df()
    con.close()

    df = df.merge(hcw, on='work_idx', how='left')
    df = df.merge(df_src_v.rename(columns={'v': 'v_host'}), on='source_idx', how='left')
    df = df.merge(df_sm[['source_idx', 'source_name']], on='source_idx', how='left')

    df['ratio_p95'] = df['v_p95'] / df['v_host'].clip(lower=1e-4)
    df = df.sort_values('ratio_p95').reset_index(drop=True)

    col_order = ['work_idx', 'source_idx', 'source_name', 'publication_year',
                 'title', 'n_intra', 'v_host',
                 'n_referencers', 'v_median', 'v_p80', 'v_p90', 'v_p95', 'v_p99',
                 'ratio_p95']
    return df[col_order]


# ── 3. Figures ────────────────────────────────────────────────────────────────

_HCW_COLOUR     = '#e07b00'              # orange for HCW count dots
_CARTEL_COLOURS = ['#c0392b', '#2980b9', '#27ae60', '#8e44ad']  # per cluster
_V_LO, _V_HI   = -2.5, 1.4
_V_TICKS        = [-2, -1, 0, 1]
_GRID_COLOUR    = '#dddddd'


def _draw_rank_panel(ax, df: pd.DataFrame, title: str,
                     flagged_ids: list | None = None,
                     label_v: bool = True,
                     label_hcw: bool = True,
                     highlight_map: dict | None = None) -> None:
    """
    Single panel: rank on x, log10(v) black curve on primary y-axis (left),
    HCW count orange dots on twin y-axis (right, log scale 1–500).
    label_v      — show ticks/label on left y-axis.
    label_hcw    — show ticks/label on right y-axis.
    highlight_map — dict {id → cluster_int}; cluster-coloured dots overlaid on HCW scatter.
    """
    n      = len(df)
    ranks  = np.arange(1, n + 1)
    log_v  = np.log10(np.clip(df['v'].values, 1e-4, None))
    id_col = 'source_idx' if 'source_idx' in df.columns else 'institution_idx'

    ax.set_facecolor('white')
    ax.grid(True, color=_GRID_COLOUR, linewidth=0.7, linestyle='-', zorder=0)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.set_facecolor('none')
    ax2.patch.set_visible(False)
    ax2.grid(False)

    # ── v rank curve ─────────────────────────────────────────────────────────
    ax.plot(ranks, log_v, color='#444444', linewidth=1.2, zorder=3)
    ax.axhline(0.0, color='#bbbbbb', linewidth=0.8, linestyle='--', zorder=2)
    ax.text(n * 0.98, 0.02, '$v=1$', ha='right', va='bottom',
            fontsize=7, color='#aaaaaa')
    ax.set_ylim(_V_LO, _V_HI)
    ax.set_yticks(_V_TICKS)
    ax.set_xlim(1, n)
    ax.set_xlabel('Rank', labelpad=4)
    ax.set_title(title, fontsize=9, pad=6)

    if label_v:
        ax.set_ylabel(r'$\log_{10}(v)$', labelpad=4)
    else:
        ax.set_ylabel('')
        ax.tick_params(axis='y', left=False, labelleft=False)
        ax.spines['left'].set_visible(False)

    # ── HCW dots ──────────────────────────────────────────────────────────────
    hcw      = df['n_hcw'].values
    hcw_plot = np.clip(hcw, 1, 500)
    mask     = hcw > 0
    ax2.scatter(ranks[mask], hcw_plot[mask],
                color=_HCW_COLOUR, s=10, alpha=0.55, linewidths=0, zorder=4)

    if flagged_ids:
        flag_mask = df[id_col].isin(set(flagged_ids)).values & mask
        ax2.scatter(ranks[flag_mask], hcw_plot[flag_mask],
                    color='darkred', s=30, alpha=0.85, linewidths=0, zorder=5)

    if highlight_map:
        ids = df[id_col].values
        for cid in sorted(set(highlight_map.values())):
            colour    = _CARTEL_COLOURS[cid % len(_CARTEL_COLOURS)]
            in_cl     = np.array([highlight_map.get(int(x), -1) == cid for x in ids])
            plot_mask = in_cl & mask
            if plot_mask.any():
                ax2.scatter(ranks[plot_mask], hcw_plot[plot_mask],
                            color=colour, s=40, alpha=0.9, linewidths=0, zorder=6)

    ax2.set_yscale('log')
    ax2.set_ylim(1, 500)
    ax2.set_xlim(1, n)
    ax2.patch.set_visible(False)

    if label_hcw:
        ax2.set_ylabel('HCW count', labelpad=6, color=_HCW_COLOUR)
        ax2.tick_params(axis='y', labelcolor=_HCW_COLOUR)
        ax2.spines['right'].set_edgecolor(_HCW_COLOUR)
    else:
        ax2.set_ylabel('')
        ax2.tick_params(axis='y', right=False, labelright=False)
        ax2.spines['right'].set_visible(False)


def plot_lowess(df: pd.DataFrame, out_path: Path) -> None:
    """
    Scatter of propensity vs log10(v) with LOWESS fit overlaid.
    Flagged sources (residual_sd >= RESID_SD_THRESH) shown in orange;
    cartel candidates (flagged AND v < LOW_V_THRESH) in red with name labels.
    """
    log_v   = np.log10(np.clip(df['v'].values, 1e-4, None))
    prop    = df['propensity'].values
    flagged = df['residual_sd'] >= RESID_SD_THRESH
    cartel  = flagged & (df['v'] < LOW_V_THRESH)

    # LOWESS curve sampled on a fine grid for smooth line
    xy_sorted = lowess(prop, log_v, frac=0.30, return_sorted=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(True, color=_GRID_COLOUR, linewidth=0.7, linestyle='-', zorder=0)
    ax.set_axisbelow(True)

    # Background sources
    ax.scatter(log_v[~flagged], prop[~flagged],
               color='#999999', s=8, alpha=0.4, linewidths=0, zorder=2)
    # Flagged (above-fit, high-v)
    ax.scatter(log_v[flagged & ~cartel], prop[flagged & ~cartel],
               color=_HCW_COLOUR, s=20, alpha=0.7, linewidths=0, zorder=3)
    # Cartel candidates
    ax.scatter(log_v[cartel], prop[cartel],
               color=_CARTEL_COLOURS[0], s=40, alpha=0.9, linewidths=0, zorder=4)

    # Labels for cartel candidates
    for _, r in df[cartel].iterrows():
        lv = np.log10(max(r['v'], 1e-4))
        ax.annotate(str(r['source_name']), xy=(lv, r['propensity']),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=6, color=_CARTEL_COLOURS[0])

    # LOWESS fit line
    ax.plot(xy_sorted[:, 0], xy_sorted[:, 1],
            color='#333333', linewidth=1.5, zorder=5)

    ax.set_xlabel(r'$\log_{10}(v)$', labelpad=4)
    ax.set_ylabel('Propensity  (n_hcw / n_works)', labelpad=4)
    ax.axvline(np.log10(LOW_V_THRESH), color='#bbbbbb', linewidth=0.8,
               linestyle='--', zorder=1)

    sup = fig.suptitle(
        f'Source HCW propensity vs influence  '
        f'(LOWESS frac={0.30}, flagged >{RESID_SD_THRESH} SD)',
        fontsize=9, y=0.99,
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12)

    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved {out_path}')

    sup.set_visible(False)
    latex_path = out_path.with_name(out_path.stem + '_latex.pdf')
    fig.savefig(latex_path, bbox_inches='tight')
    print(f'Saved {latex_path}')

    plt.close(fig)


def plot_rank_with_hcw(df_src: pd.DataFrame, df_inst: pd.DataFrame,
                       flagged_src_ids: list, out_path: Path,
                       cartel_inst_map: dict | None = None) -> None:
    """
    1×2 figure: sources (left) and institutions (right).
    Each panel: rank vs log10(v) curve + HCW count dots.
    Outer y-axes carry labels; inner axes (between facets) are hidden.
    Saves out_path (with suptitle) and out_path stem + _latex.pdf (without).
    """
    df_s = df_src.sort_values('v', ascending=False).reset_index(drop=True)
    df_s['rank_v'] = np.arange(1, len(df_s) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    _draw_rank_panel(axes[0], df_s,    'Sources',      flagged_src_ids,
                     label_v=True,  label_hcw=False)
    _draw_rank_panel(axes[1], df_inst, 'Institutions', flagged_ids=None,
                     label_v=False, label_hcw=True,
                     highlight_map=cartel_inst_map or {})

    fig.subplots_adjust(left=0.07, right=0.93, top=0.90, bottom=0.10, wspace=0.35)

    sup = fig.suptitle('HCW cartel analysis — rank vs log(v) and HCW count',
                       fontsize=9, y=0.99)
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved {out_path}')

    sup.set_visible(False)
    latex_path = out_path.with_name(out_path.stem + '_latex.pdf')
    fig.savefig(latex_path, bbox_inches='tight')
    print(f'Saved {latex_path}')

    plt.close(fig)


# ── 4. TF-IDF title terms ─────────────────────────────────────────────────────

def _tokenise(title: str) -> list[str]:
    t = title.lower()
    t = t.translate(str.maketrans('', '', string.punctuation + '–—'))
    return [w for w in t.split() if w not in _STOPWORDS and len(w) > 2]


def top_tfidf_terms(fg_titles: list[str],
                    bg_titles: list[str]) -> list[tuple[str, float]]:
    """TF-IDF: foreground = HCW titles in flagged source; background = all HCW titles."""
    fg_df = Counter(w for t in fg_titles for w in set(_tokenise(t)))
    bg_df = Counter(w for t in bg_titles for w in set(_tokenise(t)))
    fg_n  = max(len(fg_titles), 1)
    bg_n  = max(len(bg_titles), 1)
    scores = {
        w: (cnt / fg_n) * (np.log((bg_n + 1) / (bg_df.get(w, 0) + 1)) + 1)
        for w, cnt in fg_df.items()
    }
    return sorted(scores.items(), key=lambda x: -x[1])[:TOP_N_TERMS]


# ── 5. Mutual citation table ───────────────────────────────────────────────────

def mutual_citation_table(df_works: pd.DataFrame, flagged_ids: list,
                          src_names: dict, out_path: Path | None = None) -> None:
    """
    For each ordered pair of cartel-candidate sources (cited A, citer B),
    count citations from works in source B to HCW works in source A.
    Reports mean references per HCW in A from B (intensity = n_cites / n_hcw(A)).
    If out_path given, saves a wide pivot CSV (rows=cited, cols=citer, values=intensity).
    """
    if len(flagged_ids) < 2:
        print('\nFewer than 2 cartel candidates — skipping mutual citation table.')
        return

    hcw_src = (
        df_works[df_works['is_hcw'] & df_works['source_idx'].isin(flagged_ids)]
        [['work_idx', 'source_idx']].copy()
    )
    citer_src = (
        df_works[df_works['source_idx'].isin(flagged_ids)]
        [['work_idx', 'source_idx']].copy()
    )

    con = duckdb.connect(':memory:')
    con.register('_hcw_src',   hcw_src)
    con.register('_citer_src', citer_src)

    mc = con.execute(f"""
        SELECT
            h.source_idx  AS cited_source,
            c.source_idx  AS citer_source,
            COUNT(*)      AS n_citations
        FROM {REF_PATH} r
        JOIN _hcw_src   h ON h.work_idx = r.cited_idx
        JOIN _citer_src c ON c.work_idx = r.citer_idx
        WHERE h.source_idx != c.source_idx
        GROUP BY h.source_idx, c.source_idx
        ORDER BY n_citations DESC
    """).df()
    con.close()

    if mc.empty:
        print('\nNo cross-citations found among cartel candidates.')
        return

    n_hcw_by_src = hcw_src.groupby('source_idx')['work_idx'].count().to_dict()

    print(f'\n── Mutual reference table  (cartel candidates, intensity = refs per HCW) ──')
    print(f'{"Cited source":<42} {"Citer source":<42} {"n_refs":>8} {"intensity":>9}')
    print('-' * 106)
    for _, row in mc.head(50).iterrows():
        cs  = int(row['cited_source'])
        cr  = int(row['citer_source'])
        n   = int(row['n_citations'])
        tot = max(n_hcw_by_src.get(cs, 1), 1)
        print(
            f'{src_names.get(cs, str(cs)):<42} '
            f'{src_names.get(cr, str(cr)):<42} '
            f'{n:>8,}  {n/tot:>9.3f}'
        )

    if out_path is not None:
        mc['cited_name']   = mc['cited_source'].map(lambda x: src_names.get(int(x), str(x)))
        mc['citer_name']   = mc['citer_source'].map(lambda x: src_names.get(int(x), str(x)))
        mc['intensity']    = mc.apply(
            lambda r: r['n_citations'] / max(n_hcw_by_src.get(int(r['cited_source']), 1), 1),
            axis=1,
        )
        pivot = (
            mc.pivot(index='cited_name', columns='citer_name', values='intensity')
            .fillna(0)
            .round(4)
        )
        pivot.index.name   = 'cited_source'
        pivot.columns.name = 'citer_source'
        pivot.to_csv(out_path)
        print(f'Saved {out_path}')


# ── 6. Console report ─────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame, df_works: pd.DataFrame) -> list:
    """Prints full console report. Returns list of flagged low-v source_idx."""

    all_hcw_titles = df_works[df_works['is_hcw']]['title'].dropna().tolist()

    # Threshold summary
    thresh_dict = (
        df_works.groupby('publication_year')['threshold']
        .first()
        .astype(int)
        .to_dict()
    )
    print('\nYear-adjusted HCW thresholds (99th pctile of n_intra, all OAS works):')
    for yr in [2000, 2005, 2010, 2015, 2020, 2023]:
        t = thresh_dict.get(yr, '?')
        print(f'  {yr}: n_intra ≥ {t}')

    n_hcw   = int(df_works['is_hcw'].sum())
    n_total = len(df_works)
    print(f'\nTotal HCW: {n_hcw:,}  /  {n_total:,} OAS works {YEAR_LO}–{YEAR_HI}'
          f'  ({100 * n_hcw / n_total:.1f}%)')

    # Residual table — top 40 by |residual_sd|
    print('\n── Source propensity table  (top 40 by |residual_sd|) ─────────────────────')
    print(f'{"Source":<44} {"v":>6} {"n_wk":>6} {"n_hcw":>6} '
          f'{"prop":>7} {"fit":>7} {"SD":>7}  fld  {"Subfield":<36}')
    print('-' * 132)
    top40 = df.reindex(df['residual_sd'].abs().sort_values(ascending=False).index).head(40)
    for _, r in top40.iterrows():
        sf = str(r.get('subfield_name', '') or '')[:36]
        print(
            f"{str(r['source_name']):<44} {r['v']:>6.2f} {int(r['n_works']):>6,} "
            f"{int(r['n_hcw']):>6,} {r['propensity']:>7.4f} "
            f"{r['propensity_fitted']:>7.4f} {r['residual_sd']:>7.1f}  {r['field_eb']}  {sf}"
        )

    # All flagged sources
    flagged = df[df['residual_sd'] >= RESID_SD_THRESH].sort_values(
        'residual_sd', ascending=False
    )
    low_v   = flagged[flagged['v'] < LOW_V_THRESH]

    print(f'\n── Flagged sources  (residual_sd ≥ {RESID_SD_THRESH},  {len(flagged)} total) ──')
    print(f'   Of which v < {LOW_V_THRESH} (cartel candidates): {len(low_v)}')
    for _, r in flagged.iterrows():
        tag = '  *** CARTEL CANDIDATE' if r['v'] < LOW_V_THRESH else ''
        sf  = str(r.get('subfield_name', '') or '')
        print(
            f"  {str(r['source_name']):<44}  v={r['v']:>5.2f}  "
            f"n_hcw={int(r['n_hcw']):>3d}  prop={r['propensity']:.4f}  "
            f"SD={r['residual_sd']:>+5.1f}  {r['field_eb']}  {sf}{tag}"
        )

    # TF-IDF for each cartel candidate
    if low_v.empty:
        print('\nNo cartel candidates identified at this threshold.')
        return []

    print(f'\n── TF-IDF title terms  (cartel candidates vs all-HCW background) ─────────')
    for _, r in low_v.iterrows():
        src_id = r['source_idx']
        fg = (
            df_works[df_works['is_hcw'] & (df_works['source_idx'] == src_id)]
            ['title'].dropna().tolist()
        )
        if not fg:
            continue
        terms = top_tfidf_terms(fg, all_hcw_titles)
        print(
            f"\n  {r['source_name']}"
            f"  (v={r['v']:.2f}, n_hcw={int(r['n_hcw'])}, "
            f"prop={r['propensity']:.4f}, SD={r['residual_sd']:+.1f})"
        )
        print(f"  {', '.join(f'{t}({s:.3f})' for t, s in terms)}")

    return low_v['source_idx'].tolist()


# ── Institution map ───────────────────────────────────────────────────────────

def institution_hcw_map(df_works: pd.DataFrame, flagged_ids: list,
                        df_inst_v: pd.DataFrame) -> None:
    """
    For HCW works in cartel-candidate sources, show which institutions authored them.
    Aggregates: n_hcw_works (distinct HCW co-authored), n_sources (distinct cartel
    sources spanned). Joined with institution v from the baseline ranking.
    """
    hcw_flagged = (
        df_works[df_works['is_hcw'] & df_works['source_idx'].isin(flagged_ids)]
        [['work_idx', 'source_idx']].copy()
    )
    if hcw_flagged.empty:
        print('\nNo HCW works in cartel candidates — skipping institution map.')
        return

    con = duckdb.connect(':memory:')
    con.register('_hcw_flagged', hcw_flagged)

    df_inst = con.execute(f"""
        SELECT
            a.institution_idx,
            a.institution_name,
            COUNT(DISTINCT a.work_idx)  AS n_hcw_works,
            COUNT(DISTINCT h.source_idx) AS n_sources
        FROM {AU_PATH} a
        JOIN _hcw_flagged h ON h.work_idx = a.work_idx
        WHERE a.institution_idx IS NOT NULL
        GROUP BY a.institution_idx, a.institution_name
        ORDER BY n_hcw_works DESC
    """).df()
    con.close()

    if df_inst.empty:
        print('\nNo institution data found for cartel HCW.')
        return

    df_inst = df_inst.merge(df_inst_v, on='institution_idx', how='left')
    n_total = hcw_flagged['work_idx'].nunique()

    print(f'\n── Institution map  (authors of cartel-candidate HCW,  {n_total:,} HCW total,  top 50) ──')
    print(f'{"Institution":<52} {"v_inst":>7} {"n_hcw":>6} {"n_src":>6} {"pct":>6}')
    print('-' * 83)
    for _, r in df_inst.head(50).iterrows():
        v_str = f"{r['v']:.2f}" if pd.notna(r.get('v')) else '    —'
        pct   = 100 * r['n_hcw_works'] / n_total
        print(
            f"{str(r['institution_name']):<52} {v_str:>7} "
            f"{int(r['n_hcw_works']):>6,} {int(r['n_sources']):>6}  {pct:>5.1f}%"
        )


# ── Cartel cluster → institution map ─────────────────────────────────────────

def cartel_institution_map(df_works: pd.DataFrame,
                           flagged_ids: list,
                           src_names: dict) -> dict:
    """
    Run union-find connected components on the mutual citation graph among
    cartel-candidate sources to assign each source a cluster_id (0-based).
    Then map institutions that authored HCW in those sources to their cluster_id.

    Returns dict: institution_idx (int) → cluster_id (int).
    Prints cluster membership to console.
    """
    if not flagged_ids:
        return {}

    flagged_set = set(int(x) for x in flagged_ids)
    hcw_src     = df_works[df_works['is_hcw'] & df_works['source_idx'].isin(flagged_set)][['work_idx', 'source_idx']].copy()
    citer_src   = df_works[df_works['source_idx'].isin(flagged_set)][['work_idx', 'source_idx']].copy()

    con = duckdb.connect(':memory:')
    con.register('_hcw_src',   hcw_src)
    con.register('_citer_src', citer_src)
    mc = con.execute(f"""
        SELECT h.source_idx AS cited, c.source_idx AS citer
        FROM {REF_PATH} r
        JOIN _hcw_src   h ON h.work_idx = r.cited_idx
        JOIN _citer_src c ON c.work_idx = r.citer_idx
        WHERE h.source_idx != c.source_idx
        GROUP BY h.source_idx, c.source_idx
        HAVING COUNT(*) > 0
    """).df()
    con.close()

    # Union-find
    parent = {s: s for s in flagged_set}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for _, row in mc.iterrows():
        cs, cr = int(row['cited']), int(row['citer'])
        if cs in parent and cr in parent:
            union(cs, cr)

    roots = {}
    src_cluster: dict[int, int] = {}
    for s in sorted(flagged_set):
        r = find(s)
        if r not in roots:
            roots[r] = len(roots)
        src_cluster[s] = roots[r]

    print('\n── Cartel clusters (connected components on mutual citation graph) ──────')
    for cid in sorted(set(src_cluster.values())):
        members = [src_names.get(s, str(s)) for s, c in src_cluster.items() if c == cid]
        colour  = _CARTEL_COLOURS[cid % len(_CARTEL_COLOURS)]
        print(f'  Cluster {cid}  ({colour}):  {", ".join(members)}')

    # Map institutions to clusters via authorship of HCW in flagged sources
    con = duckdb.connect(':memory:')
    con.register('_hcw_src', hcw_src)
    df_ia = con.execute(f"""
        SELECT a.institution_idx, h.source_idx
        FROM {AU_PATH} a
        JOIN _hcw_src h ON h.work_idx = a.work_idx
        WHERE a.institution_idx IS NOT NULL
        GROUP BY a.institution_idx, h.source_idx
    """).df()
    con.close()

    inst_cluster: dict[int, int] = {}
    for _, row in df_ia.iterrows():
        iid = int(row['institution_idx'])
        cid = src_cluster.get(int(row['source_idx']))
        if cid is not None and (iid not in inst_cluster or cid < inst_cluster[iid]):
            inst_cluster[iid] = cid

    n_per = {cid: sum(1 for c in inst_cluster.values() if c == cid)
             for cid in sorted(set(src_cluster.values()))}
    print(f'  Institutions implicated: ' +
          ', '.join(f'cluster {c}: {n}' for c, n in n_per.items()))

    return inst_cluster


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = load_config()

    print('Loading source and institution v scores from rankings.duckdb…')
    with duckdb.connect(str(p.working / 'rankings.duckdb'), read_only=True) as db:
        df_src_v = db.execute(
            f"SELECT unit_idx AS source_idx, v FROM {RK_TABLE} WHERE unit_type='S'"
        ).df()
        df_inst_v = db.execute(
            f"SELECT unit_idx AS institution_idx, v FROM {RK_TABLE} WHERE unit_type='U'"
        ).df()

    print('Loading source master…')
    _con = duckdb.connect(':memory:')
    df_sm = _con.execute(
        f"SELECT source_idx, source_name, field_eb, subfield_name FROM {SM_PATH}"
    ).df()
    _con.close()

    print('Computing HCW flags  (joins 17M-row corpus_references parquet)…')
    df_works = compute_hcw()

    print('Aggregating by source and fitting LOWESS residuals…')
    df_src = source_stats(df_works, df_src_v, df_sm)
    df_src = add_residuals(df_src)

    print('Computing institution HCW stats…')
    df_inst = institution_hcw_stats(df_works, df_inst_v)

    flagged_ids = print_report(df_src, df_works)

    src_names       = df_sm.set_index('source_idx')['source_name'].to_dict()
    cartel_inst_map = {}
    if flagged_ids:
        mutual_citation_table(df_works, flagged_ids, src_names,
                              out_path=p.plots / 'cartel_mrt.csv')
        institution_hcw_map(df_works, flagged_ids, df_inst_v)
        cartel_inst_map = cartel_institution_map(df_works, flagged_ids, src_names)

    # Save combined propensity + flagged CSV
    csv_cols = ['source_name', 'subfield_name', 'field_eb', 'v',
                'n_works', 'n_hcw', 'propensity',
                'propensity_fitted', 'residual_sd']
    df_csv = df_src[csv_cols].copy()
    df_csv['flagged']          = df_src['residual_sd'] >= RESID_SD_THRESH
    df_csv['cartel_candidate'] = df_csv['flagged'] & (df_src['v'] < LOW_V_THRESH)
    df_csv = df_csv.sort_values('residual_sd', ascending=False).reset_index(drop=True)
    csv_path = p.plots / 'cartel_source_propensity.csv'
    df_csv.to_csv(csv_path, index=False)
    print(f'Saved {csv_path}')

    print('Computing HCW referer v-distributions…')
    df_hcw_ref = hcw_referer_stats(df_works, df_src_v, df_sm)
    ref_csv = p.plots / 'cartel_hcw_referer_stats.csv'
    df_hcw_ref.to_csv(ref_csv, index=False, float_format='%.4f')
    print(f'Saved {ref_csv}  ({len(df_hcw_ref):,} HCW)')

    print('Saving figures…')
    plot_lowess(df_src, p.plots / 'cartel_lowess.pdf')
    plot_rank_with_hcw(df_src, df_inst, flagged_ids or [],
                       p.plots / 'cartel_v_vs_hcw.pdf', cartel_inst_map)


if __name__ == '__main__':
    main()
    print('\nFINISHED!')
