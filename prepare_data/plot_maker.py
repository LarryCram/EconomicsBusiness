"""
plot_maker.py — Plots for the paper.

Plot 1: Institution and source works-count elbow.
    x = minimum works_count threshold (cutoff, in works/year)
    y = % of total works retained at that threshold
    Blue curve: institution retention (work retained if ≥1 author at institution
                with >= τ works/year)
    Green curve: source retention (work retained if its source has >= τ works/year)
    Upper x-axis row 1: sources retained at each τ tick
    Upper x-axis row 2: institutions retained at each τ tick
    Computed over the baseline window (t_x=5, 2020–2024).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_params

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

paths  = load_config()
params = load_params()

PARQUET = paths.parquet
PLOTS   = paths.plots
PLOTS.mkdir(exist_ok=True)

# Baseline window: t_x=5 (2020–2024)
_BASELINE_TX = 5
_tw = params['time_windows'][_BASELINE_TX]
_YEAR_MIN = min(_tw['census'][0], _tw['target'][0])
_YEAR_MAX = max(_tw['census'][1], _tw['target'][1])
N_YEARS   = _YEAR_MAX - _YEAR_MIN + 1   # 5


def fetch_elbow_data(db) -> pd.DataFrame:
    """
    Returns one row per distinct works_count threshold with:
        works_count            -- institution size (distinct works in the baseline window)
        institutions_count     -- institutions at exactly this size
        cum_institutions_above -- institutions with works_count >= this value
        pct_retained           -- % of works that have at least one author at an institution
                                  with works_count >= this value (pct_works metric)
    """
    return db.sql(f"""
        WITH institution_works AS (
            SELECT a.institution_idx,
                   COUNT(DISTINCT a.work_idx) AS works_count
            FROM '{PARQUET}/corpus_authorships.parquet' a
            JOIN '{PARQUET}/corpus_works.parquet' w USING (work_idx)
            WHERE a.institution_idx IS NOT NULL
              AND w.publication_year BETWEEN {_YEAR_MIN} AND {_YEAR_MAX}
            GROUP BY a.institution_idx
        ),
        -- For each work, the largest institution (by works_count) among its authors.
        -- A work is retained at threshold W iff max_inst_works >= W.
        work_max_inst AS (
            SELECT a.work_idx,
                   MAX(iw.works_count) AS max_inst_works
            FROM '{PARQUET}/corpus_authorships.parquet' a
            JOIN '{PARQUET}/corpus_works.parquet' w USING (work_idx)
            JOIN institution_works iw ON a.institution_idx = iw.institution_idx
            WHERE w.publication_year BETWEEN {_YEAR_MIN} AND {_YEAR_MAX}
            GROUP BY a.work_idx
        ),
        inst_freq AS (
            SELECT works_count, COUNT(*) AS institutions_count
            FROM institution_works
            GROUP BY works_count
        ),
        work_freq AS (
            SELECT max_inst_works AS works_count, COUNT(*) AS works_n
            FROM work_max_inst
            GROUP BY max_inst_works
        ),
        all_thresholds AS (
            SELECT works_count FROM inst_freq
            UNION
            SELECT works_count FROM work_freq
        ),
        combined AS (
            SELECT t.works_count,
                   COALESCE(i.institutions_count, 0) AS institutions_count,
                   COALESCE(wf.works_n, 0)           AS works_n
            FROM all_thresholds t
            LEFT JOIN inst_freq i USING (works_count)
            LEFT JOIN work_freq wf USING (works_count)
        ),
        totals AS (
            SELECT SUM(institutions_count) AS total_inst,
                   SUM(works_n)            AS total_works
            FROM combined
        ),
        cumul AS (
            SELECT works_count, institutions_count, works_n,
                   SUM(institutions_count) OVER (ORDER BY works_count
                       ROWS UNBOUNDED PRECEDING) AS cum_inst_to,
                   SUM(works_n) OVER (ORDER BY works_count
                       ROWS UNBOUNDED PRECEDING) AS cum_works_to,
                   total_inst, total_works
            FROM combined CROSS JOIN totals
        )
        SELECT works_count,
               institutions_count,
               (total_inst  - COALESCE(LAG(cum_inst_to)  OVER (ORDER BY works_count), 0)) AS cum_institutions_above,
               (total_works - COALESCE(LAG(cum_works_to) OVER (ORDER BY works_count), 0))
                   * 100.0 / total_works                                                   AS pct_retained
        FROM cumul
        ORDER BY works_count
    """).df()

def fetch_source_elbow_data(db) -> pd.DataFrame:
    """
    Returns one row per distinct source works_count with:
        works_count          -- source size (distinct works in the baseline window)
        sources_count        -- sources at exactly this size
        cum_sources_above    -- sources with works_count >= this value
        pct_retained         -- % of total corpus works from sources with >= this works_count
    """
    return db.sql(f"""
        WITH source_works AS (
            SELECT source_idx,
                   COUNT(DISTINCT work_idx) AS works_count
            FROM '{PARQUET}/corpus_works.parquet'
            WHERE publication_year BETWEEN {_YEAR_MIN} AND {_YEAR_MAX}
            GROUP BY source_idx
        ),
        src_freq AS (
            SELECT works_count,
                   COUNT(*)          AS sources_count,
                   SUM(works_count)  AS works_at_level
            FROM source_works
            GROUP BY works_count
        ),
        totals AS (
            SELECT SUM(sources_count)  AS total_sources,
                   SUM(works_at_level) AS total_works
            FROM src_freq
        ),
        cumul AS (
            SELECT works_count, sources_count, works_at_level,
                   SUM(sources_count) OVER (ORDER BY works_count
                       ROWS UNBOUNDED PRECEDING) AS cum_src_to,
                   SUM(works_at_level) OVER (ORDER BY works_count
                       ROWS UNBOUNDED PRECEDING) AS cum_works_to,
                   total_sources, total_works
            FROM src_freq CROSS JOIN totals
        )
        SELECT works_count,
               sources_count,
               (total_sources - COALESCE(LAG(cum_src_to)    OVER (ORDER BY works_count), 0))
                   AS cum_sources_above,
               (total_works  - COALESCE(LAG(cum_works_to)   OVER (ORDER BY works_count), 0))
                   * 100.0 / total_works AS pct_retained
        FROM cumul
        ORDER BY works_count
    """).df()


def _count_at_tick(df: pd.DataFrame, count_col: str, t: float) -> str:
    """Return formatted count from df for the lowest works_per_year >= t."""
    candidates = df[df['works_per_year'] >= t]
    if candidates.empty:
        return '0'
    return f'{int(candidates.iloc[0][count_col]):,}'


def plot1(df_inst: pd.DataFrame, df_src: pd.DataFrame) -> None:
    df_inst = df_inst.copy()
    df_inst['works_per_year'] = df_inst['works_count'] / N_YEARS
    df_src = df_src.copy()
    df_src['works_per_year'] = df_src['works_count'] / N_YEARS

    x_max = 40
    plot_inst = df_inst[df_inst['works_per_year'] <= x_max]
    plot_src  = df_src[df_src['works_per_year']  <= x_max]

    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    total_inst = int(df_inst['cum_institutions_above'].iloc[0])
    total_src  = int(df_src['cum_sources_above'].iloc[0])

    ax.plot(plot_inst['works_per_year'], plot_inst['pct_retained'],
            color='steelblue', linewidth=1.5,
            label=f'Institutions  (total = {total_inst:,})')
    ax.plot(plot_src['works_per_year'],  plot_src['pct_retained'],
            color='green',     linewidth=1.5,
            label=f'Sources  (total = {total_src:,})')

    ax.set_ylim(60, 100)
    ax.set_xlim(0, x_max)
    ax.set_xlabel(r'Annual work count threshold ($\tau_U$)', labelpad=4)
    ax.set_ylabel('% works retained', labelpad=4)
    ax.legend(loc='lower left', framealpha=1.0)

    for level in (75, 85, 90, 95, 99):
        ax.axhline(level, color='grey', linewidth=0.7, linestyle='--', alpha=0.6, zorder=0)
        ax.text(1.01, level, f'{level}%', va='center', fontsize=8, color='grey',
                transform=ax.get_yaxis_transform())

    ticks = [t for t in ax.get_xticks() if 0 < t <= x_max]

    # ── Lower secondary axis: institutions retained ───────────────────────────
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(
        [_count_at_tick(df_inst, 'cum_institutions_above', t) for t in ticks],
    )
    ax2.set_xlabel('Institutions retained', labelpad=8)

    # ── Upper secondary axis: sources retained (offset above ax2) ────────────
    ax3 = ax.twiny()
    ax3.set_xlim(ax.get_xlim())
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(
        [_count_at_tick(df_src, 'cum_sources_above', t) for t in ticks],
    )
    ax3.spines['top'].set_position(('outward', 50))
    ax3.set_xlabel('Sources retained', labelpad=8)

    sup = fig.suptitle('Institution and source retention curves', y=1.02)
    fig.tight_layout()

    out_path = PLOTS / 'plot1_institution_elbow.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved {out_path}')

    sup.set_visible(False)
    latex_path = PLOTS / 'plot1_institution_elbow_latex.pdf'
    fig.savefig(latex_path, bbox_inches='tight')
    print(f'Saved {latex_path}')
    sup.set_visible(True)

    # Console summary
    print(f'\nRetention at reference levels (baseline t_x={_BASELINE_TX}):')
    print(f'  {"τ":>4}  {"inst%":>6}  {"N_inst":>7}  {"src%":>6}  {"N_src":>6}')
    for t in [5, 10, 15, 20]:
        ri = df_inst[df_inst['works_per_year'] >= t]
        rs = df_src[df_src['works_per_year'] >= t]
        if ri.empty or rs.empty:
            continue
        print(f'  {t:>4}  {ri.iloc[0]["pct_retained"]:>6.1f}  '
              f'{int(ri.iloc[0]["cum_institutions_above"]):>7,}  '
              f'{rs.iloc[0]["pct_retained"]:>6.1f}  '
              f'{int(rs.iloc[0]["cum_sources_above"]):>6,}')


def main():
    with duckdb.connect() as db:
        df_inst = fetch_elbow_data(db)
        df_src  = fetch_source_elbow_data(db)
    plot1(df_inst, df_src)


if __name__ == '__main__':
    main()
    print('FINISHED!')
