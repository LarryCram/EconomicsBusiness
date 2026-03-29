"""
plot_maker.py — Plots for the paper.

Plot 1: Institution works-count elbow.
    x = minimum works_count threshold (cutoff, in works/year)
    y = % of total works from institutions with >= cutoff works/year
    Computed over the baseline window (t_x=5, 2020–2024) so the institution
    counts on the secondary x-axis match institution_retention.py at τ_U=10.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_params

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.objects as so
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

def plot1(df: pd.DataFrame) -> None:
    df = df.copy()
    df['works_per_year'] = df['works_count'] / N_YEARS

    plot_df = df[df['works_per_year'] <= 40].copy()

    fig = plt.figure(figsize=(9, 3.5))
    (
        so.Plot(plot_df, x='works_per_year', y='pct_retained')
        .add(so.Line(linewidth=1.5, color='steelblue'))
        .label(x=r'Annual work count threshold ($\tau_U$)',
               y='% works retained')
        .theme(sns.axes_style('whitegrid'))
        .on(fig)
        .plot()
    )

    ax = fig.axes[0]
    ax.set_ylim(60, 100)

    for level in (75, 85, 90, 95, 99):
        ax.axhline(level, color='grey', linewidth=0.7, linestyle='--', alpha=0.6)
        ax.text(1.01, level, f'{level}%', va='center', fontsize=8, color='grey',
                transform=ax.get_yaxis_transform())

    # Secondary x-axis: institutions included at each tick (in works_per_year units)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = [t for t in ax.get_xticks() if ax.get_xlim()[0] < t <= ax.get_xlim()[1]]
    ax2.set_xticks(ticks)
    label_size = ax.xaxis.label.get_size()
    ax2.set_xticklabels([
        f'{int(df.loc[df["works_per_year"] >= t, "cum_institutions_above"].iloc[0]):,}'
        if len(df[df['works_per_year'] >= t]) > 0 else '0'
        for t in ticks
    ], fontsize=ax.get_xticklabels()[0].get_size() if ax.get_xticklabels() else 10)
    ax2.set_xlabel('Institutions retained', fontsize=label_size, labelpad=8)

    sup = fig.suptitle('Institution retention curve', fontsize=label_size, y=1.08)
    fig.tight_layout()

    out_path = PLOTS / 'plot1_institution_elbow.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved {out_path}')

    sup.set_visible(False)
    latex_path = PLOTS / 'plot1_institution_elbow_latex.pdf'
    fig.savefig(latex_path, bbox_inches='tight')
    print(f'Saved {latex_path}')
    sup.set_visible(True)

    # Print threshold values at reference levels
    print(f'\nWorks-per-year cutoff at each retention level (÷{N_YEARS} years, baseline t_x={_BASELINE_TX}):')
    for level in (75, 90, 95, 99):
        row = df[df['pct_retained'] >= level].iloc[-1]
        n_inst = int(row['cum_institutions_above'])
        print(f'  >= {level}%: {row["works_per_year"]:.1f} works/yr '
              f'({int(row["works_count"])} total)  institutions = {n_inst:,}')


def main():
    with duckdb.connect() as db:
        df = fetch_elbow_data(db)
    plot1(df)


if __name__ == '__main__':
    main()
    print('FINISHED!')
