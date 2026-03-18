"""
plot_maker.py — Plots for the paper.

Plot 1: Institution works-count elbow.
    x = minimum works_count threshold (cutoff)
    y = % of total works from institutions with >= cutoff works
    Used to select the institution inclusion threshold.
"""
from pathlib import Path
import duckdb
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.objects as so
import seaborn as sns

config_path = Path('./config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)
    WORKING = Path(config['WORKING'])
    PLOTS = Path(config['PROJECT_ROOT']) / 'plots'

PARQUET = WORKING / 'parquet'
PLOTS.mkdir(exist_ok=True)


def fetch_elbow_data(db) -> pd.DataFrame:
    """
    Adapt the PLOTS.md draft SQL to corpus_authorships.parquet.
    Returns one row per distinct works_count with:
        works_count        -- institution size (number of distinct works)
        institutions_count -- number of institutions at exactly this size
        pct_retained       -- % of total works from institutions with >= works_count works
    """
    return db.sql(f"""
        WITH institution_works AS (
            SELECT institution_idx,
                   COUNT(DISTINCT work_idx) AS works_count
            FROM '{PARQUET}/corpus_authorships.parquet'
            WHERE institution_idx IS NOT NULL
            GROUP BY institution_idx
        ),
        freq AS (
            SELECT works_count,
                   COUNT(*) AS institutions_count
            FROM institution_works
            GROUP BY works_count
        ),
        cumul AS (
            SELECT works_count,
                   institutions_count,
                   SUM(institutions_count * works_count) OVER ()                                          AS total_works,
                   SUM(institutions_count * works_count) OVER (ORDER BY works_count
                       ROWS UNBOUNDED PRECEDING)                                                          AS cum_works_to
            FROM freq
        )
        SELECT works_count,
               institutions_count,
               (total_works - COALESCE(LAG(cum_works_to) OVER (ORDER BY works_count), 0))
                   * 100.0 / total_works                                                                  AS pct_retained
        FROM cumul
        ORDER BY works_count
    """).df()


YEARS = 25   # 2000–2024 inclusive

def plot1(df: pd.DataFrame) -> None:
    df = df.copy()
    df['cum_institutions_above'] = df['institutions_count'].iloc[::-1].cumsum().iloc[::-1].values
    df['works_per_year'] = df['works_count'] / YEARS

    plot_df = df[df['works_per_year'] <= 40].copy()   # 1000 works / 25 years

    fig = plt.figure(figsize=(9, 5))
    (
        so.Plot(plot_df, x='works_per_year', y='pct_retained')
        .add(so.Line(linewidth=1.5, color='steelblue'))
        .label(x='Annual work count threshold',
               y='% of total works retained')
        .theme(sns.axes_style('whitegrid'))
        .on(fig)
        .plot()
    )

    ax = fig.axes[0]
    ax.set_ylim(40, 100)

    for level in (50, 75, 90, 95, 99):
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
    print(f'\nWorks-per-year cutoff at each retention level (÷{YEARS} years):')
    for level in (50, 75, 90, 95, 99):
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
