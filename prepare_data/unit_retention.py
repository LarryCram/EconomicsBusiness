"""
unit_retention.py — Source and institution retention curves.

Builds a DISTINCT (work_idx, source_idx, institution_idx) base table for the
baseline census window (2020–2024, derived from params.csv), then counts works
per source and per institution independently.

Differences from institution_retention.py:
  - Institution type filter applied (corpus_institutions.parquet):
    education / nonprofit / government / other; company and funder excluded.
  - Each unit judged on its own distinct-work count — no best-unit-wins inflating
    the institution curve.  N counts match build_edge_lists.py exactly.
  - pct_retained for institutions: fraction of distinct works where ≥1 qualifying
    institution has works_count ≥ threshold (best-institution-wins, but now only
    over the type-filtered institution pool).

Outputs:
    plots/fig_1.pdf              — elbow plot (with title)
    plots/fig_1_latex.pdf        — same without title (for paper)
    data/unit_retention.csv      — N and pct_works at τ ∈ {5,10,15,20}
"""

import sys
from pathlib import Path
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

paths   = load_config()
PARQUET = paths.parquet
PLOTS   = paths.plots
PLOTS.mkdir(exist_ok=True)

_baseline = next(r for r in load_runs() if r['label'] == 'baseline')
YEAR_MIN  = min(_baseline['tc0'], _baseline['tt0'])
YEAR_MAX  = max(_baseline['tc1'], _baseline['tt1'])
N_YEARS   = YEAR_MAX - YEAR_MIN + 1


# ---------------------------------------------------------------------------
# Base table
# ---------------------------------------------------------------------------

def build_base(db) -> None:
    """DISTINCT (work_idx, source_idx, institution_idx) for the baseline window."""
    db.execute(f"""
        CREATE OR REPLACE TEMP TABLE base AS
        SELECT DISTINCT w.work_idx, w.source_idx, a.institution_idx
        FROM '{PARQUET}/corpus_works.parquet'        w
        JOIN '{PARQUET}/corpus_authorships.parquet'  a  USING (work_idx)
        JOIN '{PARQUET}/corpus_institutions.parquet' ci ON a.institution_idx = ci.institution_idx
        WHERE w.publication_year BETWEEN {YEAR_MIN} AND {YEAR_MAX}
          AND a.institution_idx IS NOT NULL
          AND NOT w.is_paratext
          AND NOT w.is_retracted
    """)
    n       = db.execute("SELECT COUNT(*)                    FROM base").fetchone()[0]
    n_works = db.execute("SELECT COUNT(DISTINCT work_idx)    FROM base").fetchone()[0]
    n_src   = db.execute("SELECT COUNT(DISTINCT source_idx)  FROM base").fetchone()[0]
    n_inst  = db.execute("SELECT COUNT(DISTINCT institution_idx) FROM base").fetchone()[0]
    print(f"Base: {n:,} triples  |  {n_works:,} works  |  "
          f"{n_src:,} sources  |  {n_inst:,} institutions")


# ---------------------------------------------------------------------------
# Per-unit work counts and work-level best-institution
# ---------------------------------------------------------------------------

def build_unit_counts(db) -> None:
    db.execute("""
        CREATE OR REPLACE TEMP TABLE src_works AS
        SELECT source_idx, COUNT(DISTINCT work_idx) AS works_count
        FROM base GROUP BY source_idx
    """)
    db.execute("""
        CREATE OR REPLACE TEMP TABLE inst_works AS
        SELECT institution_idx, COUNT(DISTINCT work_idx) AS works_count
        FROM base GROUP BY institution_idx
    """)
    # For each work: max works_count across its (type-filtered) institutions
    db.execute("""
        CREATE OR REPLACE TEMP TABLE work_best_inst AS
        SELECT b.work_idx, MAX(iw.works_count) AS best_inst_works
        FROM base b
        JOIN inst_works iw USING (institution_idx)
        GROUP BY b.work_idx
    """)


# ---------------------------------------------------------------------------
# Elbow data
# ---------------------------------------------------------------------------

def elbow_src(db) -> pd.DataFrame:
    """Source elbow: works_per_year × (cum_src_above, pct_retained)."""
    return db.execute(f"""
        WITH freq AS (
            SELECT works_count,
                   COUNT(*)         AS n_src,
                   SUM(works_count) AS sum_works
            FROM src_works
            GROUP BY works_count
        ),
        totals AS (
            SELECT SUM(n_src)    AS total_src,
                   SUM(sum_works) AS total_works
            FROM freq
        ),
        cumul AS (
            SELECT works_count, n_src, sum_works,
                   SUM(n_src)    OVER (ORDER BY works_count ROWS UNBOUNDED PRECEDING) AS cum_src_to,
                   SUM(sum_works) OVER (ORDER BY works_count ROWS UNBOUNDED PRECEDING) AS cum_wk_to,
                   total_src, total_works
            FROM freq CROSS JOIN totals
        )
        SELECT
            works_count / {N_YEARS}.0                                                    AS works_per_year,
            n_src,
            total_src - COALESCE(LAG(cum_src_to) OVER (ORDER BY works_count), 0)         AS cum_src_above,
            (total_works - COALESCE(LAG(cum_wk_to) OVER (ORDER BY works_count), 0))
                * 100.0 / total_works                                                     AS pct_retained
        FROM cumul
        ORDER BY works_count
    """).fetchdf()


def elbow_inst(db) -> pd.DataFrame:
    """
    Institution elbow: works_per_year × (cum_inst_above, pct_retained).
    N = institutions with works_count >= threshold (matches build_edge_lists).
    pct = fraction of distinct works whose best (type-filtered) institution passes.
    """
    return db.execute(f"""
        WITH inst_freq AS (
            SELECT works_count, COUNT(*) AS n_inst
            FROM inst_works
            GROUP BY works_count
        ),
        work_freq AS (
            SELECT best_inst_works AS works_count, COUNT(*) AS n_works
            FROM work_best_inst
            GROUP BY best_inst_works
        ),
        all_wc AS (
            SELECT works_count FROM inst_freq
            UNION
            SELECT works_count FROM work_freq
        ),
        combined AS (
            SELECT a.works_count,
                   COALESCE(i.n_inst,  0) AS n_inst,
                   COALESCE(w.n_works, 0) AS n_works
            FROM all_wc a
            LEFT JOIN inst_freq i USING (works_count)
            LEFT JOIN work_freq w USING (works_count)
        ),
        totals AS (
            SELECT SUM(n_inst)  AS total_inst,
                   SUM(n_works) AS total_works
            FROM combined
        ),
        cumul AS (
            SELECT works_count, n_inst, n_works,
                   SUM(n_inst)  OVER (ORDER BY works_count ROWS UNBOUNDED PRECEDING) AS cum_inst_to,
                   SUM(n_works) OVER (ORDER BY works_count ROWS UNBOUNDED PRECEDING) AS cum_wk_to,
                   total_inst, total_works
            FROM combined CROSS JOIN totals
        )
        SELECT
            works_count / {N_YEARS}.0                                                      AS works_per_year,
            n_inst,
            total_inst  - COALESCE(LAG(cum_inst_to) OVER (ORDER BY works_count), 0)        AS cum_inst_above,
            (total_works - COALESCE(LAG(cum_wk_to)  OVER (ORDER BY works_count), 0))
                * 100.0 / total_works                                                       AS pct_retained
        FROM cumul
        ORDER BY works_count
    """).fetchdf()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _at_tick(df: pd.DataFrame, col: str, t: float) -> str:
    rows = df[df['works_per_year'] >= t]
    return f'{int(rows.iloc[0][col]):,}' if not rows.empty else '0'


def plot_fig1(df_inst: pd.DataFrame, df_src: pd.DataFrame) -> None:
    x_max     = 40
    plot_inst = df_inst[df_inst['works_per_year'] <= x_max]
    plot_src  = df_src[df_src['works_per_year']  <= x_max]

    total_inst = int(df_inst['cum_inst_above'].iloc[0])
    total_src  = int(df_src['cum_src_above'].iloc[0])

    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(plot_inst['works_per_year'], plot_inst['pct_retained'],
            color='#377eb8', linewidth=1.5,
            label=f'Institutions  (total = {total_inst:,})')
    ax.plot(plot_src['works_per_year'],  plot_src['pct_retained'],
            color='#e41a1c', linewidth=1.5,
            label=f'Sources  (total = {total_src:,})')

    ax.set_ylim(60, 100)
    ax.set_xlim(0, x_max)
    ax.set_xlabel(r'Annual work count threshold ($\tau$)', labelpad=4)
    ax.set_ylabel('% works retained', labelpad=4)
    ax.legend(loc='lower left', framealpha=1.0)

    for level in (75, 85, 90, 95, 99):
        ax.axhline(level, color='grey', linewidth=0.7, linestyle='--', alpha=0.6, zorder=0)
        ax.text(1.01, level, f'{level}%', va='center', fontsize=8, color='grey',
                transform=ax.get_yaxis_transform())

    ticks = [t for t in ax.get_xticks() if 0 < t <= x_max]

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([_at_tick(df_inst, 'cum_inst_above', t) for t in ticks])
    ax2.set_xlabel('Institutions retained', labelpad=8)

    ax3 = ax.twiny()
    ax3.set_xlim(ax.get_xlim())
    ax3.set_xticks(ticks)
    ax3.set_xticklabels([_at_tick(df_src, 'cum_src_above', t) for t in ticks])
    ax3.spines['top'].set_position(('outward', 50))
    ax3.set_xlabel('Sources retained', labelpad=8)

    sup = fig.suptitle(
        f'Institution and source retention curves  (baseline {YEAR_MIN}–{YEAR_MAX})',
        y=1.02,
    )
    fig.tight_layout()

    for path, visible in [(PLOTS / 'fig_1.pdf', True), (PLOTS / 'fig_1_latex.pdf', False)]:
        sup.set_visible(visible)
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved {path}')

    sup.set_visible(True)

    print(f'\nRetention (baseline {YEAR_MIN}–{YEAR_MAX}):')
    print(f'  {"τ":>4}  {"inst%":>6}  {"N_inst":>7}  {"src%":>6}  {"N_src":>6}')
    for t in [5, 10, 15, 20]:
        ri = df_inst[df_inst['works_per_year'] >= t]
        rs = df_src[df_src['works_per_year']  >= t]
        if ri.empty or rs.empty:
            continue
        print(f'  {t:>4}  {ri.iloc[0]["pct_retained"]:>6.1f}  '
              f'{int(ri.iloc[0]["cum_inst_above"]):>7,}  '
              f'{rs.iloc[0]["pct_retained"]:>6.1f}  '
              f'{int(rs.iloc[0]["cum_src_above"]):>6,}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    with duckdb.connect() as db:
        db.sql(f"""
            SET temp_directory = '{paths.working}/.tmp';
            SET memory_limit   = '56GB';
        """)
        print("Building base table…")
        build_base(db)
        print("Computing per-unit work counts…")
        build_unit_counts(db)
        print("Computing source elbow…")
        df_src  = elbow_src(db)
        print("Computing institution elbow…")
        df_inst = elbow_inst(db)

        plot_fig1(df_inst, df_src)

        # Save summary CSV
        rows = []
        for t in [5, 10, 15, 20]:
            ri = df_inst[df_inst['works_per_year'] >= t]
            rs = df_src[df_src['works_per_year']  >= t]
            if ri.empty or rs.empty:
                continue
            rows.append({
                'tau':       t,
                'inst_pct':  round(float(ri.iloc[0]['pct_retained']), 1),
                'n_inst':    int(ri.iloc[0]['cum_inst_above']),
                'src_pct':   round(float(rs.iloc[0]['pct_retained']), 1),
                'n_src':     int(rs.iloc[0]['cum_src_above']),
            })
        out = paths.data / 'unit_retention.csv'
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f'\nSaved → {out}')


if __name__ == '__main__':
    main()
    print('FINISHED!')
