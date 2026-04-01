"""
institution_retention.py — Select τ_U (institution retention threshold).

For each of the 21 (t_x, F_x) corpus definitions and each τ_U ∈ {0, 5, 10, 15, 20},
reports the number of institutions retained (mean works/year ≥ τ_U) and the
proportion of total corpus works those institutions account for.

Rows:    21 (t_x, F_x) pairs
Columns: τ_U ∈ {0, 5, 10, 15, 20} × (inst_count, pct_works)

Requires source_master.parquet to have the field_subset column ('E'/'B'/NULL),
produced by journal_filter_match_oa.py.

Outputs:
    data/institution_retention.csv   -- table of retention rates across all (t_x, F_x, τ_U)
    plots/fig_1.pdf                  -- institution and source retention elbow (with title)
    plots/fig_1_latex.pdf            -- same plot without title (for paper)
"""

import sys
from pathlib import Path
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_params

paths  = load_config()
params = load_params()
PARQUET = paths.parquet
PLOTS   = paths.plots
PLOTS.mkdir(exist_ok=True)

# Time windows from params.yaml: t_x → (census_start, census_end, target_start, target_end)
_tw = params['time_windows']
TIME_WINDOWS = {
    tx: (w['census'][0], w['census'][1], w['target'][0], w['target'][1])
    for tx, w in _tw.items()
}

# Recommended τ_U floor per field subset (informational — logged at runtime)
TAU_U_FLOOR = params['tau_u_floor']   # {'E': 5, 'B': 5, 'A': 10}

TAU_U_VALUES = [0, 1, 2, 3, 4, 5, 10, 15, 20]

# Additional WHERE clause on source_master.field_subset for each F_x
FIELD_COND = {
    'E': "AND sm.field_subset = 'E'",
    'B': "AND sm.field_subset = 'B'",
    'A': "",
}

# Baseline window for elbow plot: t_x=5 (2020–2024)
_BASELINE_TX = 5
_baseline_tw  = params['time_windows'][_BASELINE_TX]
_YEAR_MIN     = min(_baseline_tw['census'][0], _baseline_tw['target'][0])
_YEAR_MAX     = max(_baseline_tw['census'][1], _baseline_tw['target'][1])
N_YEARS       = _YEAR_MAX - _YEAR_MIN + 1   # 5


# ---------------------------------------------------------------------------
# Retention table
# ---------------------------------------------------------------------------

def compute_retention(db, tx: int, fx: str) -> pd.DataFrame:
    cs, ce, ts, te = TIME_WINDOWS[tx]
    min_year = min(cs, ts)
    max_year = max(ce, te)
    n_years  = max_year - min_year + 1
    fc = FIELD_COND[fx]

    return db.sql(f"""
        WITH inst_works AS (
            -- Works per institution within the (t_x, F_x) window
            SELECT a.institution_idx,
                   COUNT(DISTINCT a.work_idx)              AS works_count,
                   COUNT(DISTINCT a.work_idx) / {n_years}.0 AS works_per_year
            FROM '{PARQUET}/corpus_works.parquet' w
            JOIN '{PARQUET}/source_master.parquet' sm ON w.source_idx = sm.source_idx
            JOIN '{PARQUET}/corpus_authorships.parquet' a  ON w.work_idx = a.work_idx
            WHERE w.publication_year BETWEEN {min_year} AND {max_year}
              AND a.institution_idx IS NOT NULL
              {fc}
            GROUP BY a.institution_idx
        ),
        total AS (
            SELECT COUNT(DISTINCT a.work_idx) AS total_works
            FROM '{PARQUET}/corpus_works.parquet' w
            JOIN '{PARQUET}/source_master.parquet' sm ON w.source_idx = sm.source_idx
            JOIN '{PARQUET}/corpus_authorships.parquet' a  ON w.work_idx = a.work_idx
            WHERE w.publication_year BETWEEN {min_year} AND {max_year}
              AND a.institution_idx IS NOT NULL
              {fc}
        ),
        thresholds AS (
            SELECT unnest([{', '.join(str(v) for v in TAU_U_VALUES)}]) AS tau_u
        ),
        -- For each τ_U: set of retained institution_idxs
        retained_inst AS (
            SELECT t.tau_u, iw.institution_idx
            FROM inst_works iw
            CROSS JOIN thresholds t
            WHERE iw.works_per_year >= t.tau_u
        ),
        -- For each τ_U: distinct works and sources covered by any retained institution
        retained_works AS (
            SELECT ri.tau_u,
                   COUNT(DISTINCT a.work_idx)  AS ret_works,
                   COUNT(DISTINCT w.source_idx) AS ret_sources
            FROM retained_inst ri
            JOIN '{PARQUET}/corpus_authorships.parquet' a  ON ri.institution_idx = a.institution_idx
            JOIN '{PARQUET}/corpus_works.parquet' w        ON a.work_idx = w.work_idx
            JOIN '{PARQUET}/source_master.parquet' sm      ON w.source_idx = sm.source_idx
            WHERE w.publication_year BETWEEN {min_year} AND {max_year}
              {fc}
            GROUP BY ri.tau_u
        )
        SELECT
            ri.tau_u,
            COUNT(DISTINCT ri.institution_idx)            AS retained_inst,
            rw.ret_works                                  AS retained_works,
            ROUND(rw.ret_works * 100.0 / t.total_works, 1) AS pct_works,
            rw.ret_sources                                AS retained_sources
        FROM retained_inst ri
        JOIN retained_works rw USING (tau_u)
        CROSS JOIN total t
        GROUP BY ri.tau_u, rw.ret_works, rw.ret_sources, t.total_works
        ORDER BY ri.tau_u
    """).df()


def build_table(db) -> pd.DataFrame:
    rows = []
    for tx in range(1, 8):
        for fx in ['E', 'B', 'A']:
            print(f"  t_x={tx}  F_x={fx} ...", end='  ', flush=True)
            df = compute_retention(db, tx, fx)
            row = {'t_x': tx, 'F_x': fx}
            for _, r in df.iterrows():
                tau = int(r['tau_u'])
                row[f'tau{tau}_inst']    = int(r['retained_inst'])
                row[f'tau{tau}_pct']     = round(float(r['pct_works']), 1)
                row[f'tau{tau}_sources'] = int(r['retained_sources'])
            floor = TAU_U_FLOOR[fx]
            floor_row = df.loc[df.tau_u == floor]
            floor_str = (f"  τ_U≥{floor}: {int(floor_row['retained_inst'].iloc[0]):,} inst, "
                         f"{floor_row['pct_works'].iloc[0]:.1f}% works"
                         if not floor_row.empty else "")
            print(f"total works = {int(df.loc[df.tau_u == 0, 'retained_works'].iloc[0]):,}{floor_str}")
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Elbow plot (Fig 1)
# ---------------------------------------------------------------------------

def fetch_elbow_data(db) -> pd.DataFrame:
    """
    Returns one row per distinct works_count threshold with:
        works_count            -- institution size (distinct works in the baseline window)
        institutions_count     -- institutions at exactly this size
        cum_institutions_above -- institutions with works_count >= this value
        pct_retained           -- % of works that have at least one author at an institution
                                  with works_count >= this value
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


def plot_fig1(df_inst: pd.DataFrame, df_src: pd.DataFrame) -> None:
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

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(
        [_count_at_tick(df_inst, 'cum_institutions_above', t) for t in ticks],
    )
    ax2.set_xlabel('Institutions retained', labelpad=8)

    ax3 = ax.twiny()
    ax3.set_xlim(ax.get_xlim())
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(
        [_count_at_tick(df_src, 'cum_sources_above', t) for t in ticks],
    )
    ax3.spines['top'].set_position(('outward', 50))
    ax3.set_xlabel('Sources retained', labelpad=8)

    sup = fig.suptitle(
        f'Institution and source retention curves  (baseline t_x={_BASELINE_TX}, '
        f'{_YEAR_MIN}–{_YEAR_MAX})',
        y=1.02,
    )
    fig.tight_layout()

    out = PLOTS / 'fig_1.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

    sup.set_visible(False)
    latex_out = PLOTS / 'fig_1_latex.pdf'
    fig.savefig(latex_out, bbox_inches='tight')
    print(f'Saved {latex_out}')
    sup.set_visible(True)

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    with duckdb.connect() as db:
        db.sql(f"""
            SET temp_directory = '{paths.working}/.tmp';
            SET memory_limit   = '56GB';
        """)
        print("Computing institution retention table (21 rows × 5 τ_U values)...")
        table = build_table(db)
        print("\nFetching elbow data for Fig 1...")
        df_inst = fetch_elbow_data(db)
        df_src  = fetch_source_elbow_data(db)

    out_path = paths.data / 'institution_retention.csv'
    table.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(table.to_string(index=False))

    plot_fig1(df_inst, df_src)


if __name__ == '__main__':
    main()
    print('FINISHED!')
