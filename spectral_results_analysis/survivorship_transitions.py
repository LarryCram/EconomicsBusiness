"""
survivorship_transitions.py — Stability of above-average influence without survivorship bias.

QSS revision, referee 3.2. The submitted stability analysis (fig_stability)
conditions on units present in both 2000–04 and 2020–24. Here every unit ranked
in either window is classified in both windows as

    v>1        ranked, above corpus-average influence
    v<1        ranked, below average
    below τ    has corpus works in the window but is not ranked (below the
               works-per-year threshold, or pruned from the connected network)
    no works   no corpus works in the window

and two transition tables are reported per unit type: forward (all units ranked in
the first window → state in the last) and backward (all units ranked in the last
window → state in the first). Intermediate window pairs are summarised too.

Reads the baseline bipartite ranking table of each window (labels t1..t4 and
baseline in params.csv) and the corpus parquets of the active config.

Output: data/survivorship_transitions.csv, printed tables.

Usage:
  ECONBUS_CONFIG=config.rev.yaml .venv/bin/python spectral_results_analysis/survivorship_transitions.py
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

WINDOWS = [('2000–04', 't1'), ('2005–09', 't2'), ('2010–14', 't3'),
           ('2015–19', 't4'), ('2020–24', 'baseline')]
STATES = ['v>1', 'v<1', 'below τ', 'no works']


def tname(r):
    chi = 'STAR' if r['chi'] == -1.0 else str(round(r['chi'] * 100))
    return (f"rk_{r['run_code']}_{r['fx']}_tauU{r['tau_u']}_tauS{r['tau_s']}_vartau"
            f"_rho{r['rho']}_m{r['m']}_chi{chi}_alpha{round(r['alpha'] * 100)}")


def main():
    paths = load_config()
    P = paths.parquet
    runs = {r['label']: r for r in load_runs()}
    rk = duckdb.connect(str(paths.working / 'rankings.duckdb'), read_only=True)
    con = duckdb.connect()

    # works per unit per window, from the corpus of the active config
    wk = con.execute(f"""
        SELECT 'S' AS unit_type, source_idx AS unit_idx,
               (publication_year - 2000) // 5 AS w, COUNT(*) AS n
        FROM '{P}/corpus_works.parquet' WHERE publication_year BETWEEN 2000 AND 2024
        GROUP BY ALL
        UNION ALL
        SELECT 'U', a.institution_idx, (w.publication_year - 2000) // 5, COUNT(DISTINCT a.work_idx)
        FROM '{P}/corpus_authorships.parquet' a
        JOIN '{P}/corpus_works.parquet' w USING (work_idx)
        WHERE a.institution_idx IN (SELECT institution_idx FROM '{P}/corpus_institutions.parquet')
          AND w.publication_year BETWEEN 2000 AND 2024
        GROUP BY ALL
    """).df()

    ranked = {}
    for k, (lab, run) in enumerate(WINDOWS):
        d = rk.execute(f"SELECT unit_idx, unit_type, v FROM {tname(runs[run])}").df()
        d['w'] = k
        ranked[k] = d

    def state(ut, k, ids):
        d = ranked[k][ranked[k].unit_type == ut].set_index('unit_idx')['v']
        has = set(wk[(wk.unit_type == ut) & (wk.w == k)].unit_idx)
        out = []
        for i in ids:
            if i in d.index:
                out.append('v>1' if d[i] > 1 else 'v<1')
            else:
                out.append('below τ' if i in has else 'no works')
        return pd.Series(out, index=ids)

    rows = []
    for ut, name in [('S', 'Sources'), ('U', 'Institutions')]:
        for a, b, direction in [(0, 4, 'forward'), (4, 0, 'backward'),
                                (0, 1, 'forward'), (1, 2, 'forward'),
                                (2, 3, 'forward'), (3, 4, 'forward')]:
            ids = ranked[a][ranked[a].unit_type == ut].unit_idx.tolist()
            sa, sb = state(ut, a, ids), state(ut, b, ids)
            tab = pd.crosstab(sa, sb).reindex(index=['v>1', 'v<1'], columns=STATES,
                                              fill_value=0)
            for s_from in tab.index:
                for s_to in tab.columns:
                    rows.append(dict(unit=name, direction=direction,
                                     from_window=WINDOWS[a][0], to_window=WINDOWS[b][0],
                                     from_state=s_from, to_state=s_to,
                                     n=int(tab.loc[s_from, s_to])))
            if {a, b} == {0, 4}:
                pct = tab.div(tab.sum(axis=1), axis=0).mul(100).round(1)
                print(f"\n{name}, {direction}: state in {WINDOWS[a][0]} (rows, all ranked units)"
                      f" -> state in {WINDOWS[b][0]} (columns)")
                print(pd.concat([tab, tab.sum(axis=1).rename('total')], axis=1).to_string())
                print(pct.to_string())

    out = pd.DataFrame(rows)
    dest = paths.data / 'survivorship_transitions.csv'
    out.to_csv(dest, index=False)

    # consecutive-window retention of v>1, counting exits
    print('\nRetention of v>1 between consecutive windows (share of v>1 units in the '
          'earlier window), counting exits:')
    c = out[(out.direction == 'forward') & (out.from_state == 'v>1')
            & ~((out.from_window == '2000–04') & (out.to_window == '2020–24'))]
    piv = c.pivot_table(index=['unit', 'from_window', 'to_window'], columns='to_state',
                        values='n', aggfunc='sum').reindex(columns=STATES, fill_value=0)
    print(piv.div(piv.sum(axis=1), axis=0).mul(100).round(1).to_string())
    print(f'\nwritten to {dest}')


if __name__ == '__main__':
    main()
