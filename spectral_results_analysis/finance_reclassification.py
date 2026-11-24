"""
finance_reclassification.py — Robustness of the E/B results to counting finance and accounting as business.

QSS revision, referee 3.1.

prepare : copy the parquets of a source working folder into the parquet folder of the
          active config, with source_master.field_eb changed from E or A to B for every
          finance/accounting journal. A journal counts as finance/accounting if any
          registry signal says so: Harzing 'F&A', a WoS category FINANCE, ERA FoR 3501
          (accounting) or 3502 (banking, finance and investment), or OpenAlex top
          subfield Finance/Accounting. X stays X.
analyse : compare the E/B results of two rankings.duckdb files (as classified vs
          finance->B): field counts, the top-10 E clique and its share of source pi,
          combined-vs-separate offsets for journal-only and bipartite models, and the
          field mix of the non-clique top 20.

Usage:
  ECONBUS_CONFIG=<fin config> .venv/bin/python spectral_results_analysis/finance_reclassification.py \
      prepare --source /home/lc/m/working/econ_bus_rev/parquet
  (then run build_edge_lists, build_institution_field_eb, filter_mode_units, run_rankings
   with the same ECONBUS_CONFIG)
  .venv/bin/python spectral_results_analysis/finance_reclassification.py analyse \
      --base /home/lc/m/working/econ_bus_rev --fin /home/lc/m/working/econ_bus_rev_fin
"""

import argparse
import shutil
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

FIN_SQL = """
      coalesce(harzing_field, '') LIKE '%F&A%'
   OR list_contains(coalesce(wos_categories, []), 'FINANCE')
   OR len(list_filter(coalesce(era_for_codes, []), x -> x IN ('3501', '3502'))) > 0
   OR coalesce(subfield_name, '') IN ('Finance', 'Accounting')
"""


def prepare(src: Path):
    paths = load_config()
    dst = paths.parquet.resolve()
    assert dst != src.resolve() and 'econ_bus_qss_submitted' not in str(dst) \
        and dst.parent.name != 'econ_bus', f'refusing to write into {dst}'
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob('*.parquet')):
        if f.name != 'source_master.parquet':
            shutil.copy2(f, dst / f.name)
    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT * REPLACE (
                CASE WHEN field_eb IN ('E', 'A') AND ({FIN_SQL}) THEN 'B' ELSE field_eb END
                AS field_eb)
            FROM '{src}/source_master.parquet'
        ) TO '{dst}/source_master.parquet' (FORMAT PARQUET)
    """)
    ch = con.execute(f"""
        SELECT a.field_eb AS before, b.field_eb AS after, COUNT(*) AS n
        FROM '{src}/source_master.parquet' a
        JOIN '{dst}/source_master.parquet' b USING (source_idx)
        GROUP BY ALL ORDER BY ALL""").df()
    print(ch.to_string(index=False))


def analyse(base: Path, fin: Path):
    bl = next(r for r in load_runs() if r['label'] == 'baseline')
    rc, tu, ts = bl['run_code'], bl['tau_u'], bl['tau_s']

    def t(fx, m):
        return f"rk_{rc}_{fx}_tauU{tu}_tauS{ts}_vartau_rho0_m{m}_chi50_alpha100"

    out = []
    for label, w in [('as classified', base), ('finance -> B', fin)]:
        sm = pd.read_parquet(w / 'parquet' / 'source_master.parquet',
                             columns=['source_idx', 'source_name', 'field_eb'])
        sm['field_eb'] = sm['field_eb'].fillna('X')
        fld = dict(zip(sm.source_idx, sm.field_eb))
        nm = dict(zip(sm.source_idx, sm.source_name))
        rk = duckdb.connect(str(w / 'rankings.duckdb'), read_only=True)

        def get(fx, m, col='v'):
            d = rk.execute(f"SELECT unit_idx, {col} FROM {t(fx, m)} WHERE unit_type='S'").df()
            return d.set_index('unit_idx')[col]

        print(f"\n=== {label} ===")
        for m, mname in [('1000', 'journal-only'), ('0110', 'bipartite')]:
            vc, pic = get('EB', m), get('EB', m, 'pi')
            n_e = sum(fld.get(i) == 'E' for i in vc.index)
            n_b = sum(fld.get(i) == 'B' for i in vc.index)
            clique = [i for i in vc.sort_values(ascending=False).index if fld.get(i) == 'E'][:10]
            share = pic.reindex(clique).sum() / pic.sum()
            rest = vc.drop(clique).sort_values(ascending=False).head(20)
            mix = {f: sum(fld.get(i) == f for i in rest.index) for f in 'EB'}
            offs = {}
            for f in 'EB':
                sep = get(f, m)
                ids = [i for i in sep.index if fld.get(i) == f and i in vc.index]
                offs[f] = float(np.median(np.log10(sep[ids] / vc[ids])))
            rel = offs['E'] - offs['B']
            print(f"  {mname:12s} N_E={n_e} N_B={n_b}  clique share of pi={share:.3f}  "
                  f"non-clique top20 E/B={mix['E']}/{mix['B']}  "
                  f"separate-vs-combined shift E={offs['E']:+.3f} B={offs['B']:+.3f} "
                  f"(E vs B {rel:+.3f} dex, x{10 ** abs(rel):.2f})")
            if m == '1000':
                print('    clique:', '; '.join(nm.get(i, '?')[:28] for i in clique))
            out.append(dict(variant=label, model=mname, N_E=n_e, N_B=n_b,
                            clique_share=share, top20_E=mix['E'], top20_B=mix['B'],
                            shift_E=offs['E'], shift_B=offs['B'], shift_E_vs_B=rel,
                            clique='; '.join(nm.get(i, '?') for i in clique)))
    paths = load_config()
    dest = paths.data / 'finance_reclassification.csv'
    pd.DataFrame(out).to_csv(dest, index=False)
    print(f"\nwritten to {dest}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('prepare'); p1.add_argument('--source', required=True)
    p2 = sub.add_parser('analyse'); p2.add_argument('--base', required=True)
    p2.add_argument('--fin', required=True)
    a = ap.parse_args()
    if a.cmd == 'prepare':
        prepare(Path(a.source))
    else:
        analyse(Path(a.base), Path(a.fin))
