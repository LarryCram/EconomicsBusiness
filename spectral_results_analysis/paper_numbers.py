"""
paper_numbers.py — Recompute the in-text numbers of the paper that are not printed by a figure script.

QSS revision. Run once per config (submitted = locked econ_bus, corrected = econ_bus_rev)
and compare. Numbers covered:

  §4.1  clique share of source pi (E+B and E-only, journal-only m=1000);
        directed link density d_XY between Top-E / Rest-E / Top-B / Rest-B
        (top ten by field-specific journal-only ranking; rho=0 citation weights);
        number of E-labelled institutions among the top 20 (E+B, institution-only)
  §5    ABDC 2025 A* journals: in corpus, among the top-N by v (N = number in corpus),
        with v>1; Ritzberger (2008) top/excellent journals: in corpus, v>1

Other in-text numbers come from figure-script outputs (fig_3, fig_5_stability.csv,
fig_stability, analyse_enclaves, table_kernel_structure); see the numbers checklist.

Usage:
  ECONBUS_CONFIG=<config> .venv/bin/python spectral_results_analysis/paper_numbers.py [--json out.json]
"""

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'spectral_ranking'))
from util import load_config, load_runs
from build_csr import build_csr
from benchmark_journals import issn_map, match, name_map, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    paths = load_config()
    bl = next(r for r in load_runs() if r['label'] == 'baseline')
    rc, tu, ts = bl['run_code'], bl['tau_u'], bl['tau_s']
    rk = duckdb.connect(str(paths.working / 'rankings.duckdb'), read_only=True)
    el = duckdb.connect(str(paths.working / 'edge_lists.duckdb'), read_only=True)
    sm = pd.read_parquet(paths.parquet / 'source_master.parquet')
    sm['field_eb'] = sm['field_eb'].fillna('X')
    fld = dict(zip(sm.source_idx, sm.field_eb))
    out = {}

    def tab(fx, m, ut='S', cols='unit_idx, v, pi'):
        t = f"rk_{rc}_{fx}_tauU{tu}_tauS{ts}_vartau_rho0_m{m}_chi50_alpha100"
        return rk.execute(f"SELECT {cols} FROM {t} WHERE unit_type='{ut}'").df() \
                 .set_index('unit_idx')

    # clique: top ten E sources by E+B journal-only v (as in fig_3eb)
    eb = tab('EB', '1000')
    clique = [i for i in eb.sort_values('v', ascending=False).index if fld.get(i) == 'E'][:10]
    out['clique_share_EB_SS'] = float(eb.pi.reindex(clique).sum() / eb.pi.sum())
    e = tab('E', '1000')
    out['clique_share_E_SS'] = float(e.pi.reindex(clique).sum() / e.pi.sum())

    # link density between communities (rho=0 C_SS of the E+B corpus)
    d = build_csr(el, rc, 'EB', tu, ts, rho=0, m=(1, 0, 0, 0))
    C = d.C_SS.tocsr()
    ids = pd.Index(d.source_ids)
    top_e = [i for i in e.sort_values('v', ascending=False).index][:10]
    b = tab('B', '1000')
    top_b = [i for i in b.sort_values('v', ascending=False).index][:10]
    groups = {
        'Top-E': set(top_e), 'Rest-E': {i for i in ids if fld.get(i) == 'E'} - set(top_e),
        'Top-B': set(top_b), 'Rest-B': {i for i in ids if fld.get(i) == 'B'} - set(top_b),
    }
    groups['All-E'] = groups['Top-E'] | groups['Rest-E']
    groups['All-B'] = groups['Top-B'] | groups['Rest-B']
    pos = {g: ids.get_indexer(sorted(s)) for g, s in groups.items()}
    order = ['Top-E', 'Rest-E', 'All-E', 'Top-B', 'Rest-B', 'All-B']
    dens = pd.DataFrame(index=order, columns=order, dtype=float)
    for x in order:
        for y in order:
            px, py = pos[x][pos[x] >= 0], pos[y][pos[y] >= 0]
            tot = C[px][:, py].sum()
            same = groups[x] & groups[y]
            pairs = len(px) * len(py) - len(same)
            if same:   # remove the diagonal (self-pairs) for overlapping groups
                diag = sum(C[i, i] for i in ids.get_indexer(sorted(same)) if i >= 0)
                tot -= diag
            dens.loc[x, y] = tot / pairs
    out['link_density'] = dens.round(3).to_dict()
    print('link density d_XY (rows citing, columns cited):')
    print(dens.round(3).to_string())

    # institutions: E-labelled among top 20 (E+B institution-only)
    ii = tab('EB', '0001', ut='U')
    ife = pd.read_parquet(paths.parquet / 'institution_field_eb.parquet')
    ilab = dict(zip(ife.unit_idx, ife.field_eb))
    top20 = ii.sort_values('v', ascending=False).index[:20]
    out['top20_inst_E'] = int(sum(ilab.get(i) == 'E' for i in top20))

    # ABDC 2025 A* and Ritzberger
    base = tab(bl['fx'], '0110')
    imap = issn_map(paths.parquet)
    a = pd.read_excel(paths.data / 'source_masters' / 'ABDC-JQL-2025-v1-260326.xlsx',
                      sheet_name='ABDC_25_A')
    a = a[a['2025 rating'].astype(str).str.strip() == 'A*']
    a['sid'] = match(a, ['ISSN', 'ISSNOnline'], imap)
    in_corpus = sorted(set(a.sid.dropna().astype('int64')) & set(base.index))
    n = len(in_corpus)
    top_n = set(base.sort_values('v', ascending=False).index[:n])
    out['abdc_astar_total'] = int(len(a))
    out['abdc_astar_in_corpus'] = n
    out['abdc_astar_in_topN'] = int(len(set(in_corpus) & top_n))
    out['abdc_astar_v_lt_1'] = int((base.v.reindex(in_corpus) < 1).sum())
    r = pd.read_excel(paths.data / 'source_masters' / 'ABDC-JQL-2025-v1-260326.xlsx',
                      sheet_name='Ritzberger')
    nmap = name_map(sm.set_index('source_idx'))
    r['sid'] = r.Title.map(lambda t: nmap.get(norm(t)))
    rin = [int(i) for i in r.sid.dropna() if int(i) in base.index]
    out['ritz_total'] = int(len(r))
    out['ritz_in_corpus'] = len(rin)
    out['ritz_v_gt_1'] = int((base.v.reindex(rin) > 1).sum())
    out['ritz_unmatched'] = r[r.sid.isna()].Title.tolist()
    out['ritz_v_lt_1'] = [sm.set_index('source_idx').source_name.get(i)
                          for i in rin if base.v[i] < 1]

    for k, v in out.items():
        if k != 'link_density':
            print(f'{k}: {v}')
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
