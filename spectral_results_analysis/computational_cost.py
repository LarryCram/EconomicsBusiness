"""
computational_cost.py — Runtime and peak memory of the baseline ranking and of simpler indicators.

QSS revision, referee 2.5 ("low-cost, auditable": how do computational cost and error
sensitivity compare with other methods?). Every method starts from the same baseline
edge list (el_20242024_EBAX_tauU20_tauS10_vartau) and is run in its own subprocess, so
peak memory is per method. Times are wall-clock on one core of the machine running the
script (one thread for NumPy/BLAS and DuckDB); "assemble" is edge list -> aggregates or sparse matrix (including the DuckDB
query), "solve" is the ranking or averaging step.

Methods
  counts      citations per journal and per institution (author-fractional) and the
              JIF-like mean per work
  mncs        field-normalised MNCS for journals and institutions (mncs_comparison.py,
              baseline only, no bootstrap)
  journal_kh  journal-only Katz-Hubbell, alpha = 0.85, uniform prior (Eigenfactor-like)
  work_kh     work-level Katz-Hubbell, alpha = 0.85 (rank_works_katz_hubbell.katz_hubbell)
  bipartite   baseline bipartite Perron-Frobenius (m = 0110)
  joint       full joint Perron-Frobenius (m = 1111, chi = 0.5)

Output: data/computational_cost.csv

Usage:
  ECONBUS_CONFIG=config.rev.yaml .venv/bin/python spectral_results_analysis/computational_cost.py
"""

import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'spectral_ranking'))
sys.path.insert(0, str(ROOT / 'spectral_results_analysis'))

METHODS = ['counts', 'mncs', 'journal_kh', 'work_kh', 'bipartite', 'joint']
LABELS = {
    'counts': 'Citation counts and JIF-like mean',
    'mncs': 'Field-normalised MNCS',
    'journal_kh': 'Journal-only Katz-Hubbell (alpha 0.85)',
    'work_kh': 'Work-level Katz-Hubbell (alpha 0.85)',
    'bipartite': 'Bipartite Perron-Frobenius (baseline)',
    'joint': 'Joint Perron-Frobenius',
}


def _baseline():
    from util import load_config, load_runs
    paths = load_config()
    bl = next(r for r in load_runs() if r['label'] == 'baseline')
    el = f"el_{bl['run_code']}_{bl['fx']}_tauU{bl['tau_u']}_tauS{bl['tau_s']}_vartau"
    return paths, bl, el


def run_method(method: str) -> dict:
    import duckdb
    import scipy.sparse as sp
    paths, bl, el = _baseline()
    db = duckdb.connect(str(paths.working / 'edge_lists.duckdb'), read_only=True)
    db.execute('SET threads TO 1')
    t0 = time.perf_counter()
    info = {}
    if method == 'counts':
        df = db.execute(f"""
            SELECT DISTINCT citer_work_idx, cited_work_idx, cited_source_idx FROM {el}""").df()
        cw = db.execute(f"""
            SELECT cited_work_idx, cited_inst_idx, ANY_VALUE(cited_inst_weight) AS w
            FROM {el} GROUP BY 1, 2""").df()
        t1 = time.perf_counter()
        c_work = df.groupby('cited_work_idx').size()
        c_src = df.groupby('cited_source_idx').size()
        n_src = df.groupby('cited_source_idx').cited_work_idx.nunique()
        jif = c_src / n_src
        cw['c'] = cw.cited_work_idx.map(c_work).to_numpy() * cw.w.to_numpy()
        c_inst = cw.groupby('cited_inst_idx').c.sum()
        info = dict(units=len(jif) + len(c_inst))
    elif method == 'mncs':
        import pandas as pd
        from mncs_comparison import (load_inputs, citation_counts, normalised_scores,
                                     unit_means, institution_weights_matrix)
        d = load_inputs(paths)
        works, iw, refs = d['works'], d['iw'], d['refs']
        W_ids = works['work_idx'].to_numpy(np.int64)
        grp = pd.factorize(works['field'] + '|' + works['year'].astype(str) + '|'
                           + works['type'].astype(str))[0]
        cited_pos = np.searchsorted(W_ids, refs['cited_work_idx'].to_numpy(np.int64))
        s_index = pd.Index(d['src_ids'])
        sj = s_index.get_indexer(works['source_idx'])
        m = sj >= 0
        W_s = sp.csr_matrix((np.ones(m.sum()), (sj[m], np.nonzero(m)[0])),
                            shape=(len(s_index), len(W_ids)))
        W_u = institution_weights_matrix(iw, pd.Index(d['inst_ids']), W_ids)
        t1 = time.perf_counter()
        ncs = normalised_scores(citation_counts(cited_pos, np.ones(len(refs)), len(W_ids)),
                                grp, grp.max() + 1)
        out = (unit_means(ncs, W_s), unit_means(ncs, W_u))
        info = dict(units=len(out[0]) + len(out[1]))
    elif method == 'work_kh':
        from spectral_ranking_works.rank_works_katz_hubbell import katz_hubbell
        df = db.execute(f"""
            SELECT DISTINCT citer_work_idx, cited_work_idx FROM {el}""").df()
        ids, inv = np.unique(np.concatenate([df.citer_work_idx.to_numpy(),
                                             df.cited_work_idx.to_numpy()]), return_inverse=True)
        n = len(ids)
        r, c = inv[:len(df)], inv[len(df):]
        C = sp.csr_matrix((np.ones(len(df)), (r, c)), shape=(n, n))
        rs = np.asarray(C.sum(axis=1)).ravel()
        H = sp.diags(np.divide(1.0, rs, out=np.zeros(n), where=rs > 0)) @ C
        t1 = time.perf_counter()
        v, iters, _ = katz_hubbell(H.tocsr(), alpha=0.85, tol=1e-8)
        info = dict(units=n, nnz=int(C.nnz), iters=int(iters))
    else:
        from build_csr import build_csr
        from katz_ranker import rank
        m, alpha, chi = {'journal_kh': ((1, 0, 0, 0), 0.85, 0.5),
                         'bipartite': ((0, 1, 1, 0), 1.0, 0.5),
                         'joint': ((1, 1, 1, 1), 1.0, 0.5)}[method]
        data = build_csr(db, bl['run_code'], bl['fx'], int(bl['tau_u']), int(bl['tau_s']),
                         rho=0, m=m)
        t1 = time.perf_counter()
        n_s, n_u = data.n_s, data.n_u
        mu = None
        res = rank(data, m, chi, alpha=alpha, mu=mu)
        info = dict(units=(n_s if m[0] or m[1] else 0) + (n_u if m[3] or m[1] else 0),
                    iters=int(res.iters))
    t2 = time.perf_counter()
    db.close()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2   # KB -> GB
    return dict(method=method, assemble_s=round(t1 - t0, 2), solve_s=round(t2 - t1, 3),
                peak_rss_gb=round(peak, 2), **info)


def main():
    from util import load_config
    paths = load_config()
    rows = []
    for m in METHODS:
        env = os.environ.copy()
        env.update({k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
        out = subprocess.run([sys.executable, __file__, '--method', m], capture_output=True,
                             text=True, env=env)
        if out.returncode != 0:
            print(out.stderr[-2000:])
            raise SystemExit(f'{m} failed')
        r = json.loads(out.stdout.strip().splitlines()[-1])
        r['label'] = LABELS[m]
        rows.append(r)
        print(r, flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    cpu = platform.machine()
    if Path('/proc/cpuinfo').exists():
        cpu = next((l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines()
                    if l.startswith('model name')), cpu)
    df['machine'] = f"{cpu}, Python {platform.python_version()}"
    df.to_csv(paths.data / 'computational_cost.csv', index=False)
    print(df.to_string(index=False))
    print(f"written {paths.data / 'computational_cost.csv'}")


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--method':
        print(json.dumps(run_method(sys.argv[2])))
    else:
        main()
