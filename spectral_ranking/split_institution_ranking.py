"""
split_institution_ranking.py — Bipartite ranking with institution nodes split by field.

QSS revision, referee 1 ("cross-subfield leakage"). In the baseline bipartite
model an institution is one node: influence it receives for works in any field
is pooled and passed on through the reference lists of all its works. Here each
institution u is replaced by up to four nodes (u, F), F in {E, B, A, X}, where F
is the field_eb of the source in which the institution's work appeared. A node
(u, F) receives citations only to u's F-source works and passes influence on only
through the reference lists of those works. Cross-field references made by a
work are kept; only the pooling of fields inside an institution is removed.

The split is a pure regrouping: summing the split C_SI / C_IS blocks over F
reproduces the baseline blocks exactly, and summing split work counts over F
reproduces the baseline institution work counts. Both are asserted at run time.

Split-node v values are not reported (small nodes have inflated v = pi/a).
Institutions are reported recombined: v_u = A * sum_F pi_uF / (pi_real * sum_F a_uF).

Reads (read-only): WORKING/edge_lists.duckdb, WORKING/rankings.duckdb,
                   PARQUET corpus_works / corpus_authorships / source_master.
Writes:            WORKING/revision.duckdb (never the baseline databases).

Usage:
  .venv/bin/python spectral_ranking/split_institution_ranking.py
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).parent))          # spectral_ranking/
sys.path.insert(0, str(Path(__file__).parent.parent))   # project root

from util import load_config, load_runs
from build_csr import build_csr, CSRData
from katz_ranker import rank

FCODE = {'E': 1, 'B': 2, 'A': 3, 'X': 4}
FNAME = {v: k for k, v in FCODE.items()}
M_BIP = (0, 1, 1, 0)
CHI = 0.5
N_CLIQUE = 10


# ── Split blocks ────────────────────────────────────────────────────────────────

def build_split_blocks(el_db, tname: str, base: CSRData, fmap: pd.DataFrame,
                       r_bar: float):
    """Return (C_SI_split, C_IS_split, node_ids) with node id = inst_idx*10 + fcode."""
    el_db.register('_fmap', fmap)
    el_db.execute(f"""
        CREATE OR REPLACE TEMP TABLE _tmp_el AS
        SELECT citer_work_idx, citer_source_idx, citer_inst_idx,
               cited_work_idx,  cited_source_idx, cited_inst_idx,
               inst_weight, cited_inst_weight,
               {r_bar} / CAST(R_i AS DOUBLE) AS rho_w
        FROM {tname}
    """)
    si = el_db.execute("""
        SELECT e.citer_source_idx, e.cited_inst_idx * 10 + f.fcode AS node,
               SUM(e.rho_w * e.cited_inst_weight) AS w
        FROM (SELECT DISTINCT citer_work_idx, citer_source_idx,
                              cited_work_idx, cited_source_idx, cited_inst_idx,
                              rho_w, cited_inst_weight
              FROM _tmp_el) e
        JOIN _fmap f ON e.cited_source_idx = f.source_idx
        GROUP BY 1, 2
    """).fetchdf()
    is_ = el_db.execute("""
        SELECT e.citer_inst_idx * 10 + f.fcode AS node, e.cited_source_idx,
               SUM(e.rho_w * e.inst_weight) AS w
        FROM (SELECT DISTINCT citer_work_idx, citer_source_idx, citer_inst_idx,
                              cited_work_idx, cited_source_idx,
                              rho_w, inst_weight
              FROM _tmp_el) e
        JOIN _fmap f ON e.citer_source_idx = f.source_idx
        GROUP BY 1, 2
    """).fetchdf()
    el_db.execute("DROP TABLE IF EXISTS _tmp_el")
    el_db.unregister('_fmap')

    # restrict to the baseline unit set, exactly as build_csr does
    inst_set = set(base.inst_ids.tolist())
    si = si[si['node'].floordiv(10).isin(inst_set)]
    is_ = is_[is_['node'].floordiv(10).isin(inst_set)]
    node_ids = np.array(sorted(set(si['node']) | set(is_['node'])), dtype=np.int64)

    s_ix, n_ix = pd.Index(base.source_ids), pd.Index(node_ids)
    r, c = s_ix.get_indexer(si['citer_source_idx']), n_ix.get_indexer(si['node'])
    k = r >= 0
    C_SI = sp.coo_matrix((si['w'].to_numpy()[k], (r[k], c[k])),
                         shape=(base.n_s, len(node_ids))).tocsr()
    r, c = n_ix.get_indexer(is_['node']), s_ix.get_indexer(is_['cited_source_idx'])
    k = c >= 0
    C_IS = sp.coo_matrix((is_['w'].to_numpy()[k], (r[k], c[k])),
                         shape=(len(node_ids), base.n_s)).tocsr()
    return C_SI, C_IS, node_ids


def split_work_counts(parquet: Path, fx: str, years: tuple, fmap: pd.DataFrame,
                      node_ids: np.ndarray) -> np.ndarray:
    """Author-fractional work count per (institution, field) node, as in build_edge_lists."""
    letters = [c for c in 'EBAX' if c in fx]
    fcond = '' if len(letters) == 4 else \
        "AND sm.field_eb IN (" + ', '.join(f"'{c}'" for c in letters) + ")"
    con = duckdb.connect()
    con.register('_fmap', fmap)
    df = con.execute(f"""
        WITH fw AS (
            SELECT w.work_idx, w.source_idx
            FROM '{parquet}/corpus_works.parquet' w
            JOIN '{parquet}/source_master.parquet' sm ON w.source_idx = sm.source_idx
            WHERE w.publication_year BETWEEN {years[0]} AND {years[1]} {fcond}
        ),
        auths AS (
            SELECT DISTINCT work_idx, author_idx, institution_idx
            FROM '{parquet}/corpus_authorships.parquet'
            WHERE institution_idx IS NOT NULL
              AND work_idx IN (SELECT work_idx FROM fw)
        ),
        wac AS (SELECT work_idx, COUNT(DISTINCT author_idx) AS n_authors
                FROM auths GROUP BY work_idx),
        aic AS (SELECT work_idx, author_idx, COUNT(DISTINCT institution_idx) AS n_ipa
                FROM auths GROUP BY work_idx, author_idx),
        iw AS (
            SELECT a.work_idx, a.institution_idx,
                   SUM(1.0 / wac.n_authors / aic.n_ipa) AS w
            FROM auths a
            JOIN wac ON a.work_idx = wac.work_idx
            JOIN aic ON a.work_idx = aic.work_idx AND a.author_idx = aic.author_idx
            GROUP BY a.work_idx, a.institution_idx
        )
        SELECT iw.institution_idx * 10 + f.fcode AS node, SUM(iw.w) AS a
        FROM iw
        JOIN fw ON iw.work_idx = fw.work_idx
        JOIN _fmap f ON fw.source_idx = f.source_idx
        GROUP BY 1
    """).fetchdf()
    con.close()
    a = pd.Series(df['a'].to_numpy(), index=df['node']).reindex(node_ids)
    return a.to_numpy(dtype=np.float64)


def check_conservation(base: CSRData, C_SI, C_IS, node_ids, all_nodes, a_all):
    """Split blocks and work counts must sum back to the baseline exactly.

    Work counts are checked over all (institution, field) combinations, because
    a field node with works but no references in the window has no edges and is
    absent from node_ids; its works still belong to the institution."""
    inst_of = pd.Index(base.inst_ids).get_indexer(node_ids // 10)
    assert (inst_of >= 0).all()
    agg = sp.csr_matrix((np.ones(len(node_ids)), (np.arange(len(node_ids)), inst_of)),
                        shape=(len(node_ids), base.n_u))
    d_si = abs(C_SI @ agg - base.C_SI).max()
    d_is = abs(agg.T @ C_IS - base.C_IS).max()
    inst_all = pd.Index(base.inst_ids).get_indexer(all_nodes // 10)
    a_sum = np.bincount(inst_all, weights=np.nan_to_num(a_all), minlength=base.n_u)
    d_a = np.max(np.abs(a_sum - base.a_u) / base.a_u)
    print(f"    conservation: max|dC_SI|={d_si:.2e}  max|dC_IS|={d_is:.2e}  "
          f"max rel|da|={d_a:.2e}")
    assert d_si < 1e-8 * base.C_SI.max() and d_is < 1e-8 * base.C_IS.max(), \
        "split blocks do not sum back to the baseline"
    assert d_a < 1e-9, "split work counts do not sum back to baseline a_p"


def prune(C_SI, C_IS, src_ids, a_s, node_ids, a_n):
    """Drop dangling/unreferenced units and keep the largest SCC, until stable."""
    while True:
        n0 = (len(src_ids), len(node_ids))
        ks = (np.asarray(C_SI.sum(axis=1)).ravel() > 0) & (np.asarray(C_IS.sum(axis=0)).ravel() > 0)
        kn = (np.asarray(C_IS.sum(axis=1)).ravel() > 0) & (np.asarray(C_SI.sum(axis=0)).ravel() > 0)
        C_SI, C_IS = C_SI[ks][:, kn], C_IS[kn][:, ks]
        src_ids, a_s, node_ids, a_n = src_ids[ks], a_s[ks], node_ids[kn], a_n[kn]
        G = sp.bmat([[None, C_SI], [C_IS, None]], format='csr')
        _, lab = connected_components(G, directed=True, connection='strong')
        big = lab == np.bincount(lab).argmax()
        ks, kn = big[:len(src_ids)], big[len(src_ids):]
        C_SI, C_IS = C_SI[ks][:, kn], C_IS[kn][:, ks]
        src_ids, a_s, node_ids, a_n = src_ids[ks], a_s[ks], node_ids[kn], a_n[kn]
        if (len(src_ids), len(node_ids)) == n0:
            return C_SI, C_IS, src_ids, a_s, node_ids, a_n


def ranked_split(base, C_SI, C_IS, node_ids, a_nodes, excl_src=()):
    """Rank the split network (optionally without some sources).

    v is normalised exactly as in the baseline: A is the total work count of the
    retained sources plus the full (all-field) work count of every retained
    institution, so an institution keeps its works in fields whose split node was
    pruned or had no edges. If the split changed nothing, the baseline v would be
    reproduced exactly."""
    src_ids, a_s = base.source_ids.copy(), base.a_s.copy()
    if excl_src:
        k = ~np.isin(src_ids, list(excl_src))
        C_SI, C_IS, src_ids, a_s = C_SI[k], C_IS[:, k], src_ids[k], a_s[k]
    C_SI, C_IS, src_ids, a_s, node_ids, a_nodes = prune(C_SI, C_IS, src_ids, a_s,
                                                        node_ids, a_nodes)
    data = CSRData(C_SS=None, C_SI=C_SI, C_IS=C_IS, C_II=None,
                   source_ids=src_ids, inst_ids=node_ids, a_s=a_s, a_u=a_nodes,
                   n_s=len(src_ids), n_u=len(node_ids))
    res = rank(data, M_BIP, CHI, alpha=1.0, mu=None)
    a_inst_full = pd.Series(base.a_u, index=base.inst_ids)
    kept_inst = np.unique(node_ids // 10)
    A = a_s.sum() + a_inst_full.reindex(kept_inst).sum()
    pi_real = res.pi_s.sum() + res.pi_u.sum()
    v_s = pd.Series(A * res.pi_s / (pi_real * a_s), index=src_ids)
    # recombine institutions: v_u = A * sum_F pi_uF / (pi_real * a_u)
    pi_inst = pd.Series(res.pi_u).groupby(node_ids // 10).sum()
    v_inst = A * pi_inst / (pi_real * a_inst_full.reindex(pi_inst.index))
    nodes = pd.DataFrame({'node': node_ids, 'inst': node_ids // 10,
                          'field': [FNAME[x] for x in node_ids % 10],
                          'pi': res.pi_u, 'a_p': a_nodes, 'v': res.v_u})
    info = dict(n_s=len(src_ids), n_nodes=len(node_ids), lam2=res.lam2,
                clique_pi=None, pi_s=pd.Series(res.pi_s, index=src_ids))
    return v_s, v_inst, nodes, info


# ── Comparison helpers ──────────────────────────────────────────────────────────

def loglog(x: pd.Series, y: pd.Series):
    d = pd.concat([x, y], axis=1, keys=['x', 'y']).dropna()
    d = d[(d.x > 0) & (d.y > 0)]
    lx, ly = np.log10(d.x), np.log10(d.y)
    b, a = np.polyfit(lx, ly, 1)
    return dict(n=len(d), slope=b, offset=a, scatter=float((ly - (a + b * lx)).std()),
                median_shift=float(np.median(ly - lx)))


def fmt(d):
    return (f"n={d['n']:5d} slope={d['slope']:.3f} offset={d['offset']:+.3f} "
            f"scatter={d['scatter']:.3f} median_shift={d['median_shift']:+.3f}")


def top_mix(v: pd.Series, field: dict, k=20, drop=()):
    t = v.drop(list(drop), errors='ignore').sort_values(ascending=False).head(k)
    f = [field.get(i) for i in t.index]
    return {c: f.count(c) for c in 'EBAX' if f.count(c)}


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    paths = load_config()
    bl = next(r for r in load_runs() if r['label'] == 'baseline')
    rc, tu, ts = bl['run_code'], bl['tau_u'], bl['tau_s']
    years = (int(bl['tc0']), int(bl['tc1']))

    el_db = duckdb.connect(str(paths.working / 'edge_lists.duckdb'), read_only=True)
    rk_db = duckdb.connect(str(paths.working / 'rankings.duckdb'), read_only=True)
    out = duckdb.connect(str(paths.working / 'revision.duckdb'))

    sm = pd.read_parquet(str(paths.parquet / 'source_master.parquet'),
                         columns=['source_idx', 'source_name', 'field_eb'])
    # one source has a NULL label; treat it as X, as build_institution_field_eb does
    sm['field_eb'] = sm['field_eb'].fillna('X')
    field = dict(zip(sm.source_idx, sm.field_eb))
    fmap = pd.DataFrame({'source_idx': sm.source_idx.astype('int64'),
                         'fcode': sm.field_eb.map(FCODE).astype('int64')})

    def rk(fx, m):
        t = f"rk_{rc}_{fx}_tauU{tu}_tauS{ts}_vartau_rho0_m{m}_chi50_alpha100"
        d = rk_db.execute(f"SELECT unit_idx, unit_type, v FROM {t}").fetchdf()
        return (d[d.unit_type == 'S'].set_index('unit_idx')['v'],
                d[d.unit_type == 'U'].set_index('unit_idx')['v'])

    summary = []
    for fx in ['EBAX', 'EB']:
        print(f"\n=== corpus F={fx} ===", flush=True)
        tname = f"el_{rc}_{fx}_tauU{tu}_tauS{ts}_vartau"
        base = build_csr(el_db, rc, fx, tu, ts, rho=0, m=M_BIP)
        r_bar = el_db.execute(
            f"SELECT AVG(CAST(R_i AS DOUBLE)) FROM "
            f"(SELECT DISTINCT citer_work_idx, R_i FROM {tname})").fetchone()[0]
        C_SI, C_IS, node_ids = build_split_blocks(el_db, tname, base, fmap, r_bar)
        all_nodes = np.array(sorted(i * 10 + f for i in base.inst_ids.tolist()
                                    for f in FCODE.values()), dtype=np.int64)
        a_all = split_work_counts(paths.parquet, fx, years, fmap, all_nodes)
        a_nodes = pd.Series(a_all, index=all_nodes).reindex(node_ids).to_numpy()
        print(f"    {base.n_u} institutions -> {len(node_ids)} split nodes with edges "
              f"({np.isnan(a_nodes).sum()} with no works in window)")
        check_conservation(base, C_SI, C_IS, node_ids, all_nodes, a_all)
        # nodes with references but no works in the window cannot carry v; give them
        # a nominal tiny count so they act only as conduits (none expected)
        a_nodes = np.where(np.isnan(a_nodes) | (a_nodes <= 0), 1e-9, a_nodes)

        # baseline (whole institutions), recomputed here for an exact like-for-like
        res0 = rank(base, M_BIP, CHI, alpha=1.0, mu=None)
        v0_s = pd.Series(res0.v_s, index=base.source_ids)
        v0_u = pd.Series(res0.v_u, index=base.inst_ids)
        stored_s, _ = rk(fx, '0110')
        print(f"    baseline recompute vs stored table: "
              f"max|dlog v_s|={np.abs(np.log10(v0_s / stored_s.reindex(v0_s.index))).max():.2e}")

        v1_s, v1_u, nodes, info = ranked_split(base, C_SI, C_IS, node_ids, a_nodes)
        print(f"    split network after pruning: n_s={info['n_s']} nodes={info['n_nodes']}  "
              f"|lam2|={abs(info['lam2']):.4f}")

        # 1. journals: split vs whole, per field
        for f in 'EBAX':
            ids = [i for i in v0_s.index if field.get(i) == f]
            if len(ids) > 5:
                d = loglog(v0_s.reindex(ids), v1_s.reindex(ids))
                print(f"    journals {f}: split vs whole  {fmt(d)}")
                summary.append(dict(corpus=fx, test='journal split vs whole', field=f, **d))
        # 2. institutions recombined vs whole
        d = loglog(v0_u, v1_u)
        print(f"    institutions recombined vs whole  {fmt(d)}")
        summary.append(dict(corpus=fx, test='institution split vs whole', field='all', **d))

        # 3. clique: top-10 E journals by whole-institution bipartite v
        clique = [i for i in v0_s.sort_values(ascending=False).index
                  if field.get(i) == 'E'][:N_CLIQUE]
        pi0 = pd.Series(res0.pi_s, index=base.source_ids)
        share0 = pi0.reindex(clique).sum() / pi0.sum()
        share1 = info['pi_s'].reindex(clique).sum() / info['pi_s'].sum()
        print(f"    clique share of source pi: whole={share0:.3f}  split={share1:.3f}")
        summary.append(dict(corpus=fx, test='clique share of source pi', field='E',
                            n=N_CLIQUE, slope=np.nan, offset=share0, scatter=share1,
                            median_shift=np.nan))
        print(f"    non-clique top-20 mix: whole={top_mix(v0_s, field, drop=clique)}  "
              f"split={top_mix(v1_s, field, drop=clique)}")

        # 4. containment: remove the clique in the split model
        v2_s, _, _, _ = ranked_split(base, C_SI, C_IS, node_ids, a_nodes,
                                     excl_src=set(clique))
        for f in 'EB':
            ids = [i for i in v1_s.index if field.get(i) == f and i not in clique]
            d = loglog(v1_s.reindex(ids), v2_s.reindex(ids))
            print(f"    split model, clique removed, {f}: {fmt(d)}")
            summary.append(dict(corpus=fx, test='split: clique removed vs present',
                                field=f, **d))
        print(f"    split model non-clique top-20 mix after removal: {top_mix(v2_s, field)}")

        # 5. common E/B scale (EB corpus only): combined vs separate-field runs.
        #    In a single-field corpus the split changes nothing, so the stored
        #    F=E and F=B bipartite runs are the separate rankings.
        if fx == 'EB':
            for f in 'EB':
                sep, _ = rk(f, '0110')
                ids = [i for i in sep.index if field.get(i) == f]
                d0 = loglog(v0_s.reindex(ids), sep.reindex(ids))
                d1 = loglog(v1_s.reindex(ids), sep.reindex(ids))
                print(f"    separate {f} vs combined: whole {fmt(d0)}")
                print(f"    separate {f} vs combined: split {fmt(d1)}")
                summary.append(dict(corpus=fx, test='separate vs combined (whole)', field=f, **d0))
                summary.append(dict(corpus=fx, test='separate vs combined (split)', field=f, **d1))

        # write results
        tag = f"{rc}_{fx}_tauU{tu}_tauS{ts}_rho0_m0110_chi50_alpha100"
        src_df = pd.DataFrame({'unit_idx': v0_s.index, 'v_whole': v0_s.values,
                               'v_split': v1_s.reindex(v0_s.index).values,
                               'field_eb': [field.get(i) for i in v0_s.index]})
        inst_df = pd.DataFrame({'unit_idx': v0_u.index, 'v_whole': v0_u.values,
                                'v_split': v1_u.reindex(v0_u.index).values})
        out.register('src_df', src_df)
        out.register('inst_df', inst_df)
        out.register('nodes_df', nodes)
        out.execute(f"CREATE OR REPLACE TABLE instsplit_sources_{tag} AS SELECT * FROM src_df")
        out.execute(f"CREATE OR REPLACE TABLE instsplit_institutions_{tag} AS SELECT * FROM inst_df")
        out.execute(f"CREATE OR REPLACE TABLE instsplit_nodes_{tag} AS SELECT * FROM nodes_df")

    s = pd.DataFrame(summary)
    out.register('s', s)
    out.execute("CREATE OR REPLACE TABLE instsplit_summary AS SELECT * FROM s")
    out.close()
    print("\nwritten to", paths.working / 'revision.duckdb')


if __name__ == '__main__':
    main()
