"""
mncs_comparison.py — Compare the combined-baseline influence v with a Leiden-style MNCS.

QSS revision (referees 1.1, 2.5, 3.0). The comparison is cardinal and band-based, not
ordinal: both v and MNCS are per-work ratio scales with 1 = average, computed on the
same corpus and the same references, so log v and log MNCS are compared directly.

MNCS (mean normalised citation score)
  population  corpus works 2020-24 holding at least one baseline institution
  citations   c_w = distinct citing works among the baseline edge-list references
              (the references v uses; self-citations included, as in v)
  expected    e = mean c over population works of the same field (journal field_eb:
              E, B, A, X), publication year and document type
  journal     MNCS_j = mean of c_w / e over the journal's population works
  institution MNCS_u = sum_w ω_wu (c_w / e) / sum_w ω_wu, ω = author-fractional
              weights (as a_p and the baseline, ω = 0)
  The fields are separated only in the expected value: citations from every corpus
  work count, which is how MNCS field normalisation works.

Paired bootstrap
  The resample replicates of bootstrap_oa_errors.py (stage4_resample) redraw the
  citing works: a work drawn k times contributes its references k times. This script
  regenerates the identical draws (same seeds) and recomputes MNCS on each, so for
  every unit there is a 95% interval of Δ = log10 v − log10 MNCS over the same 1,000
  resampled corpora.
  in band   the Δ interval contains 0
  near      it excludes 0 by less than its own width
  far       otherwise

Citer standing (the explanation v offers for a gap): the reference-weighted mean
log10 v of the citing units — citing institutions for a journal, citing journals for
an institution (the bipartite route) — minus the same mean over all references.

Outputs
  data/mncs_comparison_summary.csv                 by unit type and field
  data/mncs_comparison_by_rank.csv                 by unit type and v-rank band
  WORKING/benchmarks/mncs_comparison_units.csv     per unit

Usage:
  ECONBUS_CONFIG=config.rev.yaml .venv/bin/python spectral_results_analysis/mncs_comparison.py
"""

import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs
from spectral_ranking_bootstrap.bootstrap_oa_errors import (
    YEAR_LO, YEAR_HI, STAGE_DIR, _institution_weights)

RESAMPLE_SEED_OFFSET = 30_000     # as in bootstrap_oa_errors.build_rankings


# ── core computations (pure; tested) ──────────────────────────────────────────

def expected_by_group(c: np.ndarray, group: np.ndarray, n_groups: int) -> np.ndarray:
    """Mean of c within each group code; returns the per-work expected value."""
    s = np.bincount(group, weights=c, minlength=n_groups)
    n = np.bincount(group, minlength=n_groups)
    mean = np.divide(s, n, out=np.zeros(n_groups), where=n > 0)
    return mean[group]


def normalised_scores(c: np.ndarray, group: np.ndarray, n_groups: int) -> np.ndarray:
    """c / e with e the group mean; works in a group with no citations score 0."""
    e = expected_by_group(c, group, n_groups)
    return np.divide(c, e, out=np.zeros_like(c, dtype=float), where=e > 0)


def unit_means(ncs: np.ndarray, W: sp.csr_matrix) -> np.ndarray:
    """Weighted mean of ncs per unit; W is (units × works) with the work weights."""
    num = W @ ncs
    den = np.asarray(W.sum(axis=1)).ravel()
    return np.divide(num, den, out=np.full(len(den), np.nan), where=den > 0)


def citation_counts(cited_pos: np.ndarray, citer_mult: np.ndarray, n_works: int) -> np.ndarray:
    """c_w = sum over references to w of the citing work's multiplicity."""
    return np.bincount(cited_pos, weights=citer_mult, minlength=n_works)


def classify(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """'in' if 0 ∈ [lo, hi]; 'near' if it misses 0 by ≤ its width; else 'far'."""
    width = hi - lo
    gap = np.where(lo > 0, lo, np.where(hi < 0, -hi, 0.0))
    out = np.where(gap == 0, 'in', np.where(gap <= width, 'near', 'far'))
    return np.where(np.isnan(lo) | np.isnan(hi), 'n/a', out)


# ── data ──────────────────────────────────────────────────────────────────────

def load_inputs(paths):
    bl = next(r for r in load_runs() if r['label'] == 'baseline')
    rc, fx, tu, ts = bl['run_code'], bl['fx'], bl['tau_u'], bl['tau_s']
    el_table = f'el_{rc}_{fx}_tauU{tu}_tauS{ts}_vartau'
    units_table = f'_units_{rc}_{fx}_tauU{tu}_tauS{ts}_vartau_m0110'
    rk_table = f'rk_{rc}_{fx}_tauU{tu}_tauS{ts}_vartau_rho0_m0110_chi50_alpha100'
    pq = paths.parquet
    db = duckdb.connect(str(paths.working / 'edge_lists.duckdb'), read_only=True)
    units = db.execute(f'SELECT unit_idx, unit_type FROM "{units_table}"').fetchdf()
    refs = db.execute(f'SELECT DISTINCT citer_work_idx, cited_work_idx, R_i FROM {el_table}').fetchdf()
    db.close()
    inst_ids = units.loc[units.unit_type == 'U', 'unit_idx'].to_numpy(np.int64)
    src_ids = units.loc[units.unit_type == 'S', 'unit_idx'].to_numpy(np.int64)

    con = duckdb.connect()
    con.register('ui', pd.DataFrame({'i': inst_ids}))
    works = con.execute(f"""
        SELECT w.work_idx, w.source_idx, w.publication_year AS year, w.type,
               coalesce(sm.field_eb, 'X') AS field
        FROM '{pq}/corpus_works.parquet' w
        JOIN '{pq}/source_master.parquet' sm ON w.source_idx = sm.source_idx
        WHERE w.publication_year BETWEEN {YEAR_LO} AND {YEAR_HI}
          AND w.work_idx IN (SELECT DISTINCT work_idx FROM '{pq}/corpus_authorships.parquet'
                             WHERE institution_idx IN (SELECT i FROM ui))
        ORDER BY w.work_idx
    """).fetchdf()
    iw = _institution_weights(paths, works['work_idx'].to_numpy(np.int64), omega=0)
    iw = iw[iw['inst_idx'].isin(inst_ids)]

    rk = duckdb.connect(str(paths.working / 'rankings.duckdb'), read_only=True)
    v = rk.execute(f'SELECT unit_idx, unit_type, v FROM "{rk_table}"').fetchdf()
    names_s = pd.read_parquet(pq / 'source_master.parquet', columns=['source_idx', 'source_name', 'field_eb'])
    ci = pd.read_parquet(pq / 'corpus_institutions.parquet')
    ncol = 'display_name' if 'display_name' in ci.columns else [c for c in ci.columns if 'name' in c][0]
    ife = pd.read_parquet(pq / 'institution_field_eb.parquet')
    return dict(works=works, iw=iw, refs=refs, src_ids=src_ids, inst_ids=inst_ids, v=v,
                names_s=names_s, names_i=ci.set_index('institution_idx')[ncol],
                inst_field=ife.set_index('unit_idx')['field_eb'], rk_table=rk_table)


def main():
    t0 = time.perf_counter()
    paths = load_config()
    d = load_inputs(paths)
    works, iw, refs = d['works'], d['iw'], d['refs']
    W_ids = works['work_idx'].to_numpy(np.int64)
    n_w = len(W_ids)
    grp = pd.factorize(works['field'] + '|' + works['year'].astype(str) + '|' + works['type'].astype(str))[0]
    n_g = grp.max() + 1

    # references whose cited work is in the population (all, by construction of the edge list)
    cited_pos = np.searchsorted(W_ids, refs['cited_work_idx'].to_numpy(np.int64))
    ok = (cited_pos < n_w) & (W_ids[np.minimum(cited_pos, n_w - 1)] == refs['cited_work_idx'].to_numpy())
    print(f'population works {n_w:,}; edge-list references {len(refs):,} '
          f'({ok.mean():.2%} cite population works)', flush=True)
    refs = refs[ok]; cited_pos = cited_pos[ok]

    # unit × work weight matrices
    s_index = pd.Index(d['src_ids']); u_index = pd.Index(d['inst_ids'])
    sj = s_index.get_indexer(works['source_idx'])
    m = sj >= 0
    W_s = sp.csr_matrix((np.ones(m.sum()), (sj[m], np.nonzero(m)[0])), shape=(len(s_index), n_w))
    wpos = np.searchsorted(W_ids, iw['work_idx'].to_numpy(np.int64))
    W_u = sp.csr_matrix((iw['inst_weight'].to_numpy(float), (u_index.get_indexer(iw['inst_idx']), wpos)),
                        shape=(len(u_index), n_w))

    # baseline MNCS
    c0 = citation_counts(cited_pos, np.ones(len(refs)), n_w)
    ncs0 = normalised_scores(c0, grp, n_g)
    mncs0 = {'S': unit_means(ncs0, W_s), 'U': unit_means(ncs0, W_u)}

    # paired replicates: identical draws to stage4_resample
    boot = paths.working / STAGE_DIR / 'stage4_resample'
    meta = json.loads((boot / 'meta.json').read_text())
    assert meta.get('omega', 1) == 0 and meta.get('run_label', 'baseline') == 'baseline'
    B, seed = int(meta['n']), int(meta['seed'])
    assert meta['source_ids'] == d['src_ids'].tolist() or set(meta['source_ids']) == set(d['src_ids'])
    base_works = pd.read_parquet(paths.working / STAGE_DIR / 'stage1_base_works.parquet')
    in_win_ids = (base_works[base_works['pub_year'].between(YEAR_LO, YEAR_HI)]
                  ['work_idx'].unique().astype(np.int64))
    order = np.argsort(in_win_ids); sorted_ids = in_win_ids[order]
    cpos = np.searchsorted(sorted_ids, refs['citer_work_idx'].to_numpy(np.int64))
    assert (sorted_ids[cpos] == refs['citer_work_idx'].to_numpy()).all(), 'citer outside resample pool'
    citer_slot = order[cpos]                      # position of each citer in in_win_ids
    n = len(in_win_ids)
    # pairing check against resample_works_one for the first two replicates
    from spectral_ranking_bootstrap.bootstrap_oa_errors import resample_works_one
    for b in range(2):
        ref_mult = resample_works_one(in_win_ids, np.random.default_rng(seed + RESAMPLE_SEED_OFFSET + b))
        cnt = np.bincount(np.random.default_rng(seed + RESAMPLE_SEED_OFFSET + b).integers(0, n, size=n),
                          minlength=n)
        assert {int(in_win_ids[i]): int(k) for i, k in enumerate(cnt) if k} == ref_mult

    mncs_b = {k: np.full((B, len(mncs0[k])), np.nan) for k in 'SU'}
    for b in range(B):
        cnt = np.bincount(np.random.default_rng(seed + RESAMPLE_SEED_OFFSET + b).integers(0, n, size=n),
                          minlength=n).astype(float)
        cb = citation_counts(cited_pos, cnt[citer_slot], n_w)
        ncs = normalised_scores(cb, grp, n_g)
        mncs_b['S'][b] = unit_means(ncs, W_s)
        mncs_b['U'][b] = unit_means(ncs, W_u)
    print(f'MNCS replicates: {B} in {time.perf_counter() - t0:.0f}s', flush=True)

    # citer standing: reference-weighted mean log v of citing units (bipartite route)
    v = d['v']
    lv_s = np.log10(v[v.unit_type == 'S'].set_index('unit_idx').v)
    lv_u = np.log10(v[v.unit_type == 'U'].set_index('unit_idx').v)
    rho = refs['R_i'].astype(float).to_numpy()
    rho = rho.mean() / rho          # ρ = 0 weighting, approximate R̄ (per reference)
    src_of = works.set_index('work_idx')['source_idx']
    r = pd.DataFrame({'citer': refs['citer_work_idx'].to_numpy(), 'cited': refs['cited_work_idx'].to_numpy(),
                      'w': rho})
    # journals: standing of citing institutions
    ri = r.merge(iw.rename(columns={'work_idx': 'citer', 'inst_idx': 'ci', 'inst_weight': 'cw'})[['citer', 'ci', 'cw']],
                 on='citer')
    ri['lv'] = ri['ci'].map(lv_u); ri = ri.dropna(subset=['lv'])
    ri['src'] = ri['cited'].map(src_of)
    ri['ww'] = ri['w'] * ri['cw']
    st_s = (ri.assign(x=ri.ww * ri.lv).groupby('src')[['x', 'ww']].sum())
    stand_s = st_s.x / st_s.ww - (ri.ww * ri.lv).sum() / ri.ww.sum()
    # institutions: standing of citing journals
    rs = r.copy()
    rs['lv'] = rs['citer'].map(src_of).map(lv_s); rs = rs.dropna(subset=['lv'])
    rs = rs.merge(iw.rename(columns={'work_idx': 'cited', 'inst_idx': 'u', 'inst_weight': 'uw'})[['cited', 'u', 'uw']],
                  on='cited')
    rs['ww'] = rs['w'] * rs['uw']
    st_u = rs.assign(x=rs.ww * rs.lv).groupby('u')[['x', 'ww']].sum()
    stand_u = st_u.x / st_u.ww - (rs.ww * rs.lv).sum() / rs.ww.sum()

    # Δ intervals and classification
    rows = []
    for ut, key, f_boot, ids, stand in (
            ('S', 'source_ids', 'v_s_boot.npy', d['src_ids'], stand_s),
            ('U', 'inst_ids', 'v_u_boot.npy', d['inst_ids'], stand_u)):
        vb = np.load(boot / f_boot).astype(float)
        pos = pd.Index(meta[key]).get_indexer(ids)
        vb = vb[:, pos]
        v0 = v[v.unit_type == ut].set_index('unit_idx').v.reindex(ids).to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            delta_b = np.log10(vb) - np.log10(mncs_b[ut])
        delta_b[~np.isfinite(delta_b)] = np.nan
        lo, hi = np.nanpercentile(delta_b, [2.5, 97.5], axis=0)
        n_ok = np.isfinite(delta_b).sum(0)
        delta0 = np.log10(v0) - np.log10(mncs0[ut])
        cls = classify(lo, hi)
        if ut == 'S':
            nm = d['names_s'].set_index('source_idx')
            names, field = nm.source_name.reindex(ids).to_numpy(), nm.field_eb.reindex(ids).fillna('X').to_numpy()
            n_works = np.asarray(W_s.sum(axis=1)).ravel()
        else:
            names = d['names_i'].reindex(ids).to_numpy()
            field = d['inst_field'].reindex(ids).fillna('X').to_numpy()
            n_works = np.asarray(W_u.sum(axis=1)).ravel()
        rows.append(pd.DataFrame({
            'unit_type': ut, 'unit_idx': ids, 'name': names, 'field': field, 'n_works': n_works,
            'v': v0, 'mncs': mncs0[ut], 'delta': delta0, 'delta_lo': lo, 'delta_hi': hi,
            'n_reps': n_ok, 'class': cls, 'citer_standing': stand.reindex(ids).to_numpy()}))
    units = pd.concat(rows, ignore_index=True)
    units.loc[units.n_reps < 0.95 * B, 'class'] = units.loc[units.n_reps < 0.95 * B, 'class'] + '*'

    # summary
    summ = []
    for (ut, f), g in [((ut, 'all'), g) for ut, g in units.groupby('unit_type')] + \
                      list(units.groupby(['unit_type', 'field'])):
        lm, lv_ = np.log10(g.mncs), np.log10(g.v)
        ok = np.isfinite(lm) & np.isfinite(lv_)
        r = np.corrcoef(lm[ok], lv_[ok])[0, 1] if ok.sum() > 2 else np.nan
        slope = np.sign(r) * np.std(lv_[ok]) / np.std(lm[ok]) if ok.sum() > 2 else np.nan
        base_cls = g['class'].str.rstrip('*')
        far = g[base_cls == 'far']
        sign_ok = (np.sign(far.delta) == np.sign(far.citer_standing)).mean() if len(far) else np.nan
        summ.append(dict(unit_type={'S': 'journals', 'U': 'institutions'}[ut], field=f, n=len(g),
                         in_band=int((base_cls == 'in').sum()), near=int((base_cls == 'near').sum()),
                         far=int((base_cls == 'far').sum()), flagged_dropouts=int(g['class'].str.endswith('*').sum()),
                         median_delta=round(float(np.nanmedian(g.delta)), 3),
                         sd_delta=round(float(np.nanstd(g.delta)), 3),
                         corr_logv_logmncs=round(float(r), 3),
                         rma_slope_logv_vs_logmncs=round(float(slope), 3),
                         corr_delta_citer_standing=round(float(g[['delta', 'citer_standing']].corr().iloc[0, 1]), 3),
                         far_sign_matches_standing=round(float(sign_ok), 3) if len(far) else np.nan))
    summ = pd.DataFrame(summ)
    out = paths.data / 'mncs_comparison_summary.csv'
    summ.to_csv(out, index=False)

    # by position in the ranking: v stretches the MNCS scale at both ends
    by_rank = []
    for ut, g in units.groupby('unit_type'):
        rank = g.v.rank(ascending=False)
        bins = pd.cut(rank, [0, 10, 100, 500, 1000, np.inf],
                      labels=['1-10', '11-100', '101-500', '501-1000', '1001+'])
        for lab, h in g.groupby(bins, observed=True):
            c = h['class'].str.rstrip('*')
            by_rank.append(dict(unit_type={'S': 'journals', 'U': 'institutions'}[ut], v_rank=lab,
                                n=len(h), median_delta=round(float(h.delta.median()), 3),
                                median_citer_standing=round(float(h.citer_standing.median()), 3),
                                median_band_halfwidth=round(float(((h.delta_hi - h.delta_lo) / 2).median()), 3),
                                in_band=int((c == 'in').sum()), near=int((c == 'near').sum()),
                                far=int((c == 'far').sum())))
    by_rank = pd.DataFrame(by_rank)
    out_r = paths.data / 'mncs_comparison_by_rank.csv'
    by_rank.to_csv(out_r, index=False)
    det = paths.working / 'benchmarks' / 'mncs_comparison_units.csv'
    det.parent.mkdir(parents=True, exist_ok=True)
    units.to_csv(det, index=False)
    pd.set_option('display.width', 220)
    print(summ.to_string(index=False))
    print()
    print(by_rank.to_string(index=False))
    for ut in 'SU':
        g = units[(units.unit_type == ut) & (units['class'].str.rstrip('*') == 'far')]
        g = g.reindex(g.delta.abs().sort_values(ascending=False).index).head(12)
        print(f'\nlargest far gaps, {"journals" if ut == "S" else "institutions"}:')
        print(g[['name', 'field', 'n_works', 'v', 'mncs', 'delta', 'delta_lo', 'delta_hi', 'citer_standing']]
              .round(3).to_string(index=False))
    print(f'\nsaved {out}\n      {out_r}\n      {det}\nFINISHED ({time.perf_counter() - t0:.0f}s)')


if __name__ == '__main__':
    main()
