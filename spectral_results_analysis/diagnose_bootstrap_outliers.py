"""
diagnose_bootstrap_outliers.py — locate extreme bootstrap replicates.

For sources and institutions with baseline v > 1, finds the 8 units that
move the most upward and the 8 that move the most downward across bootstrap
replicates.  For each outlier:
  - Attributes the movement to an error type (year / source / institution /
    reference / resample) using the per-ET stage4 arrays.
  - Identifies the responsible replicate b.
  - Summarises stage-2 (work-level) and stage-3 (reference-level) deltas for
    replicate b, showing how many works the unit gained or lost and how many
    incoming citations were redirected to or from it.

Usage:
    .venv/bin/python spectral_results_analysis/diagnose_bootstrap_outliers.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

N_OUTLIERS     = 8
V_BASE_MIN     = 1.0
OA_ERROR_TYPES = ['year', 'source', 'institution', 'reference', 'resample']

_bl = next(r for r in load_runs() if r['label'] == 'baseline')
BASELINE_TABLE = (
    f"rk_{_bl['run_code']}_{_bl['fx']}"
    f"_tauU{_bl['tau_u']}_tauS{_bl['tau_s']}"
    f"_vartau_rho0_m0110_chi50_alpha100_omega1"
)


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_boot(d: Path):
    vs = np.load(d / 'v_s_boot.npy').astype(np.float32)
    vu = np.load(d / 'v_u_boot.npy').astype(np.float32)
    with open(d / 'meta.json') as f:
        meta = json.load(f)
    B = meta.get('completed', vs.shape[0])
    return vs[:B], vu[:B], meta


# ── Outlier detection ─────────────────────────────────────────────────────────

def _find_outliers(v_boot_c, v_base, unit_ids, n=N_OUTLIERS):
    """
    Return DataFrame of worst-n up and down movers.

    v_boot_c : (B, n_aligned) filtered to v_base > V_BASE_MIN
    v_base   : (n_aligned,)
    unit_ids : (n_aligned,) int64
    """
    ratio = v_boot_c / v_base[np.newaxis, :]      # (B, n_aligned)
    max_r = ratio.max(axis=0)
    min_r = ratio.min(axis=0)
    b_up  = ratio.argmax(axis=0)
    b_dn  = ratio.argmin(axis=0)

    rows = []
    for direction, score, b_arr, top_fn in [
        ('up', max_r, b_up, lambda s: np.argsort(-s)[:n]),
        ('dn', min_r, b_dn, lambda s: np.argsort(s)[:n]),
    ]:
        for i in top_fn(score):
            rows.append(dict(
                unit_idx   = int(unit_ids[i]),
                v_base     = float(v_base[i]),
                ratio      = float(score[i]),
                log10_ratio = float(np.log10(score[i])),
                replicate  = int(b_arr[i]),
                direction  = direction,
            ))
    return pd.DataFrame(rows)


def _attribute_et(col_i, direction, v_base_val, et_boot_c):
    """
    Return (error_type, et_ratio, et_replicate_b) for the ET that produces
    the most extreme ratio in the given direction for unit at position col_i.
    """
    best_et, best_r, best_b = 'unknown', (1.0 if direction == 'up' else np.inf), -1
    for et, vb in et_boot_c.items():
        if col_i >= vb.shape[1]:
            continue
        ratios = vb[:, col_i].astype(np.float64) / v_base_val
        r = float(ratios.max() if direction == 'up' else ratios.min())
        b = int(ratios.argmax() if direction == 'up' else ratios.argmin())
        if (direction == 'up' and r > best_r) or (direction == 'dn' and r < best_r):
            best_et, best_r, best_b = et, r, b
    return best_et, best_r, best_b


# ── Stage-2 delta summary ─────────────────────────────────────────────────────

def _stage2_summary(uid, unit_type, rep_b, stage_dir, db, n_corpus_works=None):
    s1 = str(stage_dir / 'stage1_base_works.parquet')
    s2 = str(stage_dir / 'stage2_work_errors.parquet')

    if unit_type == 'S':
        n_own = db.execute(
            f"SELECT COUNT(DISTINCT work_idx) FROM read_parquet('{s1}') "
            f"WHERE source_idx = {uid}"
        ).fetchone()[0]

        rows = db.execute(
            f"SELECT work_idx, source_idx, error_type "
            f"FROM read_parquet('{s2}') WHERE replicate_id = {rep_b}"
        ).df()

        if rows.empty:
            print(f'        Stage2: no deltas for replicate {rep_b}')
            return

        own_works = set(db.execute(
            f"SELECT DISTINCT work_idx FROM read_parquet('{s1}') WHERE source_idx = {uid}"
        ).df()['work_idx'].tolist())

        lost   = rows[rows['work_idx'].isin(own_works) & (rows['source_idx'] != uid)]
        gained = rows[rows['work_idx'].isin(own_works) == False]  # noqa: E712
        gained = rows[(~rows['work_idx'].isin(own_works)) & (rows['source_idx'] == uid)]
        own_et = rows[rows['work_idx'].isin(own_works)].groupby('error_type').size().to_dict()

        pct = f'  ({100 * len(rows) / n_corpus_works:.1f}% of corpus)' if n_corpus_works else ''
        print(f'        Stage2  rep={rep_b}: {len(rows):,} total corpus work deltas{pct}')
        print(f'          Baseline own works:           {n_own}')
        print(f'          Own works touched (by type):  {own_et}')
        print(f'          Works gained (→ this source): {len(gained)}')
        print(f'          Works lost  (this source →):  {len(lost)}')

    else:
        n_own = db.execute(
            f"SELECT COUNT(DISTINCT work_idx) FROM read_parquet('{s1}') "
            f"WHERE inst_idx = {uid}"
        ).fetchone()[0]

        gained = db.execute(
            f"SELECT COUNT(*) FROM read_parquet('{s2}') "
            f"WHERE replicate_id = {rep_b} AND inst_idx = {uid}"
        ).fetchone()[0]
        lost = db.execute(
            f"SELECT COUNT(*) FROM read_parquet('{s2}') "
            f"WHERE replicate_id = {rep_b} AND inst_idx_old = {uid}"
        ).fetchone()[0]
        total = db.execute(
            f"SELECT COUNT(*) FROM read_parquet('{s2}') WHERE replicate_id = {rep_b}"
        ).fetchone()[0]

        pct = f'  ({100 * total / n_corpus_works:.1f}% of corpus)' if n_corpus_works else ''
        print(f'        Stage2  rep={rep_b}: {total:,} total corpus work deltas{pct}')
        print(f'          Baseline own work-inst rows:      {n_own}')
        print(f'          Institution assignments gained:   {gained}')
        print(f'          Institution assignments lost:     {lost}')


# ── Stage-3 delta summary ─────────────────────────────────────────────────────

def _stage3_summary(uid, unit_type, rep_b, stage_dir, db):
    s1 = str(stage_dir / 'stage1_base_works.parquet')
    br = str(stage_dir / 'stage3_base_refs.parquet')
    re = str(stage_dir / 'stage3_ref_errors.parquet')

    total = db.execute(
        f"SELECT COUNT(*) FROM read_parquet('{re}') WHERE replicate_id = {rep_b}"
    ).fetchone()[0]

    if unit_type == 'S':
        # Citations originally TO this source that were redirected away
        n_lost = db.execute(
            f"SELECT COUNT(*) FROM read_parquet('{re}') e "
            f"JOIN read_parquet('{br}') r USING (ref_idx) "
            f"WHERE e.replicate_id = {rep_b} AND r.cited_source_idx = {uid}"
        ).fetchone()[0]

        # Refs now newly pointing at this source's works
        n_gained = db.execute(
            f"SELECT COUNT(*) FROM read_parquet('{re}') e "
            f"JOIN read_parquet('{s1}') w ON e.new_cited_work_idx = w.work_idx "
            f"WHERE e.replicate_id = {rep_b} AND w.source_idx = {uid}"
        ).fetchone()[0]

        print(f'        Stage3  rep={rep_b}: {total:,} total corpus ref deltas')
        print(f'          Incoming citations redirected away: {n_lost}')
        print(f'          Refs newly pointing here:           {n_gained}')
        print(f'          Net citation change:                {n_gained - n_lost:+d}')

    else:
        # Citations to institution works redirected away
        n_lost = db.execute(
            f"SELECT COUNT(*) FROM read_parquet('{re}') e "
            f"JOIN read_parquet('{br}') r USING (ref_idx) "
            f"JOIN read_parquet('{s1}') w ON r.cited_work_idx = w.work_idx "
            f"WHERE e.replicate_id = {rep_b} AND w.inst_idx = {uid}"
        ).fetchone()[0]

        # Refs now newly pointing at institution works
        n_gained = db.execute(
            f"SELECT COUNT(*) FROM read_parquet('{re}') e "
            f"JOIN read_parquet('{s1}') w ON e.new_cited_work_idx = w.work_idx "
            f"WHERE e.replicate_id = {rep_b} AND w.inst_idx = {uid}"
        ).fetchone()[0]

        print(f'        Stage3  rep={rep_b}: {total:,} total corpus ref deltas')
        print(f'          Citations to inst works redirected away: {n_lost}')
        print(f'          Refs newly pointing at inst works:       {n_gained}')
        print(f'          Net citation change:                     {n_gained - n_lost:+d}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    paths     = load_config()
    boot_base = paths.working / 'bootstrap_oa_errors'
    rk_path   = paths.working / 'rankings.duckdb'

    sm = pd.read_parquet(str(paths.parquet / 'source_master.parquet'),
                         columns=['source_idx', 'source_name'])
    ci = pd.read_parquet(str(paths.parquet / 'corpus_institutions.parquet'),
                         columns=['institution_idx', 'institution_name'])
    src_name  = dict(zip(sm['source_idx'],      sm['source_name']))
    inst_name = dict(zip(ci['institution_idx'], ci['institution_name']))

    print(f'Baseline table: {BASELINE_TABLE}')
    with duckdb.connect(str(rk_path), read_only=True) as rk_db:
        df_base = rk_db.execute(
            f'SELECT unit_idx, unit_type, v FROM {BASELINE_TABLE}'
        ).df()
    df_base_s = df_base[df_base['unit_type'] == 'S'][['unit_idx', 'v']]
    df_base_i = df_base[df_base['unit_type'] == 'U'][['unit_idx', 'v']]

    print('Loading stage4 (combined) ...', flush=True)
    v_s_all, v_u_all, meta = _load_boot(boot_base / 'stage4')
    print(f'  B={v_s_all.shape[0]}  n_s={v_s_all.shape[1]}  n_u={v_u_all.shape[1]}')

    src_dense  = {uid: i for i, uid in enumerate(meta['source_ids'])}
    inst_dense = {uid: i for i, uid in enumerate(meta['inst_ids'])}

    print('Loading per-ET boot arrays ...', flush=True)
    et_v_s, et_v_u = {}, {}
    for et in OA_ERROR_TYPES:
        d = boot_base / f'stage4_{et}'
        if (d / 'v_s_boot.npy').exists():
            vs, vu, _ = _load_boot(d)
            et_v_s[et] = vs
            et_v_u[et] = vu
            print(f'  {et}: B={vs.shape[0]}')

    db = duckdb.connect()   # in-memory; reads parquets via SQL
    s1 = str(boot_base / 'stage1_base_works.parquet')
    _row = db.execute(
        f"SELECT COUNT(DISTINCT work_idx) FROM read_parquet('{s1}')"
    ).fetchone()
    n_corpus_works = _row[0] if _row else 0
    print(f'Corpus works: {n_corpus_works:,}')

    for unit_type, v_boot_all, et_v_boot, df_bv, id_to_dense, name_map in [
        ('S', v_s_all, et_v_s, df_base_s, src_dense,  src_name),
        ('I', v_u_all, et_v_u, df_base_i, inst_dense, inst_name),
    ]:
        label = 'Sources' if unit_type == 'S' else 'Institutions'
        print(f'\n{"═" * 74}')
        print(f'  {label}  (baseline v > {V_BASE_MIN})')
        print(f'{"═" * 74}')

        id_to_v = dict(zip(df_bv['unit_idx'].astype(int), df_bv['v']))
        common  = [(uid, id_to_dense[uid])
                   for uid in (meta['source_ids'] if unit_type == 'S'
                                else meta['inst_ids'])
                   if uid in id_to_v and id_to_v[uid] > V_BASE_MIN]
        if not common:
            print('  No qualifying units.')
            continue

        unit_ids   = np.array([x[0] for x in common], dtype=np.int64)
        dense_cols = [x[1] for x in common]
        uid_to_col = {int(uid): i for i, uid in enumerate(unit_ids)}
        v_base     = np.array([id_to_v[int(uid)] for uid in unit_ids], dtype=np.float32)
        v_boot_c   = v_boot_all[:, dense_cols]

        # ET arrays aligned to the same column positions
        et_boot_c = {}
        for et, vb in et_v_boot.items():
            if vb.shape[1] > max(dense_cols):
                et_boot_c[et] = vb[:, dense_cols]

        outliers = _find_outliers(v_boot_c, v_base, unit_ids)

        for direction in ('up', 'dn'):
            sub = outliers[outliers['direction'] == direction]
            arrow = '▲ Upward' if direction == 'up' else '▼ Downward'
            print(f'\n  {arrow} top {N_OUTLIERS}  (ratio = v_boot / v_base):')
            print(f'  {"unit_idx":>12}  {"name":<38}  {"v_base":>7}  '
                  f'{"ratio":>6}  {"log10":>5}  {"error_type":<13}  {"rep":>4}')
            print('  ' + '─' * 94)

            for _, row in sub.iterrows():
                uid  = int(row['unit_idx'])
                col_i = uid_to_col[uid]
                name = name_map.get(uid, str(uid))[:38]
                et, _, et_b = _attribute_et(
                    col_i, direction, float(row['v_base']), et_boot_c)
                print(f'  {uid:>12}  {name:<38}  {row["v_base"]:>7.3f}  '
                      f'{row["ratio"]:>6.2f}  {row["log10_ratio"]:>+5.1f}  '
                      f'{et:<13}  {et_b:>4}')
                _stage2_summary(uid, unit_type, et_b, boot_base, db, n_corpus_works)
                _stage3_summary(uid, unit_type, et_b, boot_base, db)

    db.close()


if __name__ == '__main__':
    main()
    print('\nFINISHED!')
