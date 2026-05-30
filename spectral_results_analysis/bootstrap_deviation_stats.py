"""
bootstrap_deviation_stats.py — summary statistics on large bootstrap deviations.

For units with baseline v > V_BASE_MIN, counts replicates and unit×replicate
pairs where v_boot / v_base exceeds each threshold in both directions.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

V_BASE_MIN = 0.0 #1.0
THRESHOLDS = [2, 4, 8, 16]

_bl = next(r for r in load_runs() if r['label'] == 'baseline')
BASELINE_TABLE = (
    f"rk_{_bl['run_code']}_{_bl['fx']}"
    f"_tauU{_bl['tau_u']}_tauS{_bl['tau_s']}"
    f"_vartau_rho0_m0110_chi50_alpha100_omega1"
)


def _load_boot(d):
    vs = np.load(d / 'v_s_boot.npy').astype(np.float32)
    vu = np.load(d / 'v_u_boot.npy').astype(np.float32)
    with open(d / 'meta.json') as f:
        meta = json.load(f)
    B = meta.get('completed', vs.shape[0])
    return vs[:B], vu[:B], meta


def report(label, v_boot, v_base, B):
    """Print threshold table for one unit type."""
    ratio = v_boot / v_base[np.newaxis, :]   # (B, n_units)
    n_units = v_base.shape[0]
    total_pairs = B * n_units

    print(f'\n  {label}  —  B={B}, qualifying units={n_units}, '
          f'total unit×rep pairs={total_pairs:,}')
    print(f'  {"threshold":>10}  {"reps w/ ≥1 hit":>15}  {"% of reps":>10}  '
          f'{"units hit":>10}  {"% of units":>11}  '
          f'{"unit×rep pairs":>15}  {"% of pairs":>10}')
    print('  ' + '─' * 86)

    for t in THRESHOLDS:
        hit = (ratio >= t) | (ratio <= 1 / t)          # (B, n_units) bool
        reps_hit  = int(hit.any(axis=1).sum())
        units_hit = int(hit.any(axis=0).sum())
        pairs_hit = int(hit.sum())
        print(f'  {t:>8}×  {reps_hit:>15,}  {100*reps_hit/B:>9.1f}%  '
              f'{units_hit:>10,}  {100*units_hit/n_units:>10.1f}%  '
              f'{pairs_hit:>15,}  {100*pairs_hit/total_pairs:>9.3f}%')

    # Direction breakdown at 2×
    up2  = (ratio >= 2).any(axis=1).sum()
    dn2  = (ratio <= 0.5).any(axis=1).sum()
    both = ((ratio >= 2).any(axis=1) & (ratio <= 0.5).any(axis=1)).sum()
    print(f'\n  At 2× threshold:  {up2} reps have an upward outlier, '
          f'{dn2} have a downward outlier, {both} have both.')


def main():
    import duckdb
    paths     = load_config()
    boot_base = paths.working / 'bootstrap_oa_errors'
    rk_path   = paths.working / 'rankings.duckdb'

    with duckdb.connect(str(rk_path), read_only=True) as db:
        df_base = db.execute(
            f'SELECT unit_idx, unit_type, v FROM {BASELINE_TABLE}'
        ).df()

    id_to_v_s = dict(zip(
        df_base[df_base['unit_type'] == 'S']['unit_idx'].astype(int),
        df_base[df_base['unit_type'] == 'S']['v'],
    ))
    id_to_v_u = dict(zip(
        df_base[df_base['unit_type'] == 'U']['unit_idx'].astype(int),
        df_base[df_base['unit_type'] == 'U']['v'],
    ))

    print('Loading stage4 (combined) ...', flush=True)
    v_s_all, v_u_all, meta = _load_boot(boot_base / 'stage4')
    B = v_s_all.shape[0]
    print(f'  B={B}  n_s={v_s_all.shape[1]}  n_u={v_u_all.shape[1]}')

    for unit_type, v_boot_all, id_to_v, id_list, label in [
        ('S', v_s_all, id_to_v_s, meta['source_ids'], 'Sources'),
        ('U', v_u_all, id_to_v_u, meta['inst_ids'],   'Institutions'),
    ]:
        qualifying = [(uid, i) for i, uid in enumerate(id_list)
                      if uid in id_to_v and id_to_v[uid] > V_BASE_MIN]
        if not qualifying:
            continue
        cols   = [x[1] for x in qualifying]
        v_base = np.array([id_to_v[x[0]] for x in qualifying], dtype=np.float32)
        v_boot = v_boot_all[:, cols]
        report(label, v_boot, v_base, B)

    print()


if __name__ == '__main__':
    main()
    print('DONE')
