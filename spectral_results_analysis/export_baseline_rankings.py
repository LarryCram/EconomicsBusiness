"""
export_baseline_rankings.py — one wide table with v and rank for all field filters.

Output: data/rankings_all_fields.csv
    unit_idx, unit_type, name, issn, field_eb, country_code, a_p
    v_EBAX, rank_EBAX, v_E, rank_E, v_B, rank_B, v_A, rank_A, v_X, rank_X

Units absent from a field-filtered run have NaN for that run's v/rank columns.
a_p is taken from the baseline (F=EBAX) run.
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

RUNS = [
    ('baseline', 'EBAX'),
    ('F=E',      'E'),
    ('F=B',      'B'),
    ('F=A',      'A'),
    ('F=X',      'X'),
]

paths    = load_config()
all_runs = {r['label']: r for r in load_runs()}

sm = pd.read_parquet(
    str(paths.parquet / 'source_master.parquet'),
    columns=['source_idx', 'source_name', 'issn', 'field_eb'],
)
ci = pd.read_parquet(
    str(paths.parquet / 'corpus_institutions.parquet'),
    columns=['institution_idx', 'institution_name', 'country_code'],
)

db = duckdb.connect(str(paths.working / 'rankings.duckdb'), read_only=True)

# ── Load each run ─────────────────────────────────────────────────────────────

run_dfs  = {}
a_p_base = None

for label, suffix in RUNS:
    if label not in all_runs:
        print(f'[{label}] not in params.csv — skipped')
        continue
    run       = all_runs[label]
    chi_str   = 'STAR' if run['chi'] == -1.0 else str(round(run['chi'] * 100))
    alpha_int = round(run['alpha'] * 100)
    table = (
        f"rk_{run['run_code']}_{run['fx']}"
        f"_tauU{run['tau_u']}_tauS{run['tau_s']}_vartau"
        f"_rho{run['rho']}_m{run['m']}_chi{chi_str}_alpha{alpha_int}"
    )
    try:
        rk = db.execute(
            f'SELECT unit_idx, unit_type, v, rank_v, a_p FROM {table}'
        ).fetchdf()
    except Exception as e:
        print(f'[{label}] table {table} not found — skipped ({e})')
        continue

    print(f'[{label}]  {table}  ({len(rk):,} rows)')

    if suffix == 'EBAX':
        a_p_base = rk[['unit_idx', 'unit_type', 'a_p']].copy()

    run_dfs[suffix] = rk.rename(columns={
        'v':      f'v_{suffix}',
        'rank_v': f'rank_{suffix}',
    }).drop(columns=['a_p'])

db.close()

if not run_dfs:
    raise RuntimeError('No runs loaded.')

# ── Merge into one wide table (outer join preserves all units) ────────────────

ordered = [run_dfs[s] for _, s in RUNS if s in run_dfs]
if not ordered:
    raise RuntimeError('No run DataFrames to merge.')

merged = ordered[0]
for rk in ordered[1:]:
    merged = merged.merge(rk, on=['unit_idx', 'unit_type'], how='outer')

if a_p_base is not None:
    merged = merged.merge(a_p_base, on=['unit_idx', 'unit_type'], how='left')
else:
    merged['a_p'] = pd.NA

# ── Attach names ──────────────────────────────────────────────────────────────

src = merged[merged['unit_type'] == 'S'].merge(
    sm.rename(columns={'source_idx': 'unit_idx', 'source_name': 'name'}),
    on='unit_idx', how='left',
)
src['unit_type']    = 'source'
src['country_code'] = ''
src['field_eb']     = src['field_eb'].fillna('')
src['issn']         = src['issn'].fillna('')

inst = merged[merged['unit_type'] == 'U'].merge(
    ci.rename(columns={'institution_idx': 'unit_idx', 'institution_name': 'name'}),
    on='unit_idx', how='left',
)
inst['unit_type'] = 'institution'
inst['field_eb']  = ''
inst['issn']      = ''
inst['country_code'] = inst['country_code'].fillna('')

out = pd.concat([src, inst], ignore_index=True)

# ── Column order and sort ─────────────────────────────────────────────────────

v_rank_cols = []
for _, suffix in RUNS:
    if suffix in run_dfs:
        v_rank_cols += [f'v_{suffix}', f'rank_{suffix}']

out = out[['unit_idx', 'unit_type', 'name', 'issn', 'field_eb', 'country_code', 'a_p']
          + v_rank_cols]
out['name'] = out['name'].fillna('')
out = out.sort_values('rank_EBAX', na_position='last').reset_index(drop=True)

out_path = Path(__file__).parent.parent / 'data' / 'rankings_all_fields.csv'
out.to_csv(str(out_path), index=False)
print(f'\n→ {out_path.name}  ({len(out):,} rows, '
      f'{(out.unit_type=="source").sum():,} src, '
      f'{(out.unit_type=="institution").sum():,} inst)')
