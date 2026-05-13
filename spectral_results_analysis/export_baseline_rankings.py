"""
export_baseline_rankings.py — one wide table with v and rank for all field filters.

Output: data/rankings_all_fields.csv
    unit_idx, unit_type, name, issn, field_eb, country_code, a_p
    v_EBAX, rank_EBAX, v_E, rank_E, v_B, rank_B, v_A, rank_A, v_X, rank_X
    v_eps1, rank_eps1
    ext_balance

Ranks are computed separately for sources and institutions within each run.
rank_EBAX=1 for a source means the top-ranked source in that run; rank_EBAX=1
for an institution means the top-ranked institution — they do not share a sequence.

ext_balance = (in - out) / (in + out) for each unit's citation flow to/from the
external sentinel in the eps=1 run. Range [-1, +1]: +1 = pure importer from
outside OAS, -1 = pure exporter to outside OAS, NaN = no boundary flow.

Units absent from a field-filtered run have NaN for that run's v/rank columns.
a_p is taken from the baseline (F=EBAX) run.
v_eps1 / rank_eps1 are from the baseline-eps (ε=1) run; sentinel row excluded.
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
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

EPS_EL_TABLE = 'el_20242024_EBAX_tauU20_tauS20_vartau_eps1'
SENTINEL_IDX = 1

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
            f'SELECT unit_idx, unit_type, v, a_p FROM {table}'
        ).fetchdf()
    except Exception as e:
        print(f'[{label}] table {table} not found — skipped ({e})')
        continue

    print(f'[{label}]  {table}  ({len(rk):,} rows)')

    if suffix == 'EBAX':
        a_p_base = rk[['unit_idx', 'unit_type', 'a_p']].copy()

    run_dfs[suffix] = rk[['unit_idx', 'unit_type', 'v']].rename(
        columns={'v': f'v_{suffix}'}
    )

# ── ε=1 run (baseline-eps) ────────────────────────────────────────────────────

eps_df = None
if 'baseline-eps' in all_runs:
    r = all_runs['baseline-eps']
    chi_str   = 'STAR' if r['chi'] == -1.0 else str(round(r['chi'] * 100))
    alpha_int = round(r['alpha'] * 100)
    eps_table = (
        f"rk_{r['run_code']}_{r['fx']}"
        f"_tauU{r['tau_u']}_tauS{r['tau_s']}_vartau"
        f"_rho{r['rho']}_m{r['m']}_chi{chi_str}_alpha{alpha_int}_eps1"
    )
    try:
        eps_df = db.execute(
            f'SELECT unit_idx, unit_type, v '
            f'FROM {eps_table} WHERE unit_idx != {SENTINEL_IDX}'
        ).fetchdf()
        eps_df = eps_df.rename(columns={'v': 'v_eps1'})
        print(f'[baseline-eps]  {eps_table}  ({len(eps_df):,} rows)')
    except Exception as e:
        print(f'[baseline-eps] table {eps_table} not found — skipped ({e})')
else:
    print('[baseline-eps] not in params.csv — skipped')

db.close()

if not run_dfs:
    raise RuntimeError('No runs loaded.')

# ── Merge into one wide table (outer join preserves all units) ────────────────

ordered = [run_dfs[s] for _, s in RUNS if s in run_dfs]
merged  = ordered[0]
for rk in ordered[1:]:
    merged = merged.merge(rk, on=['unit_idx', 'unit_type'], how='outer')

if a_p_base is not None:
    merged = merged.merge(a_p_base, on=['unit_idx', 'unit_type'], how='left')
else:
    merged['a_p'] = pd.NA

if eps_df is not None:
    merged = merged.merge(eps_df, on=['unit_idx', 'unit_type'], how='left')

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
inst['unit_type']    = 'institution'
inst['field_eb']     = ''
inst['issn']         = ''
inst['country_code'] = inst['country_code'].fillna('')

out = pd.concat([src, inst], ignore_index=True)
out['name'] = out['name'].fillna('')

# ── Ranks: separate sequences for sources and institutions within each run ────

v_suffixes = [s for _, s in RUNS if s in run_dfs]
if eps_df is not None:
    v_suffixes.append('eps1')

for sfx in v_suffixes:
    v_col = f'v_{sfx}'
    r_col = f'rank_{sfx}'
    if v_col not in out.columns:
        continue
    out[r_col] = (
        out.groupby('unit_type')[v_col]
        .rank(method='min', ascending=False, na_option='bottom')
        .astype('Int64')
    )

# ── External balance: (in - out) / (in + out) via eps1 edge list ─────────────

el_db_path = paths.working / 'edge_lists.duckdb'
ext_balance = pd.DataFrame(columns=['unit_idx', 'unit_type', 'ext_balance'])

if el_db_path.exists():
    try:
        with duckdb.connect(str(el_db_path), read_only=True) as eldb:
            tables = {t for t in eldb.execute("SHOW TABLES").fetchdf()['name']}
            if EPS_EL_TABLE in tables:

                # Source-level boundary flows (distinct work pairs)
                src_out = eldb.execute(f"""
                    SELECT citer_source_idx AS unit_idx, COUNT(*) AS n_out
                    FROM (SELECT DISTINCT citer_work_idx, cited_work_idx,
                                 citer_source_idx, cited_source_idx
                          FROM {EPS_EL_TABLE})
                    WHERE cited_source_idx  = {SENTINEL_IDX}
                      AND citer_source_idx != {SENTINEL_IDX}
                    GROUP BY citer_source_idx
                """).fetchdf()

                src_in = eldb.execute(f"""
                    SELECT cited_source_idx AS unit_idx, COUNT(*) AS n_in
                    FROM (SELECT DISTINCT citer_work_idx, cited_work_idx,
                                 citer_source_idx, cited_source_idx
                          FROM {EPS_EL_TABLE})
                    WHERE citer_source_idx  = {SENTINEL_IDX}
                      AND cited_source_idx != {SENTINEL_IDX}
                    GROUP BY cited_source_idx
                """).fetchdf()

                src_bal = (
                    src_out.merge(src_in, on='unit_idx', how='outer')
                    .fillna(0)
                )
                src_bal['ext_balance'] = (
                    (src_bal['n_in'] - src_bal['n_out'])
                    / (src_bal['n_in'] + src_bal['n_out'])
                )
                src_bal['unit_type'] = 'source'

                # Institution-level boundary flows (distinct work pairs)
                inst_out = eldb.execute(f"""
                    SELECT citer_inst_idx AS unit_idx, COUNT(*) AS n_out
                    FROM (SELECT DISTINCT citer_work_idx, cited_work_idx,
                                 citer_inst_idx, cited_inst_idx
                          FROM {EPS_EL_TABLE})
                    WHERE cited_inst_idx  = {SENTINEL_IDX}
                      AND citer_inst_idx != {SENTINEL_IDX}
                    GROUP BY citer_inst_idx
                """).fetchdf()

                inst_in = eldb.execute(f"""
                    SELECT cited_inst_idx AS unit_idx, COUNT(*) AS n_in
                    FROM (SELECT DISTINCT citer_work_idx, cited_work_idx,
                                 citer_inst_idx, cited_inst_idx
                          FROM {EPS_EL_TABLE})
                    WHERE citer_inst_idx  = {SENTINEL_IDX}
                      AND cited_inst_idx != {SENTINEL_IDX}
                    GROUP BY cited_inst_idx
                """).fetchdf()

                inst_bal = (
                    inst_out.merge(inst_in, on='unit_idx', how='outer')
                    .fillna(0)
                )
                inst_bal['ext_balance'] = (
                    (inst_bal['n_in'] - inst_bal['n_out'])
                    / (inst_bal['n_in'] + inst_bal['n_out'])
                )
                inst_bal['unit_type'] = 'institution'

                ext_balance = pd.concat(
                    [src_bal[['unit_idx', 'unit_type', 'ext_balance']],
                     inst_bal[['unit_idx', 'unit_type', 'ext_balance']]],
                    ignore_index=True,
                )
                print(f'[ext_balance]  {EPS_EL_TABLE}  '
                      f'({(ext_balance.unit_type=="source").sum():,} src, '
                      f'{(ext_balance.unit_type=="institution").sum():,} inst)')
            else:
                print(f'[ext_balance]  {EPS_EL_TABLE} not found — skipped')
    except Exception as e:
        print(f'[ext_balance]  failed — {e}')
else:
    print(f'[ext_balance]  {el_db_path} not found — skipped')

if not ext_balance.empty:
    out = out.merge(ext_balance, on=['unit_idx', 'unit_type'], how='left')
else:
    out['ext_balance'] = np.nan

# ── Column order and sort ─────────────────────────────────────────────────────

v_rank_cols = []
for _, suffix in RUNS:
    if suffix in run_dfs:
        v_rank_cols += [f'v_{suffix}', f'rank_{suffix}']
if eps_df is not None:
    v_rank_cols += ['v_eps1', 'rank_eps1']
v_rank_cols.append('ext_balance')

out = out[['unit_idx', 'unit_type', 'name', 'issn', 'field_eb',
           'country_code', 'a_p'] + v_rank_cols]
out = out.sort_values('rank_EBAX', na_position='last').reset_index(drop=True)

out_path = Path(__file__).parent.parent / 'data' / 'rankings_all_fields.csv'
out.to_csv(str(out_path), index=False, float_format='%.6g')
print(f'\n→ {out_path.name}  ({len(out):,} rows, '
      f'{(out.unit_type=="source").sum():,} src, '
      f'{(out.unit_type=="institution").sum():,} inst)')
