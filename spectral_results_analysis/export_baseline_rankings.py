"""
Export baseline influence rankings to flat CSVs for multiple field filters.

Outputs in data/:
    baseline_rankings.csv   — F=EBAX (full corpus)
    rankings_F=E.csv        — F=E (economics-dominant sources)
    rankings_F=B.csv        — F=B (business-dominant sources)
    rankings_F=A.csv        — F=A (ambiguous E+B sources)

Columns in each file:
    unit_idx      — source_idx or institution_idx
    unit_type     — 'source' or 'institution'
    name          — source_name / institution_name
    field_eb      — E/B/A/X subfield (sources only; blank for institutions)
    country_code  — ISO country code (institutions only; blank for sources)
    v             — influence value
    rank_v        — rank by v (1 = highest)
    a_p           — works count used in ranking
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

EXPORT_LABELS = {
    'baseline': 'baseline_rankings.csv',
    'F=E':      'rankings_F=E.csv',
    'F=B':      'rankings_F=B.csv',
    'F=A':      'rankings_F=A.csv',
}

paths   = load_config()
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

for label, filename in EXPORT_LABELS.items():
    run = all_runs[label]
    chi_str   = 'STAR' if run['chi'] == -1.0 else str(round(run['chi'] * 100))
    alpha_int = round(run['alpha'] * 100)
    table = (
        f"rk_{run['run_code']}_{run['fx']}"
        f"_tauU{run['tau_u']}_tauS{run['tau_s']}_vartau"
        f"_rho{run['rho']}_m{run['m']}_chi{chi_str}_alpha{alpha_int}"
    )
    rk = db.execute(f'SELECT unit_idx, unit_type, v, rank_v, a_p FROM {table}').fetchdf()

    src = rk[rk['unit_type'] == 'S'].merge(
        sm.rename(columns={'source_idx': 'unit_idx', 'source_name': 'name'}),
        on='unit_idx', how='left',
    )
    src['unit_type']    = 'source'
    src['country_code'] = ''
    src['field_eb']     = src['field_eb'].fillna('')
    src['issn']         = src['issn'].fillna('')

    inst = rk[rk['unit_type'] == 'U'].merge(
        ci.rename(columns={'institution_idx': 'unit_idx', 'institution_name': 'name'}),
        on='unit_idx', how='left',
    )
    inst['unit_type']    = 'institution'
    inst['field_eb']     = ''
    inst['issn']         = ''
    inst['country_code'] = inst['country_code'].fillna('')

    out = pd.concat([src, inst], ignore_index=True).sort_values('rank_v')
    out = out[['unit_idx', 'unit_type', 'name', 'issn', 'field_eb', 'country_code', 'v', 'rank_v', 'a_p']]

    out_path = Path(__file__).parent.parent / 'data' / filename
    out.to_csv(str(out_path), index=False)
    print(f'[{label}]  {table}')
    print(f'  {len(rk):,} rows ({(rk.unit_type=="S").sum():,} src, {(rk.unit_type=="U").sum():,} inst)  → {out_path.name}')

db.close()
