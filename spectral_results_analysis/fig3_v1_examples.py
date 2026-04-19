"""
fig3_v1_examples.py — Table of illustrative units near v=1 for fig_3 discussion.

For sources: two pairs selected from units with v_SS near 1 (extremes of v_bip)
and units with v_bip near 1 (extremes of v_SS).
For institutions: same logic with v_II and v_bip.

field_eb=X units excluded (consistent with fig_3).

Output: plots/fig3_v1_examples.csv
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

_baseline = next(r for r in load_runs() if r['label'] == 'baseline')
_run_code = _baseline['run_code']
_tau_u    = _baseline['tau_u']
_tau_s    = _baseline['tau_s']
_fx       = _baseline['fx']

BASELINE_TABLE = f'rk_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m0110_chi50_alpha100'
SS_TABLE       = f'rk_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m1000_chi50_alpha100'
II_TABLE       = f'rk_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m0001_chi50_alpha100'
EL_TABLE       = f'el_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau'

# How close to v=1 to define "near 1"
V1_BAND = 0.35   # |v - 1| < V1_BAND


def _explain_source(row) -> str:
    """
    v_SB[s] = H_SI-weighted mean v_IB of cited institutions.
    Explain the relationship between v_bip, v_ss, and mean_v_I_bip.
    """
    v_bip = row['v_bip']
    v_ss  = row['v_ss']
    mv_i  = row['mean_v_I_bip']

    bip_vs_ss   = 'above' if v_bip > v_ss  else 'below'
    partner_rel = 'above' if mv_i   > 1.0  else 'below'
    vbip_rel    = 'above' if v_bip  > 1.0  else 'below'

    return (
        f"v_SB ({v_bip:.3f}) is {bip_vs_ss} v_SS ({v_ss:.3f}). "
        f"In the bipartite model v_SB equals the H_SI-weighted mean v_IB of cited "
        f"institutions ({mv_i:.3f}, {partner_rel} the institutional mean of 1), "
        f"placing this source {vbip_rel} the bipartite average. "
        f"v_SS reflects source-source citations, which are absent from the bipartite model."
    )


def _explain_institution(row) -> str:
    """
    v_IB[i] = H_IS-weighted mean v_SB of citing sources.
    Explain the relationship between v_bip, v_ii, and mean_v_S_bip.
    """
    v_bip = row['v_bip']
    v_ii  = row['v_ii']
    mv_s  = row['mean_v_S_bip']

    bip_vs_ii   = 'above' if v_bip > v_ii  else 'below'
    partner_rel = 'above' if mv_s   > 1.0  else 'below'
    vbip_rel    = 'above' if v_bip  > 1.0  else 'below'

    return (
        f"v_IB ({v_bip:.3f}) is {bip_vs_ii} v_II ({v_ii:.3f}). "
        f"In the bipartite model v_IB equals the H_IS-weighted mean v_SB of citing "
        f"sources ({mv_s:.3f}, {partner_rel} the source mean of 1), "
        f"placing this institution {vbip_rel} the bipartite average. "
        f"v_II reflects institution-institution citations, which are absent from the bipartite model."
    )


def _write_latex(df: pd.DataFrame, paths) -> None:
    src = df[df['type'] == 'Sources'].copy()
    ins = df[df['type'] == 'Institutions'].copy()

    case_order_s = ['min v_ss | v_bip≈1', 'max v_ss | v_bip≈1',
                    'min v_bip | v_ss≈1', 'max v_bip | v_ss≈1']
    case_order_i = ['min v_ii | v_bip≈1', 'max v_ii | v_bip≈1',
                    'min v_bip | v_ii≈1', 'max v_bip | v_ii≈1']

    case_label = {
        'min v_ss | v_bip≈1':  r'$\min v_{\rm SS}$',
        'max v_ss | v_bip≈1':  r'$\max v_{\rm SS}$',
        'min v_bip | v_ss≈1':  r'$\min v_{\rm bip}$',
        'max v_bip | v_ss≈1':  r'$\max v_{\rm bip}$',
        'min v_ii | v_bip≈1':  r'$\min v_{\rm II}$',
        'max v_ii | v_bip≈1':  r'$\max v_{\rm II}$',
        'min v_bip | v_ii≈1':  r'$\min v_{\rm bip}$',
        'max v_bip | v_ii≈1':  r'$\max v_{\rm bip}$',
    }

    def fmt(x):
        return f'{float(x):.2f}'

    def src_row(r):
        return (f'{r["name"]} & {fmt(r["v_bip"])} & {fmt(r["v_ss"])} & {fmt(r["mean_v_I_bip"])}'
                f' & {case_label[r["case"]]} \\\\\n')

    def ins_row(r):
        return (f'{r["name"]} & {fmt(r["v_bip"])} & {fmt(r["v_ii"])} & {fmt(r["mean_v_S_bip"])}'
                f' & {case_label[r["case"]]} \\\\\n')

    src_rows = ''.join(src_row(src[src['case'] == c].iloc[0]) for c in case_order_s)
    ins_rows = ''.join(ins_row(ins[ins['case'] == c].iloc[0]) for c in case_order_i)

    tex = (
        r'\begin{table}[htbp]' + '\n'
        r'\centering\small' + '\n'
        r'\caption{Illustrative sources and institutions near $v=1$, showing how' + '\n'
        r'$v_{\rm bip}$ diverges from $v_{\rm SS}$ and $v_{\rm II}$.' + '\n'
        r'$\bar{v}^{\rm bip}$ is the $H$-weighted mean bipartite rank of cross-type' + '\n'
        r'partners. Field: E\,=\,economics, B\,=\,business, A\,=\,ambiguous.}' + '\n'
        r'\label{tab:unit_effects}' + '\n'
        r'\begin{tabular}{lrrrl}' + '\n'
        r'\toprule' + '\n'
        r'\multicolumn{5}{l}{\textit{Sources}} \\[2pt]' + '\n'
        r'Name & $v_{\rm bip}$ & $v_{\rm SS}$ & $\bar{v}_{I}^{\rm bip}$ & Case \\' + '\n'
        r'\midrule' + '\n'
        + src_rows
        + r'\midrule' + '\n'
        r'\multicolumn{5}{l}{\textit{Institutions}} \\[2pt]' + '\n'
        r'Name & $v_{\rm bip}$ & $v_{\rm II}$ & $\bar{v}_{S}^{\rm bip}$ & Case \\' + '\n'
        r'\midrule' + '\n'
        + ins_rows
        + r'\bottomrule' + '\n'
        r'\end{tabular}' + '\n'
        r'\end{table}' + '\n'
    )

    out = Path(__file__).parent.parent / 'tables' / 'table_unit_effects.tex'
    out.write_text(tex)
    print(f'Saved {out}')


def main():
    paths   = load_config()
    rk_path = paths.working / 'rankings.duckdb'
    el_path = paths.working / 'edge_lists.duckdb'
    par     = paths.working / 'parquet'

    # ── Load rankings ─────────────────────────────────────────────────────────
    with duckdb.connect(str(rk_path), read_only=True) as db:
        df_bip = db.execute(
            f"SELECT unit_idx, unit_type, v FROM {BASELINE_TABLE}"
        ).df()
        df_ss = db.execute(
            f"SELECT unit_idx, v AS v_ss FROM {SS_TABLE} WHERE unit_type='S'"
        ).df()
        df_ii = db.execute(
            f"SELECT unit_idx, v AS v_ii FROM {II_TABLE} WHERE unit_type='U'"
        ).df()

    df_s_bip = df_bip[df_bip['unit_type'] == 'S'][['unit_idx', 'v']].rename(columns={'v': 'v_bip'})
    df_i_bip = df_bip[df_bip['unit_type'] == 'U'][['unit_idx', 'v']].rename(columns={'v': 'v_bip'})

    # ── Source and institution metadata ───────────────────────────────────────
    sm = pd.read_parquet(str(par / 'source_master.parquet'),
                         columns=['source_idx', 'source_name', 'field_eb'])
    sm = sm.rename(columns={'source_idx': 'unit_idx'})

    inst_eb = pd.read_parquet(str(par / 'institution_field_eb.parquet'),
                              columns=['unit_idx', 'field_eb'])

    # ── Drop field_eb=X ───────────────────────────────────────────────────────
    non_x_src  = set(sm.loc[sm['field_eb'] != 'X', 'unit_idx'])
    non_x_inst = set(inst_eb.loc[inst_eb['field_eb'] != 'X', 'unit_idx'])

    # ── Cross-type partner colours (H_SI weighted) ────────────────────────────
    with duckdb.connect(str(el_path), read_only=True) as db:
        el = db.execute(f"""
            SELECT citer_source_idx, cited_inst_idx, SUM(cited_inst_weight) AS w
            FROM {EL_TABLE}
            WHERE citer_source_idx IS NOT NULL AND cited_inst_idx IS NOT NULL
            GROUP BY citer_source_idx, cited_inst_idx
        """).df()

    # Source → H_SI-weighted mean v_IB
    el_s = (el.merge(df_i_bip.rename(columns={'unit_idx': 'cited_inst_idx', 'v_bip': 'v_i'}),
                     on='cited_inst_idx', how='inner'))
    el_s['wv'] = el_s['w'] * el_s['v_i']
    agg_s = el_s.groupby('citer_source_idx')[['wv', 'w']].sum()
    agg_s['mean_v_I_bip'] = agg_s['wv'] / agg_s['w']
    agg_s = agg_s[['mean_v_I_bip']].reset_index().rename(columns={'citer_source_idx': 'unit_idx'})

    # Institution → H_IS-weighted mean v_SB
    el_i = (el.merge(df_s_bip.rename(columns={'unit_idx': 'citer_source_idx', 'v_bip': 'v_s'}),
                     on='citer_source_idx', how='inner'))
    el_i['wv'] = el_i['w'] * el_i['v_s']
    agg_i = el_i.groupby('cited_inst_idx')[['wv', 'w']].sum()
    agg_i['mean_v_S_bip'] = agg_i['wv'] / agg_i['w']
    agg_i = agg_i[['mean_v_S_bip']].reset_index().rename(columns={'cited_inst_idx': 'unit_idx'})

    # ── Assemble source table ─────────────────────────────────────────────────
    src = (df_s_bip
           .merge(df_ss,  on='unit_idx', how='inner')
           .merge(agg_s,  on='unit_idx', how='left')
           .merge(sm[['unit_idx', 'source_name', 'field_eb']], on='unit_idx', how='inner'))
    src = src[src['unit_idx'].isin(non_x_src)].copy()

    # ── Assemble institution table ────────────────────────────────────────────
    corp_inst = pd.read_parquet(str(par / 'corpus_institutions.parquet'),
                                columns=['institution_idx', 'institution_name'])
    corp_inst = corp_inst.rename(columns={'institution_idx': 'unit_idx',
                                          'institution_name': 'source_name'})

    inst = (df_i_bip
            .merge(df_ii,       on='unit_idx', how='inner')
            .merge(agg_i,       on='unit_idx', how='left')
            .merge(inst_eb,     on='unit_idx', how='inner')
            .merge(corp_inst,   on='unit_idx', how='left'))
    inst = inst[inst['unit_idx'].isin(non_x_inst)].copy()
    inst = inst.rename(columns={'display_name': 'source_name'})

    # ── Select examples ───────────────────────────────────────────────────────
    rows = []

    # Sources near v_SS ≈ 1: extremes of v_bip
    band_s = src[(src['v_ss'] - 1.0).abs() < V1_BAND].copy()
    if not band_s.empty:
        rows.append(('Sources', 'min v_bip | v_ss≈1',
                     band_s.loc[band_s['v_bip'].idxmin()]))
        rows.append(('Sources', 'max v_bip | v_ss≈1',
                     band_s.loc[band_s['v_bip'].idxmax()]))

    # Sources near v_bip ≈ 1: extremes of v_ss
    band_sb = src[(src['v_bip'] - 1.0).abs() < V1_BAND].copy()
    if not band_sb.empty:
        rows.append(('Sources', 'min v_ss | v_bip≈1',
                     band_sb.loc[band_sb['v_ss'].idxmin()]))
        rows.append(('Sources', 'max v_ss | v_bip≈1',
                     band_sb.loc[band_sb['v_ss'].idxmax()]))

    # Institutions near v_II ≈ 1: extremes of v_bip
    band_i = inst[(inst['v_ii'] - 1.0).abs() < V1_BAND].copy()
    if not band_i.empty:
        rows.append(('Institutions', 'min v_bip | v_ii≈1',
                     band_i.loc[band_i['v_bip'].idxmin()]))
        rows.append(('Institutions', 'max v_bip | v_ii≈1',
                     band_i.loc[band_i['v_bip'].idxmax()]))

    # Institutions near v_bip ≈ 1: extremes of v_ii
    band_ib = inst[(inst['v_bip'] - 1.0).abs() < V1_BAND].copy()
    if not band_ib.empty:
        rows.append(('Institutions', 'min v_ii | v_bip≈1',
                     band_ib.loc[band_ib['v_ii'].idxmin()]))
        rows.append(('Institutions', 'max v_ii | v_bip≈1',
                     band_ib.loc[band_ib['v_ii'].idxmax()]))

    # ── Build output DataFrame ────────────────────────────────────────────────
    records = []
    for unit_type, case, r in rows:
        is_src = unit_type == 'Sources'
        rec = {
            'type':        unit_type,
            'case':        case,
            'name':        r['source_name'],
            'field_eb':    r['field_eb'],
            'v_bip':       round(float(r['v_bip']), 3),
            'v_ss':        round(float(r['v_ss']),  3) if is_src  else '',
            'mean_v_I_bip': round(float(r['mean_v_I_bip']), 3) if is_src else '',
            'v_ii':        '' if is_src else round(float(r['v_ii']),  3),
            'mean_v_S_bip': '' if is_src else round(float(r['mean_v_S_bip']), 3),
            'explanation': _explain_source(r) if is_src else _explain_institution(r),
        }
        records.append(rec)

    out = pd.DataFrame(records)
    out_path = paths.plots / 'fig3_v1_examples.csv'
    out.to_csv(str(out_path), index=False)
    print(out.to_string(index=False))
    print(f'\nSaved {out_path}')

    _write_latex(out, paths)


if __name__ == '__main__':
    main()
    print('FINISHED!')
