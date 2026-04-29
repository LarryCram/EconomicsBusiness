"""
source_communities.py — Leiden community structure of source and institution citation networks.

Builds four directed weighted graphs:

  A_SS            = C_SS                  (source-to-source direct citations)
  A_bipartite_S   = C_SI @ C_IS          (institution-mediated source-to-source)
  A_II            = C_II                  (institution-to-institution direct citations)
  A_bipartite_I   = C_IS @ C_SI          (source-mediated institution-to-institution)

Runs Leiden community detection on each, prints a modularity summary, then plots
the Fig-3 scatter coloured by community membership (2×2: sources top, institutions bottom).

All runs: baseline time window, τ_U=τ_S=20, ρ=0.
"""

import sys
from pathlib import Path

import duckdb
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs

sys.path.insert(0, str(Path(__file__).parent.parent / 'spectral_ranking'))
from build_csr import build_csr

_baseline = next(r for r in load_runs() if r['label'] == 'baseline')
_run_code = _baseline['run_code']
_tau_u    = _baseline['tau_u']
_tau_s    = _baseline['tau_s']
_fx       = _baseline['fx']

BASELINE_TABLE = f'rk_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m0110_chi50_alpha100'
SS_TABLE       = f'rk_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m1000_chi50_alpha100'
II_TABLE       = f'rk_{_run_code}_{_fx}_tauU{_tau_u}_tauS{_tau_s}_vartau_rho0_m0001_chi50_alpha100'

# Leiden resolution parameter (γ); increase to get finer communities
RESOLUTION = 1.0
N_ITERATIONS = -1   # run until convergence


# ─── Matrix construction ──────────────────────────────────────────────────────

def build_matrices(el_path: Path):
    """
    Returns
    -------
    source_ids    : (n_s,) int64 — common source set (SS ∩ bipartite)
    A_SS          : (n_s × n_s) CSR — raw C_SS
    A_bipartite_S : (n_s × n_s) CSR — C_SI @ C_IS
    inst_ids      : (n_u,) int64 — common institution set (II ∩ bipartite)
    A_II          : (n_u × n_u) CSR — raw C_II
    A_bipartite_I : (n_u × n_u) CSR — C_IS @ C_SI
    """
    with duckdb.connect(str(el_path)) as db:
        print('Building C_SS ...')
        csr_ss = build_csr(db, _run_code, _fx, _tau_u, _tau_s, 0, (1, 0, 0, 0))
        print('Building C_SI, C_IS ...')
        csr_bi = build_csr(db, _run_code, _fx, _tau_u, _tau_s, 0, (0, 1, 1, 0))
        print('Building C_II ...')
        csr_ii = build_csr(db, _run_code, _fx, _tau_u, _tau_s, 0, (0, 0, 0, 1))

    def restrict(C: sp.csr_matrix, old_ids: np.ndarray, keep_ids: np.ndarray) -> sp.csr_matrix:
        idx = pd.Index(old_ids).get_indexer(keep_ids)
        return C[idx, :][:, idx].tocsr()

    # ── Sources ───────────────────────────────────────────────────────────────
    common_s = np.array(sorted(
        set(csr_ss.source_ids.tolist()) & set(csr_bi.source_ids.tolist())
    ), dtype=np.int64)

    A_SS = restrict(csr_ss.C_SS, csr_ss.source_ids, common_s)

    idx_bi_s  = pd.Index(csr_bi.source_ids).get_indexer(common_s)
    C_SI_r    = csr_bi.C_SI[idx_bi_s, :]
    C_IS_r    = csr_bi.C_IS[:, idx_bi_s]
    A_bip_S   = (C_SI_r @ C_IS_r).tocsr()

    print(f'  Sources in SS:            {csr_ss.n_s:,}')
    print(f'  Sources in bipartite:     {csr_bi.n_s:,}')
    print(f'  Common source set:        {len(common_s):,}')
    print(f'  A_SS nnz:                 {A_SS.nnz:,}')
    print(f'  A_bipartite_S nnz:        {A_bip_S.nnz:,}')

    # ── Institutions ──────────────────────────────────────────────────────────
    common_i = np.array(sorted(
        set(csr_ii.inst_ids.tolist()) & set(csr_bi.inst_ids.tolist())
    ), dtype=np.int64)

    A_II = restrict(csr_ii.C_II, csr_ii.inst_ids, common_i)

    idx_bi_i  = pd.Index(csr_bi.inst_ids).get_indexer(common_i)
    C_IS_r2   = csr_bi.C_IS[idx_bi_i, :]
    C_SI_r2   = csr_bi.C_SI[:, idx_bi_i]
    A_bip_I   = (C_IS_r2 @ C_SI_r2).tocsr()

    print(f'  Institutions in II:       {csr_ii.n_u:,}')
    print(f'  Institutions in bipartite:{csr_bi.n_u:,}')
    print(f'  Common institution set:   {len(common_i):,}')
    print(f'  A_II nnz:                 {A_II.nnz:,}')
    print(f'  A_bipartite_I nnz:        {A_bip_I.nnz:,}')

    return common_s, A_SS, A_bip_S, common_i, A_II, A_bip_I


# ─── igraph / Leiden ──────────────────────────────────────────────────────────

def csr_to_igraph(A: sp.csr_matrix) -> ig.Graph:
    """Convert a square CSR matrix to a directed weighted igraph Graph."""
    A_coo = A.tocoo()
    mask = A_coo.row != A_coo.col
    edges = list(zip(A_coo.row[mask].tolist(), A_coo.col[mask].tolist()))
    weights = A_coo.data[mask].tolist()
    g = ig.Graph(n=A.shape[0], edges=edges, directed=True)
    g.es['weight'] = weights
    return g


def run_leiden(g: ig.Graph, label: str) -> tuple[np.ndarray, float]:
    """Run Leiden with directed modularity; return (membership, Q)."""
    print(f'\nLeiden on {label} ...')
    partition = leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition,
        weights='weight',
        n_iterations=N_ITERATIONS,
        seed=42,
    )
    membership = np.array(partition.membership)
    sizes = np.bincount(membership)
    sizes_sorted = np.sort(sizes)[::-1]
    print(f'  Communities: {len(sizes)}   '
          f'top-5 sizes: {sizes_sorted[:5].tolist()}   '
          f'modularity Q={partition.modularity:.4f}')
    return membership, partition.modularity


# ─── Ranking data ─────────────────────────────────────────────────────────────

def load_rankings(rk_path: Path, source_ids: np.ndarray) -> pd.DataFrame:
    """
    Load v from 0110 baseline and 1000 SS runs for sources.
    Returns DataFrame with columns: unit_idx, v_0110, v_1000, a_p, baseline_rank.
    """
    sid_set = set(source_ids.tolist())
    with duckdb.connect(str(rk_path), read_only=True) as db:
        tables = {r[0] for r in db.execute('SHOW TABLES').fetchall()}

        def _load(table, col, extra=''):
            if table not in tables:
                raise RuntimeError(f'{table} not found in rankings.duckdb')
            return db.execute(
                f"SELECT unit_idx, v{extra} FROM {table} WHERE unit_type='S'"
            ).df().rename(columns={'v': col})

        df0 = _load(BASELINE_TABLE, 'v_0110', ', a_p')
        df1 = _load(SS_TABLE, 'v_1000')

    df = df0.merge(df1, on='unit_idx', how='inner')
    df = df[df['unit_idx'].isin(sid_set)].copy()
    df = df.sort_values('v_0110', ascending=False).reset_index(drop=True)
    df['baseline_rank'] = np.arange(1, len(df) + 1)
    print(f'  Sources with both rankings in common set: {len(df):,}')
    return df


def load_inst_rankings(rk_path: Path, inst_ids: np.ndarray) -> pd.DataFrame:
    """
    Load v from 0110 baseline and 0001 II runs for institutions.
    Returns DataFrame with columns: unit_idx, v_0110, v_0001, baseline_rank.
    """
    iid_set = set(inst_ids.tolist())
    with duckdb.connect(str(rk_path), read_only=True) as db:
        tables = {r[0] for r in db.execute('SHOW TABLES').fetchall()}

        def _load(table, col):
            if table not in tables:
                raise RuntimeError(f'{table} not found in rankings.duckdb')
            return db.execute(
                f"SELECT unit_idx, v FROM {table} WHERE unit_type='U'"
            ).df().rename(columns={'v': col})

        df0 = _load(BASELINE_TABLE, 'v_0110')
        df1 = _load(II_TABLE,       'v_0001')

    df = df0.merge(df1, on='unit_idx', how='inner')
    df = df[df['unit_idx'].isin(iid_set)].copy()
    df = df.sort_values('v_0110', ascending=False).reset_index(drop=True)
    df['baseline_rank'] = np.arange(1, len(df) + 1)
    print(f'  Institutions with both rankings in common set: {len(df):,}')
    return df


# ─── Community report ────────────────────────────────────────────────────────

def report_communities(df: pd.DataFrame, source_ids: np.ndarray,
                       membership: np.ndarray, label: str, paths,
                       top_n: int = 8) -> None:
    """Print top_n sources by v_0110 for each community. Saves CSV to plots/."""
    sm_path = paths.parquet / 'source_master.parquet'
    if sm_path.exists():
        sm = pd.read_parquet(sm_path, columns=['source_idx', 'source_name'])
    else:
        sm = pd.read_csv(
            Path(__file__).parent.parent / 'data' / 'source_master.csv',
            usecols=['source_idx', 'source_name'],
        )
    name_map = sm.set_index('source_idx')['source_name'].to_dict()

    sid_to_idx = {sid: i for i, sid in enumerate(source_ids.tolist())}
    df = df.copy()
    df['dense_idx'] = df['unit_idx'].map(sid_to_idx)
    df = df.dropna(subset=['dense_idx']).copy()
    df['dense_idx'] = df['dense_idx'].astype(int)
    df['community'] = membership[df['dense_idx'].values]
    df['source_name'] = df['unit_idx'].map(name_map).fillna('?')

    sizes = df['community'].value_counts().sort_values(ascending=False)

    print(f'\n{"─"*70}')
    print(f'Community report: {label}')
    print(f'{"─"*70}')

    rows_out = []
    for cid in sizes.index:
        sub = df[df['community'] == cid].sort_values('v_0110', ascending=False)
        print(f'\nCommunity {cid}  (n={len(sub):,})')
        print(f'  {"source_name":<45}  {"v_0110":>7}  {"v_1000":>7}  {"a_p":>6}')
        print(f'  {"-"*45}  {"-------":>7}  {"-------":>7}  {"------":>6}')
        for _, row in sub.head(top_n).iterrows():
            v1 = f'{row["v_1000"]:.3f}' if pd.notna(row.get('v_1000')) else '  n/a '
            print(f'  {row["source_name"][:45]:<45}  '
                  f'{row["v_0110"]:7.3f}  {v1:>7}  {int(row["a_p"]):>6,}')
        for _, row in sub.iterrows():
            rows_out.append({
                'community': cid, 'community_size': len(sub),
                'source_name': row['source_name'], 'unit_idx': row['unit_idx'],
                'v_0110': row['v_0110'], 'v_1000': row.get('v_1000', np.nan),
                'a_p': row['a_p'],
            })

    out_csv = paths.plots / f'source_communities_{label.replace(" ", "_")}.csv'
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f'\n  Saved {out_csv}')


# ─── Plot ─────────────────────────────────────────────────────────────────────

_PALETTE = (
    ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    * 10
)
_SINGLETON_COLOUR = '#cccccc'
_MIN_COMMUNITY_SIZE = 10


def _community_colours(membership: np.ndarray) -> list[str]:
    sizes = np.bincount(membership)
    rank = np.argsort(sizes)[::-1]
    colour_map = {}
    ci = 0
    for cid in rank:
        if sizes[cid] >= _MIN_COMMUNITY_SIZE:
            colour_map[cid] = _PALETTE[ci]
            ci += 1
        else:
            colour_map[cid] = _SINGLETON_COLOUR
    return [colour_map[m] for m in membership]


def _draw_panel(ax, df: pd.DataFrame, dense_idx_col: str,
                v_single_col: str, v_bip_col: str,
                membership: np.ndarray, title: str) -> None:
    colours = _community_colours(membership)
    point_colours = [colours[i] for i in df[dense_idx_col].values]

    ax.scatter(
        df['baseline_rank'].values,
        df[v_single_col].values,
        c=point_colours,
        s=18, alpha=0.65, linewidths=0.0, zorder=2,
    )
    ax.plot(
        df['baseline_rank'].values,
        df[v_bip_col].values,
        color='black', linewidth=1.0, alpha=0.4, zorder=1,
    )
    ax.set_yscale('log')
    ax.set_ylim(0.002, 20)
    ax.axhline(1.0, color='#999999', linewidth=0.7, linestyle='--', zorder=0)
    ax.set_xlim(1, len(df))
    ax.set_xlabel('Baseline rank (m=0110)', labelpad=4)
    ax.set_ylabel(f'$v$ from {v_single_col} (log)', labelpad=4)
    ax.set_title(title, fontsize=9, pad=6)

    n_comm = len(np.unique(membership))
    ax.text(0.02, 0.03, f'{n_comm} communities  (grey < {_MIN_COMMUNITY_SIZE})',
            transform=ax.transAxes, fontsize=7.5, color='#555555', va='bottom')


def plot_communities(
    df_src: pd.DataFrame, source_ids: np.ndarray,
    mem_ss: np.ndarray, mem_bi_s: np.ndarray,
    df_inst: pd.DataFrame, inst_ids: np.ndarray,
    mem_ii: np.ndarray, mem_bi_i: np.ndarray,
    paths,
) -> None:

    def _attach_dense(df, ids):
        id_to_idx = {sid: i for i, sid in enumerate(ids.tolist())}
        df = df.copy()
        df['dense_idx'] = df['unit_idx'].map(id_to_idx)
        df = df.dropna(subset=['dense_idx']).copy()
        df['dense_idx'] = df['dense_idx'].astype(int)
        return df

    df_src  = _attach_dense(df_src,  source_ids)
    df_inst = _attach_dense(df_inst, inst_ids)

    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(wspace=0.32, hspace=0.40)

    _draw_panel(axes[0, 0], df_src,  'dense_idx', 'v_1000', 'v_0110', mem_ss,
                r'Sources — $\mathbf{A}_{SS}$ (direct citations)')
    _draw_panel(axes[0, 1], df_src,  'dense_idx', 'v_1000', 'v_0110', mem_bi_s,
                r'Sources — $\mathbf{A}_{bip}$ ($C_{SI}C_{IS}$)')
    _draw_panel(axes[1, 0], df_inst, 'dense_idx', 'v_0001', 'v_0110', mem_ii,
                r'Institutions — $\mathbf{A}_{II}$ (direct citations)')
    _draw_panel(axes[1, 1], df_inst, 'dense_idx', 'v_0001', 'v_0110', mem_bi_i,
                r'Institutions — $\mathbf{A}_{bip}$ ($C_{IS}C_{SI}$)')

    sup = fig.suptitle(
        'Leiden community structure  '
        '($\\gamma$=' + str(RESOLUTION) + ', coloured by community, grey = singleton)',
        fontsize=9, y=1.01,
    )

    out = paths.plots / 'source_communities.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'\nSaved {out}')

    sup.set_visible(False)
    fig.savefig(paths.plots / 'source_communities_latex.pdf', bbox_inches='tight')
    sup.set_visible(True)

    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    paths   = load_config()
    el_path = paths.working / 'edge_lists.duckdb'
    rk_path = paths.working / 'rankings.duckdb'

    for p in (el_path, rk_path):
        if not p.exists():
            raise FileNotFoundError(f'{p} not found.')

    print('=== Building adjacency matrices ===')
    source_ids, A_SS, A_bip_S, inst_ids, A_II, A_bip_I = build_matrices(el_path)

    print('\n=== Running Leiden ===')
    mem_ss,   Q_ss   = run_leiden(csr_to_igraph(A_SS),   'A_SS (direct sources)')
    mem_bi_s, Q_bi_s = run_leiden(csr_to_igraph(A_bip_S),'A_bipartite_S (SI·IS)')
    mem_ii,   Q_ii   = run_leiden(csr_to_igraph(A_II),   'A_II (direct institutions)')
    mem_bi_i, Q_bi_i = run_leiden(csr_to_igraph(A_bip_I),'A_bipartite_I (IS·SI)')

    print('\n=== Modularity summary ===')
    print(f'  {"":20s}  {"S/I-network":>12}  {"B-network":>12}')
    print(f'  {"─"*20}  {"─"*12}  {"─"*12}')
    print(f'  {"Sources":<20}  {Q_ss:12.4f}  {Q_bi_s:12.4f}')
    print(f'  {"Institutions":<20}  {Q_ii:12.4f}  {Q_bi_i:12.4f}')

    print('\n=== Loading rankings ===')
    df_src  = load_rankings(rk_path, source_ids)
    df_inst = load_inst_rankings(rk_path, inst_ids)

    print('\n=== Community reports (sources) ===')
    report_communities(df_src, source_ids, mem_ss,   'A_SS',          paths)
    report_communities(df_src, source_ids, mem_bi_s, 'A_bipartite_S', paths)

    print('\n=== Plotting ===')
    plot_communities(
        df_src,  source_ids, mem_ss,   mem_bi_s,
        df_inst, inst_ids,   mem_ii,   mem_bi_i,
        paths,
    )


if __name__ == '__main__':
    main()
    print('FINISHED!')
