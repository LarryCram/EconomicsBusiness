"""
source_communities.py — Leiden community structure of source citation networks.

Builds two directed weighted graphs on the source node set:

  A_SS          = C_SS                  (raw source-to-source citation counts)
  A_bipartite   = C_SI @ C_IS          (institution-mediated source-to-source,
                                         raw counts, no row-normalisation)

Runs Leiden community detection on each, then plots the Fig-3 scatter
(x = 0110 baseline rank, y = v from m=1000) coloured by community membership
to test whether the outliers cluster into a distinct community.

All runs: F=A, τ_U=τ_S=20, ρ=0, baseline time window.
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

BASELINE_TABLE = f'rk_{_run_code}_A_tauU{_tau_u}_tauS{_tau_s}_rho0_m0110_chi50_alpha100'
SS_TABLE       = f'rk_{_run_code}_A_tauU{_tau_u}_tauS{_tau_s}_rho0_m1000_chi50_alpha100'

# Leiden resolution parameter (γ); increase to get finer communities
RESOLUTION = 1.0
N_ITERATIONS = -1   # run until convergence


# ─── Matrix construction ──────────────────────────────────────────────────────

def row_normalise(M: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalise a CSR matrix; zero rows stay zero."""
    row_sums = np.asarray(M.sum(axis=1)).ravel()
    inv = np.where(row_sums > 0, 1.0 / row_sums, 0.0)
    return sp.diags(inv) @ M


def build_matrices(el_path: Path) -> tuple[np.ndarray, sp.csr_matrix, sp.csr_matrix]:
    """
    Returns
    -------
    source_ids   : (n_s,) int64 array of OpenAlex source IDs
    A_SS         : (n_s × n_s) CSR — raw C_SS
    A_bipartite  : (n_s × n_s) CSR — C_SI @ C_IS (raw, unscaled)
    """
    with duckdb.connect(str(el_path)) as db:
        print('Building C_SS ...')
        csr_ss = build_csr(db, _run_code, 'A', _tau_u, _tau_s, 0, (1, 0, 0, 0))
        print('Building C_SI, C_IS ...')
        csr_si_is = build_csr(db, _run_code, 'A', _tau_u, _tau_s, 0, (0, 1, 1, 0))

    # Intersect source sets (SCC filtering may differ between modes)
    ids_ss  = set(csr_ss.source_ids.tolist())
    ids_bi  = set(csr_si_is.source_ids.tolist())
    common  = np.array(sorted(ids_ss & ids_bi), dtype=np.int64)
    n_common = len(common)
    print(f'  Sources in SS:         {csr_ss.n_s:,}')
    print(f'  Sources in bipartite:  {csr_si_is.n_s:,}')
    print(f'  Common source set:     {n_common:,}')

    # Restrict both matrices to common source set
    def reindex(C: sp.csr_matrix, old_ids: np.ndarray) -> sp.csr_matrix:
        idx = pd.Index(old_ids).get_indexer(common)
        keep = idx >= 0
        # Select rows then columns
        C_rows = C[idx[keep], :][:, idx[keep]]   # Note: idx has no -1 by construction
        return C_rows.tocsr()

    A_SS = reindex(csr_ss.C_SS, csr_ss.source_ids)

    # A_bipartite = C_SI @ C_IS, restricted to common sources
    idx_bi = pd.Index(csr_si_is.source_ids).get_indexer(common)
    C_SI_r = csr_si_is.C_SI[idx_bi, :]
    C_IS_r = csr_si_is.C_IS[:, idx_bi]
    A_bipartite = (C_SI_r @ C_IS_r).tocsr()

    print(f'  A_SS        nnz: {A_SS.nnz:,}')
    print(f'  A_bipartite nnz: {A_bipartite.nnz:,}')

    return common, A_SS, A_bipartite


# ─── igraph / Leiden ──────────────────────────────────────────────────────────

def csr_to_igraph(A: sp.csr_matrix) -> ig.Graph:
    """Convert a square CSR matrix to a directed weighted igraph Graph."""
    A_coo = A.tocoo()
    # Remove self-loops
    mask = A_coo.row != A_coo.col
    edges = list(zip(A_coo.row[mask].tolist(), A_coo.col[mask].tolist()))
    weights = A_coo.data[mask].tolist()
    g = ig.Graph(n=A.shape[0], edges=edges, directed=True)
    g.es['weight'] = weights
    return g


def run_leiden(g: ig.Graph, label: str) -> np.ndarray:
    """Run Leiden with directed modularity; return integer community array."""
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
    return membership


# ─── Ranking data ─────────────────────────────────────────────────────────────

def load_rankings(rk_path: Path, source_ids: np.ndarray) -> pd.DataFrame:
    """
    Load v from 0110 baseline and 1000 SS runs.
    Returns DataFrame with columns:
      unit_idx, v_0110, v_1000, baseline_rank
    Restricted to the common source_ids.
    """
    sid_set = set(source_ids.tolist())
    with duckdb.connect(str(rk_path), read_only=True) as db:
        def load(table, extra=''):
            tables = {r[0] for r in db.execute('SHOW TABLES').fetchall()}
            if table not in tables:
                raise RuntimeError(f'{table} not found in rankings.duckdb')
            return db.execute(
                f"SELECT unit_idx, v{extra} FROM {table} WHERE unit_type='S'"
            ).df()

        df0 = load(BASELINE_TABLE, ', a_p').rename(columns={'v': 'v_0110'})
        df1 = load(SS_TABLE).rename(columns={'v': 'v_1000'})

    df = df0.merge(df1, on='unit_idx', how='inner')
    df = df[df['unit_idx'].isin(sid_set)].copy()

    df = df.sort_values('v_0110', ascending=False).reset_index(drop=True)
    df['baseline_rank'] = np.arange(1, len(df) + 1)

    print(f'\n  Sources with both rankings in common set: {len(df):,}')
    return df


# ─── Community report ────────────────────────────────────────────────────────

def report_communities(df: pd.DataFrame, source_ids: np.ndarray,
                       membership: np.ndarray, label: str, paths,
                       top_n: int = 8) -> None:
    """
    Print top_n sources by v_0110 for each community, with work count a_p.
    Also saves a CSV to plots/.
    """
    # Load source names
    sm_path = paths.parquet / 'source_master.parquet'
    if sm_path.exists():
        sm = pd.read_parquet(sm_path, columns=['source_idx', 'source_name'])
    else:
        sm = pd.read_csv(
            Path(__file__).parent.parent / 'data' / 'source_master.csv',
            usecols=['source_idx', 'source_name'],
        )
    name_map = sm.set_index('source_idx')['source_name'].to_dict()

    # Attach community and name to df
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

# Colour palette: large enough for many communities; singletons in light grey
_PALETTE = (
    ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    * 10
)
_SINGLETON_COLOUR = '#cccccc'
_MIN_COMMUNITY_SIZE = 10   # communities smaller than this shown as grey


def _community_colours(membership: np.ndarray) -> list[str]:
    sizes = np.bincount(membership)
    # Rank communities by size descending for stable colour assignment
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


def plot_communities(df: pd.DataFrame, source_ids: np.ndarray,
                     mem_ss: np.ndarray, mem_bi: np.ndarray,
                     paths) -> None:
    # Map unit_idx → dense index in source_ids array
    sid_to_idx = {sid: i for i, sid in enumerate(source_ids.tolist())}
    df = df.copy()
    df['dense_idx'] = df['unit_idx'].map(sid_to_idx)
    df = df.dropna(subset=['dense_idx']).copy()
    df['dense_idx'] = df['dense_idx'].astype(int)

    df['comm_ss'] = mem_ss[df['dense_idx'].values]
    df['comm_bi'] = mem_bi[df['dense_idx'].values]

    sns.set_theme(style='whitegrid', font_scale=0.95)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.32)

    n_base = len(df)

    for ax, comm_col, title, membership in [
        (axes[0], 'comm_ss', 'Communities in $\\mathbf{A}_{SS}$ (direct citations)', mem_ss),
        (axes[1], 'comm_bi', 'Communities in $\\mathbf{A}_{bip}$ ($C_{SI}C_{IS}$)', mem_bi),
    ]:
        colours = _community_colours(membership)
        point_colours = [colours[i] for i in df['dense_idx'].values]

        ax.scatter(
            df['baseline_rank'].values,
            df['v_1000'].values,
            c=point_colours,
            s=18,
            alpha=0.65,
            linewidths=0.0,
            zorder=2,
        )

        # Baseline v_0110 curve for reference
        ax.plot(
            df['baseline_rank'].values,
            df['v_0110'].values,
            color='black', linewidth=1.0, alpha=0.4, zorder=1,
            label='$v$ (m=0110)',
        )

        ax.set_yscale('log')
        ax.set_ylim(0.002, 20)
        ax.axhline(1.0, color='#999999', linewidth=0.7, linestyle='--', zorder=0)
        ax.set_xlim(1, n_base)
        ax.set_xlabel('Baseline rank (m=0110)', labelpad=4)
        ax.set_ylabel('$v$ from m=1000 (log)', labelpad=4)
        ax.set_title(title, fontsize=9, pad=6)

        n_comm = len(np.unique(membership))
        ax.text(0.02, 0.03, f'{n_comm} communities  (grey < {_MIN_COMMUNITY_SIZE})',
                transform=ax.transAxes, fontsize=7.5, color='#555555',
                va='bottom')

    sup = fig.suptitle(
        'Leiden community structure vs Fig-3 outliers  '
        '(coloured by community, $\\gamma$=' + str(RESOLUTION) + ')',
        fontsize=9, y=1.02,
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
    source_ids, A_SS, A_bipartite = build_matrices(el_path)

    print('\n=== Running Leiden ===')
    g_ss = csr_to_igraph(A_SS)
    g_bi = csr_to_igraph(A_bipartite)
    mem_ss = run_leiden(g_ss, 'A_SS (direct)')
    mem_bi = run_leiden(g_bi, 'A_bipartite (SI·IS)')

    print('\n=== Loading rankings ===')
    df = load_rankings(rk_path, source_ids)

    print('\n=== Community reports ===')
    report_communities(df, source_ids, mem_ss, 'A_SS', paths)
    report_communities(df, source_ids, mem_bi, 'A_bipartite', paths)

    print('\n=== Plotting ===')
    plot_communities(df, source_ids, mem_ss, mem_bi, paths)


if __name__ == '__main__':
    main()
    print('FINISHED!')
