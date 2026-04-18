"""
community.py — Second eigenpair analysis across network modes.

Standalone script; reads edge_lists.duckdb and source_master.parquet.
Writes nothing to any database.

Computes λ₂, φ₂, spectral gap and community amplification for:
    SS       m=(1,0,0,0)  — source-only
    II       m=(0,0,0,1)  — institution-only
    bipartite m=(0,1,1,0) — one-mode projection M_S = H_SI @ H_IS
    full χ=0.5 m=(1,1,1,1)
    full χ=χ*  m=(1,1,1,1)

α is a per-hop parameter with the same meaning in all modes.  For the
bipartite case, M_S = H_SI @ H_IS represents a two-hop round trip, so
power iteration uses α² as the round-trip damping.  Amplification for
M_S is therefore 1/(1−α²·λ₂(M_S)), consistent with rank().

Usage
-----
    python spectral_results_analysis/community.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import duckdb
from scipy.sparse import bmat
from scipy.sparse.linalg import eigs

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import load_config, load_runs
from spectral_ranking.build_csr import build_csr
from spectral_ranking.katz_ranker import _row_normalise


# ─── Second eigenpair helpers ─────────────────────────────────────────────────

def _second_eigenpair(M, alpha, label=''):
    """
    Compute λ₂ and φ₂ of H̃^T, where H̃ = αH + (1−α)μ1^T.

    H̃ is always primitive regardless of dangling rows, so λ₁=1 and λ₂ < 1
    always. For k≥2, λ_k(H̃) = α·λ_k(H), so community structure is
    preserved. Reported λ₂ and gap are those of H̃.

    Uses scipy eigs(H̃.T, k=2, which='LM'). Returns real parts.
    Warns if imaginary component of λ₂ is non-negligible.

    Sign convention: element of φ₂ with largest absolute value is positive.

    Returns: (lambda2, phi2, gap, amplification)
        gap           = 1 − Re(λ₂(H̃))
        amplification = 1 / (1 − α · Re(λ₂(H̃)))
          (note: λ₂(H̃) = α·λ₂(H), so amplification = 1/(1 − α²·λ₂(H)) for SS/II)
    """
    from scipy.sparse import diags as sp_diags
    N = M.shape[0]
    mu = np.full(N, 1.0 / N)
    # H̃ = α·H + (1−α)·μ·1^T  — adding rank-1 teleportation
    # For eigs we don't need to form H̃ explicitly; use LinearOperator
    from scipy.sparse.linalg import LinearOperator

    def matvec(x):
        return alpha * M.T.dot(x) + (1.0 - alpha) * mu * x.sum()

    Htilde_T = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    vals, vecs = eigs(Htilde_T, k=2, which='LM')

    # Sort by descending real part; index 1 is λ₂
    order = np.argsort(-vals.real)
    lam2  = vals[order[1]]
    phi2  = vecs[:, order[1]]

    if abs(lam2.imag) / (abs(lam2) + 1e-12) > 0.01:
        warnings.warn(
            f"[{label}] λ₂ has non-negligible imaginary part: {lam2:.4f}. "
            "Using real part only."
        )

    lam2_r = lam2.real
    phi2_r = phi2.real

    # Sign convention: unit with max |φ₂| is positive
    max_idx = np.argmax(np.abs(phi2_r))
    if phi2_r[max_idx] < 0:
        phi2_r = -phi2_r

    gap  = 1.0 - lam2_r
    ampl = 1.0 / (1.0 - alpha * lam2_r)

    return lam2_r, phi2_r, gap, ampl


def second_eigenpair_unipartite(C, alpha, label=''):
    """SS or II: row-normalise C → H, then compute second eigenpair of H^T."""
    H, _ = _row_normalise(C)
    return _second_eigenpair(H, alpha, label)


def second_eigenpair_bipartite(C_SI, C_IS, alpha, label='bipartite'):
    """
    Bipartite M_S = H_SI @ H_IS.
    α is per-hop; M_S represents a two-hop round trip, so _second_eigenpair
    is called with α² as the round-trip damping, matching rank().
    Amplification = 1/(1−α²·λ₂(M_S)).
    Returns (lambda2, phi2_S, phi2_I, gap, amplification).
    phi2_I = H_SI.T @ phi2_S, normalised to unit Euclidean norm.
    """
    H_SI, _ = _row_normalise(C_SI)
    H_IS, _ = _row_normalise(C_IS)
    M_S = H_SI.dot(H_IS)

    lam2, phi2_S, gap, ampl = _second_eigenpair(M_S, alpha ** 2, label)

    # Institution community vector induced by the source walk
    phi2_I = H_SI.T.dot(phi2_S)
    norm = np.linalg.norm(phi2_I)
    if norm > 0:
        phi2_I /= norm

    return lam2, phi2_S, phi2_I, gap, ampl


def second_eigenpair_full(C_SS, C_SI, C_IS, C_II, chi, alpha, label='full'):
    """Full joint m=(1,1,1,1): assemble χ-scaled matrix, then eigenpair."""
    C = bmat(
        [[(1 - chi)**2 * C_SS,  chi*(1-chi) * C_SI],
         [chi*(1-chi)  * C_IS,  chi**2      * C_II]],
        format='csr',
    )
    H, _ = _row_normalise(C)
    return _second_eigenpair(H, alpha, label)


# ─── SCC report ──────────────────────────────────────────────────────────────

def scc_report(C, unit_ids, label, field_labels, source_names, small_threshold=5):
    """
    Compute strongly connected components of C and print a summary.

    Reports: number of SCCs, giant SCC size, size distribution of small SCCs,
    and names of all units not in the giant SCC.

    Parameters
    ----------
    C           : scipy sparse matrix (square)
    unit_ids    : array of original idx values (length = C.shape[0])
    label       : string label for printing
    field_labels, source_names : dicts keyed by original idx (may be None for inst)
    small_threshold : SCCs with size <= this are listed by member name
    """
    from scipy.sparse.csgraph import connected_components
    import collections

    n_comp, labels = connected_components(C, directed=True, connection='strong')

    sizes = collections.Counter(labels)
    giant_label = sizes.most_common(1)[0][0]
    giant_size  = sizes[giant_label]

    non_giant_sizes = sorted(
        [(sz, lbl) for lbl, sz in sizes.items() if lbl != giant_label],
        reverse=True
    )
    n_non_giant = sum(sz for sz, _ in non_giant_sizes)
    n_singletons = sum(1 for sz, _ in non_giant_sizes if sz == 1)

    print(f"\n{label} — SCC structure")
    print(f"  Total units : {C.shape[0]}")
    print(f"  SCCs        : {n_comp}  (giant={giant_size}, non-giant={n_non_giant}, singletons={n_singletons})")

    # Zero-out-row count (sinks within any SCC)
    row_sums = np.array(C.sum(axis=1)).ravel()
    n_sinks = int((row_sums == 0).sum())
    print(f"  Zero out-row (sinks): {n_sinks}")

    # List members of small SCCs
    small_scc_labels = {lbl for sz, lbl in non_giant_sizes if sz <= small_threshold}
    if small_scc_labels:
        print(f"  Non-giant SCCs (size ≤ {small_threshold}):")
        # Group by SCC label, sorted by size desc then label
        by_lbl = collections.defaultdict(list)
        for i, lbl in enumerate(labels):
            if lbl in small_scc_labels:
                by_lbl[lbl].append(i)
        for lbl in sorted(by_lbl, key=lambda l: -len(by_lbl[l])):
            members = by_lbl[lbl]
            sz = len(members)
            print(f"    SCC size={sz}:")
            for i in members:
                uid  = int(unit_ids[i])
                fl   = (field_labels or {}).get(uid) or '-'
                name = (source_names or {}).get(uid, f'idx={uid}')
                out  = row_sums[i]
                print(f"      out={out:6.1f}  F={fl:>4}  {name}")

    # Also list larger non-giant SCCs by size only (no member names)
    large_non_giant = [(sz, lbl) for sz, lbl in non_giant_sizes if sz > small_threshold]
    if large_non_giant:
        print(f"  Larger non-giant SCCs (size > {small_threshold}): "
              + ", ".join(f"{sz}" for sz, _ in large_non_giant))

    return labels, giant_label


# ─── Community detection ─────────────────────────────────────────────────────

def community_analysis(H, unit_ids, label, field_labels=None, seed=42):
    """
    Leiden community detection on directed stochastic matrix H.

    Steps:
      1. Prune nodes with both in-degree=0 AND out-degree=0 (isolated).
      2. Restrict to the giant SCC of the remaining graph.
      3. Run Leiden (ModularityVertexPartition, directed) on the weighted
         directed igraph built from the giant-SCC submatrix.
      4. Report: n_communities, modularity Q, mean intra-community edge
         density, and community composition (size, field mix if available).

    Edge weights are the stochastic matrix entries (row sums ≤ 1).
    """
    import collections
    import igraph as ig
    import leidenalg
    from scipy.sparse.csgraph import connected_components

    H = H.tocsr()
    n = H.shape[0]

    # 1. Remove completely isolated nodes (in-degree=0 AND out-degree=0)
    out_deg = np.array(H.sum(axis=1)).ravel()
    in_deg  = np.array(H.sum(axis=0)).ravel()
    keep    = np.where((out_deg > 0) | (in_deg > 0))[0]
    H_sub   = H[keep][:, keep]
    ids_sub = unit_ids[keep]

    # 2. Giant SCC
    n_comp, comp_labels = connected_components(H_sub, directed=True,
                                               connection='strong')
    sizes       = collections.Counter(comp_labels)
    giant_label = sizes.most_common(1)[0][0]
    scc_mask    = np.where(comp_labels == giant_label)[0]
    H_scc       = H_sub[scc_mask][:, scc_mask]
    ids_scc     = ids_sub[scc_mask]
    n_scc       = H_scc.shape[0]

    print(f"\n{label} — community analysis")
    print(f"  matrix size: {n}  after isolate prune: {len(keep)}"
          f"  SCCs: {n_comp}  giant SCC: {n_scc}")

    # 3. Build igraph directed weighted graph
    H_coo = H_scc.tocoo()
    # Remove self-loops (diagonal) for community detection
    mask_off = H_coo.row != H_coo.col
    edges    = list(zip(H_coo.row[mask_off].tolist(),
                        H_coo.col[mask_off].tolist()))
    weights  = H_coo.data[mask_off].tolist()

    g = ig.Graph(n=n_scc, edges=edges, directed=True)
    g.es['weight'] = weights

    # 4. Leiden with directed modularity
    partition = leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition,
        weights='weight',
        seed=seed,
    )
    membership = np.array(partition.membership)
    Q          = partition.modularity
    n_comm     = len(partition)

    # 5. Intra-community edge density per community
    # density_c = total intra-community weight / (n_c * (n_c - 1))
    H_scc_arr = H_scc.toarray()
    densities = []
    for c in range(n_comm):
        idx_c = np.where(membership == c)[0]
        n_c   = len(idx_c)
        if n_c < 2:
            densities.append(0.0)
            continue
        sub = H_scc_arr[np.ix_(idx_c, idx_c)]
        np.fill_diagonal(sub, 0.0)
        densities.append(sub.sum() / (n_c * (n_c - 1)))
    mean_density = float(np.mean(densities))

    print(f"  communities: {n_comm}  Q={Q:.4f}  "
          f"mean intra-community edge density: {mean_density:.4f}")

    # 6. Per-community summary (size + field mix if available)
    comm_sizes = collections.Counter(membership)
    print(f"  {'Comm':>5}  {'n':>6}  {'density':>8}  field mix")
    for c in sorted(comm_sizes, key=lambda x: -comm_sizes[x]):
        idx_c = np.where(membership == c)[0]
        n_c   = len(idx_c)
        field_counts: dict[str, int] = collections.Counter()
        for i in idx_c:
            uid = int(ids_scc[i])
            if field_labels:
                field_counts[field_labels.get(uid, 'X')] += 1
        mix = '  '.join(f"{k}:{v}" for k, v in
                        sorted(field_counts.items()) if v > 0)
        print(f"  {c:>5}  {n_c:>6}  {densities[c]:>8.4f}  {mix}")

    return membership, ids_scc, Q, n_comm, densities


# ─── Field labels ─────────────────────────────────────────────────────────────

def load_source_meta(parquet_path):
    """
    Returns two dicts keyed by source_idx (int):
        field_labels  : {source_idx: 'E'|'B'|None}
        source_names  : {source_idx: source_name str}
    """
    with duckdb.connect() as db:
        df = db.sql(
            f"SELECT source_idx, source_name, field_eb FROM '{parquet_path}'"
        ).df()
    idx = df['source_idx'].astype(int)
    field_labels = dict(zip(idx, df['field_eb']))
    source_names = dict(zip(idx, df['source_name']))
    return field_labels, source_names


# ─── Summary printer ──────────────────────────────────────────────────────────

def print_summary(rows):
    """
    rows: list of (label, lambda2, gap, amplification)
    """
    print(f"\n{'Case':<22} {'λ₂':>8} {'gap':>8} {'amplification':>14}")
    print('-' * 56)
    for label, lam2, gap, ampl in rows:
        print(f"{label:<22} {lam2:>8.5f} {gap:>8.5f} {ampl:>14.4f}")


def print_phi2_top(label, phi2, unit_ids, field_labels, source_names, n=15):
    """Print top n positive and top n negative units in φ₂ with names."""
    order = np.argsort(phi2)
    print(f"\n{label} — φ₂ extremes (sign: max|φ₂| unit = positive)")
    print(f"  {'φ₂':>9}  {'F':>4}  name")
    print(f"  --- negative end ---")
    for i in order[:n]:
        idx  = int(unit_ids[i])
        fl   = field_labels.get(idx) or '-'
        name = source_names.get(idx, f'idx={idx}')
        print(f"  {phi2[i]:>9.5f}  {fl:>4}  {name}")
    print(f"  --- positive end ---")
    for i in order[-n:][::-1]:
        idx  = int(unit_ids[i])
        fl   = field_labels.get(idx) or '-'
        name = source_names.get(idx, f'idx={idx}')
        print(f"  {phi2[i]:>9.5f}  {fl:>4}  {name}")


# ─── Main driver ──────────────────────────────────────────────────────────────

def run(paths, run_code: str = '20242024', fx: str = 'A',
        tau_u: int = 20, tau_s: int = 20, rho: int = 0, alpha: float = 1.0):
    el_path = paths.working / 'edge_lists.duckdb'
    sm_path = paths.parquet / 'source_master.parquet'

    field_labels, source_names = load_source_meta(sm_path)
    print(f"Source meta loaded: {len(field_labels)} sources")
    print(f"run_code={run_code}  fx={fx}  tau_u={tau_u}  tau_s={tau_s}  "
          f"rho={rho}  alpha={alpha}")

    summary = []

    with duckdb.connect(str(el_path), read_only=True) as db:

        # ── χ* ────────────────────────────────────────────────────────────────
        uname = f'_units_{run_code}_{fx}_tauU{tau_u}_tauS{tau_s}_vartau_m0110'
        counts = {r[0]: r[1] for r in db.execute(
            f"SELECT unit_type, COUNT(*) FROM {uname} GROUP BY unit_type"
        ).fetchall()}
        n_s, n_u = counts.get('S', 0), counts.get('U', 0)
        chi_star = n_u / (n_s + n_u) if (n_s + n_u) > 0 else 0.5
        print(f"N_s={n_s}, N_u={n_u}, χ*={chi_star:.4f}")

        # ── SS ────────────────────────────────────────────────────────────────
        print("\n=== SS mode ===")
        csr = build_csr(db, run_code, fx, tau_u, tau_s, rho, (1, 0, 0, 0))
        assert csr.C_SS is not None
        scc_report(csr.C_SS, csr.source_ids, 'C_SS', field_labels, source_names)
        lam2, phi2, gap, ampl = second_eigenpair_unipartite(csr.C_SS, alpha, 'SS')
        summary.append(('SS', lam2, gap, ampl))
        print_phi2_top('SS', phi2, csr.source_ids, field_labels, source_names)
        H_SS, _ = _row_normalise(csr.C_SS)
        community_analysis(H_SS, csr.source_ids, 'H_SS', field_labels)

        # ── II ────────────────────────────────────────────────────────────────
        print("\n=== II mode ===")
        csr = build_csr(db, run_code, fx, tau_u, tau_s, rho, (0, 0, 0, 1))
        assert csr.C_II is not None
        scc_report(csr.C_II, csr.inst_ids, 'C_II', None, None)
        lam2, phi2, gap, ampl = second_eigenpair_unipartite(csr.C_II, alpha, 'II')
        summary.append(('II', lam2, gap, ampl))
        H_II, _ = _row_normalise(csr.C_II)
        community_analysis(H_II, csr.inst_ids, 'H_II')

        # ── Bipartite ─────────────────────────────────────────────────────────
        print("\n=== Bipartite SI/IS ===")
        csr = build_csr(db, run_code, fx, tau_u, tau_s, rho, (0, 1, 1, 0))
        assert csr.C_SI is not None and csr.C_IS is not None
        lam2, phi2_S, phi2_I, gap, ampl = second_eigenpair_bipartite(
            csr.C_SI, csr.C_IS, alpha, 'bipartite')
        summary.append(('bipartite M_S', lam2, gap, ampl))
        print_phi2_top('bipartite M_S (sources)', phi2_S, csr.source_ids,
                       field_labels, source_names)
        H_SI, _ = _row_normalise(csr.C_SI)
        H_IS, _ = _row_normalise(csr.C_IS)
        M_S = H_SI.dot(H_IS)
        M_I = H_IS.dot(H_SI)
        community_analysis(M_S, csr.source_ids, 'M_S (H_SI·H_IS)', field_labels)
        community_analysis(M_I, csr.inst_ids,   'M_I (H_IS·H_SI)')

        # ── Full joint χ* ─────────────────────────────────────────────────────
        print("\n=== Full joint χ=0.5 ===")
        csr = build_csr(db, run_code, fx, tau_u, tau_s, rho, (1, 1, 1, 1))
        assert (csr.C_SS is not None and csr.C_SI is not None
                and csr.C_IS is not None and csr.C_II is not None)
        from scipy.sparse import bmat as sp_bmat
        C_full = sp_bmat(
            [[csr.C_SS, csr.C_SI],
             [csr.C_IS, csr.C_II]], format='csr'
        )
        full_ids = np.concatenate([csr.source_ids, csr.inst_ids])
        scc_report(C_full, full_ids, 'C_full (joint)', field_labels, source_names)
        lam2, phi2, gap, ampl = second_eigenpair_full(
            csr.C_SS, csr.C_SI, csr.C_IS, csr.C_II, 0.5, alpha, 'full-0.5')
        summary.append(('full χ=0.5', lam2, gap, ampl))

        print(f"\n=== Full joint χ*={chi_star:.4f} ===")
        lam2, phi2, gap, ampl = second_eigenpair_full(
            csr.C_SS, csr.C_SI, csr.C_IS, csr.C_II, chi_star, alpha, 'full-chi*')
        summary.append((f'full χ*={chi_star:.3f}', lam2, gap, ampl))
        C_full_chi = sp_bmat(
            [[(1 - chi_star)**2 * csr.C_SS,  chi_star*(1-chi_star) * csr.C_SI],
             [chi_star*(1-chi_star) * csr.C_IS,  chi_star**2      * csr.C_II]],
            format='csr',
        )
        from scipy.sparse import csr_matrix as _csr
        C_full_chi = _csr(C_full_chi)
        H_ISSI, _ = _row_normalise(C_full_chi)
        community_analysis(H_ISSI, full_ids, f'H_ISSI (full χ*={chi_star:.3f})',
                           field_labels)

    print_summary(summary)


def main():
    paths = load_config()
    # Derive baseline parameters from params.csv
    baseline = next(r for r in load_runs() if r['label'] == 'baseline')
    run(paths,
        run_code=baseline['run_code'],
        fx=baseline['fx'],
        tau_u=baseline['tau_u'],
        tau_s=baseline['tau_s'],
        rho=baseline['rho'],
        alpha=float(baseline['alpha']))


if __name__ == '__main__':
    main()
