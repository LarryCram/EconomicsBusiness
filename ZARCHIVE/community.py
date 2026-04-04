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
from util import load_config, load_params
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

def run(paths, params):
    tau_u = params['tau_u_floor']['A']
    tau_s = params['tau_s_floor']['A']
    alpha = 1.0
    tx, fx, rho = 5, 'A', 0

    el_path = paths.working / 'edge_lists.duckdb'
    sm_path = paths.parquet / 'source_master.parquet'

    field_labels, source_names = load_source_meta(sm_path)
    print(f"Source meta loaded: {len(field_labels)} sources")

    summary = []

    with duckdb.connect(str(el_path), read_only=True) as db:

        # ── χ* ────────────────────────────────────────────────────────────────
        uname = f'_units_t{tx}_{fx}_tauU{tau_u}_tauS{tau_s}'
        counts = {r[0]: r[1] for r in db.execute(
            f"SELECT unit_type, COUNT(*) FROM {uname} GROUP BY unit_type"
        ).fetchall()}
        n_s, n_u = counts['S'], counts['U']
        chi_star = n_u / (n_s + n_u)
        print(f"N_s={n_s}, N_u={n_u}, χ*={chi_star:.4f}")

        # ── SS ────────────────────────────────────────────────────────────────
        print("\n=== SS mode ===")
        csr = build_csr(db, tx, fx, tau_u, tau_s, rho, (1,0,0,0))
        scc_report(csr.C_SS, csr.source_ids, 'C_SS', field_labels, source_names)
        lam2, phi2, gap, ampl = second_eigenpair_unipartite(csr.C_SS, alpha, 'SS')
        summary.append(('SS', lam2, gap, ampl))
        print_phi2_top('SS', phi2, csr.source_ids, field_labels, source_names)

        # ── II ────────────────────────────────────────────────────────────────
        print("\n=== II mode ===")
        csr = build_csr(db, tx, fx, tau_u, tau_s, rho, (0,0,0,1))
        scc_report(csr.C_II, csr.inst_ids, 'C_II', None, None)
        lam2, phi2, gap, ampl = second_eigenpair_unipartite(csr.C_II, alpha, 'II')
        summary.append(('II', lam2, gap, ampl))

        # ── Bipartite ─────────────────────────────────────────────────────────
        print("\n=== Bipartite SI/IS ===")
        csr = build_csr(db, tx, fx, tau_u, tau_s, rho, (0,1,1,0))
        lam2, phi2_S, phi2_I, gap, ampl = second_eigenpair_bipartite(
            csr.C_SI, csr.C_IS, alpha, 'bipartite')
        summary.append(('bipartite M_S', lam2, gap, ampl))
        print_phi2_top('bipartite M_S (sources)', phi2_S, csr.source_ids, field_labels, source_names)

        # ── Full joint χ=0.5 ──────────────────────────────────────────────────
        print("\n=== Full joint χ=0.5 ===")
        csr = build_csr(db, tx, fx, tau_u, tau_s, rho, (1,1,1,1))
        # SCC on the assembled block matrix (sources first, then institutions)
        from scipy.sparse import bmat as sp_bmat
        C_full = sp_bmat(
            [[csr.C_SS, csr.C_SI],
             [csr.C_IS, csr.C_II]], format='csr'
        )
        full_ids = np.concatenate([csr.source_ids, csr.inst_ids])
        # source_names only covers sources; institutions get idx=... fallback
        scc_report(C_full, full_ids, 'C_full (joint)', field_labels, source_names)
        lam2, phi2, gap, ampl = second_eigenpair_full(
            csr.C_SS, csr.C_SI, csr.C_IS, csr.C_II, 0.5, alpha, 'full-0.5')
        summary.append(('full χ=0.5', lam2, gap, ampl))

        # ── Full joint χ* ─────────────────────────────────────────────────────
        print(f"\n=== Full joint χ*={chi_star:.4f} ===")
        lam2, phi2, gap, ampl = second_eigenpair_full(
            csr.C_SS, csr.C_SI, csr.C_IS, csr.C_II, chi_star, alpha, 'full-chi*')
        summary.append((f'full χ*={chi_star:.3f}', lam2, gap, ampl))

    print_summary(summary)


def main():
    paths  = load_config()
    params = load_params()
    run(paths, params)


if __name__ == '__main__':
    main()
