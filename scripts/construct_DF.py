#!/usr/bin/env python3
# construct_DF.py
# Companion implementation for Appendix B pseudocode (minimal worked example).
# Produces D_F^{(0)}, D_F^{(1)}, D_F^{(2)} from synthetic spectral input.
#
# Requirements: numpy, scipy
# Usage: python3 construct_DF.py
#
# The script implements the algorithmic pipeline described in Appendix B.7:
#  - builds a finite basis (kernel or eigenmode truncation)
#  - orthonormalizes the basis
#  - constructs projectors and kernel matrix
#  - forms a small monomial set and evaluates coefficients by quadrature
#  - assembles D_F^{(j)}, enforces hermiticity, saves matrices and runs checks
#
# The minimal worked example uses:
#  - N_in = 40 synthetic spectral points (gaussian-distributed)
#  - N_F = 6 basis vectors (first 6 kernel modes)
#  - gaussian kernel K(lambda,lambda')
#  - simple monomials: P_i, K @ P_i, sym(K^2)
#
# All numeric outputs are saved as .npy files in the current directory and
# a short run log is printed to stdout.

from __future__ import annotations
import numpy as np
import scipy.linalg as la
import os
import json
from typing import List, Tuple, Dict

# ---------------------------
# Configuration / parameters
# ---------------------------
RNG_SEED = 123456
np.random.seed(RNG_SEED)

# Spectral input (minimal example)
N_in = 40           # number of spectral quadrature nodes
N_F = 6             # finite basis truncation
sigma_kernel = 0.5  # kernel width for gaussian kernel
w_default = 1.0     # default quadrature weight (uniform simple rule)

# Numerical tolerances for checks
EPS_ORDER = 1e-8
EPS_REALITY = 1e-10
EPS_CONV = 1e-6

# Output filenames
OUT_D0 = "D0.npy"
OUT_D1 = "D1.npy"
OUT_D2 = "D2.npy"
OUT_LOG = "construct_DF_runlog.json"

# ---------------------------
# Utility functions
# ---------------------------
def save_npy(mat: np.ndarray, fname: str) -> None:
    np.save(fname, mat)
    print(f"Saved {fname} (shape {mat.shape})")

def sha256_of_file(fname: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------
# Step 1: Synthetic spectral input
# ---------------------------
def make_spectral_input(n_in: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal synthetic spectral input: sorted positive spectral points and weights.
    """
    # Use a simple distribution of spectral points in [0.1, 5.0]
    lam = np.linspace(0.1, 5.0, n_in)
    # Optionally perturb slightly
    lam += 0.01 * np.random.randn(n_in)
    lam = np.sort(np.abs(lam))
    # Uniform weights for simple quadrature
    w = np.full(n_in, w_default * (lam[-1] - lam[0]) / float(n_in))
    return lam, w

# ---------------------------
# Step 2: Kernel definition
# ---------------------------
def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute Gaussian kernel matrix K(x_i, y_j) for vectors x,y.
    Returns matrix of shape (len(x), len(y)).
    """
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(1, -1)
    d2 = (x - y) ** 2
    return np.exp(-d2 / (2.0 * sigma ** 2))

# ---------------------------
# Step 3: Basis construction
# ---------------------------
def build_kernel_basis(lam: np.ndarray, w: np.ndarray, N_F: int, sigma: float) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Build a kernel projection basis: use the first N_F eigenvectors of the kernel
    discretized on the spectral nodes. Returns list of basis functions evaluated
    at the spectral nodes (phi_i(lambda_m)) as columns of a matrix Phi (n_in x N_F).
    """
    Kmat = gaussian_kernel(lam, lam, sigma)
    # Weighted kernel: incorporate quadrature weights symmetrically
    W = np.sqrt(np.diag(w))
    Kw = W @ Kmat @ W
    # Eigen-decomposition (symmetric)
    vals, vecs = la.eigh(Kw)
    # Sort descending
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    # Take first N_F modes
    vecs_trunc = vecs[:, :N_F]
    # Convert back to basis functions phi_i(lambda_m) = W^{-1} * vecs_trunc[:,i]
    Winv = np.linalg.inv(W)
    Phi = Winv @ vecs_trunc  # shape (n_in, N_F)
    # Normalize columns with respect to spectral inner product
    for i in range(Phi.shape[1]):
        norm = np.sqrt(np.sum(w * np.abs(Phi[:, i]) ** 2))
        if norm > 0:
            Phi[:, i] /= norm
    # Return list of column vectors and matrix
    basis_list = [Phi[:, i].copy() for i in range(Phi.shape[1])]
    return basis_list, Phi

# ---------------------------
# Step 4: Orthonormalization (numerically stable)
# ---------------------------
def orthonormalize_basis(Phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Orthonormalize columns of Phi with respect to weighted inner product
    <f,g> = sum_m w_m conj(f_m) g_m. Returns orthonormalized Phi (n_in x N_F).
    Uses weighted QR via forming Gram matrix.
    """
    # Compute Gram matrix G = Phi^* W Phi
    W = np.diag(w)
    G = Phi.conj().T @ (W @ Phi)
    # Cholesky or symmetric orthonormalization
    try:
        L = la.cholesky(G, lower=True)
        Linv = la.inv(L)
        Phi_orth = Phi @ Linv.T  # columns now orthonormal
    except la.LinAlgError:
        # fallback: symmetric orthonormalization via eigen
        vals, vecs = la.eigh(G)
        # discard tiny eigenvalues
        vals[vals < 1e-16] = 1e-16
        Ghalf_inv = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        Phi_orth = Phi @ Ghalf_inv
    # Re-normalize to ensure numerical orthonormality
    for i in range(Phi_orth.shape[1]):
        norm = np.sqrt(np.sum(w * np.abs(Phi_orth[:, i]) ** 2))
        if norm > 0:
            Phi_orth[:, i] /= norm
    return Phi_orth

# ---------------------------
# Step 5: Projectors and kernel matrix on finite basis
# ---------------------------
def compute_projectors_and_kernel(Phi: np.ndarray, lam: np.ndarray, w: np.ndarray, kernel_sigma: float) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Given Phi (n_in x N_F) with columns phi_i evaluated at spectral nodes,
    compute rank-one projectors P_i (N_F x N_F) in the finite Hilbert space
    representation (we represent operators in the basis of the phi_i themselves).
    Also compute the kernel matrix K_ij = <phi_i, K phi_j>_Espec (N_F x N_F).
    """
    n_in, N_F = Phi.shape
    # Projectors in the finite basis representation: P_i has matrix with 1 at (i,i)
    # when we represent operators in the basis {phi_i}. For clarity we will represent
    # operators as N_F x N_F matrices in the phi basis.
    P_list = []
    for i in range(N_F):
        E = np.zeros((N_F, N_F), dtype=np.complex128)
        E[i, i] = 1.0
        P_list.append(E)
    # Kernel matrix K_ij = sum_{m,n} w_m w_n conj(phi_i(lambda_m)) K(lambda_m,lambda_n) phi_j(lambda_n)
    K_full = gaussian_kernel(lam, lam, kernel_sigma)
    # Compute matrix of inner products: B_{i,m} = conj(phi_i(lambda_m)) * w_m
    B = (Phi.conj().T) * w.reshape(1, -1)  # shape (N_F, n_in)
    # Then K_ij = B @ K_full @ Phi  (N_F x N_F)
    Kmat = B @ (K_full @ Phi)
    # Ensure hermiticity by symmetrization
    Kmat = 0.5 * (Kmat + Kmat.conj().T)
    return P_list, Kmat

# ---------------------------
# Step 6: Representation projectors (toy)
# ---------------------------
def construct_representation_projectors(N_F: int) -> Dict[str, np.ndarray]:
    """
    Minimal mock of representation projectors for gauge factors.
    For the worked example we create three projectors (U1, SU2, SU3) that partition
    the finite basis indices. In a real run these are derived from the algebra embedding.
    """
    P = {}
    # Partition indices: first 2 -> SU3, next 2 -> SU2, rest -> U1 (toy)
    P_su3 = np.zeros((N_F, N_F), dtype=np.complex128)
    P_su2 = np.zeros((N_F, N_F), dtype=np.complex128)
    P_u1 = np.zeros((N_F, N_F), dtype=np.complex128)
    # assign blocks
    for i in range(N_F):
        if i < max(1, N_F // 3):
            P_su3[i, i] = 1.0
        elif i < 2 * max(1, N_F // 3):
            P_su2[i, i] = 1.0
        else:
            P_u1[i, i] = 1.0
    P['SU3'] = P_su3
    P['SU2'] = P_su2
    P['U1'] = P_u1
    return P

# ---------------------------
# Step 7: Monomial set formation
# ---------------------------
def form_monomials(P_list: List[np.ndarray], Kmat: np.ndarray, P_reps: Dict[str, np.ndarray]) -> Dict[str, List[np.ndarray]]:
    """
    Build a small set of monomials M_k^{(j)} for j=0,1,2.
    For the minimal example:
      - M^{(0)}: P_i  (rank-one projectors)
      - M^{(1)}: K @ P_i
      - M^{(2)}: sym(K^2)
    Returns dict with keys 'M0','M1','M2' each a list of N_F matrices.
    """
    M0 = [P.copy() for P in P_list]  # P_i
    M1 = [Kmat @ P for P in P_list]  # K P_i
    K2 = Kmat @ Kmat
    M2 = [0.5 * (K2 + K2.conj().T)]  # single sym(K^2)
    # Optionally include rep-projected monomials
    for a, Pa in P_reps.items():
        M0.append(Pa)
        M1.append(Kmat @ Pa)
    return {'M0': M0, 'M1': M1, 'M2': M2}

# ---------------------------
# Step 8: Coefficient evaluation by quadrature
# ---------------------------
def compute_coefficients_by_quadrature(Ms: Dict[str, List[np.ndarray]],
                                       Phi: np.ndarray,
                                       lam: np.ndarray,
                                       w: np.ndarray) -> Dict[str, np.ndarray]:
    """
    For each monomial M_k^{(j)} compute a scalar coefficient alpha_k^{(j)} by
    evaluating a model quadrature. In the minimal example we use a simple trace-based
    projection:
      alpha_k = Tr( M_k * K_proj ) / Tr(I)
    where K_proj is the kernel matrix represented in the finite basis (Kmat).
    This is a pragmatic choice for the worked example; in a real implementation
    F_k^{(j)} would be derived from cohomological kernels.
    """
    coeffs = {}
    # For the minimal example we use the finite-basis kernel representation Kmat
    # which we can obtain from Ms['M1'] building blocks (they contain K @ P_i).
    # Instead, we compute alpha as normalized trace of M_k times a reference operator R.
    # Choose R = identity for simplicity or R = sym(K)
    # Here we choose R = sym(K) approximated by averaging M1 contributions.
    # Build R:
    M1_list = Ms['M1']
    R = sum(M1_list) / max(1, len(M1_list))
    # Now compute coefficients
    for key in Ms:
        L = Ms[key]
        alphas = np.zeros(len(L), dtype=np.float64)
        for idx, M in enumerate(L):
            # scalar coefficient: real part of normalized trace
            tr = np.trace(M @ R)
            alphas[idx] = float(np.real(tr) / max(1.0, np.abs(np.trace(R))))
        coeffs[key] = alphas
    return coeffs

# ---------------------------
# Step 9: Assemble D_j and hermitize
# ---------------------------
def assemble_Dj(Ms: Dict[str, List[np.ndarray]], coeffs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble D0, D1, D2 from monomials and coefficients.
    """
    N = Ms['M0'][0].shape[0]
    D0 = np.zeros((N, N), dtype=np.complex128)
    D1 = np.zeros((N, N), dtype=np.complex128)
    D2 = np.zeros((N, N), dtype=np.complex128)
    for k, M in enumerate(Ms['M0']):
        D0 += coeffs['M0'][k] * M
    for k, M in enumerate(Ms['M1']):
        D1 += coeffs['M1'][k] * M
    for k, M in enumerate(Ms['M2']):
        D2 += coeffs['M2'][k] * M
    # Hermitize
    D0 = 0.5 * (D0 + D0.conj().T)
    D1 = 0.5 * (D1 + D1.conj().T)
    D2 = 0.5 * (D2 + D2.conj().T)
    return D0, D1, D2

# ---------------------------
# Step 10: Consistency checks (approximate)
# ---------------------------
def run_consistency_checks(D0: np.ndarray, D1: np.ndarray, D2: np.ndarray,
                           P_reps: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Run a set of simple numeric checks:
      - hermiticity residuals
      - mock order-one residuals: || [[D, a], b^circ] || for generator set
      - block projector trace (diagnostic)
      - physical U(1) generator trace and unimodularity correction (toy example)
      - reality: check J D J^{-1} - D (here J is complex conjugation in basis)
    These are illustrative; in a real implementation J and b^circ must be defined
    and the projectors/charges must reflect the true embedding of A_F.
    """
    results: Dict[str, float] = {}

    # Hermiticity residuals
    results['herm_D0'] = float(la.norm(D0 - D0.conj().T, ord=2))
    results['herm_D1'] = float(la.norm(D1 - D1.conj().T, ord=2))
    results['herm_D2'] = float(la.norm(D2 - D2.conj().T, ord=2))

    # Mock order-one residuals: pick a small generating set of algebra elements (the projectors)
    # and compute norm of double commutator with D = D0 + 0.5*D1 + 0.25*D2 at rho=0.5
    Dtot = D0 + 0.5 * D1 + 0.25 * D2
    max_res = 0.0
    reps = list(P_reps.values())
    for a in reps:
        for b in reps:
            # define b^circ as b.conj() (toy: complex conjugation)
            b_circ = b.conj()
            comm = Dtot @ a - a @ Dtot
            double = comm @ b_circ - b_circ @ comm
            norm = la.norm(double, ord=2)
            if norm > max_res:
                max_res = norm
    results['order_one_max_residual'] = float(max_res)

    # --- Block projector trace (diagnostic for basis partitioning) ---
    Q_block = P_reps.get('U1', np.zeros_like(D0))
    trQ_block = np.trace(Q_block)
    results['trace_Q_block'] = float(np.real_if_close(trQ_block))
    results['trace_Q_block_imag'] = float(np.imag(trQ_block))
    if abs(results['trace_Q_block_imag']) > 1e-12:
        print(f"Warning: non-negligible imag(trace_Q_block) = {results['trace_Q_block_imag']:.3e}")

    # --- Construct a physical U(1) generator Q_phys from charges and projectors (toy example) ---
    # Replace projectors_map and charges with the real embedding data in production.
    projectors_map = {
        'rep0': P_reps.get('SU3', np.zeros_like(Q_block)),
        'rep1': P_reps.get('SU2', np.zeros_like(Q_block)),
        'rep2': P_reps.get('U1', np.zeros_like(Q_block))
    }
    # Example placeholder charges; replace with physical charges for each subrepresentation
    charges = {'rep0': 1/3.0, 'rep1': 1/2.0, 'rep2': -1.0}

    Q_phys = np.zeros_like(Q_block, dtype=np.complex128)
    for label, Pmat in projectors_map.items():
        qval = charges.get(label, 0.0)
        Q_phys += qval * Pmat

    # trace of physical generator
    trQ_phys = np.trace(Q_phys)
    results['trace_Q_phys'] = float(np.real_if_close(trQ_phys))
    results['trace_Q_phys_imag'] = float(np.imag(trQ_phys))

    # unimodularity policy: enforce zero trace if nonzero (toy correction)
    if abs(results['trace_Q_phys']) > 1e-12:
        dimH = Q_phys.shape[0]
        correction = results['trace_Q_phys'] / float(dimH)
        Q_phys_corrected = Q_phys - correction * np.eye(dimH, dtype=np.complex128)
        results['trace_Q_phys_after_correction'] = float(np.real_if_close(np.trace(Q_phys_corrected)))
        results['unimodularity_correction'] = float(correction)
        # replace Q_phys with corrected version for downstream use if needed
        Q_phys = Q_phys_corrected
    else:
        results['trace_Q_phys_after_correction'] = results['trace_Q_phys']
        results['unimodularity_correction'] = 0.0

    # Reality check: use J = complex conjugation in this basis (i.e., J X J^{-1} = X.conj())
    D_check = Dtot.conj()
    results['reality_residual'] = float(la.norm(D_check - Dtot, ord=2))

    return results


# ---------------------------
# Main driver: minimal worked example
# ---------------------------
def main_minimal_example():
    # 1. spectral input
    lam, w = make_spectral_input(N_in)
    print(f"Spectral input: N_in={len(lam)}, lambda range [{lam[0]:.4f}, {lam[-1]:.4f}]")

    # 2. build kernel basis
    basis_list, Phi = build_kernel_basis(lam, w, N_F, sigma_kernel)
    print(f"Built kernel basis: Phi shape = {Phi.shape}")

    # 3. orthonormalize
    Phi_orth = orthonormalize_basis(Phi, w)
    print("Orthonormalized basis (weighted inner product)")

    # 4. projectors and kernel matrix
    P_list, Kmat = compute_projectors_and_kernel(Phi_orth, lam, w, sigma_kernel)
    print("Computed projectors P_i and kernel matrix Kmat")

    # 5. representation projectors (toy)
    P_reps = construct_representation_projectors(N_F)
    print("Constructed toy representation projectors for gauge factors:", list(P_reps.keys()))

    # 6. monomials
    Ms = form_monomials(P_list, Kmat, P_reps)
    print("Formed monomial sets: sizes ->", {k: len(v) for k, v in Ms.items()})

    # 7. coefficients by quadrature (toy)
    coeffs = compute_coefficients_by_quadrature(Ms, Phi_orth, lam, w)
    print("Computed coefficients (toy quadrature) for monomials")

    # 8. assemble D_j
    D0, D1, D2 = assemble_Dj(Ms, coeffs)
    print("Assembled D0, D1, D2 and enforced hermiticity")

    # 9. save matrices
    save_npy(D0, OUT_D0)
    save_npy(D1, OUT_D1)
    save_npy(D2, OUT_D2)

    # 10. run checks
    checks = run_consistency_checks(D0, D1, D2, P_reps)
    print("Consistency checks (numeric):")
    for k, v in checks.items():
        print(f"  {k}: {v:.3e}")

    # 11. save run log
    runlog = {
        "rng_seed": RNG_SEED,
        "N_in": N_in,
        "N_F": N_F,
        "sigma_kernel": sigma_kernel,
        "files": {
            "D0": {"file": OUT_D0, "sha256": sha256_of_file(OUT_D0)},
            "D1": {"file": OUT_D1, "sha256": sha256_of_file(OUT_D1)},
            "D2": {"file": OUT_D2, "sha256": sha256_of_file(OUT_D2)}
        },
        "checks": checks
    }
    with open(OUT_LOG, "w") as f:
        json.dump(runlog, f, indent=2)
    print(f"Run log saved to {OUT_LOG}")

if __name__ == "__main__":
    main_minimal_example()
