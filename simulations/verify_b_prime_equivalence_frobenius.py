# verify_b_prime_equivalence_frobenius.py

import numpy as np
import time
import traceback

try:
    import qic_core
except ImportError as e:
    print("ERROR: Failed to import qic_core.")
    exit()

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import eigh
from scipy.sparse import csc_matrix

N_VALUES_TO_VERIFY = [3, 4, 5, 6, 7, 8, 9, 10]
EQUIV_ATOL = 1e-7
EQUIV_RTOL = 1e-5

def compute_G_typeA_unitary():
    N_local = 3
    k_idx_local = 0
    try:
        qic_strings, qic_vectors = qic_core.get_qic_basis(N_local)
        V_iso = qic_core.construct_isometry_V(qic_vectors)
        V_csc = V_iso.tocsc()
        P_anyon = qic_core.get_kauffman_Pn_anyon_general(
            N_local, k_idx_local, qic_strings, delta=qic_core.PHI
        )
        P_prime = qic_core.build_P_prime_n(k_idx_local, N_local, V_csc, P_anyon)
        G_sparse = qic_core.build_B_prime_n(k_idx_local, P_prime, N_local)
        G_dense = G_sparse.toarray()
        identity = np.eye(2**N_local, dtype=complex)
        if not np.allclose(G_dense @ G_dense.conj().T, identity, atol=1e-8):
            return None
        return G_dense
    except Exception:
        return None

def compute_G_typeB_unitary(P1_dense):
    N_local = 3
    k_idx = 1
    try:
        qic_strings, qic_vectors = qic_core.get_qic_basis(N_local)
        V_iso = qic_core.construct_isometry_V(qic_vectors)
        V_csc = V_iso.tocsc()
        P1_sparse = csc_matrix(P1_dense)
        P_prime = qic_core.build_P_prime_n(k_idx, N_local, V_csc, P1_sparse)
        eigvals, eigvecs = eigh(P_prime.toarray())
        proj = np.zeros_like(P_prime.toarray(), dtype=complex)
        for i, v in enumerate(eigvals):
            if v > 0.5:
                vec = eigvecs[:, i].reshape(-1, 1)
                proj += vec @ vec.conj().T
        G_sparse = qic_core.build_B_prime_n(k_idx, csc_matrix(proj), N_local)
        G_dense = G_sparse.toarray()
        identity = np.eye(2**N_local, dtype=complex)
        if not np.allclose(G_dense @ G_dense.conj().T, identity, atol=1e-8):
            return None
        return G_dense
    except Exception:
        return None

def frobenius_norm_for_N_k(N, k_idx, G_A, G_B):
    try:
        # Original
        qic_strings, qic_vectors = qic_core.get_qic_basis(N)
        V_iso = qic_core.construct_isometry_V(qic_vectors)
        V_csc = V_iso.tocsc()
        Pk_anyon = qic_core.get_kauffman_Pn_anyon_general(N, k_idx, qic_strings, delta=qic_core.PHI)
        Pk_prime = qic_core.build_P_prime_n(k_idx, N, V_csc, Pk_anyon)
        Bk_orig = qic_core.build_B_prime_n(k_idx, Pk_prime, N).toarray()
        # Efficient
        if k_idx == 0:
            G = G_A
            tq = [0, 1, 2]
        elif k_idx == N-2:
            G = G_B
            tq = [N-3, N-2, N-1]
        else:
            G = G_A
            tq = [k_idx, k_idx+1, k_idx+2]
        qc = QuantumCircuit(N)
        qc.unitary(G, tq)
        Bk_eff = Operator(qc).data
        # Frobenius norm
        return np.linalg.norm(Bk_orig - Bk_eff, 'fro')
    except Exception:
        return None

if __name__ == "__main__":
    start = time.time()
    PHI = qic_core.PHI
    val_diag_1 = 1.0
    val_diag_2 = 1.0 / (PHI**2)
    val_diag_3 = 1.0 / PHI
    val_offdiag = 1.0 / (PHI * np.sqrt(PHI))
    P1_anyon_N3_data = np.array([
        [val_diag_1, 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., val_diag_2, 0., val_offdiag],
        [0., 0., 0., 0., 0.],
        [0., 0., val_offdiag, 0., val_diag_3]
    ], dtype=complex)
    G_A = compute_G_typeA_unitary()
    G_B = compute_G_typeB_unitary(P1_anyon_N3_data)
    if G_A is None or G_B is None:
        print("Critical error: Failed to compute G_typeA or G_typeB. Exiting.")
        exit()
    norms_by_N = {}
    for N in N_VALUES_TO_VERIFY:
        if N < 3: continue
        norms_by_N[N] = []
        for k in range(N-1):
            norm = frobenius_norm_for_N_k(N, k, G_A, G_B)
            norms_by_N[N].append(norm)
    print("\nFrobenius norm of (U_orig - V_eff) for each N and k:")
    for N in norms_by_N:
        print(f"N={N}: {['%.3e' % x if x is not None else 'ERR' for x in norms_by_N[N]]}")
    print(f"\nTotal runtime: {time.time()-start:.2f} seconds")
