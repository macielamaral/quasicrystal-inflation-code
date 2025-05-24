import os
import sys
import numpy as np

# --- Robust import of qic_core.py ---
try:
    import qic_core
    print("Successfully imported qic_core directly.")
except ImportError:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(os.path.dirname(current_dir), 'src')
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        import qic_core
        print("Successfully imported qic_core (path adjusted from simulations/ to src/).")
    except ImportError as e_path:
        print(f"ERROR: Failed to import qic_core.py even after path adjustment: {e_path}")
        exit(1)

N = 3  # Number of qubits/sites

# ---- Get QIC basis ----
basis_strings, _ = qic_core.get_qic_basis(N)
print(f"\nQIC basis strings (N={N}):")
for i, s in enumerate(basis_strings):
    print(f"  {i}: {s}")

# ---- Build P0 and P1 ----
P0 = qic_core.get_kauffman_Pn_anyon_general(N, 0, basis_strings)
P1 = qic_core.get_kauffman_Pn_anyon_general(N, 1, basis_strings)

# ---- Construct B0 and B1 using the standard formula ----
phi = (1 + np.sqrt(5)) / 2
R_I = np.exp(-4j * np.pi / 5)    # e^{-4πi/5}
R_tau = np.exp(3j * np.pi / 5)   # e^{3πi/5}
dim = len(basis_strings)
I = np.eye(dim)

# Dense matrices for display
P0_dense = P0.toarray()
P1_dense = P1.toarray()

B0_dense = R_I * P0_dense + R_tau * (I - P0_dense)
B1_dense = R_I * P1_dense + R_tau * (I - P1_dense)

def print_matrix_with_labels(M, basis_strings, name):
    print(f"\nNonzero entries of {name} (N=3):")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if abs(v) > 1e-10:
                print(f"  ({basis_strings[i]}, {basis_strings[j]}): {v:.4f}")

print_matrix_with_labels(B0_dense, basis_strings, "B0^anyon")
print_matrix_with_labels(B1_dense, basis_strings, "B1^anyon")

# Optional: Compare selected entries to Quantinuum paper for B1
print("\nComparison to Quantinuum (for B1, N=3):")
idx = {s: i for i, s in enumerate(basis_strings)}
phi_val = phi

entries = [
    ("010", "010"), # should be exp(-4j*pi/5)
    ("011", "011"), # should be exp(3j*pi/5)
    ("101", "101"), # should be (1/phi)*exp(3j*pi/5)
    ("101", "111"), # off-diagonal
    ("110", "110"), # exp(3j*pi/5)
    ("111", "111")  # should be ...
]
for r, c in entries:
    i, j = idx[r], idx[c]
    v = B1_dense[i, j]
    print(f"B1({r},{c}): {v:.4f}")

# Optionally, print the full matrix if needed:
# print("\nFull B1 matrix:\n", np.round(B1_dense, 4))


# === Embed ideal Braid (from QIC basis) into full 8x8 Hilbert space ===
from scipy.sparse import csc_matrix

# 1. Get QIC basis vectors for isometry construction
_, basis_vectors = qic_core.get_qic_basis(N) # Already loaded basis_strings above

# 2. Build isometry V: 8x5 for N=3
V = qic_core.construct_isometry_V(basis_vectors)  # V: (2^N, F_{N+2})

# 3. Braid matrix in Fibonacci space (already B1_dense above)
B1_sparse = csc_matrix(B1_dense)

# 4. Embed: B1_full = V @ B1 @ V^\dagger (shape 8x8)
V_csc = V.tocsc() if not isinstance(V, csc_matrix) else V
B1_full = V_csc @ B1_sparse @ V_csc.getH()  # getH() = conjugate transpose

B1_full_dense = B1_full.toarray()

# 5. Load G_ideal target matrix from .npy file (make sure path is correct)
TARGET_G_IDEAL_NPY_FILE = "data/optimal_local_approximators/G_tilde_N10_kop8_act7.npy"
G_ideal = np.load(TARGET_G_IDEAL_NPY_FILE)
assert G_ideal.shape == B1_full_dense.shape, f"Shape mismatch: {G_ideal.shape} vs {B1_full_dense.shape}"

# 6. Compute Frobenius norm and max difference
diff = B1_full_dense - G_ideal
fro_norm = np.linalg.norm(diff, ord='fro')
max_abs = np.max(np.abs(diff))
print(f"\n=== COMPARISON: Embedded Ideal Braid (QIC->8x8) vs Target G_ideal ===")
print(f"Frobenius norm of difference: {fro_norm:.6e}")
print(f"Max absolute difference:      {max_abs:.6e}")

# Optional: Print real/imag of both for quick visual comparison
np.set_printoptions(precision=4, suppress=True)
print("\nEmbedded Ideal Braid (real):\n", np.round(B1_full_dense.real, 4))
print("\nG_ideal (real):\n", np.round(G_ideal.real, 4))
print("\nDifference (real):\n", np.round(diff.real, 4))
print("\nEmbedded Ideal Braid (imag):\n", np.round(B1_full_dense.imag, 4))
print("\nG_ideal (imag):\n", np.round(G_ideal.imag, 4))
print("\nDifference (imag):\n", np.round(diff.imag, 4))
