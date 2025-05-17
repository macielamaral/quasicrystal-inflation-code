import numpy as np
from scipy.sparse import csc_matrix

# Import functions from the core library module
try:
    # Assuming qic_core.py is in src directory and src is in PYTHONPATH
    # or you've used the sys.path.insert(0, src_path) method
    import qic_core 
except ImportError:
    # Attempt relative import if src is a package and this script is run as a module
    # from ..src import qic_core # If run as a module e.g. python -m simulations.this_script
    # Fallback for direct execution if qic_core is in the same dir or already in path
    try:
        # If you set up sys.path in qic_core or it's discoverable
        if 'qic_core' not in globals(): # check if already imported by a previous attempt
             print("Attempting to import qic_core directly (ensure it's in PYTHONPATH or same dir)")
             import qic_core
    except ImportError as e:
        print("ERROR: Failed to import qic_core.py.")
        print("       Make sure qic_core.py is in 'src/' and 'src/' parent is in sys.path,")
        print("       or qic_core.py is in the same directory or Python path.")
        print(f"       Original error: {e}")
        exit()



N = 4  # Number of qubits/sites (small for clarity)
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1.0 / PHI
PHI_SQ_INV = PHI**(-2)
R_TAU_1 = np.exp(-1j * 4 * np.pi / 5.0)
R_TAU_TAU = np.exp(1j * 3 * np.pi / 5.0)

# --- Step 1: QIC basis ---
basis_strings, qic_basis_vectors = qic_core.get_qic_basis(N)

# --- Step 2: Isometry V ---
V = qic_core.construct_isometry_V(qic_basis_vectors)

# --- Step 3: TL generators in QIC subspace (P_k^anyon) ---
Pk_anyon_list = []
for k in range(N-1):
    Pk_anyon = qic_core.get_kauffman_Pn_anyon_general(N, k, basis_strings)
    Pk_anyon_list.append(Pk_anyon)

# --- Step 4: Embedded projectors (P'_k) ---
P_prime_list = []
for k in range(N-1):
    P_prime = qic_core.build_P_prime_n(k, N, V, Pk_anyon_list[k])
    P_prime_list.append(P_prime)

# --- Step 5: Embedded braids (B'_k) ---
B_prime_list = []
for k in range(N-1):
    B_prime = qic_core.build_B_prime_n(k, P_prime_list[k], N)
    B_prime_list.append(B_prime)

# --- Step 6: Print matrices for inspection ---
np.set_printoptions(precision=3, suppress=True)

print("=== Temperley-Lieb (P_k^anyon) matrices on QIC subspace ===")
#for k, Pk in enumerate(Pk_anyon_list):
    #if k == (N-2):
#    print(f"\nP_{k}^anyon:\n", Pk.toarray())

print("\n=== Embedded Projectors (P'_k) in full Hilbert space ===")
#for k, Pp in enumerate(P_prime_list):
    #if k == N-2:
#    print(f"\nP'_{k}:\n", Pp.toarray())
#
print("\n=== Embedded Braid Generators (B'_k) ===")
for k, Bp in enumerate(B_prime_list):
    print(f"\nB'_{k}:\n", Bp.toarray())
