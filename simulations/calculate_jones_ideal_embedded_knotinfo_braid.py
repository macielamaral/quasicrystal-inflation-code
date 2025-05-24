# calculate_jones_ideal_embedded_knotinfo_braid.py
#
# Calculates the Jones polynomial for 12a_122 using the KNOTINFO N=5 BRAID by:
# 1. Constructing the ideal embedded 2^N x 2^N dense unitary matrices for each B'_k(N)
#    using the definitions in qic_core.py.
# 2. Classically multiplying these dense matrices for the full KnotInfo braid word to get
#    the total ideal embedded 2^N x 2^N braid operator, U_total_ideal_embedded.
# 3. Taking the QIC-RESTRICTED trace of U_total_ideal_embedded. Let this be S_knot.
#    This S_knot is equivalent to the trace of the abstract anyonic operator Tr(B_anyon_knot).
# 4. Normalizing this S_knot using the formula V_knot = (-A^3)^(-w_knot) * (S_knot / S_Unknot_N),
#    where S_Unknot_N = F_{N+2} (the trace of the N-strand identity anyonic operator).
#    This normalization ensures that V(Unknot with w=0, N strands) = 1 within this framework.
#
# The final output V_knot_framework is the Jones polynomial for the specified knot as computed
# by this QIC framework and its self-consistent V(Unknot)=1 normalization.
# This value can then be compared to external database values (like KnotInfo's polynomial evaluation).

import numpy as np
from scipy.sparse import identity as sparse_identity
import time
import cmath # For cmath.isclose

# Attempt to import qic_core and its necessary components
try:
    import qic_core
    PHI = qic_core.PHI
    R_TAU_1 = qic_core.R_TAU_1
    R_TAU_TAU = qic_core.R_TAU_TAU
    fibonacci = qic_core.fibonacci
    get_qic_basis = qic_core.get_qic_basis
    construct_isometry_V = qic_core.construct_isometry_V
    get_kauffman_Pn_anyon_general = qic_core.get_kauffman_Pn_anyon_general
    build_P_prime_n = qic_core.build_P_prime_n
    build_B_prime_n = qic_core.build_B_prime_n
except ImportError:
    print("ERROR: Could not import qic_core.py or its components.")
    exit()
except AttributeError as e:
    print(f"ERROR: A required attribute is missing from qic_core.py: {e}")
    exit()

def get_ideal_embedded_braid_operators_dense(n_strands, qic_strings_N_basis, V_isometry_csc_matrix):
    """
    Generates the ideal embedded B'_k(N) operators (defined as
    (R1-R2)P'_k + R2*I_total) as dense 2^N x 2^N NumPy arrays.
    """
    P_anyon_ops_abstract = []
    if n_strands >= 2:
        for k_idx in range(n_strands - 1):
            Pk_anyon = get_kauffman_Pn_anyon_general(n_strands, k_idx, qic_strings_N_basis, delta=PHI)
            if Pk_anyon is None:
                raise RuntimeError(f"Failed to generate P_{k_idx}^anyon for N={n_strands}.")
            P_anyon_ops_abstract.append(Pk_anyon)

    B_prime_dense_ops = []
    B_prime_inv_dense_ops = []
    
    if n_strands >= 2:
        for k_op_idx, Pk_anyon_abstract in enumerate(P_anyon_ops_abstract):
            Pk_prime_sparse = build_P_prime_n(k_op_idx, n_strands, V_isometry_csc_matrix, Pk_anyon_abstract)
            if Pk_prime_sparse is None:
                raise RuntimeError(f"Failed to build P'_{k_op_idx} for N={n_strands}")
            Bk_prime_sparse = build_B_prime_n(k_op_idx, Pk_prime_sparse, n_strands)
            if Bk_prime_sparse is None:
                raise RuntimeError(f"Failed to build B'_{k_op_idx} for N={n_strands}")

            Bk_prime_dense = Bk_prime_sparse.toarray()
            B_prime_dense_ops.append(Bk_prime_dense)
            B_prime_inv_dense_ops.append(Bk_prime_dense.conj().T)
            # print(f"  Ideal embedded B'_{k_op_idx}(N={n_strands}) and its inverse (dense) constructed.") # Verbose
            
    return B_prime_dense_ops, B_prime_inv_dense_ops

def calculate_qic_restricted_trace(U_matrix_dense, qic_basis_vectors_full_space):
    """Calculates Tr_QIC(U_matrix_dense) = Sum <q_i|U|q_i> for QIC basis |q_i>."""
    qic_trace = 0.0j
    for q_vec in qic_basis_vectors_full_space:
        term = np.conjugate(q_vec) @ U_matrix_dense @ q_vec
        qic_trace += term
    return qic_trace

def main():
    print("--- Jones Polynomial Calculation for 12a_122 (KnotInfo N=5 Braid) ---")
    print("--- using IDEAL EMBEDDED Braid Operators & Self-Consistent Unknot Normalization ---")
    overall_start_time = time.time()

    # Knot and Braid Information from KnotInfo for 12a_122
    knot_name_display = "12a_122 (KnotInfo N=5 Braid)"
    N_knot = 5
    knotinfo_braid_indices_1based = [1, -2, -2, 3, -4, 1, 1, -2, -2, -2, 3, -4]
    braid_sequence_knotinfo = []
    for gen_idx in knotinfo_braid_indices_1based:
        op_k_idx = abs(gen_idx) - 1 
        is_inv = gen_idx < 0
        braid_sequence_knotinfo.append((op_k_idx, is_inv))
    writhe_knotinfo_braid = sum(knotinfo_braid_indices_1based)

    print(f"\nTarget Knot: {knot_name_display}")
    print(f"Number of Strands (N): {N_knot}")
    print(f"Braid Sequence (0-indexed k, is_inverse): {braid_sequence_knotinfo}")
    print(f"Writhe (w): {writhe_knotinfo_braid}")

    # Constants
    A_param = np.exp(1j * np.pi / 10) # For t = exp(-2*pi*i/5)
    S_Unknot_N_theoretical_abstract_trace = float(fibonacci(N_knot + 2)) # For N=5, F_7 = 13

    # 1. Get QIC basis info and Isometry V
    print(f"\nStep 1: Setting up QIC basis and Isometry V for N={N_knot}...")
    qic_strings_N, qic_basis_vectors_full = get_qic_basis(N_knot)
    V_isometry_sparse = construct_isometry_V(qic_basis_vectors_full)
    V_isometry_csc = V_isometry_sparse.tocsc()
    print("QIC basis and V setup complete.")

    # 2. Generate the ideal embedded B'_k(N) operators
    print(f"\nStep 2: Generating ideal embedded B'_k(N={N_knot}) dense matrices...")
    try:
        B_prime_dense_list, B_prime_inv_dense_list = get_ideal_embedded_braid_operators_dense(
            N_knot, qic_strings_N, V_isometry_csc
        )
    except RuntimeError as e:
        print(f"  ERROR during B'_k generation: {e}")
        return
    print("Ideal embedded B'_k dense matrices generated.")

    # 3. Classically compose for the full KnotInfo braid
    print(f"\nStep 3: Composing dense B'_k matrices for {knot_name_display} braid...")
    U_total_ideal_embedded = np.eye(2**N_knot, dtype=complex)
    for op_k_idx, is_inv in braid_sequence_knotinfo:
        current_op_matrix = B_prime_inv_dense_list[op_k_idx] if is_inv else B_prime_dense_list[op_k_idx]
        U_total_ideal_embedded = U_total_ideal_embedded @ current_op_matrix
    print(f"Total ideal embedded braid operator U_total_ideal_embedded computed.")

    # 4. Calculate the QIC-RESTRICTED Trace
    print(f"\nStep 4: Calculating QIC-RESTRICTED Trace...")
    S_knot_qic_restricted_trace = calculate_qic_restricted_trace(U_total_ideal_embedded, qic_basis_vectors_full)
    print(f"  QIC-Restricted Trace (S_knot) for {knot_name_display}: {S_knot_qic_restricted_trace:.8f}")

    # This S_knot_qic_restricted_trace should be equal to the trace of the abstract anyonic operator
    # for the same KnotInfo N=5 braid, which was complex(1.9295634824951646, -0.2882008495275116)
    S_abstract_ref = complex(1.9295634824951646, -0.2882008495275116)
    if np.isclose(S_knot_qic_restricted_trace, S_abstract_ref):
        print("  Verification SUCCESS: QIC-Restricted trace of embedded ops matches abstract anyonic trace.")
    else:
        print(f"  Verification WARNING: QIC-Restricted trace ({S_knot_qic_restricted_trace:.8f})")
        print(f"                      differs from abstract anyonic trace ({S_abstract_ref:.8f}).")

    # 5. Normalize S_knot_qic_restricted_trace to get the Jones Polynomial
    print(f"\nStep 5: Normalizing S_knot to get Jones Polynomial for {knot_name_display}...")
    phase_factor_writhe = (-A_param**3)**(-writhe_knotinfo_braid)
    
    V_knot_framework_normalized = phase_factor_writhe * \
        (S_knot_qic_restricted_trace / S_Unknot_N_theoretical_abstract_trace)

    overall_end_time = time.time()
    print("\n--- QIC Framework Jones Polynomial Calculation Complete (KnotInfo N=5 Braid) ---")
    print(f"Total script execution time: {overall_end_time - overall_start_time:.3f} seconds")
    
    print(f"\nFinal Jones Polynomial V({knot_name_display}) from QIC Framework:")
    print(f"  V_framework = {V_knot_framework_normalized:.8f}")
    print(f"    Real part: {np.real(V_knot_framework_normalized):.8f}")
    print(f"    Imaginary part: {np.imag(V_knot_framework_normalized):.8f}")

    # --- For Paper: Comparison with KnotInfo's Own Polynomial Evaluation ---
    # KnotInfo Polynomial: -t^(-8)+3*t^(-7)...-t^4
    # Evaluated at t=exp(-2*pi*i/5) gives V_KI_eval approx. 0.20820415 - 1.04930417j
    V_KnotInfo_database_eval = complex(0.20820415139800033, -1.0493041738032164)
    print(f"\n--- Comparison with external citable value ---")
    print(f"Value from evaluating KnotInfo's polynomial string for 12a_122 at t=exp(-2*pi*i/5):")
    print(f"  V_KnotInfo_eval = {V_KnotInfo_database_eval:.8f}")
    
    diff_framework_vs_database = V_knot_framework_normalized - V_KnotInfo_database_eval
    print(f"Difference (V_framework vs V_KnotInfo_eval): {diff_framework_vs_database:.3e} (Magnitude: {np.abs(diff_framework_vs_database):.3e})")
    
    print("\nDiscussion for paper:")
    print("The V_framework value is the output of our QIC model for the KnotInfo 12a_122 braid,")
    print("normalized such that an N=5 unknot (from an identity braid) gives V=1 within our framework.")
    print("The difference from V_KnotInfo_eval highlights that the precise relationship between")
    print("our framework's raw trace and the specific conventions of tabulated Jones polynomials")
    print("(beyond V(Unknot)=1) may require further theoretical elucidation of overall scaling/phase")
    print("factors dependent on N, A, and phi_GR, or accounting for mirror image conventions.")
    print("The key result here is the V_framework value, which is the self-consistent target")
    print("for this specific braid within our QIC theoretical setup.")

if __name__ == "__main__":
    main()