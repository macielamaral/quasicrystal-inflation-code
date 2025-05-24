# calculate_jones_ideal_embedded.py
#
# Calculates the Jones polynomial for 12a_122 (N=4) by:
# 1. Constructing the ideal embedded 2^N x 2^N dense unitary matrices for each B'_k(N).
# 2. Classically multiplying these dense matrices for the full braid word.
# 3. Taking the standard matrix trace of the resulting 2^N x 2^N total braid operator.
# 4. Normalizing this trace using the theoretical trace of an N=4 Unknot's
#    abstract anyonic operator (which is F_{N+2}).
# The result should match that from calculate_renormalized_jones.py.

import numpy as np
from scipy.sparse import identity as sparse_identity # For Id_anyon in B_k construction
import time

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
    build_P_prime_n = qic_core.build_P_prime_n # Needed for P'_k
    build_B_prime_n = qic_core.build_B_prime_n # Needed for B'_k (sparse)
except ImportError:
    print("ERROR: Could not import qic_core.py or its components.")
    print("Please ensure qic_core.py is in the same directory or in your Python path.")
    exit()
except AttributeError as e:
    print(f"ERROR: A required attribute is missing from qic_core.py: {e}")
    print("Please ensure qic_core.py is complete and up-to-date.")
    exit()

def get_ideal_embedded_braid_operators_dense(n_strands, qic_strings_N, V_isometry_csc):
    """
    Generates the ideal embedded B'_k(N) operators as dense 2^N x 2^N NumPy arrays.
    """
    P_anyon_ops_abstract = [] # Abstract P_k^anyon (F_N+2 x F_N+2)
    if n_strands >= 2:
        for k_idx in range(n_strands - 1):
            Pk_anyon = get_kauffman_Pn_anyon_general(n_strands, k_idx, qic_strings_N, delta=PHI)
            if Pk_anyon is None:
                raise RuntimeError(f"Failed to generate P_{k_idx}^anyon for N={n_strands}.")
            P_anyon_ops_abstract.append(Pk_anyon)

    # These will be lists of dense 2^N x 2^N matrices
    B_prime_dense_ops = []
    B_prime_inv_dense_ops = []
    
    # For constructing B'_k, we need P'_k first.
    # build_B_prime_n in qic_core takes P'_k as input.
    if n_strands >= 2:
        for k_op_idx, Pk_anyon_abstract in enumerate(P_anyon_ops_abstract):
            # 1. Build P'_k = V P_k^anyon V_dag (sparse)
            Pk_prime_sparse = build_P_prime_n(k_op_idx, n_strands, V_isometry_csc, Pk_anyon_abstract)
            if Pk_prime_sparse is None:
                raise RuntimeError(f"Failed to build P'_{k_op_idx} for N={n_strands}")

            # 2. Build B'_k = R1*P'_k + Rtau*(I_total - P'_k) (sparse)
            # Note: qic_core.build_B_prime_n uses I_total (2^N x 2^N identity)
            Bk_prime_sparse = build_B_prime_n(k_op_idx, Pk_prime_sparse, n_strands)
            if Bk_prime_sparse is None:
                raise RuntimeError(f"Failed to build B'_{k_op_idx} for N={n_strands}")

            # 3. Convert to dense and store
            Bk_prime_dense = Bk_prime_sparse.toarray()
            B_prime_dense_ops.append(Bk_prime_dense)
            
            # Inverse is the adjoint
            Bk_prime_inv_dense = Bk_prime_dense.conj().T
            B_prime_inv_dense_ops.append(Bk_prime_inv_dense)
            print(f"  Ideal embedded B'_{k_op_idx}(N={n_strands}) and its inverse (dense) constructed.")
            
    return B_prime_dense_ops, B_prime_inv_dense_ops


def main():
    print("--- Jones Polynomial Calculation using IDEAL EMBEDDED Braid Operators ---")
    overall_start_time = time.time()

    # Knot and Braid Information for 12a_122
    knot_name = "12a_122"
    N_knot = 4
    braid_sequence_12a122 = [
        (0, False), (1, True),  (0, False), (2, True),  (1, False), (0, False),
        (2, True),  (1, False), (2, False), (0, False), (1, True),  (2, False)
    ] # (operator_k_index_for_B'_k, is_inverse)
    writhe_12a122 = 4

    print(f"Target Knot: {knot_name}")
    print(f"Number of Strands (N): {N_knot}")
    print(f"Writhe (w): {writhe_12a122}")

    # Constants for Jones Polynomial normalization
    A_param = np.exp(1j * np.pi / 10) # For t = exp(-2*pi*i/5)
    
    # Theoretical trace of Unknot on N=4 strands (abstract anyonic identity op)
    # This is F_{N+2}
    S_Unknot_N4_theoretical_abstract_trace = float(fibonacci(N_knot + 2)) # F_6 = 8

    # 1. Get QIC basis info and Isometry V (needed for P'_k, B'_k construction)
    print(f"\nStep 1: Setting up QIC basis and Isometry V for N={N_knot}...")
    qic_strings_N, qic_basis_vectors_full = get_qic_basis(N_knot)
    if not qic_strings_N:
        print("ERROR: Failed to generate QIC basis strings.")
        return
    
    V_isometry_sparse = construct_isometry_V(qic_basis_vectors_full)
    if V_isometry_sparse is None:
        print("ERROR: Failed to construct isometry V.")
        return
    V_isometry_csc = V_isometry_sparse.tocsc()
    print("QIC basis and V setup complete.")

    # 2. Generate the ideal embedded B'_k(N) operators as dense matrices
    print(f"\nStep 2: Generating ideal embedded B'_k(N={N_knot}) dense matrices...")
    try:
        B_prime_dense_list, B_prime_inv_dense_list = get_ideal_embedded_braid_operators_dense(
            N_knot, qic_strings_N, V_isometry_csc
        )
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return
    print("Ideal embedded B'_k dense matrices generated.")

    # 3. Classically compose these dense matrices for the full 12a_122 braid
    print(f"\nStep 3: Composing dense B'_k matrices for {knot_name} braid...")
    # Total operator will be 2^N_knot x 2^N_knot
    U_total_ideal_embedded = np.eye(2**N_knot, dtype=complex)

    for op_k_idx, is_inv in braid_sequence_12a122:
        if op_k_idx >= len(B_prime_dense_list):
            print(f"  ERROR: Braid operator index {op_k_idx} out of range.")
            return
        
        current_op_matrix = B_prime_inv_dense_list[op_k_idx] if is_inv else B_prime_dense_list[op_k_idx]
        U_total_ideal_embedded = U_total_ideal_embedded @ current_op_matrix
        
    print(f"Total ideal embedded braid operator U_total_ideal_embedded computed (shape: {U_total_ideal_embedded.shape}).")

    # 4. Calculate the standard matrix trace of U_total_ideal_embedded
    # This S_double_prime should be equal to Tr(B_anyon_total) from the abstract calculation.
    #print(f"\nStep 4: Calculating standard matrix Trace(U_total_ideal_embedded)...")
    #S_double_prime_12a122 = np.trace(U_total_ideal_embedded)
    print(f"\nStep 4: Calculating QIC-RESTRICTED Trace of U_total_ideal_embedded...")
    S_double_prime_12a122 = 0.0j
    # qic_basis_vectors_full was generated in Step 1 of main() in your script.
    for q_vec_full in qic_basis_vectors_full: # These are the 2^N-dim QIC basis vectors
        # <q_i | U | q_i> = q_i_dag @ U @ q_i
        term = np.conjugate(q_vec_full) @ U_total_ideal_embedded @ q_vec_full
        S_double_prime_12a122 += term
    print(f"  Trace(U_total_ideal_embedded) (S''): {S_double_prime_12a122}")
    print(f"    Real part: {np.real(S_double_prime_12a122)}")
    print(f"    Imaginary part: {np.imag(S_double_prime_12a122)}")

    # For reference, the abstract anyonic trace from calculate_renormalized_jones.py was:
    # S_abstract_12a122 = -1.899186938124422 - 2.1063826071850382j
    S_abstract_12a122_ref = complex(-1.899186938124422, -2.1063826071850382)
    if np.isclose(S_double_prime_12a122, S_abstract_12a122_ref):
        print("  SUCCESS: Trace of embedded operator matches trace of abstract anyonic operator.")
    else:
        print(f"  WARNING: Trace of embedded operator ({S_double_prime_12a122}) significantly differs from")
        print(f"           abstract anyonic trace ({S_abstract_12a122_ref}). This indicates a potential issue.")
        print(f"           Difference: {S_double_prime_12a122 - S_abstract_12a122_ref}")


    # 5. Normalize this S_double_prime_12a122 to get the Jones Polynomial
    # Use the same normalization as calculate_renormalized_jones.py:
    # V_K_final(t) = (-A^3)^(-writhe_K) * (S_K / S_Unknot_N_abstract_trace)
    print(f"\nStep 5: Normalizing S''_{knot_name} to get Jones Polynomial...")
    
    phase_factor_writhe = (-A_param**3)**(-writhe_12a122)
    
    # Normalize using the THEORETICAL S_Unknot_N4_abstract_trace = F_{N+2}
    V_12a122_ideal_embedded_calc = phase_factor_writhe * (S_double_prime_12a122 / S_Unknot_N4_theoretical_abstract_trace)

    overall_end_time = time.time()
    print("\n--- Ideal Embedded Jones Polynomial Calculation Complete ---")
    print(f"Total script execution time: {overall_end_time - overall_start_time:.3f} seconds")
    
    print(f"\nRaw Trace from Ideal Embedded Operators (S''_{knot_name}): {S_double_prime_12a122}")
    print(f"Normalization Factor (S_Unknot_N4_theoretical_abstract_trace = F_6): {S_Unknot_N4_theoretical_abstract_trace}")
    print(f"Writhe-dependent Phase Factor (-A^3)^(-w): {phase_factor_writhe}")
    
    print(f"\nFinal Jones Polynomial V({knot_name}) from Ideal Embedded Calculation:")
    print(f"  V''({knot_name}) = {V_12a122_ideal_embedded_calc}")
    print(f"    Real part: {np.real(V_12a122_ideal_embedded_calc)}")
    print(f"    Imaginary part: {np.imag(V_12a122_ideal_embedded_calc)}")

    # Compare with the target V_final from calculate_renormalized_jones.py
    # (which used abstract anyonic matrices directly for the trace)
    TARGET_JONES_12A122_FINAL_ABSTRACT = complex(0.34682189257828955, 0.07347315653655925)
    print(f"\nComparison with V_final({knot_name}) from abstract anyonic calculation:")
    print(f"  Target V_final_abstract = {TARGET_JONES_12A122_FINAL_ABSTRACT}")
    
    diff_from_abstract = V_12a122_ideal_embedded_calc - TARGET_JONES_12A122_FINAL_ABSTRACT
    error_mag_from_abstract = np.abs(diff_from_abstract)
        
    print(f"  Difference: {diff_from_abstract} (Magnitude: {error_mag_from_abstract:.6e})")

    if error_mag_from_abstract < 1e-9: # Allow for small numerical precision differences
        print("  --> EXCELLENT MATCH! The ideal embedded calculation matches the abstract anyonic calculation.")
        print("      This confirms Tr(V B_anyon V_dag) = Tr(B_anyon) and the consistency of the framework.")
    else:
        print("  --> MISMATCH with abstract anyonic calculation. This is unexpected and indicates a potential issue")
        print("      in either the script logic, the understanding of Tr(V B V_dag), or qic_core functions.")

if __name__ == "__main__":
    main()