# calculate_jones_qic_core_12a122.py
#
# Computes the Jones polynomial for knot 12a_122 at t=exp(i*2*pi/5)
# using the qic_core.py framework's native representation and trace.

import numpy as np
from scipy.sparse import identity as sparse_identity
import time

# Attempt to import qic_core and its necessary components
try:
    import qic_core
    PHI = qic_core.PHI
    R_TAU_1 = qic_core.R_TAU_1
    R_TAU_TAU = qic_core.R_TAU_TAU
    TOL = qic_core.TOL
    fibonacci = qic_core.fibonacci
    get_qic_basis = qic_core.get_qic_basis
    get_kauffman_Pn_anyon_general = qic_core.get_kauffman_Pn_anyon_general
except ImportError:
    print("ERROR: Could not import qic_core.py or its components.")
    print("Please ensure qic_core.py is in the same directory or in your Python path.")
    exit()
except AttributeError as e:
    print(f"ERROR: A required attribute is missing from qic_core.py: {e}")
    print("Please ensure qic_core.py is complete and up-to-date.")
    exit()

def calculate_jones_with_qic_core():
    """
    Calculates the Jones polynomial for knot 12a_122 at t_0 = exp(i*2*pi/5)
    using the qic_core framework's native trace and normalization.
    """
    print("--- Starting Jones Polynomial Calculation for 12a_122 using qic_core.py formalism ---")
    start_time = time.time()

    # 1. Knot Parameters and Framework Setup
    knot_name = "12a_122"
    # Braid notation from knotinfo: {1,-2,-2,3,-4,1,1,-2,-2,-2,3,-4}
    # This uses generators sigma_1 through sigma_4.
    braid_word_1_indexed = [1, -2, -2, 3, -4, 1, 1, -2, -2, -2, 3, -4]
    
    # Braid index for 12a_122 is 5, meaning 5 strands.
    # The generators are sigma_1, sigma_2, sigma_3, sigma_4.
    N_strands = 5 
    
    # For qic_core.py, to represent N_strands-1 distinct generator locations (P_0 to P_{N_strands-2}),
    # we need N_sites-2 >= N_strands-2, so N_sites >= N_strands.
    # We choose N_sites = N_strands.
    N_sites = N_strands

    print(f"\nKnot: {knot_name}")
    print(f"Braid Index (N_strands): {N_strands}")
    print(f"QIC Sites (N_sites = N_strands): {N_sites}")
    print(f"Braid Word (1-indexed): {braid_word_1_indexed}")

    dim_anyon_space = fibonacci(N_sites + 2) # F_{5+2} = F_7 = 13
    print(f"QIC Anyonic Basis Dimension (F_{N_sites+2}): {dim_anyon_space}")
    if dim_anyon_space == 0:
        print("ERROR: Anyonic dimension is 0. Cannot proceed.")
        return

    print("\nStep 1: Generating QIC basis strings...")
    qic_basis_strings, _ = get_qic_basis(N_sites)
    if not qic_basis_strings or len(qic_basis_strings) != dim_anyon_space:
        print(f"ERROR: Failed to generate QIC basis strings correctly for N_sites={N_sites}.")
        return

    print("\nStep 2: Generating Anyonic Projectors (P_k^anyon)...")
    # We need projectors P_0, P_1, P_2, P_3 (for sigma_1 to sigma_4)
    num_projectors_needed = N_strands - 1 # Max index is 3 for P_k
    P_k_anyon_list = []
    for k_idx in range(num_projectors_needed): # k_idx from 0 to N_strands-2
        print(f"  Generating P_{k_idx}^anyon (for sigma_{k_idx+1})...")
        pk_a = get_kauffman_Pn_anyon_general(N_sites, k_idx, qic_basis_strings, delta=PHI)
        if pk_a is None:
            print(f"ERROR: Failed to generate P_{k_idx}^anyon.")
            return
        P_k_anyon_list.append(pk_a)

    print("\nStep 3: Constructing Anyonic Braid Operators (B_k^anyon)...")
    Id_anyon = sparse_identity(dim_anyon_space, dtype=complex, format='csc')
    B_k_anyon_list = []
    B_k_anyon_inv_list = []

    for idx, Pk_anyon in enumerate(P_k_anyon_list):
        # print(f"  Constructing B_{idx}^anyon and its inverse...")
        Bk_anyon = R_TAU_1 * Pk_anyon + R_TAU_TAU * (Id_anyon - Pk_anyon)
        B_k_anyon_list.append(Bk_anyon.tocsc())
        B_k_anyon_inv_list.append(Bk_anyon.conj().T.tocsc())

    print("\nStep 4: Constructing the full braid matrix M_beta_anyon...")
    braid_matrix_sequence = []
    max_op_idx = N_strands - 2 # P_k index for sigma_{k+1}
    for val in braid_word_1_indexed:
        op_idx_0_based = abs(val) - 1 # sigma_1 -> op_idx 0, etc.
        if op_idx_0_based < 0 or op_idx_0_based > max_op_idx :
            print(f"ERROR: Invalid operator index {op_idx_0_based} (from braid value {val}) for {N_strands} strands (max P_k index {max_op_idx}).")
            return
        if val > 0:
            braid_matrix_sequence.append(B_k_anyon_list[op_idx_0_based])
        else:
            braid_matrix_sequence.append(B_k_anyon_inv_list[op_idx_0_based])

    M_beta_anyon = sparse_identity(dim_anyon_space, dtype=complex, format='csc')
    for op_matrix in braid_matrix_sequence:
        M_beta_anyon = M_beta_anyon @ op_matrix
    M_beta_anyon = M_beta_anyon.tocsc()
    print(f"  M_beta_anyon constructed (shape {M_beta_anyon.shape}, nnz={M_beta_anyon.nnz}).")

    print("\nStep 5: Computing the Jones Polynomial value...")
    trace_M_beta = np.trace(M_beta_anyon.toarray())
    
    if np.isclose(PHI, 0):
        print("ERROR: PHI is zero, cannot divide by PHI.")
        calculated_jones_value = np.nan
    else:
        calculated_jones_value = trace_M_beta / PHI

    print(f"  Trace(M_beta_anyon) [T_QIC] = {trace_M_beta:.8f}")
    print(f"  PHI (Golden Ratio) = {PHI:.8f}")
    print(f"  Calculated Jones Value (T_QIC / PHI) = {calculated_jones_value:.8f}")
    
    end_time = time.time()
    print(f"\n--- Calculation completed in {end_time - start_time:.2f} seconds ---")

    # 6. Verification against known Jones polynomial value
    print("\n--- For Verification ---")
    print("Comparing against V_L(t = exp(i*2*pi/5)), as suggested by literature (e.g., NP paper)")
    print("for Fibonacci R-matrix traces (eigenvalues e^{-i4pi/5}, e^{i3pi/5}).")
    
    coeffs = { # For V_12a_122(t)
        -8: -1, -7:  3, -6: -8, -5: 15, -4: -21, -3: 26,
        -2: -27, -1: 25,  0: -20, 1: 14,  2: -7,  3:  3, 4: -1
    }
    def evaluate_jones_poly(t_val, coeffs_dict):
        val = 0j
        for power, coeff_val in coeffs_dict.items():
            val += coeff_val * (t_val**power)
        return val

    t_target = np.exp(1j * 2 * np.pi / 5.0)
    expected_jones_value_at_t_target = evaluate_jones_poly(t_target, coeffs)
    
    print(f"\nTheoretical Jones polynomial V(t) for 12a_122 evaluated at t_target = exp(i*2*pi/5) approx {t_target:.6f}:")
    print(f"  Expected V(t_target) approx: {expected_jones_value_at_t_target:.8f}")

    if isinstance(calculated_jones_value, complex):
        abs_diff = np.abs(calculated_jones_value - expected_jones_value_at_t_target)
        print(f"  Absolute difference: {abs_diff:.8f}")
        comparison_tol = max(TOL * 1000 * len(braid_word_1_indexed), 1e-4) # Adjusted tolerance
        if abs_diff < comparison_tol:
            print(f"  SUCCESS: Calculated value is close to expected V(exp(i*2*pi/5)) (tolerance {comparison_tol:.1e}).")
        else:
            print(f"  WARNING: Calculated value differs from expected V(exp(i*2*pi/5)) (tolerance {comparison_tol:.1e}).")
            print(f"           This could indicate that the qic_core.py trace (over F_{N_strands+2} space) and 1/PHI normalization,")
            print(f"           while using correct R-matrix eigenvalues, might not directly map to this specific Jones value without further")
            print(f"           study of the basis mapping or normalization factors for this particular QIC state space.")
            
            # Also print the value at t=exp(i*4*pi/5) for completeness of earlier discussions
            t_alternate = np.exp(1j * 4 * np.pi / 5.0)
            expected_jones_value_at_t_alternate = evaluate_jones_poly(t_alternate, coeffs)
            print(f"\nFor reference, V(t) evaluated at t_alternate = exp(i*4*pi/5) approx {t_alternate:.6f}:")
            print(f"  Expected V(t_alternate) approx: {expected_jones_value_at_t_alternate:.8f}")
            abs_diff_alt = np.abs(calculated_jones_value - expected_jones_value_at_t_alternate)
            print(f"  Absolute difference from calculated value to V(t_alternate): {abs_diff_alt:.8f}")
            if abs_diff_alt < comparison_tol:
                 print(f"  NOTE: Calculated value is closer to V(exp(i*4*pi/5)). This would align with d=PHI => t=exp(i*4*pi/5) logic.")


    else:
        print("  Could not perform numerical comparison (calculated value is not complex).")

if __name__ == "__main__":
    calculate_jones_with_qic_core()