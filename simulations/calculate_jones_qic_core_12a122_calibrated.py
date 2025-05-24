# calculate_jones_qic_core_12a122_calibrated.py
#
# Computes the Jones polynomial for knot 12a_122 (5-strands) at t=exp(i*2*pi/5)
# using the qic_core.py framework's native representation,
# and a Kauffman-style normalization calibrated by the identity braid's trace (F_N_strands+2).
# The choice of A=exp(-i*3*pi/5) is guided by literature (e.g., NP paper)
# linking Fibonacci R-matrix eigenvalues to t=exp(i*2*pi/5).

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

def calculate_jones_with_qic_core_calibrated():
    print("--- Starting Jones Polynomial Calculation for 12a_122 (5-strands) ---")
    print("--- Using qic_core.py with Kauffman-style normalization (calibrated by Tr(Identity)) ---")
    start_time = time.time()

    # Knot and Braid Information for 12a_122
    knot_name = "12a_122"
    # Braid notation from knotinfo: {1,-2,-2,3,-4,1,1,-2,-2,-2,3,-4}
    # This uses generators sigma_1 through sigma_4.
    braid_word_1_indexed = [1, -2, -2, 3, -4, 1, 1, -2, -2, -2, 3, -4]
    
    N_strands = 5 # From braid_index: 5 for 12a_122
    N_sites = N_strands # Number of sites for qic_core.py representation
    
    # Writhe of the braid B (sum of exponents in the braid word)
    writhe_B = sum(np.sign(val) for val in braid_word_1_indexed) # For this word: 1-1-1+1-1+1+1-1-1-1+1-1 = -2

    print(f"\nKnot: {knot_name}")
    print(f"Braid Index (N_strands): {N_strands}")
    print(f"QIC Sites (N_sites = N_strands): {N_sites}")
    print(f"Braid Word (1-indexed): {braid_word_1_indexed}, Writhe w_B = {writhe_B}")

    dim_anyon_space = fibonacci(N_sites + 2) # F_{5+2} = F_7 = 13
    # Calibration: Trace of identity anyonic operator is the dimension of the space
    trace_identity_anyon_for_norm = float(dim_anyon_space) 
    print(f"QIC Anyonic Basis Dimension (F_{N_sites+2}): {dim_anyon_space}")
    print(f"Normalization denominator (Tr(Identity_anyon)): {trace_identity_anyon_for_norm}")

    print("\nStep 1: Generating QIC basis strings...")
    qic_basis_strings, _ = get_qic_basis(N_sites)
    if not qic_basis_strings or len(qic_basis_strings) != dim_anyon_space:
        print(f"ERROR: Failed to generate QIC basis strings correctly for N_sites={N_sites}.")
        return

    print("\nStep 2: Generating Anyonic Projectors (P_k^anyon)...")
    # For N_strands=5, we need P_0, P_1, P_2, P_3 (for sigma_1 to sigma_4)
    num_projectors_needed = N_strands - 1 
    P_k_anyon_list = []
    for k_idx in range(num_projectors_needed):
        # print(f"  Generating P_{k_idx}^anyon (for sigma_{k_idx+1})...")
        pk_a = get_kauffman_Pn_anyon_general(N_sites, k_idx, qic_basis_strings, delta=PHI)
        if pk_a is None: print(f"ERROR: P_{k_idx}^anyon is None"); return
        P_k_anyon_list.append(pk_a)

    print("\nStep 3: Constructing Anyonic Braid Operators (B_k^anyon)...")
    Id_anyon = sparse_identity(dim_anyon_space, dtype=complex, format='csc')
    B_k_anyon_list = []
    B_k_anyon_inv_list = []
    for Pk_anyon in P_k_anyon_list:
        Bk_anyon = R_TAU_1 * Pk_anyon + R_TAU_TAU * (Id_anyon - Pk_anyon)
        B_k_anyon_list.append(Bk_anyon.tocsc())
        B_k_anyon_inv_list.append(Bk_anyon.conj().T.tocsc())

    print("\nStep 4: Constructing the full braid matrix M_beta_anyon...")
    M_beta_anyon = sparse_identity(dim_anyon_space, dtype=complex, format='csc')
    max_generator_idx = N_strands - 2 # Max k for P_k (0 to N_strands-2 for sigma_1 to sigma_{N_strands-1})
    for val in braid_word_1_indexed:
        op_idx_0_based = abs(val) - 1 # sigma_1 -> P_0 (op_idx 0), etc.
        if op_idx_0_based < 0 or op_idx_0_based > max_generator_idx:
            print(f"ERROR: Braid generator index {abs(val)} is out of range for {N_strands} strands.")
            return
        current_op = B_k_anyon_list[op_idx_0_based] if val > 0 else B_k_anyon_inv_list[op_idx_0_based]
        M_beta_anyon = M_beta_anyon @ current_op
    M_beta_anyon = M_beta_anyon.tocsc()
    # print(f"  M_beta_anyon constructed (shape {M_beta_anyon.shape}, nnz={M_beta_anyon.nnz}).")

    print("\nStep 5: Computing Trace and Applying Calibrated Jones Formula...")
    trace_M_beta_qic = np.trace(M_beta_anyon.toarray())
    
    # Parameters for Jones formula:
    # Target t = exp(i*2*pi/5). This corresponds to A = exp(-i*3*pi/5).
    A_param = np.exp(-1j * 3 * np.pi / 5.0) 
    target_t_val = A_param**(-4) # This should be exp(i*2*pi/5)

    # Writhe factor: (-A^3)^(-w_B)
    A_cubed = A_param**3
    minus_A_cubed = -A_cubed
    writhe_phase_factor = minus_A_cubed**(-writhe_B)

    if np.isclose(trace_identity_anyon_for_norm, 0):
        print("ERROR: Trace of identity for normalization is zero.")
        calculated_jones_value = np.nan
    else:
        calculated_jones_value = writhe_phase_factor * (trace_M_beta_qic / trace_identity_anyon_for_norm)

    print(f"  Trace(M_beta_qic) [T_QIC] = {trace_M_beta_qic:.8f}")
    print(f"  Normalization Denom. [F_7] = {trace_identity_anyon_for_norm:.1f}")
    print(f"  A parameter = exp(-i*3*pi/5) approx {A_param:.6f}")
    print(f"  Target t value = A^-4 approx {target_t_val:.6f}")
    print(f"  Writhe phase factor (-A^3)^(-w_B) approx {writhe_phase_factor:.6f}")
    print(f"  Calculated V_L(t=exp(i*2*pi/5)) = (PhaseFactor) * (T_QIC / F_7) = {calculated_jones_value:.8f}")
    
    end_time = time.time()
    print(f"\n--- Calculation completed in {end_time - start_time:.3f} seconds ---")

    # Verification
    print("\n--- For Verification ---")
    coeffs_12a122 = {
        -8: -1, -7:  3, -6: -8, -5: 15, -4: -21, -3: 26,
        -2: -27, -1: 25,  0: -20, 1: 14,  2: -7,  3:  3, 4: -1
    }
    def evaluate_jones_poly(t_val, coeffs_dict):
        val = 0j
        for power, coeff_val in coeffs_dict.items():
            val += coeff_val * (t_val**power)
        return val

    expected_jones_value_at_target_t = evaluate_jones_poly(target_t_val, coeffs_12a122)
    
    print(f"Theoretical V(12a_122) evaluated at t_target = exp(i*2*pi/5) approx {target_t_val:.6f}:")
    print(f"  Expected V(t_target) approx: {expected_jones_value_at_target_t:.8f}")

    if isinstance(calculated_jones_value, complex):
        abs_diff = np.abs(calculated_jones_value - expected_jones_value_at_target_t)
        print(f"  Absolute difference: {abs_diff:.8f}")
        # Tolerance can be tricky due to multiple matrix ops and float precision
        comparison_tol = 1e-4 
        if abs_diff < comparison_tol:
            print(f"  SUCCESS: Calculated value is close to expected V(exp(i*2*pi/5)) (tolerance {comparison_tol:.1e}).")
        else:
            print(f"  WARNING: Calculated value still differs from expected V(exp(i*2*pi/5)) (tolerance {comparison_tol:.1e}).")
            print(f"           If this persists, the normalization or the assumed 'A' for this N_strands={N_strands}")
            print(f"           and qic_core.py's F_{N_strands+2} dimensional trace may need further theoretical clarification.")
    else:
        print("  Could not perform numerical comparison (calculated value is not complex).")

if __name__ == "__main__":
    calculate_jones_with_qic_core_calibrated()