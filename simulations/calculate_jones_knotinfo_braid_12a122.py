# calculate_jones_knotinfo_braid_12a122.py
#
# Computes the Jones polynomial for knot 12a_122 at t=exp(i*4*pi/5)
# using the qic_core.py framework and the knot's braid representation.

import numpy as np
from scipy.sparse import identity as sparse_identity
import time

# Attempt to import qic_core and its necessary components
try:
    import qic_core
    PHI = qic_core.PHI
    R_TAU_1 = qic_core.R_TAU_1
    R_TAU_TAU = qic_core.R_TAU_TAU
    TOL = qic_core.TOL # For numerical comparisons if needed later
    fibonacci = qic_core.fibonacci
    get_qic_basis = qic_core.get_qic_basis
    get_kauffman_Pn_anyon_general = qic_core.get_kauffman_Pn_anyon_general
    # Note: construct_isometry_V, build_P_prime_n, build_B_prime_n are not
    # directly needed if we work purely in the anyonic basis space for the trace.
except ImportError:
    print("ERROR: Could not import qic_core.py or its components.")
    print("Please ensure qic_core.py is in the same directory or in your Python path.")
    exit()
except AttributeError as e:
    print(f"ERROR: A required attribute is missing from qic_core.py: {e}")
    print("Please ensure qic_core.py is complete and up-to-date.")
    exit()

def calculate_jones_for_12a122():
    """
    Calculates the Jones polynomial for knot 12a_122 at t_0 = exp(i*4*pi/5)
    using the qic_core framework.
    """
    print("--- Starting Jones Polynomial Calculation for 12a_122 ---")
    start_time = time.time()

    # 1. Knot Parameters and Framework Setup
    knot_name = "12a_122"
    # Braid notation: {1,-2,-2,3,-4,1,1,-2,-2,-2,3,-4} (1-indexed from knotinfo)
    # sigma_i means generator acting on strands (i, i+1)
    braid_word_1_indexed = [1, -2, -2, 3, -4, 1, 1, -2, -2, -2, 3, -4]
    braid_index = 5  # Number of strands (m)

    N_sites = braid_index  # Number of qubit sites (N) for qic_core
    print(f"\nKnot: {knot_name}")
    print(f"Braid Index (m): {braid_index}")
    print(f"QIC Sites (N=m): {N_sites}")
    print(f"Braid Word (1-indexed): {braid_word_1_indexed}")

    dim_anyon_space = fibonacci(N_sites + 2)
    print(f"Expected Anyonic Basis Dimension (F_N+2 = F_{N_sites+2}): {dim_anyon_space}")
    if dim_anyon_space == 0:
        print("ERROR: Anyonic dimension is 0. Cannot proceed.")
        return

    # 2. Generate Anyonic Braid Operators (B_k^anyon)
    print("\nStep 1: Generating QIC basis strings...")
    qic_basis_strings, _ = get_qic_basis(N_sites)
    if not qic_basis_strings:
        print("ERROR: Failed to generate QIC basis strings.")
        return
    if len(qic_basis_strings) != dim_anyon_space:
        print(f"ERROR: Basis string length {len(qic_basis_strings)} mismatch with expected {dim_anyon_space}")
        return

    print("\nStep 2: Generating Anyonic Projectors (P_k^anyon)...")
    # P_k^anyon acts on sites (k, k+1). For m strands, k goes from 0 to m-2.
    # Number of distinct projectors needed is m-1 = N_sites - 1
    num_projectors = N_sites - 1
    P_k_anyon_list = []
    for k_idx in range(num_projectors):
        print(f"  Generating P_{k_idx}^anyon...")
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
        print(f"  Constructing B_{idx}^anyon and its inverse...")
        Bk_anyon = R_TAU_1 * Pk_anyon + R_TAU_TAU * (Id_anyon - Pk_anyon)
        B_k_anyon_list.append(Bk_anyon.tocsc())
        # B_k are unitary, so B_k^{-1} = B_k^dagger
        B_k_anyon_inv_list.append(Bk_anyon.conj().T.tocsc())

    # 3. Constructing the Braid Representation Matrix
    print("\nStep 4: Constructing the full braid matrix M_beta_anyon...")
    # Convert 1-indexed braid word to 0-indexed list of operators
    braid_matrix_sequence = []
    for val in braid_word_1_indexed:
        op_idx = abs(val) - 1 # Convert to 0-indexed
        if op_idx < 0 or op_idx >= num_projectors:
            print(f"ERROR: Invalid operator index {op_idx} from braid word value {val}.")
            return
        if val > 0:
            braid_matrix_sequence.append(B_k_anyon_list[op_idx])
        else:
            braid_matrix_sequence.append(B_k_anyon_inv_list[op_idx])

    # Multiply them together to get the full braid matrix M_beta
    M_beta_anyon = sparse_identity(dim_anyon_space, dtype=complex, format='csc')
    for i, op_matrix in enumerate(braid_matrix_sequence):
        # print(f"  Multiplying by operator {i+1}/{len(braid_matrix_sequence)} (shape {op_matrix.shape}, M_beta shape {M_beta_anyon.shape})")
        M_beta_anyon = M_beta_anyon @ op_matrix
    M_beta_anyon = M_beta_anyon.tocsc() # Ensure CSC for trace ops
    print(f"  M_beta_anyon constructed (shape {M_beta_anyon.shape}, nnz={M_beta_anyon.nnz}).")

    # 4. Computing the Jones Polynomial Value
    print("\nStep 5: Computing the Jones Polynomial value...")
    # Using .toarray() for trace with numpy. For very large sparse matrices,
    # one might look for sparse trace methods if memory is an issue,
    # but F_7=13 is small, so 13x13 dense is fine.
    trace_M_beta = np.trace(M_beta_anyon.toarray())
    
    if np.isclose(PHI, 0):
        print("ERROR: PHI is zero, cannot divide by PHI.")
        jones_polynomial_value_at_t0 = np.nan # Or handle as error
    else:
        jones_polynomial_value_at_t0 = trace_M_beta / PHI

    print(f"  Tr(M_beta_anyon) = {trace_M_beta:.8f}")
    print(f"  PHI (Golden Ratio) = {PHI:.8f}")
    print(f"  Calculated V(t_0) = Tr(M_beta_anyon) / PHI = {jones_polynomial_value_at_t0:.8f}")
    
    end_time = time.time()
    print(f"\n--- Calculation completed in {end_time - start_time:.2f} seconds ---")

    # 5. Verification (Theoretical Value)
    print("\n--- For Verification ---")
    # Jones polynomial for 12a_122:
    # V(t) = -t^-8 + 3t^-7 - 8t^-6 + 15t^-5 - 21t^-4 + 26t^-3 - 27t^-2 + 25t^-1 - 20 + 14t - 7t^2 + 3t^3 - t^4
    # We evaluate this at t_0 = exp(i*4*pi/5)
    t0 = np.exp(1j * 4 * np.pi / 5.0)
    
    # Coefficients of the polynomial V(t) for powers -8 to 4
    # P(t) = c_[-8]*t^-8 + ... + c_0 + ... + c_4*t^4
    coeffs = {
        -8: -1, -7:  3, -6: -8, -5: 15, -4: -21, -3: 26,
        -2: -27, -1: 25,  0: -20, 1: 14,  2: -7,  3:  3, 4: -1
    }
    expected_value = 0j
    for power, coeff in coeffs.items():
        expected_value += coeff * (t0**power)
    
    print(f"Theoretical Jones polynomial for 12a_122 from database definition:")
    print(f"  V(t) = -t^-8 + 3t^-7 - 8t^-6 + 15t^-5 - 21t^-4 + 26t^-3 - 27t^-2 + 25t^-1 - 20 + 14t - 7t^2 + 3t^3 - t^4")
    print(f"Evaluating at t_0 = exp(i*4*pi/5) approx {t0:.6f}")
    print(f"  Expected V(t_0) approx: {expected_value:.8f}")

    # Compare
    if isinstance(jones_polynomial_value_at_t0, complex) and isinstance(expected_value, complex):
        abs_diff = np.abs(jones_polynomial_value_at_t0 - expected_value)
        print(f"  Absolute difference: {abs_diff:.8f}")
        # A simple tolerance check, can be adjusted based on qic_core's TOL
        # and expected numerical stability for N=5.
        # TOL in qic_core is 1e-9. Matrix products can accumulate errors.
        comparison_tol = max(TOL * 1000 * len(braid_word_1_indexed), 1e-5) # Heuristic tolerance
        if abs_diff < comparison_tol:
            print(f"  SUCCESS: Calculated value is close to the expected value (tolerance {comparison_tol:.1e}).")
        else:
            print(f"  WARNING: Calculated value differs significantly from expected value (tolerance {comparison_tol:.1e}).")
            print(f"           This could be due to numerical precision, conventions, or an issue in the calculation steps/qic_core.")
    else:
        print("  Could not perform numerical comparison (one value is not complex).")


if __name__ == "__main__":
    calculate_jones_for_12a122()