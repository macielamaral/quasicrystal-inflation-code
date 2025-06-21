# compute_jones_qic_core_B.py
#
# Computes the Jones polynomial for a given knot (3_1 or 12a_122)
# using the qic_core.py framework. This version uses the standard
# theoretical formula for the Jones polynomial normalization.

import numpy as np
from scipy.sparse import identity as sparse_identity
import time

try:
    import qic_core
    PHI = qic_core.PHI
    TOL = qic_core.TOL
    fibonacci = qic_core.fibonacci
    get_qic_basis = qic_core.get_qic_basis
    get_kauffman_Pn_anyon_general = qic_core.get_kauffman_Pn_anyon_general
    get_kauffman_Bn_anyon_general = qic_core.get_kauffman_Bn_anyon_general
except ImportError:
    print("ERROR: Could not import qic_core.py.")
    exit()
except AttributeError as e:
    print(f"ERROR: Attribute missing from qic_core.py: {e}")
    exit()

def get_knot_data(knot_name_str):
    """Returns a dictionary with knot data."""
    if knot_name_str == "3_1":
        return {
            "name": "3_1",
            "braid_word_1_indexed": [1, 1, 1], # sigma_1^3
            "N_strands": 2,
            "jones_coeffs": {0:0, 1:1, 2:0, 3:1, 4:-1}, # t + t^3 - t^4
            "writhe_B": 3
        }
    elif knot_name_str == "12a_122":
        return {
            "name": "12a_122",
            "braid_word_1_indexed": [1, -2, -2, 3, -4, 1, 1, -2, -2, -2, 3, -4],
            "N_strands": 5, # Braid index from knotinfo
            "jones_coeffs": {
                -8: -1, -7:  3, -6: -8, -5: 15, -4: -21, -3: 26,
                -2: -27, -1: 25,  0: -20,  1: 14,  2: -7,  3:  3, 4: -1
            },
            "writhe_B": sum(np.sign(val) for val in [1, -2, -2, 3, -4, 1, 1, -2, -2, -2, 3, -4]) # -2
        }
    else:
        raise ValueError(f"Unknown knot: {knot_name_str}")

def calculate_jones_final(knot_key):
    """Calculates and verifies the Jones polynomial for the given knot."""
    knot_data = get_knot_data(knot_key)
    knot_name = knot_data["name"]
    braid_word_1_indexed = knot_data["braid_word_1_indexed"]
    N_strands = knot_data["N_strands"]
    jones_coeffs = knot_data["jones_coeffs"]
    writhe_B = knot_data["writhe_B"]

    N_sites = N_strands

    print(f"\n--- Starting Jones Polynomial Calculation for {knot_name} ---")
    start_time = time.time()

    print(f"\nKnot: {knot_name}")
    print(f"Braid Index (N_strands): {N_strands}")
    print(f"QIC Sites (N_sites = N_strands): {N_sites}")
    print(f"Braid Word: {braid_word_1_indexed}, Writhe w_B = {writhe_B}")

    dim_qic_space = fibonacci(N_sites + 2)
    print(f"QIC Basis Dimension (F_{N_sites+2}): {dim_qic_space}")

    qic_basis_strings, _ = get_qic_basis(N_sites)
    if not qic_basis_strings or len(qic_basis_strings) != dim_qic_space:
        print(f"ERROR: QIC basis generation failed for N_sites={N_sites}.")
        return

    # Build P_k operators
    num_projectors_needed = N_strands - 1
    P_k_anyon_list = []
    print("\nBuilding P_k^anyon operators...")
    for k_idx in range(num_projectors_needed):
        pk_a = get_kauffman_Pn_anyon_general(N_sites, k_idx, qic_basis_strings, delta=PHI)
        if pk_a is None: print(f"ERROR: P_{k_idx}^anyon is None"); return
        P_k_anyon_list.append(pk_a)

    # Build B_k (sigma_k^-1) and B_k^-1 (sigma_k)
    B_k_anyon_list = []
    B_k_anyon_inv_list = []
    print("Building B_k^anyon and their inverses (sigma_k)...")
    for Pk_anyon in P_k_anyon_list:
        Bk_anyon = get_kauffman_Bn_anyon_general(Pk_anyon)
        if Bk_anyon is None:
            print(f"ERROR: Bk_anyon generation failed.")
            return
        B_k_anyon_list.append(Bk_anyon.tocsc())
        B_k_anyon_inv_list.append(Bk_anyon.conj().T.tocsc())

    # Build M_beta_anyon
    M_beta_anyon = sparse_identity(dim_qic_space, dtype=complex, format='csc')
    max_generator_idx = N_strands - 2
    print("Building M_beta_anyon using standard sigma_k convention...")
    for val in braid_word_1_indexed:
        op_idx_0_based = abs(val) - 1
        if op_idx_0_based < 0 or op_idx_0_based > max_generator_idx:
            print(f"ERROR: Braid generator index {abs(val)} out of range.")
            return
        current_op = B_k_anyon_inv_list[op_idx_0_based] if val > 0 else B_k_anyon_list[op_idx_0_based]
        M_beta_anyon = M_beta_anyon @ current_op
    M_beta_anyon = M_beta_anyon.tocsc()

    # Compute trace and apply the corrected, standard formula
    trace_M_beta_qic = np.trace(M_beta_anyon.toarray())

    A_param_KL = np.exp(1j * 3 * np.pi / 5.0)
    target_t_val = A_param_KL**(-4)
    A_cubed = A_param_KL**3
    minus_A_cubed = -A_cubed
    writhe_phase_factor = minus_A_cubed**(-writhe_B)
    
    # --- FINAL CORRECTION ---
    # Using the standard theoretical formula: V(A) = (-A^3)^(-w) * d^(n-2) * Tr(rho(beta))
    # where d = phi is the quantum dimension of the anyon.
    phi_power_factor = PHI**(N_strands - 2)
    calculated_jones_value = writhe_phase_factor * phi_power_factor * trace_M_beta_qic
    # --- END CORRECTION ---

    print(f"\n  Trace(M_beta_qic) [T_QIC] = {trace_M_beta_qic:.8f}")
    print(f"  A_K&L parameter = {A_param_KL:.6f}")
    print(f"  Target t value = A_K&L^-4 approx {target_t_val:.6f}")
    print(f"  Writhe factor (-A_K&L^3)^(-w_B) approx {writhe_phase_factor:.6f}")
    print(f"  Normalization (phi^(N-2)) approx {phi_power_factor:.6f}")
    print(f"  Calculated V_L(t=exp(-i*2*pi/5)) = {calculated_jones_value:.8f}")

    end_time = time.time()
    print(f"--- Calculation for {knot_name} completed in {end_time - start_time:.3f} seconds ---")

    # Verification against theoretical value
    print("\n--- Verification ---")
    def evaluate_jones_poly(t_val, coeffs_dict):
        val = 0j
        for power, coeff_val in coeffs_dict.items():
            val += coeff_val * (t_val**power)
        return val

    expected_jones_value_at_target_t = evaluate_jones_poly(target_t_val, jones_coeffs)

    print(f"Theoretical V({knot_name}) evaluated at t_target = {target_t_val:.6f}:")
    print(f"  Expected V(t_target) approx: {expected_jones_value_at_target_t:.8f}")

    if isinstance(calculated_jones_value, complex):
        abs_diff = np.abs(calculated_jones_value - expected_jones_value_at_target_t)
        print(f"  Absolute difference: {abs_diff:.8f}")
        comparison_tol = 1e-6
        if abs_diff < comparison_tol:
            print(f"  SUCCESS: Calculated value matches expected V({knot_name}, t) (tolerance {comparison_tol:.1e}).")
        else:
            print(f"  WARNING: Calculated value for {knot_name} still differs from expected (tolerance {comparison_tol:.1e}).")
    else:
        print("  Could not perform numerical comparison.")
    print("-" * 50)

# --- Main execution block ---
if __name__ == "__main__":
    calculate_jones_final("3_1")
    calculate_jones_final("12a_122")
