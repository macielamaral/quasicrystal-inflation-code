# compute_jones_qic_core_Bprime.py
#
# Computes the Jones polynomial using the EMBEDDED qic_core.py operators (B'_k)
# that act on the full 2^N qubit Hilbert space. This serves as a validation
# of the embedding procedure. The final result should be identical to the
# calculation using the smaller anyonic operators.

import numpy as np
from scipy.sparse import identity as sparse_identity
import time

try:
    import qic_core
    PHI = qic_core.PHI
    TOL = qic_core.TOL
    fibonacci = qic_core.fibonacci
    get_qic_basis = qic_core.get_qic_basis
    construct_isometry_V = qic_core.construct_isometry_V
    get_kauffman_Pn_anyon_general = qic_core.get_kauffman_Pn_anyon_general
    build_P_prime_n = qic_core.build_P_prime_n
    build_B_prime_n = qic_core.build_B_prime_n
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

def calculate_jones_with_prime_ops(knot_key):
    """Calculates Jones polynomial using B'_k operators."""
    knot_data = get_knot_data(knot_key)
    knot_name = knot_data["name"]
    braid_word_1_indexed = knot_data["braid_word_1_indexed"]
    N_strands = knot_data["N_strands"]
    jones_coeffs = knot_data["jones_coeffs"]
    writhe_B = knot_data["writhe_B"]

    N_sites = N_strands

    print(f"\n--- Starting Jones Polynomial Calculation for {knot_name} (using B'_k Embedded Ops) ---")
    start_time = time.time()

    print(f"\nKnot: {knot_name}")
    print(f"Braid Index (N_strands): {N_strands}")
    print(f"QIC Sites (N_sites = N_strands): {N_sites}")
    
    # --- Step 1: Generate QIC basis, vectors, and Isometry V ---
    qic_basis_strings, qic_basis_vectors = get_qic_basis(N_sites)
    if not qic_basis_strings:
        print(f"ERROR: QIC basis generation failed for N_sites={N_sites}.")
        return
    
    print("\nConstructing Isometry V...")
    V_isometry = construct_isometry_V(qic_basis_vectors)
    if V_isometry is None:
        print(f"ERROR: Isometry V construction failed.")
        return

    # --- Step 2: Build Embedded B'_k operators ---
    # We must first build P_k^anyon, then embed to P'_k, then build B'_k
    num_ops_needed = N_strands - 1
    B_k_prime_list = []
    B_k_prime_inv_list = []
    print("\nBuilding Embedded B'_k and their inverses (sigma'_k)...")
    for k_idx in range(num_ops_needed):
        # a) Build P_k^anyon
        Pk_anyon = qic_core.get_kauffman_Pn_anyon_general(N_sites, k_idx, qic_basis_strings, delta=PHI)
        if Pk_anyon is None:
            print(f"ERROR: P_{k_idx}^anyon could not be built.")
            return

        # b) Embed to P'_k
        Pk_prime = build_P_prime_n(k_idx, N_sites, V_isometry, Pk_anyon)
        if Pk_prime is None:
            print(f"ERROR: P'_{k_idx} could not be built.")
            return

        # c) Build B'_k from P'_k
        Bk_prime = build_B_prime_n(k_idx, Pk_prime, N_sites)
        if Bk_prime is None:
            print(f"ERROR: B'_{k_idx} could not be built.")
            return
            
        B_k_prime_list.append(Bk_prime.tocsc())
        B_k_prime_inv_list.append(Bk_prime.conj().T.tocsc())

    # --- Step 3: Build the braid representation M_beta in the full space ---
    dim_full_space = 2**N_sites
    M_beta_prime = sparse_identity(dim_full_space, dtype=complex, format='csc')
    print("\nBuilding M_beta_prime using standard sigma'_k convention...")
    for val in braid_word_1_indexed:
        op_idx_0_based = abs(val) - 1
        if op_idx_0_based >= len(B_k_prime_list):
            print(f"ERROR: Braid generator index {abs(val)} out of range.")
            return
        #in our notation the braid is the inverse from the literature
        current_op = B_k_prime_inv_list[op_idx_0_based] if val > 0 else B_k_prime_list[op_idx_0_based] 
        M_beta_prime = M_beta_prime @ current_op
    M_beta_prime = M_beta_prime.tocsc()

    # --- Step 4: Calculate the trace and the Jones polynomial ---
    # The trace must be taken over the QIC subspace. We project M_beta_prime
    # onto the subspace using Pi = V*V^dagger before tracing.
    print("\nProjecting M_beta_prime onto QIC subspace for trace calculation...")
    Pi_QIC = V_isometry @ V_isometry.conj().T
    # The trace of the representation is Tr(Pi * M_beta)
    trace_M_beta_qic = np.trace((Pi_QIC @ M_beta_prime).toarray())
    
    # The rest of the formula is identical
    A_param_KL = np.exp(1j * 3 * np.pi / 5.0)
    target_t_val = A_param_KL**(-4)
    A_cubed = A_param_KL**3
    minus_A_cubed = -A_cubed
    writhe_phase_factor = minus_A_cubed**(-writhe_B)
    
    # Using the original formula that gave the smallest error
    F_N_plus_2 = float(fibonacci(N_sites + 2))
    phi_power_factor = PHI**(N_strands - 1)
    if np.isclose(F_N_plus_2, 0):
        calculated_jones_value = np.nan
    else:
        calculated_jones_value = writhe_phase_factor * (phi_power_factor / F_N_plus_2) * trace_M_beta_qic

    print(f"\n  Trace(Pi * M_beta_prime) [T_QIC] = {trace_M_beta_qic:.8f}")
    print(f"  Normalization (phi^(N-1)/F_N+2) approx {(phi_power_factor / F_N_plus_2):.6f}")
    print(f"  Calculated V_L(t=exp(-i*2*pi/5)) = {calculated_jones_value:.8f}")

    end_time = time.time()
    print(f"--- Calculation for {knot_name} completed in {end_time - start_time:.3f} seconds ---")

    # --- Step 5: Verification ---
    print("\n--- Verification ---")
    expected_jones_value_at_target_t = get_knot_data(knot_key)["jones_coeffs"]
    def evaluate_jones_poly(t_val, coeffs_dict):
        val = 0j
        for power, coeff_val in coeffs_dict.items():
            val += coeff_val * (t_val**power)
        return val
    
    expected_value = evaluate_jones_poly(target_t_val, jones_coeffs)
    print(f"Theoretical V({knot_name}) evaluated at t_target = {target_t_val:.6f}:")
    print(f"  Expected V(t_target) approx: {expected_value:.8f}")

    if isinstance(calculated_jones_value, complex):
        abs_diff = np.abs(calculated_jones_value - expected_value)
        print(f"  Absolute difference: {abs_diff:.8f}")
        if abs_diff < TOL * 1000: # Using a slightly larger tolerance for floating point comparisons
            print(f"  SUCCESS: Calculated value matches expected V({knot_name}, t).")
        else:
            print(f"  WARNING: Calculated value for {knot_name} still differs from expected.")
    print("-" * 50)

# --- Main execution block ---
if __name__ == "__main__":
    calculate_jones_with_prime_ops("3_1")
    calculate_jones_with_prime_ops("12a_122")
