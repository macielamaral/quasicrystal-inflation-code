# calculate_jones_qic_core_unknot.py
#
# Calculates the Jones polynomial for the Unknot using the qic_core.py framework
# and the theoretically derived formula from Kauffman & Lomonaco's work.
# This serves as a base case validation for the formula.
# The calculation targets t = exp(-i*2*pi/5), consistent with A = exp(i*3*pi/5)
# from the K&L Corollary.

import numpy as np
from scipy.sparse import identity as sparse_identity
import time

try:
    import qic_core
    PHI = qic_core.PHI
    # R_TAU_1, R_TAU_TAU are not directly used for the empty braid, but good to have
    R_TAU_1 = qic_core.R_TAU_1 
    R_TAU_TAU = qic_core.R_TAU_TAU
    TOL = qic_core.TOL
    fibonacci = qic_core.fibonacci
    get_qic_basis = qic_core.get_qic_basis
    # get_kauffman_Pn_anyon_general is not called for N_strands=1
except ImportError:
    print("ERROR: Could not import qic_core.py.")
    exit()
except AttributeError as e:
    print(f"ERROR: Attribute missing from qic_core.py: {e}")
    exit()

def calculate_jones_unknot():
    knot_name = "Unknot"
    # For the unknot represented by a single strand, N_strands = 1
    # The braid word is empty.
    braid_word_1_indexed = [] 
    N_strands = 1
    N_sites = N_strands 
    # Writhe of an empty braid (or a single unknotted strand) is 0
    writhe_B = 0 

    print(f"--- Starting Jones Polynomial Calculation for the {knot_name} ---")
    print(f"--- Using qic_core.py with K&L derived formula ---")
    print(f"--- Targeting t=exp(-i*2*pi/5) via A_K&L=exp(i*3*pi/5) ---")
    start_time = time.time()

    print(f"\nKnot: {knot_name}")
    print(f"Braid Index (N_strands): {N_strands}")
    print(f"QIC Sites (N_sites = N_strands): {N_sites}")
    print(f"Braid Word: {braid_word_1_indexed if braid_word_1_indexed else '[Empty Braid]'}, Writhe w_B = {writhe_B}")

    # For N_sites=1, dim_qic_space is F_{1+2} = F_3
    dim_qic_space = fibonacci(N_sites + 2) 
    F_N_plus_2_for_norm = float(dim_qic_space) # Denominator for normalization
    print(f"QIC Basis Dimension (F_{N_sites+2}): {dim_qic_space}")

    print("\nStep 1: Generating QIC basis strings (not strictly needed for empty braid)...")
    # get_qic_basis will run and determine the dimension, which is important.
    qic_basis_strings, _ = get_qic_basis(N_sites)
    if not qic_basis_strings or len(qic_basis_strings) != dim_qic_space:
        print(f"ERROR: QIC basis generation failed for N_sites={N_sites}.")
        return

    # Step 2 & 3: Generating Projectors and Braid Operators
    # For N_strands=1, num_projectors_needed = 1-1 = 0. So, no P_k or B_k are formed.
    print("\nStep 2 & 3: No projectors/braid operators needed for N_strands=1.")
    
    print("\nStep 4: Constructing the full braid matrix M_beta_anyon...")
    # For an empty braid word, M_beta_anyon is the identity matrix of the space.
    M_beta_anyon = sparse_identity(dim_qic_space, dtype=complex, format='csc')
    print(f"  M_beta_anyon for empty braid is Identity (shape {M_beta_anyon.shape}).")

    print("\nStep 5: Computing Trace and Applying Jones Formula...")
    trace_M_beta_qic = np.trace(M_beta_anyon.toarray()) # Should be dim_qic_space
    
    A_param_KL = np.exp(1j * 3 * np.pi / 5.0) 
    target_t_val = A_param_KL**(-4)

    A_cubed = A_param_KL**3
    minus_A_cubed = -A_cubed
    writhe_phase_factor = minus_A_cubed**(-writhe_B) # Will be (-A^3)^0 = 1
    
    phi_power_factor = PHI**(N_strands - 1) # Will be PHI^0 = 1

    if np.isclose(F_N_plus_2_for_norm, 0):
        print("ERROR: F_N_plus_2_for_norm is zero.")
        calculated_jones_value = np.nan
    else:
        calculated_jones_value = writhe_phase_factor * (phi_power_factor / F_N_plus_2_for_norm) * trace_M_beta_qic

    print(f"  Trace(M_beta_qic) [T_QIC for Unknot] = {trace_M_beta_qic:.8f}") # Expect F_3 = 2.0
    print(f"  A_K&L parameter = {A_param_KL:.6f}")
    print(f"  Target t value = A_K&L^-4 approx {target_t_val:.6f}")
    print(f"  Writhe factor (-A_K&L^3)^(-w_B) = {writhe_phase_factor:.6f}") # Expect 1.0
    print(f"  Normalization (phi^(N_strands-1)/F_N_strands+2) = (PHI^0 / F_3) = {(phi_power_factor / F_N_plus_2_for_norm):.6f}") # Expect 1/2 = 0.5
    print(f"  Calculated V_L(t=exp(-i*2*pi/5)) = {calculated_jones_value:.8f}") # Expect 1.0
    
    end_time = time.time()
    print(f"\n--- Calculation for {knot_name} completed in {end_time - start_time:.3f} seconds ---")

    # Verification
    print("\n--- For Verification ---")
    # Jones Polynomial for the Unknot is 1
    expected_jones_value_at_target_t = 1.0 + 0.0j 
    
    print(f"Theoretical V({knot_name}) evaluated at t_target = {target_t_val:.6f}:")
    print(f"  Expected V(t_target) = {expected_jones_value_at_target_t:.1f}")

    if isinstance(calculated_jones_value, complex) or isinstance(calculated_jones_value, float):
        abs_diff = np.abs(calculated_jones_value - expected_jones_value_at_target_t)
        print(f"  Absolute difference: {abs_diff:.8g}") # Use 'g' for general format
        # Tolerance should be very small for this exact case
        comparison_tol = 1e-9 
        if abs_diff < comparison_tol:
            print(f"  SUCCESS: Calculated value is essentially 1.0 as expected for the Unknot (tolerance {comparison_tol:.1e}).")
        else:
            print(f"  WARNING: Calculated value for {knot_name} ({calculated_jones_value:.8f}) is not 1.0 (tolerance {comparison_tol:.1e}).")
            print(f"           This would indicate an issue with the formula's base case application.")
    else:
        print("  Could not perform numerical comparison.")
    print("-" * 50)


if __name__ == "__main__":
    calculate_jones_unknot()