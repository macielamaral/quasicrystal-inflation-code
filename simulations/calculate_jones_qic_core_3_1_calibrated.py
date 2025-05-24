# calculate_jones_qic_core_3_1_calibrated.py

import numpy as np
from scipy.sparse import identity as sparse_identity
import time

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
    print("ERROR: Could not import qic_core.py.")
    exit()
except AttributeError as e:
    print(f"ERROR: Attribute missing from qic_core.py: {e}")
    exit()

def calculate_jones_3_1():
    print("--- Starting Jones Polynomial Calculation for 3_1 (Trefoil Knot) ---")
    print("--- Using qic_core.py with Kauffman-style normalization (calibrated by Tr(Identity)) ---")
    print("--- Targeting t=exp(i*2*pi/5) by setting A=exp(-i*3*pi/5) ---")
    start_time = time.time()

    knot_name = "3_1"
    braid_word_1_indexed = [1, 1, 1] # sigma_1^3
    N_strands = 2
    N_sites = N_strands 
    writhe_B = sum(np.sign(val) for val in braid_word_1_indexed) # Should be 3

    print(f"\nKnot: {knot_name}")
    print(f"Braid Index (N_strands): {N_strands}")
    print(f"QIC Sites (N_sites = N_strands): {N_sites}")
    print(f"Braid Word (1-indexed): {braid_word_1_indexed}, Writhe w_B = {writhe_B}")

    dim_anyon_space = fibonacci(N_sites + 2) # F_{2+2} = F_4 = 3
    trace_identity_anyon_for_norm = float(dim_anyon_space) 
    print(f"QIC Anyonic Basis Dimension (F_{N_sites+2}): {dim_anyon_space}")
    print(f"Normalization denominator (Tr(Identity_anyon)): {trace_identity_anyon_for_norm}")

    print("\nStep 1: Generating QIC basis strings...")
    qic_basis_strings, _ = get_qic_basis(N_sites)
    if not qic_basis_strings or len(qic_basis_strings) != dim_anyon_space:
        print(f"ERROR: Failed to generate QIC basis strings correctly for N_sites={N_sites}.")
        return

    print("\nStep 2: Generating Anyonic Projector (P_0^anyon)...")
    # For N_strands=2, we only need P_0 (for sigma_1)
    num_projectors_needed = N_strands - 1 # = 1, so k_idx = 0 only
    if num_projectors_needed != 1:
         print(f"ERROR: Unexpected number of projectors for N_strands=2.")
         return
    
    P0_anyon = get_kauffman_Pn_anyon_general(N_sites, 0, qic_basis_strings, delta=PHI)
    if P0_anyon is None: print(f"ERROR: P_0^anyon is None"); return

    print("\nStep 3: Constructing Anyonic Braid Operator (B_0^anyon)...")
    Id_anyon = sparse_identity(dim_anyon_space, dtype=complex, format='csc')
    B0_anyon = (R_TAU_1 * P0_anyon + R_TAU_TAU * (Id_anyon - P0_anyon)).tocsc()
    B0_anyon_inv = B0_anyon.conj().T.tocsc() # Not needed for this braid, but good practice

    print("\nStep 4: Constructing the full braid matrix M_beta_anyon for sigma_1^3...")
    M_beta_anyon = B0_anyon @ B0_anyon @ B0_anyon
    M_beta_anyon = M_beta_anyon.tocsc()

    print("\nStep 5: Computing Trace and Applying Calibrated Jones Formula...")
    trace_M_beta_qic = np.trace(M_beta_anyon.toarray())
    
    A_param = np.exp(-1j * 3 * np.pi / 5.0) 
    target_t_val = A_param**(-4) 

    A_cubed = A_param**3
    minus_A_cubed = -A_cubed
    writhe_phase_factor = minus_A_cubed**(-writhe_B)

    if np.isclose(trace_identity_anyon_for_norm, 0):
        print("ERROR: Trace of identity for normalization is zero.")
        calculated_jones_value = np.nan
    else:
        calculated_jones_value = writhe_phase_factor * (trace_M_beta_qic / trace_identity_anyon_for_norm)

    print(f"  Trace(M_beta_qic=(B_0)^3) [T_QIC] = {trace_M_beta_qic:.8f}")
    print(f"  Normalization Denom. [F_4] = {trace_identity_anyon_for_norm:.1f}")
    print(f"  A parameter = exp(-i*3*pi/5) approx {A_param:.6f}")
    print(f"  Target t value = A^-4 approx {target_t_val:.6f}")
    print(f"  Writhe phase factor (-A^3)^(-w_B) approx {writhe_phase_factor:.6f}")
    print(f"  Calculated V_L(t=exp(i*2*pi/5)) = (PhaseFactor) * (T_QIC / F_4) = {calculated_jones_value:.8f}")
    
    end_time = time.time()
    print(f"\n--- Calculation completed in {end_time - start_time:.3f} seconds ---")

    # Verification
    print("\n--- For Verification ---")
    # Jones Polynomial for 3_1: t + t^3 - t^4
    def evaluate_jones_3_1(t_val):
        return t_val + t_val**3 - t_val**4

    expected_jones_value_at_target_t = evaluate_jones_3_1(target_t_val)
    
    print(f"Theoretical V(3_1) evaluated at t_target = exp(i*2*pi/5) approx {target_t_val:.6f}:")
    print(f"  Expected V(t_target) approx: {expected_jones_value_at_target_t:.8f}")
    # Expected: -0.80901700+1.31432932j

    if isinstance(calculated_jones_value, complex):
        abs_diff = np.abs(calculated_jones_value - expected_jones_value_at_target_t)
        print(f"  Absolute difference: {abs_diff:.8f}")
        comparison_tol = 1e-4 
        if abs_diff < comparison_tol:
            print(f"  SUCCESS: Calculated value is close to expected V(exp(i*2*pi/5)) for 3_1 (tolerance {comparison_tol:.1e}).")
        else:
            print(f"  WARNING: Calculated value for 3_1 differs from expected V(exp(i*2*pi/5)) (tolerance {comparison_tol:.1e}).")
    else:
        print("  Could not perform numerical comparison for 3_1.")

if __name__ == "__main__":
    calculate_jones_3_1()