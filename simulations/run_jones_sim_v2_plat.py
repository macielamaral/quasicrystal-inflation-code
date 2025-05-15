# run_jones_sim_v2_plat.py
# Simulates the Jones polynomial using QIC for PLAT CLOSURE.
# This version uses the formula derived from Quantinuum's postprocessing.py:
# V_P(B)(x) = (x)^(3*w_B) * φ^(N_qubits - 2) * <alpha|UB|alpha>
# where x = e^(i*2*pi/5)

import numpy as np
import time
import traceback
import qic_core 

# --- Configuration ---
# N_QUBITS is the number of qubits in your QIC simulation.
# For plat closure, the corresponding braid B acts on 2*N_QUBITS strands.
N_QUBITS = 4 # Example: For a B_8 braid (8 strands), N_QUBITS would be 4.

# Initial state |alpha> = |0101...0> for N_QUBITS
# Auto-generate based on N_QUBITS, or set manually if N_QUBITS is small.
if N_QUBITS % 2 == 0:
    INITIAL_STATE_STR = "01" * (N_QUBITS // 2)
else:
    # This state is typically for an even number of strands making N_QUBITS pairs.
    # If N_QUBITS is odd, the standard |0101...> state might not be standard.
    # AJL often implies N_QUBITS is the number of pairs of strands, so 2*N_QUBITS strands total.
    # For simplicity, let's assume N_QUBITS will be chosen such that this makes sense (e.g., N_QUBITS=2,3,4)
    INITIAL_STATE_STR = "01" * (N_QUBITS // 2) + "0" if N_QUBITS > 0 else "" 
    if N_QUBITS == 1: INITIAL_STATE_STR = "0" # Or "1", check convention for single qubit expectation. Usually N_qubits >= 2 for plat.


# TODO: CRITICAL USER ACTION REQUIRED!
# Replace BRAID_WORD and WRITHE_W with the correct values for the
# trefoil knot represented as a braid B on 2*N_QUBITS strands
# suitable for PLAT CLOSURE.
# The example B_1^3 below is a placeholder and INCORRECT for this context.
# You need to find the correct sequence of sigma_j from B_{2*N_QUBITS}
# and then map those sigma_j to your QIC B'_k operators.
BRAID_WORD_PLACEHOLDER = [
    {'index': 1, 'inverse': False}, 
    {'index': 1, 'inverse': False}, 
    {'index': 1, 'inverse': False}, 
]
WRITHE_W_PLACEHOLDER = 3 

# Constants for the Jones polynomial formula (k=5, Fibonacci)
# Evaluation point x = e^(i*2*pi/5) as per Quantinuum's paper/code
X_PARAM_JONES = np.exp(1j * 2 * np.pi / 5) 
PHI = qic_core.PHI 
TOL = qic_core.TOL

# ANALYTICAL TARGET for right-handed Trefoil J(x) = x + x^3 - x^4, at x = e^(i*2*pi/5)
J_TREFOIL_ANALYTICAL_AT_X = X_PARAM_JONES + X_PARAM_JONES**3 - X_PARAM_JONES**4
print(f"Target analytical J(x) for (Right-Handed) Trefoil at x=e^(i2pi/5): {J_TREFOIL_ANALYTICAL_AT_X:.6f}")
# Expected approx -0.809017 + 1.314328j

def apply_braid_sequence_to_state(psi_initial_np, braid_word_list, B_prime_ops_list, B_prime_dagger_ops_list, num_qubits_for_ops):
    """Applies a sequence of QIC braid operators B'_k to an initial state vector."""
    psi_current = psi_initial_np.copy()
    print(f"Applying QIC braid sequence to state:")
    for step, braid_op_info in enumerate(braid_word_list):
        k_qic = braid_op_info['index'] # This 'index' refers to the k in B'_k
        is_inverse = braid_op_info['inverse']
        
        # B_prime_ops are indexed 0 to num_qubits_for_ops-2
        if not (0 <= k_qic < len(B_prime_ops_list)): 
             raise ValueError(f"Invalid QIC operator index B'_{k_qic} for {num_qubits_for_ops} qubits. Max index is {len(B_prime_ops_list)-1}.")

        op_label = f"B'_{k_qic}" if not is_inverse else f"B'dag_{k_qic}"
        
        op_matrix_for_step = B_prime_dagger_ops_list[k_qic] if is_inverse else B_prime_ops_list[k_qic]
        if op_matrix_for_step is None:
            raise ValueError(f"{op_label} not available/constructed.")

        print(f"  Step {step+1}: Applying {op_label}...")
        psi_next = op_matrix_for_step @ psi_current 
        
        norm_check = np.linalg.norm(psi_next)
        if not np.isclose(norm_check, 1.0, atol=TOL*100):
             print(f"    WARNING: Norm changed to {norm_check:.6f} after applying {op_label}.")
        
        psi_current = psi_next
        
    print("QIC Braid sequence application to state complete.")
    return psi_current

def calculate_jones_plat_closure_quantinuum_code_version(overlap_alpha_UB_alpha, n_qubits, writhe_w, x_eval_point, phi_param):
    """
    Calculates V_P(B)(x) for plat closure using Quantinuum's postprocessing.py formula:
    V_P(B)(x) = (x_eval_point)^(3*w_B) * φ^(N_qubits - 2) * <alpha|UB|alpha>
    where x_eval_point = e^(i*2*pi/5).
    """
    print(f"\nCalculating Jones value (Plat Closure, Quantinuum code logic) with N_qubits={n_qubits}, writhe_w={writhe_w}:")
    print(f"  Raw overlap <alpha|UB|alpha> = {overlap_alpha_UB_alpha:.6f}")

    # Phase factor base P0 = -e^(-i*3*pi/5) = e^(i*2*pi/5), which is x_eval_point
    phase_factor_base = x_eval_point 
    phase_factor_exponent = 3 * writhe_w
    phase_factor = phase_factor_base**phase_factor_exponent 

    print(f"  Phase factor base (should be x_eval_point = e^(i2pi/5)): {phase_factor_base:.6f}")
    print(f"  Phase factor (base^({phase_factor_exponent})) = {phase_factor:.6f}")
    
    phi_exponent = n_qubits - 2
    phi_factor = phi_param**phi_exponent
    print(f"  Phi factor φ^(N_qubits-2) = {phi_param:.4f}^({phi_exponent}) = {phi_factor:.6f}")

    calculated_jones_value = phase_factor * phi_factor * overlap_alpha_UB_alpha
    print(f"  Calculated V_P(B)(x) = {phase_factor:.6f} * {phi_factor:.6f} * {overlap_alpha_UB_alpha:.6f} = {calculated_jones_value:.6f}")
    
    return calculated_jones_value

# ==============================================================
if __name__ == "__main__":

    master_start_time = time.time()
    print(f"--- Starting Jones Polynomial Simulation (Plat Closure v2) for N_QUBITS={N_QUBITS} ---")
    print(f"TARGET KNOT: Trefoil (using CONFIGURED BRAID_WORD and WRITHE_W)")
    print(f"WARNING: BRAID_WORD and WRITHE_W are placeholders and LIKELY INCORRECT for a trefoil plat closure on 2*{N_QUBITS} strands.")
    print(f"N_QUBITS (for QIC system): {N_QUBITS}")
    print(f"BRAID_WORD (for B_prime_ops): {BRAID_WORD_PLACEHOLDER}")
    print(f"WRITHE_W: {WRITHE_W_PLACEHOLDER}")
    print(f"INITIAL_STATE_STR |alpha>: |{INITIAL_STATE_STR}>")
    print(f"Parameters: Jones eval point x={X_PARAM_JONES:.4f}, φ={PHI:.4f}")

    # === Part 1: Setup QIC Basis and Isometry V ===
    print(f"\n=== PART 1: Setup QIC Basis and Isometry V (N_QUBITS={N_QUBITS}) ===")
    start_time = time.time()
    try:
        qic_strings, qic_vectors = qic_core.get_qic_basis(N_QUBITS)
        if not qic_strings or not qic_vectors: raise ValueError("Failed basis gen.")
        dim_QIC = len(qic_strings)
        print(f"QIC dim F_{N_QUBITS+2} = {dim_QIC}")
        V_iso = qic_core.construct_isometry_V(qic_vectors)
        if V_iso is None: raise ValueError("Failed V construction.")
        V_csc = V_iso.tocsc()
        V_dagger_sparse = V_csc.conj().T.tocsc() 
    except Exception as e: print(f"ERROR Part 1: {e}"); traceback.print_exc(); exit()
    end_time = time.time(); print(f"Part 1 Time: {end_time - start_time:.3f}s")

    # === Part 2: Build Embedded Operators ===
    print(f"\n=== PART 2: Build Embedded Operators B'_k for N_QUBITS={N_QUBITS} ===")
    part2_start_time = time.time()
    B_prime_ops = []
    B_prime_dagger_ops = [] 
    operators_ok = True
    num_TL_generators = N_QUBITS - 1 
    if num_TL_generators < 0 : 
        print("ERROR: N_QUBITS must be at least 1 (>=2 for any B' operators).")
        operators_ok = False 
        exit()
    if N_QUBITS == 1 and BRAID_WORD_PLACEHOLDER:
        print("Warning: N_QUBITS=1 means no B' operators, but BRAID_WORD is specified.")
        # Allow to proceed if BRAID_WORD_PLACEHOLDER is empty, otherwise it will fail in application
        if BRAID_WORD_PLACEHOLDER: operators_ok = False; exit()


    for k_idx in range(num_TL_generators): 
        print(f"\n--- Processing Embedded Operator Index k_idx = {k_idx} (for B'_{k_idx}) ---")
        try:
            # Using PHI directly from qic_core for delta in Kauffman rules
            Pk_anyon = qic_core.get_kauffman_Pn_anyon_general(N_QUBITS, k_idx, qic_strings, delta=qic_core.PHI)
            if Pk_anyon is None: raise ValueError(f"Failed P_{k_idx}^anyon.")
            Pk_prime = qic_core.build_P_prime_n(k_idx, N_QUBITS, V_csc, Pk_anyon)
            if Pk_prime is None: raise ValueError(f"Failed P'_{k_idx}.")
            Bk_prime = qic_core.build_B_prime_n(k_idx, Pk_prime, N_QUBITS) # Uses R_TAU_1, R_TAU_TAU from qic_core
            if Bk_prime is None: raise ValueError(f"Failed B'_{k_idx}.")
            B_prime_ops.append(Bk_prime)
            Bk_prime_dagger = Bk_prime.conj().T.tocsc() 
            B_prime_dagger_ops.append(Bk_prime_dagger)
            print(f"  B'_{k_idx} and B'dag_{k_idx} constructed and stored.")
        except Exception as e:
            print(f"ERROR processing k_idx={k_idx}: {e}"); traceback.print_exc(); operators_ok = False; break

    if not operators_ok or (num_TL_generators > 0 and len(B_prime_ops) != num_TL_generators) :
        print("\nERROR: Operator construction failed or incomplete."); exit()
    
    end_time = time.time(); print(f"\nPart 2 Time: {end_time - part2_start_time:.3f}s")

    # === Part 3: Simulation using Sparse Matrices (Plat Closure Version) ===
    print(f"\n=== PART 3: Jones Polynomial Simulation (Plat, N_QUBITS={N_QUBITS}) ===")
    part3_start_time = time.time()

    # --- 1. Prepare Initial State Vector |alpha> ---
    print(f"\n--- 1. Preparing Initial State |{INITIAL_STATE_STR}> ---")
    try:
        if INITIAL_STATE_STR not in qic_strings: 
            # Attempt to generate default |0101...> if N_QUBITS allows and INITIAL_STATE_STR is placeholder-like
            if N_QUBITS > 0 and N_QUBITS % 2 == 0 and INITIAL_STATE_STR == "01" * (N_QUBITS // 2):
                 print(f"Generated default initial state |{INITIAL_STATE_STR}> as it's in qic_strings.")
            elif N_QUBITS > 0 and N_QUBITS % 2 != 0 and INITIAL_STATE_STR == ("01" * (N_QUBITS // 2) + "0"):
                 print(f"Generated default initial state |{INITIAL_STATE_STR}> as it's in qic_strings.")
            else:
                 raise ValueError(f"Initial state '{INITIAL_STATE_STR}' invalid for N_QUBITS={N_QUBITS}. Valid QIC: {qic_strings}")

        initial_state_idx = qic_strings.index(INITIAL_STATE_STR)
        qic_vec_init = np.zeros(dim_QIC, dtype=complex); qic_vec_init[initial_state_idx] = 1.0
        psi_init_np = V_csc @ qic_vec_init # This is |alpha> in the 2^N_QUBITS space
        init_norm = np.linalg.norm(psi_init_np)
        if not np.isclose(init_norm, 1.0): print(f"WARNING: Initial state norm {init_norm:.6f} != 1.")
        print(f"Initial state |{INITIAL_STATE_STR}> prepared. Norm: {init_norm:.6f}")
    except Exception as e: print(f"ERROR Step 1: {e}"); traceback.print_exc(); exit()
    
    # --- 2. Apply Braid Sequence to get UB|alpha> ---
    print(f"\n--- 2. Applying Braid Sequence UB to state |alpha> ---")
    psi_final_np = np.copy(psi_init_np) # Default if BRAID_WORD is empty
    if BRAID_WORD_PLACEHOLDER: # Only apply if braid word is defined
        try:
            psi_final_np = apply_braid_sequence_to_state(psi_init_np, 
                                                        BRAID_WORD_PLACEHOLDER, 
                                                        B_prime_ops, 
                                                        B_prime_dagger_ops,
                                                        N_QUBITS)
            final_norm = np.linalg.norm(psi_final_np) 
            if not np.isclose(final_norm, 1.0):
                print("WARNING: Norm of UB|alpha> deviates significantly from 1.")
        except Exception as e:
            print(f"ERROR applying braid sequence: {e}"); traceback.print_exc(); exit()
    else:
        print("  No BRAID_WORD specified, UB is effectively Identity. Overlap will be 1.")


    # --- 3. Calculate Overlap and Resulting Jones Value ---
    print(f"\n--- 3. Calculating Final Overlap and Resulting Jones Value ---")
    try:
        overlap_value = np.vdot(psi_init_np, psi_final_np) 
        
        simulated_jones_value = calculate_jones_plat_closure_quantinuum_code_version(
            overlap_value, 
            N_QUBITS, 
            WRITHE_W_PLACEHOLDER, 
            X_PARAM_JONES,
            PHI
        )
        
        print(f"\n--- COMPARISON ---")
        print(f"Simulated V_P(B)(x) for N_QUBITS={N_QUBITS}, w(B)={WRITHE_W_PLACEHOLDER}: {simulated_jones_value:.6f}")
        print(f"Target Analytical J(x) for (Right-Handed) Trefoil at x=e^(i2pi/5): {J_TREFOIL_ANALYTICAL_AT_X:.6f}")
        
        error_diff = np.abs(simulated_jones_value - J_TREFOIL_ANALYTICAL_AT_X)
        real_error = np.abs(simulated_jones_value.real - J_TREFOIL_ANALYTICAL_AT_X.real)
        imag_error = np.abs(simulated_jones_value.imag - J_TREFOIL_ANALYTICAL_AT_X.imag)

        print(f"Absolute difference: {error_diff:.6f} (Real diff: {real_error:.6f}, Imag diff: {imag_error:.6f})")

    except Exception as e:
        print(f"ERROR calculating Jones value: {e}"); traceback.print_exc(); exit()

    # --- 4. Optional: Analyze Final State ---
    print("\n--- 4. Optional Final State Analysis ---")
    H_qic_op = qic_core.build_qic_hamiltonian_op(N_QUBITS, lam=1.0, verbose=False)
    if H_qic_op is not None and qic_core.QISKIT_AVAILABLE:
         is_gs_final, energy_final = qic_core.verify_energy(psi_final_np, H_qic_op, N_QUBITS, label="Final Braided State UB|alpha>", verbose=False)
         print(f"  Energy of final state UB|alpha>: {energy_final.real:.3e} (real), {energy_final.imag:.3e} (imag). In GS: {is_gs_final}")
    try:
         qic_final_vec = V_dagger_sparse @ psi_final_np
         print("\n  Final state UB|alpha> in QIC basis (coeffs > TOL):")
         output_str = []
         for i, coeff in enumerate(qic_final_vec):
              if abs(coeff) > TOL * 10: output_str.append(f"({coeff.real:.3f}{coeff.imag:+.3f}j)|{qic_strings[i]}>")
         if not output_str: print("    (Zero vector in QIC basis after TOL)")
         else: print("    " + " + ".join(output_str))
    except Exception as e: print(f"ERROR during back-projection: {e}"); traceback.print_exc()
            
    end_time = time.time(); print(f"\nPart 3 Time: {end_time - part3_start_time:.3f}s")
    
    master_end_time = time.time()
    print(f"\n--- N_QUBITS={N_QUBITS} Jones Polynomial Simulation (Plat Closure v2) Finished ---")
    print(f"Total execution time: {master_end_time - master_start_time:.3f} seconds")