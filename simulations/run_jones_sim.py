# run_jones_sim.py
# Simulates the evaluation of Jones polynomials using QIC braiding.
# VERSION: Markov Closure of LEFT-HANDED Trefoil (sigma_1^-3)

import numpy as np
import time
import traceback
import qic_core 

# --- Configuration for LEFT-HANDED Trefoil (3_1*) ---
# Markov closure of B = sigma_1^-3 (a 2-strand braid operation)
# QIC representation using N_QUBITS=2, where B'_0 represents sigma_1.

N_QUBITS = 2
N_STRANDS_IN_BRAID = 2 # For the formula factor phi^(N_strands-1)

# BRAID_WORD for B = (sigma_1)^-3, mapping sigma_1 to QIC B'_0
BRAID_WORD = [
    {'index': 0, 'inverse': True}, # B'_0 dagger
    {'index': 0, 'inverse': True}, # B'_0 dagger
    {'index': 0, 'inverse': True}  # B'_0 dagger
]
WRITHE_W = -3 # Writhe of sigma_1^-3

# Constants for the Jones polynomial formula (k=5, Fibonacci)
# Formula 2 for Markov closure uses q = e^(i*pi/5)
Q_PARAM = np.exp(1j * np.pi / 5) 
PHI = qic_core.PHI 
TOL = qic_core.TOL

# ANALYTICAL TARGET: Left-handed Trefoil J_LH(q) = q^-1 + q^-3 - q^-4 at q = e^(i*pi/5)
q_inv = 1.0 / Q_PARAM
J_TREFOIL_ANALYTICAL = q_inv + q_inv**3 - q_inv**4
print(f"Target analytical J_LH(q) for left-handed trefoil at q=e^(i*pi/5): {J_TREFOIL_ANALYTICAL:.6f}")
# Expected: approx 1.309017 - 0.951057j

def apply_braid_sequence_to_matrix(initial_matrix, braid_word_list, B_prime_ops_list, B_prime_dagger_ops_list, num_qubits_for_ops):
    """Applies a sequence of braid operators to an initial matrix (e.g., identity)."""
    current_matrix = initial_matrix.copy()
    print(f"Constructing UB matrix from braid sequence:")
    for step, braid_op_info in enumerate(braid_word_list):
        k = braid_op_info['index']
        is_inverse = braid_op_info['inverse']
        
        # For N_QUBITS, B_prime_ops has length N_QUBITS-1. Indices are 0 to N_QUBITS-2.
        if not (0 <= k < len(B_prime_ops_list)): 
             raise ValueError(f"Invalid braid operator index {k} for {num_qubits_for_ops} qubits (expected 0 to {num_qubits_for_ops-2}). B_prime_ops has length {len(B_prime_ops_list)}")

        op_label = f"B'_{k}" if not is_inverse else f"B'dag_{k}"
        
        op_matrix_for_step = B_prime_dagger_ops_list[k] if is_inverse else B_prime_ops_list[k]
        if op_matrix_for_step is None: # Should not happen if B_prime_ops is populated
            raise ValueError(f"{op_label} is None. B_prime_ops length: {len(B_prime_ops_list)}, k: {k}")

        print(f"  Step {step+1}: Applying {op_label} to current matrix...")
        current_matrix = op_matrix_for_step @ current_matrix
        
    print("UB matrix construction complete.")
    return current_matrix

def calculate_jones_value_from_trace(UB_matrix, n_strands_in_braid, writhe_w, q_param, phi_param):
    """
    Calculates V_M(B)(q) for Markov closure using Formula (2) from AI deep search.
    V_M(B)(q) = (-q)^(3/2 * w(B)) * φ^(N_strands - 1) * Tr[UB_matrix]
    """
    matrix_trace = np.trace(UB_matrix)
    print(f"\nCalculating Jones value (Markov) with N_strands_in_braid={n_strands_in_braid}, writhe_w={writhe_w}:")
    print(f"  Matrix Trace Tr[UB] = {matrix_trace:.6f}")

    phase_factor_exponent = (3.0 / 2.0) * writhe_w
    q_param_complex = complex(q_param)
    phase_factor = (-q_param_complex)**phase_factor_exponent
    print(f"  Phase factor (-q)^(3w/2) = ({-q_param_complex:.4f})^({phase_factor_exponent:.1f}) = {phase_factor:.6f}")
    
    phi_exponent = n_strands_in_braid - 1 
    phi_factor = phi_param**phi_exponent
    print(f"  Phi factor φ^(N_strands-1) = {phi_param:.4f}^({phi_exponent:.0f}) = {phi_factor:.6f}")

    calculated_jones_value = phase_factor * phi_factor * matrix_trace
    print(f"  Calculated V_M(B)(q) = {phase_factor:.6f} * {phi_factor:.6f} * {matrix_trace:.6f} = {calculated_jones_value:.6f}")
    
    return calculated_jones_value
# ==============================================================
if __name__ == "__main__":

    master_start_time = time.time()
    print(f"--- Starting Jones Polynomial Simulation (Markov Closure) for N_QUBITS={N_QUBITS} ---")
    print(f"Target Knot: Left-Handed Trefoil (Markov closure of B=sigma_1^-3)")
    print(f"N_STRANDS_IN_BRAID (for formula): {N_STRANDS_IN_BRAID}")
    print(f"BRAID_WORD (for B_prime_ops): {BRAID_WORD}")
    print(f"WRITHE_W: {WRITHE_W}")
    print(f"Parameters: q={Q_PARAM:.4f}, φ={PHI:.4f}")

    # === Part 1: Setup QIC Basis and Isometry V ===
    print(f"\n=== PART 1: Setup QIC Basis and Isometry V (N_QUBITS={N_QUBITS}) ===")
    start_time = time.time()
    try:
        qic_strings, qic_vectors = qic_core.get_qic_basis(N_QUBITS)
        if not qic_strings or not qic_vectors: raise ValueError("Failed basis gen.")
        dim_QIC = len(qic_strings) # For N_QUBITS=2, dim_QIC = F_4 = 3
        print(f"QIC dim F_{N_QUBITS+2} = {dim_QIC}")
        V_iso = qic_core.construct_isometry_V(qic_vectors) # V_iso is (2^N_QUBITS) x dim_QIC
        if V_iso is None: raise ValueError("Failed V construction.")
        V_csc = V_iso.tocsc()
    except Exception as e: print(f"ERROR Part 1: {e}"); traceback.print_exc(); exit()
    end_time = time.time(); print(f"Part 1 Time: {end_time - start_time:.3f}s")

    # === Part 2: Build Embedded Operators (B'_k and B'_k_dagger) ===
    print(f"\n=== PART 2: Build Embedded Operators for N_QUBITS={N_QUBITS} ===")
    part2_start_time = time.time()
    B_prime_ops = []
    B_prime_dagger_ops = [] 
    operators_ok = True
    num_TL_generators = N_QUBITS - 1 # For N_QUBITS=2, this is 1 (B'_0)
    
    if num_TL_generators < 0 : # Should only happen if N_QUBITS < 1
        print(f"Error: num_TL_generators is {num_TL_generators}. N_QUBITS must be >= 1 (and >=2 for braids).")
        operators_ok = False
        exit()
    if N_QUBITS == 1 and num_TL_generators == 0: # N_QUBITS=1 means no braid ops B'_k
        print(f"N_QUBITS=1, B_prime_ops will be empty. Cannot apply braid word.")
        # This path should ideally not be taken for braid simulations.
        # If BRAID_WORD is non-empty, it will fail later.
        # For sigma_1^3, we need N_QUBITS=2 so B'_0 exists.
        if BRAID_WORD: # Check if BRAID_WORD expects operators
             print("ERROR: BRAID_WORD is specified but N_QUBITS=1 means no B' operators.")
             exit()


    for k_idx in range(num_TL_generators): 
        print(f"\n--- Processing Embedded Operator Index k_idx = {k_idx} (for B'_{k_idx}) ---")
        try:
            Pk_anyon = qic_core.get_kauffman_Pn_anyon_general(N_QUBITS, k_idx, qic_strings, delta=PHI)
            if Pk_anyon is None: raise ValueError(f"Failed P_{k_idx}^anyon.")
            
            Pk_prime = qic_core.build_P_prime_n(k_idx, N_QUBITS, V_csc, Pk_anyon)
            if Pk_prime is None: raise ValueError(f"Failed P'_{k_idx}.")

            Bk_prime = qic_core.build_B_prime_n(k_idx, Pk_prime, N_QUBITS)
            if Bk_prime is None: raise ValueError(f"Failed B'_{k_idx}.")
            B_prime_ops.append(Bk_prime)

            Bk_prime_dagger = Bk_prime.conj().T.tocsc() 
            B_prime_dagger_ops.append(Bk_prime_dagger)
            print(f"  B'_{k_idx} and B'dag_{k_idx} constructed and stored.")

        except Exception as e:
            print(f"ERROR processing k_idx={k_idx}: {e}"); traceback.print_exc(); operators_ok = False; break

    # If N_QUBITS=2, num_TL_generators=1. B_prime_ops should have 1 element.
    if not operators_ok or (num_TL_generators > 0 and len(B_prime_ops) != num_TL_generators) :
        print("\nERROR: Operator construction failed or incomplete."); exit()
    
    end_time = time.time(); print(f"\nPart 2 Time: {end_time - part2_start_time:.3f}s")

    # === Part 3: Simulation using Sparse Matrices (Markov Trace Version) ===
    print(f"\n=== PART 3: Jones Polynomial Simulation (Markov, N_QUBITS={N_QUBITS}) ===")
    part3_start_time = time.time()

    # --- 1. Construct the Braid Unitary UB_matrix ---
    print(f"\n--- 1. Constructing Braid Unitary UB ---")
    try:
        hilbert_dim = 2**N_QUBITS # For N_QUBITS=2, hilbert_dim=4
        identity_H = np.eye(hilbert_dim, dtype=complex) 
        
        UB_matrix = apply_braid_sequence_to_matrix(identity_H, 
                                                   BRAID_WORD, 
                                                   B_prime_ops, 
                                                   B_prime_dagger_ops,
                                                   N_QUBITS) # Pass N_QUBITS here
        
        UB_dagger_UB = UB_matrix.conj().T @ UB_matrix
        if not np.allclose(UB_dagger_UB, identity_H, atol=TOL*100):
            print("WARNING: UB_matrix is not unitary! ||UB_dag*UB - I|| =", np.linalg.norm(UB_dagger_UB - identity_H))
        else:
            print("  UB_matrix unitarity check passed.")

    except Exception as e:
        print(f"ERROR constructing UB_matrix: {e}"); traceback.print_exc(); exit()
    
    # --- 2. Calculate Trace and Resulting Jones Value ---
    print(f"\n--- 2. Calculating Matrix Trace and Jones Value ---")
    try:
        simulated_jones_value = calculate_jones_value_from_trace(
            UB_matrix, 
            N_STRANDS_IN_BRAID, # This is 2 for sigma_1^-3
            WRITHE_W, 
            Q_PARAM, 
            PHI
        )
        
        print(f"\n--- COMPARISON ---")
        print(f"Simulated V_M(B)(q) for B=sigma_1^-3 (N_strands={N_STRANDS_IN_BRAID}, w(B)={WRITHE_W}): {simulated_jones_value:.6f}")
        print(f"Target Analytical J_LH(q) for Left-Handed Trefoil: {J_TREFOIL_ANALYTICAL:.6f}")
        
        error_diff = np.abs(simulated_jones_value - J_TREFOIL_ANALYTICAL)
        real_error = np.abs(simulated_jones_value.real - J_TREFOIL_ANALYTICAL.real)
        imag_error = np.abs(simulated_jones_value.imag - J_TREFOIL_ANALYTICAL.imag)
        print(f"Absolute difference: {error_diff:.6f} (Real diff: {real_error:.6f}, Imag diff: {imag_error:.6f})")

    except Exception as e:
        print(f"ERROR calculating Jones value: {e}"); traceback.print_exc(); exit()
            
    end_time = time.time(); print(f"\nPart 3 Time: {end_time - part3_start_time:.3f}s")
    
    master_end_time = time.time()
    print(f"\n--- N_QUBITS={N_QUBITS} Jones Polynomial Simulation (Markov) Finished ---")
    print(f"Total execution time: {master_end_time - master_start_time:.3f} seconds")