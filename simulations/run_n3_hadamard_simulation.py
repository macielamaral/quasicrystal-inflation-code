# run_n3_hadamard_simulation.py
# Simulates the Hadamard gate approximation using a braid sequence
# from arXiv:2210.12145 for N=3 QIC.

import numpy as np
import time
import traceback
from scipy.sparse.linalg import inv as sparse_inv # For inverse/dagger

# Import functions from the core library module
try:
    import qic_core
except ImportError:
    print("ERROR: Failed to import qic_core.py.")
    print("       Make sure qic_core.py is in the same directory or Python path.")
    exit()

# === Configuration ===
N = 3 # Fixed N=3 for this simulation

# =======================

# --- Helper function for applying sequence ---
def apply_braid_sequence(psi_init, sequence_ops):
    """Applies a list of operators sequentially to an initial state."""
    psi_current = psi_init.copy()
    # Assume sequence_ops contains sparse matrices (CSC format preferred for matvec)
    print(f"Applying sequence of {len(sequence_ops)} operator terms...")
    step = 0
    total_ops = len(sequence_ops)
    for op_matrix in sequence_ops:
        step += 1
        # Ensure sparse @ dense vector = dense vector
        psi_current = op_matrix @ psi_current
        # Optional: Progress indicator
        # if step % 5 == 0 or step == total_ops:
        #    print(f"  ...applied step {step}/{total_ops}")
        # Optional: Check norm preservation periodically if desired (costly)
        # current_norm = np.linalg.norm(psi_current)
        # if not np.isclose(current_norm, 1.0):
        #    print(f"Warning: Norm became {current_norm:.6f} after step {step}. Renormalizing.")
        #    psi_current /= current_norm
    print("Sequence application complete.")
    # Return final state (potentially renormalized if check added)
    return psi_current / np.linalg.norm(psi_current) # Ensure final normalization
    
# === Main Script Execution ===
if __name__ == "__main__":

    master_start_time = time.time()
    print(f"--- Starting N={N} Hadamard Gate Simulation ---")

    qic_strings = []
    qic_vectors = []
    V = None
    V_csc = None
    V_dagger_sparse = None
    P_prime_ops = []
    B_prime_ops = []
    operators_ok = False

    # === Part 1: Setup QIC Basis and Isometry V ===
    print(f"\n=== PART 1: Setup QIC Basis and Isometry V (N={N}) ===")
    start_time = time.time()
    try:
        qic_strings, qic_vectors = qic_core.get_qic_basis(N)
        if not qic_strings or not qic_vectors: raise ValueError(f"Failed to generate QIC basis.")
        dim_QIC = len(qic_strings)
        V = qic_core.construct_isometry_V(qic_vectors)
        if V is None: raise ValueError("Failed to construct isometry V.")
        V_csc = V.tocsc(); V_dagger_sparse = V_csc.conj().T.tocsc()
    except Exception as e: print(f"ERROR during Part 1: {e}"); traceback.print_exc(); exit()
    end_time = time.time(); print(f"Part 1 Execution Time: {end_time - start_time:.3f} seconds")

    # === Part 2: Build N=3 Embedded Braid Operators ===
    print(f"\n=== PART 2: Build Embedded Operators B'_k (N={N}) ===")
    part2_start_time = time.time()
    temp_operators_ok = True
    num_ops_expected = N - 1
    for k in range(num_ops_expected):
        try:
            Pk_anyon = qic_core.get_kauffman_Pn_anyon_general(N, k, qic_strings, delta=qic_core.PHI)
            if Pk_anyon is None: raise ValueError(f"Failed P_anyon gen k={k}")
            Pk_prime = qic_core.build_P_prime_n(k, N, V, Pk_anyon)
            if Pk_prime is None: raise ValueError(f"Failed P' gen k={k}")
            # P_prime_ops.append(Pk_prime) # Not needed unless checking props
            Bk_prime = qic_core.build_B_prime_n(k, Pk_prime, N)
            if Bk_prime is None: raise ValueError(f"Failed B' gen k={k}")
            B_prime_ops.append(Bk_prime)
        except Exception as e: print(f"ERROR op k={k}: {e}"); temp_operators_ok = False; break
    if temp_operators_ok and len(B_prime_ops) == num_ops_expected: operators_ok = True
    else: print("\nERROR: Operator construction failed.")
    end_time = time.time(); print(f"Part 2 Execution Time: {end_time - part2_start_time:.3f} seconds")



    # === Part 3: Define Logical Basis and W_H Sequence ===
    print(f"\n=== PART 3: Define Logical Basis & W_H Sequence ===")
    part3_start_time = time.time()

    if not operators_ok:
        print("Skipping Part 3: Operator construction failed.")
        exit()

    psi_0L_np = None
    psi_1L_np = None
    psi_WH0_ideal = None
    psi_WH1_ideal = None
    WH_sequence_ops = []

    try:
        B0_prime_sparse = B_prime_ops[0].tocsc()
        B1_prime_sparse = B_prime_ops[1].tocsc()

        # --- Precompute needed operators/powers for W_H ---
        print("Precomputing operator powers...")
        B0_sq = (B0_prime_sparse @ B0_prime_sparse).tocsc()
        B0_pow4 = (B0_sq @ B0_sq).tocsc()
        B1_sq = (B1_prime_sparse @ B1_prime_sparse).tocsc()
        B1_pow4 = (B1_sq @ B1_sq).tocsc()
        # Inverse = dagger for unitary B
        B0_inv = B0_prime_sparse.conj().T.tocsc() # Needed if sequence had odd powers
        B1_inv = B1_prime_sparse.conj().T.tocsc()
        B0_inv_sq = (B0_inv @ B0_inv).tocsc()
        B1_inv_sq = (B1_inv @ B1_inv).tocsc()
        print("Operator powers computed.")
        # --- END PRECOMPUTATION ---

        # --- Define W_H sequence from Rouabah Eq. (11) ---
        # W_H = s1^4 s2^-2 s1^2 s2^-2 s1^2 s2^2 s1^-2 s2^4 s1^2 s2^-2 s1^-2 s2^2 s1^2
        # Apply operators right-to-left from formula
        WH_sequence_ops = [
            B0_sq,       # sigma_1^2 (Rightmost term)
            B1_sq,       # sigma_2^2
            B0_inv_sq,   # sigma_1^-2
            B1_inv_sq,   # sigma_2^-2
            B0_sq,       # sigma_1^2
            B1_pow4,     # sigma_2^4
            B0_inv_sq,   # sigma_1^-2
            B1_sq,       # sigma_2^2
            B0_sq,       # sigma_1^2
            B1_inv_sq,   # sigma_2^-2
            B0_sq,       # sigma_1^2
            B1_inv_sq,   # sigma_2^-2
            B0_pow4      # sigma_1^4 (Leftmost term)
        ]
        print(f"Defined W_H braid sequence with {len(WH_sequence_ops)} operator terms (30 elementary braids).")
        # --- END SEQUENCE DEFINITION ---

        # Define logical basis states (simple mapping for simulation)
        idx_L0 = qic_strings.index('101') # |0>_L = |101>
        idx_L1 = qic_strings.index('010') # |1>_L = |010>

        qic_vec_L0 = np.zeros(dim_QIC, dtype=complex); qic_vec_L0[idx_L0] = 1.0
        qic_vec_L1 = np.zeros(dim_QIC, dtype=complex); qic_vec_L1[idx_L1] = 1.0

        psi_0L_np = V_csc @ qic_vec_L0
        psi_1L_np = V_csc @ qic_vec_L1
        print(f"Logical |0>_L mapped to |{qic_strings[idx_L0]}>")
        print(f"Logical |1>_L mapped to |{qic_strings[idx_L1]}>")

        # --- Define Ideal Target States for -iH ---
        target_phase = -1j # Target is -iH according to Rouabah paper text
        psi_WH0_ideal = target_phase * (psi_0L_np + psi_1L_np) / np.sqrt(2)
        psi_WH1_ideal = target_phase * (psi_0L_np - psi_1L_np) / np.sqrt(2)
        print("Defined ideal target states for -iH.")
        # --- END TARGET STATES ---

    except Exception as e:
        print(f"ERROR during Part 3 setup: {e}"); traceback.print_exc(); exit()
    end_time = time.time(); print(f"Part 3 Setup Time: {end_time - part3_start_time:.3f} seconds")
    
    # === Part 4: Simulate Braiding and Calculate Fidelity ===
    print(f"\n=== PART 4: Simulate W_H Sequence and Calculate Fidelity ===")
    part4_start_time = time.time()
    psi_final_0 = None
    psi_final_1 = None
    fidelity_0 = 0.0
    fidelity_1 = 0.0

    if not operators_ok or not WH_sequence_ops: # Check if sequence was built
        print("Skipping Part 4: Operator construction failed or sequence empty.")
    else:
        try:
            # Simulate W_H|0>_L
            print(f"\nSimulating sequence W_H on |0>_L = |{qic_strings[idx_L0]}>...")
            psi_final_0 = apply_braid_sequence(psi_0L_np, WH_sequence_ops)
            norm_final_0 = np.linalg.norm(psi_final_0)
            print(f"Norm of final state (from |0>_L): {norm_final_0:.6f}")
            # Final state should be normalized by apply_braid_sequence now

            # Calculate Fidelity F0 = |<-iH0_ideal|psi_final_0>|^2
            # Use vdot for <bra|ket> = bra^dagger * ket
            fidelity_0 = np.abs(np.vdot(psi_WH0_ideal, psi_final_0))**2
            print(f"--> Fidelity F_0 = |<-iH0_ideal|psi_final_0>|^2 = {fidelity_0:.6f}")

            # Simulate W_H|1>_L
            print(f"\nSimulating sequence W_H on |1>_L = |{qic_strings[idx_L1]}>...")
            psi_final_1 = apply_braid_sequence(psi_1L_np, WH_sequence_ops)
            norm_final_1 = np.linalg.norm(psi_final_1)
            print(f"Norm of final state (from |1>_L): {norm_final_1:.6f}")

            # Calculate Fidelity F1 = |<-iH1_ideal|psi_final_1>|^2
            fidelity_1 = np.abs(np.vdot(psi_WH1_ideal, psi_final_1))**2
            print(f"--> Fidelity F_1 = |<-iH1_ideal|psi_final_1>|^2 = {fidelity_1:.6f}")

            # Optional: Analyze final states in QIC basis
            print("\nAnalyzing final states in QIC basis...")
            if psi_final_0 is not None:
                qic_final_0 = V_dagger_sparse @ psi_final_0
                output_str0 = []
                for i, coeff in enumerate(qic_final_0):
                    if abs(coeff) > qic_core.TOL * 10: output_str0.append(f"({coeff.real:.3f}{coeff.imag:+.3f}j)|{qic_strings[i]}>")
                print("  Final state from |0>_L:")
                print("  " + (" + ".join(output_str0) if output_str0 else "(Zero vector)"))

            if psi_final_1 is not None:
                qic_final_1 = V_dagger_sparse @ psi_final_1
                output_str1 = []
                for i, coeff in enumerate(qic_final_1):
                    if abs(coeff) > qic_core.TOL * 10: output_str1.append(f"({coeff.real:.3f}{coeff.imag:+.3f}j)|{qic_strings[i]}>")
                print("  Final state from |1>_L:")
                print("  " + (" + ".join(output_str1) if output_str1 else "(Zero vector)"))

        except Exception as e:
            print(f"ERROR during Part 4 simulation: {e}"); traceback.print_exc()

    end_time = time.time(); print(f"\nPart 4 Execution Time: {end_time - part4_start_time:.3f} seconds")

    master_end_time = time.time()
    print(f"\n--- N={N} W_H Simulation Finished ---")
    print(f"Total execution time: {master_end_time - master_start_time:.3f} seconds")
    