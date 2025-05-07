# run_n3_simulation_cycle.py
# Performs the Prepare->Braid->Project->Analyze simulation cycle for N=3.

import numpy as np
import time
import traceback

# Import functions from the core library module
try:
    import qic_core
except ImportError:
    print("ERROR: Failed to import qic_core.py.")
    print("       Make sure qic_core.py is in the same directory or Python path.")
    exit()

# Optionally import Qiskit (needed for this script's core functionality)
if qic_core.QISKIT_AVAILABLE:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    # Note: AerEstimator is imported within qic_core if available
else:
    print("ERROR: Qiskit is required for this script (run_n3_simulation_cycle.py) but is not available.")
    exit() # Exit if Qiskit is essential and not found


# === Configuration ===
N = 3 # Choose the fixed N for this simulation cycle
LAMBDA = 1.0 # QIC Hamiltonian coupling
INITIAL_STATE_STR = '101' # Choose initial QIC basis state string for the N level,  
# for example N=4, choose a valid QIC basis string for the new N. For N=4, the basis is ['0101', '0110', '0111', '1010', '1011', '1101', '1110', '1111']. So you could choose:  INITIAL_STATE_STR = '1010'
BRAID_TO_APPLY_IDX = 0    # Choose which braid operator B'_k to apply (0 or 1 for N=3)
PROJECTOR_TO_APPLY_IDX = N-2 # Choose final projector P'_k (0 or 1 for N=3) - usually N-2 = 1 for deflation

# =======================

if __name__ == "__main__":

    master_start_time = time.time()
    print(f"--- Starting N={N} Simulation Cycle ---")
    print(f"Initial State: |{INITIAL_STATE_STR}>")
    print(f"Braid Operator: B'_{BRAID_TO_APPLY_IDX}")
    print(f"Projector: P'_{PROJECTOR_TO_APPLY_IDX}")

    qic_strings = []
    qic_vectors = []
    V = None
    V_csc = None
    V_dagger_sparse = None
    P_anyon_ops = []
    P_prime_ops = []
    B_prime_ops = []
    H_qic_op = None
    operators_ok = False # Flag

    # === Part 1: Setup QIC Basis and Isometry V ===
    print(f"\n=== PART 1: Setup QIC Basis and Isometry V (N={N}) ===")
    start_time = time.time()
    try:
        qic_strings, qic_vectors = qic_core.get_qic_basis(N)
        if not qic_strings or not qic_vectors:
            raise ValueError(f"Failed to generate QIC basis for N={N}.")

        dim_QIC = len(qic_strings)
        print(f"QIC subspace dimension F_{N+2} = {dim_QIC}")

        V = qic_core.construct_isometry_V(qic_vectors)
        if V is None:
             raise ValueError("Failed to construct isometry V.")

        print(f"Isometry V constructed (shape {V.shape}).")
        V_csc = V.tocsc() # Ensure CSC format
        V_dagger_sparse = V_csc.conj().T.tocsc() # Precompute V_dagger

    except Exception as e:
        print(f"ERROR during Part 1: {e}")
        traceback.print_exc()
        exit()
    end_time = time.time(); print(f"Part 1 Execution Time: {end_time - start_time:.3f} seconds")


    # === Part 2: Build N=3 Embedded Operators ===
    # (Skipping property checks here, assuming they passed in verification script)
    print(f"\n=== PART 2: Build Embedded Operators P'_k, B'_k (N={N}) ===")
    part2_start_time = time.time()
    temp_operators_ok = True
    num_ops_expected = N - 1
    for k in range(num_ops_expected): # Loops k=0, 1 for N=3
        print(f"\n--- Processing Operator Index k = {k} ---")
        try:
            # Build Anyonic Pk
            Pk_anyon = qic_core.get_kauffman_Pn_anyon_general(N, k, qic_strings, delta=qic_core.PHI)
            if Pk_anyon is None: raise ValueError(f"Failed to get P_{k}^anyon matrix.")
            # P_anyon_ops.append(Pk_anyon) # Don't necessarily need to store

            # Build Embedded P'k
            Pk_prime = qic_core.build_P_prime_n(k, N, V, Pk_anyon)
            if Pk_prime is None: raise ValueError(f"Failed to build P'_{k}.")
            P_prime_ops.append(Pk_prime)

            # Build Embedded B'k
            Bk_prime = qic_core.build_B_prime_n(k, Pk_prime, N)
            if Bk_prime is None: raise ValueError(f"Failed to build B'_{k}.")
            B_prime_ops.append(Bk_prime)

        except Exception as e:
            print(f"ERROR processing operator index k={k}: {e}")
            traceback.print_exc()
            temp_operators_ok = False
            break

    if temp_operators_ok and len(P_prime_ops) == num_ops_expected:
        print(f"\nSuccessfully constructed {len(P_prime_ops)} P' and B' operators for N={N}.")
        operators_ok = True
    else:
        print("\nERROR: Operator construction failed.")
        operators_ok = False

    end_time = time.time(); print(f"Part 2 Execution Time: {end_time - part2_start_time:.3f} seconds")

    # === Part 3: Qiskit Simulation Cycle ===
    print(f"\n=== PART 3: Qiskit Simulation Cycle (N={N}) ===")
    part3_start_time = time.time()

    if not operators_ok:
        print("Skipping Part 3: Operator construction failed.")
    else:
        # --- 1. Choose and Prepare Initial State Vector ---
        print(f"\n--- 1. Preparing Initial State |{INITIAL_STATE_STR}> ---")
        if INITIAL_STATE_STR not in qic_strings:
            print(f"ERROR: Initial state '{INITIAL_STATE_STR}' is not a valid QIC state for N={N}.")
            exit()

        try:
             initial_state_idx = qic_strings.index(INITIAL_STATE_STR)
        except ValueError:
             print(f"ERROR: Could not find index for initial state '{INITIAL_STATE_STR}'.")
             exit()

        qic_vec_5d = np.zeros(dim_QIC, dtype=complex)
        qic_vec_5d[initial_state_idx] = 1.0

        try:
            psi_init_np = V_csc @ qic_vec_5d
            init_norm = np.linalg.norm(psi_init_np)
            print(f"Initial 8D state |{INITIAL_STATE_STR}> prepared. Norm: {init_norm:.6f}")
            if not np.isclose(init_norm, 1.0):
                print("WARNING: Initial state norm is not 1. Check Isometry V.")
        except Exception as e:
            print(f"ERROR applying isometry V: {e}"); traceback.print_exc(); exit()

        # --- 2. Create Qiskit Circuit and Initialize State ---
        print("\n--- 2. Creating and Initializing Qiskit Circuit ---")
        try:
            n_qubits = N
            circuit = QuantumCircuit(n_qubits, name="QIC_Sim")
            circuit.initialize(psi_init_np, list(range(n_qubits)))
            print(f"Circuit initialized with state |{INITIAL_STATE_STR}>.")
        except Exception as e:
            print(f"ERROR initializing Qiskit circuit: {e}"); traceback.print_exc(); exit()

        # --- 3. Apply Braid Operator Gate ---
        print(f"\n--- 3. Applying Braid Operator B'_{BRAID_TO_APPLY_IDX} ---")
        if not (0 <= BRAID_TO_APPLY_IDX < len(B_prime_ops)):
            print(f"ERROR: Invalid BRAID_TO_APPLY_IDX ({BRAID_TO_APPLY_IDX})."); exit()
        try:
            B_k_prime_sparse = B_prime_ops[BRAID_TO_APPLY_IDX]
            B_k_prime_dense = B_k_prime_sparse.toarray()
            circuit.unitary(B_k_prime_dense, list(range(n_qubits)), label=f"B'_{BRAID_TO_APPLY_IDX}")
            print(f"Braid gate B'_{BRAID_TO_APPLY_IDX} applied using circuit.unitary().")
        except Exception as e:
            print(f"ERROR applying braid gate: {e}"); traceback.print_exc(); exit()

        # --- 4. Simulate Circuit to Get Braided State ---
        print(f"\n--- 4. Simulating Circuit to get Braided State ---")
        try:
            sv_braided = Statevector.from_instruction(circuit)
            psi_braided_np = sv_braided.data
            braided_norm = np.linalg.norm(sv_braided.data)
            print(f"Statevector norm after B'_{BRAID_TO_APPLY_IDX}: {braided_norm:.6f}")
            if not np.isclose(braided_norm, 1.0):
                print("WARNING: State norm changed after braiding. Check unitarity of B'.")
        except Exception as e:
            print(f"ERROR simulating circuit statevector: {e}"); traceback.print_exc(); exit()

        # --- 5. Apply Final Projector Mathematically ---
        print(f"\n--- 5. Applying Final Projector P'_{PROJECTOR_TO_APPLY_IDX} ---")
        final_projector_idx = PROJECTOR_TO_APPLY_IDX
        if not (0 <= final_projector_idx < len(P_prime_ops)):
            print(f"ERROR: Invalid PROJECTOR_TO_APPLY_IDX ({final_projector_idx})."); exit()
        try:
            P_final_prime_sparse = P_prime_ops[final_projector_idx]
            P_final_prime_dense = P_final_prime_sparse.toarray()
            psi_projected_np = P_final_prime_dense @ psi_braided_np
            projection_prob = np.linalg.norm(psi_projected_np)**2
            print(f"Projection Probability (Norm^2 after P'_{final_projector_idx}): {projection_prob:.6f}")

            psi_final_np = None
            if projection_prob > qic_core.TOL:
                psi_final_np = psi_projected_np / np.sqrt(projection_prob)
                final_norm = np.linalg.norm(psi_final_np)
                print(f"Final state norm after projection and normalization: {final_norm:.6f}")
            else:
                psi_final_np = np.zeros_like(psi_projected_np)
                print("Final state is zero vector (projection annihilated the state).")
        except Exception as e:
            print(f"ERROR applying final projector: {e}"); traceback.print_exc(); exit()

        # --- 6. Analyze and Verify ---
        print("\n--- 6. Analysis and Verification ---")
        H_qic_op = qic_core.build_qic_hamiltonian_op(N, lam=LAMBDA, verbose=False)

        if projection_prob > qic_core.TOL and H_qic_op is not None:
            print("Verifying energy of final projected state...")
            is_gs_final, energy_final = qic_core.verify_energy(psi_final_np, H_qic_op, n_qubits, label="Final Projected State")
            if is_gs_final: print("  Energy check PASSED: Final state remains in QIC subspace (Energy ~ 0).")
            else: print(f"  Energy check FAILED: Final state energy is non-zero ({energy_final:.4f})!")
        elif H_qic_op is None: print("Skipping final energy verification (Hamiltonian not built).")
        else: print("Skipping final energy verification (Final state is zero).")

        if projection_prob > qic_core.TOL:
            try:
                qic_final_vec_5d = V_dagger_sparse @ psi_final_np
                print("\nFinal state represented in N=3 QIC basis (coefficients):")
                output_str = []
                for i, coeff in enumerate(qic_final_vec_5d):
                    if abs(coeff) > qic_core.TOL * 10:
                         output_str.append(f"({coeff.real:.3f}{coeff.imag:+.3f}j)|{qic_strings[i]}>")
                if not output_str: print("  (Zero vector in QIC basis after TOL)")
                else: print("  " + " + ".join(output_str))
            except Exception as e:
                print(f"ERROR during back-projection to QIC basis: {e}"); traceback.print_exc()

    end_time = time.time()
    print(f"\nPart 3 Execution Time: {end_time - part3_start_time:.3f} seconds")

    master_end_time = time.time()
    print(f"\n--- N={N} Simulation Cycle Finished ---")
    print(f"Total execution time: {master_end_time - master_start_time:.3f} seconds")