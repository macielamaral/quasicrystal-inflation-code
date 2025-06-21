# run_operator_verification.py
# Generates QIC operators for a given N and verifies their properties.

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

# Optionally import Qiskit (only needed for Part 4 energy checks)
if qic_core.QISKIT_AVAILABLE:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    # Note: AerEstimator is imported within qic_core if available
else:
    print("Note: Qiskit not available, skipping Part 4 energy checks.")

# === Configuration ===
N = 10  # Set the desired N value for verification (e.g., 3, 4, 5, ..., 10)
LAMBDA = 1.0 # Hamiltonian coupling constant (for energy checks)
RUN_ENERGY_CHECKS = True # Set to False to skip Qiskit-dependent energy checks

# =======================

if __name__ == "__main__":

    master_start_time = time.time()
    print(f"--- Starting Operator Verification for N={N} ---")

    qic_strings = []
    qic_vectors = []
    V = None
    V_csc = None
    V_dagger_sparse = None
    P_anyon_ops = []
    P_prime_ops = []
    B_prime_ops = []
    H_qic_op = None
    operators_ok = False # Flag to track if operators were built successfully

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
        exit() # Exit if setup fails
    end_time = time.time(); print(f"Part 1 Execution Time: {end_time - start_time:.3f} seconds")

    # === Part 2: Build Anyonic and Embedded Operators ===
    print(f"\n=== PART 2: Build Anyonic and Embedded Operators (N={N}) ===")
    part2_start_time = time.time()
    if N >= 2:
        num_ops_expected = N - 1
        print(f"Building {num_ops_expected} operators (k=0 to {num_ops_expected-1})...")
        temp_operators_ok = True
        for k in range(num_ops_expected):
            print(f"\n--- Processing Operator Index k = {k} ---")
            try:
                # Build Anyonic Pk
                Pk_anyon = qic_core.get_kauffman_Pn_anyon_general(N, k, qic_strings, delta=qic_core.PHI)
                if Pk_anyon is None: raise ValueError(f"Failed to get P_{k}^anyon matrix.")
                P_anyon_ops.append(Pk_anyon)

                # print(Pk_anyon.toarray())  # Convert sparse matrix to dense NumPy array and print

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
                break # Stop processing if one operator fails

        print("\n--- Finished Operator Construction Loop ---")
        if temp_operators_ok and len(P_prime_ops) == num_ops_expected:
            print(f"Successfully constructed {len(P_prime_ops)} P_anyon, P', and B' operators.")
            operators_ok = True
        elif not temp_operators_ok:
            print("\nERROR: Operator construction failed or aborted.")
        else:
            print(f"\nWARNING: Operator list length mismatch? Expected {num_ops_expected}, Got {len(P_prime_ops)}")
            operators_ok = False
    else: # N < 2
        print(f"Skipping operator build for N={N} < 2.")
        operators_ok = True # No operators expected

    end_time = time.time(); print(f"Part 2 Execution Time: {end_time - part2_start_time:.3f} seconds")

    # === Part 3: Run Operator Property Checks ===
    print(f"\n=== PART 3: Operator Property Verification (N={N}) ===")
    part3_start_time = time.time()
    anyon_checks_passed = False
    embedded_checks_passed = False

    if operators_ok and N >= 2:
        # Check Anyonic Operators first
        anyon_checks_passed = qic_core.check_anyon_operator_properties(
            P_anyon_ops, N, delta=qic_core.PHI, tol=qic_core.TOL
        )
        if anyon_checks_passed:
            print("\n**********************************************************")
            print("SUCCESS: Anyonic operator algebra checks passed for P_k!")
            print("**********************************************************")
        else:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("WARNING: Anyonic operator algebra checks FAILED for P_k!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # Check Embedded Operators
        embedded_checks_passed = qic_core.check_embedded_operator_properties(
            P_prime_ops, B_prime_ops, N, tol=qic_core.TOL
        )
        if embedded_checks_passed:
            print("\n**********************************************************")
            print("SUCCESS: Embedded operator algebra checks passed for P'_k / B'_k!")
            print("**********************************************************")
        else:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("WARNING: Embedded operator algebra checks FAILED for P'_k / B'_k!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    elif N < 2:
        print("Skipping operator verification for N<2.")
        anyon_checks_passed = True # Pass trivially
        embedded_checks_passed = True # Pass trivially
    else:
        print("Skipping operator verification due to construction errors.")

    end_time = time.time(); print(f"Part 3 Execution Time: {end_time - part3_start_time:.3f} seconds")

    # === Part 4: (Optional) Energy Check on Superposition State ===
    print(f"\n=== PART 4: (Optional) Energy & Braiding Check (N={N}) ===")
    part4_start_time = time.time()

    if not RUN_ENERGY_CHECKS:
        print("Skipping Part 4: RUN_ENERGY_CHECKS set to False.")
    elif not qic_core.QISKIT_AVAILABLE:
        print("Skipping Part 4: Qiskit is not available.")
    elif not operators_ok:
        print("Skipping Part 4: Operator construction or verification failed.")
    elif N < 2:
         print("Skipping Part 4: N < 2.")
    elif not P_prime_ops: # Need operators to run check
         print("Skipping Part 4: Operators list is empty.")
    else:
        try:
            print("\n--- Preparing Superposition State & Hamiltonian ---")
            # Create equal superposition state in QIC space, then embed
            qic_superpos_5d = np.ones(dim_QIC, dtype=complex) / np.sqrt(dim_QIC)
            psi_superpos_np = V_csc @ qic_superpos_5d
            superpos_norm = np.linalg.norm(psi_superpos_np)
            if not np.isclose(superpos_norm, 1.0):
                 print(f"WARNING: Superposition state norm is {superpos_norm:.4f}, expected 1.0.")
            print(f"Created equal superposition QIC state |psi_QIC_superpos> (norm={superpos_norm:.4f})")

            # Build Hamiltonian
            H_qic_op = qic_core.build_qic_hamiltonian_op(N, lam=LAMBDA, verbose=True)

            if H_qic_op is not None and psi_superpos_np is not None:
                # Verify initial energy
                print("\nVerifying energy of initial |psi_QIC_superpos>...")
                is_gs_init, _ = qic_core.verify_energy(
                    psi_superpos_np, H_qic_op, N, label="|psi_QIC_superpos> init", verbose=True
                )
                if not is_gs_init:
                     print("WARNING: Initial superposition state energy is not close to zero!")

                # Apply a braid operator (e.g., middle one)
                target_k = (N - 1) // 2 # Choose middle index
                print(f"\nApplying braid B'_{target_k} to |psi_QIC_superpos>...")
                Bk_prime_target = B_prime_ops[target_k].tocsc()
                psi_braided_np = Bk_prime_target @ psi_superpos_np
                norm_braided = np.linalg.norm(psi_braided_np)
                print(f"Norm after braiding B'_{target_k}: {norm_braided:.6f}")

                if not np.isclose(norm_braided, 1.0, atol=qic_core.TOL*100):
                    print(f"ERROR: Braiding significantly changed state norm! Check B' unitarity.")
                else:
                    # Verify energy after braiding
                    print(f"\nVerifying energy of braided state B'_{target_k}|psi_QIC_superpos>...")
                    is_gs_final, _ = qic_core.verify_energy(
                        psi_braided_np, H_qic_op, N, label=f"B'_{target_k}|psi_QIC_superpos>", verbose=True
                    )
                    if is_gs_final:
                        print(f"SUCCESS: Braided state remains in QIC ground state (Energy ~ 0).")
                    else:
                        print(f"WARNING: Braided state has non-zero energy!")
                        print(f"         This indicates B'_{target_k} might not preserve the QIC subspace perfectly (check algebra/implementation).")
            else:
                print("Skipping energy checks (Hamiltonian or state prep failed).")

        except Exception as e:
            print(f"\nERROR during Part 4: {e}")
            traceback.print_exc()

    end_time = time.time(); print(f"Part 4 Execution Time: {end_time - part4_start_time:.3f} seconds")

    master_end_time = time.time()
    print(f"\n--- Verification Script Finished for N={N} ---")
    print(f"Total execution time: {master_end_time - master_start_time:.3f} seconds")
    print(f"Overall Status: Anyonic Checks {'Passed' if anyon_checks_passed else 'Failed/Skipped'}, Embedded Checks {'Passed' if embedded_checks_passed else 'Failed/Skipped'}")