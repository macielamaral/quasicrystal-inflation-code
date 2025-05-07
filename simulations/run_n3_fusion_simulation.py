# run_n3_fusion_simulation.py
# Calculates projection probabilities P(1) = || P'_1 B'_k |s> ||^2
# for N=3 QIC states |s>, braids B'_k.

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

# Optional: For plotting results later
# import matplotlib.pyplot as plt

# === Configuration ===
N = 3 # Fixed N for this simulation
PROJECTOR_TO_APPLY_IDX = 1 # Index of the projector P'_k to apply (usually N-2 = 1)

# =======================

if __name__ == "__main__":

    master_start_time = time.time()
    print(f"--- Starting N={N} Fusion Probability Simulation ---")
    print(f"Calculating Prob(1) = || P'_{PROJECTOR_TO_APPLY_IDX} B'_k |s> ||^2")

    qic_strings = []
    qic_vectors = []
    V = None
    V_csc = None
    P_prime_ops = []
    B_prime_ops = []
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

    except Exception as e:
        print(f"ERROR during Part 1: {e}")
        traceback.print_exc()
        exit()
    end_time = time.time(); print(f"Part 1 Execution Time: {end_time - start_time:.3f} seconds")

    # === Part 2: Build N=3 Embedded Operators ===
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
        # Get the specific projector matrix we need
        if not (0 <= PROJECTOR_TO_APPLY_IDX < len(P_prime_ops)):
            print(f"ERROR: Invalid PROJECTOR_TO_APPLY_IDX ({PROJECTOR_TO_APPLY_IDX}).")
            operators_ok = False
        else:
            P_final_prime_sparse = P_prime_ops[PROJECTOR_TO_APPLY_IDX].tocsc()
            print(f"Using P'_{PROJECTOR_TO_APPLY_IDX} for final projection.")

    else:
        print("\nERROR: Operator construction failed.")
        operators_ok = False

    end_time = time.time(); print(f"Part 2 Execution Time: {end_time - part2_start_time:.3f} seconds")


    # === Part 3: Calculate Fusion Probabilities ===
    print(f"\n=== PART 3: Calculating Fusion Probabilities (N={N}) ===")
    part3_start_time = time.time()

    results = {} # Dictionary to store results: results[(init_state, braid_idx)] = probability

    if not operators_ok:
        print("Skipping Part 3: Operator construction failed.")
    else:
        print(f"\nApplying Braids B'_k and Projector P'_{PROJECTOR_TO_APPLY_IDX}...")
        # Loop through all initial basis states
        for init_idx, init_state_str in enumerate(qic_strings):
            print(f"\n--- Initial State: |{init_state_str}> ---")
            # Create the 5D QIC vector
            qic_vec_5d = np.zeros(dim_QIC, dtype=complex)
            qic_vec_5d[init_idx] = 1.0
            # Embed into 8D space
            psi_init_np = V_csc @ qic_vec_5d

            # Loop through all braid operators B'_k
            for braid_idx, Bk_prime_sparse in enumerate(B_prime_ops):
                try:
                    # Apply Braid Mathematically
                    psi_braided_np = Bk_prime_sparse @ psi_init_np

                    # Apply Final Projector Mathematically
                    psi_projected_np = P_final_prime_sparse @ psi_braided_np

                    # Calculate Projection Probability (Norm Squared)
                    projection_prob = np.linalg.norm(psi_projected_np)**2

                    print(f"  Braid: B'_{braid_idx} -> Projection Prob (P'_{PROJECTOR_TO_APPLY_IDX}): {projection_prob:.6f}")
                    results[(init_state_str, braid_idx)] = projection_prob

                except Exception as e:
                    print(f"ERROR during calculation for |{init_state_str}>, B'_{braid_idx}: {e}")
                    traceback.print_exc()
                    results[(init_state_str, braid_idx)] = np.nan # Mark as failed

    # --- Simple summary of results ---
    print("\n--- Results Summary ---")
    target_prob = qic_core.PHI**(-2)
    print(f"Target probability P(1) = phi^-2 = {target_prob:.6f}")
    consistent_count = 0
    total_count = 0
    for (init_state, braid_idx), prob in results.items():
         total_count += 1
         print(f"  State |{init_state}>, Braid B'_{braid_idx} -> Prob = {prob:.6f}", end="")
         if np.isnan(prob):
             print(" (Calculation FAILED)")
         elif np.isclose(prob, target_prob, atol=qic_core.TOL*100):
             print(" (Consistent)")
             consistent_count += 1
         else:
             print(" (INCONSISTENT!)")
    print(f"\n{consistent_count} out of {total_count} calculated probabilities are consistent with phi^-2.")


    end_time = time.time(); print(f"\nPart 3 Execution Time: {end_time - part3_start_time:.3f} seconds")

    master_end_time = time.time()
    print(f"\n--- N={N} Fusion Simulation Finished ---")
    print(f"Total execution time: {master_end_time - master_start_time:.3f} seconds")