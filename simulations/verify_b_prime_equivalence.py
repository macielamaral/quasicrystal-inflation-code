# verify_b_prime_equivalence.py
# Verifies the efficient local gate implementation of B'_k operators
# against the original full matrix construction for small N.

import numpy as np
import time
import traceback
import os
import sys

# --- Ensure qic_core is importable ---
try:
    import qic_core
except ImportError:
    try:
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # src_dir = os.path.join(os.path.dirname(current_dir), 'src')
        # if src_dir not in sys.path:
        #    sys.path.insert(0, src_dir)
        import qic_core
        print("Successfully imported qic_core (path might have been adjusted).")
    except ImportError as e:
        print("ERROR: Failed to import qic_core.py.")
        print("       Please ensure qic_core.py is accessible.")
        print(f"       Original error: {e}")
        exit()

# Import Qiskit components
if qic_core.QISKIT_AVAILABLE:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit.quantum_info import average_gate_fidelity
    from scipy.linalg import eigh # For purification if needed, or np.linalg.eigh
    from scipy.sparse import csc_matrix
else:
    print("ERROR: Qiskit is required for this script but not available.")
    exit()

# ===== Configuration =====
N_VALUES_TO_VERIFY = [3, 4, 5, 6, 7, 8, 9, 10] # List of N_QUBITS to verify (N>=3 for 3-qubit gates)
# Tolerances for equivalence checks
EQUIV_ATOL = 1e-7 # Absolute tolerance for Operator.equiv()
EQUIV_RTOL = 1e-5 # Relative tolerance for Operator.equiv()
# =======================

# --- Functions to compute G_typeA and G_typeB (copied from efficient sweep script) ---
def compute_G_typeA_unitary():
    print("\n--- Computing Local 3-Qubit Unitary G_typeA (from B'_0 for N=3) ---")
    N_local = 3
    k_idx_local = 0 
    try:
        qic_strings_N3, qic_vectors_N3 = qic_core.get_qic_basis(N_local)
        if not qic_strings_N3: raise ValueError("Failed QIC basis N=3 for G_typeA.")
        V_N3_iso = qic_core.construct_isometry_V(qic_vectors_N3)
        if V_N3_iso is None: raise ValueError("Failed V_iso N=3 for G_typeA.")
        V_N3_csc = V_N3_iso.tocsc()
        P_anyon_N3_k0_sparse = qic_core.get_kauffman_Pn_anyon_general(
            N_local, k_idx_local, qic_strings_N3, delta=qic_core.PHI
        )
        if P_anyon_N3_k0_sparse is None: raise ValueError("Failed P_anyon N=3,k=0 for G_typeA.")
        P_prime_N3_k0_sparse = qic_core.build_P_prime_n(
            k_idx_local, N_local, V_N3_csc, P_anyon_N3_k0_sparse
        )
        if P_prime_N3_k0_sparse is None: raise ValueError("Failed P'_local N=3 for G_typeA.")
        G_sparse = qic_core.build_B_prime_n(
            k_idx_local, P_prime_N3_k0_sparse, N_local
        )
        if G_sparse is None: raise ValueError("Failed B'_local N=3 for G_typeA.")
        G_dense = G_sparse.toarray() 
        identity_8x8 = np.eye(2**N_local, dtype=complex)
        if np.allclose(G_dense @ G_dense.conj().T, identity_8x8, atol=1e-8):
            print("G_typeA (B'_0 for N=3) is unitary. Dimensions: ", G_dense.shape)
        else:
            print("WARNING: Computed G_typeA is NOT unitary.")
            return None # Crucial to return None if not unitary
        return G_dense
    except Exception as e:
        print(f"ERROR during G_typeA computation: {e}")
        traceback.print_exc()
        return None

def compute_G_typeB_unitary(P1_anyon_N3_dense_input):
    print("\n--- Computing Local 3-Qubit Unitary G_typeB (from B'_1 for N=3) ---")
    N_local = 3
    k_idx_B_type = 1 
    try:
        qic_strings_N3, qic_vectors_N3 = qic_core.get_qic_basis(N_local)
        if not qic_strings_N3: raise ValueError("Failed QIC basis N=3 for G_typeB.")
        V_N3_iso = qic_core.construct_isometry_V(qic_vectors_N3)
        if V_N3_iso is None: raise ValueError("Failed V_iso N=3 for G_typeB.")
        V_N3_csc = V_N3_iso.tocsc()
        P1_anyon_N3_sparse = csc_matrix(P1_anyon_N3_dense_input)
        P_prime_initial_sparse = qic_core.build_P_prime_n(
            k_idx_B_type, N_local, V_N3_csc, P1_anyon_N3_sparse
        )
        if P_prime_initial_sparse is None: raise ValueError("Failed initial P'_prime for G_typeB.")
        P_prime_initial_dense = P_prime_initial_sparse.toarray()
        print("Initial P'_prime (for G_typeB) properties before purification:")
        is_idempotent_initial = np.allclose(P_prime_initial_dense @ P_prime_initial_dense, P_prime_initial_dense, atol=1e-9)
        is_hermitian_initial = np.allclose(P_prime_initial_dense, P_prime_initial_dense.conj().T, atol=1e-9)
        print(f"  Initial P'_prime is idempotent? {is_idempotent_initial}")
        print(f"  Initial P'_prime is Hermitian? {is_hermitian_initial}")
        P_prime_to_purify = P_prime_initial_dense
        if not is_hermitian_initial:
            print("  Making P_prime_initial_dense Hermitian explicitly for purification...")
            P_prime_to_purify = (P_prime_initial_dense + P_prime_initial_dense.conj().T) / 2.0
        
        eigvals, eigvecs = eigh(P_prime_to_purify)
        P_prime_purified_dense = np.zeros_like(P_prime_to_purify, dtype=complex)
        eigenvalue_one_threshold = 0.5
        num_selected_eigvals = 0
        for i in range(len(eigvals)):
            if eigvals[i] > eigenvalue_one_threshold:
                num_selected_eigvals +=1
                vec = eigvecs[:, i].reshape(-1, 1)
                P_prime_purified_dense += vec @ vec.conj().T
        print(f"  Purified P'_prime by selecting {num_selected_eigvals} eigenvalues > {eigenvalue_one_threshold}.")
        is_idempotent_purified = np.allclose(P_prime_purified_dense @ P_prime_purified_dense, P_prime_purified_dense, atol=1e-9)
        is_hermitian_purified = np.allclose(P_prime_purified_dense, P_prime_purified_dense.conj().T, atol=1e-9)
        print(f"  Purified P'_prime is idempotent? {is_idempotent_purified}")
        print(f"  Purified P'_prime is Hermitian? {is_hermitian_purified}")
        if not (is_idempotent_purified and is_hermitian_purified):
             print("  WARNING: Purified P'_prime is STILL not a good numerical projector!")
        
        P_prime_for_B_build_sparse = csc_matrix(P_prime_purified_dense)
        G_sparse = qic_core.build_B_prime_n(
            k_idx_B_type, P_prime_for_B_build_sparse, N_local
        )
        if G_sparse is None: raise ValueError("Failed G_typeB construction with purified P'.")
        G_dense = G_sparse.toarray()
        identity_8x8 = np.eye(2**N_local, dtype=complex)
        is_unitary_by_numpy = np.allclose(G_dense @ G_dense.conj().T, identity_8x8, atol=1e-8)
        print(f"G_typeB (from purified P') is unitary (by np.allclose)? {is_unitary_by_numpy}. Dimensions: {G_dense.shape}")
        if not is_unitary_by_numpy:
            diff_norm = np.linalg.norm((G_dense @ G_dense.conj().T) - identity_8x8)
            print(f"  Norm of (G G_dag - I) for G_typeB: {diff_norm:.3e}")
            print("  WARNING: G_typeB computed as NOT unitary by numpy check.")
            return None # Crucial to return None if not unitary by our own check
        return G_dense
    except Exception as e:
        print(f"ERROR during G_typeB computation: {e}")
        traceback.print_exc()
        return None

def verify_operator_equivalence(N, k_idx, G_A, G_B):
    """
    Verifies B'_k for a given N and k_idx.
    Compares original full matrix method with efficient local gate method.
    """
    op_name = f"B'_{k_idx}"
    print(f"\n--- Verifying {op_name} for N_QUBITS = {N} ---")

    Operator_original = None
    Bk_original_matrix = None # To store the matrix for Frobenius norm calculation

    # 1. Original Method: Construct full B'_k matrix
    try:
        print(f"  Constructing {op_name} (N={N}) using original full matrix method...")
        qic_strings_N, qic_vectors_N = qic_core.get_qic_basis(N)
        if not qic_strings_N: raise ValueError("Failed QIC basis for original method.")
        
        V_N_iso = qic_core.construct_isometry_V(qic_vectors_N)
        if V_N_iso is None: raise ValueError("Failed V_iso for original method.")
        V_N_csc = V_N_iso.tocsc()

        Pk_anyon_orig = qic_core.get_kauffman_Pn_anyon_general(
            N, k_idx, qic_strings_N, delta=qic_core.PHI
        )
        if Pk_anyon_orig is None: raise ValueError("Failed P_anyon for original method.")

        Pk_prime_orig_sparse = qic_core.build_P_prime_n(
            k_idx, N, V_N_csc, Pk_anyon_orig
        )
        if Pk_prime_orig_sparse is None: raise ValueError("Failed P'_prime for original method.")

        Bk_original_sparse = qic_core.build_B_prime_n(
            k_idx, Pk_prime_orig_sparse, N
        )
        if Bk_original_sparse is None: raise ValueError("Failed B'_original for original method.")
        Bk_original_matrix = Bk_original_sparse.toarray() # Keep the numpy array
        Operator_original = Operator(Bk_original_matrix)
        
        # Check if original matrix is unitary (important for meaningful fidelity)
        if not Operator_original.is_unitary(atol=EQUIV_ATOL): 
             print(f"  WARNING: Original B'_{k_idx} matrix (from full construction) is NOT unitary by Qiskit check! Fidelity/distance might be misleading.")
        print(f"  Original {op_name} matrix constructed.")
    except Exception as e:
        print(f"  ERROR constructing B'_{k_idx} with original method: {e}")
        traceback.print_exc()
        return False

    Operator_efficient = None
    Bk_efficient_matrix = None # To store the matrix for Frobenius norm calculation

    # 2. Efficient Method: Construct circuit with local G gate
    try:
        print(f"  Constructing {op_name} (N={N}) using efficient local gate method...")
        current_G_local = None
        target_qubits = []

        if k_idx == 0: 
            current_G_local = G_A
            target_qubits = [0, 1, 2]
        elif k_idx == N - 2: 
            current_G_local = G_B
            target_qubits = [N - 3, N - 2, N - 1] 
        else: # Middle operators: B'_1 to B'_{N-3}
            current_G_local = G_A # Using G_A as a proxy for G_middle
            target_qubits = [k_idx, k_idx + 1, k_idx + 2]
        
        if current_G_local is None: 
            print(f"  ERROR: Local gate (G_A or G_B) is None for {op_name}. Efficient construction failed.")
            return False # Cannot proceed if G_A or G_B was not computed

        qc_efficient = QuantumCircuit(N, name=f"{op_name}_eff")
        qc_efficient.unitary(current_G_local, target_qubits, label=f"G({k_idx})")
        
        Operator_efficient = Operator(qc_efficient)
        Bk_efficient_matrix = Operator_efficient.data # Get the numpy array for efficient operator
        print(f"  Efficient {op_name} circuit operator constructed.")

    except Exception as e:
        print(f"  ERROR constructing B'_{k_idx} with efficient method: {e}")
        traceback.print_exc()
        return False

    # 3. Compare the two operators
    try:
        print(f"  Comparing operators for {op_name} (N={N})...")
        are_equiv = Operator_original.equiv(Operator_efficient, rtol=EQUIV_RTOL, atol=EQUIV_ATOL)
        
        if are_equiv:
            print(f"  SUCCESS: {op_name} (N={N}) - Efficient method MATCHES original method (by Operator.equiv).")
        else:
            print(f"  FAILURE: {op_name} (N={N}) - Efficient method DOES NOT MATCH original method (by Operator.equiv).")
            
            # Calculate and print distance metrics
            U_matrix = Bk_original_matrix # This is Operator_original.data
            V_matrix = Bk_efficient_matrix # This is Operator_efficient.data
            dim_d_squared = (2**N)**2

            # Average gate fidelity: F_avg = ( |Tr(U_dag V)|^2 + d ) / (d^2 + d), where d=2^N
            # Simpler version: F_avg = |Tr(U_dag V)|^2 / d^2 (often used for unitary comparison)
            # Let's use the simpler |Tr(U_dag V)|^2 / d^2 as it's common for comparing unitaries.
            # Qiskit's process_fidelity uses ( |Tr(U_dag V)|^2 / d^2 ) if target is unitary.
            # Or average_gate_fidelity from qiskit.quantum_info
            
        
            # average_gate_fidelity requires target to be a QuantumChannel or UnitaryGate
            # For two statevectors: state_fidelity(state_a, state_b)
            # For two operators: process_fidelity(Operator(V_matrix), Operator(U_matrix)) might be more robust
            # or calculate manually.
            
            # Manual calculation for average gate fidelity between two unitaries U and V of dimension D:
            # F_avg = (D + |Tr(U^\dagger V)|^2) / (D * (D+1)) -- this is for channels.
            # For unitaries, often F = | <psi_U | psi_V> |^2 where states are related to Choi matrices.
            # Or simply: F_avg = |Tr(U^\dagger V)|^2 / D^2
            # Let's use the simpler |Tr(U^\dagger V)/D|^2 which qiskit calls average_gate_fidelity(A,B)
            # when A and B are unitaries.
            
            # Ensure matrices are complex type for trace and correct dimensions
            if U_matrix is not None and V_matrix is not None and U_matrix.shape == V_matrix.shape:
                op_U = Operator(U_matrix)
                op_V = Operator(V_matrix)

                # Check if both are unitary before calculating fidelity in a specific way
                # (Original one is checked above, efficient one from circuit is unitary by construction)
                if op_U.is_unitary(atol=EQUIV_ATOL) and op_V.is_unitary(atol=EQUIV_ATOL):
                        # Using a definition of fidelity more aligned with average gate fidelity
                        # F = |Tr(U_target^dag U_actual)|^2 / dim^2 when both are unitary
                        # (or use Qiskit's average_gate_fidelity if it applies directly to Operators)
                    trace_val = np.trace(np.conj(U_matrix).T @ V_matrix)
                    avg_fidelity = (np.abs(trace_val)**2) / dim_d_squared
                    distance_metric = np.sqrt(np.abs(1 - avg_fidelity)) # abs for potential small neg due to precision
                    print(f"      Average gate fidelity ( |Tr(UdagV)|^2/d^2 ): {avg_fidelity:.8f}")
                    print(f"      Distance metric (sqrt(1-F_avg)): {distance_metric:.8f}")
                else:
                    print("      Skipping fidelity/distance: one or both operators not reliably unitary.")

                frobenius_diff = np.linalg.norm(U_matrix - V_matrix, 'fro')
                print(f"      Frobenius norm of difference (U_orig - V_eff): {frobenius_diff:.3e}")

            else:
                print("      Could not compute distance/fidelity: matrices unavailable or mismatched.")

        return are_equiv
    except Exception as e:
        print(f"  ERROR during operator comparison for {op_name} (N={N}): {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    overall_start_time = time.time()
    print("Starting B'_k operator equivalence verification...")

    # --- Compute G_typeA and G_typeB once ---
    G_A_unitary = compute_G_typeA_unitary()
    if G_A_unitary is None:
        print("Critical error: Failed to compute G_typeA_unitary. Exiting verification.")
        exit()

    # Define P1_anyon_N3_data using symbolic constants for precision
    PHI = qic_core.PHI 
    val_diag_1 = 1.0
    val_diag_2 = 1.0 / (PHI**2) 
    val_diag_3 = 1.0 / PHI      
    val_offdiag = 1.0 / (PHI * np.sqrt(PHI)) 

    P1_anyon_N3_data = np.array([
        [val_diag_1, 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , val_diag_2, 0.        , val_offdiag],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , val_offdiag, 0.        , val_diag_3]
    ], dtype=complex)
    
    G_B_unitary = compute_G_typeB_unitary(P1_anyon_N3_data)
    if G_B_unitary is None:
        print("Critical error: Failed to compute G_typeB_unitary. Exiting verification.")
        exit()
    
    all_tests_passed = True
    for N_val in N_VALUES_TO_VERIFY:
        if N_val < 3: # Current G_A and G_B are 3-qubit gates
            print(f"\nSkipping N={N_val}: Verification script requires N>=3 for current 3-qubit local gates.")
            continue
        
        print(f"\n===== Verifying for N_QUBITS = {N_val} =====")
        num_operators = N_val - 1 # B'_0 to B'_{N-2}
        
        for k_val in range(num_operators):
            success = verify_operator_equivalence(N_val, k_val, G_A_unitary, G_B_unitary)
            if not success:
                all_tests_passed = False
    
    print("\n--- Verification Sweep Finished ---")
    if all_tests_passed:
        print("All tested operators are equivalent between methods!")
    else:
        print("Some operators FAILED the equivalence check.")
        
    total_duration = time.time() - overall_start_time
    print(f"Total verification time: {total_duration:.3f} seconds")