# find_optimal_local_approximator.py
# For each N and k, finds the best 3-qubit local unitary G_tilde(N,k)
# such that G_tilde(N,k) @ I approximates B'_k(N)_original,
# stores G_tilde, and reports Frobenius norms for comparison.

import numpy as np
import time
import traceback
import csv
import os
import sys

# --- Ensure qic_core is importable ---
try:
    import qic_core
except ImportError:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(os.path.dirname(current_dir), 'src') 
        if src_dir not in sys.path:
           sys.path.insert(0, src_dir)
        import qic_core
        print("Successfully imported qic_core (path might have been adjusted).")
    except ImportError as e:
        print(f"ERROR: Failed to import qic_core.py: {e}")
        print("       Please ensure qic_core.py is in an importable location (e.g., src/).")
        exit()

# Import Qiskit components
if qic_core.QISKIT_AVAILABLE:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from scipy.linalg import polar, eigh # Added eigh for purification
    from scipy.sparse import csc_matrix
else:
    print("ERROR: Qiskit (and Scipy) is required for this script.")
    exit()

# ===== Configuration =====
N_VALUES_TO_ANALYZE = [3, 4, 5, 6, 7, 8, 9, 10] 
SAVE_G_TILDE_MATRICES = True 
RESULTS_DIR = 'data/optimal_local_approximators' 
CSV_RESULTS_FILE = os.path.join(RESULTS_DIR, 'optimal_approx_summary.csv')
# =======================

PHI = getattr(qic_core, 'PHI', (1 + np.sqrt(5)) / 2)
stored_G_tildes = {}

def get_original_Bk_matrix_bk(N, k_idx):
    op_name = f"B'_{k_idx}(N={N})"
    try:
        # --- These lines were missing/incomplete in the user-provided version ---
        qic_strings_N, qic_vectors_N = qic_core.get_qic_basis(N)
        if not qic_strings_N: raise ValueError(f"Failed QIC basis for {op_name}.")
        
        V_N_iso = qic_core.construct_isometry_V(qic_vectors_N)
        if V_N_iso is None: raise ValueError(f"Failed V_iso for {op_name}.")
        V_N_csc = V_N_iso.tocsc()

        Pk_anyon_orig = qic_core.get_kauffman_Pn_anyon_general(
            N, k_idx, qic_strings_N, delta=PHI
        )
        #print(f"\nP_{k_idx}^anyon:\n", Pk_anyon_orig.toarray())
        if Pk_anyon_orig is None: raise ValueError(f"Failed P_anyon for {op_name}.")
        # --- End of missing lines ---

        Pk_prime_orig_sparse_initial = qic_core.build_P_prime_n(
            k_idx, N, V_N_csc, Pk_anyon_orig
        )
        if Pk_prime_orig_sparse_initial is None: raise ValueError("Failed P'_prime initial.")
        
        P_prime_initial_dense = Pk_prime_orig_sparse_initial.toarray()
        P_prime_to_purify = (P_prime_initial_dense + P_prime_initial_dense.conj().T) / 2.0
        
        eigvals, eigvecs = eigh(P_prime_to_purify) # Using scipy.linalg.eigh
        P_prime_purified_dense = np.zeros_like(P_prime_to_purify, dtype=complex)
        eigenvalue_one_threshold = 0.5 
        num_selected_eigvals = 0
        for i in range(len(eigvals)):
            if eigvals[i] > eigenvalue_one_threshold:
                num_selected_eigvals +=1
                vec = eigvecs[:, i].reshape(-1, 1)
                P_prime_purified_dense += vec @ vec.conj().T
        
        # Optional: print a message if purification actually changed P_prime significantly
        if num_selected_eigvals < P_prime_to_purify.shape[0] and not np.allclose(P_prime_to_purify, P_prime_purified_dense):
             print(f"    Note: P'_orig for {op_name} was purified (selected {num_selected_eigvals} eigenvalues).")
        Pk_prime_orig_sparse_purified = csc_matrix(P_prime_purified_dense)

        Bk_original_sparse = qic_core.build_B_prime_n(
            k_idx, Pk_prime_orig_sparse_purified, N 
        )
        if Bk_original_sparse is None: raise ValueError("Failed B'_original after purification.")
        
        Bk_original_dense = Bk_original_sparse.toarray()
        # Inside get_original_Bk_matrix, after Bk_original_dense is computed:
        op_check = Operator(Bk_original_dense)
        if not op_check.is_unitary(atol=1e-8): # Use a stricter atol if needed, e.g., 1e-9
            u_udag_diff_norm = np.linalg.norm(Bk_original_dense @ Bk_original_dense.conj().T - np.eye(2**N), 'fro')
            print(f"    WARNING: {op_name} from get_original_Bk_matrix is NOT unitary! ||UUdag-I||_F = {u_udag_diff_norm:.3e}")
        # else:
        #     print(f"    {op_name} from get_original_Bk_matrix IS unitary (atol=1e-8).")

        if not Operator(Bk_original_dense).is_unitary(atol=1e-7): # Stricter check
            norm_diff_unitary = np.linalg.norm((Bk_original_dense @ Bk_original_dense.conj().T) - np.eye(2**N), 'fro')
            print(f"    WARNING: Purified B'_{k_idx}(N={N}) still not quite unitary by Qiskit check! ||UUdag-I||_F = {norm_diff_unitary:.2e}")
        return Bk_original_dense
    except Exception as e:
        print(f"    ERROR constructing original {op_name}: {e}")
        # traceback.print_exc() # Uncomment for full traceback during debugging
        return None

def get_original_Bk_matrix(N, k_idx):
    op_name = f"B'_{k_idx}(N={N})"
    try:
        # Get QIC basis
        qic_strings_N, qic_vectors_N = qic_core.get_qic_basis(N)
        if not qic_strings_N: raise ValueError(f"Failed QIC basis for {op_name}.")

        V_N_iso = qic_core.construct_isometry_V(qic_vectors_N)
        if V_N_iso is None: raise ValueError(f"Failed V_iso for {op_name}.")
        V_N_csc = V_N_iso.tocsc()

        # Get P_k^anyon
        Pk_anyon_orig = qic_core.get_kauffman_Pn_anyon_general(N, k_idx, qic_strings_N, delta=PHI)
        if Pk_anyon_orig is None: raise ValueError(f"Failed P_anyon for {op_name}.")
        #print(f"\nP_{k_idx}^anyon(N={N}):\n", Pk_anyon_orig.toarray())  # PRINT HERE

        # Build and purify P'_prime
        Pk_prime_orig_sparse_initial = qic_core.build_P_prime_n(k_idx, N, V_N_csc, Pk_anyon_orig)
        if Pk_prime_orig_sparse_initial is None: raise ValueError("Failed P'_prime initial.")

        P_prime_initial_dense = Pk_prime_orig_sparse_initial.toarray()
        P_prime_to_purify = (P_prime_initial_dense + P_prime_initial_dense.conj().T) / 2.0

        # Purification (project to eigenvalue ≈ 1 eigenspace)
        eigvals, eigvecs = eigh(P_prime_to_purify)
        P_prime_purified_dense = np.zeros_like(P_prime_to_purify, dtype=complex)
        eigenvalue_one_threshold = 0.5
        for i in range(len(eigvals)):
            if eigvals[i] > eigenvalue_one_threshold:
                vec = eigvecs[:, i].reshape(-1, 1)
                P_prime_purified_dense += vec @ vec.conj().T
        Pk_prime_orig_sparse_purified = csc_matrix(P_prime_purified_dense)

        # Build B'_k from purified P'
        Bk_original_sparse = qic_core.build_B_prime_n(k_idx, Pk_prime_orig_sparse_purified, N)
        if Bk_original_sparse is None: raise ValueError("Failed B'_original after purification.")
        Bk_original_dense = Bk_original_sparse.toarray()

        # Unitarity check (Frobenius norm)
        UUdag = Bk_original_dense @ Bk_original_dense.conj().T
        eye = np.eye(2**N)
        norm_diff = np.linalg.norm(UUdag - eye, 'fro')
        print(f"  || B'_orig @ B'_orig^dag - I ||_F = {norm_diff:.6e}")
        # if N == 3 and k_idx == 0:
        #     print("\n--- DEBUG: Pk_anyon_orig (5x5) ---")
        #     print(Pk_anyon_orig.toarray())
        #     # After purification of P'
        #     print("\n--- DEBUG: Pk_prime_orig_sparse_purified (8x8) ---")
        #     print(Pk_prime_orig_sparse_purified.toarray())
        #     # Rank: count number of eigenvalues > threshold (e.g., 0.5)
        #     eigvals = np.linalg.eigvalsh(Pk_prime_orig_sparse_purified.toarray())
        #     rank = np.sum(eigvals > 0.5)
        #     print(f"Rank of purified P'_k: {rank} (eigvals > 0.5: {eigvals[eigvals > 0.5]})")
        #     # After B'_original construction
        #     print("\n--- DEBUG: Bk_original_dense (8x8) ---")
        #     print(Bk_original_dense)


        # Qiskit-style unitarity check
        if not Operator(Bk_original_dense).is_unitary(atol=1e-8):
            print(f"    WARNING: {op_name} is NOT unitary! (||UUdag-I||_F = {norm_diff:.6e})")
        else:
            print(f"    {op_name} IS unitary within atol=1e-8.")

        return Bk_original_dense

    except Exception as e:
        print(f"    ERROR constructing original {op_name}: {e}")
        return None


def extract_averaged_local_block(U_original_Nk, N, k_active_start_idx):
    N_active = 3
    if N < N_active or k_active_start_idx + N_active > N:
        print(f"Error in extract_averaged_local_block: N={N} or k_active_start_idx={k_active_start_idx} invalid for 3-qubit block.")
        return None
    
    N_env = N - N_active
    d_A = 2**N_active
    d_E = 2**N_env if N_env >= 0 else 1 
    
    G_avg = np.zeros((d_A, d_A), dtype=complex)
    
    active_qubit_indices = list(range(k_active_start_idx, k_active_start_idx + N_active))
    env_qubit_indices = [i for i in range(N) if i not in active_qubit_indices]

    for alpha_local_idx in range(d_A): 
        alpha_local_str = format(alpha_local_idx, f'0{N_active}b')
        for beta_local_idx in range(d_A): 
            beta_local_str = format(beta_local_idx, f'0{N_active}b')
            sum_val = 0.0j
            
            num_env_states = d_E 
            if N_env == 0: 
                 gamma_env_str_list = [""] 
            else:
                 gamma_env_str_list = [format(idx, f'0{N_env}b') for idx in range(num_env_states)]

            for gamma_env_str in gamma_env_str_list:
                input_N_q_list = ['0'] * N
                output_N_q_list = ['0'] * N
                
                for i_act_map_idx in range(N_active):
                    global_q_idx = active_qubit_indices[i_act_map_idx]
                    input_N_q_list[global_q_idx] = beta_local_str[i_act_map_idx]
                    output_N_q_list[global_q_idx] = alpha_local_str[i_act_map_idx]
                
                for i_env_map_idx in range(N_env): 
                    global_q_idx = env_qubit_indices[i_env_map_idx]
                    input_N_q_list[global_q_idx] = gamma_env_str[i_env_map_idx]
                    output_N_q_list[global_q_idx] = gamma_env_str[i_env_map_idx]
                
                idx_in_global = int("".join(input_N_q_list)[::-1], 2)
                idx_out_global = int("".join(output_N_q_list)[::-1], 2)
                sum_val += U_original_Nk[idx_out_global, idx_in_global]
            
            G_avg[alpha_local_idx, beta_local_idx] = sum_val / num_env_states 
            
    return G_avg

def get_closest_unitary_bk(matrix_M):
    if matrix_M is None: return None
    try:
        U_polar, _ = polar(matrix_M)
        return U_polar
    except Exception as e:
        print(f"    Error during polar decomposition: {e}")
        return None

def get_closest_unitary(matrix_M, N_val_debug=None, k_op_idx_debug=None, original_matrix_if_known_unitary=None):
    if matrix_M is None: return None
    try:
        # If we already know matrix_M is perfectly unitary from a prior reliable check,
        # its SVD should ideally yield itself back via V @ Wh.
        # The polar decomposition should also yield itself.
        # This direct return can bypass potential numerical quirks if matrix_M IS unitary.
        if original_matrix_if_known_unitary is not None: # Pass the original if it was checked
             # Check if matrix_M (which is G_avg) is indeed the original and if original was unitary
             identity_check_matrix = np.eye(matrix_M.shape[0], dtype=complex)
             if np.allclose(matrix_M @ matrix_M.conj().T, identity_check_matrix, atol=1e-14): # Very strict check
                 print(f"    DEBUG get_closest_unitary: Input matrix_M for N={N_val_debug}, k={k_op_idx_debug} is already highly unitary. Returning as is.")
                 return matrix_M.copy() # Return a copy

        # Proceed with SVD-based closest unitary if not taking the shortcut
        V_svd, S_svd, Wh_svd = np.linalg.svd(matrix_M)
        
        if N_val_debug is not None and k_op_idx_debug is not None: # Print SVD for the matrix it received
            if N_val_debug == 3 and k_op_idx_debug == 0 : # Focus on the problematic case
                 print(f"    --- DEBUG INSIDE get_closest_unitary (N={N_val_debug}, k={k_op_idx_debug}) ---")
                 print(f"    Singular values of input matrix_M (G_avg):")
                 print(S_svd)
                 print(f"    Norm of (G_avg @ G_avg_dag - I): {np.linalg.norm(matrix_M @ matrix_M.conj().T - np.eye(matrix_M.shape[0]), 'fro'):.4e}")

        G_tilde = V_svd @ Wh_svd
        return G_tilde
    except Exception as e:
        print(f"    Error during SVD-based closest unitary computation: {e}")
        traceback.print_exc()
        return None
    
def construct_global_approx_operator_matrix(G_local_8x8, N, k_active_start_idx):
    if k_active_start_idx < 0 or k_active_start_idx + 3 > N :
        print(f"Error: Cannot place 3-qubit gate starting at {k_active_start_idx} in N={N} system for tensor construction.")
        return None
    try:
        qc = QuantumCircuit(N)
        target_qubits = [k_active_start_idx, k_active_start_idx + 1, k_active_start_idx + 2]
        qc.unitary(G_local_8x8, target_qubits)
        return Operator(qc).data
    except Exception as e:
        print(f"    Error during construction of global approx operator matrix: {e}")
        return None

# --- Main execution block ---
if __name__ == "__main__":
    overall_start_time = time.time()
    print("Starting Optimal Local Approximator Search...")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created results directory: {RESULTS_DIR}")
    
    csv_header = ['N', 'k_op_idx', 'k_active_start', 'status', 
                  'min_frobenius_delta', 'norm_G_vs_G(N-1,k_op)', 'norm_G_vs_G(N,k_op-1)']
    if not os.path.exists(CSV_RESULTS_FILE) or os.path.getsize(CSV_RESULTS_FILE) == 0:
        with open(CSV_RESULTS_FILE, 'w', newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(csv_header)
        print(f"Created CSV results file: {CSV_RESULTS_FILE}")
    else:
        print(f"Appending to existing CSV results file: {CSV_RESULTS_FILE}")

    results_summary_list = [] # Changed variable name for clarity

    for N_val in N_VALUES_TO_ANALYZE:
        if N_val < 3: 
            print(f"\nSkipping N={N_val}: Method requires N>=3 for 3-qubit local blocks.")
            continue
        
        print(f"\n===== Analyzing for N_QUBITS = {N_val} =====")
        num_b_prime_ops = N_val - 1 
        
        for k_op_idx in range(num_b_prime_ops): 
            op_name = f"B'_{k_op_idx}(N={N_val})"
            print(f"\n--- Processing {op_name} ---")

            k_active_start = k_op_idx 
            if k_active_start + 2 >= N_val: 
                k_active_start = N_val - 3 
            
            current_result_dict = {'N': N_val, 'k_op_idx': k_op_idx, 'k_active_start': k_active_start}

            print(f"  1. Computing original {op_name} matrix (with internal P' purification)...")
            Bk_orig_N_matrix = get_original_Bk_matrix(N_val, k_op_idx)
            if Bk_orig_N_matrix is None:
                current_result_dict.update({'status': 'Error Original Construction', 'min_frobenius_delta': 'N/A', 'norm_G_vs_G(N-1,k_op)': 'N/A', 'norm_G_vs_G(N,k_op-1)': 'N/A'})
                results_summary_list.append(current_result_dict)
                print(f"  Skipping {op_name} due to error in original matrix construction.")
                continue
            
            # Verify unitarity of the (now hopefully purified) original matrix
            op_orig_check = Operator(Bk_orig_N_matrix)
            if not op_orig_check.is_unitary(atol=1e-7): # Use a reasonably strict tolerance
                norm_diff_unitary = np.linalg.norm((Bk_orig_N_matrix @ Bk_orig_N_matrix.conj().T) - np.eye(2**N_val), 'fro')
                print(f"  WARNING: Original {op_name} matrix is NOT strictly unitary even after P' purification! ||UUdag-I||_F = {norm_diff_unitary:.2e}")
            
            print(f"  2. Extracting averaged local block G_avg for {op_name} from active qubits [{k_active_start}, {k_active_start+1}, {k_active_start+2}]...")
            G_avg_Nk = extract_averaged_local_block(Bk_orig_N_matrix, N_val, k_active_start)
            if G_avg_Nk is None:
                current_result_dict.update({'status': 'Error G_avg Extraction', 'min_frobenius_delta': 'N/A', 'norm_G_vs_G(N-1,k_op)': 'N/A', 'norm_G_vs_G(N,k_op-1)': 'N/A'})
                results_summary_list.append(current_result_dict)
                print(f"  Skipping {op_name} due to error in G_avg extraction.")
                continue

            print(f"  3. Finding closest unitary G_tilde to G_avg for {op_name}...")
            #G_tilde_Nk = get_closest_unitary(G_avg_Nk)
            #G_tilde_Nk = get_closest_unitary(G_avg_Nk, N_val, k_op_idx) # NEW CALL
            
            if N_val == 3: # For N=3, G_avg is Bk_orig_N_matrix
                G_avg_Nk = Bk_orig_N_matrix 
                # Pass Bk_orig_N_matrix to the debug parameter
                G_tilde_Nk = get_closest_unitary(G_avg_Nk, N_val, k_op_idx, original_matrix_if_known_unitary=Bk_orig_N_matrix)
            else: # N > 3
                G_avg_Nk = extract_averaged_local_block(Bk_orig_N_matrix, N_val, k_active_start)
                if G_avg_Nk is None: # Check if extraction failed
                    # ... handle error ...
                    continue
                G_tilde_Nk = get_closest_unitary(G_avg_Nk, N_val, k_op_idx)

            if G_tilde_Nk is None:
                current_result_dict.update({'status': 'Error Closest Unitary', 'min_frobenius_delta': 'N/A', 'norm_G_vs_G(N-1,k_op)': 'N/A', 'norm_G_vs_G(N,k_op-1)': 'N/A'})
                results_summary_list.append(current_result_dict)
                print(f"  Skipping {op_name} due to error in finding closest unitary.")
                continue
            
            #if N_val == 3 and k_op_idx == 100: #k_op_idx == 0
            #    print("\n--- DEBUG: Main Loop ---")
            #    print("Bk_orig_N_matrix:")
            #    print(Bk_orig_N_matrix)
            #    print("G_avg_Nk:")
            #    print(G_avg_Nk)
            #    print("G_tilde_Nk:")
            #    print(G_tilde_Nk)
            #    frob_norm_unitary = np.linalg.norm(Bk_orig_N_matrix @ Bk_orig_N_matrix.conj().T - np.eye(8), 'fro')
            #    print(f"||Bk_orig_N_matrix @ Bk_orig_N_matrix^dag - I||_F = {frob_norm_unitary:.4e}")

            # debug:
            #if N_val == 3 and k_op_idx == 0:
            #    if Bk_orig_N_matrix is not None and G_tilde_Nk is not None:
            #        direct_diff_norm = np.linalg.norm(Bk_orig_N_matrix - G_tilde_Nk, 'fro')
            #        print(f"    DEBUG (N=3,k=0): || Bk_orig - G_tilde ||_F = {direct_diff_norm:.4e}")
            #        print(f"\nBorig:\n", Bk_orig_N_matrix)
            #        print(f"\nBtilde:\n", G_tilde_Nk)


            if not Operator(G_tilde_Nk).is_unitary(atol=1e-7):
                 print(f"  WARNING: G_tilde_Nk for {op_name} is NOT strictly unitary after polar decomp! Norm of G G_dag - I: {np.linalg.norm((G_tilde_Nk @ G_tilde_Nk.conj().T) - np.eye(8)):.2e}")

            # Inside the k_op_idx loop, after Bk_orig_N_matrix is confirmed unitary
            #if N_val == 3 and k_op_idx == 0 and Bk_orig_N_matrix is not None:
            #    print("\n--- SVD Analysis of Bk_orig_N_matrix (N=3, k=0) ---")
            #    try:
            #        V_svd, S_svd, Wh_svd = np.linalg.svd(Bk_orig_N_matrix)
            #        print("Singular values (should all be ~1.0):")
            #        print(S_svd)
            #        print("Singular values - 1.0:")
            #        print(S_svd - 1.0)

            #        U_reconstructed_from_svd = V_svd @ Wh_svd
            #        norm_diff_orig_vs_svd_unitary = np.linalg.norm(Bk_orig_N_matrix - U_reconstructed_from_svd, 'fro')
            #        norm_diff_tilde_vs_svd_unitary = np.linalg.norm(G_tilde_Nk - U_reconstructed_from_svd, 'fro')
                    
            #        print(f"|| Bk_orig - (V @ Wh) ||_F = {norm_diff_orig_vs_svd_unitary:.4e}")
            #        print(f"|| G_tilde (from polar) - (V @ Wh) ||_F = {norm_diff_tilde_vs_svd_unitary:.4e}")

            #    except Exception as e_svd:
            #        print(f"Error during SVD analysis: {e_svd}")

            stored_G_tildes[(N_val, k_op_idx)] = G_tilde_Nk
            if SAVE_G_TILDE_MATRICES:
                g_tilde_filename = f"G_tilde_N{N_val}_kop{k_op_idx}_act{k_active_start}.npy"
                np.save(os.path.join(RESULTS_DIR, g_tilde_filename), G_tilde_Nk)

            print(f"  4. Constructing B_approx = G_tilde_Nk @ I for {op_name}...")
            Bk_approx_Nk_matrix = construct_global_approx_operator_matrix(G_tilde_Nk, N_val, k_active_start)
            if Bk_approx_Nk_matrix is None:
                current_result_dict.update({'status': 'Error Tensor Construction', 'min_frobenius_delta': 'N/A', 'norm_G_vs_G(N-1,k_op)': 'N/A', 'norm_G_vs_G(N,k_op-1)': 'N/A'})
                results_summary_list.append(current_result_dict)
                print(f"  Skipping {op_name} due to error constructing tensor product operator.")
                continue

            minimized_delta = np.linalg.norm(Bk_orig_N_matrix - Bk_approx_Nk_matrix, 'fro')
            print(f"  SUCCESS: Minimized Frobenius norm || B_orig - (G_tilde @ I) || for {op_name}: {minimized_delta:.4e}")
            current_result_dict['min_frobenius_delta'] = f"{minimized_delta:.4e}"
            current_result_dict['status'] = 'Success'

            norm_G_vs_prev_N_val = 'N/A'
            if N_val > N_VALUES_TO_ANALYZE[0] : 
                if (N_val - 1, k_op_idx) in stored_G_tildes:
                    G_prev_N_same_k = stored_G_tildes[(N_val - 1, k_op_idx)]
                    norm_G_vs_prev_N_val = f"{np.linalg.norm(G_tilde_Nk - G_prev_N_same_k, 'fro'):.4e}"
            current_result_dict['norm_G_vs_G(N-1,k_op)'] = norm_G_vs_prev_N_val

            norm_G_vs_prev_k_val = 'N/A'
            if k_op_idx > 0:
                if (N_val, k_op_idx - 1) in stored_G_tildes:
                    G_prev_k_same_N = stored_G_tildes[(N_val, k_op_idx - 1)]
                    norm_G_vs_prev_k_val = f"{np.linalg.norm(G_tilde_Nk - G_prev_k_same_N, 'fro'):.4e}"
            current_result_dict['norm_G_vs_G(N,k_op-1)'] = norm_G_vs_prev_k_val
            
            results_summary_list.append(current_result_dict)

    print("\n--- Optimal Local Approximator Analysis Finished ---")
    with open(CSV_RESULTS_FILE, 'a', newline='') as f_csv: 
        writer = csv.DictWriter(f_csv, fieldnames=csv_header)
        for row_dict in results_summary_list: # Write new results collected in this run
            writer.writerow(row_dict)
    print(f"All results for this run appended to {CSV_RESULTS_FILE}")
    
    print("\nResults Summary (from this run):")
    print(f"{'N':<3} | {'k_op':<4} | {'k_act':<5} | {'Status':<28} | {'Min Frob Delta':<18} | {'GvG(N-1,k)':<15} | {'GvG(N,k-1)':<15}")
    print("-" * 100)
    for res in results_summary_list:
        print(f"{res['N']:<3} | {res['k_op_idx']:<4} | {res['k_active_start']:<5} | {res['status']:<28} | {res.get('min_frobenius_delta', 'N/A'):<18} | {res.get('norm_G_vs_G(N-1,k_op)', 'N/A'):<15} | {res.get('norm_G_vs_G(N,k_op-1)', 'N/A'):<15}")
        
    total_duration = time.time() - overall_start_time
    print(f"\nTotal script execution time: {total_duration:.2f} seconds.")