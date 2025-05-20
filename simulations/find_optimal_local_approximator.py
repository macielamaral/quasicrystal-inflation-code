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

import qic_synthesis_tools as qst

# ===== Configuration =====
N_VALUES_TO_ANALYZE = [3, 4, 5, 6, 7, 8, 9, 10] 
SAVE_G_TILDE_MATRICES = True 
RESULTS_DIR = 'data/optimal_local_approximators' 
CSV_RESULTS_FILE = os.path.join(RESULTS_DIR, 'optimal_approx_summary.csv')
# =======================

PHI = getattr(qic_core, 'PHI', (1 + np.sqrt(5)) / 2)
stored_G_tildes = {}

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
            Bk_orig_N_matrix = qst.get_original_Bk_matrix(N_val, k_op_idx)
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
            G_avg_Nk = qst.extract_averaged_local_block(Bk_orig_N_matrix, N_val, k_active_start)
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
                G_tilde_Nk = qst.get_closest_unitary(G_avg_Nk, N_val, k_op_idx, original_matrix_if_known_unitary=Bk_orig_N_matrix)
            else: # N > 3
                G_avg_Nk = qst.extract_averaged_local_block(Bk_orig_N_matrix, N_val, k_active_start)
                if G_avg_Nk is None: # Check if extraction failed
                    # ... handle error ...
                    continue
                G_tilde_Nk = qst.get_closest_unitary(G_avg_Nk, N_val, k_op_idx)

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
            Bk_approx_Nk_matrix = qst.construct_global_approx_operator_matrix(G_tilde_Nk, N_val, k_active_start)
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