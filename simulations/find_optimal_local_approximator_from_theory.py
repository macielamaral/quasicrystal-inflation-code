# find_optimal_local_approximator_from_theory.py
#
# This script extends find_optimal_local_approximator.py.
# For each N and k, it:
# 1. Computes the original B'_k(N) matrix from qic_core (via qst).
# 2. Derives the best 3-qubit local unitary G_tilde(N,k) that approximates B'_k(N).
# 3. Introduces Quantinuum's theoretical 3-qubit unitary U_Q.
# 4. Compares how well G_tilde(N,k) vs. U_Q (and U_Q_inv) approximate B'_k(N).
# 5. Compares G_tilde(N,k) directly to U_Q and U_Q_inv.
# 6. Stores G_tilde matrices and reports all comparison Frobenius norms to a CSV.

import numpy as np
import time
import traceback
import csv
import os
import sys

# --- Ensure qic_core is importable ---
try:
    import qic_core
    print("Successfully imported qic_core directly.")
except ImportError:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Assuming this script is in 'simulations/' and 'src/' is one level up
        src_dir = os.path.join(os.path.dirname(current_dir), 'src')
        if src_dir not in sys.path:
           sys.path.insert(0, src_dir)
        import qic_core
        print("Successfully imported qic_core (path adjusted from simulations/ to src/).")
    except ImportError as e_path:
        print(f"ERROR: Failed to import qic_core.py even after path adjustment: {e_path}")
        print("       Please ensure qic_core.py is in an importable location (e.g., src/).")
        exit()

# Import Qiskit components (check availability via qic_core)
if qic_core.QISKIT_AVAILABLE:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from scipy.linalg import polar # For closest unitary if qst uses it
    # from scipy.sparse import csc_matrix # Not directly used here, but qst might
else:
    print("ERROR: Qiskit (and Scipy) is required for this script but qic_core reports it's not available.")
    exit()

# --- Ensure qic_synthesis_tools is importable ---
try:
    # Assuming qic_synthesis_tools.py is in the same directory as qic_core.py (e.g. src/)
    # If this script is in 'simulations/', and qic_synthesis_tools.py is in 'src/',
    # the path adjustment for qic_core should also make qic_synthesis_tools findable.
    import qic_synthesis_tools as qst
    print("Successfully imported qic_synthesis_tools.")
except ImportError as e_qst:
    print(f"ERROR: Failed to import qic_synthesis_tools.py: {e_qst}")
    print("       Ensure it's in the same directory as qic_core.py or in the Python path.")
    exit()


# ===== Configuration =====
N_VALUES_TO_ANALYZE = [3, 4, 5] # Reduced for a quicker example run; expand as needed [3, 4, 5, 6, 7, 8, 9, 10]
SAVE_G_TILDE_MATRICES = True
RESULTS_DIR = 'data/optimal_local_approximators_from_theory' # New directory for these results
CSV_RESULTS_FILE = os.path.join(RESULTS_DIR, 'approx_summary_with_theory_gate.csv')
# =======================

# --- Quantinuum's U_Q Gate Definition ---
# Using PHI from qic_core for consistency
PHI_CONST = getattr(qic_core, 'PHI', (1 + np.sqrt(5)) / 2)

def get_quantinuum_3q_unitary_matrix(phi_val=PHI_CONST, alpha_000=2*np.pi/5, non_fib_diag_phase=0.0):
    """
    Constructs the 8x8 numpy matrix for the Quantinuum braid generator U_sigma_i.
    Qubit order for matrix indices: q2, q1, q0 (Qiskit standard)
    |000> = 0, |001> = 1, ..., |111> = 7
    """
    Uq_matrix = np.zeros((8, 8), dtype=complex)
    def bstr_to_idx(s): return int(s[::-1], 2)

    idx_010, idx_011, idx_101, idx_110, idx_111 = (
        bstr_to_idx("010"), bstr_to_idx("011"), bstr_to_idx("101"),
        bstr_to_idx("110"), bstr_to_idx("111")
    )
    Uq_matrix[idx_010, idx_010] = np.exp(-1j * 4 * np.pi / 5)
    Uq_matrix[idx_011, idx_011] = np.exp(1j * 3 * np.pi / 5)
    Uq_matrix[idx_101, idx_101] = (1/phi_val) * np.exp(1j * 4 * np.pi / 5)
    Uq_matrix[idx_110, idx_110] = np.exp(1j * 3 * np.pi / 5)
    Uq_matrix[idx_111, idx_111] = -(1/phi_val)
    val_101_111 = (phi_val**(-0.5)) * np.exp(-1j * 3 * np.pi / 5)
    Uq_matrix[idx_101, idx_111] = val_101_111
    Uq_matrix[idx_111, idx_101] = val_101_111

    idx_000, idx_001, idx_100 = (
        bstr_to_idx("000"), bstr_to_idx("001"), bstr_to_idx("100")
    )
    Uq_matrix[idx_000, idx_000] = np.exp(1j * alpha_000)
    Uq_matrix[idx_001, idx_001] = np.exp(1j * non_fib_diag_phase)
    Uq_matrix[idx_100, idx_100] = np.exp(1j * non_fib_diag_phase)

    if not Operator(Uq_matrix).is_unitary(atol=1e-8):
        print("WARNING: Constructed Quantinuum 3Q unitary U_Q_matrix is NOT unitary!")
    return Uq_matrix

# --- Main execution block ---
if __name__ == "__main__":
    overall_start_time = time.time()
    print("Starting Optimal Local Approximator Search (with Theoretical U_Q Comparison)...")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created results directory: {RESULTS_DIR}")

    # Define U_Q and its inverse once
    U_Q_THEORY_GATE = get_quantinuum_3q_unitary_matrix()
    U_Q_THEORY_GATE_INV = U_Q_THEORY_GATE.conj().T # Assuming U_Q is unitary

    csv_header = [
        'N', 'k_op_idx', 'k_active_start', 'status_Gtilde',
        'delta_Gtilde', # || B_orig - (G_tilde @ I) ||
        'delta_UQ',     # || B_orig - (U_Q @ I) ||
        'delta_UQ_inv', # || B_orig - (U_Q_inv @ I) ||
        'norm_Gtilde_vs_UQ',
        'norm_Gtilde_vs_UQinv',
        'norm_G_vs_G(N-1,k_op)', # From original script
        'norm_G_vs_G(N,k_op-1)'  # From original script
    ]
    if not os.path.exists(CSV_RESULTS_FILE) or os.path.getsize(CSV_RESULTS_FILE) == 0:
        with open(CSV_RESULTS_FILE, 'w', newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(csv_header)
        print(f"Created CSV results file: {CSV_RESULTS_FILE}")
    else:
        print(f"Appending to existing CSV results file: {CSV_RESULTS_FILE}")

    results_this_run = []
    stored_G_tildes_this_run = {} # For GvG(N-1,k) and GvG(N,k-1) comparisons

    for N_val in N_VALUES_TO_ANALYZE:
        if N_val < 3:
            print(f"\nSkipping N={N_val}: Method requires N>=3 for 3-qubit local blocks.")
            continue

        print(f"\n===== Analyzing for N_QUBITS = {N_val} =====")
        num_b_prime_ops = N_val - 1

        for k_op_idx in range(num_b_prime_ops):
            op_name = f"B'_{k_op_idx}(N={N_val})"
            print(f"\n--- Processing {op_name} ---")

            # Determine the starting qubit for the 3-qubit active block
            # This logic ensures the 3-qubit block is always within N_val qubits
            k_active_start = k_op_idx
            if k_active_start + 2 >= N_val: # If k_op_idx is N-2 (rightmost B_k) or N-1 (not possible for B_k)
                k_active_start = N_val - 3  # Place the 3-qubit block at the right end [N-3, N-2, N-1]
            
            current_result_dict = {
                'N': N_val, 'k_op_idx': k_op_idx, 'k_active_start': k_active_start,
                'status_Gtilde': 'Pending', 'delta_Gtilde': 'N/A',
                'delta_UQ': 'N/A', 'delta_UQ_inv': 'N/A',
                'norm_Gtilde_vs_UQ': 'N/A', 'norm_Gtilde_vs_UQinv': 'N/A',
                'norm_G_vs_G(N-1,k_op)': 'N/A', 'norm_G_vs_G(N,k_op-1)': 'N/A'
            }

            print(f"  1. Computing original {op_name} matrix (with internal P' purification)...")
            # Assuming qst.get_original_Bk_matrix returns the 2^N x 2^N numpy array
            Bk_orig_N_matrix = qst.get_original_Bk_matrix(N_val, k_op_idx, verbose=False) # verbose=False for cleaner loop
            if Bk_orig_N_matrix is None:
                current_result_dict['status_Gtilde'] = 'Error Original B_k Construction'
                results_this_run.append(current_result_dict)
                print(f"  Skipping {op_name} due to error in original matrix construction.")
                continue

            # --- Derive G_tilde(N,k) from B'_k(N)_original (your existing method) ---
            print(f"  2. Deriving G_tilde(N,k) for {op_name}...")
            G_tilde_Nk = None
            if N_val == 3: # For N=3, B'_k(N=3) is already 8x8. G_avg is B'_k.
                G_avg_Nk_for_Gtilde = Bk_orig_N_matrix
            else:
                G_avg_Nk_for_Gtilde = qst.extract_averaged_local_block(Bk_orig_N_matrix, N_val, k_active_start)

            if G_avg_Nk_for_Gtilde is None:
                current_result_dict['status_Gtilde'] = 'Error G_avg Extraction for G_tilde'
            else:
                # Pass original_matrix_if_known_unitary if G_avg_Nk_for_Gtilde is Bk_orig_N_matrix and Bk_orig_N_matrix is unitary
                original_for_closest_unitary = Bk_orig_N_matrix if N_val == 3 else None
                G_tilde_Nk = qst.get_closest_unitary(
                    G_avg_Nk_for_Gtilde,
                    N_val_debug=N_val, k_op_idx_debug=k_op_idx, # For qst's internal debug prints
                    original_matrix_if_known_unitary=original_for_closest_unitary
                )
                if G_tilde_Nk is None:
                    current_result_dict['status_Gtilde'] = 'Error Closest Unitary for G_tilde'
                else:
                    current_result_dict['status_Gtilde'] = 'Success G_tilde Derivation'
                    stored_G_tildes_this_run[(N_val, k_op_idx)] = G_tilde_Nk
                    if SAVE_G_TILDE_MATRICES:
                        g_tilde_filename = f"G_tilde_N{N_val}_kop{k_op_idx}_act{k_active_start}.npy"
                        np.save(os.path.join(RESULTS_DIR, g_tilde_filename), G_tilde_Nk)

            # --- Calculate Delta for G_tilde(N,k) ---
            if G_tilde_Nk is not None:
                Bk_approx_Gtilde_matrix = qst.construct_global_approx_operator_matrix(
                    G_tilde_Nk, N_val, k_active_start
                )
                if Bk_approx_Gtilde_matrix is not None:
                    delta_gtilde_val = np.linalg.norm(Bk_orig_N_matrix - Bk_approx_Gtilde_matrix, 'fro')
                    current_result_dict['delta_Gtilde'] = f"{delta_gtilde_val:.6e}"
                    print(f"    || B_orig - (G_tilde @ I) ||_F = {delta_gtilde_val:.4e}")
                else:
                    current_result_dict['delta_Gtilde'] = 'Error G_tilde Tensor Construct'

            # --- Calculate Deltas for U_Q and U_Q_inv ---
            print(f"  3. Comparing B_orig with U_Q and U_Q_inv based approximators...")
            # U_Q
            Bk_approx_UQ_matrix = qst.construct_global_approx_operator_matrix(
                U_Q_THEORY_GATE, N_val, k_active_start
            )
            if Bk_approx_UQ_matrix is not None:
                delta_uq_val = np.linalg.norm(Bk_orig_N_matrix - Bk_approx_UQ_matrix, 'fro')
                current_result_dict['delta_UQ'] = f"{delta_uq_val:.6e}"
                print(f"    || B_orig - (U_Q @ I) ||_F     = {delta_uq_val:.4e}")
            else:
                current_result_dict['delta_UQ'] = 'Error U_Q Tensor Construct'
            # U_Q_inv
            Bk_approx_UQ_inv_matrix = qst.construct_global_approx_operator_matrix(
                U_Q_THEORY_GATE_INV, N_val, k_active_start
            )
            if Bk_approx_UQ_inv_matrix is not None:
                delta_uq_inv_val = np.linalg.norm(Bk_orig_N_matrix - Bk_approx_UQ_inv_matrix, 'fro')
                current_result_dict['delta_UQ_inv'] = f"{delta_uq_inv_val:.6e}"
                print(f"    || B_orig - (U_Q_inv @ I) ||_F = {delta_uq_inv_val:.4e}")
            else:
                current_result_dict['delta_UQ_inv'] = 'Error U_Q_inv Tensor Construct'

            # --- Compare G_tilde_Nk directly with U_Q and U_Q_inv ---
            if G_tilde_Nk is not None:
                print(f"  4. Comparing derived G_tilde_Nk with U_Q and U_Q_inv...")
                norm_gtilde_vs_uq = np.linalg.norm(G_tilde_Nk - U_Q_THEORY_GATE, 'fro')
                current_result_dict['norm_Gtilde_vs_UQ'] = f"{norm_gtilde_vs_uq:.6e}"
                print(f"    || G_tilde_Nk - U_Q ||_F     = {norm_gtilde_vs_uq:.4e}")

                norm_gtilde_vs_uqinv = np.linalg.norm(G_tilde_Nk - U_Q_THEORY_GATE_INV, 'fro')
                current_result_dict['norm_Gtilde_vs_UQinv'] = f"{norm_gtilde_vs_uqinv:.6e}"
                print(f"    || G_tilde_Nk - U_Q_inv ||_F = {norm_gtilde_vs_uqinv:.4e}")

            # --- Comparisons from original script (G_tilde vs. previous G_tildes) ---
            if G_tilde_Nk is not None:
                # Compare with G_tilde from N-1, same k_op_idx
                if N_val > N_VALUES_TO_ANALYZE[0] and (N_val - 1, k_op_idx) in stored_G_tildes_this_run:
                    G_prev_N_same_k = stored_G_tildes_this_run[(N_val - 1, k_op_idx)]
                    norm_val = np.linalg.norm(G_tilde_Nk - G_prev_N_same_k, 'fro')
                    current_result_dict['norm_G_vs_G(N-1,k_op)'] = f"{norm_val:.6e}"
                # Compare with G_tilde from N, k_op_idx-1
                if k_op_idx > 0 and (N_val, k_op_idx - 1) in stored_G_tildes_this_run:
                    G_prev_k_same_N = stored_G_tildes_this_run[(N_val, k_op_idx - 1)]
                    norm_val = np.linalg.norm(G_tilde_Nk - G_prev_k_same_N, 'fro')
                    current_result_dict['norm_G_vs_G(N,k_op-1)'] = f"{norm_val:.6e}"
            
            results_this_run.append(current_result_dict)
            # End of k_op_idx loop
        # End of N_val loop

    print("\n--- Optimal Local Approximator Analysis (with Theory Gate) Finished ---")
    # Append all collected results from this run to the CSV file
    try:
        with open(CSV_RESULTS_FILE, 'a', newline='') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=csv_header)
            # If file was empty or just created, and we are appending,
            # ensure header is written if it wasn't (though initial check should handle it)
            # f_csv.seek(0, os.SEEK_END) # Go to end of file
            # if f_csv.tell() == 0: # Check if file is empty
            #     writer.writeheader() # Write header if we are the first to write
            
            for row_dict in results_this_run:
                writer.writerow(row_dict)
        print(f"All results for this run appended to {CSV_RESULTS_FILE}")
    except IOError as e:
        print(f"ERROR: Could not write to CSV file {CSV_RESULTS_FILE}: {e}")


    print("\nResults Summary (from this run):")
    # Print header row directly from csv_header
    header_line = " | ".join([f"{h:<18}" if 'delta' in h.lower() or 'norm' in h.lower() else \
                              f"{h:<35}" if 'status' in h.lower() else \
                              f"{h:<14}" if 'k_active' in h.lower() else \
                              f"{h:<8}" if 'k_op' in h.lower() else \
                              f"{h:<3}" for h in csv_header]) # Adjust widths as needed
    print(header_line)
    print("-" * len(header_line)) # Separator based on header length

    for res_dict in results_this_run:
        # Ensure all keys are present in res_dict, defaulting to 'N/A' if not
        row_to_print_values = []
        for h in csv_header:
            val = res_dict.get(h, 'N/A')
            # Format numerical strings that might be 'N/A'
            if isinstance(val, str) and val != 'N/A':
                try: # Attempt to format as float if it's a number string
                    val_float = float(val)
                    if 'delta' in h.lower() or 'norm' in h.lower():
                         val_str = f"{val_float:<18.6e}" # Scientific notation for these
                    else:
                         val_str = f"{val:<18}" # General case for other potential numbers
                except ValueError:
                    val_str = f"{val:<18}" # If not a float string, format as string
            elif isinstance(val, float): # If already a float
                 if 'delta' in h.lower() or 'norm' in h.lower():
                    val_str = f"{val:<18.6e}"
                 else:
                    val_str = f"{val:<18}"
            else: # For N, k_op_idx, status, etc.
                 val_str = f"{val:<35}" if 'status' in h.lower() else \
                           f"{val:<14}" if 'k_active' in h.lower() else \
                           f"{val:<8}" if 'k_op' in h.lower() else \
                           f"{val:<3}"


            # Apply specific widths for alignment (matching header logic)
            if 'delta' in h.lower() or 'norm' in h.lower():
                formatted_val = f"{str(res_dict.get(h, 'N/A')):<18}"
            elif 'status' in h.lower():
                formatted_val = f"{str(res_dict.get(h, 'N/A')):<35}"
            elif 'k_active' in h.lower():
                formatted_val = f"{str(res_dict.get(h, 'N/A')):<14}"
            elif 'k_op' in h.lower():
                formatted_val = f"{str(res_dict.get(h, 'N/A')):<8}"
            elif h == 'N':
                formatted_val = f"{str(res_dict.get(h, 'N/A')):<3}"
            else:
                formatted_val = f"{str(res_dict.get(h, 'N/A')):<18}" # default
            row_to_print_values.append(formatted_val)
        print(" | ".join(row_to_print_values))

    total_duration = time.time() - overall_start_time
    print(f"\nTotal script execution time: {total_duration:.2f} seconds.")