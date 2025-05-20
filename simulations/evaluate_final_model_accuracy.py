# evaluate_final_model_accuracy.py
# Computes the overall accuracy (fidelity and Frobenius norm difference)
# between B'_k(N)_original and an efficient model constructed from
# PRE-SYNTHESIZED fixed local gate circuits (C_L, C_M, C_R).

import numpy as np
import time
import traceback
import csv
import os
import sys

# --- Ensure qic_synthesis_tools (and implicitly qic_core) is importable ---
try:
    # Assuming qic_synthesis_tools.py contains necessary qic_core imports and PHI
    import qic_synthesis_tools as qst 
    PHI = qst.PHI 
    QISKIT_AVAILABLE = qst.QISKIT_AVAILABLE
except ImportError:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Adjust this path if your 'src' or 'tools' directory is elsewhere
        src_dir = os.path.join(os.path.dirname(current_dir), 'src') 
        if src_dir not in sys.path:
           sys.path.insert(0, src_dir)
        import qic_synthesis_tools as qst
        PHI = qst.PHI
        QISKIT_AVAILABLE = qst.QISKIT_AVAILABLE
        print("Successfully imported qic_synthesis_tools (path might have been adjusted).")
    except ImportError as e_qst:
        print(f"ERROR: Could not import qic_synthesis_tools.py: {e_qst}")
        print("       Ensure qic_synthesis_tools.py is in your Python path or src directory,")
        print("       and it can import/define qic_core related constants.")
        # Define PHI as a fallback if qst or its PHI is not available
        PHI = (1 + np.sqrt(5)) / 2
        QISKIT_AVAILABLE = True # Assume Qiskit is available for the rest of the script
        print(f"WARNING: qic_synthesis_tools.py import failed. Using fallback PHI. Qiskit import will be attempted.")


if not QISKIT_AVAILABLE: 
    print("ERROR: Qiskit is required (as per qic_synthesis_tools or direct check) but not available.")
    exit()

from qiskit import QuantumCircuit, transpile, qpy
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.quantum_info import Operator
# Scipy imports (eigh, polar) are expected to be used within qst.get_original_Bk_matrix if needed

# ===== Configuration =====
# THIS IS WHERE N_QUBITS_LIST IS DEFINED GLOBALLY
N_QUBITS_LIST = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # N values for accuracy evaluation

TARGET_BACKEND_NAME = "ibm_brisbane" # For resource estimation
OPTIMIZATION_LEVEL_FINAL_TRANSPILE = 1 # For transpiling the N-qubit circuit with the composed C_X

# --- Paths to PRE-SYNTHESIZED QPY Circuit Files ---
SYNTHESIZED_CIRCUITS_DIR = "data/synthesized_circuits" 
PATH_C_L_QPY = os.path.join(SYNTHESIZED_CIRCUITS_DIR, "C_L.qpy")
PATH_C_M_QPY = os.path.join(SYNTHESIZED_CIRCUITS_DIR, "C_M.qpy")
PATH_C_R_QPY = os.path.join(SYNTHESIZED_CIRCUITS_DIR, "C_R.qpy")

# --- KNOWN SYNTHESIS FIDELITIES for the loaded C_L, C_M, C_R ---
# Replace with your actual achieved fidelities for the chosen C_L, C_M, C_R circuits
KNOWN_SYNTH_FID_L = 0.99906195 
KNOWN_SYNTH_FID_M = 0.99923507 # Make sure this is the corrected high fidelity value
KNOWN_SYNTH_FID_R = 0.99176594 

# --- Output CSV Configuration ---
RESULTS_DIR_MAIN = 'data'
RESULTS_SUBDIR_ACCURACY = 'final_model_accuracy_eval' 
CSV_ACCURACY_FILE = os.path.join(RESULTS_DIR_MAIN, RESULTS_SUBDIR_ACCURACY, 'overall_model_accuracy_resources.csv')
# =======================

# Main execution block
if __name__ == "__main__":
    overall_start_time = time.time()
    print("Starting Overall Model Accuracy and Resource Evaluation (using pre-synthesized local circuits)...")

    # Setup CSV and directories
    if not os.path.exists(RESULTS_DIR_MAIN): os.makedirs(RESULTS_DIR_MAIN)
    results_subdir_path = os.path.join(RESULTS_DIR_MAIN, RESULTS_SUBDIR_ACCURACY)
    if not os.path.exists(results_subdir_path):
        os.makedirs(results_subdir_path)
        print(f"Created results directory: {results_subdir_path}")

    csv_header = ['N', 'k_op_idx', 'Active_Qubits', 'Gate_Type_Used', 'Synth_Fidelity_IdealG_to_C', 
                  'Overall_Fidelity_Orig_to_Eff', 'Overall_Frobenius_Diff',
                  'Circuit_Depth_Eff', 'Total_Gates_Eff', 'Native_2Q_Gates_Eff', 'Native_2Q_Types_Eff',
                  'Native_Ops_Counts_Eff', 'Transpilation_Time_S_Eff', 'Status'] # Added Status
    
    write_header_flag = not os.path.exists(CSV_ACCURACY_FILE) or os.path.getsize(CSV_ACCURACY_FILE) == 0
    if write_header_flag:
        with open(CSV_ACCURACY_FILE, 'w', newline='') as f_csv:
            writer = csv.writer(f_csv); writer.writerow(csv_header)
        print(f"Created CSV results file with header: {CSV_ACCURACY_FILE}")
    else:
        print(f"Appending to existing CSV results file: {CSV_ACCURACY_FILE}")

    # 1. Load Pre-synthesized C_L, C_M, C_R circuits
    print("\nLoading pre-synthesized local circuits C_L, C_M, C_R...")
    synthesized_circuits = {}
    synthesis_fidelities = {
        "L": KNOWN_SYNTH_FID_L, 
        "M": KNOWN_SYNTH_FID_M, 
        "R": KNOWN_SYNTH_FID_R
    }
    paths_to_circs = {"L": PATH_C_L_QPY, "M": PATH_C_M_QPY, "R": PATH_C_R_QPY}
    
    all_circs_loaded = True
    for gate_type, qpy_path in paths_to_circs.items():
        if not os.path.exists(qpy_path):
            print(f"ERROR: QPY file not found for C_{gate_type}: {qpy_path}")
            all_circs_loaded = False; break
        try:
            with open(qpy_path, 'rb') as fd:
                loaded_circs = qpy.load(fd)
                if not loaded_circs or not isinstance(loaded_circs[0], QuantumCircuit):
                    raise ValueError(f"QPY file {qpy_path} did not contain a valid QuantumCircuit.")
                synthesized_circuits[gate_type] = loaded_circs[0]
            print(f"  Circuit C_{gate_type} loaded from {qpy_path} (Known Synth Fidelity: {synthesis_fidelities[gate_type]:.6f})")
        except Exception as e:
            print(f"ERROR loading pre-synthesized circuit C_{gate_type} from {qpy_path}: {e}")
            all_circs_loaded = False; break
            
    if not all_circs_loaded:
        print("Exiting due to failure in loading one or more pre-synthesized circuits.")
        exit()

    # Optional: Connect to IBM Backend
    target_backend_instance = None
    if TARGET_BACKEND_NAME:
        try:
            print(f"\nConnecting to IBM Quantum for backend {TARGET_BACKEND_NAME}...")
            service = QiskitRuntimeService() 
            # N_QUBITS_LIST is defined globally and should be accessible here
            min_q_for_sweep = 3 
            if N_QUBITS_LIST: min_q_for_sweep = max(3, max(N_QUBITS_LIST))
            else: print("Warning: N_QUBITS_LIST is empty or not defined for backend sweep min_qubits.")

            target_backend_instance = qst.get_ibm_backend(service, TARGET_BACKEND_NAME, min_qubits=min_q_for_sweep)
            if target_backend_instance is None:
                print(f"WARNING: Could not get target backend. Transpilation resource counts will be generic.")
        except Exception as e:
            print(f"ERROR connecting to IBM Quantum: {e}"); target_backend_instance = None
    
    all_run_results_data = []

    # THIS IS THE LOOP WHERE THE NameError OCCURRED (line 138 in user's log)
    for N_val in N_QUBITS_LIST: 
        if N_val < 3: continue
        if target_backend_instance and N_val > target_backend_instance.num_qubits:
            print(f"Skipping N={N_val} (exceeds backend {target_backend_instance.name} capacity).")
            continue
            
        print(f"\n===== Evaluating N_QUBITS = {N_val} =====")
        num_operators = N_val - 1 

        for k_op_idx in range(num_operators):
            op_name_full = f"B'_{k_op_idx}(N={N_val})"
            print(f"\n--- Processing {op_name_full} ---")

            gate_type_str, active_qubits_list_int = "", []
            if k_op_idx == 0:
                gate_type_str, active_qubits_list_int = "L", [0, 1, 2]
            elif k_op_idx == N_val - 2:
                gate_type_str, active_qubits_list_int = "R", [N_val - 3, N_val - 2, N_val - 1]
            else:
                gate_type_str, active_qubits_list_int = "M", [k_op_idx, k_op_idx + 1, k_op_idx + 2]
            
            chosen_synth_circuit = synthesized_circuits[gate_type_str]
            synthesis_fidelity = synthesis_fidelities[gate_type_str]

            current_result_row = {
                'N': N_val, 'k_op_idx': k_op_idx, 
                'Active_Qubits': str(active_qubits_list_int), 
                'Gate_Type_Used': gate_type_str,
                'Synth_Fidelity_IdealG_to_C': f"{synthesis_fidelity:.6f}",
                'Overall_Fidelity_Orig_to_Eff': 'N/A', 
                'Overall_Frobenius_Diff': 'N/A',
                'Circuit_Depth_Eff': 'N/A', 'Total_Gates_Eff': 'N/A', 
                'Native_2Q_Gates_Eff': 'N/A', 'Native_2Q_Types_Eff': 'N/A', 
                'Native_Ops_Counts_Eff': 'N/A', 'Transpilation_Time_S_Eff': 'N/A',
                'Status': 'Pending'
            }

            print(f"  a. Computing B'_k(N)_original for {op_name_full}...")
            U_orig_matrix = qst.get_original_Bk_matrix(N_val, k_op_idx, verbose=False)
            if U_orig_matrix is None:
                current_result_row.update({'Status': 'Error (Orig Construct)'})
                # --- SAVE ROW TO CSV ---
                with open(CSV_ACCURACY_FILE, 'a', newline='') as f_csv:
                    writer = csv.DictWriter(f_csv, fieldnames=csv_header)
                    writer.writerow(current_result_row)
                continue

            Operator_orig = Operator(U_orig_matrix)
            if not Operator_orig.is_unitary(atol=1e-7):
                u_udag_diff_norm = np.linalg.norm((U_orig_matrix @ U_orig_matrix.conj().T) - np.eye(2**N_val), 'fro')
                print(f"  WARNING: {op_name_full}_original is not strictly unitary! ||UUdag-I||_F = {u_udag_diff_norm:.3e}.")

            print(f"  b. Constructing U_eff_model using circuit C_{gate_type_str} on qubits {active_qubits_list_int}...")
            U_eff_model_matrix = None
            try:
                qc_eff_model = QuantumCircuit(N_val)
                qc_eff_model.compose(chosen_synth_circuit, qubits=active_qubits_list_int, inplace=True)
                U_eff_model_matrix = Operator(qc_eff_model).data
                Operator_eff_model = Operator(U_eff_model_matrix)
            except Exception as e:
                print(f"  ERROR creating U_eff_model for {op_name_full}: {e}")
                current_result_row.update({'Status': 'Error (Eff Model Construct)'})
                # --- SAVE ROW TO CSV ---
                with open(CSV_ACCURACY_FILE, 'a', newline='') as f_csv:
                    writer = csv.DictWriter(f_csv, fieldnames=csv_header)
                    writer.writerow(current_result_row)
                continue

            overall_frob_diff = np.linalg.norm(U_orig_matrix - U_eff_model_matrix, 'fro')
            current_result_row['Overall_Frobenius_Diff'] = f"{overall_frob_diff:.4e}"
            print(f"  c. Overall Frobenius Diff: {overall_frob_diff:.4e}")

            try:
                if Operator_orig.is_unitary(atol=1e-7) and Operator_eff_model.is_unitary(atol=1e-7):
                    trace_val_overall = np.trace(np.conj(U_orig_matrix).T @ U_eff_model_matrix)
                    overall_fidelity = (np.abs(trace_val_overall)**2) / ((2**N_val)**2)
                    current_result_row['Overall_Fidelity_Orig_to_Eff'] = f"{overall_fidelity:.6f}"
                    print(f"  d. Overall Avg Gate Fidelity: {overall_fidelity:.6f}")
                else:
                    current_result_row['Overall_Fidelity_Orig_to_Eff'] = "N/A (Non-Unitary)"
            except Exception:
                current_result_row['Overall_Fidelity_Orig_to_Eff'] = "Error (Fid Calc)"
            
            print(f"  e. Transpiling final model circuit for {op_name_full}...")
            transpile_start_time = time.time()
            transpiled_qc = transpile(qc_eff_model, backend=target_backend_instance, optimization_level=OPTIMIZATION_LEVEL_FINAL_TRANSPILE)
            transpile_duration = time.time() - transpile_start_time
            
            ops_counts_dict = dict(transpiled_qc.count_ops())
            current_result_row.update({
                'Circuit_Depth_Eff': transpiled_qc.depth(), 
                'Total_Gates_Eff': sum(ops_counts_dict.values()),
                'Native_Ops_Counts_Eff': str(ops_counts_dict), 
                'Transpilation_Time_S_Eff': f"{transpile_duration:.3f}", 
                'Status': 'Success'
            })
            num_native_2q_gates = 0; native_2q_types_set = set()
            basis_gates_to_check = getattr(target_backend_instance.configuration(), 'basis_gates', []) if target_backend_instance else ['cx','ecr','cz','swap']
            known_2q = {'cx', 'ecr', 'cz', 'swap', 'rzz', 'rzx', 'zz'}
            for gate, count in ops_counts_dict.items():
                if gate in basis_gates_to_check and gate in known_2q: 
                    num_native_2q_gates += count; native_2q_types_set.add(gate)
                elif not target_backend_instance and gate in known_2q:
                    num_native_2q_gates += count; native_2q_types_set.add(gate)
            current_result_row['Native_2Q_Gates_Eff'] = num_native_2q_gates
            current_result_row['Native_2Q_Types_Eff'] = ', '.join(sorted(list(native_2q_types_set))) if native_2q_types_set else "None"
            print(f"     Depth: {current_result_row['Circuit_Depth_Eff']}, Total Gates: {current_result_row['Total_Gates_Eff']}, Native 2Q Gates: {current_result_row['Native_2Q_Gates_Eff']}")

            # --- SAVE ROW TO CSV ---
            with open(CSV_ACCURACY_FILE, 'a', newline='') as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=csv_header)
                writer.writerow(current_result_row)

    # # Write all results to CSV
    # with open(CSV_ACCURACY_FILE, 'a', newline='') as f_csv:
    #     writer = csv.DictWriter(f_csv, fieldnames=csv_header, extrasaction='ignore')
    #     if write_header_flag: # Only write header if it's a new file
    #          # This check might be redundant if already handled, but safe
    #         if f_csv.tell() == 0: # Double check if file is actually empty before writing header
    #             writer.writeheader()
    #     for row_dict in all_run_results_data:
    #         writer.writerow(row_dict)
    # print(f"\nAll accuracy results for this run appended to {CSV_ACCURACY_FILE}")
    
    print("\nOverall Model Accuracy Summary (from this run):")
    header_titles_map = { # Map keys to slightly shorter titles for display
        'N': 'N', 'k_op_idx': 'k_op', 'Active_Qubits': 'ActiveQ', 'Gate_Type_Used': 'Type',
        'Synth_Fidelity_IdealG_to_C': 'SynFid', 'Overall_Fidelity_Orig_to_Eff': 'OvrFid',
        'Overall_Frobenius_Diff': 'OvrFrobDiff', 'Circuit_Depth_Eff': 'Depth',
        'Total_Gates_Eff': 'TotG', 'Native_2Q_Gates_Eff': 'N2QG', 'Native_2Q_Types_Eff': 'N2QType',
        'Native_Ops_Counts_Eff': 'OpsCounts', 'Transpilation_Time_S_Eff': 'TransTime',
        'Status': 'Status'
    }
    # Construct format string dynamically based on max widths or fixed widths
    header_fmt_str = ("{N:<3} | {k_op_idx:<4} | {Active_Qubits:<12} | {Gate_Type_Used:<4} | "
                      "{Synth_Fidelity_IdealG_to_C:<10} | {Overall_Fidelity_Orig_to_Eff:<12} | {Overall_Frobenius_Diff:<18} | "
                      "{Circuit_Depth_Eff:<6} | {Total_Gates_Eff:<6} | {Native_2Q_Gates_Eff:<5} | {Native_2Q_Types_Eff:<10} | "
                      "{Transpilation_Time_S_Eff:<10} | {Status:<20}") # OpsCounts removed for brevity here
    
    print(header_fmt_str.format(**{k:v for k,v in header_titles_map.items() if k in csv_header and k != 'Native_Ops_Counts_Eff'})) # Print header
    print("-" * (len(header_fmt_str.format(**{k:v for k,v in header_titles_map.items() if k in csv_header and k != 'Native_Ops_Counts_Eff'})) + 5 ))

    for res in all_run_results_data:
        res_to_print = {key: res.get(key, 'N/A') for key in csv_header}
        # if len(str(res_to_print['Native_Ops_Counts_Eff'])) > 28 : res_to_print['Native_Ops_Counts_Eff'] = str(res_to_print['Native_Ops_Counts_Eff'])[:25] + "..."
        print(header_fmt_str.format(**res_to_print).replace(res_to_print['Native_Ops_Counts_Eff'], str(res_to_print['Native_Ops_Counts_Eff'])[:15]+"..." if len(str(res_to_print['Native_Ops_Counts_Eff'])) > 18 else str(res_to_print['Native_Ops_Counts_Eff'])))
        
    total_duration = time.time() - overall_start_time
    print(f"\nTotal script execution time: {total_duration:.2f} seconds.")

