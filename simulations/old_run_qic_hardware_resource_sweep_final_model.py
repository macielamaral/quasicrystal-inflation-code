# run_qic_hardware_resource_sweep_final_model.py
# Estimates resources for QIC braid operators B'_k using pre-synthesized
# ideal local gate circuits (C_L, C_M, C_R).

import numpy as np
import time
import traceback
import csv
import os
import sys

import qic_synthesis_tools as qst
# --- Ensure qic_core is importable (primarily for PHI if not defined, and QISKIT_AVAILABLE) ---
try:
    import qic_core
    PHI_from_core = getattr(qic_core, 'PHI', None)
    QISKIT_AVAILABLE_from_core = getattr(qic_core, 'QISKIT_AVAILABLE', False)
except ImportError:
    print("Warning: qic_core.py not found or QISKIT_AVAILABLE/PHI not defined within it.")
    print("         This script primarily needs Qiskit and Scipy for its core functionality.")
    print("         Defining PHI numerically and assuming Qiskit is available.")
    PHI_from_core = None
    QISKIT_AVAILABLE_from_core = True # Assume Qiskit is available if this script is run
    # Fallback for PHI if not found in qic_core
    PHI = (1 + np.sqrt(5)) / 2

if QISKIT_AVAILABLE_from_core:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit.providers.exceptions import QiskitBackendNotFoundError
    from qiskit.quantum_info import Operator
    from scipy.optimize import minimize # For synthesize_ideal_gate
else:
    print("ERROR: Qiskit (and Scipy for synthesis) is required for this script but not available.")
    exit()

# Use PHI from qic_core if available, otherwise use the fallback
PHI = PHI_from_core if PHI_from_core is not None else (1 + np.sqrt(5)) / 2

# ===== Configuration =====
N_QUBITS_LIST = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
TARGET_BACKEND_NAME = "ibm_brisbane"
OPTIMIZATION_LEVEL_FINAL_TRANSPILE = 1 # Since local circuits are already somewhat optimized
RESULTS_DIR_MAIN = 'data' # Main data directory
RESULTS_SUBDIR_FINAL_MODEL = 'final_model_resources' # Subdirectory for these specific results
CSV_RESULTS_FILE = os.path.join(RESULTS_DIR_MAIN, RESULTS_SUBDIR_FINAL_MODEL, 'qic_final_model_resource_summary.csv')

# Paths to your chosen pre-analyzed ideal G matrices (8x8 numpy arrays)
# These should be the .npy files of G_tilde(N_large, k_type) that you selected as ideal.
# Ensure these files exist and contain 8x8 unitary matrices.
IDEAL_GATES_DIR = 'data/optimal_local_approximators' # Directory where G_tilde .npy files are
PATH_G_IDEAL_L = os.path.join(IDEAL_GATES_DIR, "G_tilde_N10_kop0_act0.npy")
PATH_G_IDEAL_M = os.path.join(IDEAL_GATES_DIR, "G_tilde_N10_kop4_act4.npy")
PATH_G_IDEAL_R = os.path.join(IDEAL_GATES_DIR, "G_tilde_N10_kop8_act7.npy")

# Ansatz and optimization settings for synthesizing the G_ideal matrices ONCE
# These should match the settings that gave you good fidelity for these ideal gates
SYNTHESIS_NUM_ANSATZ_LAYERS = 4 
SYNTHESIS_PARAMS_PER_LAYER = 3 * 3 
SYNTHESIS_MAX_OPTIMIZATION_ITERATIONS = 200
# =======================

# --- Main execution block ---
if __name__ == "__main__":
    master_start_time = time.time()
    
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR_MAIN):
        os.makedirs(RESULTS_DIR_MAIN)
    if not os.path.exists(os.path.join(RESULTS_DIR_MAIN, RESULTS_SUBDIR_FINAL_MODEL)):
        os.makedirs(os.path.join(RESULTS_DIR_MAIN, RESULTS_SUBDIR_FINAL_MODEL))
        print(f"Created results directory: {os.path.join(RESULTS_DIR_MAIN, RESULTS_SUBDIR_FINAL_MODEL)}")

    csv_header = ['N_QUBITS', 'OPERATOR_K_IDX', 'ACTIVE_QUBITS', 'GATE_TYPE_USED', 
                  'SYNTHESIS_FIDELITY', 'CIRCUIT_DEPTH', 'TOTAL_GATES', 
                  'NATIVE_2Q_GATES', 'NATIVE_2Q_TYPES', 'NATIVE_OPS_COUNTS', 
                  'TRANSPILATION_TIME_S']
    if not os.path.exists(CSV_RESULTS_FILE) or os.path.getsize(CSV_RESULTS_FILE) == 0:
        with open(CSV_RESULTS_FILE, 'w', newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(csv_header)
        print(f"Created CSV results file: {CSV_RESULTS_FILE}")
    else:
        print(f"Appending to existing CSV results file: {CSV_RESULTS_FILE}")

    service = None
    target_backend_instance = None
    try:
        print("\nConnecting to IBM Quantum Runtime Service...")
        service = QiskitRuntimeService() 
        print("Connected to IBM Quantum.")
        min_q_for_sweep = 3 
        if N_QUBITS_LIST: min_q_for_sweep = max(3, max(N_QUBITS_LIST))
        target_backend_instance = qst.get_ibm_backend(service, TARGET_BACKEND_NAME, min_qubits=min_q_for_sweep)
        if target_backend_instance is None:
            print("Could not get a target backend. Exiting.")
            exit()
    except Exception as e:
        print(f"ERROR connecting to IBM Quantum or getting backend: {e}")
        traceback.print_exc()
        exit()
    
    print("\nLoading Ideal Local Gate Matrices...")
    try:
        G_ideal_L_matrix = np.load(PATH_G_IDEAL_L)
        G_ideal_M_matrix = np.load(PATH_G_IDEAL_M)
        G_ideal_R_matrix = np.load(PATH_G_IDEAL_R)
        print("Ideal G matrices loaded successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: Could not load one of the ideal G matrices: {e}")
        print("Please check PATH_G_IDEAL_L, _M, _R variables and ensure files exist.")
        exit()

    # Synthesize these ideal fixed gates ONCE to get QuantumCircuit objects
    # These circuits (C_L, C_M, C_R) are the building blocks.
    # The synthesize_ideal_gate function now returns the circuit.
    # We also need to store their synthesis fidelities to report.
    
    C_L_synth_results = {}
    C_M_synth_results = {}
    C_R_synth_results = {}

    C_L = qst.synthesize_ideal_gate(G_ideal_L_matrix, SYNTHESIS_NUM_ANSATZ_LAYERS, SYNTHESIS_PARAMS_PER_LAYER, SYNTHESIS_MAX_OPTIMIZATION_ITERATIONS, "G_ideal_L")
    if C_L: C_L_synth_results['fidelity'] = 1.0 - qst.objective_function(C_L.parameters, C_L, list(C_L.parameters), G_ideal_L_matrix) if C_L.parameters else 1.0 - objective_function([], C_L, [], G_ideal_L_matrix)


    C_M = qst.synthesize_ideal_gate(G_ideal_M_matrix, SYNTHESIS_NUM_ANSATZ_LAYERS, SYNTHESIS_PARAMS_PER_LAYER, SYNTHESIS_MAX_OPTIMIZATION_ITERATIONS, "G_ideal_M")
    if C_M: C_M_synth_results['fidelity'] = 1.0 - qst.objective_function(C_M.parameters, C_M, list(C_M.parameters), G_ideal_M_matrix) if C_M.parameters else 1.0 - objective_function([], C_M, [], G_ideal_M_matrix)


    C_R = qst.synthesize_ideal_gate(G_ideal_R_matrix, SYNTHESIS_NUM_ANSATZ_LAYERS, SYNTHESIS_PARAMS_PER_LAYER, SYNTHESIS_MAX_OPTIMIZATION_ITERATIONS, "G_ideal_R")
    if C_R: C_R_synth_results['fidelity'] = 1.0 - qst.objective_function(C_R.parameters, C_R, list(C_R.parameters), G_ideal_R_matrix) if C_R.parameters else 1.0 - objective_function([], C_R, [], G_ideal_R_matrix)


    if not all([C_L, C_M, C_R]):
        print("ERROR: Failed to synthesize one or more ideal local gate circuits. Exiting.")
        exit()
    
    all_run_results_data = [] # To collect data for CSV

    for N_val in N_QUBITS_LIST:
        if N_val < 3: continue
        if N_val > target_backend_instance.num_qubits:
            print(f"Skipping N_QUBITS={N_val} (exceeds backend capacity).")
            continue
            
        print(f"\n===== Estimating resources for N_QUBITS = {N_val} using synthesized fixed local gates =====")
        num_operators = N_val - 1 

        for k_op_idx in range(num_operators):
            op_name = f"B'_{k_op_idx}(N={N_val})"
            print(f"\n-- Processing {op_name} --")

            chosen_local_circuit_template = None
            target_qubits = []
            gate_type_name_str = ""
            synthesis_fidelity_for_type = 0.0

            if k_op_idx == 0:
                chosen_local_circuit_template = C_L
                target_qubits = [0, 1, 2]
                gate_type_name_str = "L"
                synthesis_fidelity_for_type = C_L_synth_results.get('fidelity', 0.0)
            elif k_op_idx == N_val - 2:
                chosen_local_circuit_template = C_R
                target_qubits = [N_val - 3, N_val - 2, N_val - 1]
                gate_type_name_str = "R"
                synthesis_fidelity_for_type = C_R_synth_results.get('fidelity', 0.0)
            else: 
                chosen_local_circuit_template = C_M
                target_qubits = [k_op_idx, k_op_idx + 1, k_op_idx + 2]
                gate_type_name_str = "M"
                synthesis_fidelity_for_type = C_M_synth_results.get('fidelity', 0.0)
            
            active_qubits_str = str(target_qubits)
            current_result_row = {
                'N_QUBITS': N_val, 'OPERATOR_K_IDX': k_op_idx, 
                'ACTIVE_QUBITS': active_qubits_str, 
                'GATE_TYPE_USED': gate_type_name_str,
                'SYNTHESIS_FIDELITY': f"{synthesis_fidelity_for_type:.6f}",
                'CIRCUIT_DEPTH': 'Error', 'TOTAL_GATES': 'Error', 
                'NATIVE_2Q_GATES': 'Error', 'NATIVE_2Q_TYPES': 'Error', 
                'NATIVE_OPS_COUNTS': 'Error', 'TRANSPILATION_TIME_S': 'Error'
            }

            try:
                qc_final_model = QuantumCircuit(N_val, name=op_name)
                qc_final_model.compose(chosen_local_circuit_template, qubits=target_qubits, inplace=True)
                
                print(f"Transpiling {op_name} (with pre-synthesized '{gate_type_name_str}' block on {target_qubits}) for {target_backend_instance.name}...")
                start_transpile_time = time.time()
                transpiled_qc = transpile(qc_final_model, backend=target_backend_instance, optimization_level=OPTIMIZATION_LEVEL_FINAL_TRANSPILE)
                end_transpile_time = time.time()
                transpile_duration = end_transpile_time - start_transpile_time
                print(f"Transpilation finished ({transpile_duration:.3f} s).")

                depth = transpiled_qc.depth()
                ops_counts = transpiled_qc.count_ops()
                total_gates = sum(ops_counts.values())
                
                backend_config = target_backend_instance.configuration()
                backend_basis_gates_list = getattr(backend_config, 'basis_gates', [])
                known_2q_gate_names = {'cx', 'ecr', 'cz', 'swap', 'rzz', 'rzx', 'zz'} 
                native_2q_gates_found = {gate for gate in ops_counts.keys() if gate in known_2q_gate_names}
                
                num_native_two_qubit_gates = 0
                for gate_name, count in ops_counts.items():
                    if gate_name in native_2q_gates_found:
                        num_native_two_qubit_gates += count
                
                current_result_row.update({
                    'CIRCUIT_DEPTH': depth, 'TOTAL_GATES': total_gates,
                    'NATIVE_2Q_GATES': num_native_two_qubit_gates, 
                    'NATIVE_2Q_TYPES': ', '.join(sorted(list(native_2q_gates_found))),
                    'NATIVE_OPS_COUNTS': str(dict(ops_counts)), # Convert OrderedDict for easier CSV
                    'TRANSPILATION_TIME_S': f"{transpile_duration:.3f}"
                })
                print(f"  {op_name} - Depth: {depth}, Total Gates: {total_gates}, 2Q Gates: {num_native_two_qubit_gates}")

            except Exception as e:
                print(f"ERROR during processing for {op_name} at N={N_val}: {e}")
                current_result_row['NATIVE_OPS_COUNTS'] = str(e) # Store error
            
            all_run_results_data.append(current_result_row)
    
    # Write all results to CSV at the end
    with open(CSV_RESULTS_FILE, 'a', newline='') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_header)
        # If file was empty or newly created, write header (already handled)
        for row_dict in all_run_results_data:
            writer.writerow(row_dict)
    print(f"\nAll results for this run appended to {CSV_RESULTS_FILE}")
    
    # Print summary to console
    print("\nSummary of this Run (also in CSV):")
    print(f"{'N':<4} | {'k_op':<4} | {'ActQ':<12} | {'Type':<4} | {'SynthFid':<10} | {'Depth':<6} | {'TotG':<5} | {'2QG':<4} | {'2QType':<10} | {'TransTime':<10}")
    print("-" * 90)
    for res in all_run_results_data:
        print(f"{res['N_QUBITS']:<4} | {res['OPERATOR_K_IDX']:<4} | {res['ACTIVE_QUBITS']:<12} | {res['GATE_TYPE_USED']:<4} | "
              f"{res['SYNTHESIS_FIDELITY']:<10} | {res['CIRCUIT_DEPTH']:<6} | {res['TOTAL_GATES']:<5} | "
              f"{res['NATIVE_2Q_GATES']:<4} | {res['NATIVE_2Q_TYPES']:<10} | {res['TRANSPILATION_TIME_S']:<10}")

    total_duration = time.time() - master_start_time
    print(f"\nTotal script execution time: {total_duration:.2f} seconds")