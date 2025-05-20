# batch_unitary_synthesis_analysis.py
# Analyze multiple target 8x8 unitaries (G_ideal), trying several ansatz depths.

import numpy as np
import os
import time
import csv
import glob

# Qiskit imports (assuming QISKIT_AVAILABLE is handled by qst or checked before use)
from qiskit import QuantumCircuit, transpile
# ParameterVector not strictly needed here if qst.create_3q_ansatz handles parameters internally
# from qiskit.circuit import ParameterVector 
from qiskit.quantum_info import Operator
from scipy.optimize import minimize

# Import your synthesis tools module
try:
    import qic_synthesis_tools as qst
except ImportError:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Assuming 'simulations' is sibling to 'src' where qic_synthesis_tools might be,
        # or qic_synthesis_tools is in 'src' and 'src' is added to path.
        # This path adjustment might need to be specific to your project structure.
        # For simplicity, let's assume qic_synthesis_tools.py is in a directory
        # that's already in sys.path or in the same directory as this script.
        # If it's in ../src relative to this script's parent:
        # src_dir = os.path.join(os.path.dirname(current_dir), 'src') 
        # if src_dir not in sys.path:
        #    sys.path.insert(0, src_dir)
        import qic_synthesis_tools as qst
        print("Successfully imported qic_synthesis_tools (path might have been adjusted).")
    except ImportError as e_qst:
        print(f"ERROR: Could not import qic_synthesis_tools.py: {e_qst}")
        print("       Ensure qic_synthesis_tools.py is in your Python path or src directory.")
        exit()


# ==== User Configuration ====
IDEAL_GATES_DATA_DIR = "data/optimal_local_approximators"
GIDEAL_FILES = sorted(
    glob.glob(
        os.path.join(IDEAL_GATES_DATA_DIR, "*.npy")
    )
)
if not GIDEAL_FILES:
    print(f"WARNING: No .npy files found in {IDEAL_GATES_DATA_DIR}. Check path and files.")

NUM_LAYERS_LIST = [1, 2, 3, 4] # List of ansatz layers to try for each target
MAX_OPTIMIZATION_ITER = 200    # Max iterations for scipy.minimize

# Configuration for the ansatz structure within qst.create_3q_ansatz
# This should match the expectation of qst.create_3q_ansatz
# For RZ-SX-RZ-SX-RZ sequence on each qubit, this is 3 parameterized RZ gates.
NUM_ROT_PARAMS_PER_QUBIT_PER_LAYER = 3 

# Output summary CSV
RESULTS_DIR_MAIN = 'data'
RESULTS_SUBDIR_BATCH_SYNTH = 'batch_synthesis_results'
RESULTS_CSV = os.path.join(RESULTS_DIR_MAIN, RESULTS_SUBDIR_BATCH_SYNTH, "gate_synthesis_batch_summary.csv")
# =========================

def run_synthesis_for_target(target_unitary_file, num_ansatz_layers, 
                             num_rot_params_config, max_iters):
    """
    Loads a target unitary, creates an ansatz, optimizes, and returns results.
    """
    print(f"  Synthesizing {os.path.basename(target_unitary_file)} with {num_ansatz_layers} layer(s)...")
    
    # 1. Load target unitary
    try:
        target_unitary_matrix = np.load(target_unitary_file)
        if target_unitary_matrix.shape != (8,8):
            print(f"    ERROR: Target matrix in {target_unitary_file} has shape {target_unitary_matrix.shape}, expected (8,8). Skipping.")
            return None
        if not Operator(target_unitary_matrix).is_unitary(atol=1e-7): # Check if loaded is unitary
            u_udag_diff_norm = np.linalg.norm(target_unitary_matrix @ target_unitary_matrix.conj().T - np.eye(8), 'fro')
            print(f"    WARNING: Loaded target matrix from {target_unitary_file} may not be perfectly unitary itself! ||UUdag-I||_F = {u_udag_diff_norm:.3e}")
    except Exception as e:
        print(f"    ERROR loading target unitary from {target_unitary_file}: {e}")
        return None

    # 2. Create ansatz using the imported tool
    #    qst.create_3q_ansatz now expects (num_layers, num_rot_params_per_qubit_per_layer)
    #    and returns only the QuantumCircuit object (qc_template).
    ansatz_qc_template = qst.create_3q_ansatz(num_ansatz_layers, num_rot_params_config)
    
    # Get the actual Parameter objects present in the circuit template and sort them
    unbound_params_in_circuit = sorted(list(ansatz_qc_template.parameters), key=lambda p: p.name)
    num_actual_params = len(unbound_params_in_circuit)

    if num_actual_params == 0 and num_ansatz_layers > 0 : # Check if any parameters were actually added
        print(f"    ERROR: Ansatz for {num_ansatz_layers} layers resulted in no parameters. Check qst.create_3q_ansatz.")
        return None
    if num_actual_params == 0 and num_ansatz_layers == 0:
        print(f"    Ansatz for 0 layers has no parameters. Evaluating fixed structure.")
        # Fall through to objective_function with empty params

    # 3. Optimize
    initial_params = np.random.rand(num_actual_params) * 2 * np.pi
    
    # print(f"    Starting optimization with {num_actual_params} parameters...")
    optimization_start_time = time.time()
    res = minimize(
        qst.objective_function, # Using the objective function from the tools module
        initial_params,
        args=(ansatz_qc_template, unbound_params_in_circuit, target_unitary_matrix),
        method='L-BFGS-B',
        options={'maxiter': max_iters, 'disp': False, 'eps': 1e-9, 'ftol': 1e-10, 'gtol': 1e-7}
    )
    optimization_duration = time.time() - optimization_start_time
    # print(f"    Optimization finished in {optimization_duration:.2f}s.")

    if not res.success:
        print(f"    WARNING: Optimization for {os.path.basename(target_unitary_file)} (layers={num_ansatz_layers}) not fully converged: {res.message}")

    min_infidelity = res.fun
    optimal_params_values = res.x
    achieved_fidelity = 1.0 - min_infidelity

    # 4. Gate count (transpile to basis for resource count)
    optimal_param_dict = {unbound_params_in_circuit[i]: optimal_params_values[i] for i in range(len(optimal_params_values))}
    optimized_circuit = ansatz_qc_template.assign_parameters(optimal_param_dict)
    
    # Transpile for a generic basis to count 1Q/2Q gates, not specific to a backend here
    # Using a common basis set for estimation
    try:
        transpiled_optimized_circuit = transpile(optimized_circuit, basis_gates=['u3', 'cx', 'ecr', 'id'], optimization_level=3)
        ops_counts = transpiled_optimized_circuit.count_ops()
        # More robust count for any 2Q gate names Qiskit might use after transpilation
        two_qubit_gates = 0
        known_2q_gate_names = {'cx', 'ecr', 'cz', 'swap', 'rzz', 'rzx', 'zz', 'csx', 'cy', 'cp', 'crx', 'cry', 'crz', 'ch'} # Expanded list
        for gate, count in ops_counts.items():
            if gate in known_2q_gate_names:
                two_qubit_gates += count
    except Exception as e_transpile:
        print(f"    Error during transpilation for resource counting: {e_transpile}")
        ops_counts = {"error": str(e_transpile)}
        two_qubit_gates = -1 # Indicate error

    return {
        "file": os.path.basename(target_unitary_file),
        "num_layers": num_ansatz_layers,
        "params_values": optimal_params_values,
        "min_infidelity": min_infidelity,
        "achieved_fidelity": achieved_fidelity,
        "num_gates_total": sum(ops_counts.values()) if isinstance(ops_counts, dict) else -1,
        "num_two_qubit_gates": two_qubit_gates,
        "ops_counts": dict(ops_counts) if isinstance(ops_counts, dict) else ops_counts,
    }

def main():
    all_synthesis_results = []
    script_start_time = time.time()

    print("Starting Batch Local Unitary Synthesis Analysis...\n")
    if not GIDEAL_FILES:
        print("No target .npy files found to process. Exiting.")
        return

    for target_file_path in GIDEAL_FILES:
        print(f"Processing Target Unitary: {os.path.basename(target_file_path)}")
        for layers_count in NUM_LAYERS_LIST:
            print(f"  Trying with {layers_count} ansatz layer(s)...")
            synthesis_output = run_synthesis_for_target(
                target_file_path, 
                layers_count, 
                NUM_ROT_PARAMS_PER_QUBIT_PER_LAYER, # Pass the config for num_rot_params
                MAX_OPTIMIZATION_ITER
            )
            if synthesis_output:
                all_synthesis_results.append(synthesis_output)
                print(f"    Fidelity: {synthesis_output['achieved_fidelity']:.6f} | 2Q gates: {synthesis_output['num_two_qubit_gates']} | Total gates: {synthesis_output['num_gates_total']}")
            else:
                print(f"    Synthesis failed for {os.path.basename(target_file_path)} with {layers_count} layer(s).")
        print("-" * 30)


    # Prepare CSV output
    results_output_dir = os.path.join(RESULTS_DIR_MAIN, RESULTS_SUBDIR_BATCH_SYNTH)
    if not os.path.exists(results_output_dir):
        os.makedirs(results_output_dir)
        print(f"Created results directory: {results_output_dir}")

    print(f"\nWriting all synthesis results to {RESULTS_CSV}")
    with open(RESULTS_CSV, 'w', newline='') as f_csv:
        fieldnames = [
            "file", "num_layers", "min_infidelity", "achieved_fidelity",
            "num_gates_total", "num_two_qubit_gates", "ops_counts", "params_values"
        ]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for r_data in all_synthesis_results:
            row_to_write = r_data.copy()
            # Convert numpy array of params to string for CSV
            if 'params_values' in row_to_write and isinstance(row_to_write['params_values'], np.ndarray):
                row_to_write['params_values'] = np.array2string(row_to_write['params_values'], separator=',', precision=5, max_line_width=np.inf)
            if 'ops_counts' in row_to_write and isinstance(row_to_write['ops_counts'], dict):
                 row_to_write['ops_counts'] = str(row_to_write['ops_counts']) # Convert dict to string
            writer.writerow(row_to_write)
    
    print("\n=== Best Synthesis per Target Unitary (Highest Fidelity, then Lowest 2Q Gates) ===")
    for target_file_path in GIDEAL_FILES:
        target_filename = os.path.basename(target_file_path)
        candidates_for_file = [r for r in all_synthesis_results if r['file'] == target_filename and r['num_two_qubit_gates'] != -1]
        if not candidates_for_file:
            print(f"No successful synthesis results for {target_filename}")
            continue
        
        # Sort by fidelity (descending), then by num_two_qubit_gates (ascending)
        best_candidate = sorted(candidates_for_file, key=lambda x: (x['achieved_fidelity'], -x['num_two_qubit_gates']), reverse=True)[0]
        
        print(f"{best_candidate['file']:<35} | Fidelity: {best_candidate['achieved_fidelity']:.6f} | Layers: {best_candidate['num_layers']:<2} | "
              f"2Q gates: {best_candidate['num_two_qubit_gates']:<3} | Total gates: {best_candidate['num_gates_total']:<4}")
        # print(f"  OpCounts: {best_candidate['ops_counts']}") # Can be verbose

    total_script_duration = time.time() - script_start_time
    print(f"\nTotal script execution time: {total_script_duration:.1f} seconds.")

if __name__ == "__main__":
    main()
