# approximate_local_unitary_synthesis_from_optimal.py
# Finds a local circuit approximation for a specific 8x8 target unitary 
# (loaded from a .npy file, e.g., a G_tilde(N,k) matrix)
# using a parameterized ansatz and optimization.

import numpy as np
import time
import traceback
import os
import sys

# --- Ensure qic_core is importable (for PHI if needed, though not for core logic here) ---
try:
    import qic_core
    PHI = getattr(qic_core, 'PHI', (1 + np.sqrt(5)) / 2)
except ImportError:
    print("Warning: qic_core.py not found. PHI will be defined numerically.")
    print("         This script primarily needs Qiskit and Scipy.")
    PHI = (1 + np.sqrt(5)) / 2 # Define PHI if qic_core is missing

# Import Qiskit components
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit_ibm_runtime import QiskitRuntimeService # Optional: for backend transpilation
    from qiskit.providers.exceptions import QiskitBackendNotFoundError # Optional
    from qiskit.quantum_info import Operator
    from scipy.optimize import minimize
except ImportError as e:
    print(f"ERROR: Missing Qiskit or Scipy. Please ensure they are installed. Error: {e}")
    exit()

# ===== Configuration =====
# Specify the path to the .npy file containing the 8x8 target unitary matrix
# This should be one of the G_tilde_N{N}_kop{k_op_idx}_act{k_active_start}.npy files
# Example:
# TARGET_G_TILDE_FILE = 'data/optimal_local_approximators/G_tilde_N4_kop1_act1.npy'
# Ensure this path is correct for your file structure.
# Please SET THIS PATH before running:
TARGET_G_TILDE_FILE = 'data/optimal_local_approximators/G_tilde_N4_kop1_act1.npy' # <--- !!! SET THIS !!!
TARGET_G_TILDE_FILE = 'data/optimal_local_approximators/G_tilde_N4_kop2_act1.npy' # <--- !!! SET THIS !!!
TARGET_G_TILDE_FILE = 'data/optimal_local_approximators/G_tilde_N5_kop0_act0.npy' # <--- !!! SET THIS !!!
TARGET_G_TILDE_FILE = 'data/optimal_local_approximators/G_tilde_N5_kop1_act1.npy' # <--- !!! SET THIS !!!
TARGET_G_TILDE_FILE = 'data/optimal_local_approximators/G_tilde_N5_kop2_act2.npy' # <--- !!! SET THIS !!!
TARGET_G_TILDE_FILE = 'data/optimal_local_approximators/G_tilde_N5_kop3_act2.npy' # <--- !!! SET THIS !!!

TARGET_G_TILDE_FILE = 'data/optimal_local_approximators/G_tilde_N10_kop0_act0.npy' # <--- !!! SET THIS !!!

# Ansatz Configuration
NUM_ANSATZ_LAYERS = 2
PARAMS_PER_SINGLE_QUBIT_LAYER = 3 * 3 
PARAMS_PER_LAYER = PARAMS_PER_SINGLE_QUBIT_LAYER 

# Optimization Configuration
MAX_OPTIMIZATION_ITERATIONS = 200

# Optional: Backend for final transpilation of the optimized ansatz
# TARGET_BACKEND_NAME = "ibm_brisbane" 
TARGET_BACKEND_NAME = None # Set to a backend name to transpile, or None to skip

# =======================

def create_3q_ansatz(num_layers, num_params_per_single_qubit_layer):
    num_qubits = 3
    total_params_needed = num_layers * num_params_per_single_qubit_layer
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector('θ', total_params_needed)
    param_idx = 0

    for l_idx in range(num_layers):
        for q_idx in range(num_qubits):
            if param_idx + 3 > total_params_needed:
                raise ValueError(f"Not enough parameters for single qubit rotations in ansatz. Needed {param_idx+3}, have {total_params_needed}")
            qc.rz(params[param_idx], q_idx); param_idx += 1
            qc.sx(q_idx)
            qc.rz(params[param_idx], q_idx); param_idx += 1
            qc.sx(q_idx)
            qc.rz(params[param_idx], q_idx); param_idx += 1
        
        if num_layers > 0 and l_idx < num_layers : 
            qc.ecr(0, 1)
            qc.ecr(1, 2)
            if num_qubits > 2 and l_idx % 2 == 0 : 
                 qc.ecr(0,2) # Add another ECR for more entanglement connectivity

    if param_idx != total_params_needed:
         print(f"Warning: Ansatz created for {total_params_needed} parameters, but structure used {param_idx}")
    return qc, params

def objective_function(param_values, ansatz_circuit_template, unbound_params_vector, target_unitary_matrix_np):
    param_dict = {unbound_params_vector[i]: param_values[i] for i in range(len(param_values))}
    bound_circuit = ansatz_circuit_template.assign_parameters(param_dict)
    
    try:
        ansatz_unitary_op = Operator(bound_circuit)
        ansatz_unitary_np = ansatz_unitary_op.data 
    except Exception as e:
        print(f"Error getting unitary from ansatz during optimization: {e}")
        return 2.0 

    d_squared = (2**ansatz_circuit_template.num_qubits)**2
    trace_val = np.trace(np.conj(target_unitary_matrix_np).T @ ansatz_unitary_np)
    fidelity = (np.abs(trace_val)**2) / d_squared
    
    infidelity = 1.0 - fidelity
    return infidelity

# --- Main execution block ---
if __name__ == "__main__":
    overall_start_time = time.time()
    print("Starting Approximate Local Unitary Synthesis from Optimal G_tilde File...")

    # 1. Load the target ideal local unitary from file
    if not os.path.exists(TARGET_G_TILDE_FILE):
        print(f"ERROR: Target G_tilde file not found: {TARGET_G_TILDE_FILE}")
        print(f"Please set the TARGET_G_TILDE_FILE variable in the script.")
        exit()

    try:
        target_ideal_unitary = np.load(TARGET_G_TILDE_FILE)
        print(f"\nSuccessfully loaded target unitary from: {TARGET_G_TILDE_FILE}")
        if target_ideal_unitary.shape != (8,8):
            print(f"Error: Loaded matrix is shape {target_ideal_unitary.shape}, expected (8,8).")
            exit()
        if not Operator(target_ideal_unitary).is_unitary(atol=1e-7): # Check if loaded is unitary
            print("WARNING: The loaded target G_tilde matrix may not be perfectly unitary itself!")
            print(f"  ||UUdag-I||_F for loaded target: {np.linalg.norm(target_ideal_unitary @ target_ideal_unitary.conj().T - np.eye(8), 'fro'):.3e}")

    except Exception as e:
        print(f"Error loading target unitary from {TARGET_G_TILDE_FILE}: {e}")
        exit()

    # 2. Create ansatz
    ansatz_qc, ansatz_params = create_3q_ansatz(NUM_ANSATZ_LAYERS, PARAMS_PER_LAYER)
    print(f"\nAnsatz Circuit ({NUM_ANSATZ_LAYERS} layer(s), {len(ansatz_params)} params):")
    print(ansatz_qc.draw(output='text'))

    if len(ansatz_params) == 0 and PARAMS_PER_LAYER > 0 and NUM_ANSATZ_LAYERS > 0:
        print("Error: Ansatz has no parameters, but parameters were expected. Check create_3q_ansatz function.")
        exit()
        
    # 3. Perform optimization
    if len(ansatz_params) > 0:
        initial_params = np.random.rand(len(ansatz_params)) * 2 * np.pi
        
        print(f"\nStarting optimization to approximate loaded target unitary...")
        print(f"Number of parameters to optimize: {len(initial_params)}")
        
        optimization_start_time = time.time()
        res = minimize(objective_function, 
                       initial_params, 
                       args=(ansatz_qc, ansatz_params, target_ideal_unitary), 
                       method='L-BFGS-B', 
                       options={'maxiter': MAX_OPTIMIZATION_ITERATIONS, 'disp': True, 'eps': 1e-9}) # eps for gradient step size
        optimization_duration = time.time() - optimization_start_time
        print(f"Optimization finished in {optimization_duration:.2f}s.")

        print("\nOptimization Result:")
        print(f"  Success: {res.success}")
        print(f"  Message: {res.message}")
        print(f"  Number of iterations: {res.nit}")
        print(f"  Number of function evaluations: {res.nfev}")
        
        optimal_params = res.x
        min_infidelity = res.fun
        print(f"  Optimal parameters found: {optimal_params}")
        print(f"  Minimum infidelity achieved (1 - F_avg): {min_infidelity:.6e}")
        achieved_fidelity = 1.0 - min_infidelity
        print(f"  Achieved average gate fidelity: {achieved_fidelity:.8f}")

        # 4. Construct the optimized circuit
        optimized_circuit = ansatz_qc.assign_parameters(
            {ansatz_params[i]: optimal_params[i] for i in range(len(optimal_params))}
        )
    else: # No parameters in ansatz, just use the fixed ansatz structure
        print("\nAnsatz has no parameters. Using fixed ansatz structure.")
        min_infidelity = objective_function([], ansatz_qc, ansatz_params, target_ideal_unitary)
        achieved_fidelity = 1.0 - min_infidelity
        print(f"  Infidelity of fixed ansatz (1 - F_avg): {min_infidelity:.6e}")
        print(f"  Average gate fidelity of fixed ansatz: {achieved_fidelity:.8f}")
        optimized_circuit = ansatz_qc


    print("\nOptimized Ansatz Circuit (with found parameters):")
    print(optimized_circuit.draw(output='text'))

    if TARGET_BACKEND_NAME:
        service = None
        target_backend_instance = None
        try:
            print(f"\nConnecting to IBM Quantum to transpile for {TARGET_BACKEND_NAME}...")
            service = QiskitRuntimeService()
            target_backend_instance = service.backend(TARGET_BACKEND_NAME)
            if target_backend_instance:
                print(f"Transpiling optimized circuit for {target_backend_instance.name}...")
                transpiled_optimized_circuit = transpile(optimized_circuit, backend=target_backend_instance, optimization_level=3)
                print("\nTranspiled Optimized Circuit Gate Counts:")
                ops_counts = transpiled_optimized_circuit.count_ops()
                print(ops_counts)
                
                backend_config = target_backend_instance.configuration()
                backend_basis_gates_list = getattr(backend_config, 'basis_gates', [])
                known_2q_gate_names = {'cx', 'ecr', 'cz', 'swap', 'rzz', 'rzx', 'zz'} 
                backend_2q_gates = {gate for gate in backend_basis_gates_list if gate in known_2q_gate_names}
                num_two_qubit_gates = 0
                for gate_name_count, count_val in ops_counts.items():
                    if gate_name_count in backend_2q_gates: # Check if the gate name matches a known 2Q type
                        num_two_qubit_gates += count_val
                print(f"Estimated 2-Qubit native gate count: {num_two_qubit_gates}")

        except Exception as e:
            print(f"Could not transpile for backend {TARGET_BACKEND_NAME}: {e}")
            print("Skipping backend transpilation.")
            
    total_script_duration = time.time() - overall_start_time
    print(f"\nTotal script execution time: {total_script_duration:.2f} seconds.")