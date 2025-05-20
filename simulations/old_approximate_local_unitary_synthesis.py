# approximate_local_unitary_synthesis.py
# Finds a local circuit approximation for an ideal 3-qubit unitary (e.g., G_typeA)
# using a parameterized ansatz and optimization.

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
        # Example: If this script is in 'simulations/' and qic_core is in 'src/'
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
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit_ibm_runtime import QiskitRuntimeService # Optional: for backend transpilation
    from qiskit.providers.exceptions import QiskitBackendNotFoundError # Optional
    from qiskit.quantum_info import Operator
    from scipy.linalg import eigh # For purification, or np.linalg.eigh
    from scipy.sparse import csc_matrix
    from scipy.optimize import minimize
else:
    print("ERROR: Qiskit (and Scipy) is required for this script but not available.")
    exit()

# ===== Configuration =====
# Target ideal unitary to approximate (e.g., 'G_typeA' or 'G_typeB')
TARGET_GATE_TO_APPROXIMATE = 'G_typeA' 

# Ansatz Configuration
NUM_ANSATZ_LAYERS = 2  # Number of layers in the ansatz
# Parameters per layer: 3 params per U3 gate (RZ-SX-RZ-SX-RZ like) on each of 3 qubits = 3*3=9
PARAMS_PER_SINGLE_QUBIT_LAYER = 3 * 3 
# We'll use fixed ECRs, so params per layer mainly from single-qubit gates.
# If you add parameterized entanglers, increase this.
PARAMS_PER_LAYER = PARAMS_PER_SINGLE_QUBIT_LAYER 

# Optimization Configuration
MAX_OPTIMIZATION_ITERATIONS = 200

# Optional: Backend for final transpilation of the optimized ansatz
TARGET_BACKEND_NAME = "ibm_brisbane" 
# TARGET_BACKEND_NAME = None # Set to None if not transpiling or using default simulator

# Define PHI if not directly available from qic_core (though it should be)
PHI = (1 + np.sqrt(5)) / 2 if not hasattr(qic_core, 'PHI') else qic_core.PHI
# =======================


# --- Functions to compute G_typeA and G_typeB (adapted from verify_b_prime_equivalence.py) ---
# These will provide the target 8x8 unitary matrices we want to approximate.
def compute_G_type_unitary(gate_type_name, k_idx_for_N3_rules, P_anyon_N3_5x5_data=None):
    """
    Computes a local 3-qubit unitary G_type (e.g., G_typeA from B'_0(N=3) or G_typeB from B'_1(N=3)).
    k_idx_for_N3_rules: 0 for G_typeA (P_0^anyon), 1 for G_typeB (P_1^anyon).
    P_anyon_N3_5x5_data: The 5x5 numpy array for P_k^anyon(N=3) if k_idx_for_N3_rules corresponds to it.
                         If None (for k_idx=0), it will be computed.
    Includes projector purification.
    """
    print(f"\n--- Computing Ideal Local 3-Qubit Unitary {gate_type_name} (from B'_{k_idx_for_N3_rules} for N=3) ---")
    N_local = 3
    
    try:
        qic_strings_N3, qic_vectors_N3 = qic_core.get_qic_basis(N_local)
        if not qic_strings_N3: raise ValueError(f"Failed QIC basis N=3 for {gate_type_name}.")
        
        V_N3_iso = qic_core.construct_isometry_V(qic_vectors_N3)
        if V_N3_iso is None: raise ValueError(f"Failed V_iso N=3 for {gate_type_name}.")
        V_N3_csc = V_N3_iso.tocsc()

        if P_anyon_N3_5x5_data is not None:
            P_anyon_N3_k_sparse = csc_matrix(P_anyon_N3_5x5_data)
        else: # Compute P_anyon for k_idx_for_N3_rules (typically k=0 for G_typeA)
            P_anyon_N3_k_sparse = qic_core.get_kauffman_Pn_anyon_general(
                N_local, k_idx_for_N3_rules, qic_strings_N3, delta=PHI
            )
        if P_anyon_N3_k_sparse is None: raise ValueError(f"Failed P_anyon N=3,k={k_idx_for_N3_rules} for {gate_type_name}.")

        P_prime_initial_sparse = qic_core.build_P_prime_n(
            k_idx_for_N3_rules, N_local, V_N3_csc, P_anyon_N3_k_sparse
        )
        if P_prime_initial_sparse is None: raise ValueError(f"Failed initial P'_prime for {gate_type_name}.")
        P_prime_initial_dense = P_prime_initial_sparse.toarray()

        print(f"Initial P'_prime (for {gate_type_name}) properties before purification:")
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
        
        P_prime_for_B_build_sparse = csc_matrix(P_prime_purified_dense)

        G_sparse = qic_core.build_B_prime_n(
            k_idx_for_N3_rules, P_prime_for_B_build_sparse, N_local
        )
        if G_sparse is None: raise ValueError(f"Failed {gate_type_name} construction with purified P'.")
        
        G_dense = G_sparse.toarray()
        identity_8x8 = np.eye(2**N_local, dtype=complex)
        is_unitary_by_numpy = np.allclose(G_dense @ G_dense.conj().T, identity_8x8, atol=1e-8)
        print(f"{gate_type_name} (from purified P') is unitary (by np.allclose)? {is_unitary_by_numpy}. Dimensions: {G_dense.shape}")
        if not is_unitary_by_numpy:
            diff_norm = np.linalg.norm((G_dense @ G_dense.conj().T) - identity_8x8)
            print(f"  Norm of (G G_dag - I) for {gate_type_name}: {diff_norm:.3e}")
            print(f"  WARNING: {gate_type_name} computed as NOT unitary by numpy check.")
            return None
        return G_dense
    except Exception as e:
        print(f"ERROR during {gate_type_name} computation: {e}")
        traceback.print_exc()
        return None

def create_3q_ansatz(num_layers, num_params_per_single_qubit_layer):
    """
    Creates a 3-qubit ansatz with specified number of layers.
    Each layer consists of single-qubit rotations (RZ-SX-RZ-SX-RZ like) and fixed ECR entanglers.
    """
    num_qubits = 3
    total_params_needed = num_layers * num_params_per_single_qubit_layer
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector('θ', total_params_needed)
    param_idx = 0

    for l_idx in range(num_layers):
        # Single-qubit rotations: RZ(p1) SX RZ(p2) SX RZ(p3) for each qubit
        for q_idx in range(num_qubits):
            if param_idx + 3 > total_params_needed: # Should not happen if params_per_layer is correct
                raise ValueError("Not enough parameters for single qubit rotations in ansatz.")
            qc.rz(params[param_idx], q_idx); param_idx += 1
            qc.sx(q_idx)
            qc.rz(params[param_idx], q_idx); param_idx += 1
            qc.sx(q_idx)
            qc.rz(params[param_idx], q_idx); param_idx += 1
        
        # Entangling layer (fixed ECR pattern for simplicity)
        if num_layers > 0 and l_idx < num_layers : # Apply entanglers in each layer (or less frequently)
            qc.ecr(0, 1)
            qc.ecr(1, 2)
            if num_qubits > 2 and l_idx % 2 == 0 : # Example: add another ECR for more entanglement
                 qc.ecr(0,2)


    if param_idx != total_params_needed:
         print(f"Warning: Ansatz created for {total_params_needed} parameters, but structure used {param_idx}")
    return qc, params

def objective_function(param_values, ansatz_circuit_template, unbound_params_vector, target_unitary_matrix_np):
    """
    Calculates infidelity: 1.0 - AverageGateFidelity(target, ansatz_unitary).
    """
    # Assign current parameter values to the ansatz circuit template
    # Ensure parameters are correctly mapped if unbound_params_vector is a ParameterVector list
    param_dict = {unbound_params_vector[i]: param_values[i] for i in range(len(param_values))}
    bound_circuit = ansatz_circuit_template.assign_parameters(param_dict)
    
    try:
        ansatz_unitary_op = Operator(bound_circuit)
        ansatz_unitary_np = ansatz_unitary_op.data 
    except Exception as e:
        print(f"Error getting unitary from ansatz during optimization: {e}")
        return 2.0 # Return a large infidelity value on error to steer optimizer away

    # Average Gate Fidelity F_avg = |Tr(U_target_dag @ U_ansatz)|^2 / d^2
    # where d is the dimension (2^num_qubits)
    d_squared = (2**ansatz_circuit_template.num_qubits)**2
    
    # Ensure target_unitary_matrix_np is complex, as is ansatz_unitary_np
    trace_val = np.trace(np.conj(target_unitary_matrix_np).T @ ansatz_unitary_np)
    fidelity = (np.abs(trace_val)**2) / d_squared
    
    infidelity = 1.0 - fidelity
    return infidelity

# --- Main execution block ---
if __name__ == "__main__":
    overall_start_time = time.time()
    print("Starting Approximate Local Unitary Synthesis...")

    # 1. Compute the target ideal local unitary (e.g., G_typeA)
    if TARGET_GATE_TO_APPROXIMATE == 'G_typeA':
        target_ideal_unitary = compute_G_type_unitary("G_typeA", k_idx_for_N3_rules=0)
    elif TARGET_GATE_TO_APPROXIMATE == 'G_typeB':
        P1_anyon_N3_data = np.array([ # Data for P_1^anyon(N=3) from previous user input
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0/(PHI**2), 0.0, 1.0/(PHI*np.sqrt(PHI))],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0/(PHI*np.sqrt(PHI)), 0.0, 1.0/PHI]
        ], dtype=complex)
        target_ideal_unitary = compute_G_type_unitary("G_typeB", k_idx_for_N3_rules=1, P_anyon_N3_5x5_data=P1_anyon_N3_data)
    else:
        print(f"Error: Unknown TARGET_GATE_TO_APPROXIMATE: {TARGET_GATE_TO_APPROXIMATE}")
        exit()

    if target_ideal_unitary is None:
        print(f"Critical error: Failed to compute target ideal unitary {TARGET_GATE_TO_APPROXIMATE}. Exiting.")
        exit()

    # 2. Create ansatz
    ansatz_qc, ansatz_params = create_3q_ansatz(NUM_ANSATZ_LAYERS, PARAMS_PER_LAYER)
    print(f"\nAnsatz Circuit ({NUM_ANSATZ_LAYERS} layer(s), {len(ansatz_params)} params):")
    print(ansatz_qc.draw(output='text'))

    if len(ansatz_params) == 0:
        print("Error: Ansatz has no parameters. Check create_3q_ansatz function.")
        exit()
        
    # 3. Perform optimization
    initial_params = np.random.rand(len(ansatz_params)) * 2 * np.pi # Random initial parameters
    
    print(f"\nStarting optimization to approximate {TARGET_GATE_TO_APPROXIMATE}...")
    print(f"Number of parameters to optimize: {len(initial_params)}")
    
    optimization_start_time = time.time()
    res = minimize(objective_function, 
                   initial_params, 
                   args=(ansatz_qc, ansatz_params, target_ideal_unitary), 
                   method='L-BFGS-B', 
                   options={'maxiter': MAX_OPTIMIZATION_ITERATIONS, 'disp': True, 'eps': 1e-7}) # eps for gradient step size
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

    # 4. Construct and optionally transpile the optimized circuit
    optimized_circuit = ansatz_qc.assign_parameters(
        {ansatz_params[i]: optimal_params[i] for i in range(len(optimal_params))}
    )
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
                print("\nTranspiled Optimized Circuit:")
                # print(transpiled_optimized_circuit.draw(output='text')) # Can be very long
                ops_counts = transpiled_optimized_circuit.count_ops()
                print("Ops counts for transpiled optimized circuit:", ops_counts)
                
                backend_config = target_backend_instance.configuration()
                backend_basis_gates_list = getattr(backend_config, 'basis_gates', [])
                known_2q_gate_names = {'cx', 'ecr', 'cz', 'swap', 'rzz', 'rzx', 'zz'} 
                backend_2q_gates = {gate for gate in backend_basis_gates_list if gate in known_2q_gate_names}
                num_two_qubit_gates = 0
                for gate_name_count, count_val in ops_counts.items():
                    if gate_name_count in backend_2q_gates:
                        num_two_qubit_gates += count_val
                print(f"Estimated 2-Qubit (ECR/CX etc.) gate count: {num_two_qubit_gates}")

        except Exception as e:
            print(f"Could not transpile for backend {TARGET_BACKEND_NAME}: {e}")
            print("Skipping backend transpilation.")
            
    total_script_duration = time.time() - overall_start_time
    print(f"\nTotal script execution time: {total_script_duration:.2f} seconds.")