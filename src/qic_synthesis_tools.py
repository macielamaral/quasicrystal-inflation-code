# qic_synthesis_tools.py
# Tools for QIC operator analysis, local unitary approximation, and circuit synthesis.

import numpy as np
import time
import traceback
import os
import sys

# --- Try to import qic_core for constants and Qiskit check ---
try:
    import qic_core
    PHI = getattr(qic_core, 'PHI', (1 + np.sqrt(5)) / 2)
    QISKIT_AVAILABLE = getattr(qic_core, 'QISKIT_AVAILABLE', False)
except ImportError:
    print("Warning: qic_core.py not found or PHI/QISKIT_AVAILABLE not defined within it.")
    print("         Defining PHI numerically and attempting to import Qiskit/Scipy directly.")
    PHI = (1 + np.sqrt(5)) / 2
    QISKIT_AVAILABLE = True # Assume Qiskit is available if this module is imported

if QISKIT_AVAILABLE:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter # ParameterVector is not strictly needed for the refined ansatz
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit.providers.exceptions import QiskitBackendNotFoundError
    from qiskit.quantum_info import Operator
    from scipy.optimize import minimize
    from scipy.linalg import eigh, polar # eigh for purification, polar for closest unitary
    from scipy.sparse import csc_matrix
else:
    # This allows the script to be imported to see docstrings, but functions will fail.
    print("CRITICAL ERROR: Qiskit (and Scipy for some functions) is required for qic_synthesis_tools.py but not available.")
    # Define dummy classes or functions if needed for basic import without Qiskit
    class QuantumCircuit: pass
    class Operator: pass
    # etc. or simply let it fail on use.

# ===== IBM Backend Utility =====
def get_ibm_backend(service, backend_name=None, min_qubits=3):
    """
    Gets the specified IBM Quantum backend or the least busy one.

    Args:
        service (QiskitRuntimeService): Authenticated QiskitRuntimeService instance.
        backend_name (str, optional): Name of the target backend. Defaults to None (least busy).
        min_qubits (int, optional): Minimum number of qubits for least_busy. Defaults to 3.

    Returns:
        IBMBackend or None: The backend instance or None if not found/error.
    """
    backend = None
    if not service:
        print("ERROR: QiskitRuntimeService instance not provided to get_ibm_backend.")
        return None
    if backend_name:
        print(f"Attempting to get specified backend: {backend_name}...")
        try:
            backend = service.backend(backend_name)
            print(f"Successfully got backend: {backend.name}")
        except QiskitBackendNotFoundError:
            print(f"ERROR: Backend '{backend_name}' not found or access denied.")
            available_backends = service.backends(simulator=False, operational=True)
            if available_backends:
                print("Available operational backends you have access to:")
                for b_config in available_backends: # Iterate over BackendConfiguration
                    print(f"  - {b_config.name} ({b_config.num_qubits} qubits)")
            else:
                print("No operational backends found.")
            return None
        except Exception as e_service: # Catch other potential service errors
             print(f"ERROR: Could not retrieve backend '{backend_name}' due to service error: {e_service}")
             return None
    else:
        print(f"Finding least busy operational backend with >= {min_qubits} qubits...")
        try:
            backend = service.least_busy(min_num_qubits=min_qubits, operational=True, simulator=False)
        except Exception as e:
            print(f"ERROR: Could not find a least busy backend: {e}")
            return None
    
    if backend:
        backend_config = backend.configuration()
        backend_basis_gates = getattr(backend_config, 'basis_gates', 'N/A')
        print(f"Using backend: {backend.name} (Qubits: {backend.num_qubits}, Basis Gates: {backend_basis_gates})")
    return backend

# ===== Original B'_k Matrix Construction (with Purification) =====
def get_original_Bk_matrix(N, k_idx, verbose=False):
    """
    Computes the original B'_k(N) full matrix (2^N x 2^N) using qic_core functions,
    including purification of the internal P'_k projector.

    Args:
        N (int): Total number of qubits in the chain.
        k_idx (int): Index of the B' operator (0 to N-2).
        verbose (bool): If True, print more debug information.

    Returns:
        np.ndarray or None: The 2^N x 2^N unitary matrix for B'_k(N), or None on error.
    """
    op_name = f"B'_{k_idx}(N={N})"
    try:
        if verbose: print(f"    Constructing {op_name} original matrix...")
        qic_strings_N, qic_vectors_N = qic_core.get_qic_basis(N)
        if not qic_strings_N: raise ValueError(f"Failed QIC basis for {op_name}.")
        
        V_N_iso = qic_core.construct_isometry_V(qic_vectors_N)
        if V_N_iso is None: raise ValueError(f"Failed V_iso for {op_name}.")
        V_N_csc = V_N_iso.tocsc()

        Pk_anyon_orig_sparse = qic_core.get_kauffman_Pn_anyon_general(N, k_idx, qic_strings_N, delta=PHI)
        if Pk_anyon_orig_sparse is None: raise ValueError(f"Failed P_anyon for {op_name}.")
        if verbose and N <= 3: # Only print for small N to avoid large output
             print(f"\n    P_{k_idx}^anyon(N={N}) (from qic_core):\n", Pk_anyon_orig_sparse.toarray() if hasattr(Pk_anyon_orig_sparse, "toarray") else Pk_anyon_orig_sparse)

        Pk_prime_orig_sparse_initial = qic_core.build_P_prime_n(k_idx, N, V_N_csc, Pk_anyon_orig_sparse)
        if Pk_prime_orig_sparse_initial is None: raise ValueError("Failed P'_prime initial.")
        
        P_prime_initial_dense = Pk_prime_orig_sparse_initial.toarray()
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
        
        if verbose and N <= 3:
            print(f"    Purified P'_orig for {op_name} selected {num_selected_eigvals} eigenvalues.")
            # print("\n--- DEBUG: Pk_prime_orig_sparse_purified ---")
            # print(P_prime_purified_dense)
        Pk_prime_orig_sparse_purified = csc_matrix(P_prime_purified_dense)

        Bk_original_sparse = qic_core.build_B_prime_n(k_idx, Pk_prime_orig_sparse_purified, N)
        if Bk_original_sparse is None: raise ValueError("Failed B'_original after purification.")
        Bk_original_dense = Bk_original_sparse.toarray()

        # Final Unitarity check
        identity_N = np.eye(2**N, dtype=complex)
        norm_diff_unitary = np.linalg.norm((Bk_original_dense @ Bk_original_dense.conj().T) - identity_N, 'fro')
        is_unitary_qiskit_check = Operator(Bk_original_dense).is_unitary(atol=1e-8) # Qiskit's check

        if verbose:
            print(f"    || {op_name}_orig @ ({op_name}_orig)^dag - I ||_F = {norm_diff_unitary:.3e}")
            if not is_unitary_qiskit_check:
                print(f"    WARNING: {op_name} constructed is NOT strictly unitary by Qiskit check (atol=1e-8) despite purification!")
            else:
                print(f"    {op_name} constructed IS unitary (atol=1e-8).")
        
        if norm_diff_unitary > 1e-7: # If our own check finds it too non-unitary
             print(f"    CRITICAL WARNING: {op_name} is significantly non-unitary (norm {norm_diff_unitary:.3e}) even after purification. Check P_anyon definition or qic_core functions.")

        return Bk_original_dense
    except Exception as e:
        print(f"    ERROR constructing original {op_name}: {e}")
        if verbose: traceback.print_exc()
        return None

# ===== Functions for Optimal Local Approximator Search =====

def extract_averaged_local_block(U_original_Nk, N, k_active_start_idx):
    """
    Extracts an 8x8 matrix G_avg by averaging B'_k(N) over environment states.
    U_original_Nk: The 2^N x 2^N matrix for B'_k(N).
    k_active_start_idx: The global index of the first qubit in the 3-qubit active block.
    """
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
            
            G_avg[alpha_local_idx, beta_local_idx] = sum_val / num_env_states if num_env_states > 0 else sum_val
            
    return G_avg

def get_closest_unitary(matrix_M, N_val_debug=None, k_op_idx_debug=None):
    """
    Finds the closest unitary matrix to matrix_M using SVD: G_tilde = V @ Wh.
    Includes debug prints for singular values for specific N,k cases.
    """
    if matrix_M is None: return None
    try:
        V_svd, S_svd, Wh_svd = np.linalg.svd(matrix_M) # Wh_svd is W_dagger
        
        # Debug singular values, especially for cases that were problematic
        if N_val_debug is not None and k_op_idx_debug is not None:
            is_problem_case = (N_val_debug == 3 and k_op_idx_debug == 0) # Example focus
            if is_problem_case:
                 print(f"    --- DEBUG SVD inside get_closest_unitary (N={N_val_debug}, k_op={k_op_idx_debug}) ---")
                 print(f"    Singular values of input matrix_M (G_avg): {S_svd}")
                 # Check unitarity of input matrix_M to get_closest_unitary more directly
                 input_M_unitarity_err = np.linalg.norm(matrix_M @ matrix_M.conj().T - np.eye(matrix_M.shape[0]), 'fro')
                 print(f"    Input matrix_M ||MMdag-I||_F = {input_M_unitarity_err:.3e}")

        G_tilde = V_svd @ Wh_svd # V @ W_dagger gives the unitary part
        return G_tilde
    except Exception as e:
        print(f"    Error during SVD-based closest unitary computation: {e}")
        # traceback.print_exc() # Uncomment for full traceback during debugging
        return None

def construct_global_approx_operator_matrix(G_local_8x8, N, k_active_start_idx):
    """
    Constructs the 2^N x 2^N matrix for G_local_8x8 @ I_{N-3},
    where G_local_8x8 acts on qubits k_active_start_idx, k+1, k+2.
    """
    if G_local_8x8 is None:
        print("Error: G_local_8x8 is None in construct_global_approx_operator_matrix.")
        return None
    if k_active_start_idx < 0 or k_active_start_idx + 3 > N :
        print(f"Error: Cannot place 3-qubit gate starting at {k_active_start_idx} in N={N} system.")
        return None
    try:
        qc = QuantumCircuit(N)
        target_qubits = [k_active_start_idx, k_active_start_idx + 1, k_active_start_idx + 2]
        qc.unitary(G_local_8x8, target_qubits)
        return Operator(qc).data
    except Exception as e:
        print(f"    Error during construction of global approx operator matrix for N={N}, k_start={k_active_start_idx}: {e}")
        return None

# ===== Variational Circuit Synthesis Functions =====

def create_3q_ansatz(num_layers, num_rot_params_per_qubit_per_layer):
    """
    Creates a 3-qubit ansatz with specified number of layers.
    Each layer: 3 RZ params per qubit, plus fixed ECRs.
    """
    num_qubits = 3
    total_rotation_params_needed = num_layers * num_rot_params_per_qubit_per_layer * num_qubits

    qc = QuantumCircuit(num_qubits)
    # Create unique Parameter objects that will be bound to the circuit
    # Their names will be p_0, p_1, ...
    parameter_list = [Parameter(f'p_{i}') for i in range(total_rotation_params_needed)]
    param_idx = 0

    for l_idx in range(num_layers):
        # Single-qubit rotations: RZ(p1) SX RZ(p2) SX RZ(p3) for each qubit
        for q_idx in range(num_qubits):
            # Ensure we have enough parameters for the RZ-SX-RZ-SX-RZ sequence (3 params)
            if param_idx + 2 < total_rotation_params_needed : 
                qc.rz(parameter_list[param_idx], q_idx); param_idx += 1
                qc.sx(q_idx)
                qc.rz(parameter_list[param_idx], q_idx); param_idx += 1
                qc.sx(q_idx)
                qc.rz(parameter_list[param_idx], q_idx); param_idx += 1
            elif param_idx < total_rotation_params_needed: # Use remaining params if any
                print(f"Warning in create_3q_ansatz: Layer {l_idx}, Qubit {q_idx}: Not enough params for full RZ-SX-RZ-SX-RZ. Using fewer.")
                qc.rz(parameter_list[param_idx], q_idx); param_idx += 1
                if param_idx < total_rotation_params_needed: # One more param?
                    qc.sx(q_idx)
                    qc.rz(parameter_list[param_idx], q_idx); param_idx += 1
                break # Not enough for full sequence on this qubit
            else: # No parameters left
                break # Break from q_idx loop
        if param_idx >= total_rotation_params_needed and l_idx < num_layers -1 :
             break # Break from l_idx loop if all rotation parameters are used up early

        # Entangling layer (fixed ECR pattern)
        qc.ecr(0, 1)
        qc.ecr(1, 2)
        if l_idx % 2 == 0 : # Add another ECR for more entanglement every other layer
             qc.ecr(0,2) 
    
    if param_idx < total_rotation_params_needed:
         print(f"Warning in create_3q_ansatz: Expected to use {total_rotation_params_needed} rotation parameters, but structure only placed {param_idx}")
    
    # The actual parameters are those bound to the circuit.
    # `qc.parameters` returns a set of unique Parameter objects.
    return qc # The circuit template; its parameters are qc.parameters

def objective_function(param_values_array, ansatz_circuit_template, unbound_params_list_sorted, target_unitary_matrix_np):
    """
    Calculates infidelity: 1.0 - AverageGateFidelity(target, ansatz_unitary).
    unbound_params_list_sorted: a sorted list of Parameter objects from the ansatz.
    """
    if len(param_values_array) != len(unbound_params_list_sorted):
        print(f"ERROR in objective_function: Mismatch len(param_values_array)={len(param_values_array)} vs len(unbound_params_list_sorted)={len(unbound_params_list_sorted)}")
        return 2.0 # High infidelity to penalize this
        
    param_dict = {unbound_params_list_sorted[i]: param_values_array[i] for i in range(len(param_values_array))}
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

def synthesize_ideal_gate(target_ideal_unitary_matrix, num_ansatz_layers, params_per_single_qubit_rot_seq, max_opt_iters, gate_name="G_ideal"):
    """
    Takes an 8x8 target ideal unitary, synthesizes it using variational ansatz.
    Returns the optimized Qiskit QuantumCircuit and its achieved fidelity.
    """
    print(f"\n--- Synthesizing Ideal Local Gate: {gate_name} ---")
    if not isinstance(target_ideal_unitary_matrix, np.ndarray) or target_ideal_unitary_matrix.shape != (8,8):
        print(f"ERROR: Target matrix for {gate_name} is not an 8x8 numpy array. Shape: {target_ideal_unitary_matrix.shape if isinstance(target_ideal_unitary_matrix, np.ndarray) else type(target_ideal_unitary_matrix)}")
        return None, 0.0
    if not Operator(target_ideal_unitary_matrix).is_unitary(atol=1e-7): # Check target
        print(f"WARNING: Target matrix {gate_name} itself is not strictly unitary! Synthesis may be poor.")
        print(f"  ||UUdag-I||_F for target {gate_name}: {np.linalg.norm(target_ideal_unitary_matrix @ target_ideal_unitary_matrix.conj().T - np.eye(8), 'fro'):.3e}")


    print(f"Target matrix for {gate_name} loaded. Shape: {target_ideal_unitary_matrix.shape}")

    ansatz_qc_template = create_3q_ansatz(num_ansatz_layers, params_per_single_qubit_rot_seq)
    
    # Get actual Parameter objects present in the circuit template and sort them for consistency
    unbound_params_in_circuit = sorted(list(ansatz_qc_template.parameters), key=lambda p: p.name)
    num_actual_params = len(unbound_params_in_circuit)
    
    print(f"Using {num_ansatz_layers}-layer ansatz with {num_actual_params} unique parameters for {gate_name}.")

    if num_actual_params == 0: 
        print(f"Ansatz for {gate_name} has no parameters (e.g. 0 layers or no params in structure). Evaluating fixed ansatz structure.")
        # If no params, objective_function needs empty list for param_values and unbound_params
        infidelity = objective_function(np.array([]), ansatz_qc_template, [], target_ideal_unitary_matrix)
        achieved_fidelity = 1.0 - infidelity
        print(f"  Fidelity of fixed (no-param) ansatz for {gate_name}: {achieved_fidelity:.6f}")
        return ansatz_qc_template, achieved_fidelity 

    initial_params = np.random.rand(num_actual_params) * 2 * np.pi
    
    print(f"Starting optimization for {gate_name} ({num_actual_params} params)...")
    optimization_start_time = time.time()
    res = minimize(objective_function, 
                   initial_params, 
                   args=(ansatz_qc_template, unbound_params_in_circuit, target_ideal_unitary_matrix), 
                   method='L-BFGS-B', 
                   options={'maxiter': max_opt_iters, 'disp': False, 'eps': 1e-9, 'ftol': 1e-10, 'gtol': 1e-7}) # Added ftol, gtol 
    optimization_duration = time.time() - optimization_start_time
    print(f"Optimization for {gate_name} finished in {optimization_duration:.2f}s.")

    if not res.success:
        print(f"WARNING: Optimization for {gate_name} not fully converged: {res.message}")
    
    min_infidelity = res.fun
    achieved_fidelity = 1.0 - min_infidelity
    print(f"  Synthesis for {gate_name} done. Optimized Fidelity: {achieved_fidelity:.8f} (Infidelity: {min_infidelity:.6e})")
    # print(f"  Optimal parameters found: {res.x}") # Can be very long

    optimal_param_dict = {unbound_params_in_circuit[i]: res.x[i] for i in range(len(res.x))}
    optimized_circuit = ansatz_qc_template.assign_parameters(optimal_param_dict)
    return optimized_circuit, achieved_fidelity



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


# Example usage (if this file is run directly, mainly for testing the tools)
if __name__ == "__main__":
    print("This is qic_synthesis_tools.py - usually imported as a module.")
    print(f"PHI constant is defined as: {PHI}")
    if not QISKIT_AVAILABLE:
        print("Qiskit not available, cannot run tests from here.")
    else:
        print("Testing ansatz creation (1 layer, 3 rot params per qubit):")
        qc_test_ansatz = create_3q_ansatz(1, 3)
        print(qc_test_ansatz.draw(output='text'))
        print(f"Parameters in test ansatz: {[p.name for p in qc_test_ansatz.parameters]}")

        # Example test of synthesize_ideal_gate with a random unitary
        print("\nTesting synthesize_ideal_gate with a random 3Q unitary:")
        random_3q_unitary = Operator.random(8, seed=42).data 
        # Ensure it's unitary for the test
        U_rand, _ = polar(random_3q_unitary)

        synthesized_random_gate_circuit, achieved_fid = synthesize_ideal_gate(
            U_rand, 
            num_ansatz_layers=2, 
            params_per_single_qubit_rot_seq=3, 
            max_opt_iters=50, # Shorter for a quick test
            gate_name="RandomTest"
        )
        if synthesized_random_gate_circuit:
            print("Synthesized circuit for random target:")
            print(synthesized_random_gate_circuit.draw(output='text'))
            print(f"Achieved fidelity for random target: {achieved_fid:.6f}")

            # Example of getting original B0 matrix for N=3
            print("\nTesting get_original_Bk_matrix for N=3, k=0:")
            b0_n3_orig = get_original_Bk_matrix(N=3, k_idx=0, verbose=True)
            if b0_n3_orig is not None:
                print(f"Shape of B'_0(N=3)_orig: {b0_n3_orig.shape}")
                # Further analysis could be done here
