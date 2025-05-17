# run_qic_hardware_resource_sweep_eff.py
# Estimates resources for QIC braid operators B'_k (EFFICIENT METHOD)
# for various N_QUBITS on a specified IBM Quantum hardware backend.

import numpy as np
import time
import traceback
import csv
import os
import sys # For sys.path modifications if needed

# --- Ensure qic_core is importable ---
try:
    import qic_core
except ImportError:
    try:
        # Basic attempt if src is a direct sibling or on path from execution dir
        # For more complex structures, sys.path.append might be needed before this.
        # Example: If this script is in 'simulations/' and qic_core is in 'src/'
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # src_dir = os.path.join(os.path.dirname(current_dir), 'src')
        # if src_dir not in sys.path:
        #    sys.path.insert(0, src_dir)
        import qic_core
        print("Successfully imported qic_core (path might have been adjusted).")
    except ImportError as e:
        print("ERROR: Failed to import qic_core.py.")
        print("       Make sure qic_core.py is in 'src/' and 'src/' parent is in sys.path,")
        print("       or qic_core.py is in the same directory or Python path.")
        print(f"       Original error: {e}")
        exit()

# Import Qiskit components
if qic_core.QISKIT_AVAILABLE:
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit.providers.exceptions import QiskitBackendNotFoundError
    from qiskit.quantum_info import Operator # For verification if needed
else:
    print("ERROR: Qiskit and QiskitRuntimeService are required for this script but not available.")
    exit()

# ===== Configuration =====
N_QUBITS_LIST = [3, 4, 5, 6, 7, 8, 9, 10] 
TARGET_BACKEND_NAME = "ibm_brisbane"  
OPTIMIZATION_LEVEL = 3
RESULTS_CSV_FILE = 'data/qic_hardware_resource_results_efficient.csv'
# =======================

def get_ibm_backend(service, backend_name=None, min_qubits=3):
    backend = None
    if backend_name:
        print(f"Attempting to get specified backend: {backend_name}...")
        try:
            backend = service.backend(backend_name)
        except QiskitBackendNotFoundError:
            print(f"ERROR: Backend '{backend_name}' not found or access denied.")
            print("Available backends you have access to:")
            for b_config in service.backends():
                if not b_config.simulator: print(f"  - {b_config.name} ({b_config.num_qubits} qubits)")
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

def compute_G_typeA_unitary(): # Formerly compute_local_B_prime_unitary
    """
    Computes the local 3-qubit unitary G_typeA (representing B'_0 type,
    derived from B'_0 for N=3 system).
    """
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
        ) # Corrected to match common signature patterns if positional
        if P_prime_N3_k0_sparse is None: raise ValueError("Failed P'_local N=3 for G_typeA.")

        G_sparse = qic_core.build_B_prime_n(
            k_idx_local, P_prime_N3_k0_sparse, N_local
        ) # Corrected to match common signature patterns if positional
        if G_sparse is None: raise ValueError("Failed B'_local N=3 for G_typeA.")
        
        G_dense = G_sparse.toarray() 
        identity_8x8 = np.eye(2**N_local, dtype=complex)
        if np.allclose(G_dense @ G_dense.conj().T, identity_8x8, atol=1e-8):
            print("G_typeA (B'_0 for N=3) is unitary. Dimensions: ", G_dense.shape)
        else:
            print("WARNING: Computed G_typeA is NOT unitary.")
        return G_dense
    except Exception as e:
        print(f"ERROR during G_typeA computation: {e}")
        traceback.print_exc()
        return None

def compute_G_typeB_unitary(P1_anyon_N3_dense_input):
    """
    Computes G_typeB (representing B'_{N-2} type), derived from B'_1 for N=3 system.
    """
    print("\n--- Computing Local 3-Qubit Unitary G_typeB (from B'_1 for N=3) ---")
    N_local = 3
    k_idx_B_type = 1 

    try:
        qic_strings_N3, qic_vectors_N3 = qic_core.get_qic_basis(N_local)
        if not qic_strings_N3: raise ValueError("Failed QIC basis N=3 for G_typeB.")
        
        V_N3_iso = qic_core.construct_isometry_V(qic_vectors_N3)
        if V_N3_iso is None: raise ValueError("Failed V_iso N=3 for G_typeB.")
        V_N3_csc = V_N3_iso.tocsc()

        from scipy.sparse import csc_matrix
        P1_anyon_N3_sparse = csc_matrix(P1_anyon_N3_dense_input)

        P_prime_for_G_typeB_sparse = qic_core.build_P_prime_n(
            k_idx_B_type, N_local, V_N3_csc, P1_anyon_N3_sparse
        ) # Corrected to match common signature patterns if positional
        if P_prime_for_G_typeB_sparse is None: raise ValueError("Failed P'_prime for G_typeB.")

        # Checking P'_prime properties for G_typeB construction
        #P_prime_dense = P_prime_for_G_typeB_sparse.toarray()
        #print("Checking P'_prime properties for G_typeB construction:")
        #is_idempotent = np.allclose(P_prime_dense @ P_prime_dense, P_prime_dense, atol=1e-9)
        #is_hermitian = np.allclose(P_prime_dense, P_prime_dense.conj().T, atol=1e-9)
        #print(f"  P'_prime for G_typeB is idempotent? {is_idempotent}")
        #print(f"  P'_prime for G_typeB is Hermitian? {is_hermitian}")
        #if not (is_idempotent and is_hermitian):
        #    print("  WARNING: P'_prime used for G_typeB is not a good numerical projector!")
            # Further diagnostics:
            # eigvals = np.linalg.eigvalsh(P_prime_dense) # Use eigvalsh for Hermitian
            # print(f"  Eigenvalues of P'_prime for G_typeB: {np.sort(eigvals)}")
            # Expected eigenvalues are 0s and 1s.

        G_sparse = qic_core.build_B_prime_n(
            k_idx_B_type, P_prime_for_G_typeB_sparse, N_local
        ) # Corrected to match common signature patterns if positional
        if G_sparse is None: raise ValueError("Failed G_typeB.")
        
        G_dense = G_sparse.toarray()
        identity_8x8 = np.eye(2**N_local, dtype=complex)
        if np.allclose(G_dense @ G_dense.conj().T, identity_8x8, atol=1e-8):
            print("G_typeB is unitary. Dimensions: ", G_dense.shape)
        else:
            print("WARNING: Computed G_typeB is NOT unitary.")
        return G_dense
    except Exception as e:
        print(f"ERROR during G_typeB computation: {e}")
        traceback.print_exc()
        return None

def run_resource_estimation_for_n_efficient(N, G_A, G_B, target_backend): # G_A for left/mid, G_B for right
    N_RESULTS = []
    if N < 3: 
         print(f"N={N} is too small for 3-qubit local gates G_A/G_B. Skipping.")
         return N_RESULTS

    num_operators = N - 1 # B'_0 to B'_{N-2}
    print(f"\n--- Efficient Transpiling for N_QUBITS = {N} (using local gates) ---")
    
    for k_idx in range(num_operators): 
        op_name = f"B'_{k_idx}"
        print(f"\n-- Processing {op_name} --")
        
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
        
        try:
            qc = QuantumCircuit(N, name=op_name)
            qc.unitary(current_G_local, target_qubits, label=f"G({k_idx})")
            
            print(f"Transpiling {op_name} (local G on {target_qubits}) for {target_backend.name} (Opt Level: {OPTIMIZATION_LEVEL})...")
            start_transpile_time = time.time()
            transpiled_qc = transpile(qc, backend=target_backend, optimization_level=OPTIMIZATION_LEVEL)
            end_transpile_time = time.time()
            transpile_duration = end_transpile_time - start_transpile_time
            print(f"Transpilation finished ({transpile_duration:.3f} s).")

            depth = transpiled_qc.depth()
            ops_counts = transpiled_qc.count_ops()
            total_ops_count = sum(ops_counts.values())
            
            backend_config = target_backend.configuration()
            backend_basis_gates_list = getattr(backend_config, 'basis_gates', [])
            known_2q_gate_names = {'cx', 'ecr', 'cz', 'swap', 'rzz', 'rzx', 'zz'} 
            backend_2q_gates = {gate for gate in backend_basis_gates_list if gate in known_2q_gate_names}

            num_two_qubit_gates = 0
            types_two_qubit_gates = set()
            for gate_name_count, count_val in ops_counts.items():
                if gate_name_count in backend_2q_gates:
                    num_two_qubit_gates += count_val
                    types_two_qubit_gates.add(gate_name_count)
            
            print(f"  {op_name} - Transpiled Depth: {depth}")
            print(f"  {op_name} - Operations Count: {ops_counts}")
            print(f"  {op_name} - Total Primitive Ops: {total_ops_count}")
            print(f"  {op_name} - Estimated 2-Qubit Gate Count: {num_two_qubit_gates} (Types: {types_two_qubit_gates if types_two_qubit_gates else 'None Found'})")

            N_RESULTS.append({
                'N_QUBITS': N, 'OPERATOR': op_name, 'DEPTH': depth,
                'TOTAL_OPS': total_ops_count, '2Q_GATE_COUNT': num_two_qubit_gates,
                '2Q_GATE_TYPES': ', '.join(sorted(list(types_two_qubit_gates))),
                'OPS_COUNTS': str(ops_counts), 'TRANSPILATION_TIME_S': f"{transpile_duration:.3f}"
            })

        except Exception as e:
            print(f"ERROR during processing for {op_name} at N={N}: {e}")
            traceback.print_exc()
            N_RESULTS.append({
                'N_QUBITS': N, 'OPERATOR': op_name, 'DEPTH': 'Error',
                'TOTAL_OPS': 'Error', '2Q_GATE_COUNT': 'Error', '2Q_GATE_TYPES': 'Error',
                'OPS_COUNTS': str(e), 'TRANSPILATION_TIME_S': 'Error'
            })
    return N_RESULTS

if __name__ == "__main__":
    master_start_time = time.time()
    all_results = []

    results_dir = os.path.dirname(RESULTS_CSV_FILE)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")

    if not os.path.exists(RESULTS_CSV_FILE) or os.path.getsize(RESULTS_CSV_FILE) == 0:
        with open(RESULTS_CSV_FILE, 'w', newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(['N_QUBITS', 'OPERATOR', 'DEPTH', 'TOTAL_OPS', 
                             '2Q_GATE_COUNT', '2Q_GATE_TYPES', 'OPS_COUNTS', 'TRANSPILATION_TIME_S'])
        print(f"Created results file: {RESULTS_CSV_FILE}")
    else:
        print(f"Appending to existing results file: {RESULTS_CSV_FILE}")

    service = None
    target_backend_instance = None
    try:
        print("\nConnecting to IBM Quantum Runtime Service...")
        service = QiskitRuntimeService() 
        print("Connected to IBM Quantum.")
        min_q_for_sweep = 3 
        if N_QUBITS_LIST: min_q_for_sweep = max(3, max(N_QUBITS_LIST)) # Ensure at least 3 for G_local
            
        target_backend_instance = get_ibm_backend(service, TARGET_BACKEND_NAME, min_qubits=min_q_for_sweep)
        if target_backend_instance is None:
            print("Could not get a target backend. Exiting.")
            exit()
    except Exception as e:
        print(f"ERROR connecting to IBM Quantum or getting backend: {e}")
        traceback.print_exc()
        exit()
    
    G_typeA_unitary = compute_G_typeA_unitary()
    if G_typeA_unitary is None:
        print("Failed to compute G_typeA_unitary. Exiting.")
        exit()

    PHI = qic_core.PHI # Or (1 + np.sqrt(5)) / 2 directly
    val_diag_1 = 1.0
    val_diag_2 = 1/PHI**2 # Approx 0.382
    val_diag_3 = 1/PHI   # Approx 0.618
    val_offdiag = 1/(PHI * np.sqrt(PHI)) # Approx 0.486


    P1_anyon_N3_data = np.array([
    [val_diag_1, 0.        , 0.        , 0.        , 0.        ],
    [0.        , 0.        , 0.        , 0.        , 0.        ],
    [0.        , 0.        , val_diag_2, 0.        , val_offdiag],
    [0.        , 0.        , 0.        , 0.        , 0.        ],
    [0.        , 0.        , val_offdiag, 0.        , val_diag_3]
], dtype=complex)

    G_typeB_unitary = compute_G_typeB_unitary(P1_anyon_N3_data)
    if G_typeB_unitary is None:
        print("Failed to compute G_typeB_unitary. Exiting.")
        exit()
    
    for N_val in N_QUBITS_LIST:
        if N_val > target_backend_instance.num_qubits:
            print(f"Skipping N_QUBITS={N_val} as it exceeds backend max qubits ({target_backend_instance.num_qubits}).")
            continue
        if N_val < 3: 
            print(f"Skipping N_QUBITS={N_val} as it's < 3 (method uses 3-qubit local gates).")
            continue
            
        print(f"\n===== Processing for N_QUBITS = {N_val} (Efficient Method) =====")
        
        results_for_N = run_resource_estimation_for_n_efficient(
            N_val, G_typeA_unitary, G_typeB_unitary, target_backend_instance
        )
        # all_results.extend(results_for_N) # Removed to avoid large memory use if not needed

        with open(RESULTS_CSV_FILE, 'a', newline='') as f_csv:
            writer = csv.writer(f_csv)
            for res_dict in results_for_N:
                writer.writerow([res_dict['N_QUBITS'], res_dict['OPERATOR'], res_dict['DEPTH'], 
                                 res_dict['TOTAL_OPS'], res_dict['2Q_GATE_COUNT'],
                                 res_dict['2Q_GATE_TYPES'], res_dict['OPS_COUNTS'],
                                 res_dict['TRANSPILATION_TIME_S']])
        print(f"Results for N_QUBITS={N_val} (Efficient Method) appended to {RESULTS_CSV_FILE}")

    print("\n--- QIC Hardware Resource Sweep (Efficient Method) Finished ---")
    total_duration = time.time() - master_start_time
    print(f"Total execution time: {total_duration:.3f} seconds")
    print(f"All results saved to {RESULTS_CSV_FILE}")