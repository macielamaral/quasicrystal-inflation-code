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
    # Example: If qic_core.py is in a 'src' directory at the same level as this script's parent
    # current_script_path = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.join(current_script_path, '..') # Adjust if structure is different
    # src_path = os.path.join(project_root, 'src')
    # if src_path not in sys.path:
    #     sys.path.insert(0, src_path)
    try:
        import qic_core
        print("Successfully imported qic_core (check sys.path if it was modified here).")
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
N_QUBITS_LIST = [3, 4, 5, 6, 7, 8, 9, 10] # List of N_QUBITS to sweep through
                                       # Min N=3 for a 3-qubit local operator.
TARGET_BACKEND_NAME = "ibm_brisbane"   # Or your preferred backend, or None for least_busy
OPTIMIZATION_LEVEL = 3
RESULTS_CSV_FILE = 'data/qic_hardware_resource_results_efficient.csv'
# =======================

def get_ibm_backend(service, backend_name=None, min_qubits=3):
    """Gets the specified backend or the least busy one."""
    backend = None
    if backend_name:
        print(f"Attempting to get specified backend: {backend_name}...")
        try:
            backend = service.backend(backend_name)
        except QiskitBackendNotFoundError:
            print(f"ERROR: Backend '{backend_name}' not found or access denied.")
            print("Available backends you have access to:")
            for b in service.backends():
                if not b.simulator: print(f"  - {b.name} ({b.num_qubits} qubits)")
            return None
    else:
        print(f"Finding least busy operational backend with >= {min_qubits} qubits...")
        try:
            backend = service.least_busy(min_num_qubits=min_qubits, operational=True, simulator=False)
        except Exception as e:
            print(f"ERROR: Could not find a least busy backend: {e}")
            return None
    
    if backend:
        backend_basis_gates = getattr(backend.configuration(), 'basis_gates', 'N/A')
        print(f"Using backend: {backend.name} (Qubits: {backend.num_qubits}, Basis Gates: {backend_basis_gates})")
    return backend

def compute_local_B_prime_unitary():
    """
    Computes the local 3-qubit unitary G_local (B'_0 for N=3 system).
    This uses the "robust method" by leveraging existing qic_core functions.
    """
    print("\n--- Computing Local 3-Qubit Unitary G_local (B'_0 for N=3) ---")
    N_local = 3
    k_idx_local = 0 # For B'_0, this is the operator index for an N=3 system

    try:
        # 1. Get QIC basis and V for N=3
        qic_strings_N3, qic_vectors_N3 = qic_core.get_qic_basis(N_local)
        if not qic_strings_N3:
            raise ValueError("Failed to generate QIC basis for N=3.")
        
        V_N3_iso = qic_core.construct_isometry_V(qic_vectors_N3)
        if V_N3_iso is None:
            raise ValueError("Failed to construct isometry V for N=3.")
        V_N3_csc = V_N3_iso.tocsc()

        # 2. Get P_anyon for N=3, k_idx=0
        #    Call with positional arguments for N, operator_index, basis_strings
        P_anyon_N3_k0_sparse = qic_core.get_kauffman_Pn_anyon_general(
            N_local,                # First positional argument: N
            k_idx_local,            # Second positional argument: operator index
            qic_strings_N3,         # Third positional argument: basis_strings
            delta=qic_core.PHI      # Keyword argument for delta (or 4th positional)
        )
        if P_anyon_N3_k0_sparse is None:
            raise ValueError("Failed to generate P_anyon for N=3, k_idx=0.")

        # 3. Build P'_local = V P_anyon_local V_dag for N=3
        P_prime_N3_k0_sparse = qic_core.build_P_prime_n(
            n_operator_idx=k_idx_local,
            n_qubits=N_local,
            V_isometry=V_N3_csc,
            Pn_anyon_matrix=P_anyon_N3_k0_sparse
        )
        if P_prime_N3_k0_sparse is None:
            raise ValueError("Failed to build P'_local for N=3.")

        # 4. Build B'_local (G_local) = R_tau1 P'_local + R_tau_tau (Pi_QIC - P'_local) for N=3
        G_local_sparse = qic_core.build_B_prime_n(
            n_operator_idx=k_idx_local,
            P_prime_n_matrix=P_prime_N3_k0_sparse,
            n_qubits=N_local
        )
        if G_local_sparse is None:
            raise ValueError("Failed to build B'_local (G_local) for N=3.")
        
        G_local_dense = G_local_sparse.toarray() 

        # 5. Verify unitarity
        identity_8x8 = np.eye(2**N_local, dtype=complex)
        if np.allclose(G_local_dense @ G_local_dense.conj().T, identity_8x8, atol=1e-8):
            print("G_local (B'_0 for N=3) is unitary. Dimensions: ", G_local_dense.shape)
        else:
            print("WARNING: Computed G_local is NOT unitary. Please check qic_core implementations.")
            # diff_norm = np.linalg.norm((G_local_dense @ G_local_dense.conj().T) - identity_8x8)
            # print(f"Norm of (G G_dag - I): {diff_norm}")
            # return None # Or raise error

        return G_local_dense

    except Exception as e:
        print(f"ERROR during G_local computation: {e}")
        traceback.print_exc()
        return None

def run_resource_estimation_for_n_efficient(N, G_local, target_backend):
    """
    Builds B'_k for a given N using the local G_local gate, transpiles them,
    and returns a list of dictionaries with resource info.
    Processes B'_0 to B'_{N-3} as 3-qubit gates.
    """
    N_RESULTS = []
    # For an N-qubit system, a 3-qubit gate G_local applied at q_k, q_{k+1}, q_{k+2}
    # means k can go from 0 to N-3. This gives N-2 such operators.
    num_local_ops = max(0, N - 2) 

    if N < 3:
        print(f"N={N} is too small for 3-qubit B'_k operators based on G_local. Skipping B' operator analysis.")
        return N_RESULTS
    
    print(f"\n--- Efficient Transpiling for N_QUBITS = {N} (using G_local) ---")
    
    for k_idx in range(num_local_ops): # k_idx from 0 to N-3
        op_name = f"B'_{k_idx}"
        print(f"\n-- Processing {op_name} --")
        try:
            target_qubits = [k_idx, k_idx + 1, k_idx + 2]
            
            qc = QuantumCircuit(N, name=op_name)
            # Apply the pre-computed 3-qubit G_local to the target qubits
            qc.unitary(G_local, target_qubits, label=f"G_loc({k_idx})")

            print(f"Transpiling {op_name} (local G_local on {target_qubits}) for {target_backend.name} (Opt Level: {OPTIMIZATION_LEVEL})...")
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
            backend_2q_gates = {gate for gate in backend_basis_gates_list if gate in ['cx', 'ecr', 'cz', 'swap', 'rzz', 'rzx']} # Added rzx

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
                'N_QUBITS': N,
                'OPERATOR': op_name,
                'DEPTH': depth,
                'TOTAL_OPS': total_ops_count,
                '2Q_GATE_COUNT': num_two_qubit_gates,
                '2Q_GATE_TYPES': ', '.join(sorted(list(types_two_qubit_gates))),
                'OPS_COUNTS': str(ops_counts),
                'TRANSPILATION_TIME_S': f"{transpile_duration:.3f}"
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

    # --- Create results directory if it doesn't exist ---
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
        service = QiskitRuntimeService() # Assumes account is saved
        print("Connected to IBM Quantum.")
        min_q_for_sweep = max(N_QUBITS_LIST) if N_QUBITS_LIST else 3
        target_backend_instance = get_ibm_backend(service, TARGET_BACKEND_NAME, min_qubits=min_q_for_sweep)
        if target_backend_instance is None:
            print("Could not get a target backend. Exiting.")
            exit()
    except Exception as e:
        print(f"ERROR connecting to IBM Quantum or getting backend: {e}")
        traceback.print_exc()
        exit()
    
    # --- Compute G_local (once) ---
    G_local_unitary = compute_local_B_prime_unitary()
    if G_local_unitary is None:
        print("Failed to compute G_local_unitary. Exiting.")
        exit()

    # --- Main Sweep ---
    for N_val in N_QUBITS_LIST:
        if N_val > target_backend_instance.num_qubits:
            print(f"Skipping N_QUBITS={N_val} as it exceeds backend max qubits ({target_backend_instance.num_qubits}).")
            continue
        
        print(f"\n===== Processing for N_QUBITS = {N_val} (Efficient Method) =====")
        
        results_for_N = run_resource_estimation_for_n_efficient(N_val, G_local_unitary, target_backend_instance)
        all_results.extend(results_for_N)

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