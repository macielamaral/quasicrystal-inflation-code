# run_qic_hardware_resource_sweep.py
# Estimates resources for QIC braid operators B'_k
# for various N_QUBITS on a specified IBM Quantum hardware backend.

import numpy as np
import time
import traceback
import csv
import os

# Import functions from the core library module
try:
    # Assuming qic_core.py is in src directory and src is in PYTHONPATH
    # or you've used the sys.path.insert(0, src_path) method
    import qic_core 
except ImportError:
    # Attempt relative import if src is a package and this script is run as a module
    # from ..src import qic_core # If run as a module e.g. python -m simulations.this_script
    # Fallback for direct execution if qic_core is in the same dir or already in path
    try:
        # If you set up sys.path in qic_core or it's discoverable
        if 'qic_core' not in globals(): # check if already imported by a previous attempt
             print("Attempting to import qic_core directly (ensure it's in PYTHONPATH or same dir)")
             import qic_core
    except ImportError as e:
        print("ERROR: Failed to import qic_core.py.")
        print("       Make sure qic_core.py is in 'src/' and 'src/' parent is in sys.path,")
        print("       or qic_core.py is in the same directory or Python path.")
        print(f"       Original error: {e}")
        exit()


# Import Qiskit components
if qic_core.QISKIT_AVAILABLE:
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, IBMBackend as RuntimeIBMBackend
    from qiskit.providers.exceptions import QiskitBackendNotFoundError
else:
    print("ERROR: Qiskit and QiskitRuntimeService are required for this script but not available.")
    exit()

# ===== Configuration =====
N_QUBITS_LIST = [2, 3, 4] # List of N_QUBITS to sweep through (e.g., start small: [2, 3])
                          # N=4 means 16x16 unitaries, N=5 means 32x32, transpilation can get slow.

# --- IBM Backend Selection ---
TARGET_BACKEND_NAME = "ibm_brisbane" #  Or set to None to try least_busy (less predictable for specific architecture)
#   service = QiskitRuntimeService()
#   ibm_brisbane (127 qubits)
#   ibm_sherbrooke (127 qubits)


OPTIMIZATION_LEVEL = 3 # Qiskit optimization level (0 to 3)
RESULTS_CSV_FILE = 'data/qic_hardware_resource_results.csv'
# =======================

def get_ibm_backend(service, backend_name=None, min_qubits=2):
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
        print(f"Using backend: {backend.name} (Qubits: {backend.num_qubits}, Basis Gates: {backend.basis_gates})")
    return backend

def run_resource_estimation_for_n(N, V_csc, qic_strings, target_backend):
    """
    Builds B'_k for a given N, transpiles them for the target_backend,
    and returns a list of dictionaries with resource info.
    """
    N_RESULTS = []
    num_tl_generators = N - 1
    if num_tl_generators <= 0 and N < 2 : # B_prime_ops only exist for N>=2
        print(f"N={N} is too small for B' operators. Skipping B' operator analysis.")
        return N_RESULTS

    print(f"\n--- Building and Transpiling for N_QUBITS = {N} ---")
    
    for k_idx in range(num_tl_generators):
        op_name = f"B'_{k_idx}"
        print(f"\n-- Processing {op_name} --")
        try:
            # 1. Build the QIC B'_k operator
            Pk_anyon = qic_core.get_kauffman_Pn_anyon_general(N, k_idx, qic_strings, delta=qic_core.PHI)
            if Pk_anyon is None: raise ValueError(f"Failed P_anyon gen for {op_name}")
            Pk_prime = qic_core.build_P_prime_n(k_idx, N, V_csc, Pk_anyon)
            if Pk_prime is None: raise ValueError(f"Failed P' gen for {op_name}")
            Bk_prime_sparse = qic_core.build_B_prime_n(k_idx, Pk_prime, N)
            if Bk_prime_sparse is None: raise ValueError(f"Failed B' gen for {op_name}")
            
            Bk_prime_dense = Bk_prime_sparse.toarray()

            # 2. Create a circuit with just this unitary
            qc = QuantumCircuit(N, name=op_name)
            # Apply to qubits 0 to N-1
            qc.unitary(Bk_prime_dense, list(range(N)), label=op_name)

            # 3. Transpile the circuit
            print(f"Transpiling {op_name} for {target_backend.name} (Opt Level: {OPTIMIZATION_LEVEL})...")
            start_transpile_time = time.time()
            # Use the target_backend directly; its properties (basis_gates, coupling_map) will be used
            transpiled_qc = transpile(qc, backend=target_backend, optimization_level=OPTIMIZATION_LEVEL)
            end_transpile_time = time.time()
            transpile_duration = end_transpile_time - start_transpile_time
            print(f"Transpilation finished ({transpile_duration:.3f} s).")

            # 4. Analyze the transpiled circuit
            depth = transpiled_qc.depth()
            ops_counts = transpiled_qc.count_ops()
            total_ops_count = sum(ops_counts.values())
            
            # Identify 2-qubit gates based on backend's known 2Q gates
            # Common IBM 2Q gates: 'cx', 'ecr', 'cz'. Check target_backend.basis_gates
            backend_2q_gates = {gate for gate in target_backend.basis_gates if gate in ['cx', 'ecr', 'cz', 'swap', 'rzz']}

            
            num_two_qubit_gates = 0
            types_two_qubit_gates = set()
            for gate_name, count in ops_counts.items():
                # A simple check: if a gate name is in the known 2Q list and in ops_counts
                if gate_name in backend_2q_gates: # Check if the gate is a known 2Q gate type for this backend
                    num_two_qubit_gates += count
                    types_two_qubit_gates.add(gate_name)
            
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
                'OPS_COUNTS': str(ops_counts), # Store full dict as string
                'TRANSPILATION_TIME_S': f"{transpile_duration:.3f}"
            })

        except Exception as e:
            print(f"ERROR during processing for {op_name} at N={N}: {e}")
            traceback.print_exc()
            N_RESULTS.append({
                'N_QUBITS': N,
                'OPERATOR': op_name,
                'DEPTH': 'Error',
                'TOTAL_OPS': 'Error',
                '2Q_GATE_COUNT': 'Error',
                '2Q_GATE_TYPES': 'Error',
                'OPS_COUNTS': str(e),
                'TRANSPILATION_TIME_S': 'Error'
            })
    return N_RESULTS

if __name__ == "__main__":
    master_start_time = time.time()
    all_results = []

    # --- Setup CSV File ---
    if not os.path.exists(RESULTS_CSV_FILE) or os.path.getsize(RESULTS_CSV_FILE) == 0:
        with open(RESULTS_CSV_FILE, 'w', newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(['N_QUBITS', 'OPERATOR', 'DEPTH', 'TOTAL_OPS', 
                             '2Q_GATE_COUNT', '2Q_GATE_TYPES', 'OPS_COUNTS', 'TRANSPILATION_TIME_S'])
        print(f"Created results file: {RESULTS_CSV_FILE}")
    else:
        print(f"Appending to existing results file: {RESULTS_CSV_FILE}")

    # --- Connect to IBM Quantum Service ---
    service = None
    target_backend_instance = None
    try:
        print("\nConnecting to IBM Quantum Runtime Service...")
        # Make sure you have your account saved, e.g., QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_API_TOKEN")
        service = QiskitRuntimeService()
        print("Connected to IBM Quantum.")
        target_backend_instance = get_ibm_backend(service, TARGET_BACKEND_NAME, min_qubits=max(N_QUBITS_LIST) if N_QUBITS_LIST else 2)
        if target_backend_instance is None:
            print("Could not get a target backend. Exiting.")
            exit()
    except Exception as e:
        print(f"ERROR connecting to IBM Quantum or getting backend: {e}")
        traceback.print_exc()
        exit()
    
    # --- Main Sweep ---
    for N_val in N_QUBITS_LIST:
        if N_val > target_backend_instance.num_qubits:
            print(f"Skipping N_QUBITS={N_val} as it exceeds backend max qubits ({target_backend_instance.num_qubits}).")
            continue
        
        print(f"\n===== Processing for N_QUBITS = {N_val} =====")
        # Setup QIC basis and V for current N_val
        print(f"Setting up QIC basis for N_QUBITS = {N_val}...")
        current_qic_strings, current_qic_vectors = qic_core.get_qic_basis(N_val)
        if not current_qic_strings:
            print(f"Failed to generate QIC basis for N={N_val}. Skipping.")
            continue
        current_V_iso = qic_core.construct_isometry_V(current_qic_vectors)
        if current_V_iso is None:
            print(f"Failed to construct isometry V for N={N_val}. Skipping.")
            continue
        current_V_csc = current_V_iso.tocsc()
        print("QIC basis and V setup complete.")

        # Run resource estimation for this N
        results_for_N = run_resource_estimation_for_n(N_val, current_V_csc, current_qic_strings, target_backend_instance)
        all_results.extend(results_for_N)

        # Save to CSV incrementally
        with open(RESULTS_CSV_FILE, 'a', newline='') as f_csv:
            writer = csv.writer(f_csv)
            for res_dict in results_for_N:
                writer.writerow([res_dict['N_QUBITS'], res_dict['OPERATOR'], res_dict['DEPTH'], 
                                 res_dict['TOTAL_OPS'], res_dict['2Q_GATE_COUNT'],
                                 res_dict['2Q_GATE_TYPES'], res_dict['OPS_COUNTS'],
                                 res_dict['TRANSPILATION_TIME_S']])
        print(f"Results for N_QUBITS={N_val} appended to {RESULTS_CSV_FILE}")

    print("\n--- QIC Hardware Resource Sweep Finished ---")
    # print("\nCollected Results:")
    # for r in all_results:
    # print(r) # You can print or process further if needed
    
    total_duration = time.time() - master_start_time
    print(f"Total execution time: {total_duration:.3f} seconds")
    print(f"All results saved to {RESULTS_CSV_FILE}")