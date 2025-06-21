# run_n3_resource_estimation.py
# Estimates resources for N=3 braid operators on IBM Quantum hardware.

import numpy as np
import time
import traceback

# Import functions from the core library module
try:
    import qic_core
except ImportError:
    print("ERROR: Failed to import qic_core.py.")
    print("       Make sure qic_core.py is in the same directory or Python path.")
    exit()

# Import Qiskit components
if qic_core.QISKIT_AVAILABLE:
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime import QiskitRuntimeService
    # Statevector needed if we were simulating the transpiled circuit
    # from qiskit.quantum_info import Statevector
else:
    print("ERROR: Qiskit is required for this script but is not available.")
    exit()

# === Configuration ===
N = 3 # Fixed N=3
# --- IBM Backend Selection ---
# Option 1: Specify a known backend name
# backend_name = "ibm_brisbane"
# Option 2: Find the least busy operational backend (ensure it has >= N qubits)
backend_name = None # Set to None to use least_busy
min_qubits = N
# ---------------------------
OPTIMIZATION_LEVEL = 3 # Qiskit optimization level (0 to 3)

# =======================

if __name__ == "__main__":

    master_start_time = time.time()
    print(f"--- Starting N={N} Hardware Resource Estimation ---")

    qic_strings = []
    qic_vectors = []
    V = None
    B_prime_ops = [] # Only need B' operators for this script
    operators_ok = False

    # === Part 1: Setup (Only Basis Strings and V needed) ===
    print(f"\n=== PART 1: Setup QIC Basis and Isometry V (N={N}) ===")
    start_time = time.time()
    try:
        qic_strings, qic_vectors = qic_core.get_qic_basis(N)
        if not qic_strings or not qic_vectors: raise ValueError(f"Failed to generate QIC basis for N={N}.")
        V = qic_core.construct_isometry_V(qic_vectors)
        if V is None: raise ValueError("Failed to construct isometry V.")
        print(f"Isometry V constructed.")
    except Exception as e:
        print(f"ERROR during Part 1: {e}"); traceback.print_exc(); exit()
    end_time = time.time(); print(f"Part 1 Execution Time: {end_time - start_time:.3f} seconds")

    # === Part 2: Build N=3 Embedded Braid Operators ===
    print(f"\n=== PART 2: Build Embedded Braid Operators B'_k (N={N}) ===")
    part2_start_time = time.time()
    temp_operators_ok = True
    num_ops_expected = N - 1
    for k in range(num_ops_expected): # Loops k=0, 1 for N=3
        print(f"\n--- Processing Operator Index k = {k} ---")
        try:
            Pk_anyon = qic_core.get_kauffman_Pn_anyon_general(N, k, qic_strings, delta=qic_core.PHI)
            if Pk_anyon is None: raise ValueError(f"Failed to get P_{k}^anyon matrix.")
            Pk_prime = qic_core.build_P_prime_n(k, N, V, Pk_anyon)
            if Pk_prime is None: raise ValueError(f"Failed to build P'_{k}.")
            Bk_prime = qic_core.build_B_prime_n(k, Pk_prime, N)
            if Bk_prime is None: raise ValueError(f"Failed to build B'_{k}.")
            B_prime_ops.append(Bk_prime)
        except Exception as e:
            print(f"ERROR processing operator index k={k}: {e}"); traceback.print_exc(); temp_operators_ok = False; break

    if temp_operators_ok and len(B_prime_ops) == num_ops_expected:
        print(f"\nSuccessfully constructed {len(B_prime_ops)} B' operators for N={N}.")
        operators_ok = True
    else:
        print("\nERROR: Operator construction failed.")
        operators_ok = False
    end_time = time.time(); print(f"Part 2 Execution Time: {end_time - part2_start_time:.3f} seconds")


    # === Part 3: Connect to IBM Quantum and Transpile ===
    print(f"\n=== PART 3: Transpile Braid Operators for IBM Backend (N={N}) ===")
    part3_start_time = time.time()

    if not operators_ok:
        print("Skipping Part 3: Operator construction failed.")
    else:
        try:
            print("Connecting to IBM Quantum Runtime Service...")
            # Load saved credentials or provide them:
            # service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_API_TOKEN")
            service = QiskitRuntimeService() # Assumes account is saved
            print("Connected.")

            # Select Backend
            backend = None
            if backend_name:
                print(f"Attempting to get specified backend: {backend_name}...")
                backend = service.get_backend(backend_name)
            else:
                print(f"Finding least busy operational backend with >= {min_qubits} qubits...")
                backend = service.least_busy(min_num_qubits=min_qubits, operational=True, simulator=False)

            if backend:
                print(f"Using backend: {backend.name} (Max qubits: {backend.num_qubits})")
                print(f"Backend basis gates: {backend.basis_gates}")
            else:
                raise RuntimeError("Could not find a suitable IBM Quantum backend.")

            # --- Transpile each Braid Operator ---
            print(f"\nTranspiling with Optimization Level {OPTIMIZATION_LEVEL}...")
            for k, Bk_prime_sparse in enumerate(B_prime_ops):
                print(f"\n--- Transpiling B'_{k} ---")
                try:
                    B_k_prime_dense = Bk_prime_sparse.toarray()

                    # Create a circuit with just the unitary
                    qc = QuantumCircuit(N, name=f"B_prime_{k}")
                    qc.unitary(B_k_prime_dense, list(range(N)), label=f"B'_{k}")

                    # Transpile the circuit
                    print(f"Running transpile for B'_{k}...")
                    start_transpile_time = time.time()
                    transpiled_qc = transpile(qc, backend=backend, optimization_level=OPTIMIZATION_LEVEL)
                    end_transpile_time = time.time()
                    print(f"Transpilation finished ({end_transpile_time - start_transpile_time:.3f} s).")

                    # Analyze the transpiled circuit
                    depth = transpiled_qc.depth()
                    ops_counts = transpiled_qc.count_ops()
                    print(f"  Transpiled Depth: {depth}")
                    print(f"  Operations Count: {ops_counts}")

                    two_qubit_gates_in_circuit = set()
                    two_qubit_count = 0
                    # Get basis gates from the backend object used for transpilation
                    # Use backend.configuration().basis_gates if needed, but backend.basis_gates usually works
                    backend_basis_gates = backend.basis_gates
                    # Identify known 2-qubit gates present in the backend's basis
                    known_backend_2q_gates = {gate for gate in backend_basis_gates if gate in ['ecr', 'cx', 'cz', 'swap']} # Add others if relevant

                    for gate_name, count in ops_counts.items():
                        # Check if the gate from the circuit count is in our known list for this backend
                        if gate_name in known_backend_2q_gates:
                            two_qubit_gates_in_circuit.add(gate_name)
                            two_qubit_count += count

                    print(f"  Estimated 2-Qubit Gate Count: {two_qubit_count} (Types: {two_qubit_gates_in_circuit})")

                except Exception as e:
                    print(f"ERROR during transpilation for B'_{k}: {e}")
                    traceback.print_exc()

        except Exception as e:
            print(f"ERROR during IBM Quantum connection or main transpilation loop: {e}")
            traceback.print_exc()

    end_time = time.time(); print(f"\nPart 3 Execution Time: {end_time - part3_start_time:.3f} seconds")

    master_end_time = time.time()
    print(f"\n--- N={N} Resource Estimation Finished ---")
    print(f"Total execution time: {master_end_time - master_start_time:.3f} seconds")