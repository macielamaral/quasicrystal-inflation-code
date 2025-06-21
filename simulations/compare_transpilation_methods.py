# compare_transpilation_methods.py
#
# This script directly compares two methods of constructing a braid circuit
# to see which one results in a more efficient transpiled circuit for
# real quantum hardware.
#
# Method A (Local Composition): Build the circuit by composing a sequence
#           of local, 3-qubit B-gates.
# Method B (Global Unitary): Classically compute the single, giant, non-local
#           unitary matrix for the entire braid, and then ask Qiskit to
#           transpile that single matrix.
#
# The hypothesis is that providing the transpiler with the local gate
# structure (Method A) allows for better optimization.

import numpy as np
import time
import os

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_ibm_runtime import QiskitRuntimeService
from scipy.sparse import identity as sparse_identity

# --- Core Logic Import ---
# This script requires the qic_core.py file to be accessible
try:
    import qic_core
except ImportError:
    print("ERROR: Could not import qic_core.py.")
    print("Please ensure qic_core.py is in the same directory or your PYTHONPATH.")
    exit()

# ==============================================================================
# ===== CONFIGURATION =====
# ==============================================================================
# Use a real backend name to get a realistic transpilation result
BACKEND_NAME = "ibm_brisbane"
KNOT_TO_RUN = "12a_122" # The 3_1 knot is simple and good for a clear comparison
OPTIMIZATION_LEVEL = 1 # 1 is a good balance, 3 is most aggressive.

# Path to the LOCAL 3-qubit braid operator matrix
LOCAL_GATE_FILE = "data/gates/b_local_matrix.npy"


# --- Helper Function from previous script ---
def get_knot_data(knot_name_str):
    """Returns a dictionary with knot data."""
    if knot_name_str == "3_1": return { "name": "3_1", "braid_word_1_indexed": [1, 1, 1] }
    elif knot_name_str == "12a_122": return { "name": "12a_122", "braid_word_1_indexed": [1, -2, -2, 3, -4, 1, 1, -2, -2, -2, 3, -4] }
    else: raise ValueError(f"Unknown knot: {knot_name_str}")

# ==============================================================================
# ===== MAIN EXECUTION =====
# ==============================================================================

if __name__ == "__main__":
    print("--- Transpilation Method Comparison ---")
    print(f"Target Backend: {BACKEND_NAME}")
    print(f"Knot: {KNOT_TO_RUN}")

    # --- 1. General Setup ---
    print("\nSetting up backend and operators...")
    try:
        service = QiskitRuntimeService()
        backend = service.backend(BACKEND_NAME)
    except Exception as e:
        print(f"Could not access backend '{BACKEND_NAME}'. Please check your configuration.")
        print(f"Error: {e}")
        exit()

    knot_data = get_knot_data(KNOT_TO_RUN)
    braid_word = knot_data["braid_word_1_indexed"]
    max_generator_idx = max(abs(g) for g in braid_word)
    N_SITES = max_generator_idx + 2
    print(f"Knot requires N={N_SITES} qubits.")

    # --- 2. Method A: Compose-then-Transpile (Local Structure) ---
    print("\n--- Method A: Composing Local 3-Qubit Gates ---")
    if not os.path.exists(LOCAL_GATE_FILE):
        raise FileNotFoundError(f"Gate file not found: {LOCAL_GATE_FILE}.")
    
    local_b_matrix = np.load(LOCAL_GATE_FILE)
    local_b_gate = UnitaryGate(local_b_matrix, label="B_loc")
    local_b_gate_inv = UnitaryGate(local_b_matrix.T.conj(), label="B_loc†")

    # Build the logical circuit
    qc_A = QuantumCircuit(N_SITES, name="local_composition")
    for g in braid_word:
        idx = abs(g)
        target_qubits = [idx - 1, idx, idx + 1]
        gate_to_apply = local_b_gate_inv if g > 0 else local_b_gate
        qc_A.append(gate_to_apply, target_qubits)
    
    print("Transpiling circuit built from local gates...")
    start_time_A = time.time()
    transpiled_A = transpile(qc_A, backend=backend, optimization_level=OPTIMIZATION_LEVEL)
    duration_A = time.time() - start_time_A

    print("\nRESULTS (Method A):")
    print(f"  Transpilation Time: {duration_A:.3f} s")
    print(f"  Transpiled Depth:   {transpiled_A.depth()}")
    print(f"  Operations Count:   {transpiled_A.count_ops()}")


    # --- 3. Method B: Multiply-then-Transpile (Global Unitary) ---
    print("\n\n--- Method B: Transpiling a Single Global Unitary Matrix ---")
    print("Building the B'_k operators using qic_core...")
    # This block uses the logic from qic_core.py to generate the B' operators
    qic_strings, qic_vectors = qic_core.get_qic_basis(N_SITES)
    V_iso = qic_core.construct_isometry_V(qic_vectors)
    B_prime_ops = []
    for k in range(N_SITES - 1):
        Pk_anyon = qic_core.get_kauffman_Pn_anyon_general(N_SITES, k, qic_strings)
        Pk_prime = qic_core.build_P_prime_n(k, N_SITES, V_iso, Pk_anyon)
        Bk_prime = qic_core.build_B_prime_n(k, Pk_prime, N_SITES)
        B_prime_ops.append(Bk_prime)
    
    print("\nClassically computing the final global unitary matrix...")
    # U_final = U_gN * ... * U_g2 * U_g1
    # So we must left-multiply in reverse order of the braid word
    U_B_global = sparse_identity(2**N_SITES, format='csc')
    for g in reversed(braid_word):
        idx = abs(g)
        # Braid g_i acts on strands i,i+1, which is operator B'_{i-1}
        op_idx = idx - 1
        
        B_op = B_prime_ops[op_idx]
        if g > 0: # Positive braid needs inverse
            B_op = B_op.T.conj()
            
        U_B_global = B_op @ U_B_global

    # Now create a circuit with this single giant matrix
    qc_B = QuantumCircuit(N_SITES, name="global_unitary")
    qc_B.unitary(U_B_global.toarray(), range(N_SITES), label="U_Braid_Global")

    print("Transpiling circuit with the single global matrix...")
    start_time_B = time.time()
    transpiled_B = transpile(qc_B, backend=backend, optimization_level=OPTIMIZATION_LEVEL)
    duration_B = time.time() - start_time_B

    print("\nRESULTS (Method B):")
    print(f"  Transpilation Time: {duration_B:.3f} s")
    print(f"  Transpiled Depth:   {transpiled_B.depth()}")
    print(f"  Operations Count:   {transpiled_B.count_ops()}")
    print("-" * 50)