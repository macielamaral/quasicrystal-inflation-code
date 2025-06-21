# compute_jones_hybrid_simulation.py
#
# This script uses a hybrid quantum/classical approach to validate that the
# quantum circuit built from synthesized gates correctly implements the
# intended *approximated* physical model.
#
# Workflow:
# 1. (Quantum) Build the full braid circuit from the synthesized C_General gate.
# 2. (Quantum) Compute the exact unitary matrix for this circuit.
# 3. (Classical) Calculate the trace of the circuit's matrix over the QIC subspace.
# 4. (Verification) Build a classical matrix representation of the *same approximated
#    model* using standard Qiskit tools to ensure convention matching.
# 5. (Verification) Compare the trace from the quantum circuit to the trace from
#    the classical approximation. They should match to high precision.

import numpy as np
import time
import os
import sys

# --- Qiskit Imports ---
try:
    from qiskit import QuantumCircuit, qpy
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library import UnitaryGate
except ImportError:
    print("ERROR: Qiskit is required for this script.")
    exit()

# --- Project Imports ---
try:
    import qic_core
    PHI = qic_core.PHI
    fibonacci = qic_core.fibonacci
    get_qic_basis = qic_core.get_qic_basis
except ImportError:
    print("ERROR: Could not import qic_core.py or qic_synthesis_tools.py.")
    exit()

# ===== Configuration =====
SYNTHESIZED_GATE_QPY_FILE = "data/synthesized_circuits/C_General.qpy"
# The .npy file corresponding to the universal gate C_General.qpy
UNIVERSAL_G_TILDE_NPY_FILE = "data/optimal_local_approximators/G_tilde_N10_kop4_act4.npy"
# =======================

def get_knot_data(knot_name_str):
    """Returns a dictionary with knot data."""
    if knot_name_str == "12a_122":
        return {
            "name": "12a_122",
            "braid_word_1_indexed": [1, -2, -2, 3, -4, 1, 1, -2, -2, -2, 3, -4],
            "N_strands": 5,
            "writhe_B": -2
        }
    else:
        raise ValueError(f"Knot {knot_name_str} not configured for this script.")

def calculate_jones_hybrid_style(knot_key):
    """Calculates and verifies the Jones polynomial from a quantum circuit."""
    knot_data = get_knot_data(knot_key)
    if knot_data is None: return

    knot_name = knot_data["name"]
    braid_word_1_indexed = knot_data["braid_word_1_indexed"]
    N_strands = knot_data["N_strands"]

    print(f"\n--- Starting Hybrid Quantum/Classical Calculation for {knot_name} ---")
    start_time = time.time()

    # --- Step 1: Load the pre-synthesized 3-qubit gate ---
    print(f"\n1. Loading synthesized gate from {SYNTHESIZED_GATE_QPY_FILE}...")
    with open(SYNTHESIZED_GATE_QPY_FILE, 'rb') as fd:
        C_general_circuit = qpy.load(fd)[0]
    C_general_gate = C_general_circuit.to_gate(label="C_Gen")
    print("   Gate loaded successfully.")

    # --- Step 2: Build the full braid circuit (U_beta) ---
    print("\n2. Building full quantum braid circuit (U_beta)...")
    braid_circuit = QuantumCircuit(N_strands, name="U_beta")
    for val in braid_word_1_indexed:
        op_idx_0_based = abs(val) - 1
        k_active_start = op_idx_0_based
        if k_active_start + 2 >= N_strands:
             k_active_start = N_strands - 3
        target_qubits = [k_active_start, k_active_start + 1, k_active_start + 2]
        gate_to_append = C_general_gate if val > 0 else C_general_gate.inverse()
        braid_circuit.append(gate_to_append, target_qubits)

    U_beta_matrix_op = Operator(braid_circuit.decompose()).reverse_qargs()
    U_beta_matrix = U_beta_matrix_op.data
    print("   Quantum circuit's unitary matrix computed.")

    # --- Step 3: Calculate trace from the quantum circuit's matrix ---
    print("\n3. Calculating trace from the quantum circuit's matrix...")
    qic_basis_strings, qic_basis_vectors = get_qic_basis(N_strands)
    trace_from_circuit = 0.0 + 0.0j
    for basis_vec in qic_basis_vectors:
        trace_from_circuit += basis_vec.conj().T @ U_beta_matrix @ basis_vec
    print(f"   Trace from Circuit = {trace_from_circuit:.8f}")

    # --- Step 4 (Verification): Build the classical approximated matrix ---
    print("\n4. (Verification) Building classical matrix using standard Qiskit tools...")
    # Load the big-endian matrix from the file
    universal_g_tilde_matrix_big_endian = np.load(UNIVERSAL_G_TILDE_NPY_FILE)
    # Convert it to little-endian for use with Qiskit's UnitaryGate object
    op_big = Operator(universal_g_tilde_matrix_big_endian)
    universal_g_tilde_matrix_little_endian = op_big.reverse_qargs().data
    
    M_approx_classical_matrix = np.identity(2**N_strands, dtype=complex)

    for val in braid_word_1_indexed:
        op_idx_0_based = abs(val) - 1
        k_active_start = op_idx_0_based
        if k_active_start + 2 >= N_strands:
            k_active_start = N_strands - 3
        
        # --- FIX: Use the correctly-ordered little-endian matrix for UnitaryGate ---
        gate_matrix_little_endian = universal_g_tilde_matrix_little_endian if val > 0 else universal_g_tilde_matrix_little_endian.conj().T
        
        # Build the N-qubit operator for a single step using Qiskit
        temp_qc = QuantumCircuit(N_strands)
        target_qubits = [k_active_start, k_active_start + 1, k_active_start + 2]
        temp_qc.append(UnitaryGate(gate_matrix_little_endian), target_qubits)
        
        # Get the big-endian matrix for this single operation to match the trace convention
        single_op_matrix = Operator(temp_qc).reverse_qargs().data

        # Correct the order of matrix multiplication to match circuit application
        M_approx_classical_matrix = single_op_matrix @ M_approx_classical_matrix
        
    trace_from_classical_approx = 0.0 + 0.0j
    for basis_vec in qic_basis_vectors:
        trace_from_classical_approx += basis_vec.conj().T @ M_approx_classical_matrix @ basis_vec
    print(f"   Trace from Classical Approx = {trace_from_classical_approx:.8f}")

    # --- Step 5: Final Verification and Conclusion ---
    print("\n--- Final Verification ---")
    trace_diff = np.abs(trace_from_circuit - trace_from_classical_approx)
    print(f"Difference between circuit trace and classical approx trace: {trace_diff:.8e}")

    if trace_diff < 1e-9:
        print("\nSUCCESS: The quantum circuit correctly implements the approximated model.")
        print("The simulation pipeline is validated.")
    else:
        print("\nFAILURE: The quantum circuit does NOT correctly implement the approximated model.")

    end_time = time.time()
    print(f"--- Calculation for {knot_name} completed in {end_time - start_time:.3f} seconds ---")

if __name__ == "__main__":
    calculate_jones_hybrid_style("12a_122")
