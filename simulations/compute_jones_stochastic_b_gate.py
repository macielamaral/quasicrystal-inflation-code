# compute_jones_stochastic_b_gate.py
#
# Computes the Jones polynomial using a stochastic, hardware-centric approach.
# This script builds a batch of circuits for all basis states and runs them
# with a distributed number of shots, simulating a single, large statistical
# experiment suitable for real quantum hardware.

import numpy as np
import os
import time
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import Aer

# Import functions from the core library module
try:
    import qic_core
    PHI = qic_core.PHI
except ImportError:
    print("ERROR: Could not import qic_core.py.")
    exit()

# --- Configuration ---
KNOT_NAME = "12a_122"  # Choose "3_1" or "12a_122"
# We can use a slightly smaller shot budget since the stochastic method is robust
TOTAL_SHOTS = 40960 

# Path to the LOCAL 3-qubit braid operator matrix
LOCAL_GATE_FILE = "data/gates/b_local_matrix.npy"

# Use the QASM simulator for shot-based experiments
SIMULATOR = Aer.get_backend('qasm_simulator')


def get_knot_data(knot_name_str):
    """Returns a dictionary with knot data."""
    if knot_name_str == "3_1":
        return { "name": "3_1", "braid_word_1_indexed": [1, 1, 1], "N_strands": 2, "jones_coeffs": {0:0, 1:1, 2:0, 3:1, 4:-1}, "writhe_B": 3 }
    elif knot_name_str == "12a_122":
        return { "name": "12a_122", "braid_word_1_indexed": [1, -2, -2, 3, -4, 1, 1, -2, -2, -2, 3, -4], "N_strands": 5, "jones_coeffs": { -8: -1, -7:  3, -6: -8, -5: 15, -4: -21, -3: 26, -2: -27, -1: 25,  0: -20,  1: 14,  2: -7,  3:  3, 4: -1 }, "writhe_B": -2 }
    else: raise ValueError(f"Unknown knot: {knot_name_str}")

def create_braid_circuit(braid_word, num_qubits, local_b_gate):
    """Constructs a quantum circuit for U_B from a braid word."""
    qc = QuantumCircuit(num_qubits, name="U_Braid")
    local_b_gate_inv = local_b_gate.inverse()
    local_b_gate_inv.label = "B_loc†"
    for g in braid_word:
        generator_idx = abs(g)
        target_qubits = [generator_idx - 1, generator_idx, generator_idx + 1]
        gate_to_apply = local_b_gate_inv if g > 0 else local_b_gate
        qc.append(gate_to_apply, target_qubits)
    return qc

def run_stochastic_trace_estimation(braid_circuit, qic_basis_strings, total_shot_budget):
    """
    Estimates the normalized trace by creating a batch of circuits
    and distributing the total shot budget among them.
    """
    num_qic_states = len(qic_basis_strings)
    if num_qic_states == 0: return 0j
    
    shots_per_state = max(1, total_shot_budget // num_qic_states)
    print(f"\nBeginning stochastic estimation.")
    print(f"  Total Shot Budget: {total_shot_budget}")
    print(f"  Distributing {shots_per_state} shots to each of the {num_qic_states} basis states.")

    real_circuits, imag_circuits = [], []
    num_qubits = braid_circuit.num_qubits
    controlled_braid_gate = braid_circuit.to_gate().control(1)
    
    for basis_state_str in qic_basis_strings:
        # Create circuit for Real part
        qc_real = QuantumCircuit(num_qubits + 1, 1, name=f"real_{basis_state_str}")
        for i, bit in enumerate(reversed(basis_state_str)):
            if bit == '1': qc_real.x(i)
        qc_real.h(num_qubits)
        qc_real.append(controlled_braid_gate, [num_qubits] + list(range(num_qubits)))
        qc_real.h(num_qubits)
        qc_real.measure(num_qubits, 0)
        real_circuits.append(qc_real)

        # Create circuit for Imaginary part
        qc_imag = QuantumCircuit(num_qubits + 1, 1, name=f"imag_{basis_state_str}")
        for i, bit in enumerate(reversed(basis_state_str)):
            if bit == '1': qc_imag.x(i)
        qc_imag.h(num_qubits)
        qc_imag.sdg(num_qubits)
        qc_imag.append(controlled_braid_gate, [num_qubits] + list(range(num_qubits)))
        qc_imag.h(num_qubits)
        qc_imag.measure(num_qubits, 0)
        imag_circuits.append(qc_imag)

    print("  Transpiling and running circuit batches...")
    transpiled_real = transpile(real_circuits, SIMULATOR)
    real_results = SIMULATOR.run(transpiled_real, shots=shots_per_state).result()
    
    transpiled_imag = transpile(imag_circuits, SIMULATOR)
    imag_results = SIMULATOR.run(transpiled_imag, shots=shots_per_state).result()
    
    total_p0_real, total_p1_real = 0, 0
    for i in range(num_qic_states):
        counts = real_results.get_counts(i)
        total_p0_real += counts.get('0', 0)
        total_p1_real += counts.get('1', 0)

    total_p0_imag, total_p1_imag = 0, 0
    for i in range(num_qic_states):
        counts = imag_results.get_counts(i)
        total_p0_imag += counts.get('0', 0)
        total_p1_imag += counts.get('1', 0)

    total_shots_real = total_p0_real + total_p1_real
    total_shots_imag = total_p0_imag + total_p1_imag

    avg_real_part = (total_p0_real - total_p1_real) / total_shots_real if total_shots_real > 0 else 0
    avg_imag_part = (total_p0_imag - total_p1_imag) / total_shots_imag if total_shots_imag > 0 else 0

    return avg_real_part + 1j * avg_imag_part


if __name__ == "__main__":
    master_start_time = time.time()
    print(f"--- Starting Quantum Jones Polynomial Calculation for {KNOT_NAME} (Stochastic Method) ---")
    
    knot_data = get_knot_data(KNOT_NAME)
    braid_word = knot_data["braid_word_1_indexed"]
    max_generator_idx = max(abs(g) for g in braid_word)
    N_SITES = max_generator_idx + 2
    
    print(f"\nKnot '{KNOT_NAME}' requires a system of N_SITES = {N_SITES} to implement.")

    print(f"Loading local gate and building braid circuit for U_B...")
    b_local_matrix = np.load(LOCAL_GATE_FILE)
    local_b_gate = UnitaryGate(b_local_matrix, label="B_loc")
    
    braid_circuit = create_braid_circuit(braid_word, N_SITES, local_b_gate)
    
    print("Generating QIC basis states...")
    qic_basis_strings, _ = qic_core.get_qic_basis(N_SITES)
    
    norm_trace_quantum = run_stochastic_trace_estimation(braid_circuit, qic_basis_strings, TOTAL_SHOTS)
    
    print(f"\nQuantum Estimation Complete.")
    print(f"  Normalized Trace Estimate [Tr(Pi*M_beta)/F_N+2] ≈ {norm_trace_quantum:.8f}")

    print("\nApplying classical pre-factors to get the Jones polynomial...")
    writhe_B, N_strands = knot_data["writhe_B"], knot_data["N_strands"]
    A_param_KL = np.exp(1j * 3 * np.pi / 5.0)
    
    writhe_phase_factor = (-A_param_KL**3)**(-writhe_B)
    phi_power_factor = PHI**(N_strands - 1)
    
    calculated_jones_value = writhe_phase_factor * phi_power_factor * norm_trace_quantum
    
    print(f"  Calculated V({KNOT_NAME}) at t=exp(-2*pi*i/5) ≈ {calculated_jones_value:.8f}")
    
    # --- Corrected Verification Section ---
    print("\n--- Verification ---")
    # Corrected one-liner function with the 'return' statement
    def evaluate_jones_poly(t_val, coeffs_dict):
        return sum(coeff * (t_val**power) for power, coeff in coeffs_dict.items())
    
    # Corrected definition of target_t_val to match the classical script
    target_t_val = A_param_KL**(-4)
    
    expected_value = evaluate_jones_poly(target_t_val, knot_data["jones_coeffs"])
    
    print(f"  Expected Theoretical Value ≈ {expected_value:.8f}")

    abs_diff = np.abs(calculated_jones_value - expected_value)
    print(f"  Absolute Difference: {abs_diff:.8f}")

    if abs_diff < 0.1: 
        print("  SUCCESS: Quantum result is close to the theoretical value.")
    else: 
        print("  NOTE: Difference is larger than tolerance. Try increasing TOTAL_SHOTS.")

    print("-" * 50)
    print(f"Total Execution Time: {time.time() - master_start_time:.2f} seconds")