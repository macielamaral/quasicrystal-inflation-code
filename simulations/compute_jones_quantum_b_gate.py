# compute_jones_quantum_b_gate.py
#
# Computes the Jones polynomial for a given knot on a quantum simulator.
# This script implements the full algorithm by estimating the normalized trace
# of the braid operator using a series of Hadamard Tests.

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
KNOT_NAME = "3_1"  # Choose "3_1" or "12a_122"
NUM_SHOTS = 8192   # Shots for each expectation value. Higher is more accurate.

# Path to the LOCAL 3-qubit braid operator matrix
LOCAL_GATE_FILE = "data/gates/b_local_matrix.npy"

# Use the QASM simulator for shot-based experiments
SIMULATOR = Aer.get_backend('qasm_simulator')


def get_knot_data(knot_name_str):
    """Returns a dictionary with knot data."""
    if knot_name_str == "3_1":
        return {
            "name": "3_1",
            "braid_word_1_indexed": [1, 1, 1],
            "N_strands": 2, # Theoretical minimum number of strands
            "jones_coeffs": {0:0, 1:1, 2:0, 3:1, 4:-1},
            "writhe_B": 3
        }
    elif knot_name_str == "12a_122":
        return {
            "name": "12a_122",
            "braid_word_1_indexed": [1, -2, -2, 3, -4, 1, 1, -2, -2, -2, 3, -4],
            "N_strands": 5,
            "jones_coeffs": {
                -8: -1, -7:  3, -6: -8, -5: 15, -4: -21, -3: 26,
                -2: -27, -1: 25,  0: -20,  1: 14,  2: -7,  3:  3, 4: -1
            },
            "writhe_B": -2
        }
    else:
        raise ValueError(f"Unknown knot: {knot_name_str}")

def create_braid_circuit(braid_word, num_qubits, local_b_gate):
    """Constructs a quantum circuit for U_B from a braid word."""
    qc = QuantumCircuit(num_qubits, name="U_Braid")
    local_b_gate_inv = local_b_gate.inverse()
    local_b_gate_inv.label = "B_loc†"

    for g in braid_word:
        generator_idx = abs(g)
        target_qubits = [generator_idx - 1, generator_idx, generator_idx + 1]
        
        # --- THIS IS THE CORRECTED LINE ---
        # To match the Kauffman/classical formalism, a positive braid
        # generator sigma_k corresponds to the INVERSE B-gate.
        gate_to_apply = local_b_gate_inv if g > 0 else local_b_gate
        
        qc.append(gate_to_apply, target_qubits)

    return qc

def estimate_expectation_value(braid_circuit, basis_state_str, part='real'):
    """
    Estimates Re(<s|U|s>) or Im(<s|U|s>) using a Hadamard test circuit.
    """
    num_qubits = braid_circuit.num_qubits
    ancilla_idx = num_qubits
    
    qc = QuantumCircuit(num_qubits + 1, 1)
    
    for i, bit in enumerate(reversed(basis_state_str)):
        if bit == '1':
            qc.x(i)

    qc.h(ancilla_idx)

    if part == 'imag':
        qc.sdg(ancilla_idx)

    controlled_braid_gate = braid_circuit.to_gate().control(1)
    qc.append(controlled_braid_gate, [ancilla_idx] + list(range(num_qubits)))

    qc.h(ancilla_idx)
    qc.measure(ancilla_idx, 0)

    transpiled_qc = transpile(qc, SIMULATOR)
    result = SIMULATOR.run(transpiled_qc, shots=NUM_SHOTS).result()
    counts = result.get_counts()
    
    p0 = counts.get('0', 0) / NUM_SHOTS
    p1 = counts.get('1', 0) / NUM_SHOTS
    
    return p0 - p1


if __name__ == "__main__":
    master_start_time = time.time()
    print(f"--- Starting Quantum Jones Polynomial Calculation for {KNOT_NAME} ---")
    
    knot_data = get_knot_data(KNOT_NAME)
    
    braid_word = knot_data["braid_word_1_indexed"]
    max_generator_idx = max(abs(g) for g in braid_word)
    N_SITES = max_generator_idx + 2
    
    print(f"\nKnot '{KNOT_NAME}' requires a system of N_SITES = {N_SITES} to implement.")

    print(f"Loading local gate and building braid circuit for U_B...")
    b_local_matrix = np.load(LOCAL_GATE_FILE)
    local_b_gate = UnitaryGate(b_local_matrix, label="B_loc")
    
    braid_circuit = create_braid_circuit(braid_word, N_SITES, local_b_gate)
    
    ### To prepare for quantum hardware###
    print("\n--- Circuit Resource Analysis ---")
    print(f"Circuit Depth: {braid_circuit.depth()}")
    print(f"Gate Counts: {braid_circuit.count_ops()}")
    ###

    print("Generating QIC basis states...")
    qic_basis_strings, _ = qic_core.get_qic_basis(N_SITES)
    num_qic_states = len(qic_basis_strings)
    print(f"Found {num_qic_states} states in the QIC subspace for N={N_SITES}.")

    print(f"\nBeginning quantum estimation of normalized trace ({NUM_SHOTS} shots each)...")
    total_real_part = 0
    total_imag_part = 0

    for i, s in enumerate(qic_basis_strings):
        state_start_time = time.time()
        
        real_part = estimate_expectation_value(braid_circuit, s, part='real')
        imag_part = estimate_expectation_value(braid_circuit, s, part='imag')
        
        total_real_part += real_part
        total_imag_part += imag_part
        
        state_time = time.time() - state_start_time
        print(f"  ({i+1}/{num_qic_states}) For basis state |{s}>:  <s|U_B|s> ≈ {real_part:+.4f} + {imag_part:+.4f}j  ({state_time:.2f}s)")

    norm_trace_quantum = (total_real_part / num_qic_states) + 1j * (total_imag_part / num_qic_states)
    
    print(f"\nQuantum Estimation Complete.")
    print(f"  Normalized Trace Estimate [Tr(Pi*M_beta)/F_N+2] ≈ {norm_trace_quantum:.8f}")

    print("\nApplying classical pre-factors to get the Jones polynomial...")
    writhe_B = knot_data["writhe_B"]
    N_strands = knot_data["N_strands"]
    A_param_KL = np.exp(1j * 3 * np.pi / 5.0)
    target_t_val = A_param_KL**(-4)

    writhe_phase_factor = (-A_param_KL**3)**(-writhe_B)
    phi_power_factor = PHI**(N_strands - 1)
    
    calculated_jones_value = writhe_phase_factor * phi_power_factor * norm_trace_quantum
    
    print(f"  Calculated V({KNOT_NAME}) at t=exp(-2*pi*i/5) ≈ {calculated_jones_value:.8f}")
    
    print("\n--- Verification ---")
    def evaluate_jones_poly(t_val, coeffs_dict):
        return sum(coeff * (t_val**power) for power, coeff in coeffs_dict.items())
    
    expected_value = evaluate_jones_poly(target_t_val, knot_data["jones_coeffs"])
    print(f"  Expected Theoretical Value ≈ {expected_value:.8f}")

    abs_diff = np.abs(calculated_jones_value - expected_value)
    print(f"  Absolute Difference: {abs_diff:.8f}")

    if abs_diff < 0.1:
        print("  SUCCESS: Quantum result is close to the theoretical value.")
    else:
        print("  NOTE: Difference is larger than tolerance. Try increasing NUM_SHOTS for better accuracy.")

    print("-" * 50)
    print(f"Total Execution Time: {time.time() - master_start_time:.2f} seconds")