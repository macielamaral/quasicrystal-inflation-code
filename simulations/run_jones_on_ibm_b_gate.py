# run_jones_on_ibm_b_gate.py
#
# This script takes the validated, physically-motivated local B_gate and
# executes a core part of the Jones polynomial algorithm on a real IBM Quantum
# computer or a local, ideal simulator. It compares ideal, noiseless, noisy,
# and (optionally) real hardware results.

import numpy as np
import os
import time

# --- Qiskit Imports ---
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
# --- Use the IBM Runtime for interacting with hardware ---
#from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2


# --- Use AerSimulator for local testing ---
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer import AerSimulator

# --- Qiskit Noise Model (Optional) ---
from qiskit.utils.optionals import HAS_AER
if HAS_AER:
    from qiskit_aer.noise import NoiseModel, depolarizing_error

# ==============================================================================
# ===== CONFIGURATION =====
# ==============================================================================

# --- TARGET HARDWARE: Name of the IBM backend you want to use ---
# Leave as None to use local simulators.
# To run on a cloud simulator: "ibmq_qasm_simulator"
# To run on real hardware: "ibm_brisbane", "ibm_kyoto", etc.
BACKEND_NAME = "ibm_brisbane" # <-- CHANGE THIS FOR A REAL RUN

# --- INPUT: The pre-built, theoretical 3-qubit gate ---
LOCAL_GATE_FILE = "data/gates/b_local_matrix.npy"

# --- EXPERIMENT SETTINGS ---
KNOT_TO_RUN = "12a_122"
SHOTS = 1024 #8192
# We will test the expectation value for a single, representative basis state
BASIS_STATE_TO_TEST = "101"


# --- Helper Functions ---
def get_knot_data(knot_name_str):
    """Returns a dictionary with knot data."""
    if knot_name_str == "3_1": return { "name": "3_1", "braid_word_1_indexed": [1, 1, 1] }
    elif knot_name_str == "12a_122": return { "name": "12a_122", "braid_word_1_indexed": [1, -2, -2, 3, -4, 1, 1, -2, -2, -2, 3, -4] }
    else: raise ValueError(f"Unknown knot: {knot_name_str}")

def create_braid_circuit(braid_word, num_qubits, local_b_gate):
    """Constructs a quantum circuit for U_B from a braid word."""
    qc = QuantumCircuit(num_qubits, name="U_Braid")
    local_b_gate_inv = local_b_gate.inverse()
    local_b_gate_inv.label = "B_loc†"
    for g in braid_word:
        generator_idx = abs(g)
        # Braid generator g_i acts on strands (i, i+1), which corresponds to
        # our local 3-qubit gate acting on qubits (i-1, i, i+1)
        target_qubits = [generator_idx - 1, generator_idx, generator_idx + 1]
        
        # Check if target qubits are within the valid range
        if any(q >= num_qubits for q in target_qubits):
            raise ValueError(f"Generator index {g} requires {max(target_qubits)+1} qubits, but only {num_qubits} are available.")

        # Braid g > 0 corresponds to B_local_inv, g < 0 corresponds to B_local
        gate_to_apply = local_b_gate_inv if g > 0 else local_b_gate
        qc.append(gate_to_apply, target_qubits)
    return qc

def print_results(counts, title, shots):
    """Helper to print formatted results from a Sampler run."""
    print(f"\n--- {title} ---")
    if not counts:
        print("No results found.")
        return
        
    # Sort results by counts descending
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    
    print(f"Total shots: {shots}")
    print("Outcome | Counts | Probability")
    print("---------------------------------")
    for i, (outcome, count) in enumerate(sorted_counts):
        # Qiskit uses little-endian bit order, so we reverse for standard reading
        readable_outcome = outcome[::-1]
        probability = count / shots
        print(f"{readable_outcome:<7} | {count:<6} | {probability:.4f}")
        if i >= 9: # Print top 10 results
            break
    print("---------------------------------")


# ==============================================================================
# ===== MAIN EXECUTION =====
# ==============================================================================

if __name__ == "__main__":
    master_start_time = time.time()
    print(f"--- Running Jones Algorithm Test for Knot '{KNOT_TO_RUN}' ---")
    print(f"Targeting basis state |{BASIS_STATE_TO_TEST}> with {SHOTS} shots.")

    # --- 1. SETUP ---
    knot_data = get_knot_data(KNOT_TO_RUN)
    braid_word = knot_data["braid_word_1_indexed"]
    
    # Determine the number of qubits required
    max_generator_idx = max(abs(g) for g in braid_word)
    N_SITES = max_generator_idx + 2 # B_i acts on qubits i-1, i, i+1
    print(f"\nKnot requires N={N_SITES} qubits.")

    # Load the local B-gate from file
    if not os.path.exists(LOCAL_GATE_FILE):
        raise FileNotFoundError(f"Gate file not found: {LOCAL_GATE_FILE}. Please run qic_core.py to generate it.")
    b_local_matrix = np.load(LOCAL_GATE_FILE)
    local_b_gate = UnitaryGate(b_local_matrix, label="B_loc")
    print(f"Loaded 3-qubit B_local gate from '{LOCAL_GATE_FILE}'.")

    # --- 2. BUILD THE QUANTUM CIRCUIT ---
    # Construct the full braid operator circuit U_B
    braid_circuit_op = create_braid_circuit(braid_word, N_SITES, local_b_gate)

    # Create the final test circuit: |psi_in> -> U_B |psi_in> -> Measure
    test_circuit = QuantumCircuit(N_SITES)
    # Initialize to the basis state to test (Qiskit uses little-endian)
    for i, bit in enumerate(reversed(BASIS_STATE_TO_TEST)):
        if bit == '1':
            test_circuit.x(i)
    
    # Instead of appending, compose the circuits to inline the braid gates.
    test_circuit.compose(braid_circuit_op, inplace=True)
    
    # Measure all qubits
    test_circuit.measure_all()
    
    print("\nConstructed final test circuit:")
    # The drawing will now show the individual B_loc gates
    print(test_circuit.draw(output='text', idle_wires=False))

    # --- 3. EXECUTE ON SIMULATORS AND/OR HARDWARE ---
    
    # Dictionary to store results from different runs
    results_collection = {}

    # --- Run 3a: Ideal, noiseless simulation ---
    print("\nExecuting on local IDEAL simulator (AerSampler)...")
    ideal_sampler = AerSampler()
    
    # No explicit transpilation is needed for the default AerSampler.
    # It will handle the compilation internally during the run.
    job_ideal = ideal_sampler.run([test_circuit], shots=SHOTS)
    result_ideal = job_ideal.result()[0]
    counts_ideal = result_ideal.data.meas.get_counts()
    results_collection["Ideal"] = counts_ideal
    print("Ideal simulation complete.")

    # --- Run 3b: Noisy simulation (optional) ---
    if HAS_AER:
        print("\nExecuting on local NOISY simulator (AerSimulator with NoiseModel)...")
        # Create a simple depolarizing noise model
        noise_model = NoiseModel()
        p1_error = 0.001
        p2_error = 0.01
        error_1 = depolarizing_error(p1_error, 1)
        error_2 = depolarizing_error(p2_error, 2)
        noise_model.add_all_qubit_quantum_error(error_1, ['rz', 'sx', 'x'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

        # Create AerSimulator with the noise model
        noisy_sim = AerSimulator(noise_model=noise_model)
        
        # Transpile the circuit for the noisy simulator
        transpiled_circuit_noisy = transpile(test_circuit, noisy_sim)

        # Run with classic Qiskit execution (not Sampler primitive)
        job_noisy = noisy_sim.run(transpiled_circuit_noisy, shots=SHOTS)
        counts_noisy = job_noisy.result().get_counts()
        results_collection["Noisy"] = counts_noisy
        print("Noisy simulation complete.")
    else:
        print("\nSkipping noisy simulation (Qiskit Aer not installed).")


    # --- Run 3c: Real hardware or cloud simulation ---
    if BACKEND_NAME:
        print(f"\nExecuting on IBM Quantum backend: '{BACKEND_NAME}'...")
        try:
            service = QiskitRuntimeService()
            backend = service.backend(BACKEND_NAME)

            # Transpile for the backend (this is still needed!)
            print(f"Transpiling circuit for {backend.name}...")
            transpiled_circuit_hw = transpile(test_circuit, backend)

            # Use SamplerV2 for measurement counts
            sampler = SamplerV2(backend)
            job_hw = sampler.run([transpiled_circuit_hw], shots=SHOTS)
            print(f"Job submitted to {backend.name}. Job ID: {job_hw.job_id()}")
            print("Waiting for results...")
            result_hw = job_hw.result()[0]
            counts_hw = result_hw.data.meas.get_counts()
            results_collection[f"Hardware ({BACKEND_NAME})"] = counts_hw
            print("Hardware execution complete.")


        except Exception as e:
            print(f"ERROR: Could not run on backend '{BACKEND_NAME}'.")
            print(f"Reason: {e}")
    else:
        print("\nSkipping hardware run (BACKEND_NAME is not set).")



    # --- 4. ANALYZE AND COMPARE RESULTS ---
    print("\n\n================= FINAL RESULTS COMPARISON =================")
    for title, counts in results_collection.items():
        print_results(counts, title, SHOTS)
    
    print("\n==========================================================")
    print(f"Total script execution time: {time.time() - master_start_time:.2f} seconds.")