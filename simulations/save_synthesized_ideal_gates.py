# save_synthesized_ideal_gates.py
# Loads a target ideal 8x8 unitary, synthesizes it, and saves the QuantumCircuit.

import numpy as np
import os
import sys
import time # To record synthesis time
from qiskit import QuantumCircuit, qpy # For QPY serialization
from qiskit.circuit import Parameter # If create_3q_ansatz needs it explicitly
from qiskit.quantum_info import Operator
from scipy.optimize import minimize

# Assuming qic_synthesis_tools.py is in an importable location
try:
    import qic_synthesis_tools as qst
except ImportError:
    # Add path modification if necessary
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(os.path.dirname(current_dir), 'src') 
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    try:
        import qic_synthesis_tools as qst
        print("Imported qic_synthesis_tools.")
    except ImportError as e:
        print(f"ERROR: qic_synthesis_tools.py not found: {e}")
        exit()


# ===== Configuration =====
# --- INPUT: Path to the .npy file containing the 8x8 target G_ideal matrix ---
#TARGET_G_IDEAL_NPY_FILE = "data/optimal_local_approximators/G_tilde_N10_kop0_act0.npy" # EXAMPLE for G_ideal_L
#TARGET_G_IDEAL_NPY_FILE = "data/optimal_local_approximators/G_tilde_N10_kop4_act4.npy" 
TARGET_G_IDEAL_NPY_FILE = "data/optimal_local_approximators/G_tilde_N10_kop8_act7.npy" 

# --- OUTPUT: Path to save the synthesized QuantumCircuit object ---
# You'll run this script 3 times, changing TARGET_G_IDEAL_NPY_FILE and OUTPUT_CIRCUIT_QPY_FILE

#OUTPUT_CIRCUIT_QPY_FILE = "data/synthesized_circuits/C_L.qpy" 
#OUTPUT_CIRCUIT_QPY_FILE = "data/synthesized_circuits/C_M.qpy"
OUTPUT_CIRCUIT_QPY_FILE = "data/synthesized_circuits/C_R.qpy"

# --- SYNTHESIS SETTINGS (use your best found settings) ---
SYNTHESIS_NUM_ANSATZ_LAYERS = 4
# This should be the num_rot_params_per_qubit_per_layer for qst.create_3q_ansatz
SYNTHESIS_PARAMS_PER_QUBIT_ROT_SEQ = 3 
SYNTHESIS_MAX_OPTIMIZATION_ITERATIONS = 200 # Or more if needed for target fidelity
TARGET_SYNTHESIS_FIDELITY = 0.999 # Set a target, retry if not met? (Optional)
# =======================

if __name__ == "__main__":
    start_time = time.time()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_CIRCUIT_QPY_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 1. Load the target ideal unitary
    if not os.path.exists(TARGET_G_IDEAL_NPY_FILE):
        print(f"ERROR: Target G_ideal file not found: {TARGET_G_IDEAL_NPY_FILE}")
        exit()
    
    target_G_matrix = np.load(TARGET_G_IDEAL_NPY_FILE)
    print(f"Loaded target ideal unitary from: {TARGET_G_IDEAL_NPY_FILE}")
    if target_G_matrix.shape != (8,8):
        print(f"Error: Loaded matrix is shape {target_G_matrix.shape}, expected (8,8).")
        exit()
    if not Operator(target_G_matrix).is_unitary(atol=1e-7):
        print(f"WARNING: Loaded target matrix from {TARGET_G_IDEAL_NPY_FILE} may not be perfectly unitary!")
        print(f"  ||UUdag-I||_F for loaded target: {np.linalg.norm(target_G_matrix @ target_G_matrix.conj().T - np.eye(8), 'fro'):.3e}")


    # 2. Synthesize it using the function from qic_synthesis_tools
    # Assuming synthesize_ideal_gate is now in qst
    synthesized_circuit, achieved_fidelity = qst.synthesize_ideal_gate(
        target_ideal_unitary_matrix=target_G_matrix,
        num_ansatz_layers=SYNTHESIS_NUM_ANSATZ_LAYERS,
        params_per_single_qubit_rot_seq=SYNTHESIS_PARAMS_PER_QUBIT_ROT_SEQ,
        max_opt_iters=SYNTHESIS_MAX_OPTIMIZATION_ITERATIONS,
        gate_name=os.path.basename(OUTPUT_CIRCUIT_QPY_FILE).replace('.qpy','')
    )

    if synthesized_circuit is None:
        print(f"ERROR: Synthesis failed for {TARGET_G_IDEAL_NPY_FILE}")
        exit()

    print(f"\nSynthesis of {os.path.basename(TARGET_G_IDEAL_NPY_FILE)} completed.")
    print(f"  Target Fidelity: > {TARGET_SYNTHESIS_FIDELITY}" if TARGET_SYNTHESIS_FIDELITY else "(no specific target)")
    print(f"  Achieved Fidelity: {achieved_fidelity:.8f}")
    print("  Optimized Circuit Structure:")
    print(synthesized_circuit.draw(output='text'))

    # 3. Save the synthesized QuantumCircuit object using QPY
    try:
        with open(OUTPUT_CIRCUIT_QPY_FILE, 'wb') as fd:
            qpy.dump(synthesized_circuit, fd)
        print(f"Successfully saved synthesized circuit to: {OUTPUT_CIRCUIT_QPY_FILE}")
    except Exception as e:
        print(f"Error saving circuit with QPY: {e}")

    end_time = time.time()
    print(f"Total time for this synthesis and save: {end_time - start_time:.2f} seconds.")