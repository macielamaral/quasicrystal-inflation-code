# run_qic_noise_resilience.py
import numpy as np
import time
import qic_core # Make sure this is importable

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import state_fidelity # For direct fidelity if needed, or manual vdot

# --- Configuration ---
N_QUBITS = 3
INITIAL_STATE_STR_QIC = '101' # Example for N=3

# Define a simple braid sequence using indices of B_prime_ops
# e.g., B'_0 then B'_1_dagger then B'_0
BRAID_SEQUENCE_CONFIG = [
    {'index': 0, 'inverse': False}, # B'_0
    {'index': 1, 'inverse': True},  # B'_1_dagger
    {'index': 0, 'inverse': False}  # B'_0
]

# Noise parameters
P_1Q_ERROR =0.05# 0.001 # 0.1% single-qubit depolarizing error
P_2Q_ERROR = 0.1 #0.01  # 1% two-qubit depolarizing error

# --- Helper: Apply QIC sequence (noiseless) ---
def apply_qic_sequence_noiseless(psi_init_np, braid_sequence_config, B_prime_ops, B_prime_dagger_ops):
    psi_current = psi_init_np.copy()
    # ... (Similar to apply_braid_sequence_to_state from previous scripts) ...
    # For each op in braid_sequence_config:
    #   k_qic = op['index']
    #   op_matrix = B_prime_dagger_ops[k_qic] if op['inverse'] else B_prime_ops[k_qic]
    #   psi_current = op_matrix @ psi_current
    # psi_current /= np.linalg.norm(psi_current) # Ensure normalization
    # return psi_current
    # (Using the one from run_jones_sim_v2_plat.py as a base)
    print(f"Applying QIC braid sequence (noiseless):")
    for step, braid_op_info in enumerate(braid_sequence_config):
        k_qic = braid_op_info['index'] 
        is_inverse = braid_op_info['inverse']
        
        if not (0 <= k_qic < len(B_prime_ops)): 
             raise ValueError(f"Invalid QIC operator index B'_{k_qic} for {len(B_prime_ops)} operators.")

        op_label = f"B'_{k_qic}" if not is_inverse else f"B'dag_{k_qic}"
        op_matrix_for_step = B_prime_dagger_ops[k_qic] if is_inverse else B_prime_ops[k_qic]

        # print(f"  Step {step+1}: Applying {op_label} (noiseless)...")
        psi_current = op_matrix_for_step @ psi_current 
        
        # Noisy simulation handles normalization; for ideal, ensure it if ops aren't perfectly unitary numerically
        # if not np.isclose(np.linalg.norm(psi_current), 1.0):
        #    psi_current /= np.linalg.norm(psi_current)
            
    print("Noiseless QIC sequence application complete.")
    # Final normalization for safety, though individual B' should be unitary
    return psi_current / np.linalg.norm(psi_current)


# --- Main ---
if __name__ == "__main__":
    print(f"--- QIC Noise Resilience Simulation for N_QUBITS={N_QUBITS} ---")
    # ... (Timer) ...

    # Part 1: QIC Setup (qic_strings, V_csc, B_prime_ops, B_prime_dagger_ops)
    # ... (similar to other scripts, ensure V_csc, B_prime_ops are populated) ...
    print(f"\n=== PART 1: Setup QIC Basis and Operators (N_QUBITS={N_QUBITS}) ===")
    start_time = time.time()
    try:
        qic_strings, qic_vectors = qic_core.get_qic_basis(N_QUBITS)
        if not qic_strings: raise ValueError("Failed basis gen.")
        dim_QIC = len(qic_strings)
        V_iso = qic_core.construct_isometry_V(qic_vectors)
        if V_iso is None: raise ValueError("Failed V construction.")
        V_csc = V_iso.tocsc()

        B_prime_ops = []
        B_prime_dagger_ops = []
        num_TL_generators = N_QUBITS - 1
        if num_TL_generators < 0: raise ValueError("N_QUBITS must be >= 1")
        for k_idx in range(num_TL_generators):
            Pk_anyon = qic_core.get_kauffman_Pn_anyon_general(N_QUBITS, k_idx, qic_strings, delta=qic_core.PHI)
            Pk_prime = qic_core.build_P_prime_n(k_idx, N_QUBITS, V_csc, Pk_anyon)
            Bk_prime = qic_core.build_B_prime_n(k_idx, Pk_prime, N_QUBITS)
            B_prime_ops.append(Bk_prime.tocsc()) # Ensure CSC for consistent @ behavior
            B_prime_dagger_ops.append(Bk_prime.conj().T.tocsc())
        print(f"Constructed {len(B_prime_ops)} B' operators and their daggers.")
    except Exception as e: print(f"ERROR Part 1: {e}"); traceback.print_exc(); exit()
    print(f"Part 1 Time: {time.time() - start_time:.3f}s")


    # Prepare initial state
    try:
        initial_state_idx = qic_strings.index(INITIAL_STATE_STR_QIC)
        qic_vec_init = np.zeros(dim_QIC, dtype=complex); qic_vec_init[initial_state_idx] = 1.0
        psi_in_np = V_csc @ qic_vec_init
        psi_in_np /= np.linalg.norm(psi_in_np) # Ensure normalized
        print(f"Initial QIC state |{INITIAL_STATE_STR_QIC}> prepared for {N_QUBITS} qubits.")
    except Exception as e: print(f"ERROR preparing initial state: {e}"); traceback.print_exc(); exit()


    # Part 2: Ideal Noiseless Simulation
    print(f"\n=== PART 2: Ideal Noiseless Simulation ===")
    start_time = time.time()
    psi_ideal_np = apply_qic_sequence_noiseless(psi_in_np, BRAID_SEQUENCE_CONFIG, B_prime_ops, B_prime_dagger_ops)
    print(f"Ideal final state norm: {np.linalg.norm(psi_ideal_np):.6f}")
    print(f"Part 2 Time: {time.time() - start_time:.3f}s")

    
    # Part 3: Noisy Qiskit Simulation
    print(f"\n=== PART 3: Noisy Qiskit Simulation ===")
    start_time = time.time()
    psi_noisy_np = None # Initialize
    
    # Define Noise Model
    noise_model = NoiseModel()
    error_1q = depolarizing_error(P_1Q_ERROR, 1)
    error_2q = depolarizing_error(P_2Q_ERROR, 2)
    
    # Define a set of basis gates that are common and universal
    # The noise model will apply to these specific gates.
    # 'id' (identity) is important for a complete basis for the noise model.
    # 'rz', 'sx', 'x' can form any single qubit op. 'cx' for two-qubit.
    # Qiskit's default transpilation often targets u3 (or u), cx.
    # Let's use a common set that AerSimulator and noise models work well with.
    # Standard gates that can compose any unitary:
    primitive_1q_gates = ['id', 'rz', 'sx', 'x'] # 'u3' can be formed from these
    primitive_2q_gates = ['cx']
    
    noise_model.add_all_qubit_quantum_error(error_1q, primitive_1q_gates)
    noise_model.add_all_qubit_quantum_error(error_2q, primitive_2q_gates)
    print(f"Noise model defined with P_1Q={P_1Q_ERROR}, P_2Q={P_2Q_ERROR}")
    print(f"Noise model custom basis gates: {noise_model.basis_gates}") # Shows what gates have noise defined

    # Construct Qiskit Circuit
    qc = QuantumCircuit(N_QUBITS)
    qc.initialize(psi_in_np, range(N_QUBITS))
    # qc.barrier() # Barriers can sometimes interfere with transpiler optimizations unless needed for specific ordering

    print("Constructing Qiskit circuit with QIC unitary operations...")
    for step, op_info in enumerate(BRAID_SEQUENCE_CONFIG):
        k_qic = op_info['index']
        is_inverse = op_info['inverse']
        qic_op_matrix_sparse = B_prime_dagger_ops[k_qic] if is_inverse else B_prime_ops[k_qic]
        qic_op_matrix_dense = qic_op_matrix_sparse.toarray()
        
        op_label = f"B'_{k_qic}" if not is_inverse else f"B'dag_{k_qic}"
        qc.unitary(qic_op_matrix_dense, range(N_QUBITS), label=f"{op_label}_step{step}")

    qc.save_statevector() 
    
    # Transpile and Simulate
    # Create simulator instance with the noise model
    sim_aer_noisy = AerSimulator(noise_model=noise_model)
    
    print("Transpiling circuit for noisy simulation...")
    # Let transpile use the backend's (simulator's) knowledge of its basis gates,
    # which includes those defined in the noise model.
    # The transpiler will decompose qc.initialize and qc.unitary into these.
    tqc = transpile(qc, backend=sim_aer_noisy, optimization_level=1) 
    
    print("Transpiled Circuit Depth:", tqc.depth())
    ops_counts = tqc.count_ops()
    print(f"Transpiled Circuit Ops ({sum(ops_counts.values())} total): {ops_counts}")
    # For even more detail:
    #print(tqc.draw(output='text'))


    print("Running noisy simulation...")
    result_noisy = sim_aer_noisy.run(tqc).result() 
    
    # Retrieve the statevector correctly
    # When a circuit has a single save_statevector instruction without a label,
    # Qiskit Aer often stores it by default with the key 'statevector'.
    # However, accessing through result.data(0) is more robust.
    if 'statevector' in result_noisy.data(0):
        psi_noisy_sv_data = result_noisy.data(0)['statevector']
        psi_noisy_np = np.array(psi_noisy_sv_data)
    else:
        # If it has a different label or structure (less common for default save_statevector)
        # This indicates a problem or a need to inspect result_noisy.data(0) keys
        print("ERROR: 'statevector' key not found in simulation results data.")
        print("Available keys in result.data(0):", result_noisy.data(0).keys())
        exit()
    
    if psi_noisy_np is not None:
        psi_noisy_np /= np.linalg.norm(psi_noisy_np) 
        print(f"Noisy final state norm: {np.linalg.norm(psi_noisy_np):.6f}")
    else: # Should be caught by the key check above
        print("ERROR: Failed to retrieve noisy statevector.")
        exit()
        
    print(f"Part 3 Time: {time.time() - start_time:.3f}s")



    # Part 4: Calculate Fidelity
    print(f"\n=== PART 4: Fidelity Calculation ===")
    fidelity = np.abs(np.vdot(psi_ideal_np, psi_noisy_np))**2
    print(f"Fidelity F = |<ideal|noisy>|^2 = {fidelity:.8f}")
    
    # ... (Timer) ...
    print(f"\n--- Simulation Finished ---")