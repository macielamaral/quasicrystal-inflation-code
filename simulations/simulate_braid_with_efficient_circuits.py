# simulate_braid_with_efficient_circuits.py
#
# This script demonstrates how to:
# 1. Construct the N=4 qubit quantum circuit for the 12a_122 braid
#    using pre-synthesized efficient local unitary blocks (C_L, C_M, C_R).
# 2. Simulate this circuit to get its unitary matrix.
# 3. Calculate the trace of this unitary restricted to the QIC subspace.
# 4. Normalize this "efficient circuit raw trace" to get an estimate of the
#    Jones polynomial for 12a_122, to be compared with the theoretical value
#    obtained from abstract anyonic matrices.

import numpy as np
from scipy.sparse import csc_matrix # For potential sparse operations if needed later
import time
import os
import sys


try:
    # --- Qiskit Imports ---
    from qiskit import QuantumCircuit, transpile
    from qiskit import qpy 
    from qiskit.quantum_info import Operator
    # FROM QISKIT_PROVIDERS_AER (QISKIT < 0.46) to QISKIT_AER (QISKIT >= 0.46)
    # Try new import path first for Qiskit 1.0+ (qiskit-aer 0.13+)
    from qiskit_aer.primitives import SamplerV2 as AerSampler 
    # For UnitarySimulator, Qiskit Aer has changed how backends are accessed.
    # It's now through qiskit_aer.AerSimulator()
    from qiskit_aer import AerSimulator 
    
    QISKIT_AVAILABLE = True
except ImportError:
    print("ERROR: Qiskit is required for this script.")
    print("       Please ensure Qiskit and Qiskit Aer are installed.")
    QISKIT_AVAILABLE = False
    exit() # This exit is being triggered

# --- Try to import qic_core for constants and QIC basis generation ---
try:
    # Assuming qic_core.py is in a directory that Python can find
    # (e.g., same directory, or src/ if simulations/ is run from parent)
    # If simulations/ is run directly and qic_core is in ../src, adjust path:
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    import qic_core
    PHI = qic_core.PHI
    fibonacci = qic_core.fibonacci
    get_qic_basis = qic_core.get_qic_basis
    construct_isometry_V = qic_core.construct_isometry_V # Make sure this is exposed by qic_core if needed
except ImportError as e:
    print(f"ERROR: Could not import qic_core.py: {e}")
    print("       Ensure qic_core.py is in the Python path (e.g., src/ and you run from the project root, or it's in the same directory).")
    exit()
except AttributeError as e:
    print(f"ERROR: A required attribute is missing from qic_core.py: {e}")
    exit()

# ===== Configuration =====
N_QUBITS = 4 # For 12a_122 knot

# --- Paths to PRE-SYNTHESIZED QPY Circuit Files ---
# These should be the 3-qubit circuits C_L, C_M, C_R
SYNTHESIZED_CIRCUITS_DIR = "data/synthesized_circuits" 
PATH_C_L_QPY = os.path.join(SYNTHESIZED_CIRCUITS_DIR, "C_L.qpy")
PATH_C_M_QPY = os.path.join(SYNTHESIZED_CIRCUITS_DIR, "C_M.qpy")
PATH_C_R_QPY = os.path.join(SYNTHESIZED_CIRCUITS_DIR, "C_R.qpy")


# (Optional) KNOWN SYNTHESIS FIDELITIES for C_L, C_M, C_R (for reference)
# These come from your qic_synthesis_tools.py runs
KNOWN_SYNTH_FID_L = 0.99906 # Example, replace with actual
KNOWN_SYNTH_FID_M = 0.99924 # Example, replace with actual
KNOWN_SYNTH_FID_R = 0.99177 # Example, replace with actual

# Target theoretical Jones Polynomial value for 12a_122 (N=4, t=exp(-2pi*i/5))
# This is the V_final from calculate_renormalized_jones.py
TARGET_JONES_12A122_FINAL = complex(0.34682189257828955, 0.07347315653655925)
# =======================

def load_efficient_circuits():
    """Loads the pre-synthesized C_L, C_M, C_R circuits."""
    print("\nLoading pre-synthesized local circuits C_L, C_M, C_R...")
    circs = {}
    paths = {"L": PATH_C_L_QPY, "M": PATH_C_M_QPY, "R": PATH_C_R_QPY}
    fids = {"L": KNOWN_SYNTH_FID_L, "M": KNOWN_SYNTH_FID_M, "R": KNOWN_SYNTH_FID_R}

    all_loaded = True
    for type_key, qpy_path in paths.items():
        if not os.path.exists(qpy_path):
            print(f"ERROR: QPY file not found for C_{type_key}: {qpy_path}")
            all_loaded = False; continue
        try:
            with open(qpy_path, 'rb') as fd:
                loaded_qpy_data = qpy.load(fd)
                # qpy.load returns a list of circuits
                if not loaded_qpy_data or not isinstance(loaded_qpy_data[0], QuantumCircuit):
                    raise ValueError(f"QPY file {qpy_path} did not contain a valid QuantumCircuit.")
                circs[type_key] = loaded_qpy_data[0] # Assuming one circuit per file
            print(f"  Circuit C_{type_key} loaded (Known Synth Fidelity: {fids[type_key]:.5f})")
        except Exception as e:
            print(f"ERROR loading C_{type_key} from {qpy_path}: {e}")
            all_loaded = False
    
    if not all_loaded or len(circs) != 3:
        print("Critical error: Not all efficient circuits (C_L, C_M, C_R) could be loaded.")
        return None
    return circs

def construct_full_braid_circuit(n_qubits, braid_op_indices, braid_op_inverses, efficient_circs):
    """
    Constructs the N-qubit quantum circuit for the full braid sequence
    using the efficient C_L, C_M, C_R blocks.
    """
    if n_qubits < 3:
        raise ValueError("Efficient circuits are 3-qubit; n_qubits must be at least 3.")

    full_circuit = QuantumCircuit(n_qubits, name=f"Braid_N{n_qubits}")
    
    for i in range(len(braid_op_indices)):
        op_k_idx = braid_op_indices[i] # This is k for B_k (0 to N-2)
        is_inv = braid_op_inverses[i]

        gate_type_str = ""
        active_qubits_list = []

        if op_k_idx == 0: # B_0 acts on qubits 0,1,2
            gate_type_str = "L"
            active_qubits_list = [0, 1, 2]
        elif op_k_idx == n_qubits - 2: # B_{N-2} acts on N-3, N-2, N-1
            gate_type_str = "R"
            active_qubits_list = [n_qubits - 3, n_qubits - 2, n_qubits - 1]
        else: # Middle operator B_k acts on k, k+1, k+2
            gate_type_str = "M"
            active_qubits_list = [op_k_idx, op_k_idx + 1, op_k_idx + 2]

        if any(q >= n_qubits for q in active_qubits_list):
            raise ValueError(f"Active qubits {active_qubits_list} out of bounds for {n_qubits} qubits.")

        chosen_efficient_block = efficient_circs[gate_type_str]
        
        # Apply the block or its inverse
        op_to_apply = chosen_efficient_block.inverse() if is_inv else chosen_efficient_block
        full_circuit.compose(op_to_apply, qubits=active_qubits_list, inplace=True)
        full_circuit.barrier() # For visualization

    return full_circuit

def get_qic_projector(n_qubits, qic_basis_vectors_full_space):
    """
    Constructs the projector Pi_QIC = V V_dagger onto the QIC subspace.
    V maps the F_{N+2} QIC basis to the 2^N Hilbert space.
    """
    V_isometry_sparse = construct_isometry_V(qic_basis_vectors_full_space)
    if V_isometry_sparse is None:
        raise RuntimeError("Failed to construct isometry V for QIC projector.")
    
    V_dense = V_isometry_sparse.toarray()
    Pi_QIC_dense = V_dense @ V_dense.conj().T # Pi_QIC = V V_dag
    return Pi_QIC_dense

def main():
    if not QISKIT_AVAILABLE:
        return

    print("--- Simulating 12a_122 Braid with Efficient Circuits ---")
    sim_start_time = time.time()

    # Load efficient C_L, C_M, C_R circuits
    efficient_circs = load_efficient_circuits()
    if efficient_circs is None:
        return

    # Define 12a_122 knot parameters
    knot_name = "12a_122"
    N_knot = N_QUBITS # Should be 4
    braid_sequence_12a122 = [
        (0, False), (1, True),  (0, False), (2, True),  (1, False), (0, False),
        (2, True),  (1, False), (2, False), (0, False), (1, True),  (2, False)
    ] # op_k_idx for B_k (0 to N-2)
    writhe_12a122 = 4
    
    # Constants for Jones Polynomial normalization
    A_param = np.exp(1j * np.pi / 10) # For t = exp(-2*pi*i/5)
    S_Unknot_N4_theoretical = float(fibonacci(N_knot + 2)) # F_6 = 8 for N=4

    # 1. Construct the full N=4 qubit circuit for 12a_122
    print(f"\nStep 1: Constructing N={N_knot} qubit circuit for {knot_name} using efficient blocks...")
    try:
        U_qic_braid_circuit = construct_full_braid_circuit(N_knot, 
                                                           [item[0] for item in braid_sequence_12a122], 
                                                           [item[1] for item in braid_sequence_12a122], 
                                                           efficient_circs)
        print(f"  Full braid circuit constructed. Depth: {U_qic_braid_circuit.depth()}")
        # print(U_qic_braid_circuit.draw(output='text', fold=120)) # Optional: print circuit
    except ValueError as e:
        print(f"  Error during circuit construction: {e}")
        return

    # 2. Simulate to get the unitary matrix of the circuit
    print("\nStep 2: Simulating circuit to get its unitary matrix...")
    # Use AerSimulator and set the option to save the unitary
    simulator = AerSimulator(method='unitary') # Specify the simulation method

    # Create a copy of the circuit for simulation as run() can modify it
    # or some backends expect circuits not previously run.
    # For unitary method, usually direct circuit is fine.
    # Transpilation for AerSimulator's unitary method is often minimal if gates are standard.
    # However, it's good practice to transpile for the simulator's basis gates if unsure.
    
    # Option 1: Minimal transpilation (often sufficient for unitary method)
    # circuit_to_run = U_qic_braid_circuit 
    
    # Option 2: Transpile to simulator's basis (more robust)
    # First, get a backend instance to know its basis gates, or use a generic set
    # For just getting the unitary, often not strictly needed if ops are Qiskit objects
    # For simplicity here, let's assume the composed circuit can be run directly
    # or that a basic transpile step is enough.
    
    # The circuit needs to be an experiment.
    # We need to tell AerSimulator to save the unitary.
    # We make a copy of the circuit to apply the save_unitary instruction
    
    circuit_for_unitary_sim = U_qic_braid_circuit.copy()
    circuit_for_unitary_sim.save_unitary() # Tell the simulator to save the final unitary

    # Execute the simulation
    result = simulator.run(circuit_for_unitary_sim).result()
    
    # Get the unitary
    # The .get_unitary() method is on the result object when save_unitary() was used.
    # It typically doesn't need the circuit argument anymore if only one unitary was saved.
    U_qic_braid_matrix = result.get_unitary().data # Get the unitary data
    
    print(f"  Unitary matrix of shape {U_qic_braid_matrix.shape} obtained.")

    # 3. Calculate trace restricted to QIC subspace: Sum <q_i| U |q_i> for QIC basis |q_i>
    print("\nStep 3: Calculating trace of the unitary restricted to QIC subspace...")
    qic_strings, qic_basis_vectors_full = get_qic_basis(N_knot)
    
    S_prime_12a122 = 0.0j
    for q_vec_full in qic_basis_vectors_full: # q_vec_full is a 2^N vector
        # <q_i | U | q_i> = q_i_dag @ U @ q_i
        term = np.conjugate(q_vec_full) @ U_qic_braid_matrix @ q_vec_full
        S_prime_12a122 += term
    
    print(f"  Effective Raw Trace from Efficient Circuits (S'_12a122): {S_prime_12a122}")
    print(f"    Real part: {np.real(S_prime_12a122)}")
    print(f"    Imaginary part: {np.imag(S_prime_12a122)}")

    # 4. Normalize this S'_12a122 to get the Jones Polynomial estimate
    print(f"\nStep 4: Normalizing S'_12a122 to estimate Jones Polynomial...")
    phase_factor_writhe = (-A_param**3)**(-writhe_12a122)
    
    # Normalize using the THEORETICAL S_Unknot_N4 = F_{N+2}
    # (as S_Unknot_N4 from efficient circuits would also have approximation errors)
    V_12a122_eff_circuit = phase_factor_writhe * (S_prime_12a122 / S_Unknot_N4_theoretical)

    sim_end_time = time.time()
    print("\n--- Efficient Circuit Simulation and Jones Estimation Complete ---")
    print(f"Total simulation and calculation time: {sim_end_time - sim_start_time:.2f} seconds")

    print(f"\nEstimated Jones Polynomial V({knot_name}) from Efficient Circuits:")
    print(f"  V'_({knot_name}) = {V_12a122_eff_circuit}")
    print(f"    Real part: {np.real(V_12a122_eff_circuit)}")
    print(f"    Imaginary part: {np.imag(V_12a122_eff_circuit)}")

    print(f"\nComparison with target V_final({knot_name}) from abstract anyonic calculation:")
    print(f"  Target V_final = {TARGET_JONES_12A122_FINAL}")
    
    diff = V_12a122_eff_circuit - TARGET_JONES_12A122_FINAL
    error_magnitude = np.abs(diff)
    relative_error = error_magnitude / np.abs(TARGET_JONES_12A122_FINAL) if np.abs(TARGET_JONES_12A122_FINAL) > 1e-9 else error_magnitude
    
    print(f"  Difference: {diff} (Magnitude: {error_magnitude:.6e})")
    print(f"  Relative Error: {relative_error:.6f}")

    if relative_error < 0.01: # Example: 1%
        print("  --> Excellent agreement! The efficient circuit model accurately reproduces the target.")
    elif relative_error < 0.1: # Example: 10%
        print("  --> Good agreement. The efficient circuit model is a reasonable approximation.")
    else:
        print("  --> Notable difference. The approximation error from the efficient circuits is significant for the full braid.")
        print("      This could be due to accumulated errors from C_L/M/R synthesis fidelities not being perfect,")
        print("      or the 3-qubit local model not perfectly capturing the full B_k^anyon action over many operations.")

if __name__ == "__main__":
    main()