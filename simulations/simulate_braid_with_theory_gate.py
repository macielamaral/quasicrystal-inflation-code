# simulations/simulate_braid_with_quantinuum_gate.py
# (Or modify your existing script)

import numpy as np
from scipy.sparse import csc_matrix
import time
import os
import sys

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit import qpy
    from qiskit.quantum_info import Operator
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    print("ERROR: Qiskit is required.")
    QISKIT_AVAILABLE = False
    exit()

# --- Try to import qic_core ---
try:
    # Adjust path if necessary, e.g., if qic_core is in ../src/
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    import qic_core
    PHI_CORE = qic_core.PHI # Use PHI from qic_core to ensure consistency
    fibonacci = qic_core.fibonacci
    get_qic_basis = qic_core.get_qic_basis
    # construct_isometry_V is not directly needed if we only use get_qic_basis for trace
except ImportError as e:
    print(f"ERROR: Could not import qic_core.py: {e}")
    exit()
except AttributeError as e:
    print(f"ERROR: A required attribute is missing from qic_core.py: {e}")
    exit()

# ===== Configuration =====
N_QUBITS = 4 # For 12a_122 knot
KNOT_NAME = "12a_122"

# Target theoretical Jones Polynomial value for 12a_122 (N=4, t=exp(-2pi*i/5))
TARGET_JONES_12A122_FINAL = complex(0.34682189257828955, 0.07347315653655925)
# =======================

def get_quantinuum_3q_unitary(phi_val, alpha_000=2*np.pi/5, non_fib_diag_phase=0.0):
    """
    Constructs the 8x8 unitary matrix for the Quantinuum braid generator U_sigma_i.
    Qubit order for matrix indices: q2, q1, q0 (Qiskit standard)
    |000> = 0, |001> = 1, ..., |111> = 7
    """
    Uq = np.zeros((8, 8), dtype=complex)

    # Map 3-char strings to Qiskit's integer basis index
    def bstr_to_idx(s): return int(s[::-1], 2) # Qiskit: q2q1q0 -> s[0]s[1]s[2] -> rev then int

    # Fibonacci states relevant to Quantinuum's U_sigma_i definition
    # These are |010>, |011>, |101>, |110>, |111>
    # Their string representations (q2q1q0) and corresponding Qiskit indices:
    idx_010 = bstr_to_idx("010") # 2
    idx_011 = bstr_to_idx("011") # 3
    idx_101 = bstr_to_idx("101") # 5
    idx_110 = bstr_to_idx("110") # 6
    idx_111 = bstr_to_idx("111") # 7

    # Matrix elements from Quantinuum paper (Eq. 3)
    # <s'|U|s> (row_idx = idx_s_prime, col_idx = idx_s)
    Uq[idx_010, idx_010] = np.exp(-1j * 4 * np.pi / 5)
    Uq[idx_011, idx_011] = np.exp(1j * 3 * np.pi / 5)
    Uq[idx_101, idx_101] = (1/phi_val) * np.exp(1j * 4 * np.pi / 5)
    Uq[idx_110, idx_110] = np.exp(1j * 3 * np.pi / 5)
    Uq[idx_111, idx_111] = -(1/phi_val)

    val_101_111 = (phi_val**(-0.5)) * np.exp(-1j * 3 * np.pi / 5)
    Uq[idx_101, idx_111] = val_101_111
    Uq[idx_111, idx_101] = val_101_111 # Symmetric part

    # Handle non-Fibonacci states for the 3-qubit system: |000>, |001>, |100>
    idx_000 = bstr_to_idx("000") # 0
    idx_001 = bstr_to_idx("001") # 1
    idx_100 = bstr_to_idx("100") # 4

    Uq[idx_000, idx_000] = np.exp(1j * alpha_000)
    # Act as identity (phase 0) on other non-Fib states to make the 8x8 unitary
    Uq[idx_001, idx_001] = np.exp(1j * non_fib_diag_phase)
    Uq[idx_100, idx_100] = np.exp(1j * non_fib_diag_phase)

    # Verify Unitarity
    identity_8x8 = np.eye(8, dtype=complex)
    if not np.allclose(Uq @ Uq.conj().T, identity_8x8, atol=1e-8):
        print("WARNING: Constructed Quantinuum 3Q unitary U_Q is NOT unitary!")
        # print("U_Q @ U_Q^dagger = \n", np.round(Uq @ Uq.conj().T, 3))
    # else:
    # print("Constructed Quantinuum 3Q unitary U_Q is unitary.")
    return Uq

def construct_full_braid_circuit_from_3q_unitary(
    n_qubits, braid_op_start_indices, braid_op_inverses, u_3q_operator # Changed u_3q_matrix_np to u_3q_operator
):
    """
    Constructs the N-qubit quantum circuit for the full braid sequence
    using a provided 3-qubit Qiskit Operator.
    'braid_op_start_indices' are the starting qubit index for the 3-qubit action.
    """
    if n_qubits < 3:
        raise ValueError("3-qubit unitary requires n_qubits to be at least 3.")

    full_circuit = QuantumCircuit(n_qubits, name=f"Braid_N{n_qubits}_Custom3Q")
    
    # u_3q_operator is already a Qiskit Operator passed in

    for i in range(len(braid_op_start_indices)):
        start_idx = braid_op_start_indices[i]
        is_inv = braid_op_inverses[i]

        active_qubits_list = [start_idx, start_idx + 1, start_idx + 2]

        if any(q >= n_qubits for q in active_qubits_list) or \
           any(q < 0 for q in active_qubits_list) or \
           len(set(active_qubits_list)) != 3:
            raise ValueError(
                f"Braid generator on start_idx {start_idx} resulted in invalid active qubits "
                f"{active_qubits_list} for {n_qubits} total qubits."
            )

        # Determine the operator to apply (original or its inverse)
        if is_inv:
            gate_to_apply = u_3q_operator.power(-1) # This creates the inverse Operator
            gate_label = f"U3Q_k{start_idx}_inv"
        else:
            gate_to_apply = u_3q_operator
            gate_label = f"U3Q_k{start_idx}"
        
        # Use circuit.unitary() to apply the operator
        full_circuit.unitary(gate_to_apply, qubits=active_qubits_list, label=gate_label)
        
        if i < len(braid_op_start_indices) - 1: # Optional: add barrier except after last op
            full_circuit.barrier() 

    return full_circuit

def main():
    if not QISKIT_AVAILABLE:
        return

    print(f"--- Simulating {KNOT_NAME} Braid with Quantinuum's 3Q Gate Definition ---")
    sim_start_time = time.time()

    # Define Quantinuum's 3Q unitary numpy matrix
    print("\nDefining Quantinuum 3Q Unitary (U_Q) matrix...")
    U_Q_3q_np = get_quantinuum_3q_unitary(PHI_CORE, alpha_000=2*np.pi/5, non_fib_diag_phase=0.0)

    # Convert the numpy matrix to a Qiskit Operator
    U_Q_3q_operator = Operator(U_Q_3q_np)
    # Verify it's unitary (Operator has an is_unitary method)
    if not U_Q_3q_operator.is_unitary(atol=1e-8):
        print("CRITICAL: The defined Quantinuum 3Q gate Operator is not unitary. Aborting.")
        return
    print("Quantinuum 3Q Operator created and checked for unitarity.")


    # Define 12a_122 knot parameters (rest of this section is the same as before)
    N_knot = N_QUBITS 
    braid_op_details_12a122 = [
        (0, False), (1, True),  (0, False), (2, True),  (1, False), (0, False),
        (2, True),  (1, False), (2, False), (0, False), (1, True),  (2, False)
    ] 
    braid_op_start_indices = []
    for op_k_idx_abstract, _ in braid_op_details_12a122:
        if op_k_idx_abstract == 0: 
            braid_op_start_indices.append(0)
        elif op_k_idx_abstract == N_knot - 2: 
            braid_op_start_indices.append(N_knot - 3) 
        else: 
            braid_op_start_indices.append(op_k_idx_abstract)

    braid_op_inverses = [item[1] for item in braid_op_details_12a122]
    writhe_12a122 = 4 

    A_param = np.exp(1j * np.pi / 10)
    S_Unknot_N4_theoretical = float(fibonacci(N_knot + 2)) 

    # 1. Construct the full N=4 qubit circuit
    print(f"\nStep 1: Constructing N={N_knot} qubit circuit for {KNOT_NAME} using U_Q Operator...")
    try:
        U_braid_circuit = construct_full_braid_circuit_from_3q_unitary( # Function name updated for clarity if you changed it
            N_knot,
            braid_op_start_indices,
            braid_op_inverses,
            U_Q_3q_operator # Pass the Operator object
        )
        print(f"  Full braid circuit constructed. Depth: {U_braid_circuit.depth()}")
    except ValueError as e:
        print(f"  Error during circuit construction: {e}")
        traceback.print_exc() # Add traceback for more details on error
        return

    # ... (rest of your main function: Step 2, 3, 4 remain the same) ...
    # 2. Simulate to get the unitary matrix
    print("\nStep 2: Simulating circuit to get its unitary matrix...")
    simulator = AerSimulator(method='unitary')
    circuit_for_unitary_sim = U_braid_circuit.copy()
    circuit_for_unitary_sim.save_unitary()
    # Ensure the simulator can handle the custom unitaries. Transpilation might be needed if there are issues.
    # For basic unitary method, it should be fine.
    try:
        result = simulator.run(circuit_for_unitary_sim).result()
        U_braid_matrix_np = result.get_unitary().data # No circuit argument needed here
    except Exception as e:
        print(f"Error during AerSimulator run or getting unitary: {e}")
        traceback.print_exc()
        return
        
    print(f"  Unitary matrix of shape {U_braid_matrix_np.shape} obtained.")

    # 3. Calculate trace restricted to QIC subspace
    print("\nStep 3: Calculating trace of the unitary restricted to QIC subspace...")
    qic_strings, qic_basis_vectors_full = get_qic_basis(N_knot) # Your QIC basis
    
    S_prime_knot = 0.0j
    for q_vec_full in qic_basis_vectors_full:
        term = np.conjugate(q_vec_full) @ U_braid_matrix_np @ q_vec_full
        S_prime_knot += term
    
    print(f"  Effective Raw Trace from Quantinuum U_Q (S'_{KNOT_NAME}): {S_prime_knot}")
    print(f"    Real part: {np.real(S_prime_knot)}")
    print(f"    Imaginary part: {np.imag(S_prime_knot)}")

    # 4. Normalize S_prime_knot using YOUR FRAMEWORK'S normalization
    print(f"\nStep 4: Normalizing S'_{KNOT_NAME} to estimate Jones Polynomial (Your Framework)...")
    phase_factor_writhe = (-A_param**3)**(-writhe_12a122)
    V_knot_from_UQ = phase_factor_writhe * (S_prime_knot / S_Unknot_N4_theoretical)

    sim_end_time = time.time()
    print("\n--- Simulation with Quantinuum U_Q and Jones Estimation Complete ---")
    print(f"Total simulation and calculation time: {sim_end_time - sim_start_time:.2f} seconds")

    print(f"\nEstimated Jones Polynomial V({KNOT_NAME}) using Quantinuum U_Q:")
    print(f"  V'_UQ_({KNOT_NAME}) = {V_knot_from_UQ}")
    print(f"    Real part: {np.real(V_knot_from_UQ)}")
    print(f"    Imaginary part: {np.imag(V_knot_from_UQ)}")

    print(f"\nComparison with target V_final({KNOT_NAME}) from abstract anyonic calculation:")
    print(f"  Target V_final = {TARGET_JONES_12A122_FINAL}")
    
    diff = V_knot_from_UQ - TARGET_JONES_12A122_FINAL
    error_magnitude = np.abs(diff)
    target_mag = np.abs(TARGET_JONES_12A122_FINAL)
    relative_error = error_magnitude / target_mag if target_mag > 1e-12 else error_magnitude # Avoid division by zero
    
    print(f"  Difference: {diff} (Magnitude: {error_magnitude:.6e})")
    print(f"  Relative Error: {relative_error:.6f}")

    # Comparison messages (assuming your old error magnitude was around 1.136)
    old_error_magnitude = 1.136316 
    if relative_error < 0.01:
        print("  --> Excellent agreement! Quantinuum's U_Q gate significantly improved accuracy.")
    elif error_magnitude < old_error_magnitude * 0.5 : # Significantly better
        print("  --> Significant Improvement! Quantinuum's U_Q provided a much better result than the previous C_L/M/R model.")
    elif error_magnitude < old_error_magnitude - 1e-3: # Noticeably better
        print("  --> Improved! Quantinuum's U_Q provided a better result than the previous C_L/M_R model.")
    elif relative_error < 0.1: # If not necessarily better than old, but still good
        print("  --> Good agreement. Quantinuum's U_Q gate shows reasonable approximation.")
    else:
        print("  --> Notable difference. The result with Quantinuum's U_Q is still significantly different.")
        print("        This could indicate differences in how the abstract anyonic $B_k$ is defined/expected vs. this specific local unitary,")
        print("        or that the target V_final is based on a different idealization, or an issue in the starting qubit index mapping.")

if __name__ == "__main__":
    # Import traceback at the top of the file for use in main
    import traceback 
    main()