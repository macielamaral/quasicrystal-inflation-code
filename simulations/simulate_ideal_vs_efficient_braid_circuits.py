# simulate_ideal_vs_efficient_braid_circuits.py
#
# This script compares three ways of obtaining the Jones polynomial for 12a_122:
# 1. Abstract Anyonic Calculation (theoretical target using 8x8 matrices).
# 2. Ideal Embedded Full Braid Unitary Simulation:
#    - Classically compute the single 16x16 matrix for the ideal full braid.
#    - Create a Qiskit circuit with this single 16x16 unitary.
#    - Simulate this circuit, get its QIC-restricted trace, and normalize.
# 3. Efficient Local Circuit Simulation:
#    - Compose C_L, C_M, C_R circuits for the full braid.
#    - Simulate this circuit, get its QIC-restricted trace, and normalize.
#
# This will clearly show the target, what an ideal (but potentially costly)
# quantum implementation would yield, and what the current efficient approximation yields.

import numpy as np
import time
import os
import sys

# --- Qiskit Imports ---
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit import qpy
    from qiskit.quantum_info import Operator
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    print("ERROR: Qiskit is required for this script.")
    QISKIT_AVAILABLE = False
    exit()

# --- qic_core Imports ---
try:
    import qic_core
    PHI = qic_core.PHI
    R_TAU_1 = qic_core.R_TAU_1
    R_TAU_TAU = qic_core.R_TAU_TAU
    fibonacci = qic_core.fibonacci
    get_qic_basis = qic_core.get_qic_basis
    construct_isometry_V = qic_core.construct_isometry_V
    get_kauffman_Pn_anyon_general = qic_core.get_kauffman_Pn_anyon_general
    build_P_prime_n = qic_core.build_P_prime_n
    build_B_prime_n = qic_core.build_B_prime_n
except ImportError:
    print("ERROR: Could not import qic_core.py.")
    exit()
except AttributeError as e:
    print(f"ERROR: Attribute missing from qic_core.py: {e}")
    exit()

# ===== Configuration =====
N_QUBITS = 4
KNOT_NAME = "12a_122"
BRAID_SEQUENCE_12A122 = [
    (0, False), (1, True),  (0, False), (2, True),  (1, False), (0, False),
    (2, True),  (1, False), (2, False), (0, False), (1, True),  (2, False)
]
WRITHE_12A122 = 4
A_PARAM = np.exp(1j * np.pi / 10)
S_UNKNOT_N4_THEORETICAL_ABSTRACT_TRACE = float(fibonacci(N_QUBITS + 2)) # F_6 = 8

# --- Paths for Efficient Circuits (C_L, C_M, C_R) ---
SYNTHESIZED_CIRCUITS_DIR = "data/synthesized_circuits"
PATH_C_L_QPY = os.path.join(SYNTHESIZED_CIRCUITS_DIR, "C_L.qpy")
PATH_C_M_QPY = os.path.join(SYNTHESIZED_CIRCUITS_DIR, "C_M.qpy")
PATH_C_R_QPY = os.path.join(SYNTHESIZED_CIRCUITS_DIR, "C_R.qpy")

# Target from purely abstract calculation (calculate_renormalized_jones.py)
TARGET_JONES_ABSTRACT_12A122 = complex(0.34682189257828955, 0.07347315653655925)


# --- Helper Functions (from previous scripts, slightly adapted) ---

def get_ideal_embedded_braid_operators_dense_list(n_strands, qic_strings_N_basis, V_isometry_csc_matrix):
    """Generates ideal embedded B'_k(N) as dense NumPy arrays."""
    P_anyon_ops_abstract = []
    for k_idx in range(n_strands - 1):
        Pk_anyon = get_kauffman_Pn_anyon_general(n_strands, k_idx, qic_strings_N_basis, delta=PHI)
        P_anyon_ops_abstract.append(Pk_anyon)

    B_prime_dense_ops = []
    B_prime_inv_dense_ops = []
    for k_op_idx, Pk_anyon_abstract in enumerate(P_anyon_ops_abstract):
        Pk_prime_sparse = build_P_prime_n(k_op_idx, n_strands, V_isometry_csc_matrix, Pk_anyon_abstract)
        Bk_prime_sparse = build_B_prime_n(k_op_idx, Pk_prime_sparse, n_strands)
        Bk_prime_dense = Bk_prime_sparse.toarray()
        B_prime_dense_ops.append(Bk_prime_dense)
        B_prime_inv_dense_ops.append(Bk_prime_dense.conj().T)
    return B_prime_dense_ops, B_prime_inv_dense_ops

def compose_dense_matrices_for_braid(n_qubits, braid_op_indices, braid_op_inverses,
                                     B_prime_dense_list, B_prime_inv_dense_list):
    """Classically composes dense B'_k matrices for the full braid."""
    U_total_dense = np.eye(2**n_qubits, dtype=complex)
    for op_k_idx, is_inv in zip(braid_op_indices, braid_op_inverses):
        current_op_matrix = B_prime_inv_dense_list[op_k_idx] if is_inv else B_prime_dense_list[op_k_idx]
        U_total_dense = U_total_dense @ current_op_matrix
    return U_total_dense

def calculate_qic_restricted_trace(U_matrix_dense, qic_basis_vectors_full_space):
    """Calculates Tr_QIC(U_matrix_dense) = Sum <q_i|U|q_i>."""
    qic_trace = 0.0j
    for q_vec in qic_basis_vectors_full_space:
        term = np.conjugate(q_vec) @ U_matrix_dense @ q_vec
        qic_trace += term
    return qic_trace

def load_efficient_circuits_from_qpy():
    """Loads C_L, C_M, C_R from QPY files."""
    circs = {}
    paths = {"L": PATH_C_L_QPY, "M": PATH_C_M_QPY, "R": PATH_C_R_QPY}
    all_loaded = True
    for type_key, qpy_path in paths.items():
        if not os.path.exists(qpy_path):
            print(f"ERROR: QPY file not found for C_{type_key}: {qpy_path}")
            all_loaded = False; continue
        try:
            with open(qpy_path, 'rb') as fd:
                circs[type_key] = qpy.load(fd)[0]
        except Exception as e:
            print(f"ERROR loading C_{type_key} from {qpy_path}: {e}")
            all_loaded = False
    return circs if all_loaded and len(circs) == 3 else None

def construct_efficient_full_braid_circuit(n_qubits, braid_op_indices, braid_op_inverses, efficient_circs_dict):
    """Constructs full braid circuit from C_L, C_M, C_R blocks."""
    full_circuit = QuantumCircuit(n_qubits, name=f"EfficientBraid_N{n_qubits}")
    for op_k_idx, is_inv in zip(braid_op_indices, braid_op_inverses):
        gate_type_str, active_qubits_list = "", []
        if op_k_idx == 0:
            gate_type_str, active_qubits_list = "L", [0, 1, 2]
        elif op_k_idx == n_qubits - 2:
            gate_type_str, active_qubits_list = "R", [n_qubits - 3, n_qubits - 2, n_qubits - 1]
        else:
            gate_type_str, active_qubits_list = "M", [op_k_idx, op_k_idx + 1, op_k_idx + 2]
        
        chosen_block = efficient_circs_dict[gate_type_str]
        op_to_apply = chosen_block.inverse() if is_inv else chosen_block
        full_circuit.compose(op_to_apply, qubits=active_qubits_list, inplace=True)
    return full_circuit

# --- Main Calculation ---
def main():
    if not QISKIT_AVAILABLE: return
    print(f"--- Comparing Jones for {KNOT_NAME} (N={N_QUBITS}) ---")

    # Common setup
    qic_strings, qic_basis_vecs_full = get_qic_basis(N_QUBITS)
    V_iso_sparse = construct_isometry_V(qic_basis_vecs_full)
    V_iso_csc = V_iso_sparse.tocsc()
    
    phase_norm_factor = (-A_PARAM**3)**(-WRITHE_12A122)

    # --- 1. Ideal Embedded Operator Full Braid Simulation ---
    print("\nApproach 1: Simulating Ideal Embedded Full Braid Unitary...")
    start_ideal_embedded = time.time()
    ideal_B_prime_ops, ideal_B_prime_inv_ops = get_ideal_embedded_braid_operators_dense_list(
        N_QUBITS, qic_strings, V_iso_csc
    )
    U_ideal_total_embedded_dense = compose_dense_matrices_for_braid(
        N_QUBITS, [b[0] for b in BRAID_SEQUENCE_12A122], [b[1] for b in BRAID_SEQUENCE_12A122],
        ideal_B_prime_ops, ideal_B_prime_inv_ops
    )

    # Create Qiskit circuit for this ideal unitary and simulate it
    qc_ideal_braid = QuantumCircuit(N_QUBITS, name="IdealBraidUnitary")
    qc_ideal_braid.unitary(U_ideal_total_embedded_dense, range(N_QUBITS))
    
    simulator_unitary = AerSimulator(method='unitary')
    circuit_to_sim_ideal = qc_ideal_braid.copy() # Use a copy
    circuit_to_sim_ideal.save_unitary()
    result_ideal = simulator_unitary.run(circuit_to_sim_ideal).result()
    U_simulated_ideal_matrix = result_ideal.get_unitary().data

    S_double_prime_12a122 = calculate_qic_restricted_trace(U_simulated_ideal_matrix, qic_basis_vecs_full)
    V_12a122_ideal_sim = phase_norm_factor * (S_double_prime_12a122 / S_UNKNOT_N4_THEORETICAL_ABSTRACT_TRACE)
    end_ideal_embedded = time.time()
    print(f"  Raw QIC-Trace (S''): {S_double_prime_12a122:.6f}")
    print(f"  Normalized Jones (V''): {V_12a122_ideal_sim:.6f}")
    print(f"  Time for Ideal Embedded Sim: {end_ideal_embedded - start_ideal_embedded:.3f}s")

    # --- 2. Efficient Local Circuit Simulation ---
    print("\nApproach 2: Simulating Efficient Local Circuit Decomposition...")
    start_efficient_sim = time.time()
    efficient_circs = load_efficient_circuits_from_qpy()
    if not efficient_circs: return

    qc_efficient_braid = construct_efficient_full_braid_circuit(
        N_QUBITS, [b[0] for b in BRAID_SEQUENCE_12A122], [b[1] for b in BRAID_SEQUENCE_12A122],
        efficient_circs
    )
    
    circuit_to_sim_efficient = qc_efficient_braid.copy()
    circuit_to_sim_efficient.save_unitary()
    result_efficient = simulator_unitary.run(circuit_to_sim_efficient).result()
    U_simulated_efficient_matrix = result_efficient.get_unitary().data

    S_prime_12a122 = calculate_qic_restricted_trace(U_simulated_efficient_matrix, qic_basis_vecs_full)
    V_12a122_efficient_sim = phase_norm_factor * (S_prime_12a122 / S_UNKNOT_N4_THEORETICAL_ABSTRACT_TRACE)
    end_efficient_sim = time.time()
    print(f"  Raw QIC-Trace (S'): {S_prime_12a122:.6f}")
    print(f"  Normalized Jones (V'): {V_12a122_efficient_sim:.6f}")
    print(f"  Time for Efficient Circuit Sim: {end_efficient_sim - start_efficient_sim:.3f}s")
    
    # --- Comparison ---
    print("\n--- FINAL COMPARISON ---")
    print(f"Target Jones from Abstract Anyonic Calc (V_final_abstract): {TARGET_JONES_ABSTRACT_12A122:.6f}")
    print(f"Jones from Ideal Embedded Unitary Sim (V''):              {V_12a122_ideal_sim:.6f}")
    print(f"Jones from Efficient Circuit Sim (V'):                   {V_12a122_efficient_sim:.6f}")

    diff_ideal_vs_abstract = np.abs(V_12a122_ideal_sim - TARGET_JONES_ABSTRACT_12A122)
    diff_eff_vs_abstract = np.abs(V_12a122_efficient_sim - TARGET_JONES_ABSTRACT_12A122)

    print(f"\nMagnitude of difference (Ideal Sim vs Abstract): {diff_ideal_vs_abstract:.3e}")
    if diff_ideal_vs_abstract < 1e-9:
        print("  ==> Ideal Embedded Unitary simulation matches Abstract Calculation (as expected).")
    else:
        print("  ==> WARNING: Ideal Embedded Unitary simulation does NOT match Abstract Calculation. Check logic.")

    print(f"\nMagnitude of difference (Efficient Sim vs Abstract): {diff_eff_vs_abstract:.3e}")
    rel_error_eff = diff_eff_vs_abstract / np.abs(TARGET_JONES_ABSTRACT_12A122) if np.abs(TARGET_JONES_ABSTRACT_12A122) > 1e-9 else diff_eff_vs_abstract
    print(f"  Relative Error of Efficient Circuit Model: {rel_error_eff:.4f}")

if __name__ == "__main__":
    main()