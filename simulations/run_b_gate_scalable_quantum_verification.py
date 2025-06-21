# run_b_gate_scalable_quantum_verification.py
#
# Implements a SCALABLE quantum verification of the QIC braid operators.
# This script correctly uses a single, local 3-qubit gate as a resource
# and applies it sequentially to a larger N-qubit register. This avoids
# building large 2^N x 2^N matrices and is true to how a quantum
# computer would execute the braid algorithm.

import numpy as np
import os
import traceback
import time
from qiskit import QuantumCircuit
from qiskit.quantum_info import state_fidelity, Statevector
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import Aer

# Import functions from the core library module
try:
    import qic_core
except ImportError:
    print("ERROR: Failed to import qic_core.py.")
    print("       Make sure qic_core.py is in the same directory or Python path.")
    exit()

# --- Configuration ---
N = 5  # Set the desired N for verification (e.g., 4, 5, ... up to ~20)

# Path to the LOCAL 3-qubit braid operator matrix
LOCAL_GATE_FILE = "data/gates/b_local_matrix.npy"

# Use the statevector simulator for ideal fidelity/energy calculations
SIMULATOR = Aer.get_backend('statevector_simulator')


def prepare_qic_superposition_state(n_qubits):
    """
    Uses the qic_core library to prepare an equal superposition of all
    valid QIC basis states for a given number of qubits.
    """
    print(f"\nPreparing QIC superposition state for N={n_qubits}...")
    qic_strings, qic_vectors = qic_core.get_qic_basis(n_qubits)
    if not qic_strings:
        raise ValueError("QIC basis generation failed.")

    dim_qic = len(qic_strings)
    V = qic_core.construct_isometry_V(qic_vectors)
    if V is None:
        raise ValueError("Isometry V construction failed.")

    qic_superpos_anyon = np.ones(dim_qic, dtype=complex) / np.sqrt(dim_qic)
    psi_superpos_full = V.tocsc() @ qic_superpos_anyon
    print(f"  State prepared successfully.")
    return Statevector(psi_superpos_full)


def run_yang_baxter_test(initial_state, local_b_gate, n_qubits):
    """
    Verifies B_k B_{k+1} B_k = B_{k+1} B_k B_{k+1} for all possible k.
    This is the key test of the local gate implementation.
    """
    if n_qubits < 4:
        print("\nSkipping Yang-Baxter test (requires N >= 4).")
        return True

    print("\n--- Running Quantum Braid Relation Test (Yang-Baxter Eq.) ---")
    all_ybe_passed = True
    # Loop through all adjacent triplets, e.g., (B0, B1), (B1, B2), etc.
    for k in range(n_qubits - 3):
        q_k = [k, k+1, k+2]       # Qubits for B_k
        q_kp1 = [k+1, k+2, k+3] # Qubits for B_{k+1}

        print(f"  Testing relation for (B_{k}, B_{k+1}) on qubits {q_k} and {q_kp1}...")

        # --- LHS Circuit: B_k B_{k+1} B_k |psi> ---
        qc_lhs = QuantumCircuit(n_qubits)
        qc_lhs.initialize(initial_state)
        qc_lhs.append(local_b_gate, q_k)
        qc_lhs.append(local_b_gate, q_kp1)
        qc_lhs.append(local_b_gate, q_k)
        sv_lhs = Statevector(SIMULATOR.run(qc_lhs).result().get_statevector())

        # --- RHS Circuit: B_{k+1} B_k B_{k+1} |psi> ---
        qc_rhs = QuantumCircuit(n_qubits)
        qc_rhs.initialize(initial_state)
        qc_rhs.append(local_b_gate, q_kp1)
        qc_rhs.append(local_b_gate, q_k)
        qc_rhs.append(local_b_gate, q_kp1)
        sv_rhs = Statevector(SIMULATOR.run(qc_rhs).result().get_statevector())

        fidelity = state_fidelity(sv_lhs, sv_rhs)
        print(f"    ✅ Result: Fidelity(LHS, RHS) = {fidelity:.8f}")
        if not np.isclose(fidelity, 1.0):
            all_ybe_passed = False
            print(f"    ❌ FAILED for k={k}")

    return all_ybe_passed


def run_subspace_preservation_test(initial_state, local_b_gate, n_qubits):
    """
    Verifies that applying any B_k keeps the state in the QIC subspace (Energy=0).
    """
    print("\n--- Running Quantum Subspace Preservation Test (Energy Measurement) ---")
    hamiltonian = qic_core.build_qic_hamiltonian_op(n_qubits, verbose=False)
    if hamiltonian is None: return False

    # First, confirm the initial state is in the ground state
    is_gs_init, E_init = qic_core.verify_energy(
        initial_state.data, hamiltonian, n_qubits, verbose=False
    )
    print(f"  Initial state energy: {E_init.real:.8f}")
    if not is_gs_init:
        print("  ❌ ERROR: Initial state is not in the ground state! Aborting test.")
        return False

    all_subspace_tests_passed = True
    # Loop through applying B_k for each possible k
    for k in range(n_qubits - 2):
        print(f"  Testing action of B_{k} on qubits {k, k+1, k+2}...")
        
        qc = QuantumCircuit(n_qubits)
        qc.initialize(initial_state)
        qc.append(local_b_gate, [k, k+1, k+2])
        final_state_vector = SIMULATOR.run(qc).result().get_statevector().data

        is_gs, energy = qic_core.verify_energy(
            final_state_vector, hamiltonian, n_qubits, verbose=False
        )
        print(f"    ✅ Result: Energy after B_{k} = {energy.real:.8f}")
        if not is_gs:
            all_subspace_tests_passed = False
            print(f"    ❌ FAILED for k={k}")

    return all_subspace_tests_passed


if __name__ == "__main__":
    master_start_time = time.time()
    print("==========================================================")
    print(f"== Scalable Quantum Verification for N={N} Qubits ==")
    print("==========================================================")

    # === Part 1: Load the LOCAL 3-qubit Gate Resource ===
    if not os.path.exists(LOCAL_GATE_FILE):
        print(f"ERROR: Local gate file not found at '{LOCAL_GATE_FILE}'")
        exit()

    print(f"\nLoading LOCAL 3-qubit B-gate from '{LOCAL_GATE_FILE}'...")
    b_local_matrix = np.load(LOCAL_GATE_FILE)
    local_b_gate = UnitaryGate(b_local_matrix, label="B_loc")
    
    # Basic check: The local gate itself must be unitary
    is_unitary = np.allclose(b_local_matrix @ b_local_matrix.conj().T, np.identity(8))
    print(f"  Local 3-qubit gate is unitary? {'Yes' if is_unitary else 'No'}")
    if not is_unitary:
        print("  ❌ ERROR: The loaded local gate matrix is not unitary. Aborting.")
        exit()

    # === Part 2: Prepare Initial State for N Qubits ===
    try:
        initial_qic_state = prepare_qic_superposition_state(N)
    except Exception as e:
        print(f"\nERROR during state preparation: {e}")
        traceback.print_exc()
        exit()

    # === Part 3: Run Quantum Verification Protocols ===
    ybe_passed = run_yang_baxter_test(initial_qic_state, local_b_gate, N)
    subspace_passed = run_subspace_preservation_test(initial_qic_state, local_b_gate, N)
    
    all_tests_passed = ybe_passed and subspace_passed

    print("\n-------------------")
    print("--- Test Summary ---")
    print("-------------------")
    print(f"Yang-Baxter Relations Check:        {'PASSED' if ybe_passed else 'FAILED'}")
    print(f"Subspace Preservation Check: {'PASSED' if subspace_passed else 'FAILED'}")
    print("==========================================================")
    print(f"Total Script Execution Time: {time.time() - master_start_time:.3f} seconds")

    if all_tests_passed:
        print(f"\n🎉 All scalable quantum verification checks passed for N={N}! 🎉")
    else:
        print(f"\n❗ One or more checks failed for N={N}.")