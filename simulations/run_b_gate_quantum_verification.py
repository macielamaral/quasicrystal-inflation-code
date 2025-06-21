# run_b_gate_quantum_verification.py
#
# Implements quantum verification protocols for the custom 3-qubit B-gate.
# This script uses Qiskit's statevector simulator to perform ideal, noiseless
# checks for Unitarity, the Yang-Baxter Equation, and QIC Subspace Preservation.

import numpy as np
import os
import traceback
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
# Path to the pre-built and saved 3-qubit braid operator matrix
GATE_MATRIX_FILE = "data/gates/b_local_matrix.npy"

# Use the statevector simulator for ideal fidelity/energy calculations
SIMULATOR = Aer.get_backend('statevector_simulator')


def prepare_qic_superposition_state(n_qubits):
    """
    Uses the qic_core library to prepare an equal superposition of all
    valid QIC basis states for a given number of qubits.

    Returns:
        np.ndarray: A normalized statevector for the initial state.
    """
    print(f"Preparing QIC superposition state for N={n_qubits}...")
    try:
        qic_strings, qic_vectors = qic_core.get_qic_basis(n_qubits)
        if not qic_strings:
            raise ValueError("QIC basis generation failed.")

        dim_qic = len(qic_strings)
        V = qic_core.construct_isometry_V(qic_vectors)
        if V is None:
            raise ValueError("Isometry V construction failed.")

        # Create equal superposition in the QIC basis
        qic_superpos_anyon = np.ones(dim_qic, dtype=complex) / np.sqrt(dim_qic)

        # Embed the state into the full Hilbert space
        psi_superpos_full = V.tocsc() @ qic_superpos_anyon
        norm = np.linalg.norm(psi_superpos_full)
        print(f"  State prepared successfully (norm={norm:.4f}).")

        return psi_superpos_full

    except Exception as e:
        print(f"  ERROR during state preparation: {e}")
        traceback.print_exc()
        return None

# ----------------------------------------------------------------------
# --- Test 1: Quantum Unitarity Verification (Echo Test)             ---
# ----------------------------------------------------------------------
def run_unitarity_test(initial_state, b_gate):
    """
    Verifies B * B_dagger = I by running a quantum echo test.
    The protocol is: |psi_final> = B_dagger * B |psi_initial>.
    Fidelity between |psi_final> and |psi_initial> should be 1.
    """
    print("\n--- 1. Running Quantum Unitarity Test (Echo Protocol) ---")
    n_qubits = initial_state.num_qubits
    b_gate_dg = b_gate.inverse()
    b_gate_dg.label = "B†"

    # Create the quantum circuit
    qc = QuantumCircuit(n_qubits)
    qc.initialize(initial_state, qc.qubits)
    qc.barrier()
    qc.append(b_gate, range(n_qubits))    # Apply B
    qc.append(b_gate_dg, range(n_qubits)) # Apply B_dagger
    print("  Circuit: |psi_final> = B† B |psi_initial>")
    
    # Execute and get the final statevector
    final_sv = Statevector(SIMULATOR.run(qc).result().get_statevector())

    # Calculate fidelity
    fidelity = state_fidelity(initial_state, final_sv)
    print(f"  ✅ Result: Fidelity(|psi_initial>, |psi_final|) = {fidelity:.8f}")

    if not np.isclose(fidelity, 1.0):
        print("  ❌ WARNING: Unitarity test failed! Fidelity is not 1.")
    else:
        print("  SUCCESS: B-gate is unitary.")
    return np.isclose(fidelity, 1.0)

# ----------------------------------------------------------------------
# --- Test 2: Quantum Braid Relation Verification (Yang-Baxter)    ---
# ----------------------------------------------------------------------
def run_yang_baxter_test(initial_state_n4, b_gate):
    """
    Verifies the Yang-Baxter equation: B_0 B_1 B_0 = B_1 B_0 B_1.
    We generate the LHS and RHS states and check their fidelity.
    """
    print("\n--- 2. Running Quantum Braid Relation Test (Yang-Baxter Eq.) ---")
    n_qubits = 4

    # --- LHS Circuit: B_0 B_1 B_0 |psi> ---
    qc_lhs = QuantumCircuit(n_qubits, name="LHS")
    qc_lhs.initialize(initial_state_n4, qc_lhs.qubits)
    qc_lhs.barrier()
    qc_lhs.append(b_gate, [0, 1, 2]) # B_0
    qc_lhs.append(b_gate, [1, 2, 3]) # B_1
    qc_lhs.append(b_gate, [0, 1, 2]) # B_0
    print("  Circuit A (LHS): |psi_LHS> = B_0 B_1 B_0 |psi_initial>")
    sv_lhs = Statevector(SIMULATOR.run(qc_lhs).result().get_statevector())

    # --- RHS Circuit: B_1 B_0 B_1 |psi> ---
    qc_rhs = QuantumCircuit(n_qubits, name="RHS")
    qc_rhs.initialize(initial_state_n4, qc_rhs.qubits)
    qc_rhs.barrier()
    qc_rhs.append(b_gate, [1, 2, 3]) # B_1
    qc_rhs.append(b_gate, [0, 1, 2]) # B_0
    qc_rhs.append(b_gate, [1, 2, 3]) # B_1
    print("  Circuit B (RHS): |psi_RHS> = B_1 B_0 B_1 |psi_initial>")
    sv_rhs = Statevector(SIMULATOR.run(qc_rhs).result().get_statevector())

    # Calculate fidelity between the two resulting states
    fidelity = state_fidelity(sv_lhs, sv_rhs)
    print(f"  ✅ Result: Fidelity(|psi_LHS>, |psi_RHS|) = {fidelity:.8f}")

    if not np.isclose(fidelity, 1.0):
        print("  ❌ WARNING: Yang-Baxter relation test failed! States do not match.")
    else:
        print("  SUCCESS: B-gate satisfies the Yang-Baxter equation.")
    return np.isclose(fidelity, 1.0)


# ----------------------------------------------------------------------
# --- Test 3: Quantum Subspace Preservation Verification (Energy)  ---
# ----------------------------------------------------------------------
def run_subspace_preservation_test(initial_state, b_gate):
    """
    Verifies that the B-gate preserves the QIC subspace.
    We start with a state in the subspace (energy=0), apply the gate,
    and measure the energy of the final state. It should still be 0.
    """
    print("\n--- 3. Running Quantum Subspace Preservation Test (Energy Measurement) ---")
    n_qubits = initial_state.num_qubits

    # 1. Build the Hamiltonian H_QIC
    hamiltonian = qic_core.build_qic_hamiltonian_op(n_qubits, verbose=False)
    if hamiltonian is None:
        print("  Could not build Hamiltonian. Skipping test.")
        return False

    # 2. Prepare the braided state
    qc = QuantumCircuit(n_qubits)
    qc.initialize(initial_state, qc.qubits)
    qc.barrier()
    # Apply the braid gate on the first 3 qubits as an example
    qc.append(b_gate, range(3))
    print(f"  Circuit: |psi_final> = B_0 |psi_initial> (on N={n_qubits} qubits)")
    final_state_vector = SIMULATOR.run(qc).result().get_statevector().data

    # 3. Verify the energy of the final state using qic_core's function
    print("  Measuring energy of the final state <psi_final|H|psi_final>...")
    is_gs, energy = qic_core.verify_energy(
        final_state_vector, hamiltonian, n_qubits, label="|psi_final>", verbose=False
    )
    print(f"  ✅ Result: Measured Energy = {energy.real:.8f}")

    if not is_gs:
        print("  ❌ WARNING: Subspace preservation test failed! Energy is non-zero.")
    else:
        print("  SUCCESS: B-gate preserves the QIC subspace.")
    return is_gs


if __name__ == "__main__":
    print("==========================================================")
    print("=== Quantum Verification Suite for Local 3-Qubit B-Gate ===")
    print("==========================================================")

    # 1. Load the custom gate from the file
    if not os.path.exists(GATE_MATRIX_FILE):
        print(f"ERROR: Gate file not found at '{GATE_MATRIX_FILE}'")
        print("Please run the qic_core.py main script first to generate it.")
        exit()

    print(f"\nLoading B-gate matrix from '{GATE_MATRIX_FILE}'...")
    b_local_matrix = np.load(GATE_MATRIX_FILE)
    b_gate = UnitaryGate(b_local_matrix, label="B")
    print("  Custom Qiskit UnitaryGate 'B' created successfully.")

    # 2. Prepare initial states needed for the tests
    # For Unitarity test on the 3-qubit gate
    psi_init_n3 = Statevector(prepare_qic_superposition_state(n_qubits=3))
    # For Yang-Baxter test which needs 4 qubits
    psi_init_n4 = Statevector(prepare_qic_superposition_state(n_qubits=4))

    if psi_init_n3 is None or psi_init_n4 is None:
        print("\nAborting due to state preparation failure.")
        exit()

    # 3. Run the verification protocols
    results = []
    results.append(run_unitarity_test(psi_init_n3, b_gate))
    results.append(run_yang_baxter_test(psi_init_n4, b_gate))
    results.append(run_subspace_preservation_test(psi_init_n4, b_gate))

    print("\n-------------------")
    print("--- Test Summary ---")
    print("-------------------")
    print(f"Unitarity Check:          {'PASSED' if results[0] else 'FAILED'}")
    print(f"Yang-Baxter Check:        {'PASSED' if results[1] else 'FAILED'}")
    print(f"Subspace Preservation Check: {'PASSED' if results[2] else 'FAILED'}")
    print("==========================================================")

    if all(results):
        print("\n🎉 All quantum verification checks passed successfully! 🎉")
    else:
        print("\n❗ One or more quantum verification checks failed.")