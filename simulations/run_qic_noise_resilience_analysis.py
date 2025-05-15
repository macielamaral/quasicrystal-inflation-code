import numpy as np
import time
import csv
import os
import qic_core

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ===== Parameters to sweep =====

N_QUBITS_LIST = [2, 3, 4]  # Add more as needed
# No longer need INITIAL_STATE_STR_QIC global
P_1Q_ERROR_LIST = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2]
P_2Q_ERROR_LIST = [1e-3, 5e-3, 1e-2, 5e-2, 0.1]
SEQUENCE_LENGTH_LIST = [1, 3, 5, 7, 10]
DEFAULT_BRAID_INDEX = 0  # Can randomize if needed

RESULTS_CSV = 'data/qic_noise_resilience_results.csv'  # Save in data/

# Ensure output dir exists
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

def get_initial_qic_string(N_QUBITS):
    # Returns a simple valid QIC string: '1010...', works for all N >= 1
    # Alternatively, could load from qic_strings[0] for more randomness
    return ''.join(['1' if i%2==0 else '0' for i in range(N_QUBITS)])

def apply_qic_sequence_noiseless(psi_init_np, braid_sequence_config, B_prime_ops, B_prime_dagger_ops):
    psi_current = psi_init_np.copy()
    for braid_op_info in braid_sequence_config:
        k_qic = braid_op_info['index']
        is_inverse = braid_op_info['inverse']
        op_matrix = B_prime_dagger_ops[k_qic] if is_inverse else B_prime_ops[k_qic]
        psi_current = op_matrix @ psi_current
    return psi_current / np.linalg.norm(psi_current)

def run_experiment(N_QUBITS, P_1Q, P_2Q, sequence_length):
    # Setup QIC
    qic_strings, qic_vectors = qic_core.get_qic_basis(N_QUBITS)
    dim_QIC = len(qic_strings)
    V_iso = qic_core.construct_isometry_V(qic_vectors)
    V_csc = V_iso.tocsc()
    B_prime_ops = []
    B_prime_dagger_ops = []
    for k_idx in range(N_QUBITS - 1):
        Pk_anyon = qic_core.get_kauffman_Pn_anyon_general(N_QUBITS, k_idx, qic_strings, delta=qic_core.PHI)
        Pk_prime = qic_core.build_P_prime_n(k_idx, N_QUBITS, V_csc, Pk_anyon)
        Bk_prime = qic_core.build_B_prime_n(k_idx, Pk_prime, N_QUBITS)
        B_prime_ops.append(Bk_prime.tocsc())
        B_prime_dagger_ops.append(Bk_prime.conj().T.tocsc())

    # Dynamic initial state selection
    current_initial_state_str = get_initial_qic_string(N_QUBITS)
    if current_initial_state_str not in qic_strings:
        # Fallback: use first QIC string if our guess isn't present
        current_initial_state_str = qic_strings[0]
    initial_state_idx = qic_strings.index(current_initial_state_str)
    qic_vec_init = np.zeros(dim_QIC, dtype=complex)
    qic_vec_init[initial_state_idx] = 1.0
    psi_in_np = V_csc @ qic_vec_init
    psi_in_np /= np.linalg.norm(psi_in_np)

    # Build braid sequence (alternating B'_0, B'_1, ...), wrap if needed
    num_braid_types = len(B_prime_ops)
    braid_seq = []
    for i in range(sequence_length):
        braid_idx = i % num_braid_types
        braid_seq.append({'index': braid_idx, 'inverse': (i % 2 == 1)})

    # --- Noiseless simulation ---
    psi_ideal_np = apply_qic_sequence_noiseless(psi_in_np, braid_seq, B_prime_ops, B_prime_dagger_ops)

    # --- Noisy Qiskit simulation ---
    noise_model = NoiseModel()
    error_1q = depolarizing_error(P_1Q, 1)
    error_2q = depolarizing_error(P_2Q, 2)
    primitive_1q_gates = ['id', 'rz', 'sx', 'x']
    primitive_2q_gates = ['cx']
    noise_model.add_all_qubit_quantum_error(error_1q, primitive_1q_gates)
    noise_model.add_all_qubit_quantum_error(error_2q, primitive_2q_gates)

    qc = QuantumCircuit(N_QUBITS)
    qc.initialize(psi_in_np, range(N_QUBITS))
    for step, op_info in enumerate(braid_seq):
        k_qic = op_info['index']
        is_inverse = op_info['inverse']
        qic_op_matrix_sparse = B_prime_dagger_ops[k_qic] if is_inverse else B_prime_ops[k_qic]
        qic_op_matrix_dense = qic_op_matrix_sparse.toarray()
        op_label = f"B'_{k_qic}" if not is_inverse else f"B'dag_{k_qic}"
        qc.unitary(qic_op_matrix_dense, range(N_QUBITS), label=f"{op_label}_step{step}")
    qc.save_statevector()

    sim_aer_noisy = AerSimulator(noise_model=noise_model)
    tqc = transpile(qc, backend=sim_aer_noisy, optimization_level=1)
    result_noisy = sim_aer_noisy.run(tqc).result()
    if 'statevector' in result_noisy.data(0):
        psi_noisy_np = np.array(result_noisy.data(0)['statevector'])
    else:
        raise RuntimeError("Statevector not found in result.")

    psi_noisy_np /= np.linalg.norm(psi_noisy_np)
    fidelity = np.abs(np.vdot(psi_ideal_np, psi_noisy_np)) ** 2
    return {
        'N_QUBITS': N_QUBITS,
        'P_1Q_ERROR': P_1Q,
        'P_2Q_ERROR': P_2Q,
        'SEQ_LEN': sequence_length,
        'INIT_STATE': current_initial_state_str,
        'FIDELITY': fidelity,
        'CIRCUIT_DEPTH': tqc.depth(),
        'TOTAL_OPS': sum(tqc.count_ops().values()),
    }


if __name__ == "__main__":
    results = []
    t0 = time.time()
    print("--- QIC Noise Resilience Sweep ---")

    # Header for CSV
    with open(RESULTS_CSV, 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['N_QUBITS', 'P_1Q_ERROR', 'P_2Q_ERROR', 'SEQ_LEN', 'INIT_STATE', 'FIDELITY', 'CIRCUIT_DEPTH', 'TOTAL_OPS'])

    for N_QUBITS in N_QUBITS_LIST:
        for P_1Q in P_1Q_ERROR_LIST:
            for P_2Q in P_2Q_ERROR_LIST:
                for SEQ_LEN in SEQUENCE_LENGTH_LIST:
                    try:
                        print(f"Running: N={N_QUBITS} P1Q={P_1Q} P2Q={P_2Q} L={SEQ_LEN}")
                        res = run_experiment(N_QUBITS, P_1Q, P_2Q, SEQ_LEN)
                        results.append(res)
                        with open(RESULTS_CSV, 'a', newline='') as f_csv:
                            writer = csv.writer(f_csv)
                            writer.writerow([res['N_QUBITS'], res['P_1Q_ERROR'], res['P_2Q_ERROR'],
                                             res['SEQ_LEN'], res['INIT_STATE'], res['FIDELITY'],
                                             res['CIRCUIT_DEPTH'], res['TOTAL_OPS']])
                    except Exception as e:
                        print(f"Error for N={N_QUBITS}, P1Q={P_1Q}, P2Q={P_2Q}, L={SEQ_LEN}: {e}")

    print(f"Done. Results saved to {RESULTS_CSV}")
    print(f"Elapsed: {time.time() - t0:.1f}s")
