# batch_unitary_synthesis_analysis.py
# Analyze multiple target 8x8 unitaries (G_ideal), trying several ansatz depths.

import numpy as np
import os
import time
import csv
import glob

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Operator
from scipy.optimize import minimize

# ==== User Configuration ====

#GIDEAL_FILES = [
#    'data/optimal_local_approximators/Gideal_L.npy',
#    'data/optimal_local_approximators/Gideal_M.npy',
#    'data/optimal_local_approximators/Gideal_R.npy',
#    # Add more as needed
#]
# Automatically find all .npy files in the directory
GIDEAL_FILES = sorted(
    glob.glob(
        os.path.join(
            "data", "optimal_local_approximators", "*.npy"
        )
    )
)

NUM_LAYERS_LIST = [1, 2, 3, 4]
MAX_OPTIMIZATION_ITER = 200

# Output summary CSV
RESULTS_CSV = "data/gate_synthesis_results.csv"

# ==== Ansatz Definition ====

def create_3q_ansatz(num_layers):
    num_qubits = 3
    # Each layer: 3 qubits × 3 params (Rz, sx, Rz, sx, Rz), as before
    params_per_single_layer = 3 * num_qubits
    total_params = num_layers * params_per_single_layer
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector('θ', total_params)
    param_idx = 0

    for l_idx in range(num_layers):
        for q_idx in range(num_qubits):
            qc.rz(params[param_idx], q_idx); param_idx += 1
            qc.sx(q_idx)
            qc.rz(params[param_idx], q_idx); param_idx += 1
            qc.sx(q_idx)
            qc.rz(params[param_idx], q_idx); param_idx += 1
        qc.ecr(0, 1)
        qc.ecr(1, 2)
        if num_qubits > 2 and l_idx % 2 == 0:
            qc.ecr(0, 2)

    return qc, params

def objective_function(param_values, ansatz_circuit_template, unbound_params_vector, target_unitary_matrix_np):
    param_dict = {unbound_params_vector[i]: param_values[i] for i in range(len(param_values))}
    bound_circuit = ansatz_circuit_template.assign_parameters(param_dict)
    try:
        ansatz_unitary_op = Operator(bound_circuit)
        ansatz_unitary_np = ansatz_unitary_op.data 
    except Exception as e:
        print(f"Error in unitary extraction: {e}")
        return 2.0 
    d_squared = (2 ** ansatz_circuit_template.num_qubits) ** 2
    trace_val = np.trace(np.conj(target_unitary_matrix_np).T @ ansatz_unitary_np)
    fidelity = (np.abs(trace_val) ** 2) / d_squared
    infidelity = 1.0 - fidelity
    return infidelity

# ==== Main Analysis Loop ====

def run_synthesis(target_file, num_layers, max_iters):
    # 1. Load
    target = np.load(target_file)
    if target.shape != (8,8):
        print(f"ERROR: {target_file} has shape {target.shape}, expected (8,8). Skipping.")
        return None
    if not Operator(target).is_unitary(atol=1e-7):
        print(f"WARNING: {target_file} is not perfectly unitary.")

    # 2. Ansatz
    qc, params = create_3q_ansatz(num_layers)
    if len(params) == 0:
        print("ERROR: Ansatz has no parameters.")
        return None

    # 3. Optimize
    initial_params = np.random.rand(len(params)) * 2 * np.pi
    res = minimize(
        objective_function,
        initial_params,
        args=(qc, params, target),
        method='L-BFGS-B',
        options={'maxiter': max_iters, 'disp': False, 'eps': 1e-9}
    )
    min_infidelity = res.fun
    optimal_params = res.x
    achieved_fidelity = 1.0 - min_infidelity

    # 4. Gate count (transpile to basis for resource count)
    optimized_circuit = qc.assign_parameters({params[i]: optimal_params[i] for i in range(len(optimal_params))})
    transpiled = transpile(optimized_circuit, optimization_level=3)
    ops_counts = transpiled.count_ops()
    two_qubit_gates = sum(ops_counts.get(g, 0) for g in ['cx', 'ecr', 'cz', 'swap', 'rzz', 'rzx', 'zz'])

    return {
        "file": os.path.basename(target_file),
        "num_layers": num_layers,
        "params": optimal_params,
        "min_infidelity": min_infidelity,
        "achieved_fidelity": achieved_fidelity,
        "num_gates_total": sum(ops_counts.values()),
        "num_two_qubit_gates": two_qubit_gates,
        "ops_counts": dict(ops_counts),
    }

def main():
    results = []
    start = time.time()

    print("Starting batch local unitary synthesis analysis...\n")

    for gfile in GIDEAL_FILES:
        for layers in NUM_LAYERS_LIST:
            print(f"Analyzing {gfile} | Layers: {layers}")
            out = run_synthesis(gfile, layers, MAX_OPTIMIZATION_ITER)
            if out:
                results.append(out)
                print(f"  Fidelity: {out['achieved_fidelity']:.6f} | 2Q gates: {out['num_two_qubit_gates']} | Total gates: {out['num_gates_total']}")
            else:
                print("  Synthesis failed for this configuration.")

    # Print best result per file (highest fidelity, then lowest two-qubit gate count)
    print("\n=== Best synthesis per target unitary ===")
    for gfile in GIDEAL_FILES:
        candidates = [r for r in results if r['file'] == os.path.basename(gfile)]
        if not candidates: continue
        best = max(candidates, key=lambda x: (x['achieved_fidelity'], -x['num_two_qubit_gates']))
        print(f"{best['file']} | Fidelity: {best['achieved_fidelity']:.6f} | Layers: {best['num_layers']} | 2Q gates: {best['num_two_qubit_gates']} | Total gates: {best['num_gates_total']}")
        print(f"  OpCounts: {best['ops_counts']}")

    # Write to CSV
    print(f"\nWriting results to {RESULTS_CSV}")
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "num_layers", "min_infidelity", "achieved_fidelity",
            "num_gates_total", "num_two_qubit_gates", "ops_counts", "params"
        ])
        writer.writeheader()
        for r in results:
            row = r.copy()
            row['ops_counts'] = str(row['ops_counts'])
            row['params'] = np.array2string(row['params'], separator=',', precision=4)
            writer.writerow(row)


    print(f"\nTotal elapsed time: {time.time()-start:.1f} seconds.")

if __name__ == "__main__":
    main()
