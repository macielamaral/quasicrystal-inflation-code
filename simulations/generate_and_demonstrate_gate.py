import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
import traceback
import os

# --- Constants ---
# We'll define these locally so the script is self-contained,
# but ideally, you would import them from your qic_core.py.
print("--- Initializing Constants and QIC Subspace ---")
PHI = (1 + np.sqrt(5)) / 2
# Braid generator eigenvalues from qic_core.py
R_TAU_1 = np.exp(1j * 4 * np.pi / 5.0)
R_TAU_TAU = np.exp(-1j * 3 * np.pi / 5.0)
TOL = 1e-9

# to save the gate
OUTPUT_GATE_MATRIX_NPY_FILE = "data/gates/b_local_matrix.npy"

def get_3_qubit_qic_basis_map():
    """
    Defines the valid 3-qubit QIC subspace and creates helper maps.
    The "no 00" rule on 3 qubits forbids |000>, |001>, and |100>.
    """
    # The 5 valid states forming the QIC subspace within the 8-state space
    valid_strings = ['010', '011', '101', '110', '111']
    
    # Map from string to its integer value in the full 8-dim space
    # e.g., '101' -> 5
    string_to_full_idx = {s: int(s, 2) for s in valid_strings}

    # Map from string to its index within the 5-dim QIC subspace
    # e.g., '010' -> 0, '011' -> 1, ...
    string_to_subspace_idx = {s: i for i, s in enumerate(valid_strings)}

    # Reverse map for convenience
    subspace_idx_to_string = {i: s for s, i in string_to_subspace_idx.items()}

    print(f"3-Qubit QIC Subspace (5 states): {valid_strings}")
    return string_to_full_idx, string_to_subspace_idx, subspace_idx_to_string


def build_local_projector_matrix(delta=PHI):
    """
    Constructs the 8x8 matrix for the local projector P_k.
    This is the core of the logic, translating the abstract Kauffman rules
    from your qic_core.py into a concrete 3-qubit operator matrix.
    
    The logic implemented here corresponds to P_0 acting on a 3-site chain.
    """
    print("\n--- Building 8x8 Local Projector Matrix P_local ---")
    
    # Get basis maps
    s_to_full, s_to_sub, sub_to_s = get_3_qubit_qic_basis_map()

    # Initialize an 8x8 zero matrix for our P_local operator
    p_local = np.zeros((8, 8), dtype=complex)
    
    # Kauffman algebra constants
    a = 1.0 / delta
    b_sq = 1.0 - a**2
    b = np.sqrt(b_sq)
    
    # --- Apply Kauffman's rules to define the action on the valid subspace ---
    # We are building the matrix column by column. The action on a basis state |s>
    # defines the column corresponding to that state's index.

    # Rule for |101> ('P*P'): P|101> = (1/delta^2)|101> + (b/delta)|111>
    in_str = '101'
    col_idx = s_to_full[in_str]
    p_local[s_to_full['101'], col_idx] = 1 / (delta**2)
    p_local[s_to_full['111'], col_idx] = b / delta
    print(f"Rule for '{in_str}': Mapped to linear combination of '101' and '111'.")

    # Rule for |111> ('PPP'): P|111> = (b/delta)|101> + b^2|111>
    in_str = '111'
    col_idx = s_to_full[in_str]
    p_local[s_to_full['101'], col_idx] = b / delta
    p_local[s_to_full['111'], col_idx] = b_sq
    print(f"Rule for '{in_str}': Mapped to linear combination of '101' and '111'.")

    # Rule for |010> ('*P*'): P|010> = 1 * |010>
    in_str = '010'
    col_idx = s_to_full[in_str]
    p_local[s_to_full['010'], col_idx] = 1.0
    print(f"Rule for '{in_str}': Mapped to itself.")
    
    # Rule for |011> ('*PP') and |110> ('PP*'): P projects them to the zero vector.
    # We don't need to do anything as the matrix was initialized to zeros.
    print(f"Rule for '011' and '110': Mapped to zero vector.")

    # On the forbidden subspace {'000', '001', '100'}, P acts as the zero operator.
    # Again, nothing to do as the matrix is already zero there.
    print("Action on forbidden subspace (e.g., |000>, |001>, |100>) is the zero operator.")
    
    return p_local


def build_local_braid_generator(p_local):
    """
    Constructs the 8x8 unitary braid generator B_local using the formula
    B = R1*P + Rtau*(I - P).
    """
    print("\n--- Building 8x8 Braid Generator Matrix B_local ---")
    if p_local.shape != (8, 8):
        raise ValueError("Input projector matrix must be 8x8.")
        
    identity_8x8 = np.identity(8, dtype=complex)
    
    # The formula correctly defines the action on the full 8-dim space.
    # On QIC subspace: B = R1*P + Rtau*(I-P)
    # On forbidden subspace (where P=0): B = R1*0 + Rtau*(I-0) = Rtau*I
    b_local = R_TAU_1 * p_local + R_TAU_TAU * (identity_8x8 - p_local)
    
    # --- Sanity Check: Verify Unitarity ---
    print("Verifying that the resulting B_local matrix is unitary (B_dag * B = I)...")
    try:
        check_matrix = b_local.conj().T @ b_local
        if np.allclose(check_matrix, identity_8x8, atol=TOL):
            print("SUCCESS: B_local is unitary.")
        else:
            print("WARNING: B_local matrix is NOT unitary.")
    except Exception:
        print(f"Error during unitarity check: {traceback.format_exc()}")
        
    return b_local


def demonstrate_usage_in_qiskit():
    """
    Demonstrates how to use the generated B_local matrix as a custom gate
    in Qiskit to build a full braid circuit.
    """
    print("\n--- Demonstrating Usage in a Qiskit Circuit ---")
    
    # 1. Build our custom 3-qubit braid generator gate
    p_local_matrix = build_local_projector_matrix()
    b_local_matrix = build_local_braid_generator(p_local_matrix)

    # ---- Save the 8×8 matrix once it is built ----
       
    os.makedirs(os.path.dirname(OUTPUT_GATE_MATRIX_NPY_FILE), exist_ok=True)  # make sure the folder exists

    np.save(OUTPUT_GATE_MATRIX_NPY_FILE, b_local_matrix)  
    print(f"B_local saved to {OUTPUT_GATE_MATRIX_NPY_FILE}")  
    
    # 2. Create a Qiskit UnitaryGate from our matrix
    # This gate now acts as our fundamental building block, sigma_i.
    B_gate = UnitaryGate(b_local_matrix, label="B")
    
    # 3. Build a circuit for a specific braid on N=5 qubits
    # Example: Braid word [1, -2, 3] for the 12a_122 knot preamble
    num_qubits = 5
    qc = QuantumCircuit(num_qubits, name="Braid [1, -2, 3]")
    
    print(f"\nBuilding circuit for braid word [1, -2, 3] on {num_qubits} qubits...")
    
    # Apply sigma_1 (our B_gate on qubits 0, 1, 2)
    qc.append(B_gate, [0, 1, 2])
    qc.barrier()
    
    # Apply sigma_2^-1 (the inverse of our B_gate on qubits 1, 2, 3)
    qc.append(B_gate.inverse(), [1, 2, 3])
    qc.barrier()
    
    # Apply sigma_3 (our B_gate on qubits 2, 3, 4)
    qc.append(B_gate, [2, 3, 4])
    qc.barrier()
    
    # 4. Print the resulting circuit
    print("\nFinal Quantum Circuit:")
    print(qc.draw(output='text'))
    
if __name__ == "__main__":
    demonstrate_usage_in_qiskit()