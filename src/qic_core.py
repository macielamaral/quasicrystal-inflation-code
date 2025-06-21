# -*- coding: utf-8 -*-
"""
qic_core.py
===========

Core utilities for the *Quasicrystal Inflation Code* (QIC) framework.

This module provides the algebraic and numerical backbone used in
“Consistent Simulation of Fibonacci Anyon Braiding within a Qubit
Quasicrystal Inflation Code” (Amaral 2025).  It is designed to be
imported by small driver scripts (e.g. `run_qic_ibm_leakage.py`) as well
as used interactively in Jupyter.

-----------------------------------------------------------------------
Key capabilities
-----------------------------------------------------------------------
* **Basis generation**
  - Enumerate all binary strings of length *N* obeying the “no 00” rule.
  - Produce explicit basis vectors (NumPy) and check the dimension
    equals F\_{N+2}.

* **Isometry construction**
  - Build the tall, column-orthonormal matrix *V* that embeds the QIC
    subspace into the full 2ⁿ-dimensional Hilbert space.

* **Anyonic operators (abstract level)**
  - Implement Kauffman–Lomonaco rules to construct Temperley-Lieb
    projectors \(P_k^{\text{anyon}}\) and braid generators
    \(B_k^{\text{anyon}}\) on the F\_{N+2} basis.

* **Embedded operators (physical qubit level)**
  - Compute \(P_k' = V P_k^{\text{anyon}} V^\dagger\) and
    \(B_k' = R_I P_k' + R_τ (I - P_k')\).
  - Provide consistency checks: idempotency, Hermiticity, TL relations,
    Yang–Baxter, unitarity.

* **Direct 3-qubit construction (N = 3 sanity check)**
  - Build an 8×8 projector and braid gate without the general pipeline
    and verify they coincide with the embedded versions.

* **Hamiltonian and energy checks (optional, requires Qiskit)**
  - Generate the sparse-Pauli representation of
    \(H_\text{QIC} = \sum_i \Pi^{(00)}_{i+1,i}\).
  - Use Aer EstimatorV2 to confirm that candidate states have zero
    energy (ground-state compliance).

-----------------------------------------------------------------------
Quick example
-----------------------------------------------------------------------
>>> from qic_core import get_qic_basis, construct_isometry_V
>>> strings, vecs = get_qic_basis(5)
>>> V = construct_isometry_V(vecs)
>>> print(strings[:5])
['01010', '01011', '01101', '01110', '01111']

-----------------------------------------------------------------------
Dependencies
-----------------------------------------------------------------------
* NumPy, SciPy (≥ 1.10)
* **Optional:** Qiskit (≥ 0.46) and Qiskit Aer for Hamiltonian and energy
  utilities.

The module degrades gracefully if Qiskit is absent: QIC algebra still
works; only Hamiltonian/energy helpers are disabled.

-----------------------------------------------------------------------
Author & version
-----------------------------------------------------------------------
Marcelo M. Amaral with assistence of Google AI
v1.1 — June 2025

-----------------------------------------------------------------------
"""

import numpy as np
# Use scipy sparse for matrix operations and checks
from scipy.sparse import identity as sparse_identity
from scipy.sparse import kron, csc_matrix, csr_matrix, eye as sparse_eye, diags, block_diag, lil_matrix
from scipy.sparse.linalg import norm as sparse_norm
import time
import traceback
import os

# --- Qiskit Imports (Optional, needed for H_QIC and energy checks) ---
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit.circuit.library import UnitaryGate
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator
    QISKIT_AVAILABLE = True
except ImportError:
    print("WARNING: qic_core.py - Qiskit or Qiskit Aer not found.")
    print("         Functions build_qic_hamiltonian_op and verify_energy will not work.")
    QISKIT_AVAILABLE = False
    # Define dummy classes/functions if Qiskit is not available
    class SparsePauliOp: pass
    class AerEstimator: pass
    class UnitaryGate: pass


# --- Constants ---
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = (np.sqrt(5) - 1) / 2
PHI_SQ_INV = 1 / PHI**2 # Constant for TL check Pk Pk+/-1 Pk = delta^-2 Pk
R_TAU_1 = np.exp(1j * 4 * np.pi / 5.0) # Phase for Pk component in Bk
R_TAU_TAU = np.exp(-1j * 3 * np.pi / 5.0) # Phase for (I-Pk) component in Bk
TOL = 1e-9 # Numerical tolerance for checks

# --- Helper: Fibonacci Calculation ---
def fibonacci(k):
    """
    Calculates the k-th Fibonacci number using the convention F0=0, F1=1.
    (F0=0, F1=1, F2=1, F3=2, F4=3, F5=5, F6=8, F7=13, F8=21, ...)
    This means the QIC dimension for N sites is F_{N+2}.
    """
    if k <= 0: return 0
    if k == 1: return 1
    a, b = 0, 1 # Start with F0, F1
    for _ in range(k - 1): a, b = b, a + b
    return b

# --- Step 1: Generate QIC Basis States and Vectors ---
def get_qic_basis(n):
    """
    Generates QIC computational basis strings (no '00') and corresponding
    state vectors in the full 2^n Hilbert space.

    Args:
        n (int): Number of qubits (sites).

    Returns:
        tuple: (list[str], list[np.ndarray])
               - List of sorted QIC basis strings.
               - List of corresponding state vectors (numpy arrays of size 2^n).
               Returns ([], []) on error or if n=0.
    """
    print(f"Generating QIC basis for N={n}...")
    if n == 0: return [], []
    if n == 1:
        strings = ['0', '1']
        vec0 = np.array([1,0], dtype=complex); vec1 = np.array([0,1], dtype=complex)
        expected_dim = fibonacci(n + 2) # F_3 = 2
        print(f"  Found {len(strings)} QIC states for N={n}, expected F_{n+2}={expected_dim}")
        print(f"  Basis strings: {strings}")
        return strings, [vec0, vec1]

    valid_strings = set()
    def generate_recursive_fixed(current_string):
        if len(current_string) == n:
            if "00" not in current_string:
                 valid_strings.add(current_string)
            return
        generate_recursive_fixed(current_string + '1')
        if not current_string or current_string[-1] == '1':
            generate_recursive_fixed(current_string + '0')

    generate_recursive_fixed("")
    # Sort to ensure consistent basis order
    filtered_strings = sorted(list(valid_strings))

    # Check dimension against F_{N+2}
    dim_QIC = len(filtered_strings)
    expected_dim = fibonacci(n + 2)
    print(f"  Found {dim_QIC} QIC states for N={n}, expected F_{n+2}={expected_dim}")
    if dim_QIC != expected_dim:
        raise ValueError(f"Dimension mismatch for N={n}: Expected {expected_dim}, Got {dim_QIC}")

    # Convert strings to vectors
    dim_H = 2**n
    qic_basis_vectors = []
    for s in filtered_strings:
        vec = np.zeros(dim_H, dtype=complex)
        try:
            # Qiskit endianness: rightmost char is qubit 0
            idx = int(s[::-1], 2)
            vec[idx] = 1.0
            qic_basis_vectors.append(vec)
        except Exception as e:
            print(f"Error converting string {s} to vector: {e}")
            return [], [] # Return empty lists on error
    print(f"  Basis vector generation complete.")
    return filtered_strings, qic_basis_vectors

# --- Step 2: Construct Isometry V ---
def construct_isometry_V(qic_basis_vectors):
    """
    Constructs the 2^N x F_{N+2} isometry matrix V from QIC basis vectors.

    Args:
        qic_basis_vectors (list[np.ndarray]): List of 2^N-dimensional vectors
                                             forming the orthonormal QIC basis.

    Returns:
        scipy.sparse.csc_matrix: The isometry V, or None on error.
    """
    if not qic_basis_vectors:
        print("Error: No QIC basis vectors provided for V.")
        return None
    dim_H = len(qic_basis_vectors[0])
    dim_QIC = len(qic_basis_vectors)

    print(f"  Stacking {dim_QIC} vectors of dim {dim_H} into V...")
    try:
        V_matrix = np.stack(qic_basis_vectors, axis=-1)
    except ValueError as e:
        print(f"Error stacking basis vectors: {e}")
        return None

    if V_matrix.shape != (dim_H, dim_QIC):
        print(f"Error: Isometry V shape mismatch. Expected ({dim_H}, {dim_QIC}), got {V_matrix.shape}")
        return None

    print(f"  Converting V to sparse CSC matrix...")
    V_sparse = csc_matrix(V_matrix)

    print("  Verifying V_dag @ V = I...")
    try:
        V_dagger_V = V_sparse.conj().T @ V_sparse
        Identity_QIC = sparse_identity(dim_QIC, dtype=complex, format='csc')
        diff_norm = sparse_norm(V_dagger_V - Identity_QIC)
        if diff_norm > TOL * np.sqrt(dim_QIC):
            print(f"WARNING: V does not satisfy V_dag @ V = I accurately. ||Diff||={diff_norm:.3e}")
        else:
            print(f"  V_dag @ V = I check passed (||Diff||={diff_norm:.3e}).")
    except Exception as e:
        print(f"  Error during V_dag @ V check: {e}")
        traceback.print_exc()

    return V_sparse

# --- Step 3: Build Anyonic Projector P_k^anyon ---
def get_kauffman_Pn_anyon_general(N, k, basis_strings, delta=PHI):
    """
    Constructs the P_k^anyon = U_{k+1}/delta matrix acting on the N-site QIC basis.
    Matrix dimension is F_{N+2} x F_{N+2}. Based on Kauffman's Theorem 2 rules.

    Args:
        N (int): Number of qubits (sites).
        k (int): Projector index (0 <= k <= N-2). Corresponds to U_{k+1}.
        basis_strings (list): Sorted list of QIC strings ('0','1') for size N.
        delta (float/complex): Loop value (usually PHI).

    Returns:
        scipy.sparse.csc_matrix: The F_{N+2} x F_{N+2} P_k^anyon matrix, or None on error.
    """
    print(f"  Building P_{k}^anyon for N={N} using Kauffman rules (index k={k})...")
    if N < 2:
        print(f"ERROR: Kauffman projectors not defined for N < 2.")
        return None
    if not (0 <= k <= N - 2):
        print(f"ERROR: Projector index k={k} invalid for N={N} (must be 0 to {N-2}).")
        return None

    dim_QIC = fibonacci(N + 2)
    if len(basis_strings) != dim_QIC:
        print(f"ERROR: Number of basis strings ({len(basis_strings)}) doesn't match expected dimension F_{N+2}={dim_QIC}.")
        return None

    basis_map = {s: i for i, s in enumerate(basis_strings)}

    a = 1.0 / delta
    b_sq_arg = 1.0 - a**2
    if b_sq_arg < -TOL:
         raise ValueError(f"Cannot compute real b for delta={delta}, 1-1/delta^2 < 0.")
    b = np.sqrt(max(0, b_sq_arg))
    b_sq = max(0, b_sq_arg)

    Pk_anyon_matrix = lil_matrix((dim_QIC, dim_QIC), dtype=complex)

    def to_kauffman(s): return s.replace('0', '*').replace('1', 'P')
    def from_kauffman(s): return s.replace('*', '0').replace('P', '1')

    for col_idx, in_string_01 in enumerate(basis_strings):
        in_string_kauff = to_kauffman(in_string_01)
        results = []

        padded_kauff = 'P' + in_string_kauff + 'P'
        pattern = padded_kauff[k : k + 3]

        if pattern == "P*P":
             results.append((in_string_kauff, a))
             list_in = list(in_string_kauff); list_in[k] = 'P'; results.append(("".join(list_in), b))
        elif pattern == "PPP":
             results.append((in_string_kauff, delta * b_sq))
             list_in = list(in_string_kauff); list_in[k] = '*'; results.append(("".join(list_in), b))
        elif pattern == "*P*":
            results.append((in_string_kauff, delta))
        elif pattern == "*PP" or pattern == "PP*":
             pass
        else:
             pass

        for out_kauff, coeff_U in results:
            out_01 = from_kauffman(out_kauff)
            if out_01 in basis_map:
                row_idx = basis_map[out_01]
                Pk_anyon_matrix[row_idx, col_idx] += coeff_U / delta

    print(f"  Finished building P_{k}^anyon.")
    return Pk_anyon_matrix.tocsc()

# --- Step 3b: Build Anyonic Braid Operator B_k^anyon ---
def get_kauffman_Bn_anyon_general(Pk_anyon):
    """
    Constructs the anyonic braid operator B_k^anyon from a given P_k^anyon.
    The operator is unitary and follows B_k = R1*P_k + Rtau*(I - P_k).

    Args:
        Pk_anyon (scipy.sparse.csc_matrix): The anyonic projector P_k.

    Returns:
        scipy.sparse.csc_matrix: The corresponding anyonic braid operator B_k,
                                 or None on error.
    """
    if Pk_anyon is None:
        print("ERROR: Pk_anyon input to get_kauffman_Bn_anyon_general cannot be None.")
        return None

    print(f"  Building B_k^anyon from P_k^anyon (shape {Pk_anyon.shape})...")
    try:
        dim_qic_space = Pk_anyon.shape[0]
        Id_anyon = sparse_identity(dim_qic_space, dtype=complex, format='csc')
        Pk_anyon_csc = Pk_anyon.tocsc()
        Bk_anyon = R_TAU_1 * Pk_anyon_csc + R_TAU_TAU * (Id_anyon - Pk_anyon_csc)
        print(f"  B_k^anyon built (shape {Bk_anyon.shape}, nnz={Bk_anyon.nnz}).")
        return Bk_anyon.tocsc()
    except Exception as e:
        print(f"  Error during B_k^anyon calculation: {e}")
        traceback.print_exc()
        return None


# --- Step 4: Build Embedded Projector P'_k ---
def build_P_prime_n(n_operator_idx, n_qubits, V_isometry, Pn_anyon_matrix):
    """
    Computes the embedded projector P'_k = V P_k^anyon V^dagger.

    Args:
        n_operator_idx (int): Index k of the projector.
        n_qubits (int): Number of qubits N.
        V_isometry (scipy.sparse.csc_matrix): Isometry V (2^N x F_{N+2}).
        Pn_anyon_matrix (scipy.sparse.csc_matrix): Anyon projector P_k^anyon (F_{N+2} x F_{N+2}).

    Returns:
        scipy.sparse.csc_matrix: Embedded projector P'_k (2^N x 2^N), or None on error.
    """
    if V_isometry is None or Pn_anyon_matrix is None:
        print("Error: Missing V or Pn_anyon for P' construction.")
        return None
    expected_dim_QIC = fibonacci(n_qubits + 2)
    if V_isometry.shape[1] != expected_dim_QIC or Pn_anyon_matrix.shape[0] != expected_dim_QIC:
         print(f"ERROR: Dimension mismatch for V P V_dag calculation.")
         return None

    print(f"  Building Embedded Projector P'_{n_operator_idx} = V P_{n_operator_idx}^anyon V^dagger...")
    try:
        V_csc = V_isometry.tocsc()
        V_dagger_csc = V_csc.conj().T.tocsc()
        Pn_anyon_csc = Pn_anyon_matrix.tocsc()
        temp = V_csc @ Pn_anyon_csc
        P_prime_n = temp @ V_dagger_csc
        print(f"  P'_{n_operator_idx} built (shape {P_prime_n.shape}, nnz={P_prime_n.nnz}).")
        return P_prime_n.tocsc()
    except Exception as e:
        print(f"  Error during P' calculation: {e}")
        traceback.print_exc()
        return None

# --- Step 5: Build Embedded Braid Operator B'_k ---
def build_B_prime_n(n_operator_idx, P_prime_n_matrix, n_qubits):
    """
    Builds the embedded braid operator B'_k = R1*P'_k + Rtau*(Pi_QIC - P'_k),
    where Pi_QIC = V V^dagger is the projector onto the QIC subspace.
    Note: The formula B' = R1*P' + Rtau*(I - P') is used for simplicity, which is
    equivalent within the QIC subspace and acts as Rtau*I outside of it.

    Args:
        n_operator_idx (int): Index k of the operator.
        P_prime_n_matrix (scipy.sparse.csc_matrix): Embedded projector P'_k (2^N x 2^N).
        n_qubits (int): Number of qubits N.

    Returns:
        scipy.sparse.csc_matrix: Embedded braid operator B'_k (2^N x 2^N), or None on error.
    """
    if P_prime_n_matrix is None:
        print(f"Error: P'_{n_operator_idx} matrix is None, cannot build B'_{n_operator_idx}.")
        return None
    dim = 2**n_qubits
    if P_prime_n_matrix.shape != (dim, dim):
        print(f"Error: P'_{n_operator_idx} shape {P_prime_n_matrix.shape} mismatch with expected ({dim},{dim}).")
        return None

    print(f"  Building Braid operator B'_{n_operator_idx} from P'_{n_operator_idx}...")
    Id_N = sparse_identity(dim, format='csc', dtype=complex)
    try:
        P_prime_n_csc = P_prime_n_matrix.tocsc()
        B_prime_n = R_TAU_1 * P_prime_n_csc + R_TAU_TAU * (Id_N - P_prime_n_csc)
        print(f"  B'_{n_operator_idx} built (shape {B_prime_n.shape}, nnz={B_prime_n.nnz}).")
        return B_prime_n.tocsc()
    except Exception as e:
        print(f"  Error during B' calculation: {e}")
        traceback.print_exc()
        return None

# --- Step 6: Alternative: Direct 3-Qubit Matrix Construction ---
# The functions below provide a direct, concrete construction of the 3-qubit
# embedded operators in the full 8x8 Hilbert space. This serves as an
# alternative to the general-N workflow (get_anyon -> build_P_prime) and
# is useful for verification and understanding the N=3 case.

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
    from qic_core.py into a concrete 3-qubit operator matrix.

    The logic implemented here corresponds to P_0 acting on a 3-site chain.
    The resulting matrix is equivalent to P'_0 from `build_P_prime_n` for N=3.
    """
    print("\n--- Building 8x8 Local Projector Matrix P_local (Direct Method) ---")

    # Get basis maps
    s_to_full, _, _ = get_3_qubit_qic_basis_map()

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

    Args:
        p_local (np.ndarray): The 8x8 projector matrix, e.g., from build_local_projector_matrix.

    Returns:
        np.ndarray: The 8x8 braid generator matrix.
    """
    print("\n--- Building 8x8 Braid Generator Matrix B_local (Direct Method) ---")
    if p_local.shape != (8, 8):
        raise ValueError("Input projector matrix must be 8x8.")

    identity_8x8 = np.identity(8, dtype=complex)

    # The formula correctly defines the action on the full 8-dim space.
    # On QIC subspace: B = R1*P + Rtau*(I-P)
    # On forbidden subspace (where P=0): B = Rtau*I
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

# --- Step 7a: Check Anyonic Operator Properties ---
def check_anyon_operator_properties(P_anyon_ops, N, delta=PHI, tol=TOL):
    """
    Performs numerical checks on the list of anyon P_k^anyon operators.
    Checks: P^2=P, P=P^dagger, TL Relations, Commutation.
    """
    print("\n--- Verifying Anyonic Operator Properties (P_k^anyon) ---")
    if N < 2:
         print("Skipping checks for N<2 (no operators).")
         return True
    if not P_anyon_ops or len(P_anyon_ops) != N - 1:
        print(f"ERROR: Incorrect number of anyonic operators provided for N={N}.")
        return False

    all_checks_passed = True
    t = 1.0 / (delta**2) # TL parameter

    for k, Pk_anyon in enumerate(P_anyon_ops):
        label = f"P_{k}^anyon"
        Pk = Pk_anyon.tocsc()

        # Check P^2 = P (Idempotency)
        norm_diff = sparse_norm(Pk @ Pk - Pk)
        holds = norm_diff < tol
        print(f"  {label}^2 = {label}? {holds}. ||P^2-P|| = {norm_diff:.3e}")
        all_checks_passed &= holds

        # Check P = P^dagger (Hermiticity)
        norm_diff = sparse_norm(Pk - Pk.conj().T)
        holds = norm_diff < tol
        print(f"  {label} = {label}^d? {holds}. ||P-P^d|| = {norm_diff:.3e}")
        all_checks_passed &= holds

    # Check TL Relations
    if N >= 3:
        for k in range(N - 2):
            Pk = P_anyon_ops[k].tocsc()
            Pkp1 = P_anyon_ops[k+1].tocsc()
            # Pk Pk+1 Pk = t Pk
            norm_diff = sparse_norm(Pk @ Pkp1 @ Pk - t * Pk)
            holds = norm_diff < tol
            print(f"  TL1 ({k},{k+1}): P{k}P{k+1}P{k} = t P{k}? {holds}. ||Diff|| = {norm_diff:.3e}")
            all_checks_passed &= holds

    # Check Commutation
    if N >= 4:
        for k in range(N - 1):
            for j in range(k + 2, N - 1):
                norm_comm = sparse_norm(P_anyon_ops[k] @ P_anyon_ops[j] - P_anyon_ops[j] @ P_anyon_ops[k])
                holds = norm_comm < tol
                print(f"  Comm ({k},{j}): [P{k},P{j}] = 0? {holds}. ||Comm|| = {norm_comm:.3e}")
                all_checks_passed &= holds

    print("\n--- Anyonic Operator Verification Complete ---")
    return all_checks_passed


# --- Step 7b: Check Embedded Operator Properties ---
def check_embedded_operator_properties(P_prime_ops, B_prime_ops, n, tol=TOL):
    """
    Performs numerical checks on embedded P'_k and B'_k operators.
    Checks: P'^2=P', P'=P^d, TL for P', B B^d=I, Braid Relations for B'.
    """
    print("\n--- Verifying Embedded Operator Properties (P'_k, B'_k) ---")
    if n < 2: return True
    if not P_prime_ops or not B_prime_ops or len(P_prime_ops) != len(B_prime_ops) or len(P_prime_ops) != n - 1:
        print("ERROR: Embedded operator lists are invalid. Skipping checks.")
        return False

    all_checks_passed = True
    dim_H = 2**n
    Id_N = sparse_identity(dim_H, format='csc', dtype=complex)

    print("\n* Checking P' Properties *")
    for idx, P_prime_i in enumerate(P_prime_ops):
        # Idempotency & Hermiticity
        norm_proj = sparse_norm(P_prime_i @ P_prime_i - P_prime_i)
        norm_herm = sparse_norm(P_prime_i - P_prime_i.conj().T)
        holds_proj = norm_proj < tol
        holds_herm = norm_herm < tol
        print(f"  P'_{idx}: P'^2=P'? {holds_proj} (||d||={norm_proj:.2e}), P'=P'^d? {holds_herm} (||d||={norm_herm:.2e})")
        all_checks_passed &= (holds_proj and holds_herm)

    print("\n* Checking B' Properties (Unitarity) *")
    for idx, B_prime_i in enumerate(B_prime_ops):
        norm_unit = sparse_norm(B_prime_i @ B_prime_i.conj().T - Id_N)
        holds = norm_unit < tol
        print(f"  B'_{idx}: BB^d=I? {holds}. ||BB^d-I|| = {norm_unit:.3e}")
        all_checks_passed &= holds

    print("\n* Checking Braid & TL Relations *")
    if n >= 3:
        for idx in range(n - 2):
            P0p, P1p = P_prime_ops[idx], P_prime_ops[idx+1]
            B0p, B1p = B_prime_ops[idx], B_prime_ops[idx+1]
            t = 1 / PHI**2

            # TL Relation
            norm_tl = sparse_norm(P0p @ P1p @ P0p - t * P0p)
            holds_tl = norm_tl < tol
            print(f"  TL(P'_{idx},P'_{idx+1}): P'P'+1P' = t P'? {holds_tl}. ||Diff|| = {norm_tl:.3e}")
            all_checks_passed &= holds_tl

            # Yang-Baxter
            norm_ybe = sparse_norm(B0p @ B1p @ B0p - B1p @ B0p @ B1p)
            holds_ybe = norm_ybe < tol
            print(f"  YBE(B'_{idx},B'_{idx+1}): B'B'+1B' = B'+1B'B'? {holds_ybe}. ||Diff|| = {norm_ybe:.3e}")
            all_checks_passed &= holds_ybe

    print("\n--- Embedded Operator Verification Complete ---")
    return all_checks_passed

# --- Step 8: Build QIC Hamiltonian ---
def build_qic_hamiltonian_op(n, lam=1.0, verbose=True):
    """Builds the QIC Hamiltonian H = lambda * Sum P_i^{00}."""
    if not QISKIT_AVAILABLE:
        print("Qiskit not available, cannot build Hamiltonian.")
        return None
    if n < 2: return SparsePauliOp(['I'*n], coeffs=[0.0 + 0.0j])
    if verbose: print(f"  Building Hamiltonian H_QIC for N={n}...")
    pauli_list, coeff_list = [], []
    term_coeff = 0.25 * lam
    for i in range(n - 1):
        q_i, q_ip1 = n - 1 - i, n - 2 - i
        pauli_i = ['I'] * n; pauli_ip1 = ['I'] * n; pauli_i_ip1 = ['I'] * n
        pauli_i[q_i] = 'Z'
        pauli_ip1[q_ip1] = 'Z'
        pauli_i_ip1[q_i] = 'Z'; pauli_i_ip1[q_ip1] = 'Z'
        pauli_list.extend(['I'*n, "".join(pauli_i), "".join(pauli_ip1), "".join(pauli_i_ip1)])
        coeff_list.extend([term_coeff] * 4)

    hamiltonian_op = SparsePauliOp(pauli_list, coeffs=np.array(coeff_list, dtype=complex)).simplify()
    if verbose: print(f"  H_QIC built (SparsePauliOp, {len(hamiltonian_op)} terms).")
    return hamiltonian_op

# --- Step 9: Verify Energy (using Qiskit EstimatorV2) ---
def verify_energy(state_input, hamiltonian_op, n, label="State", verbose=True):
    """Calculates <state|H|state> using Qiskit EstimatorV2."""
    if not QISKIT_AVAILABLE:
        print("Qiskit not available, cannot verify energy.")
        return False, 0.0j
    if state_input is None or hamiltonian_op is None: return False, 0.0j

    state_data = state_input.flatten()
    if np.linalg.norm(state_data) < TOL:
         print(f"  State for {label} is zero vector. Energy is 0.")
         return True, 0.0j

    try:
        temp_qc = QuantumCircuit(n)
        temp_qc.initialize(state_data, temp_qc.qubits)
    except Exception as e:
        print(f"Error: Failed to initialize QC for {label}: {e}"); return False, 0.0j

    if verbose: print(f"  Verifying energy for {label} (N={n})...")
    try:
        pub = (temp_qc, [hamiltonian_op])
        job = AerEstimator().run(pubs=[pub])
        pub_result = job.result()[0]
        energy = pub_result.data.evs[0]
        is_gs = np.isclose(np.real(energy), 0, atol=TOL*100)
        if verbose:
             print(f"  <{label}|H_QIC|{label}>: {energy:.6f}")
             print(f"  --> Ground State (Energy ~ 0): {is_gs}")
        return is_gs, energy
    except Exception as e:
        print(f"  Estimator V2 calculation failed for {label}: {e}"); return False, 0.0j

# === Example Usage (Optional - typically called from another script) ===
if __name__ == "__main__":
    print("--- Testing Core Functions ---")
    N_test = 3
    print(f"\nTesting with N={N_test}, Expected QIC Dimension: F_{N_test+2} = {fibonacci(N_test+2)}")

    try:
        # --- Method 1: General-N Workflow ---
        print("\n--- METHOD 1: Generating Operators with General Workflow ---")
        test_strings, test_vectors = get_qic_basis(N_test)
        test_V = construct_isometry_V(test_vectors)
        P0_anyon = get_kauffman_Pn_anyon_general(N_test, 0, test_strings)
        P1_anyon = get_kauffman_Pn_anyon_general(N_test, 1, test_strings)
        anyon_checks_passed = check_anyon_operator_properties([P0_anyon, P1_anyon], N_test)
        print(f"Anyon Checks Passed: {anyon_checks_passed}")

        P0_prime = build_P_prime_n(0, N_test, test_V, P0_anyon)
        P1_prime = build_P_prime_n(1, N_test, test_V, P1_anyon)
        B0_prime = build_B_prime_n(0, P0_prime, N_test)
        B1_prime = build_B_prime_n(1, P1_prime, N_test)
        embedded_checks_passed = check_embedded_operator_properties(
            [P0_prime, P1_prime], [B0_prime, B1_prime], N_test
        )
        print(f"\nEmbedded Checks Passed (General Method): {embedded_checks_passed}")

        # --- Method 2: Direct 3-Qubit Construction ---
        print("\n\n--- METHOD 2: Generating Operators with Direct N=3 Functions ---")
        p0_local_direct = build_local_projector_matrix()
        b0_local_direct = build_local_braid_generator(p0_local_direct)

        # --- Verification: Compare Method 1 and Method 2 ---
        print("\n\n--- VERIFICATION: Comparing results from both methods for P'_0 ---")
        p0_prime_dense = P0_prime.toarray()
        are_matrices_close = np.allclose(p0_prime_dense, p0_local_direct, atol=TOL)
        print(f"Is P'_0 (General) == P_local (Direct)?  --> {are_matrices_close}")
        if not are_matrices_close:
            print("Matrices do not match!")
            #print("P'_0 from General Method:\n", p0_prime_dense)
            #print("P_local from Direct Method:\n", p0_local_direct)
        else:
            print("SUCCESS: The general and direct construction methods yield the same projector matrix for N=3, k=0.")

        # Save the verified gate to a file if desired
        OUTPUT_GATE_MATRIX_NPY_FILE = "data/gates/b_local_matrix.npy"
        if b0_local_direct is not None:
             output_dir = os.path.dirname(OUTPUT_GATE_MATRIX_NPY_FILE)
             if not os.path.exists(output_dir):
                 os.makedirs(output_dir)
             np.save(OUTPUT_GATE_MATRIX_NPY_FILE, b0_local_direct)
             print(f"\nSuccessfully saved B_local matrix to {OUTPUT_GATE_MATRIX_NPY_FILE}")


    except Exception as e:
        print(f"\n--- ERROR during qic_core self-test: {e} ---")
        traceback.print_exc()