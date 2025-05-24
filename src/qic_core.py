# -*- coding: utf-8 -*-
"""
qic_core.py

Core library functions for simulating 
the Quasicrystal Inflation Code (QIC) framework.

Provides functionalities for:
- Generating the QIC basis (no "00" constraint) and corresponding state vectors.
  (Uses Fibonacci convention F0=0, F1=1, dimension F_{N+2})
- Constructing the isometry V mapping the QIC basis to the full qubit space.
- Building abstract anyonic Temperley-Lieb projectors (P_k^anyon) based on
  Kauffman algebra rules (Ref: Kauffman & Lomonaco, arXiv:0705.4349).
- Embedding anyonic operators into the full qubit Hilbert space (P'_k, B'_k)
  using the isometry V.
- Verifying algebraic properties of both anyonic and embedded operators
  (TL algebra, braid relations, unitarity, etc.).
- (Optional, requires Qiskit) Building the QIC Hamiltonian (H_QIC) as a
  SparsePauliOp.
- (Optional, requires Qiskit) Verifying state energy using Qiskit Aer's
  EstimatorV2 to check for ground state preservation.

Main Collaborators: Marcelo Amaral, Google AI
Date: April 2025 [Update as needed]
Version: [e.g., 1.0]
"""

import numpy as np
# Use scipy sparse for matrix operations and checks
from scipy.sparse import identity as sparse_identity
from scipy.sparse import kron, csc_matrix, csr_matrix, eye as sparse_eye, diags, block_diag, lil_matrix
from scipy.sparse.linalg import norm as sparse_norm
import time
import traceback

# --- Qiskit Imports (Optional, needed for H_QIC and energy checks) ---
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    # from qiskit.extensions import UnitaryGate # Use circuit.unitary() instead
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator
    QISKIT_AVAILABLE = True
except ImportError:
    print("WARNING: qic_core.py - Qiskit or Qiskit Aer not found.")
    print("         Functions build_qic_hamiltonian_op and verify_energy will not work.")
    QISKIT_AVAILABLE = False
    # Define dummy classes/functions if Qiskit is not available
    class SparsePauliOp: pass
    class AerEstimator: pass

# --- Constants ---
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = (np.sqrt(5) - 1) / 2
PHI_SQ_INV = PHI # Constant delta for TL check Pk Pk+/-1 Pk = delta^-2 Pk or Ui = delta Pi and Ui^2 = delta Ui
R_TAU_1 = np.exp(1j * 4 * np.pi / 5.0) # Phase for Pk component in Bk
R_TAU_TAU = np.exp(-1j * 3 * np.pi / 5.0) # Phase for (I-Pk) component in Bk R_TAU_TAU = A^-1 from Amaral2022 or KauffmanLomonaco
# Phi * A = R_TAU_1 - R_TAU_TAU such that we can go back to Eq.14 in Amaral2022. In KauffmanLomonaco or B is their B^-1
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
    #print(f"  Generating {dim_QIC} vectors (dim {dim_H}) for basis: {filtered_strings}")
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
    # Infer N from vector dimension
    dim_H = len(qic_basis_vectors[0])
    # n_qubits = int(np.log2(dim_H)) # Calculate N if needed later
    dim_QIC = len(qic_basis_vectors) # F_{N+2}

    print(f"  Stacking {dim_QIC} vectors of dim {dim_H} into V...")
    try:
        V_matrix = np.stack(qic_basis_vectors, axis=-1) # Stack as columns
        print(f"  Stacked V shape: {V_matrix.shape}")
    except ValueError as e:
        print(f"Error stacking basis vectors: {e}")
        return None

    # Verify shape
    if V_matrix.shape != (dim_H, dim_QIC):
        print(f"Error: Isometry V shape mismatch. Expected ({dim_H}, {dim_QIC}), got {V_matrix.shape}")
        return None

    print(f"  Converting V to sparse CSC matrix...")
    V_sparse = csc_matrix(V_matrix)

    # --- Verification V_dag @ V = I (Optional but recommended) ---
    print("  Verifying V_dag @ V = I...")
    try:
        V_dagger_V = V_sparse.conj().T @ V_sparse
        Identity_QIC = sparse_identity(dim_QIC, dtype=complex, format='csc')
        # Use Frobenius norm for difference check
        diff_norm = sparse_norm(V_dagger_V - Identity_QIC)
        # Check absolute difference norm
        if diff_norm > TOL * np.sqrt(dim_QIC): # Allow roughly TOL per element
            print(f"WARNING: V does not satisfy V_dag @ V = I accurately. ||Diff||={diff_norm:.3e}")
        else:
            print(f"  V_dag @ V = I check passed (||Diff||={diff_norm:.3e}).")
    except Exception as e:
        print(f"  Error during V_dag @ V check: {e}")
        traceback.print_exc()
        # Don't necessarily fail here, allow continuation
        # return None
    # --- End Verification ---

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

    dim_QIC = fibonacci(N + 2) # Expect F_{N+2} dimension
    if len(basis_strings) != dim_QIC:
        print(f"ERROR: Number of basis strings ({len(basis_strings)}) doesn't match expected dimension F_{N+2}={dim_QIC}.")
        return None

    # Map basis strings to indices for quick lookup
    basis_map = {s: i for i, s in enumerate(basis_strings)}

    # Constants from Theorem 2
    a = 1.0 / delta
    if np.isclose(delta, PHI):
        b = 1.0 / np.sqrt(PHI)
        b_sq = 1.0 / PHI
    else: # Add handling for other delta if needed
        print(f"WARNING: Using delta = {delta}, not PHI. Ensure constants a,b,b_sq are correct.")
        b_sq_arg = 1.0 - a**2
        if b_sq_arg < -TOL: # Check if significantly negative
             raise ValueError(f"Cannot compute real b for delta={delta}, 1-1/delta^2 < 0.")
        b = np.sqrt(max(0, b_sq_arg)) # Ensure non-negative argument for sqrt
        b_sq = max(0, b_sq_arg)

    # Use lil_matrix for efficient construction (needs import)
    Pk_anyon_matrix = lil_matrix((dim_QIC, dim_QIC), dtype=complex)

    # Helper to map '0'/'1' string to 'P'/'*' string (Kauffman notation)
    def to_kauffman(s): return s.replace('0', '*').replace('1', 'P')
    # Helper to map 'P'/'*' string back to '0'/'1'
    def from_kauffman(s): return s.replace('*', '0').replace('P', '1')

    for col_idx, in_string_01 in enumerate(basis_strings):
        in_string_kauff = to_kauffman(in_string_01)
        results = [] # Store tuples of (output_kauffman_string, coefficient_U)

        # Pad with 'P' for boundary conditions
        padded_kauff = 'P' + in_string_kauff + 'P'
        # Check pattern around original sites k, k+1 (indices k, k+1, k+2 in padded)
        pattern = padded_kauff[k : k + 3] # Slice indices k, k+1, k+2

        # Apply the simplified "middle rules" everywhere approach
        # Based on U_{k+1} action from Theorem 2
        if pattern == "P*P":
             results.append((in_string_kauff, a)) # U|..P*P..> = a|..P*P..> + ...
             list_in = list(in_string_kauff); list_in[k] = 'P'; results.append(("".join(list_in), b)) # ... + b|..PPP..>
        elif pattern == "PPP":
             results.append((in_string_kauff, delta * b_sq)) # U|..PPP..> = ... + delta*b^2|..PPP..>
             list_in = list(in_string_kauff); list_in[k] = '*'; results.append(("".join(list_in), b)) # U|..PPP..> = b|..P*P..> + ...
        elif pattern == "*P*":
            results.append((in_string_kauff, delta)) # U|..*P*..> = delta |..*P*..>
        elif pattern == "*PP" or pattern == "PP*":
             pass # U acts as 0
        else:
             pass # Should not happen for valid QIC inputs

        # Process results: P_k = U_{k+1} / delta
        for out_kauff, coeff_U in results:
            out_01 = from_kauffman(out_kauff)
            if out_01 in basis_map:
                row_idx = basis_map[out_01]
                Pk_anyon_matrix[row_idx, col_idx] += coeff_U / delta
            # else: output string not in basis (e.g., contains "00"), contribution is 0

    print(f"  Finished building P_{k}^anyon.")
    return Pk_anyon_matrix.tocsc()


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
    dim_H, dim_QIC_V = V_isometry.shape
    dim_QIC_P_r, dim_QIC_P_c = Pn_anyon_matrix.shape

    # Dimension check against expected F_{N+2}
    expected_dim_QIC = fibonacci(n_qubits + 2)
    if dim_QIC_V != expected_dim_QIC or dim_QIC_P_r != expected_dim_QIC or dim_QIC_P_c != expected_dim_QIC:
         print(f"ERROR: Dimension mismatch for V P V_dag calculation.")
         print(f"  N={n_qubits}, Expected QIC Dim F_{N+2}={expected_dim_QIC}")
         print(f"  V shape: {V_isometry.shape} (QIC dim {dim_QIC_V})")
         print(f"  P_anyon shape: {Pn_anyon_matrix.shape}")
         return None

    print(f"  Building Embedded Projector P'_{n_operator_idx} = V P_{n_operator_idx}^anyon V^dagger...")
    try:
        # Ensure correct formats for sparse matrix multiplication
        V_csc = V_isometry.tocsc()
        V_dagger_csc = V_csc.conj().T.tocsc()
        Pn_anyon_csc = Pn_anyon_matrix.tocsc() # Should already be CSC

        # Perform multiplication (CSC @ CSC -> CSC)
        temp = V_csc @ Pn_anyon_csc
        P_prime_n = temp @ V_dagger_csc
        print(f"  P'_{n_operator_idx} built (shape {P_prime_n.shape}, nnz={P_prime_n.nnz}).")
        return P_prime_n.tocsc() # Keep as CSC
    except Exception as e:
        print(f"  Error during P' calculation: {e}")
        traceback.print_exc()
        return None

# --- Step 5: Build Embedded Braid Operator B'_k ---
def build_B_prime_n(n_operator_idx, P_prime_n_matrix, n_qubits):
    """
    Builds the embedded braid operator B'_k = R1*P'_k + Rtau*(I - P'_k),
    where I is identity on the full 2^N space. Note: This acts as
    B'_k = R1*P'_k + Rtau*(Pi_QIC - P'_k) within the QIC subspace,
    where Pi_QIC = V V_dagger.

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
    Id_N = sparse_identity(dim, format='csc', dtype=complex) # Use sparse_identity
    try:
        P_prime_n_csc = P_prime_n_matrix.tocsc()
        # Formula uses P'_k (embedded projector)
        # This definition ensures B'_k acts as identity outside the QIC subspace if P'_k is zero there.
        #B_prime_n = R_TAU_1 * P_prime_n_csc + R_TAU_TAU * (Id_N - P_prime_n_csc)
        B_prime_n = R_TAU_1 * P_prime_n_csc + R_TAU_TAU * (Id_N - P_prime_n_csc)
        #B_prime_n = PHI * R_TAU_1 * P_prime_n_csc + PHI * R_TAU_TAU * (Id_N - P_prime_n_csc)
        print(f"  B'_{n_operator_idx} built (shape {B_prime_n.shape}, nnz={B_prime_n.nnz}).")
        return B_prime_n.tocsc()
    except Exception as e:
        print(f"  Error during B' calculation: {e}")
        traceback.print_exc()
        return None

# --- Step 6a: Check Anyonic Operator Properties ---
def check_anyon_operator_properties(P_anyon_ops, N, delta=PHI, tol=TOL):
    """
    Performs numerical checks on the list of anyon P_k^anyon operators.
    Checks: P^2=P, P=P^dagger, TL Relations, Commutation.

    Args:
        P_anyon_ops (list[scipy.sparse.csc_matrix]): List of P_k^anyon operators.
        N (int): Number of qubits (sites).
        delta (float/complex): Loop value used in construction.
        tol (float): Numerical tolerance.

    Returns:
        bool: True if all checks pass, False otherwise.
    """
    print("\n--- Verifying Anyonic Operator Properties (P_k^anyon) ---")
    if N < 2:
         print("Skipping checks for N<2 (no operators).")
         return True
    if not P_anyon_ops:
        print("ERROR: Anyonic operator list is empty. Skipping checks.")
        return False

    num_ops_expected = N - 1
    if len(P_anyon_ops) != num_ops_expected:
        print(f"ERROR: Expected {num_ops_expected} anyonic operators for N={N}, but found {len(P_anyon_ops)}.")
        return False

    all_checks_passed = True
    t = 1.0 / (delta**2) # TL parameter
    pk_dim = P_anyon_ops[0].shape[0]
    Id_pk = sparse_identity(pk_dim, format='csc', dtype=complex)

    for k, Pk_anyon in enumerate(P_anyon_ops):
        label = f"P_{k}^anyon"
        Pk = Pk_anyon.tocsc() # Ensure CSC

        # Check P^2 = P (Idempotency)
        Pk_sq = Pk @ Pk
        diff_proj = Pk_sq - Pk
        norm_Pk = sparse_norm(Pk)
        norm_diff = sparse_norm(diff_proj)
        holds = norm_diff < tol * max(1e-12, norm_Pk**2) # Relative check with floor
        print(f"  {label}^2 = {label}? {holds}. ||P^2-P|| = {norm_diff:.3e}")
        if not holds: print(f"     Max element |P^2-P| = {np.max(np.abs(diff_proj.toarray())) if diff_proj.nnz > 0 else 0:.3e}")
        all_checks_passed &= holds

        # Check P = P^dagger (Hermiticity)
        Pk_dag = Pk.conj().T
        diff_herm = Pk - Pk_dag
        norm_diff = sparse_norm(diff_herm)
        holds = norm_diff < tol * max(1e-12, norm_Pk) # Relative check with floor
        print(f"  {label} = {label}^d? {holds}. ||P-P^d|| = {norm_diff:.3e}")
        if not holds: print(f"     Max element |P-P^d| = {np.max(np.abs(diff_herm.toarray())) if diff_herm.nnz > 0 else 0:.3e}")
        all_checks_passed &= holds

    # Check TL Pk Pk+1 Pk = t Pk
    if N >= 3:
        print("\n* Checking TL Relations for P_k^anyon *")
        for k in range(num_ops_expected - 1):
            Pk = P_anyon_ops[k].tocsc()
            Pkp1 = P_anyon_ops[k+1].tocsc()

            # Check Pk Pk+1 Pk = t Pk
            lhs = Pk @ Pkp1 @ Pk
            rhs = t * Pk
            diff = lhs - rhs
            norm_diff = sparse_norm(diff)
            norm_rhs = sparse_norm(rhs)
            holds = norm_diff < tol * max(1e-12, norm_rhs)
            print(f"  TL1 ({k},{k+1}): P{k}P{k+1}P{k} = t P{k}? {holds}. ||Diff|| = {norm_diff:.3e} (t={t:.3f})")
            if not holds: print(f"     Max element |Diff| = {np.max(np.abs(diff.toarray())) if diff.nnz > 0 else 0:.3e}")
            all_checks_passed &= holds

            # Check Pk+1 Pk Pk+1 = t Pk+1
            lhs = Pkp1 @ Pk @ Pkp1
            rhs = t * Pkp1
            diff = lhs - rhs
            norm_diff = sparse_norm(diff)
            norm_rhs = sparse_norm(rhs)
            holds = norm_diff < tol * max(1e-12, norm_rhs)
            print(f"  TL2 ({k},{k+1}): P{k+1}P{k}P{k+1} = t P{k+1}? {holds}. ||Diff|| = {norm_diff:.3e} (t={t:.3f})")
            if not holds: print(f"     Max element |Diff| = {np.max(np.abs(diff.toarray())) if diff.nnz > 0 else 0:.3e}")
            all_checks_passed &= holds
    else:
        print("  Skipping TL relations check (requires N >= 3).")

    # Check TL Commutation Pk Pj = Pj Pk for |k-j|>1
    if N >= 4:
        print("\n* Checking Commutation for P_k^anyon *")
        for k in range(num_ops_expected):
            for j in range(k + 2, num_ops_expected):
                Pk = P_anyon_ops[k].tocsc()
                Pj = P_anyon_ops[j].tocsc()
                comm = Pk @ Pj - Pj @ Pk
                norm_comm = sparse_norm(comm)
                norm_PkPj = sparse_norm(Pk @ Pj) # Norm for relative check
                holds = norm_comm < tol * max(1e-12, norm_PkPj)
                print(f"  TL Comm ({k},{j}): [P{k},P{j}] = 0? {holds}. ||Comm|| = {norm_comm:.3e}")
                if not holds: print(f"     Max element |Comm| = {np.max(np.abs(comm.toarray())) if comm.nnz > 0 else 0:.3e}")
                all_checks_passed &= holds
    else:
         print("  Skipping Commutation check (requires N >= 4).")

    print("\n--- Anyonic Operator Verification Complete ---")
    return all_checks_passed


# --- Step 6b: Check Embedded Operator Properties ---
def check_embedded_operator_properties(P_prime_ops, B_prime_ops, n, tol=TOL):
    """
    Performs numerical checks on the lists of embedded P'_k and B'_k operators.
    Checks: P'^2=P', P'=P^d, TL for P', B B^d=I, Braid Relations for B'.

    Args:
        P_prime_ops (list[scipy.sparse.csc_matrix]): List of P'_k operators.
        B_prime_ops (list[scipy.sparse.csc_matrix]): List of B'_k operators.
        n (int): Number of qubits N.
        tol (float): Numerical tolerance.

    Returns:
        bool: True if all checks pass, False otherwise.
    """
    print("\n--- Verifying Embedded Operator Properties (P'_k, B'_k) ---")
    # Input validation (same as in check_anyon_operator_properties)
    if n < 2:
        print("Skipping checks for N<2 (no operators).")
        return True
    if not P_prime_ops or not B_prime_ops or len(P_prime_ops) != len(B_prime_ops):
        print("ERROR: Embedded operator lists missing or mismatched. Skipping checks.")
        return False
    num_ops = len(P_prime_ops)
    expected_num_ops = n - 1
    if num_ops != expected_num_ops:
         print(f"ERROR: Expected {expected_num_ops} embedded operators for N={n}, but found {num_ops}.")
         return False
    dim_H = 2**n
    Id_N = sparse_identity(dim_H, format='csc', dtype=complex)
    if P_prime_ops[0].shape != (dim_H, dim_H):
        print(f"ERROR: Operator dimension mismatch.")
        return False

    all_checks_passed = True

    # --- Check P' Properties ---
    print("\n* Checking P' Properties *")
    # (Identical checks as in check_anyon_operator_properties, but on P'_k)
    for idx, P_prime_i in enumerate(P_prime_ops):
        label = f"P'_{idx}"
        P_prime_i_csc = P_prime_i.tocsc()
        # Idempotency
        P_prime_sq = P_prime_i_csc @ P_prime_i_csc
        diff_proj = P_prime_sq - P_prime_i_csc
        norm_P = sparse_norm(P_prime_i_csc)
        norm_diff = sparse_norm(diff_proj)
        holds = norm_diff < tol * max(1e-12, norm_P**2) # Relative check
        print(f"  {label}^2 = {label}? {holds}. ||P'^2-P'|| = {norm_diff:.3e}")
        if not holds: print(f"     Max element |P'^2-P'| = {np.max(np.abs(diff_proj.toarray())) if diff_proj.nnz > 0 else 0:.3e}")
        all_checks_passed &= holds
        # Hermiticity
        P_prime_dag = P_prime_i_csc.conj().T
        diff_herm = P_prime_i_csc - P_prime_dag
        norm_diff = sparse_norm(diff_herm)
        holds = norm_diff < tol * max(1e-12, norm_P) # Relative check
        print(f"  {label} = {label}^d? {holds}. ||P'-P'^d|| = {norm_diff:.3e}")
        if not holds: print(f"     Max element |P'-P'^d| = {np.max(np.abs(diff_herm.toarray())) if diff_herm.nnz > 0 else 0:.3e}")
        all_checks_passed &= holds

    # --- Check TL for P' ---
    print("\n* Checking Temperley-Lieb Relation for P' *")
    if n >= 3:
        for idx1 in range(num_ops - 1):
            idx2 = idx1 + 1
            P0p = P_prime_ops[idx1].tocsc()
            P1p = P_prime_ops[idx2].tocsc()
            t = PHI_SQ_INV**(-2)

            # Check P'_i P'_{i+1} P'_i = t P'_i
            lhs_tl = P0p @ P1p @ P0p
            rhs_tl = t * P0p
            diff_tl = lhs_tl - rhs_tl
            norm_P0 = sparse_norm(P0p)
            norm_diff = sparse_norm(diff_tl)
            holds = norm_diff < tol * max(1e-12, t * norm_P0) # Relative check
            print(f"  TL (P'_{idx1}P'_{idx2}P'_{idx1} = t P'_{idx1})? {holds}. ||Diff|| = {norm_diff:.3e} (t=PHI^-2)")
            if not holds: print(f"     Max element |Diff| = {np.max(np.abs(diff_tl.toarray())) if diff_tl.nnz > 0 else 0:.3e}")
            all_checks_passed &= holds

            # Check P'_{i+1} P'_i P'_{i+1} = t P'_{i+1}
            lhs_tl2 = P1p @ P0p @ P1p
            rhs_tl2 = t * P1p
            diff_tl2 = lhs_tl2 - rhs_tl2
            norm_P1 = sparse_norm(P1p)
            norm_diff = sparse_norm(diff_tl2)
            holds = norm_diff < tol * max(1e-12, t * norm_P1) # Relative check
            print(f"  TL (P'_{idx2}P'_{idx1}P'_{idx2} = t P'_{idx2})? {holds}. ||Diff|| = {norm_diff:.3e} (t=PHI^-2)")
            if not holds: print(f"     Max element |Diff| = {np.max(np.abs(diff_tl2.toarray())) if diff_tl2.nnz > 0 else 0:.3e}")
            all_checks_passed &= holds
    else:
        print("  Skipping TL relations check (requires N >= 3).")

    # --- Check B' Properties ---
    print("\n* Checking B' Properties *")
    for idx, B_prime_i in enumerate(B_prime_ops):
        label = f"B'_{idx}"
        B_prime_i_csc = B_prime_i.tocsc()
        # Unitarity
        B_prime_i_dagger = B_prime_i_csc.conj().T
        BBd = B_prime_i_csc @ B_prime_i_dagger
        diff_unit = BBd - Id_N
        norm_diff = sparse_norm(diff_unit)
        holds = norm_diff < tol * np.sqrt(dim_H) # Allow roughly TOL per element
        print(f"  Unitarity ({label}{label}^d=I)? {holds}. ||BB^d-I|| = {norm_diff:.3e}")
        if not holds: print(f"     Max element |BB^d-I| = {np.max(np.abs(diff_unit.toarray())) if diff_unit.nnz > 0 else 0:.3e}")
        all_checks_passed &= holds

    # --- Check Braid Relations for B' ---
    print("\n* Checking Braid Relations for B' *")
    # Yang-Baxter
    if n >= 3:
        for idx1 in range(num_ops - 1):
            idx2 = idx1 + 1
            B0p = B_prime_ops[idx1].tocsc()
            B1p = B_prime_ops[idx2].tocsc()

            lhs_ybe = B0p @ B1p @ B0p
            rhs_ybe = B1p @ B0p @ B1p
            diff_ybe = lhs_ybe - rhs_ybe
            norm_diff = sparse_norm(diff_ybe)
            norm_rhs = sparse_norm(rhs_ybe)
            # Relative check, guarding against norm_rhs being zero
            if norm_rhs > tol * 10: holds = (norm_diff / norm_rhs) < tol
            else: holds = norm_diff < tol * 10 # Absolute check if rhs norm is small

            print(f"  YBE (B'_{idx1}B'_{idx2}B'_{idx1}=B'_{idx2}B'_{idx1}B'_{idx2})? {holds}. ||Diff|| = {norm_diff:.3e}")
            if not holds: print(f"     Max element |Diff| = {np.max(np.abs(diff_ybe.toarray())) if diff_ybe.nnz > 0 else 0:.3e}")
            all_checks_passed &= holds
    else:
        print("  Skipping Yang-Baxter (requires N >= 3).")

    # Commutation
    if n >= 4:
         print("\n* Checking Commutation Relation for B' *")
         for idx1 in range(num_ops):
             for idx2 in range(idx1 + 2, num_ops):
                 Bi = B_prime_ops[idx1].tocsc()
                 Bj = B_prime_ops[idx2].tocsc()
                 comm = Bi @ Bj - Bj @ Bi
                 norm_comm = sparse_norm(comm)
                 norm_BiBj = sparse_norm(Bi @ Bj)
                 # Relative check, guarding against norm being zero
                 if norm_BiBj > tol * 10: holds = (norm_comm / norm_BiBj) < tol
                 else: holds = norm_comm < tol * 10

                 print(f"  Commutation (B'_{idx1}B'_{idx2}=B'_{idx2}B'_{idx1})? {holds}. ||[B'_{idx1}, B'_{idx2}]|| = {norm_comm:.3e}")
                 if not holds: print(f"     Max element |Comm| = {np.max(np.abs(comm.toarray())) if comm.nnz > 0 else 0:.3e}")
                 all_checks_passed &= holds
    else:
         print("  Skipping Commutation check (requires N >= 4).")

    print("\n--- Embedded Operator Verification Complete ---")
    return all_checks_passed

# --- Step 7: Build QIC Hamiltonian ---
def build_qic_hamiltonian_op(n, lam=1.0, verbose=True):
    """Builds the QIC Hamiltonian H = lambda * Sum P_i^{00}."""
    if not QISKIT_AVAILABLE:
        print("Qiskit not available, cannot build Hamiltonian.")
        return None
    if n < 2: return SparsePauliOp(['I'*n], coeffs=[0.0 + 0.0j]) # Return zero op
    if verbose: print(f"  Building Hamiltonian H_QIC for N={n}...")
    pauli_list = []
    coeff_list = []
    # P_i^{00} = |00><00|_{i,i+1} = (I+Z_i)/2 * (I+Z_{i+1})/2
    # = 0.25 * (I + Z_i + Z_{i+1} + Z_i Z_{i+1})
    for i in range(n - 1):
        qiskit_i = n - 1 - i
        qiskit_ip1 = n - 1 - (i+1)
        # Term I
        pauli_list.append('I' * n); coeff_list.append(0.25 * lam)
        # Term +Z_i
        pauli_str_zi = list('I' * n); pauli_str_zi[qiskit_i] = 'Z'
        pauli_list.append("".join(pauli_str_zi)); coeff_list.append(+0.25 * lam)
        # Term +Z_{i+1}
        pauli_str_zip1 = list('I' * n); pauli_str_zip1[qiskit_ip1] = 'Z'
        pauli_list.append("".join(pauli_str_zip1)); coeff_list.append(+0.25 * lam)
        # Term +Z_i Z_{i+1}
        pauli_str_zz = list('I' * n); pauli_str_zz[qiskit_i] = 'Z'; pauli_str_zz[qiskit_ip1] = 'Z'
        pauli_list.append("".join(pauli_str_zz)); coeff_list.append(0.25 * lam)

    hamiltonian_op = SparsePauliOp(pauli_list, coeffs=np.array(coeff_list, dtype=complex))
    simplified_op = hamiltonian_op.simplify(atol=1e-12) # Simplify sums terms
    if verbose: print(f"  H_QIC built (SparsePauliOp, {len(simplified_op)} terms).")
    return simplified_op

# --- Step 8: Verify Energy (using Qiskit EstimatorV2) ---
def verify_energy(state_input, hamiltonian_op, n, label="State", verbose=True):
    """Calculates <state|H|state> using Qiskit EstimatorV2."""
    if not QISKIT_AVAILABLE:
        print("Qiskit not available, cannot verify energy.")
        return False, 0.0j
    if state_input is None or hamiltonian_op is None:
        print(f"Error: Cannot verify energy for {label} - input missing.")
        return False, 0.0 + 0.0j

    expected_dim = 2**n
    if not isinstance(state_input, np.ndarray) or not (state_input.shape == (expected_dim,) or state_input.shape == (expected_dim, 1)):
         print(f"Error: verify_energy input must be numpy array of shape ({expected_dim},) or ({expected_dim},1). Got {type(state_input)} shape {state_input.shape if isinstance(state_input, np.ndarray) else 'N/A'}.")
         return False, 0.0 + 0.0j

    state_data = state_input.flatten()
    norm = np.linalg.norm(state_data)

    # Handle zero vector case
    if np.isclose(norm, 0.0, atol=TOL*100):
         print(f"  State for {label} is zero vector. Energy is 0.")
         return True, 0.0 + 0.0j
    # Check normalization
    if not np.isclose(norm, 1.0, atol=TOL*100):
        print(f"Warning: Input ndarray for {label} not normalized (norm={norm:.4f}). Cannot reliably calculate energy.")
        return False, 0.0 + 0.0j

    # --- Create circuit and initialize state ---
    try:
        temp_qc = QuantumCircuit(n, name=f"Temp_{label}")
        temp_qc.initialize(state_data, temp_qc.qubits)
    except Exception as e:
        print(f"Error: Failed to initialize QC for {label}: {e}"); traceback.print_exc(); return False, 0.0 + 0.0j

    if verbose: print(f"  Verifying energy for {label} (N={n})...")

    # --- Use EstimatorV2 ---
    try:
        estimator = AerEstimator()
        if not isinstance(hamiltonian_op, (SparsePauliOp)):
            print(f"Error: hamiltonian_op is not valid type ({type(hamiltonian_op)}).")
            return False, 0.0+0.0j
        pub = (temp_qc, [hamiltonian_op])
        job = estimator.run(pubs=[pub])
        result = job.result()
        pub_result = result[0]

        if hasattr(pub_result, 'data') and hasattr(pub_result.data, 'evs') and \
           isinstance(pub_result.data.evs, (np.ndarray, list)) and len(pub_result.data.evs) > 0:
             energy_complex = pub_result.data.evs[0]
             energy_real = np.real(energy_complex)
             energy_imag = np.imag(energy_complex)
             # Increase tolerance for estimator result
             is_gs = np.isclose(energy_real, 0, atol=TOL*1000) and np.isclose(energy_imag, 0, atol=TOL*1000)
             if verbose:
                  print(f"  <{label}|H_QIC|{label}>: {energy_complex:.6f} ({energy_real:.6f} R, {energy_imag:.6f} I)")
                  print(f"  --> Ground State (Energy ~ 0): {is_gs}")
             return is_gs, energy_complex
        else:
             print(f"  Estimator result format unexpected for {label}. Result data: {pub_result.data}"); return False, 0.0+0.0j
    except NameError as ne: print(f"\nERROR: Qiskit class/function not found? ({ne})"); return False, 0.0 + 0.0j
    except TypeError as te: print(f"  Estimator V2 TypeError for {label}: {te}"); traceback.print_exc(); return False, 0.0 + 0.0j
    except Exception as e: print(f"  Estimator V2 calculation failed for {label}: {e}"); traceback.print_exc(); return False, 0.0 + 0.0j

# === Example Usage (Optional - typically called from another script) ===
if __name__ == "__main__":
    # This block can be used for testing the core functions directly
    print("--- Testing Core Functions ---")
    N_test = 3
    print(f"\nTesting Fibonacci for N={N_test}: F_{N_test+2} = {fibonacci(N_test+2)}")

    try:
        test_strings, test_vectors = get_qic_basis(N_test)
        if test_strings:
            print(f"\nGenerated basis for N={N_test}.")
            test_V = construct_isometry_V(test_vectors)
            if test_V:
                print("\nGenerated Isometry V.")
                P0_anyon = get_kauffman_Pn_anyon_general(N_test, 0, test_strings)
                P1_anyon = get_kauffman_Pn_anyon_general(N_test, 1, test_strings)
                if P0_anyon is not None and P1_anyon is not None:
                    print("\nGenerated Anyon Projectors P0, P1.")
                    # Check anyon ops
                    anyon_checks_passed = check_anyon_operator_properties([P0_anyon, P1_anyon], N_test)
                    print(f"Anyon Checks Passed: {anyon_checks_passed}")

                    P0_prime = build_P_prime_n(0, N_test, test_V, P0_anyon)
                    P1_prime = build_P_prime_n(1, N_test, test_V, P1_anyon)
                    if P0_prime is not None and P1_prime is not None:
                        print("\nGenerated Embedded Projectors P'0, P'1.")
                        B0_prime = build_B_prime_n(0, P0_prime, N_test)
                        B1_prime = build_B_prime_n(1, P1_prime, N_test)
                        if B0_prime is not None and B1_prime is not None:
                            print("\nGenerated Embedded Braids B'0, B'1.")
                            # Check embedded ops
                            embedded_checks_passed = check_embedded_operator_properties(
                                [P0_prime, P1_prime], [B0_prime, B1_prime], N_test
                            )
                            print(f"Embedded Checks Passed: {embedded_checks_passed}")

                            # Test Hamiltonian and Energy (if Qiskit available)
                            if QISKIT_AVAILABLE:
                                H_op = build_qic_hamiltonian_op(N_test)
                                if H_op:
                                    # Test energy of a basis state
                                    init_state_vec = test_V @ np.array([0,0,1,0,0], dtype=complex) # |101>
                                    verify_energy(init_state_vec, H_op, N_test, label="|101> state")


    except Exception as e:
        print(f"\n--- ERROR during qic_core self-test: {e} ---")
        traceback.print_exc()
