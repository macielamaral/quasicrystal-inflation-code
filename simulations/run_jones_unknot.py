# run_jones_unknot.py
#
# Calculates the Jones polynomial for the Unknot using the abstract
# anyonic representation from the qic_core.py library and the
# same normalization formula used for 12a_122.
# The evaluation is for t = exp(-2*pi*i/5).

import numpy as np
from scipy.sparse import identity as sparse_identity
import time

# Attempt to import qic_core and its necessary components
try:
    import qic_core
    PHI = qic_core.PHI
    R_TAU_1 = qic_core.R_TAU_1
    R_TAU_TAU = qic_core.R_TAU_TAU
    fibonacci = qic_core.fibonacci
    get_qic_basis = qic_core.get_qic_basis
    get_kauffman_Pn_anyon_general = qic_core.get_kauffman_Pn_anyon_general
except ImportError:
    print("ERROR: Could not import qic_core.py or its components.")
    print("Please ensure qic_core.py is in the same directory or in your Python path.")
    exit()
except AttributeError as e:
    print(f"ERROR: A required attribute is missing from qic_core.py: {e}")
    print("Please ensure qic_core.py is complete and up-to-date.")
    exit()

def calculate_jones_for_unknot(n_strands, braid_name, braid_sequence, writhe):
    """
    Calculates the Jones polynomial for a given unknot braid representation.
    
    Args:
        n_strands (int): Number of strands (N).
        braid_name (str): A descriptive name for the unknot braid.
        braid_sequence (list): List of tuples (operator_index, is_inverse)
                               representing the braid word.
        writhe (int): The writhe of the braid.
    """
    print(f"\n--- Starting Jones Polynomial Calculation for: {braid_name} ---")
    start_time = time.time()

    print(f"Knot: Unknot (via {braid_name})")
    print(f"Number of Strands (N): {n_strands}")
    print(f"Writhe (w): {writhe}")

    # Normalization constants for Jones Polynomial V(L,t)
    # t = exp(-2*pi*i/5), so A = t^(-1/4) = exp(pi*i/10)
    A_param = np.exp(1j * np.pi / 10)
    neg_phi_gr = -PHI

    # 1. Generate QIC Basis Strings (needed for P_k^anyon construction)
    print(f"\nStep 1: Generating QIC basis for N={n_strands}...")
    qic_strings_N, _ = get_qic_basis(n_strands)
    if not qic_strings_N:
        print(f"ERROR: Failed to generate QIC basis strings for N={n_strands}.")
        return None
    dim_anyon_space = fibonacci(n_strands + 2)
    print(f"Anyonic subspace dimension for N={n_strands}: F_{n_strands+2} = {dim_anyon_space}")

    # Check if braid operators are needed (N>=2 for P_0)
    if n_strands < 2 and braid_sequence:
        print(f"Warning: Braid sequence provided for N={n_strands}, but P_k^anyon are typically defined for N>=2.")
        # For N=1, unknot trace is often handled as a special case (e.g. = d)
        # Our framework for P_k starts at N=2.
        # If braid_sequence is empty (e.g. for N=1 unknot), this path is okay.

    P_anyon_ops = []
    if n_strands >= 2:
        # 2. Generate Anyonic Temperley-Lieb Projectors (P_k^anyon)
        print(f"\nStep 2: Generating Anyonic Temperley-Lieb Projectors P_k^anyon for N={n_strands}...")
        for k_idx in range(n_strands - 1):
            print(f"  Constructing P_{k_idx}^anyon...")
            Pk_anyon = get_kauffman_Pn_anyon_general(n_strands, k_idx, qic_strings_N, delta=PHI)
            if Pk_anyon is None:
                print(f"ERROR: Failed to generate P_{k_idx}^anyon.")
                return None
            P_anyon_ops.append(Pk_anyon)
        print("P_k^anyon generation complete.")
    elif braid_sequence: # N < 2 but braid sequence exists
        print(f"ERROR: Cannot process braid sequence for N={n_strands} < 2 with P_k operators.")
        return None


    B_anyon_ops = []
    B_anyon_inv_ops = []
    Id_anyon = sparse_identity(dim_anyon_space, dtype=complex, format='csc')

    if n_strands >= 2:
        # 3. Construct Anyonic Braid Operators (B_k^anyon and their inverses)
        print(f"\nStep 3: Constructing Anyonic Braid Operators B_k^anyon for N={n_strands}...")
        for idx, Pk_anyon in enumerate(P_anyon_ops):
            Bk_anyon = R_TAU_1 * Pk_anyon + R_TAU_TAU * (Id_anyon - Pk_anyon)
            B_anyon_ops.append(Bk_anyon.tocsc())
            Bk_anyon_inv = Bk_anyon.conj().T # Inverse is the adjoint
            B_anyon_inv_ops.append(Bk_anyon_inv.tocsc())
            print(f"  B_{idx}^anyon and its inverse constructed.")
        print("B_k^anyon construction complete.")

    # 4. Compute the Total Braid Operator (B_total_anyon)
    print(f"\nStep 4: Computing total braid operator B_total_anyon for {braid_name}...")
    B_total_anyon = sparse_identity(dim_anyon_space, dtype=complex, format='csc') # Start with identity

    if braid_sequence: # If there are actual braid operations
        for op_idx, is_inv in braid_sequence:
            if op_idx >= len(B_anyon_ops):
                print(f"ERROR: Braid operator index {op_idx} out of range for N={n_strands}.")
                return None
            
            if is_inv:
                current_op_matrix = B_anyon_inv_ops[op_idx]
            else:
                current_op_matrix = B_anyon_ops[op_idx]
            B_total_anyon = B_total_anyon @ current_op_matrix # M_total = M_total @ M_op
    else: # Empty braid word (e.g. for N strands, 0 crossings, representing N unknots or 1 unknot depending on closure context)
        print("  Empty braid word, B_total_anyon remains Identity.")

    print(f"Total braid operator B_total_anyon computed (shape: {B_total_anyon.shape}).")

    # 5. Calculate the Trace
    print(f"\nStep 5: Calculating Trace(B_total_anyon)...")
    raw_trace_val = B_total_anyon.trace()
    print(f"Raw Trace Tr(B_total_anyon): {raw_trace_val}")
    # Trace of Identity_D is D. For sigma_1 sigma_1^-1, B_total_anyon is Identity.
    # So, raw_trace_val should be dim_anyon_space = F_{n_strands+2}.
    if np.isclose(raw_trace_val, dim_anyon_space):
        print(f"  Trace matches dimension F_{n_strands+2} = {dim_anyon_space}, as expected for an identity braid.")
    else:
        print(f"  WARNING: Trace {raw_trace_val} does not match dimension {dim_anyon_space}. Check braid word if it should be identity.")


    # 6. Apply Normalization for Jones Polynomial
    # V_L(t) = (-phi_GR)^-(N-1) * (-A^3)^(-writhe) * Tr(B_total_anyon)
    print(f"\nStep 6: Applying normalization for Jones Polynomial...")
    if n_strands == 1:
        # The factor (-phi_GR)^-(N-1) is problematic for N=1.
        # Standard Jones for Unknot on 1 strand is often taken as 1 directly,
        # or d (quantum dimension of the anyon type, phi_GR for Fibonacci).
        # Let's see what the formula gives if N-1 becomes 0.
        norm_factor_phi = (neg_phi_gr)**(-(n_strands - 1)) # This would be 1 if n_strands=1
        print(f"  Note: For N=1, norm_factor_phi ( (-phi_GR)^-(N-1) ) = {norm_factor_phi}")
    else:
        norm_factor_phi = (neg_phi_gr)**(-(n_strands - 1))
        
    norm_factor_A_writhe = (-A_param**3)**(-writhe) # For writhe=0, this is 1
    
    jones_polynomial_value_unknot = norm_factor_phi * norm_factor_A_writhe * raw_trace_val
    
    end_time = time.time()
    print("\n--- Jones Polynomial Calculation Complete ---")
    print(f"Total calculation time for {braid_name}: {end_time - start_time:.3f} seconds")
    print(f"\nFinal Calculated Jones Polynomial V(Unknot via {braid_name}, t=exp(-2*pi*i/5)):")
    print(f"  {jones_polynomial_value_unknot}")
    print(f"  Real part: {np.real(jones_polynomial_value_unknot)}")
    print(f"  Imaginary part: {np.imag(jones_polynomial_value_unknot)}")

    # The standard Jones polynomial for the Unknot is 1.
    # Let's see how far this value is from 1.
    diff_from_one = jones_polynomial_value_unknot - 1.0
    print(f"\nDifference from standard V(Unknot)=1: {diff_from_one} (Magnitude: {np.abs(diff_from_one)})")
    if np.isclose(jones_polynomial_value_unknot, 1.0):
        print("  --> This framework and normalization correctly yields V(Unknot) = 1 for this N.")
    else:
        print(f"  --> This framework and normalization yields V_0 = {jones_polynomial_value_unknot:.6f} for this N.")
        print(f"      To renormalize other knots (like 12a_122) to match V(Unknot)=1 standard,")
        print(f"      you would divide their calculated Jones value by this V_0 = {jones_polynomial_value_unknot:.6f}.")

    return jones_polynomial_value_unknot

if __name__ == "__main__":
    # --- Test Case 1: Unknot on N=2 strands ---
    # Braid: sigma_1 * sigma_1^-1 (operator index 0)
    # Writhe: 1 - 1 = 0
    N2_unknot_name = "Unknot (N=2, sigma_1 * sigma_1^-1)"
    N2_unknot_braid_seq = [(0, False), (0, True)]
    N2_unknot_writhe = 0
    V0_N2 = calculate_jones_for_unknot(2, N2_unknot_name, N2_unknot_braid_seq, N2_unknot_writhe)
    # Expected based on manual calc: -3/PHI ~ -1.8541

    # --- Test Case 2: Unknot on N=4 strands ---
    # Braid: sigma_1 * sigma_1^-1 (operator index 0, other strands trivial)
    # Writhe: 1 - 1 = 0
    N4_unknot_name = "Unknot (N=4, sigma_1 * sigma_1^-1)"
    N4_unknot_braid_seq = [(0, False), (0, True)] # Only uses B_0 and its inverse
    N4_unknot_writhe = 0
    V0_N4 = calculate_jones_for_unknot(4, N4_unknot_name, N4_unknot_braid_seq, N4_unknot_writhe)
    # Expected based on manual calc: 8/(-PHI)^3 ~ -1.88854

    # --- Test Case 3: Unknot on N=4 strands, more complex braid ---
    # Braid: sigma_1 * sigma_2 * sigma_2^-1 * sigma_1^-1
    # Writhe: 1 + 1 - 1 - 1 = 0
    N4_unknot_name_complex = "Unknot (N=4, s1*s2*s2^-1*s1^-1)"
    N4_unknot_braid_seq_complex = [
        (0, False), (1, False), # sigma_1, sigma_2
        (1, True), (0, True)    # sigma_2^-1, sigma_1^-1
    ]
    N4_unknot_writhe_complex = 0
    V0_N4_complex = calculate_jones_for_unknot(4, N4_unknot_name_complex, N4_unknot_braid_seq_complex, N4_unknot_writhe_complex)
    # Expected: Should also be 8/(-PHI)^3 ~ -1.88854 if braid is truly an unknot

    print("\n\n--- Summary of Unknot Values with current normalization formula: ---")
    if V0_N2 is not None:
        print(f"V(Unknot, N=2) calculated = {V0_N2:.6f}")
    if V0_N4 is not None:
        print(f"V(Unknot, N=4, braid s1*s1^-1) calculated = {V0_N4:.6f}")
    if V0_N4_complex is not None:
        print(f"V(Unknot, N=4, braid s1*s2*s2^-1*s1^-1) calculated = {V0_N4_complex:.6f}")

    print("\nIf these values are not 1, they represent the normalization constant V_0(N)")
    print("produced by your framework for an N-strand unknot with writhe 0.")
    print("To get other Jones polynomials (like for 12a_122) to the V(Unknot)=1 standard,")
    print("you would calculate V_K (as in run_jones_12a122.py) and then compute V_K / V_0(N).")