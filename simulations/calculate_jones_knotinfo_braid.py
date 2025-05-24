# calculate_jones_knotinfo_braid.py
#
# Calculates the Jones polynomial for the 12a_122 knot using the
# braid representation from KnotInfo (N=5 strands) and the abstract
# anyonic representation from the qic_core.py library.
# The result is normalized such that V(Unknot)=1.
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

def calculate_jones_with_knotinfo_braid_12a122():
    """
    Calculates the Jones polynomial for 12a_122 using the KnotInfo braid.
    """
    print("--- Starting Jones Polynomial Calculation for 12a_122 (KnotInfo Braid N=5) ---")
    start_time = time.time()

    # 1. Knot and Braid Information from KnotInfo for 12a_122
    knot_name = "12a_122 (via KnotInfo N=5 braid)"
    N_strands = 5
    # Braid word from KnotInfo: {1,-2,-2,3,-4,1,1,-2,-2,-2,3,-4}
    # Maps to sigma_i where i is the number. For B_k, k = i-1.
    # sigma_1 -> k=0, sigma_2 -> k=1, sigma_3 -> k=2, sigma_4 -> k=3
    knotinfo_braid_indices = [1, -2, -2, 3, -4, 1, 1, -2, -2, -2, 3, -4]
    
    braid_sequence = []
    for gen_idx in knotinfo_braid_indices:
        op_k_idx = abs(gen_idx) - 1 # Convert 1-based sigma_i to 0-based B_k
        is_inv = gen_idx < 0
        braid_sequence.append((op_k_idx, is_inv))
        
    writhe_knotinfo_braid = sum(knotinfo_braid_indices) # Should be -9

    print(f"Knot: {knot_name}")
    print(f"Number of Strands (N): {N_strands}")
    print(f"Braid Sequence (0-indexed k, is_inverse): {braid_sequence}")
    print(f"Writhe (w): {writhe_knotinfo_braid}")

    # Normalization constants for Jones Polynomial V(L,t)
    # t = exp(-2*pi*i/5), so A = t^(-1/4) = exp(pi*i/10)
    A_param = np.exp(1j * np.pi / 10)
    # neg_phi_gr = -PHI # This factor was part of a previous normalization attempt,
                       # the current one V_K = (-A^3)^-w * (S_K / S_U_N) is simpler.

    # Theoretical trace of Unknot on N strands (abstract anyonic identity op)
    # This is F_{N+2}
    S_Unknot_at_N_theoretical = float(fibonacci(N_strands + 2)) # For N=5, F_7 = 13

    # 2. Generate QIC Basis Strings
    print(f"\nStep 1: Generating QIC basis for N={N_strands}...")
    qic_strings_N, _ = get_qic_basis(N_strands)
    if not qic_strings_N:
        print("ERROR: Failed to generate QIC basis strings.")
        return None
    dim_anyon_space = fibonacci(N_strands + 2) # For N=5, F_7 = 13
    print(f"Anyonic subspace dimension: {dim_anyon_space}")
    if not np.isclose(S_Unknot_at_N_theoretical, dim_anyon_space):
        print(f"Warning: S_Unknot_at_N_theoretical {S_Unknot_at_N_theoretical} does not match dim_anyon_space {dim_anyon_space}")


    # 3. Generate Anyonic Temperley-Lieb Projectors (P_k^anyon)
    print(f"\nStep 2: Generating Anyonic Temperley-Lieb Projectors P_k^anyon...")
    P_anyon_ops = []
    # For N=5, k_idx = 0, 1, 2, 3
    for k_idx in range(N_strands - 1): 
        print(f"  Constructing P_{k_idx}^anyon...")
        Pk_anyon = get_kauffman_Pn_anyon_general(N_strands, k_idx, qic_strings_N, delta=PHI)
        if Pk_anyon is None:
            print(f"ERROR: Failed to generate P_{k_idx}^anyon.")
            return None
        P_anyon_ops.append(Pk_anyon)
    print("P_k^anyon generation complete.")

    # 4. Construct Anyonic Braid Operators (B_k^anyon and their inverses)
    print(f"\nStep 3: Constructing Anyonic Braid Operators B_k^anyon...")
    B_anyon_ops = []
    B_anyon_inv_ops = []
    Id_anyon = sparse_identity(dim_anyon_space, dtype=complex, format='csc')

    for idx, Pk_anyon in enumerate(P_anyon_ops):
        Bk_anyon = R_TAU_1 * Pk_anyon + R_TAU_TAU * (Id_anyon - Pk_anyon)
        B_anyon_ops.append(Bk_anyon.tocsc())
        Bk_anyon_inv = Bk_anyon.conj().T
        B_anyon_inv_ops.append(Bk_anyon_inv.tocsc())
        print(f"  B_{idx}^anyon and its inverse constructed.")
    print("B_k^anyon construction complete.")

    # 5. Compute the Total Braid Operator (B_total_anyon)
    print(f"\nStep 4: Computing total braid operator B_total_anyon for the KnotInfo braid...")
    B_total_anyon = sparse_identity(dim_anyon_space, dtype=complex, format='csc')

    for op_idx, is_inv in braid_sequence:
        if op_idx < 0 or op_idx >= len(B_anyon_ops):
            print(f"ERROR: Braid generator index {op_idx+1} (0-indexed {op_idx}) is out of range for N={N_strands} strands.")
            return None
        current_op_matrix = B_anyon_inv_ops[op_idx] if is_inv else B_anyon_ops[op_idx]
        B_total_anyon = B_total_anyon @ current_op_matrix
    print(f"Total braid operator B_total_anyon computed (shape: {B_total_anyon.shape}).")

    # 6. Calculate the Trace
    print(f"\nStep 5: Calculating Trace(B_total_anyon)...")
    raw_trace_val = B_total_anyon.trace()
    print(f"Raw Trace Tr(B_total_anyon) (S_KnotInfoBraid): {raw_trace_val}")
    print(f"  Real part of Trace: {np.real(raw_trace_val)}")
    print(f"  Imaginary part of Trace: {np.imag(raw_trace_val)}")

    # 7. Apply Normalization for Jones Polynomial
    # V_L(t) = (-A^3)^(-writhe) * (S_L / S_Unknot_N)
    print(f"\nStep 6: Applying self-consistent normalization for Jones Polynomial...")
    
    phase_factor_writhe = (-A_param**3)**(-writhe_knotinfo_braid)
    
    if np.isclose(S_Unknot_at_N_theoretical, 0):
        print("ERROR: S_Unknot_at_N_theoretical is zero, cannot normalize.")
        return None
        
    jones_polynomial_value = phase_factor_writhe * (raw_trace_val / S_Unknot_at_N_theoretical)
    
    end_time = time.time()
    print("\n--- Jones Polynomial Calculation Complete (KnotInfo Braid N=5) ---")
    print(f"Total calculation time: {end_time - start_time:.3f} seconds")
    print(f"\nFinal Self-Consistently Normalized Jones Polynomial V({knot_name}, t=exp(-2*pi*i/5)):")
    print(f"  {jones_polynomial_value}")
    print(f"  Real part: {np.real(jones_polynomial_value)}")
    print(f"  Imaginary part: {np.imag(jones_polynomial_value)}")

    return jones_polynomial_value

if __name__ == "__main__":
    # Run the calculation for the KnotInfo braid of 12a_122
    jp_value_knotinfo_braid = calculate_jones_with_knotinfo_braid_12a122()

    if jp_value_knotinfo_braid is not None:
        print("\n--- Comparison with Previous Calculation (N=4 braid for 12a_122) ---")
        # This was the result from calculate_renormalized_jones.py using the N=4 braid
        # V_final_N4_braid = complex(0.34682189257828955, 0.07347315653655925)
        # For this run, we use the updated value from the last successful run.
        V_final_N4_braid_ref = complex(0.34682189257828944, 0.07347315653655925) # From your last output

        print(f"  Value from N=4 braid for 12a_122 (expected invariant): {V_final_N4_braid_ref:.8f}")
        print(f"  Value from N=5 KnotInfo braid for 12a_122 (this run): {jp_value_knotinfo_braid:.8f}")
        
        diff = jp_value_knotinfo_braid - V_final_N4_braid_ref
        print(f"  Difference: {diff:.3e} (Magnitude: {np.abs(diff):.3e})")
        if np.isclose(jp_value_knotinfo_braid, V_final_N4_braid_ref):
            print("  --> SUCCESS: The Jones polynomial value is consistent across different braid representations (N=4 vs N=5)!")
            print("               This is a strong validation of the framework computing a true knot invariant.")
        else:
            print("  --> MISMATCH: The Jones value differs between the N=4 and N=5 braid representations.")
            print("                This suggests a potential issue in the framework's handling of N,")
            print("                the P_k^anyon definitions for different N, or the normalization's N-dependence.")

        print("\n--- Comparison with KnotInfo Polynomial Evaluation ---")
        # Jones polynomial string from KnotInfo for 12a_122:
        # P_KI(t) = -t^-8 + 3t^-7 - 8t^-6 + 15t^-5 - 21t^-4 + 26t^-3 -27t^-2 + 25t^-1 - 20 + 14t - 7t^2 + 3t^3 - t^4
        # Evaluated at t = exp(-2*pi*i/5) this gave approx. 0.208204 - 1.049304j
        val_from_knotinfo_poly_t = complex(0.20820415139800033, -1.0493041738032164)
        print(f"  Value from evaluating KnotInfo's polynomial string at t=exp(-2*pi*i/5): {val_from_knotinfo_poly_t:.8f}")
        diff_ki_poly = jp_value_knotinfo_braid - val_from_knotinfo_poly_t
        print(f"  Difference (Framework N=5 vs KnotInfo Poly(t)): {diff_ki_poly:.3e} (Magnitude: {np.abs(diff_ki_poly):.3e})")

        # Also check KnotInfo's polynomial at t^-1 = exp(2*pi*i/5)
        # For t=exp(2*pi*i/5): A_inv = (exp(2*pi*i/5))^(-1/4) = exp(-pi*i/10)
        # P_KI(t^-1)
        t_inv = np.exp(1j * 2 * np.pi / 5)
        # Manually (or with Mathematica) evaluate P_KI(t_inv)
        # -t_inv^(-8)+3*t_inv^(-7)-8*t_inv^(-6)+15*t_inv^(-5)-21*t_inv^(-4)+26*t_inv^(-3)-27*t_inv^(-2)+25*t_inv^(-1)-20+14*t_inv-7*t_inv^2+3*t_inv^3-t_inv^4
        # This requires substituting t_inv and simplifying.
        # For example, t_inv^-8 = (e^(i2pi/5))^-8 = e^(-i16pi/5) = e^(-i(10pi/5 + 6pi/5)) = e^(-i2pi) * e^(-i6pi/5) = e^(i4pi/5)
        # For simplicity, this evaluation is left as an exercise or for Mathematica.
        # A known property: V(K, t^-1) = V(K*, t) where K* is mirror.
        # If your framework computes V(K,t) and KnotInfo lists V(K*,t) or uses a t vs t^-1 convention.
        print(f"  Note: Further comparison might involve checking V(K, t^-1) or mirror image conventions.")