# run_jones_sim.py
# Reads a braid definition from a JSON file (e.g., from CQCL's generate.py)
# and constructs the abstract anyonic braid operator M_braid^anyon using qic_core.py.

import numpy as np
import time
import traceback
import json
import pathlib

# Import functions from the core library module
try:
    import qic_core # Assuming qic_core.py is in PYTHONPATH or same directory
except ImportError:
    print("ERROR: Failed to import qic_core.py.")
    print("       Make sure qic_core.py is in the same directory or Python path.")
    exit()

# Qiskit is NOT required for this script's primary goal (constructing M_braid^anyon)

# === Configuration ===
# The path to the JSON file will be an argument to the script or can be hardcoded for testing
# Example: JSON_FILE_PATH = "trefoil_data_quantinuum/b_n4_l10.json"
# We will use argparse to make it more flexible.

# =======================

def construct_abstract_anyon_braid_operator(n_strands, braid_word_list, qic_basis_strings_list):
    """
    Constructs the full abstract anyonic braid operator M_braid^anyon.

    Args:
        n_strands (int): Number of strands for the braid.
        braid_word_list (list[int]): The braid word, e.g., [-1, 2, -3].
        qic_basis_strings_list (list[str]): List of QIC basis strings for N_strands.

    Returns:
        scipy.sparse.csc_matrix: The M_braid^anyon operator, or None on error.
    """
    dim_anyon_space = qic_core.fibonacci(n_strands + 2)
    if dim_anyon_space == 0:
        print(f"Error: Anyonic space dimension is 0 for N={n_strands}.")
        return None
    
    print(f"Anyonic space dimension for N={n_strands} is F_{n_strands+2} = {dim_anyon_space}")

    # Initialize M_braid_anyon as an identity matrix
    # Operators will be left-multiplied: M_final = B_last @ ... @ B_first @ I
    M_braid_anyon = qic_core.sparse_identity(dim_anyon_space, dtype=complex, format='csc')
    Id_anyon = qic_core.sparse_identity(dim_anyon_space, dtype=complex, format='csc') # For (I - Pk)

    print(f"Processing braid word: {braid_word_list}")
    for generator_val in braid_word_list:
        if generator_val == 0:
            print("Warning: Braid generator value 0 encountered, skipping.")
            continue

        # Determine k for P_k^anyon: P_k acts on sites k, k+1 (0-indexed),
        # corresponds to braid generator sigma_{k+1}.
        # So, for sigma_j (where j is 1-indexed from braid word), k_P_idx = j - 1.
        k_P_idx = abs(generator_val) - 1

        if not (0 <= k_P_idx <= n_strands - 2):
            print(f"ERROR: Calculated P_k index {k_P_idx} from generator {generator_val} "
                  f"is out of bounds (0 to {n_strands-2}) for N={n_strands}.")
            return None

        # print(f"  Building P_{k_P_idx}^anyon for generator value {generator_val}...")
        Pk_anyon = qic_core.get_kauffman_Pn_anyon_general(n_strands, k_P_idx, qic_basis_strings_list,
                                                          delta=qic_core.PHI)
        if Pk_anyon is None:
            print(f"ERROR: Failed to build P_{k_P_idx}^anyon.")
            return None

        # Construct the abstract anyonic braid generator B_k^anyon
        # B_k = R_I * P_k + R_tau * (I - P_k)
        # Using constants from qic_core: R_I = R_TAU_1, R_TAU = R_TAU_TAU
        Bk_anyon_std = qic_core.R_TAU_1 * Pk_anyon + qic_core.R_TAU_TAU * (Id_anyon - Pk_anyon)

        if generator_val < 0:
            # Inverse: (B_k)^-1 = R_I^* * P_k + R_tau^* * (I - P_k)
            # print(f"  Taking inverse for sigma_{abs(generator_val)}^-1 (using B_{k_P_idx}^-1 anyon)")
            Bk_anyon_to_apply = np.conjugate(qic_core.R_TAU_1) * Pk_anyon + \
                                np.conjugate(qic_core.R_TAU_TAU) * (Id_anyon - Pk_anyon)
        else:
            # print(f"  Using sigma_{abs(generator_val)} (using B_{k_P_idx} anyon)")
            Bk_anyon_to_apply = Bk_anyon_std
        
        # Apply this generator to the overall braid matrix
        # M_final = B_last @ ... @ B_first @ I.
        # If braid word is [g1, g2, g3], this means B3 @ B2 @ B1.
        # So, each new Bk_anyon_to_apply should pre-multiply the current M_braid_anyon.
        # However, standard convention is that product notation B3 B2 B1 means B1 acts first.
        # If braid_word_list represents B1, B2, B3 in order of application,
        # then M_braid_anyon = B3 @ B2 @ B1.
        # So, M_braid_anyon should be updated as: M_braid_anyon_new = Bk_anyon_to_apply @ M_braid_anyon_old
        M_braid_anyon = Bk_anyon_to_apply @ M_braid_anyon
        # print(f"    Applied operator for {generator_val}. Norm of M_braid_anyon: {qic_core.sparse_norm(M_braid_anyon):.3e}")


    print("\nFinished constructing M_braid_anyon.")
    return M_braid_anyon.tocsc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Construct Abstract Anyonic Braid Operator M_braid^anyon.")
    parser.add_argument("json_file", type=str, help="Path to the JSON file containing braid information.")
    args = parser.parse_args()

    JSON_FILE_PATH = pathlib.Path(args.json_file)

    if not JSON_FILE_PATH.is_file():
        print(f"ERROR: JSON file not found at {JSON_FILE_PATH}")
        exit()

    master_start_time = time.time()
    print(f"--- Starting Jones Simulation Prep: Constructing M_braid^anyon ---")
    print(f"Loading braid data from: {JSON_FILE_PATH}")

    try:
        with open(JSON_FILE_PATH, 'r') as f:
            braid_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not read or parse JSON file: {e}")
        traceback.print_exc()
        exit()

    # Extract necessary data from JSON
    try:
        # Prefer 'source' for N_strands as 'optimized' might not reflect original intent if braid is trivial for N>1
        # However, the anyonic space must match the actual number of strands the braid operator acts on.
        # The 'optimized.word' generators will define the minimum N needed.
        # Let's infer N from the max generator index in the optimized word if source.strands is too large.
        
        optimized_word = braid_data["optimized"]["word"]
        if not optimized_word: # Empty braid word
             n_strands_from_word = 0
        else:
             n_strands_from_word = max(abs(g) for g in optimized_word) # if sigma_k, then k strands are involved. Max(k) means N-1 if k goes up to N-1
                                                                    # if max(|g|) is M, then N_strands is M+1 (e.g. sigma_3 needs 4 strands)
                                                                    # Or, if sigma_k means k is the *first* of two strands (1-indexed), then N_strands is max(|g|)+1
        
        # The CQCL generate.py output 'strands' is the number of strands the braid is defined on.
        N_strands_from_source = braid_data["source"]["strands"]
        
        # Determine actual N_strands for operator construction.
        # The operators P_k are defined up to k=N-2. If braid uses sigma_j, k = j-1. Max j implies N.
        # Max |g_val| in word gives the highest index of sigma_j used. This sigma_j acts on j and j+1. So N_strands must be at least max_j + 1.
        min_strands_for_word = 0
        if optimized_word:
            min_strands_for_word = max(abs(g) for g in optimized_word) + 1
        
        # Use the N_strands specified in the source, as this is what the benchmark value is for.
        # Ensure it's consistent with the word.
        N_strands = N_strands_from_source
        if min_strands_for_word > N_strands:
            print(f"Warning: Optimized word {optimized_word} requires at least {min_strands_for_word} strands, "
                  f"but source specifies N={N_strands}. Using {min_strands_for_word}.")
            N_strands = min_strands_for_word
        
        # If the word is empty (e.g. trivial braid from the start)
        if not optimized_word and N_strands == 0 : N_strands = 1 # Default to 1 strand for trivial case if N=0
        elif not optimized_word and N_strands > 0: print(f"Note: Braid word is empty. M_braid_anyon will be Identity for N={N_strands} strands.")
        elif N_strands == 0 and optimized_word : # Should not happen if generate.py is correct
             N_strands = min_strands_for_word
             print(f"Warning: N_strands was 0 but word is not empty. Set N_strands to {N_strands}")


        braid_word = optimized_word
        target_jones_real = braid_data["jones"][0]
        target_jones_imag = braid_data["jones"][1]
        writhe = sum(np.sign(g) for g in braid_word if g != 0) # Calculate writhe

        print(f"Parameters: N_strands = {N_strands}, Braid Word = {braid_word}, Target Jones = {target_jones_real:.6f} + {target_jones_imag:.6f}j, Writhe = {writhe}")

    except KeyError as e:
        print(f"ERROR: JSON file missing expected key: {e}")
        traceback.print_exc()
        exit()
    except Exception as e:
        print(f"ERROR: Unexpected error processing JSON data: {e}")
        traceback.print_exc()
        exit()
        
    if N_strands == 0 and not braid_word: # If word is empty and N_strands became 0
        print("Braid word is empty and N_strands is 0. Setting N_strands to 1 for identity.")
        N_strands = 1 # Smallest non-trivial N for identity. Or handle as special case.
    elif N_strands == 1 and braid_word : # A braid word like [1] implies N_strands=2.
        print(f"ERROR: N_strands=1 but braid word is {braid_word}. A braid word requires N_strands >= 2 if word not empty.")
        exit()
    elif N_strands < 2 and braid_word: # General case
        print(f"ERROR: N_strands={N_strands} but braid word is {braid_word}. Any non-empty braid word requires N_strands >= 2.")
        exit()


    # === Part 1: Setup QIC Basis (from your script) ===
    print(f"\n=== PART 1: Setup QIC Basis (N={N_strands}) ===")
    part1_start_time = time.time()
    qic_basis_strings = []
    try:
        qic_basis_strings, _ = qic_core.get_qic_basis(N_strands) # We only need strings for Pk_anyon
        if not qic_basis_strings and N_strands > 0 : # N=0 get_qic_basis returns [],[] which is fine
             raise ValueError(f"Failed to generate QIC basis strings for N={N_strands}.")
        elif N_strands == 0 and not qic_basis_strings: # Handle N=0 case explicitly if get_qic_basis returns empty
             print("N=0, QIC basis is trivially empty. M_braid will be 1x1 Identity if word is empty.")
             # For N=0, F_2 = 1. So dim_anyon_space should be 1.
             # get_qic_basis(0) returns ([],[]). fibonacci(0+2)=1.
             # This needs careful handling if we allow N=0.
             # For Jones Poly, N>=2 usually. If word is [-1], N=2.
             # The CQCL script b_n4_l10.json gave "strands":4, word:[-1]. This implies N=4.
             # If word is [-1], it means sigma_1^-1. This implies at least N=2 strands.
             # The "strands" field in JSON should be the definitive N.
    except Exception as e:
        print(f"ERROR during Part 1 (QIC Basis Setup): {e}")
        traceback.print_exc()
        exit()
    end_time = time.time(); print(f"Part 1 Execution Time: {end_time - part1_start_time:.3f} seconds")

    # === Part 2: Construct Abstract Anyonic Braid Operator M_braid^anyon ===
    print(f"\n=== PART 2: Constructing M_braid^anyon (N={N_strands}) ===")
    part2_start_time = time.time()
    M_braid_anyon_matrix = None
    try:
        # Handle the N=0 or N=1 case if the braid_word is empty
        if not braid_word:
            print("Braid word is empty. M_braid^anyon is Identity.")
            dim_anyon_space_empty_braid = qic_core.fibonacci(N_strands + 2)
            if dim_anyon_space_empty_braid == 0 and N_strands == 0: # F2=1 for N=0
                dim_anyon_space_empty_braid = 1
            elif dim_anyon_space_empty_braid == 0 and N_strands > 0 :
                 raise ValueError(f"Cannot form Identity for N={N_strands} as F_{N_strands+2} is 0.")
            M_braid_anyon_matrix = qic_core.sparse_identity(dim_anyon_space_empty_braid, dtype=complex, format='csc')
        elif N_strands < 2:
             # This case should have been caught earlier if word is not empty
             print(f"ERROR: N_strands={N_strands} is too small for a non-empty braid word. Requires N>=2.")
             exit()
        else:
            M_braid_anyon_matrix = construct_abstract_anyon_braid_operator(
                N_strands, braid_word, qic_basis_strings
            )

        if M_braid_anyon_matrix is None:
            raise ValueError("Failed to construct M_braid^anyon.")

        print(f"\nSuccessfully constructed M_braid^anyon for N={N_strands}.")
        print(f"  Shape: {M_braid_anyon_matrix.shape}")
        print(f"  Number of non-zero elements: {M_braid_anyon_matrix.nnz}")
        # For small N, you can print the dense matrix:
        # if N_strands <= 4:
        # print("M_braid_anyon (dense):\n", M_braid_anyon_matrix.toarray())

    except Exception as e:
        print(f"ERROR during Part 2 (M_braid^anyon Construction): {e}")
        traceback.print_exc()
        exit()
    end_time = time.time(); print(f"Part 2 Execution Time: {end_time - part2_start_time:.3f} seconds")

    
    # === PART 3: Analyzing M_braid_anyon and Searching for est_raw ===
    print(f"\n=== PART 3: Analyzing M_braid_anyon and Searching for est_raw ===")
    part3_start_time = time.time()

    if M_braid_anyon_matrix is None:
        print("ERROR: M_braid_anyon_matrix is None. Cannot proceed.")
        exit()

    target_jones_complex = complex(target_jones_real, target_jones_imag)

    t_cqcl_val = np.exp(1j * 2 * np.pi / 5)
    factor_writhe_cqcl = t_cqcl_val**(3 * writhe)
    if N_strands < 1 and not braid_word: factor_phi_cqcl = 1
    elif N_strands == 1 and not braid_word: factor_phi_cqcl = 1
    elif N_strands >= 1: factor_phi_cqcl = qic_core.PHI**(N_strands - 1)
    else: print(f"ERROR: Invalid N_strands ({N_strands}) for factor_phi calculation."); exit()

    denominator_cqcl = factor_writhe_cqcl * factor_phi_cqcl
    est_raw_target_from_cqcl = complex(np.nan, np.nan)
    if abs(denominator_cqcl) > 1e-12:
        est_raw_target_from_cqcl = target_jones_complex / denominator_cqcl
    
    print(f"  Target Jones (from JSON):             {target_jones_complex.real:.8f} + {target_jones_complex.imag:.8f}j")
    print(f"  CQCL Writhe Factor (t_cqcl^(3W)):     {factor_writhe_cqcl.real:.8f} + {factor_writhe_cqcl.imag:.8f}j (W={writhe})")
    print(f"  CQCL Phi Factor (PHI^(N-1)):          {factor_phi_cqcl:.8f} (N={N_strands})")
    if not np.isnan(est_raw_target_from_cqcl.real):
        print(f"  Derived est_raw_target for CQCL:      {est_raw_target_from_cqcl.real:.8f} + {est_raw_target_from_cqcl.imag:.8f}j")
    else:
        print("  Derived est_raw_target for CQCL: Could not be determined (denominator issue).")


    M_dense = M_braid_anyon_matrix.toarray()
    est_raw_candidate_M00 = complex(np.nan, np.nan)
    est_raw_candidate_Mlastlast = complex(np.nan, np.nan)
    
    if M_dense.shape[0] > 0 and M_dense.shape[1] > 0:
        est_raw_candidate_M00 = M_dense[0,0]
        last_idx = M_dense.shape[0] - 1
        est_raw_candidate_Mlastlast = M_dense[last_idx, last_idx]
        print(f"\n  Candidate est_raw from M[0,0]:          {est_raw_candidate_M00.real:.8f} + {est_raw_candidate_M00.imag:.8f}j")
        print(f"  Candidate est_raw from M[last={last_idx},last={last_idx}]: {est_raw_candidate_Mlastlast.real:.8f} + {est_raw_candidate_Mlastlast.imag:.8f}j")
    else:
        print("\n  M_braid_anyon_matrix is empty or ill-defined, cannot extract diagonal candidates.")

    TOLERANCE = 1e-7 

    # Test with est_raw_candidate_M00
    if not np.isnan(est_raw_candidate_M00.real):
        jones_calc_M00 = factor_writhe_cqcl * factor_phi_cqcl * est_raw_candidate_M00
        print(f"\n  Calculated Jones (using M[0,0] as est_raw): {jones_calc_M00.real:.8f} + {jones_calc_M00.imag:.8f}j")
        if abs(jones_calc_M00.real - target_jones_complex.real) < TOLERANCE and \
           abs(jones_calc_M00.imag - target_jones_complex.imag) < TOLERANCE:
            print("    VERIFICATION SUCCESS with M[0,0]")
        else:
            print("    VERIFICATION FAILURE with M[0,0]")

    # Test with est_raw_candidate_Mlastlast
    if not np.isnan(est_raw_candidate_Mlastlast.real):
        jones_calc_Mlastlast = factor_writhe_cqcl * factor_phi_cqcl * est_raw_candidate_Mlastlast
        print(f"\n  Calculated Jones (using M[last,last] as est_raw): {jones_calc_Mlastlast.real:.8f} + {jones_calc_Mlastlast.imag:.8f}j")
        if abs(jones_calc_Mlastlast.real - target_jones_complex.real) < TOLERANCE and \
           abs(jones_calc_Mlastlast.imag - target_jones_complex.imag) < TOLERANCE:
            print("    VERIFICATION SUCCESS with M[last,last]")
        else:
            print("    VERIFICATION FAILURE with M[last,last]")
            
    end_time = time.time(); print(f"\nPart 3 Execution Time: {end_time - part3_start_time:.3f} seconds")
    
    
    master_end_time = time.time()
    print(f"\n--- Full Jones Polynomial Calculation from Abstract Anyons Finished ---")
    print(f"Total execution time: {master_end_time - master_start_time:.3f} seconds")

