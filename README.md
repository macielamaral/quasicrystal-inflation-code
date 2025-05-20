# Quasicrystal Inflation Code
Consistent Simulation of Fibonacci Anyon Braiding within a Qubit Quasicrystal Inflation Code.

**Authors:** Marcelo Amaral et al. (QGR and new company)
**Date:** May 6, 2025

## Abstract

The realization of fault-tolerant topological quantum computation often relies on manipulating non-Abelian anyons, such as Fibonacci anyons. Implementing their braiding operations on standard qubit platforms requires robust encoding schemes. This work investigates the simulation of standard Fibonacci anyon braiding within the protected subspace of a one-dimensional Quasicrystal Inflation Code realized on qubits. We define a local Hamiltonian whose ground state manifold enforces the code's constraints, reflecting physical Fibonacci tiling rules, and possesses the required Fibonacci degeneracy consistent with anyon fusion spaces. We confirm numerically that standard local qubit operators fail to satisfy the crucial Temperley-Lieb algebra governing anyon braiding when acting on the full qubit Hilbert space. We then present and validate a successful construction, implemented and verified for up to ten qubits: abstract Temperley-Lieb projector matrices, derived from established representations valid for the code's basis, are embedded into the qubit Hilbert space via a computationally constructed isometric mapping. Extensive numerical simulations confirm that the resulting embedded operators rigorously satisfy the defining algebraic relations of the Temperley-Lieb algebra for all tested system sizes. Consequently, the derived braid operators are unitary, satisfy the fundamental braid relations including the Yang-Baxter equation, and preserve the code subspace. This provides a strongly validated pathway for simulating Fibonacci topological quantum computation on qubits using the Quasicrystal Inflation Code framework, leveraging physically motivated constraints.

## Project Overview

This repository contains the code and resources for the research project "Consistent Simulation of Fibonacci Anyon Braiding within a Qubit Quasicrystal Inflation Code." The project focuses on:
- Defining a Quasicrystal Inflation Code (QIC) on qubits based on physical Fibonacci tiling rules.
- Constructing and validating Temperley-Lieb algebra generators and braid operators ($B'_k(N)_{original}$) within the QIC's protected subspace using a formal, $N$-dependent matrix representation.
- Simulating Fibonacci anyon braiding using $B'_k(N)_{original}$ and analyzing its resource scaling.
- Developing and evaluating an efficient, $N$-scalable circuit model for these $B'_k(N)$ operators based on fixed, local few-qubit unitary building blocks, suitable for near-term quantum hardware.
- Providing a framework for simulating Fibonacci TQC on standard qubit platforms.


## Project Structure

```text
quasicrystal-inflation-code/
├── .git/                     # Git internal files
├── .venv/                    # Python virtual environment (ignored by Git)
├── .gitignore                # Specifies intentionally untracked files
├── LICENSE                   # (To be added if you choose one)
├── README.md                 # This file
├── paper/                    # LaTeX source for the research paper
│   ├── fibonacci_anyon_qic_simulation.tex
│   └── references.bib        # And other paper-related files (images, etc.)
├── src/                      # Core library code
│   └── qic_core.py             # Core logic for QIC basis, formal operators B'_k(N)_original
│   └── qic_synthesis_tools.py  # Tools for variational circuit synthesis & local operator analysis
├── simulations/
│   ├── run_N3_verification.py        # Example: Verification of B'_k(N=3)_original properties
│   ├── run_full_Bk_resource_sweep.py # Estimates resources for B'_k(N)_original (exponential scaling)
│   ├── find_optimal_local_approximator.py # Analyzes B'_k(N)_original to find best G_tilde(N,k)
│   ├── batch_unitary_synthesis_analysis.py # Synthesizes many G_tilde targets to find best C_L, C_M, C_R candidates
│   ├── save_synthesized_ideal_gates.py   # One-time synthesis and saving of chosen C_L, C_M, C_R
│   └── evaluate_final_model_accuracy.py  # Evaluates efficient model (using saved C_L,C_M,C_R) against B'_k(N)_original
├── data/
│   ├── optimal_local_approximators/  # Stores G_tilde(N,k).npy files and summary CSV
│   ├── synthesized_circuits/         # Stores C_L.qpy, C_M.qpy, C_R.qpy
│   └── final_model_accuracy_eval/    # Stores overall accuracy and resource summary CSV
├── notebooks/
│   └── N3_results_analysis.ipynb
└── requirements.txt          # Python package dependencies
```

## Setup and Installation

1.  **Prerequisites:**
    * Git
    * Python 3 (version >= 3.9 recommended, as per Qiskit requirements)

2.  **Clone the repository:**
    ```bash
    git clone [https://github.com/macielamaral/quasicrystal-inflation-code.git](https://github.com/macielamaral/quasicrystal-inflation-code.git)
    cd quasicrystal-inflation-code
    ```

3.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project's functionalities are primarily located in two Python modules within the `src/` directory:
- `qic_core.py`: Contains the core logic for defining the Quasicrystal Inspired Code (QIC), generating its basis states, constructing the formal Temperley-Lieb algebra projectors $P_k^{anyon}(N)$, and building the corresponding $N$-qubit embedded operators $P'_k(N)_{original}$ and braid generators $B'_k(N)_{original}$.
- `qic_synthesis_tools.py`: Provides a suite of tools for analyzing the $B'_k(N)_{original}$ operators, extracting optimal local unitary components, and performing variational quantum circuit synthesis to find efficient, low-gate-count approximations for these local components.


Simulation scripts in the `simulations/` directory demonstrate how to use the core library. For example:
```bash
# Ensure your virtual environment is active: source .venv/bin/activate
python simulations/run_N3_verification.py
python simulations/run_N10_scaling_tests.py
```
## Key Functionalities (`src/qic_core.py`)

The `qic_core.py` library provides core functions for simulating the Quasicrystal Inflation Code (QIC) framework.

Key functionalities include:
- Generating the QIC basis (no "00" constraint) and corresponding state vectors.
  (Uses Fibonacci convention F0=0, F1=1, dimension $F_{N+2}$)
- Constructing the isometry V mapping the QIC basis to the full qubit space.
- Building abstract anyonic Temperley-Lieb projectors ($P_k^{\text{anyon}}$) based on Kauffman algebra rules (Ref: Kauffman & Lomonaco, arXiv:0705.4349).
- Embedding anyonic operators into the full qubit Hilbert space ($P'_k, B'_k$) using the isometry V.
- Verifying algebraic properties of both anyonic and embedded operators (TL algebra, braid relations, unitarity, etc.).
- (Optional, requires Qiskit) Building the QIC Hamiltonian ($\mathcal{H}_{\text{QIC}}$) as a `SparsePauliOp`.
- (Optional, requires Qiskit) Verifying state energy using Qiskit Aer's `EstimatorV2` to check for ground state preservation.


### Workflow for Developing and Evaluating Efficient Braid Operators:

The recommended workflow involves several stages:

**Stage 1: Understanding the Formal $B'_k(N)_{original}$ Operators (Optional Initial Analysis)**
This stage uses the original, non-efficient method to understand the baseline complexity.
* **Script:** `simulations/run_full_Bk_resource_sweep.py` (This is your original script that constructs the full $2^N \times 2^N$ matrices and transpiles them).
* **Purpose:** To observe and quantify the exponential scaling of resources when directly implementing $B'_k(N)_{original}$, motivating the need for an efficient local model.
* **Example:**
    ```bash
    python simulations/run_full_Bk_resource_sweep.py
    ```

**Stage 2: Analyzing $B'_k(N)_{original}$ for Optimal Local Components & Identifying Ideal Gates**
This stage aims to find the best possible 3-qubit local unitary parts $\tilde{G}(N,k)$ of $B'_k(N)_{original}$ and select $N$-independent ideal target gates.
* **Script:** `simulations/find_optimal_local_approximator.py`
* **Purpose:**
    1.  Computes $B'_k(N)_{original}$ for various $N$ and $k_{op\_idx}$ (ensuring unitarity via internal $P'$ purification).
    2.  Extracts the optimal $8 \times 8$ local unitary $\tilde{G}(N,k_{op\_idx})$ from $B'_k(N)_{original}$.
    3.  Calculates $\Delta(N,k_{op\_idx}) = || B'_k(N)_{original} - (\tilde{G}(N,k_{op\_idx}) \otimes I) ||_F$, quantifying the best 3-qubit local approximation error.
    4.  Saves $\tilde{G}(N,k_{op\_idx})$ matrices to `.npy` files (e.g., in `data/optimal_local_approximators/`).
    5.  Outputs comparison norms ($||\tilde{G}(N,k) - \tilde{G}(N-1,k)||_F$ and $||\tilde{G}(N,k) - \tilde{G}(N,k-1)||_F$) to help analyze convergence and distinctness of local gate types.
* **Example:**
    ```bash
    python simulations/find_optimal_local_approximator.py
    ```
* **Follow-up Analysis:** Manually analyze the output CSV and saved $\tilde{G}(N,k)$ matrices to identify a small set (e.g., 1 to 3) of distinct, converged $N$-independent ideal local $8 \times 8$ unitaries ($G_{ideal\_L}, G_{ideal\_M}, G_{ideal\_R}$), typically based on data from a sufficiently large $N$ (e.g., $N=10$ or $N=11$). Note down the paths to these chosen `.npy` files.

**Stage 3: Synthesizing Ideal Local Gates into Efficient Circuits ($C_L, C_M, C_R$)**
This stage takes the chosen ideal $8 \times 8$ unitary matrices and finds low-gate-count circuit implementations for them.
* **Script Option A (Systematic Batch Analysis):** `simulations/batch_unitary_synthesis_analysis.py`
    * **Purpose:** To systematically try different ansatz configurations (e.g., varying number of layers) for each of your chosen $G_{ideal}$ matrices (and potentially other $\tilde{G}(N,k)$ files) to find the best synthesis parameters.
    * **Example:**
        ```bash
        # (Configure GIDEAL_FILES and NUM_LAYERS_LIST in the script)
        python simulations/batch_unitary_synthesis_analysis.py
        ```
    * Use the output of this script (e.g., `data/batch_synthesis_results/gate_synthesis_batch_summary.csv`) to select the best circuit structure (e.g., number of layers) that achieves high synthesis fidelity for each of your $G_{ideal\_L}, G_{ideal\_M}, G_{ideal\_R}$.

* **Script Option B (Targeted Synthesis and Saving):** `simulations/save_synthesized_ideal_gates.py`
    * **Purpose:** To perform a final, focused synthesis for each of your chosen $G_{ideal\_L}, G_{ideal\_M}, G_{ideal\_R}$ using the best ansatz settings determined from Stage 3A (or chosen directly), and then save the resulting Qiskit `QuantumCircuit` objects (e.g., as $C_L.qpy, C_M.qpy, C_R.qpy$) and their achieved synthesis fidelities.
    * **Example (run once for each ideal gate):**
        ```bash
        # For G_ideal_L
        python simulations/save_synthesized_ideal_gates.py \
            --target_npy data/optimal_local_approximators/G_tilde_N10_kop0_act0.npy \
            --output_qpy data/synthesized_circuits/C_L.qpy \
            --layers 4 --max_iters 200 
            # Add other synthesis parameters if needed

        # Repeat for G_ideal_M and G_ideal_R, changing input and output file paths
        ```

**Stage 4: Evaluating the Final Efficient Model (Using Pre-Synthesized $C_L, C_M, C_R$)**
This is the final stage where you use your efficient local gate model and assess its performance.
* **Script:** `simulations/evaluate_final_model_accuracy.py`
* **Purpose:**
    1.  Loads your pre-synthesized local circuits $C_L, C_M, C_R$ from their `.qpy` files (and their known synthesis fidelities).
    2.  For various $N$ and $k_{op\_idx}$:
        * Constructs the "efficient model operator" $U_{eff\_model}(N,k)$ by placing the appropriate $C_L, C_M,$ or $C_R$ onto the target 3 qubits in an $N$-qubit system.
        * Computes the "ground truth" $B'_k(N)_{original}$ matrix.
        * Calculates the Overall Average Gate Fidelity and Overall Frobenius Difference between $B'_k(N)_{original}$ and $U_{eff\_model}(N,k)$.
        * Transpiles the $N$-qubit circuit (containing the efficient local block) for a target hardware backend and records the resource counts (depth, ECR gates, etc.).
* **Example:**
    ```bash
    # (Ensure paths to C_L.qpy, C_M.qpy, C_R.qpy and their synthesis fidelities are set in the script)
    python simulations/evaluate_final_model_accuracy.py
    ```
* **Output:** A CSV file (e.g., in `data/final_model_accuracy_eval/`) summarizing the synthesis fidelity of the used local block, the overall model accuracy (fidelity and Frobenius difference to $B'_{original}$), and the hardware resource metrics for your defined efficient model. This data demonstrates the $N$-independent low gate cost and quantifies the model approximation error.

This structured workflow allows for systematic development, analysis, and validation of your efficient circuit implementation for QIC braid operators.


## Citing This Work

If you use this code or research, please cite the associated paper:

M. Amaral et al. (2025). Consistent Simulation of Fibonacci Anyon Braiding within a Qubit Quasicrystal Inflation Code. (Journal/Preprint details to be added upon availability).

## License

License to be determined.

## Collaborators

- Marcelo Amaral
- ...

