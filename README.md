# Quasicrystal Inflation Code
Consistent Simulation of Fibonacci Anyon Braiding within a Qubit Quasicrystal Inflation Code.

**Authors:** Marcelo Amaral et al. (QGR and new company)
**Date:** May 6, 2025

## Abstract

The realization of fault-tolerant topological quantum computation often relies on manipulating non-Abelian anyons, such as Fibonacci anyons. Implementing their braiding operations on standard qubit platforms requires robust encoding schemes. This work investigates the simulation of standard Fibonacci anyon braiding within the protected subspace of a one-dimensional Quasicrystal Inflation Code realized on qubits. We define a local Hamiltonian whose ground state manifold enforces the code's constraints, reflecting physical Fibonacci tiling rules, and possesses the required Fibonacci degeneracy consistent with anyon fusion spaces. We confirm numerically that standard local qubit operators fail to satisfy the crucial Temperley-Lieb algebra governing anyon braiding when acting on the full qubit Hilbert space. We then present and validate a successful construction, implemented and verified for up to ten qubits: abstract Temperley-Lieb projector matrices, derived from established representations valid for the code's basis, are embedded into the qubit Hilbert space via a computationally constructed isometric mapping. Extensive numerical simulations confirm that the resulting embedded operators rigorously satisfy the defining algebraic relations of the Temperley-Lieb algebra for all tested system sizes. Consequently, the derived braid operators are unitary, satisfy the fundamental braid relations including the Yang-Baxter equation, and preserve the code subspace. This provides a strongly validated pathway for simulating Fibonacci topological quantum computation on qubits using the Quasicrystal Inflation Code framework, leveraging physically motivated constraints.

## Project Overview

This repository contains the code and resources for the research project "Consistent Simulation of Fibonacci Anyon Braiding within a Qubit Quasicrystal Inflation Code." The project focuses on:
- Defining a Quasicrystal Inflation Code (QIC) on qubits based on physical Fibonacci tiling rules.
- Constructing and validating Temperley-Lieb algebra generators and braid operators within the QIC's protected subspace.
- Simulating Fibonacci anyon braiding for up to N=10 qubits.
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
│   └── qic_core.py
├── simulations/              # Example scripts to run simulations and verifications
│   ├── run_N3_verification.py  # Example script for N=3 verification
│   ├── run_N10_scaling_tests.py # Example script for N=10 scaling
│   └── (other simulation scripts...)
├── notebooks/                # (Optional) Jupyter notebooks for analysis, visualization
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

The core logic for generating QIC basis states, constructing Temperley-Lieb operators, performing embeddings, and verification is located in `src/qic_core.py`.

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

## Citing This Work

If you use this code or research, please cite the associated paper:

M. Amaral et al. (2025). Consistent Simulation of Fibonacci Anyon Braiding within a Qubit Quasicrystal Inflation Code. (Journal/Preprint details to be added upon availability).

## License

License to be determined.

## Collaborators

- Marcelo Amaral
- ...

