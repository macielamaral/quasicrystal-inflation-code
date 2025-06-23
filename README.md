# Quasicrystal Inflation Code (QIC)

<sub>Validated framework for simulating Fibonacci-anyon braiding on qubits</sub>

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Lead author:** Marcelo M. Amaral  
**Version:** 1.1 · **Last update:** June 2025

---

## TL;DR

*QIC* encodes Fibonacci anyons in a 1-D qubit chain whose basis strings obey the simple “no 00” rule, mirroring the growth rules of Fibonacci quasicrystals.  
This repo contains:

* **`src/qic_core.py`** – the algebraic engine (basis, isometry *V*, Temperley-Lieb projectors, braid operators, Hamiltonian, checks).  
* **`data/gates/b_local_matrix.npy`** – the canonical **3-qubit `B-gate`**, a translationally invariant unitary that is the scalable building block for braids.  
* **`simulations/`** – driver scripts that verify operators, measure hardware leakage, calculate the Jones polynomial, and compare transpilation strategies.  
* **`paper/`** – LaTeX source of the companion manuscript.

The framework has been numerically verified up to **17 qubits** and demonstrated on an **IBM Quantum** processor.

---

## Repository layout
```text
quasicrystal-inflation-code/
│
├── data/                         # Binary assets (e.g. verified B-gate)
│   └── gates/
│       └── b\_local\_matrix.npy
│
├── paper/                        # LaTeX of the research paper
│
├── simulations/                  # Reproducible experiments & analyses
│   ├── compare\_transpilation\_methods.py
│   ├── compute\_jones\_quantum\_b\_gate.py
│   ├── run\_qic\_ibm\_leakage.py
│   └── ...  (see doc-strings inside)
│
├── src/
│   └── qic\_core.py               # Core library (import this!)
│
├── requirements.txt              # Minimal Python deps
├── pyproject.toml (optional)     # If you prefer PEP-517 installs
└── README.md                     # You are here
```

---

## Installation

```bash
git clone https://github.com/macielamaral/quasicrystal-inflation-code.git
cd quasicrystal-inflation-code

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt  # NumPy, SciPy, Qiskit, etc.
````

> **Note**
> Qiskit and Qiskit Aer are optional but required for the Hamiltonian and hardware-back-end examples.
> The code degrades gracefully if they are absent.

---

## Quick start

```python
from src.qic_core import get_qic_basis, construct_isometry_V, build_local_projector_matrix

# 1. Generate the QIC basis for N = 5 qubits
strings, vecs = get_qic_basis(5)
print(strings[:5])  # ['01010', '01011', '01101', ...]

# 2. Build the isometry V (2^N × F_{N+2})
V = construct_isometry_V(vecs)

# 3. Load the canonical 3-qubit B-gate
import numpy as np
B_gate = np.load("data/gates/b_local_matrix.npy")
```

Or run an end-to-end example:

```bash
python simulations/generate_and_demonstrate_gate.py         # build + save B-gate
python simulations/run_b_gate_scalable_quantum_verification.py
```

---

## Typical workflow

| Stage                        | Script                                        | What it does                                         |
| ---------------------------- | --------------------------------------------- | ---------------------------------------------------- |
| **1. Verify algebra**        | `run_operator_verification.py`                | Checks Temperley-Lieb & braid relations for any *N*. |
| **2. Build canonical gate**  | `generate_and_demonstrate_gate.py`            | Saves the local 3-qubit `B-gate`.                    |
| **3. Quantum verification**  | `run_b_gate_scalable_quantum_verification.py` | Proves unitarity & Yang-Baxter via simulator.        |
| **4. Hardware experiments**  | `run_qic_ibm_leakage.py`                      | Measures code-space leakage on IBM backend.          |
| **5. Topological invariant** | `compute_jones_quantum_b_gate.py`             | Computes the Jones polynomial with Hadamard tests.   |
| **6. Resource analysis**     | `compare_transpilation_methods.py`            | Shows >10× depth reduction using the local gate.     |

All scripts include detailed doc-strings explaining parameters and expected output.

---

## License

This project is released under the **MIT license** (see `LICENSE`).

---

### Contact / Collaborators

Marcelo M. Amaral
Pull requests and issues are welcome.
