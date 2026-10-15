import os

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def _next_power_of_two(n):
    return 1 << (n - 1).bit_length()


def matrix_to_statevector(mat, order="F"):
    mat = np.asarray(mat)
    vec = mat.astype(np.complex128).reshape(-1, order=order)
    n_raw = vec.size
    n_target = _next_power_of_two(n_raw)
    nrm = np.linalg.norm(vec)
    if nrm <= 1e-12:
        raise ValueError("zero matrix")
    vec = vec / nrm
    if n_target != n_raw:
        vec = np.pad(vec, (0, n_target - n_raw))
    meta = {"shape": tuple(mat.shape), "flat_size": n_raw, "padded_size": n_target,
            "num_qubits": int(np.log2(n_target)), "order": order, "norm": float(nrm)}
    return vec, meta


def amplitude_circuit(mat, order="F"):
    """Amplitude-encoding circuit of a 2-D array. Returns (qc, meta)."""
    vec, meta = matrix_to_statevector(mat, order=order)
    qc = QuantumCircuit(meta["num_qubits"])
    qc.initialize(vec, list(range(qc.num_qubits)))
    return qc, meta


def decode_amplitude_statevector(qc, meta):
    """Exact simulation -> |amplitude| image (shape ``meta['shape']``)."""
    sv = Statevector.from_instruction(qc).data
    flat = np.abs(sv[: meta["flat_size"]])
    return flat.reshape(meta["shape"], order=meta["order"])


def counts_to_prob_vector(counts, num_qubits):
    probs = np.zeros(2 ** num_qubits, dtype=np.float64)
    total = sum(counts.values())
    if total == 0:
        return probs
    for bitstr, c in counts.items():
        probs[int(bitstr.replace(" ", ""), 2)] += c / total
    return probs


def decode_amplitude_counts(counts, meta):
    """Counts -> probability image (shape ``meta['shape']``); sqrt gives |amplitude|."""
    p = counts_to_prob_vector(counts, meta["num_qubits"])
    return p[: meta["flat_size"]].reshape(meta["shape"], order=meta["order"])


def load_mnisq_circuit(root, idx):
    """Load ``<root>/qasm/<idx>`` (MNISQ ``base_test_mnist_784_f90`` layout)."""
    path = os.path.join(root, "qasm", str(idx))
    with open(path) as f:
        qc = QuantumCircuit.from_qasm_str(f.read())
    qc.remove_final_measurements(inplace=True)
    return qc


def decode_mnisq_statevector(qc, side=28):
    sv = Statevector.from_instruction(qc).data
    return np.abs(sv[: side * side]).reshape(side, side)


def decode_mnisq_counts(counts, num_qubits, side=28):
    p = counts_to_prob_vector(counts, num_qubits)
    return p[: side * side].reshape(side, side)
