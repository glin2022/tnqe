import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def _is_power_of_two(x):
    return x > 0 and (x & (x - 1)) == 0


def build_unitary_circuit(theta, L, rank, n_layers):
    """TN-shaped circuit matching ``utils.unitary_qtt.block_unitary`` (bond register shared)."""
    theta = np.asarray(theta)
    rank = int(rank)
    if not _is_power_of_two(rank):
        raise ValueError(f"rank must be a power of 2, got {rank}")
    m = int(np.log2(rank))
    params_per_block = 2 * (m + 2) * n_layers
    assert theta.shape == (L, params_per_block), (theta.shape, (L, params_per_block))

    qc = QuantumCircuit(m + 2 * L)
    bond = list(range(m))
    for k in range(L):
        phys0, phys1 = m + 2 * k, m + 2 * k + 1
        th = theta[k]
        idx = 0
        for _ in range(n_layers):
            for q in bond + [phys0, phys1]:
                qc.u(float(th[idx]), float(th[idx + 1]), 0.0, q)
                idx += 2
            for q in range(m - 1):
                qc.cx(bond[q], bond[q + 1])
            for qi, q in enumerate(bond):
                qc.cx(q, phys0 if qi % 2 == 0 else phys1)
            qc.cx(phys0, phys1)
    return qc


def data_vector_to_image(vec, L):
    """Length 4^L vector indexed by the physical register (little-endian; qubit 2k = i_{k+1},
    qubit 2k+1 = j_{k+1}) -> (2^L, 2^L) image with the QTT row/column convention."""
    vec = np.asarray(vec)
    t = vec.reshape([2] * (2 * L), order="F")
    row_axes = list(range(0, 2 * L, 2))
    col_axes = list(range(1, 2 * L, 2))
    t = np.transpose(t, axes=row_axes + col_axes)
    return t.reshape((2 ** L, 2 ** L))


def postselect_bond_zero_statevector(sv_data, num_bond, num_data):
    """Amplitudes of the data register where the (lowest-index) bond qubits are all 0."""
    amps = np.zeros(2 ** num_data, dtype=np.complex128)
    if num_bond == 0:
        amps[:] = sv_data
        return amps
    mask = (1 << num_bond) - 1
    idx = np.arange(len(sv_data))
    sel = (idx & mask) == 0
    amps[idx[sel] >> num_bond] = sv_data[sel]
    return amps


def decode_unitary_statevector(qc, L):
    """Exact simulation -> normalised probability image (post-selected on bond = |0>)."""
    num_data = 2 * L
    num_bond = qc.num_qubits - num_data
    sv = Statevector.from_instruction(qc).data
    amps = postselect_bond_zero_statevector(sv, num_bond, num_data)
    prob = np.abs(amps) ** 2
    prob = prob / (prob.sum() + 1e-12)
    return data_vector_to_image(prob, L)


def decode_unitary_counts(counts, L, num_qubits):
    """Measurement counts (``measure_all`` bitstrings) -> probability image, post-selected on
    bond = |0>.  Returns (image, keep_rate)."""
    num_data = 2 * L
    num_bond = num_qubits - num_data
    prob = np.zeros(2 ** num_data, dtype=np.float64)
    kept = total = 0
    mask = (1 << num_bond) - 1
    for bitstr, c in counts.items():
        total += c
        full_idx = int(bitstr.replace(" ", "").zfill(num_qubits), 2)
        if num_bond == 0 or (full_idx & mask) == 0:
            prob[full_idx >> num_bond] += c
            kept += c
    prob = prob / (prob.sum() + 1e-12)
    return data_vector_to_image(prob, L), (kept / max(total, 1))
