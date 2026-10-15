import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector


def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def cores_to_mps_4level(cores):
    """cores[k]: (r_{k-1},2,2,r_k) -> mps[k]: (r_{k-1},4,r_k) with p = 2*i + j."""
    mps = []
    for G in cores:
        G = _to_numpy(G)
        r0, _, _, r1 = G.shape
        mps.append(G.reshape(r0, 4, r1).astype(np.complex128))
    return mps


def right_canonicalize_mps(mps):
    """Right-canonicalise by SVD sweeping from the last tensor to the second (Appendix A)."""
    mps = [A.copy() for A in mps]
    for k in range(len(mps) - 1, 0, -1):
        A = mps[k]
        rL, d, rR = A.shape
        U, S, Vh = np.linalg.svd(A.reshape(rL, d * rR), full_matrices=False)
        chi = Vh.shape[0]
        mps[k] = Vh.reshape(chi, d, rR)
        mps[k - 1] = np.tensordot(mps[k - 1], U * S, axes=(2, 0))
    return mps


def normalize_first_tensor(mps, eps=1e-12):
    """After right-canonicalisation the global norm sits in A^(1); rescale it to 1."""
    A0 = mps[0]
    assert A0.shape[0] == 1, "expected left boundary rank 1"
    nrm = np.linalg.norm(A0[0].reshape(-1))
    if nrm < eps:
        raise ValueError("state norm is ~0")
    mps[0] = A0 / nrm
    return mps


def orthonormal_completion_keep_first(Q_valid, n_total, rng=None, eps=1e-12):
    """Extend the orthonormal columns of ``Q_valid`` (N, r) to an (N, n_total) matrix with
    orthonormal columns by Gram-Schmidt on random vectors (Eq. 1: U = [V W])."""
    if rng is None:
        rng = np.random.default_rng(0)
    N, r_valid = Q_valid.shape
    Q = np.zeros((N, n_total), dtype=np.complex128)
    Q[:, :r_valid] = Q_valid
    for j in range(r_valid, n_total):
        for _ in range(20):
            v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            v = v - Q[:, :j] @ (Q[:, :j].conj().T @ v)
            nv = np.linalg.norm(v)
            if nv > eps:
                break
        else:
            raise ValueError("failed to complete isometry basis")
        Q[:, j] = v / nv
    return Q


def mps_to_isometry_circuit(mps, seed=0):
    """Right-canonical MPS (physical dim 4) -> sequential circuit of ``UnitaryGate`` blocks.

    Register order: ``bond`` (m = ceil(log2 r_max) qubits, lowest indices) then ``q`` (2L data
    qubits).  Block k acts on bond qubits + [q[2k+1], q[2k]] (little-endian: q[2k] = i_{k+1}).
    """
    L = len(mps)
    if mps[0].shape[1] != 4:
        raise ValueError("physical dimension must be 4 (two qubits per site)")
    bonds = [mps[0].shape[0]] + [A.shape[2] for A in mps]
    rmax = max(bonds)
    m = int(np.ceil(np.log2(rmax))) if rmax > 1 else 0
    R = 2 ** m

    q_data = QuantumRegister(2 * L, "q")
    if m > 0:
        q_bond = QuantumRegister(m, "bond")
        qc = QuantumCircuit(q_bond, q_data)
        bond_qubits = list(q_bond)
    else:
        qc = QuantumCircuit(q_data)
        bond_qubits = []
    rng = np.random.default_rng(seed)

    for k, A in enumerate(mps):
        r_in, _, r_out = A.shape
        V_valid = np.zeros((4 * R, r_in), dtype=np.complex128)
        V_valid[np.add.outer(np.arange(4) * R, np.arange(r_out)).reshape(-1), :] = (
            np.transpose(A, (1, 2, 0)).reshape(4 * r_out, r_in))
        gram = V_valid.conj().T @ V_valid
        if not np.allclose(gram, np.eye(r_in), atol=1e-6):
            raise ValueError("MPS is not right-canonical; call right_canonicalize_mps first")
        V_iso = orthonormal_completion_keep_first(V_valid, R, rng=rng)
        U = orthonormal_completion_keep_first(V_iso, 4 * R, rng=rng)
        site_qubits = [q_data[2 * k + 1], q_data[2 * k]]
        qc.append(UnitaryGate(U), bond_qubits + site_qubits)
    return qc


def build_full_circuit(cores, seed=0):
    """Convenience: QTT cores -> TNQE-full circuit."""
    mps = normalize_first_tensor(right_canonicalize_mps(cores_to_mps_4level(cores)))
    return mps_to_isometry_circuit(mps, seed=seed)


def bits_interleaved_to_rowcol(bits, L):
    """bits = [i1, j1, i2, j2, ...] -> (row, col) with i1 the MSB of the row."""
    row = sum(bits[2 * k] << (L - 1 - k) for k in range(L))
    col = sum(bits[2 * k + 1] << (L - 1 - k) for k in range(L))
    return row, col


def amps_to_abs_image(amps, L):
    """Data-register amplitudes (little-endian index) -> |amplitude| image."""
    side = 2 ** L
    img = np.zeros((side, side), dtype=np.float64)
    for data_idx, a in enumerate(amps):
        bits = [(data_idx >> b) & 1 for b in range(2 * L)]
        r, c = bits_interleaved_to_rowcol(bits, L)
        img[r, c] = np.abs(a)
    return img


def counts_to_prob_image(counts, L):
    """Counts over the data register (Qiskit bitstrings, qubit 0 rightmost) -> prob image."""
    side = 2 ** L
    probs = np.zeros((side, side), dtype=np.float64)
    total = sum(counts.values())
    if total == 0:
        return probs
    for bitstr, c in counts.items():
        bits = [int(b) for b in bitstr.replace(" ", "")[::-1]]
        r, col = bits_interleaved_to_rowcol(bits, L)
        probs[r, col] += c / total
    return probs


def decode_full_statevector(qc, L):
    """Exact simulation -> |amplitude| image (bond post-selected on |0>, L2-normalised)."""
    num_data = 2 * L
    num_bond = qc.num_qubits - num_data
    sv = Statevector.from_instruction(qc).data
    if num_bond == 0:
        amps = sv
    else:
        mask = (1 << num_bond) - 1
        idx = np.arange(len(sv))
        sel = (idx & mask) == 0
        amps = np.zeros(2 ** num_data, dtype=np.complex128)
        amps[idx[sel] >> num_bond] = sv[sel]
    nrm = np.linalg.norm(amps)
    if nrm > 0:
        amps = amps / nrm
    return amps_to_abs_image(amps, L)


def decode_full_counts(counts, L, num_qubits):
    """Counts of the whole circuit -> probability image of the data register post-selected on
    bond = |0...0>.  Returns (prob_image, keep_rate)."""
    num_data = 2 * L
    num_bond = num_qubits - num_data
    post = {}
    kept = total = 0
    for bitstr, c in counts.items():
        b = bitstr.replace(" ", "").zfill(num_qubits)
        total += c
        if num_bond == 0:
            data_bits, ok = b, True
        else:
            data_bits, ok = b[:-num_bond], (b[-num_bond:] == "0" * num_bond)
        if ok:
            post[data_bits] = post.get(data_bits, 0) + c
            kept += c
    return counts_to_prob_image(post, L), kept / max(total, 1)
