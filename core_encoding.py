import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from utils.qtt import contract_cores_np


def _next_power_of_two(n):
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def cores_to_chain_mats(cores):
    cores_np = [_to_numpy(c) for c in cores]
    L = len(cores_np)
    mats = []
    for k, G in enumerate(cores_np):
        rL, _, _, rR = G.shape
        if k == 0:
            mats.append(G.reshape(4, rR))
        elif k == L - 1:
            mats.append(G.reshape(rL, 4))
        else:
            mats.append(G.reshape(2 * rL, 2 * rR))
    return mats


def mats_to_cores(mats):
    L = len(mats)
    cores = []
    for k, M in enumerate(mats):
        if k == 0:
            cores.append(M.reshape(1, 2, 2, M.shape[1]))
        elif k == L - 1:
            cores.append(M.reshape(M.shape[0], 2, 2, 1))
        else:
            cores.append(M.reshape(M.shape[0] // 2, 2, 2, M.shape[1] // 2))
    return cores


def contract_mats(mats):
    """Chain matrices -> (2^L, 2^L) image (classical contraction of the decoded cores)."""
    return contract_cores_np(mats_to_cores([np.asarray(m) for m in mats]))


def matrix_to_block_statevector(mat, mode="amplitude"):
    """Flatten (N1,N2) column-major, zero-pad to a power of two, normalise.

    mode="amplitude": amplitudes proportional to the entries (signed) - used for cores.
    mode="sqrt_prob": amplitudes = sqrt(mat / sum(mat)) for non-negative images.
    Returns (vec, meta); ``meta`` holds what is needed to invert the map.
    """
    mat = np.asarray(mat)
    N1, N2 = mat.shape
    total = N1 * N2
    padded = _next_power_of_two(total)
    if mode == "sqrt_prob":
        s = float(mat.sum())
        amp = np.sqrt(mat.astype(np.float64) / s).astype(np.complex128)
        extra = {"sum": s}
    elif mode == "amplitude":
        amp = mat.astype(np.complex128)
        extra = {}
    else:
        raise ValueError("mode must be 'amplitude' or 'sqrt_prob'")
    vec = amp.reshape(-1, order="F")
    if padded != total:
        vec = np.concatenate([vec, np.zeros(padded - total, dtype=np.complex128)])
    nrm = np.linalg.norm(vec)
    if nrm < 1e-12:
        raise ValueError("all-zero matrix")
    vec = vec / nrm
    meta = {"shape": (N1, N2), "total": total, "padded_total": padded, "mode": mode,
            "norm": float(nrm), "num_qubits": int(np.log2(padded)), **extra}
    return vec, meta


def block_statevector_to_matrix(vec, meta):
    """Inverse of ``matrix_to_block_statevector`` (multiplies the stored norm back)."""
    vec = np.asarray(vec, dtype=np.complex128)
    N1, N2 = meta["shape"]
    v = (vec * meta["norm"])[: meta["total"]]
    amp = v.reshape((N1, N2), order="F")
    if meta["mode"] == "amplitude":
        return amp
    return (np.abs(amp) ** 2) * meta["sum"]


def block_circuit(vec):
    """``initialize`` circuit preparing ``vec`` on log2(len(vec)) qubits."""
    n = int(np.log2(len(vec)))
    qc = QuantumCircuit(n)
    qc.initialize(np.asarray(vec, dtype=np.complex128), list(range(n)))
    return qc


def build_core_circuit(mats, mode="amplitude"):
    """All registers in one circuit (disjoint ``initialize`` blocks).

    Returns (qc, block_vecs, block_meta, groups) where ``groups[k]`` are the qubit indices
    of register k.
    """
    block_vecs, block_meta, groups = [], [], []
    off = 0
    for mat in mats:
        vec, meta = matrix_to_block_statevector(mat, mode=mode)
        block_vecs.append(vec)
        block_meta.append(meta)
        groups.append(list(range(off, off + meta["num_qubits"])))
        off += meta["num_qubits"]
    qc = QuantumCircuit(off)
    for vec, qubits in zip(block_vecs, groups):
        qc.initialize(vec, qubits)
    return qc, block_vecs, block_meta, groups


def _fix_global_phase(psi, ref):
    ov = np.vdot(ref, psi)
    return psi if abs(ov) < 1e-12 else psi * np.exp(-1j * np.angle(ov))


def decode_block_statevector(vec, meta):
    """Exact simulation of the register -> decoded matrix (complex amplitudes)."""
    sv = Statevector.from_instruction(block_circuit(vec)).data
    return block_statevector_to_matrix(_fix_global_phase(sv, vec), meta)


def counts_to_prob_vector(counts, num_qubits):
    probs = np.zeros(2 ** num_qubits, dtype=np.float64)
    total = sum(counts.values())
    if total == 0:
        return probs
    for bitstr, c in counts.items():
        probs[int(bitstr.replace(" ", ""), 2)] += c / total
    return probs


def decode_block_counts(counts, meta, sign_template=None):
    """Counts of one register -> decoded matrix ``sign * sqrt(p) * norm``.

    ``sign_template``: the classical matrix whose signs are re-applied (readout is
    sign-blind); if ``None`` all signs are taken positive.
    """
    p = counts_to_prob_vector(counts, meta["num_qubits"])
    mag = np.sqrt(p)
    if sign_template is not None:
        s = np.sign(np.real(np.asarray(sign_template)).reshape(-1, order="F"))
        s[s == 0] = 1.0
        s = np.concatenate([s, np.ones(meta["padded_total"] - meta["total"])])
        mag = mag * s
    return block_statevector_to_matrix(mag.astype(np.complex128), meta)


def decode_core_statevector(mats):
    """Exact per-register simulation -> contracted |image| (L2-normalised)."""
    rec = []
    for mat in mats:
        vec, meta = matrix_to_block_statevector(mat, mode="amplitude")
        rec.append(decode_block_statevector(vec, meta))
    x = np.abs(contract_mats(rec))
    return x / (np.linalg.norm(x) + 1e-12)
