from qiskit import transpile


def aer_counts(qc, shots, seed=None, measure=True):
    """Shot-based simulation with ``AerSimulator`` (noiseless)."""
    from qiskit_aer import AerSimulator

    sim = AerSimulator(seed_simulator=seed) if seed is not None else AerSimulator()
    qm = qc.copy()
    if measure:
        qm.measure_all()
    compiled = transpile(qm, sim)
    return sim.run(compiled, shots=shots).result().get_counts(compiled)


def get_ibm_backend(name=None):
    """Named backend (e.g. ``ibm_boston``) or the least busy real device."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    if name:
        return service.backend(name)
    return service.least_busy(operational=True, simulator=False)


def ibm_counts(qc, backend, shots, optimization_level=3):
    """Execute ``qc`` (measure_all appended) with SamplerV2; returns a counts dict.

    Circuits are transpiled with the Qiskit preset pass manager for the device; no error
    mitigation is applied (Appendix E).
    """
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    qm = qc.copy()
    qm.measure_all()
    compiled = transpile(qm, backend=backend, optimization_level=optimization_level)
    job = Sampler(mode=backend).run([compiled], shots=shots)
    return job.result()[0].data.meas.get_counts()
