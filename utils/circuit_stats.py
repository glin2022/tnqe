def circuit_stats(qc, reps=10):
    dec = qc.decompose(reps=reps)
    ops = dec.count_ops()
    u = int(sum(v for k, v in ops.items() if k != "cx" and k not in ("measure", "barrier", "reset")))
    cx = int(ops.get("cx", 0))
    return {"qubits": int(qc.num_qubits), "depth": int(dec.depth()), "unitary": u,
            "cnot": cx, "operators": u + cx, "ops": dict(ops)}


def format_stats_row(name, st):
    return f"| {name} | {st['qubits']} | {st['depth']} | {st['unitary']} | {st['cnot']} | {st['operators']} |"


STATS_HEADER = "| Method | Qubits | Depth | Unitary | CNOT | Operators |\n|---|---|---|---|---|---|"
