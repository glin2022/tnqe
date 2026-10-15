import argparse
import csv
import os

import numpy as np

import _bootstrap
from utils.amplitude import amplitude_circuit, load_mnisq_circuit
from utils.circuit_stats import STATS_HEADER, circuit_stats, format_stats_row
from core_encoding import build_core_circuit, cores_to_chain_mats
from utils.data_utils import load_mnist32
from full_encoding import build_full_circuit
from utils.qtt import train_qtt
from unitary_encoding import build_unitary_circuit


def random_cores(L, rank, seed=0):
    rng = np.random.default_rng(seed)
    cores = []
    for k in range(L):
        rl = 1 if k == 0 else rank
        rr = 1 if k == L - 1 else rank
        cores.append(rng.standard_normal((rl, 2, 2, rr)))
    return cores


def random_theta(L, rank, n_layers, seed=0):
    rng = np.random.default_rng(seed)
    m = int(np.log2(rank)) if rank > 1 else 0
    return 0.01 * rng.standard_normal((L, 2 * (m + 2) * n_layers))


def _p(st):
    return {k: v for k, v in st.items() if k != "ops"}


def table2(args):
    img, _ = load_mnist32(1, root=args.data_root)
    img = img[0]
    rows = {}
    qc, _ = amplitude_circuit(img)
    rows["Baseline (Amplitude)"] = circuit_stats(qc)
    if args.mnisq_root:
        rows["Automatic encoding"] = circuit_stats(load_mnisq_circuit(args.mnisq_root, 0))
    model, _ = train_qtt(img, rank=4, steps=args.steps, device="cpu", seed=0)
    cores = model.cores_numpy()
    rows["TNQE-full (r=4)"] = circuit_stats(build_full_circuit(cores))
    rows["TNQE-core (r=4)"] = circuit_stats(build_core_circuit(cores_to_chain_mats(cores))[0])
    rows["TNQE-unitary (r=8, N_l=4)"] = circuit_stats(build_unitary_circuit(random_theta(5, 8, 4), 5, 8, 4))
    print("\n### Table 2 (32x32 MNIST)\n" + STATS_HEADER)
    for k, v in rows.items():
        print(format_stats_row(k, v))
    _save(args, "table2.csv", rows)


def rank_sweep(args):
    ranks = [1, 2, 3, 4, 5, 8, 9, 16]
    rows = {}
    for r in ranks:
        cores = random_cores(5, r)
        rows[f"full r={r}"] = circuit_stats(build_full_circuit(cores))
        rows[f"core r={r}"] = circuit_stats(build_core_circuit(cores_to_chain_mats(cores))[0])
    for r in [1, 2, 4, 8, 16]:
        rows[f"unitary r={r} N_l=4"] = circuit_stats(build_unitary_circuit(random_theta(5, r, 4), 5, r, 4))
    print("\n### Rank sweep (Table 4 / 7 circuit rows)\n" + STATS_HEADER)
    for k, v in rows.items():
        print(format_stats_row(k, v))
    _save(args, "rank_sweep.csv", rows)


def layer_sweep(args):
    rows = {}
    for nl in [1, 2, 3, 4, 5, 6, 8, 10, 14, 20]:
        rows[f"unitary r=8 N_l={nl}"] = circuit_stats(build_unitary_circuit(random_theta(5, 8, nl), 5, 8, nl))
    print("\n### Layer sweep (Table 8 depth row)\n" + STATS_HEADER)
    for k, v in rows.items():
        print(format_stats_row(k, v))
    _save(args, "layer_sweep.csv", rows)


def scalability(args):
    """Figure 4a-c: image size 2^L for L = 2..9 (4x4 ... 512x512).

    Ranks follow the paper's setting: full/core rank_list = [2,2,4,4,8,8,16,16];
    unitary rank = 2^L/4 with layers = [2,2,4,4,16,32,64,128].
    """
    full_core_ranks = [2, 2, 4, 4, 8, 8, 16, 16]
    uni_layers = [2, 2, 4, 4, 16, 32, 64, 128]
    rows = {}
    rng = np.random.default_rng(0)
    for L in range(2, 10):
        s = 2 ** L
        r = full_core_ranks[L - 2]
        if s <= args.max_amp_size:
            qc, _ = amplitude_circuit(rng.normal(size=(s, s)))
            rows[f"amplitude s={s}"] = circuit_stats(qc)
        cores = random_cores(L, r, seed=L)
        rows[f"full s={s} r={r}"] = circuit_stats(build_full_circuit(cores))
        rows[f"core s={s} r={r}"] = circuit_stats(build_core_circuit(cores_to_chain_mats(cores))[0])
        ur, nl = max(1, s // 4), uni_layers[L - 2]
        rows[f"unitary s={s} r={ur} N_l={nl}"] = circuit_stats(build_unitary_circuit(random_theta(L, ur, nl), L, ur, nl))
        print(f"s={s}: " + "; ".join(f"{k.split()[0]}={_p(v)['depth']}" for k, v in rows.items() if f"s={s} " in k or k.endswith(f"s={s}")))
    print("\n### Scalability (Figure 4a-c)\n" + STATS_HEADER)
    for k, v in rows.items():
        print(format_stats_row(k, v))
    _save(args, "scalability.csv", rows)


def _save(args, name, rows):
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "qubits", "depth", "unitary", "cnot", "operators"])
        for k, v in rows.items():
            w.writerow([k, v["qubits"], v["depth"], v["unitary"], v["cnot"], v["operators"]])
    print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table2", action="store_true")
    ap.add_argument("--rank_sweep", action="store_true")
    ap.add_argument("--layer_sweep", action="store_true")
    ap.add_argument("--scalability", action="store_true")
    ap.add_argument("--mnisq_root", default=None)
    ap.add_argument("--steps", type=int, default=2000, help="QTT training steps for --table2")
    ap.add_argument("--max_amp_size", type=int, default=512,
                    help="largest image side for the amplitude-encoding curve (synthesis of "
                         "an 18-qubit initialize is slow)")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_dir", default="./outputs/resources")
    args = ap.parse_args()
    if not any([args.table2, args.rank_sweep, args.layer_sweep, args.scalability]):
        ap.error("choose at least one of --table2 --rank_sweep --layer_sweep --scalability")
    if args.table2:
        table2(args)
    if args.rank_sweep:
        rank_sweep(args)
    if args.layer_sweep:
        layer_sweep(args)
    if args.scalability:
        scalability(args)


if __name__ == "__main__":
    main()
