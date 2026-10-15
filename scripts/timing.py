import argparse
import time

import numpy as np

import _bootstrap
from utils.amplitude import amplitude_circuit, decode_amplitude_counts
from utils.backends import aer_counts
from core_encoding import (block_circuit, build_core_circuit, contract_mats,
                           cores_to_chain_mats, decode_block_counts)
from utils.data_utils import load_mnist32
from full_encoding import build_full_circuit, decode_full_counts
from utils.qtt import train_qtt
from unitary_encoding import build_unitary_circuit, decode_unitary_counts
from utils.unitary_qtt import train_unitary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["full", "core", "unitary", "amplitude"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--shots", type=int, default=80000)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data_root", default="./data")
    args = ap.parse_args()
    rank = args.rank or (8 if args.method == "unitary" else 4)
    imgs, _ = load_mnist32(args.n + 1, root=args.data_root)
    L = 5
    t_train, t_build, t_inf = [], [], []
    model = None
    for i in range(args.n + 1):
        img = imgs[i]
        t0 = time.time()
        if args.method == "amplitude":
            mats = None
        elif args.method == "unitary":
            model, _ = train_unitary(img, rank=rank, n_layers=args.layers, device=args.device, model=model)
        else:
            model, _ = train_qtt(img, rank=rank, device=args.device)
        t1 = time.time()
        if args.method == "amplitude":
            qc, meta = amplitude_circuit(img)
        elif args.method == "full":
            qc = build_full_circuit(model.cores_numpy())
        elif args.method == "core":
            mats = cores_to_chain_mats(model.cores_numpy())
            qc, vecs, metas, _ = build_core_circuit(mats)
        else:
            qc = build_unitary_circuit(model.theta_numpy(), L, rank, args.layers)
        t2 = time.time()
        if args.method == "amplitude":
            decode_amplitude_counts(aer_counts(qc, args.shots), meta)
        elif args.method == "full":
            decode_full_counts(aer_counts(qc, args.shots), L, qc.num_qubits)
        elif args.method == "core":
            rec = [decode_block_counts(aer_counts(block_circuit(v), args.shots), m, sign_template=mat)
                   for mat, v, m in zip(mats, vecs, metas)]
            contract_mats(rec)
        else:
            decode_unitary_counts(aer_counts(qc, args.shots), L, qc.num_qubits)
        t3 = time.time()
        if i > 0:  # skip warm-up
            t_train.append(t1 - t0)
            t_build.append(t2 - t1)
            t_inf.append(t3 - t2)
    print(f"{args.method}: Img. to TN = {np.mean(t_train):.4f} s | TN to QC = {np.mean(t_build):.4f} s | "
          f"inference = {np.mean(t_inf):.4f} s   (mean over {len(t_inf)} images)")


if __name__ == "__main__":
    main()
