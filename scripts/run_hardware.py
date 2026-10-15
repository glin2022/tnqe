import argparse
import json
import os

import numpy as np

import _bootstrap
from utils.amplitude import (amplitude_circuit, decode_amplitude_counts, decode_mnisq_counts,
                             load_mnisq_circuit)
from utils.backends import aer_counts, get_ibm_backend, ibm_counts
from core_encoding import (block_circuit, contract_mats, cores_to_chain_mats,
                           decode_block_counts, matrix_to_block_statevector)
from utils.data_utils import crop28, load_mnist32, save_gray_png
from utils.qtt import train_qtt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["amp", "auto", "core"])
    ap.add_argument("--backend", default=None, help="e.g. ibm_boston; default = least busy")
    ap.add_argument("--simulate", action="store_true", help="use AerSimulator instead of hardware")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--shots", type=int, default=50000)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--mnisq_root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_root", default="./outputs")
    args = ap.parse_args()

    if args.simulate:
        backend, bname = None, "aer"
        run = lambda qc: aer_counts(qc, args.shots)
    else:
        backend = get_ibm_backend(args.backend)
        bname = backend.name
        print("backend:", bname)
        run = lambda qc: ibm_counts(qc, backend, args.shots)

    fig_dir = os.path.join(args.out_root, "figures", "real", bname, args.method)
    cnt_dir = os.path.join(args.out_root, "counts", bname, args.method)
    os.makedirs(cnt_dir, exist_ok=True)
    imgs, _ = load_mnist32(args.start + args.n, root=args.data_root)

    for i in range(args.start, args.start + args.n):
        if args.method == "amp":
            qc, meta = amplitude_circuit(imgs[i])
            counts = run(qc)
            img = decode_amplitude_counts(counts, meta)          # 32x32 probabilities
            json.dump(counts, open(os.path.join(cnt_dir, f"test_{i}.json"), "w"))
        elif args.method == "auto":
            qc = load_mnisq_circuit(args.mnisq_root, i)
            counts = run(qc)
            img = decode_mnisq_counts(counts, qc.num_qubits)     # 28x28 probabilities
            json.dump(counts, open(os.path.join(cnt_dir, f"test_{i}.json"), "w"))
        else:
            model, _ = train_qtt(imgs[i], rank=args.rank, steps=args.steps, device=args.device, seed=i)
            mats = cores_to_chain_mats(model.cores_numpy())
            rec, all_counts = [], []
            for mat in mats:
                vec, meta = matrix_to_block_statevector(mat)
                counts = run(block_circuit(vec))
                all_counts.append(counts)
                rec.append(decode_block_counts(counts, meta, sign_template=mat))
            x = np.abs(contract_mats(rec))
            img = crop28(x / (np.linalg.norm(x) + 1e-12))
            json.dump(all_counts, open(os.path.join(cnt_dir, f"test_{i}.json"), "w"))
        save_gray_png(os.path.join(fig_dir, f"test_{i}.png"), img)
        print(f"[{i}] done")
    print("saved to", fig_dir)


if __name__ == "__main__":
    main()
