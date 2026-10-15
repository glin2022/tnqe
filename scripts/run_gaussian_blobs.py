import argparse
import os

import numpy as np

import _bootstrap
from utils.amplitude import amplitude_circuit, decode_amplitude_counts, decode_amplitude_statevector
from utils.backends import get_ibm_backend, ibm_counts
from core_encoding import (block_circuit, contract_mats, cores_to_chain_mats,
                           decode_block_counts, decode_core_statevector,
                           matrix_to_block_statevector)
from utils.data_utils import gaussian_blob, minmax01, save_gray_png
from utils.qtt import train_qtt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--hardware", default=None, help="IBM backend name")
    ap.add_argument("--hw_amp_max_size", type=int, default=32)
    ap.add_argument("--shots", type=int, default=50000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", default="./outputs/gaussian")
    args = ap.parse_args()

    backend = get_ibm_backend(args.hardware) if args.hardware else None
    for n in args.sizes:
        img = gaussian_blob(n)
        save_gray_png(os.path.join(args.out_dir, "gt", f"{n}.png"), img, autoscale=False)

        qc, meta = amplitude_circuit(img)
        amp_img = decode_amplitude_statevector(qc, meta) * meta["norm"]
        save_gray_png(os.path.join(args.out_dir, "amp", f"test_{n}.png"), amp_img, autoscale=False)
        if backend is not None and n <= args.hw_amp_max_size:
            counts = ibm_counts(qc, backend, args.shots)
            p = decode_amplitude_counts(counts, meta)
            save_gray_png(os.path.join(args.out_dir, "real", "amp", f"test_{n}.png"), minmax01(p), autoscale=False)

        model, recon = train_qtt(img, rank=args.rank, steps=args.steps, device=args.device, seed=n)
        mats = cores_to_chain_mats(model.cores_numpy())
        ours = decode_core_statevector(mats)
        save_gray_png(os.path.join(args.out_dir, "ours", f"test_{n}.png"), minmax01(ours), autoscale=False)
        print(f"n={n}: qtt mse={np.mean((recon - img) ** 2):.3e}")
        if backend is not None:
            rec = []
            for mat in mats:
                vec, bmeta = matrix_to_block_statevector(mat)
                counts = ibm_counts(block_circuit(vec), backend, args.shots)
                rec.append(decode_block_counts(counts, bmeta, sign_template=mat))
            x = np.abs(contract_mats(rec))
            save_gray_png(os.path.join(args.out_dir, "real", "ours", f"test_{n}.png"), minmax01(x), autoscale=False)
    print("done ->", args.out_dir)


if __name__ == "__main__":
    main()
