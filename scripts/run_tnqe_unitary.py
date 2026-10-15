import argparse
import json
import os
import time

import numpy as np

import _bootstrap
from utils.backends import aer_counts
from utils.circuit_stats import circuit_stats
from utils.data_utils import crop28, get_dataset, save_gray_png, save_ground_truth
from unitary_encoding import (build_unitary_circuit, decode_unitary_counts,
                              decode_unitary_statevector)
from utils.unitary_qtt import train_unitary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=8, help="power of two")
    ap.add_argument("--layers", type=int, default=4, help="N_l layers per block")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--max_lr", type=float, default=3e-2)
    ap.add_argument("--shots", type=int, default=0, help="0 = statevector decode only")
    ap.add_argument("--warm_start", action="store_true",
                    help="reuse the model of the previous image as initialisation")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_root", default="./outputs")
    ap.add_argument("--tag", default="amp", help="sub-folder under figures/uni/")
    args = ap.parse_args()

    imgs, _ = get_dataset(args.dataset, args.start + args.n, root=args.data_root)
    prefix = "" if args.dataset == "mnist" else f"{args.dataset}/"
    fig_root = os.path.join(args.out_root, "figures", prefix)
    L = int(np.log2(imgs.shape[1]))
    crop = crop28 if args.dataset == "mnist" else (lambda x: x)

    stats, model, thetas = None, None, {}
    t0 = time.time()
    for i in range(args.start, args.start + args.n):
        img = imgs[i]
        if img.sum() <= 0:
            continue
        if args.tag == "amp":
            save_ground_truth(fig_root, i, img, args.dataset)
        model, recon = train_unitary(img, rank=args.rank, n_layers=args.layers, steps=args.steps,
                                     max_lr=args.max_lr, device=args.device,
                                     model=model if args.warm_start else None, seed=args.seed + i)
        theta = model.theta_numpy()
        thetas[str(i)] = theta
        qc = build_unitary_circuit(theta, L, args.rank, args.layers)
        if stats is None:
            stats = circuit_stats(qc)
            print("TNQE-unitary circuit:", {k: v for k, v in stats.items() if k != "ops"})

        if args.tag == "amp":
            save_gray_png(os.path.join(fig_root, "uni", "tn", f"test_{i}.png"), crop(recon))
        p_img = decode_unitary_statevector(qc, L)
        save_gray_png(os.path.join(fig_root, "uni", args.tag, f"test_{i}.png"), crop(p_img))
        if args.shots > 0:
            counts = aer_counts(qc, args.shots, seed=args.seed + i)
            p_meas, keep = decode_unitary_counts(counts, L, qc.num_qubits)
            save_gray_png(os.path.join(fig_root, "uni", "prob", f"test_{i}.png"), crop(p_meas))
        if (i - args.start) % 10 == 0:
            print(f"[{i}] mse={np.mean((recon - img) ** 2):.6f}  elapsed={time.time() - t0:.1f}s")

    out_dir = os.path.join(fig_root, "uni", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "theta.npz"), **thetas)
    with open(os.path.join(fig_root, "uni", f"circuit_stats_r{args.rank}_nl{args.layers}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
