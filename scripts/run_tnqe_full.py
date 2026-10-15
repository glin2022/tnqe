import argparse
import json
import os
import time

import numpy as np

import _bootstrap
from utils.backends import aer_counts
from utils.circuit_stats import circuit_stats
from utils.data_utils import crop28, get_dataset, save_gray_png, save_ground_truth
from full_encoding import build_full_circuit, decode_full_counts, decode_full_statevector
from utils.qtt import train_qtt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--shots", type=int, default=0, help="0 = statevector decode only")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_root", default="./outputs")
    ap.add_argument("--tag", default="amp", help="sub-folder under figures/full/")
    ap.add_argument("--save_tn", action="store_true", help="also save the classical QTT recon")
    args = ap.parse_args()

    imgs, _ = get_dataset(args.dataset, args.start + args.n, root=args.data_root)
    prefix = "" if args.dataset == "mnist" else f"{args.dataset}/"
    fig_root = os.path.join(args.out_root, "figures", prefix)
    L = int(np.log2(imgs.shape[1]))
    crop = crop28 if args.dataset == "mnist" else (lambda x: x)

    stats = None
    t0 = time.time()
    for i in range(args.start, args.start + args.n):
        img = imgs[i]
        if img.sum() <= 0:
            continue
        if args.tag == "amp":
            save_ground_truth(fig_root, i, img, args.dataset)
        model, recon = train_qtt(img, rank=args.rank, steps=args.steps, lr=args.lr,
                                 device=args.device, seed=args.seed + i)
        qc = build_full_circuit(model.cores_numpy())
        if stats is None:
            stats = circuit_stats(qc)
            print("TNQE-full circuit:", {k: v for k, v in stats.items() if k != "ops"})

        if args.save_tn or args.tag == "amp":
            save_gray_png(os.path.join(fig_root, "TN", f"test_{i}.png"), crop(recon), autoscale=False)
        amp_img = decode_full_statevector(qc, L)
        save_gray_png(os.path.join(fig_root, "full", args.tag, f"test_{i}.png"), crop(amp_img))
        if args.shots > 0:
            counts = aer_counts(qc, args.shots, seed=args.seed + i)
            prob_img, keep = decode_full_counts(counts, L, qc.num_qubits)
            save_gray_png(os.path.join(fig_root, "full", "prob", f"test_{i}.png"), crop(np.sqrt(prob_img)))
        if (i - args.start) % 10 == 0:
            print(f"[{i}] tn_mse={np.mean((recon - img) ** 2):.6f}  elapsed={time.time() - t0:.1f}s")

    os.makedirs(os.path.join(fig_root, "full"), exist_ok=True)
    with open(os.path.join(fig_root, "full", f"circuit_stats_r{args.rank}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
