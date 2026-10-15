import argparse
import json
import os
import time

import numpy as np

import _bootstrap
from utils.backends import aer_counts
from utils.circuit_stats import circuit_stats
from core_encoding import (block_circuit, build_core_circuit, contract_mats,
                           cores_to_chain_mats, decode_block_counts,
                           decode_core_statevector)
from utils.data_utils import crop28, get_dataset, save_gray_png, save_ground_truth
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
    ap.add_argument("--tag", default="amp", help="sub-folder under figures/core/")
    args = ap.parse_args()

    imgs, _ = get_dataset(args.dataset, args.start + args.n, root=args.data_root)
    prefix = "" if args.dataset == "mnist" else f"{args.dataset}/"
    fig_root = os.path.join(args.out_root, "figures", prefix)
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
        mats = cores_to_chain_mats(model.cores_numpy())
        qc, vecs, metas, groups = build_core_circuit(mats)
        if stats is None:
            stats = circuit_stats(qc)
            stats["register_sizes"] = [len(g) for g in groups]
            print("TNQE-core circuit:", {k: v for k, v in stats.items() if k != "ops"})

        amp_img = decode_core_statevector(mats)  # |contracted|, L2-normalised
        save_gray_png(os.path.join(fig_root, "core", args.tag, f"test_{i}.png"), crop(amp_img))
        if args.shots > 0:
            rec = []
            for mat, vec, meta in zip(mats, vecs, metas):
                counts = aer_counts(block_circuit(vec), args.shots, seed=args.seed + i)
                rec.append(decode_block_counts(counts, meta, sign_template=mat))
            x = np.abs(contract_mats(rec))
            x = x / (np.linalg.norm(x) + 1e-12)
            save_gray_png(os.path.join(fig_root, "core", "prob", f"test_{i}.png"), crop(x))
        if (i - args.start) % 10 == 0:
            print(f"[{i}] tn_mse={np.mean((recon - img) ** 2):.6f}  elapsed={time.time() - t0:.1f}s")

    os.makedirs(os.path.join(fig_root, "core"), exist_ok=True)
    with open(os.path.join(fig_root, "core", f"circuit_stats_r{args.rank}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
