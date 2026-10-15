import argparse
import json
import os

import _bootstrap
from utils.amplitude import amplitude_circuit, decode_amplitude_statevector
from utils.circuit_stats import circuit_stats
from utils.data_utils import crop28, load_mnist32, save_gray_png, save_ground_truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_root", default="./outputs")
    args = ap.parse_args()

    imgs, _ = load_mnist32(args.start + args.n, root=args.data_root)
    out_dir = os.path.join(args.out_root, "figures", "AE")
    stats = None
    for i in range(args.start, args.start + args.n):
        save_ground_truth(os.path.join(args.out_root, "figures"), i, imgs[i])
        qc, meta = amplitude_circuit(imgs[i], order="F")
        if stats is None:
            stats = circuit_stats(qc)
            print("amplitude circuit:", {k: v for k, v in stats.items() if k != "ops"})
        img = decode_amplitude_statevector(qc, meta)
        save_gray_png(os.path.join(out_dir, f"test_{i}.png"), crop28(img))
    with open(os.path.join(out_dir, "circuit_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
