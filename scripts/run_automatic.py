import argparse
import json
import os

import _bootstrap
from utils.amplitude import decode_mnisq_statevector, load_mnisq_circuit
from utils.circuit_stats import circuit_stats
from utils.data_utils import save_gray_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnisq_root", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out_root", default="./outputs")
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, "figures", "auto")
    stats = None
    for i in range(args.start, args.start + args.n):
        qc = load_mnisq_circuit(args.mnisq_root, i)
        if stats is None:
            stats = circuit_stats(qc)
            print("automatic encoding circuit (image 0):", {k: v for k, v in stats.items() if k != "ops"})
        save_gray_png(os.path.join(out_dir, f"test_{i}.png"), decode_mnisq_statevector(qc))
    with open(os.path.join(out_dir, "circuit_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
