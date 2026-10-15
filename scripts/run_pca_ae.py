import argparse
import os

import numpy as np

import _bootstrap
from utils.amplitude import amplitude_circuit
from utils.circuit_stats import STATS_HEADER, circuit_stats, format_stats_row
from utils.data_utils import crop28, load_mnist32, save_gray_png


def pca_reconstruct(img, k):
    mean = img.mean(axis=0, keepdims=True)
    X = img - mean
    C = np.cov(X, rowvar=False)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    W = evecs[:, order[:k]]
    Z = X @ W
    return Z @ W.T + mean, Z, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--components", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_root", default="./outputs")
    args = ap.parse_args()

    imgs, _ = load_mnist32(args.n, root=args.data_root)
    print(STATS_HEADER)
    for k in args.components:
        stats = None
        for i in range(args.n):
            rec, Z, W = pca_reconstruct(imgs[i], k)
            save_gray_png(os.path.join(args.out_root, "figures", "pca", str(k), f"test_{i}.png"), crop28(rec))
            if stats is None:
                zw = np.concatenate([Z.reshape(-1, order="F"), W.reshape(-1, order="F")])
                qc, _ = amplitude_circuit(zw.reshape(1, -1))
                stats = circuit_stats(qc)
        print(format_stats_row(f"PCA+AE k={k}", stats))


if __name__ == "__main__":
    main()
