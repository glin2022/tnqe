import argparse
import os

import numpy as np

import _bootstrap
from utils.classifiers import load_classifier, predict
from utils.data_utils import load_gray_png, load_mnist28


def load_folder(folder, n, start=0):
    imgs = []
    for i in range(start, start + n):
        arr = load_gray_png(os.path.join(folder, f"test_{i}.png")) * 255.0
        if arr.shape == (32, 32):
            arr = arr[2:-2, 2:-2]
        imgs.append(arr.astype(np.float32))
    return np.stack(imgs)


def plot_confusion(cm, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.xticks(range(10))
    plt.yticks(range(10))
    for i in range(10):
        for j in range(10):
            plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if cm[i, j] > 0.5 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help='entries "Name=folder"')
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ckpt_dir", default="./checkpoints")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_dir", default="./outputs/classification")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    _, labels = load_mnist28(args.start + args.n, root=args.data_root)
    y = labels[args.start:args.start + args.n]
    models = {k: load_classifier(k, os.path.join(args.ckpt_dir, f"mnist_{k}.pth"), args.device)
              for k in ["cnn", "mlp"]}
    os.makedirs(args.out_dir, exist_ok=True)

    print("| Images | CNN acc (%) | MLP acc (%) |\n|---|---|---|")
    for entry in args.dirs:
        name, folder = entry.split("=", 1)
        x = load_folder(folder, args.n, args.start)
        accs = {}
        for k, m in models.items():
            pred = predict(m, x, device=args.device)
            accs[k] = 100.0 * float((pred == y).mean())
            cm = np.zeros((10, 10))
            for t, p in zip(y, pred):
                cm[t, p] += 1
            cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            safe = name.replace(" ", "_").replace("/", "_")
            plot_confusion(cm, os.path.join(args.out_dir, f"confusion_{safe}_{k}.png"), f"{name} ({k.upper()})")
        print(f"| {name} | {accs['cnn']:.2f} | {accs['mlp']:.2f} |")


if __name__ == "__main__":
    main()
