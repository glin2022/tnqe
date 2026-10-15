import os

import numpy as np


def load_mnist32(n, root="./data", train=False):
    """First ``n`` MNIST images, zero-padded to 32x32, in [0,1]. Returns (imgs, labels)."""
    from torchvision import datasets, transforms

    ds = datasets.MNIST(root=root, train=train, download=True,
                        transform=transforms.Compose([transforms.Pad(2)]))
    n = min(n, len(ds))
    imgs = np.stack([np.asarray(ds[i][0], dtype=np.float64) / 255.0 for i in range(n)])
    labels = np.array([int(ds[i][1]) for i in range(n)])
    return imgs, labels


def load_mnist28(n, root="./data", train=False):
    """First ``n`` raw 28x28 MNIST images in [0,1]. Returns (imgs, labels)."""
    from torchvision import datasets

    ds = datasets.MNIST(root=root, train=train, download=True)
    n = min(n, len(ds))
    imgs = np.stack([np.asarray(ds[i][0], dtype=np.float64) / 255.0 for i in range(n)])
    labels = np.array([int(ds[i][1]) for i in range(n)])
    return imgs, labels


def load_cifar_gray32(n, root="./data", train=False):
    """First ``n`` CIFAR-10 images converted to grayscale (PIL ``L`` mode), 32x32, in [0,1]."""
    from torchvision import datasets, transforms

    ds = datasets.CIFAR10(root=root, train=train, download=True,
                          transform=transforms.Grayscale(num_output_channels=1))
    n = min(n, len(ds))
    imgs, labels = [], []
    for i in range(n):
        im = np.asarray(ds[i][0], dtype=np.float64) / 255.0
        if im.ndim == 3:
            im = im[..., 0]
        imgs.append(im)
        labels.append(int(ds[i][1]))
    return np.stack(imgs), np.array(labels)


def get_dataset(name, n, root="./data"):
    """Uniform access: ``name`` in {"mnist", "cifar"}; returns (imgs (n,32,32), labels)."""
    if name == "mnist":
        return load_mnist32(n, root=root)
    if name == "cifar":
        return load_cifar_gray32(n, root=root)
    raise ValueError(f"unknown dataset {name!r}")


def gaussian_blob(n, sigma=None):
    """n x n image with a centred Gaussian blob (sigma defaults to n/4), values in (0,1]."""
    if sigma is None:
        sigma = n / 4.0
    x = np.arange(n) - (n - 1) / 2.0
    xx, yy = np.meshgrid(x, x)
    return np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))


def save_gray_png(path, img, autoscale=True):
    """Save a 2-D array as an 8-bit grayscale PNG.

    autoscale=True  : ``plt.imsave(path, 1-img, cmap="Greys")`` (convention behind all reported metrics) -> the
                      stored intensity is ``(img-min)/(max-min)``.
    autoscale=False : ``plt.imsave(path, img, cmap="gray", vmin=0, vmax=1)`` -> stored
                      intensity is ``clip(img,0,1)`` (used for the classical TN reconstruction).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img = np.asarray(img, dtype=np.float64)
    if autoscale:
        plt.imsave(path, 1.0 - img, cmap="Greys")
    else:
        plt.imsave(path, img, cmap="gray", vmin=0.0, vmax=1.0)


def load_gray_png(path):
    """Read a PNG as grayscale float array in [0,1]."""
    from PIL import Image

    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def crop28(img):
    """32x32 -> central 28x28 (undo the MNIST zero padding)."""
    return img[2:-2, 2:-2]


def minmax01(img, eps=1e-12):
    """Min-max normalise to [0,1] (what ``save_gray_png(autoscale=True)`` stores)."""
    img = np.asarray(img, dtype=np.float64)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < eps:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def save_ground_truth(fig_root, i, img, dataset="mnist"):
    """Store the ground-truth image used by the metric / classification scripts
    (``<fig_root>/org/test_<i>.png``; MNIST is saved at 28x28, CIFAR at 32x32)."""
    gt = crop28(img) if dataset == "mnist" else img
    save_gray_png(os.path.join(fig_root, "org", f"test_{i}.png"), gt, autoscale=True)
