import os

import numpy as np

from .data_utils import load_gray_png


def mse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def bce(target, pred, eps=1e-12):
    """Binary cross entropy with ``target`` the ground truth and ``pred`` the reconstruction."""
    t = np.clip(np.asarray(target, dtype=np.float64), eps, 1 - eps)
    p = np.clip(np.asarray(pred, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def psnr(a, b, max_val=1.0, eps=1e-12):
    m = max(mse(a, b), eps)
    return float(10.0 * np.log10(max_val ** 2 / m))


def ssim(a, b, max_val=1.0, k1=0.01, k2=0.03, eps=1e-12):
    """Global SSIM (statistics over the whole image, no sliding window)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {x.shape} vs {y.shape}")
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    mu_x, mu_y = x.mean(), y.mean()
    x0, y0 = x - mu_x, y - mu_y
    sxx, syy, sxy = (x0 * x0).mean(), (y0 * y0).mean(), (x0 * y0).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2)
    return float(num / (den + eps))


def all_metrics(target, pred):
    return {"MSE": mse(target, pred), "BCE": bce(target, pred),
            "PSNR": psnr(target, pred), "SSIM": ssim(target, pred)}


def mean_metrics(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def std_metrics(rows):
    keys = rows[0].keys()
    return {k: float(np.std([r[k] for r in rows])) for k in keys}


def evaluate_folders(pred_dir, org_dir, indices, pattern="test_{}.png", crop_pred=False,
                     skip_missing=False):
    """Average metrics between ``pred_dir/test_i.png`` and ``org_dir/test_i.png``.

    ``crop_pred``: crop a 32x32 prediction to its 28x28 centre before comparing.
    Returns (mean_dict, per_image_rows).
    """
    rows = []
    for i in indices:
        pp = os.path.join(pred_dir, pattern.format(i))
        op = os.path.join(org_dir, pattern.format(i))
        if skip_missing and not os.path.exists(pp):
            continue
        t = load_gray_png(op)
        p = load_gray_png(pp)
        if crop_pred and p.shape[0] == t.shape[0] + 4:
            p = p[2:-2, 2:-2]
        rows.append(all_metrics(t, p))
    if not rows:
        raise FileNotFoundError(f"no images found in {pred_dir}")
    return mean_metrics(rows), rows


def format_table(name_to_metrics, title=None):
    """Markdown table of {row_name: metrics_dict}."""
    keys = list(next(iter(name_to_metrics.values())).keys())
    lines = []
    if title:
        lines.append(f"**{title}**\n")
    lines.append("| Method | " + " | ".join(keys) + " |")
    lines.append("|---|" + "---|" * len(keys))
    for name, m in name_to_metrics.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[k]:.6f}" for k in keys) + " |")
    return "\n".join(lines)
