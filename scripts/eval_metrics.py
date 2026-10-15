import argparse

import _bootstrap
from utils.metrics import evaluate_folders, format_table, mean_metrics, std_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True, help="reference folder (test_<i>.png)")
    ap.add_argument("--pred", nargs="+", required=True, help='entries "Name=folder"')
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--subsets", type=int, default=1)
    ap.add_argument("--crop_pred", action="store_true")
    ap.add_argument("--skip_missing", action="store_true")
    args = ap.parse_args()

    table = {}
    for entry in args.pred:
        name, folder = entry.split("=", 1)
        if args.subsets <= 1:
            m, _ = evaluate_folders(folder, args.org, range(args.start, args.start + args.n),
                                    crop_pred=args.crop_pred, skip_missing=args.skip_missing)
            table[name] = m
        else:
            per = args.n // args.subsets
            sub_means = []
            for j in range(args.subsets):
                idx = range(args.start + j * per, args.start + (j + 1) * per)
                m, _ = evaluate_folders(folder, args.org, idx, crop_pred=args.crop_pred,
                                        skip_missing=args.skip_missing)
                sub_means.append(m)
                table[f"{name} / subset {j + 1}"] = m
            mu, sd = mean_metrics(sub_means), std_metrics(sub_means)
            table[f"{name} / mean"] = mu
            table[f"{name} / std"] = sd
    print(format_table(table, title=f"n={args.n}, reference={args.org}"))


if __name__ == "__main__":
    main()
