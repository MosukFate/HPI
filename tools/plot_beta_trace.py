#!/usr/bin/env python3
"""Plot HPI beta traces recorded by BetaTrackerHook."""

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot clip/dino beta curves from beta_trace.csv.')
    parser.add_argument('csv', help='Path to beta_trace.csv')
    parser.add_argument(
        '--out',
        default=None,
        help='Output image path. Defaults to <csv stem>.png.')
    parser.add_argument('--title', default='HPI beta during training')
    return parser.parse_args()


def _read_float(row, key):
    value = row.get(key, '')
    return None if value == '' else float(value)


def read_trace(path):
    iters, clip_beta, dino_beta = [], [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row['iter']))
            clip_beta.append(_read_float(row, 'clip_beta'))
            dino_beta.append(_read_float(row, 'dino_beta'))
    return iters, clip_beta, dino_beta


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path.with_suffix('.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    iters, clip_beta, dino_beta = read_trace(csv_path)
    if not iters:
        raise RuntimeError(f'No rows found in {csv_path}')

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.0, 3.6), dpi=200)
    plt.plot(iters, clip_beta, label='CLIP branch beta', linewidth=2.0)
    plt.plot(iters, dino_beta, label='DINOv2 branch beta', linewidth=2.0)
    plt.xlabel('Training iteration')
    plt.ylabel('Beta = sigmoid(logit)')
    plt.title(args.title)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f'Saved beta plot to {out_path}')


if __name__ == '__main__':
    main()
