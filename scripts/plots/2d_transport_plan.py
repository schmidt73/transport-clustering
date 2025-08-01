#!/usr/bin/env python3
"""
Plot a transport plan between two point distributions in ℝ².

Usage
-----
python plot_transport_plan.py --x X_points.txt --y Y_points.txt --p plan.txt [-o out.png] [--thresh 0.0]

All inputs are plain-text files produced by numpy.savetxt,
with shape:
    X : (n, 2)
    Y : (m, 2)
    P : (n, m)
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise an optimal transport plan between two 2-D point sets.")
    p.add_argument("--x", required=True, help="text file with first set of points (n × 2)")
    p.add_argument("--y", required=True, help="text file with second set of points (m × 2)")
    p.add_argument("--p", required=True, help="text file with transport plan matrix P (n × m)")
    p.add_argument("-o", "--output", default="transport_plan.png", help="output image filename (PNG)")
    p.add_argument("--thresh", type=float, default=0.0, help="draw lines only for P[i,j] > thresh (default: 0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load data
    X = np.loadtxt(args.x)
    Y = np.loadtxt(args.y)
    P = np.loadtxt(args.p)

    if X.shape[0] != P.shape[0] or Y.shape[0] != P.shape[1]:
        raise ValueError("Shape mismatch: X is %s, Y is %s, but P is %s"
                         % (X.shape, Y.shape, P.shape))

    max_w = P.max()
    if max_w == 0:
        raise ValueError("Transport matrix is all zeros – nothing to plot.")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X[:, 0], X[:, 1], c="tab:red", label="X points", zorder=3)
    ax.scatter(Y[:, 0], Y[:, 1], c="tab:blue", label="Y points", zorder=3)

    # Draw transport lines
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            w = P[i, j]
            if w <= args.thresh:
                continue
            # Scale line width and alpha by transported mass
            lw = 0.5 + 4.0 * (w / max_w)
            alpha = 0.1  + 0.15 * (w / max_w)
            ax.plot(
                [X[i, 0], Y[j, 0]],
                [X[i, 1], Y[j, 1]],
                color="gray",
                linewidth=lw,
                alpha=alpha,
                zorder=1,
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend(loc="upper right")
    ax.set_title("Transport plan (|P|>%.3g)" % args.thresh)
    fig.tight_layout()

    out_path = Path(args.output)
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path.resolve()}")


if __name__ == "__main__":
    main()
