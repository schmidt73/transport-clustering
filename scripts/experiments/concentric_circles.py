#!/usr/bin/env python3
"""
Generate two circles in ℝ², compute the pair-wise distance cost matrix,
add Gaussian white noise, and write everything to **text** files.

File naming scheme (prefix = args.output):
    <prefix>_C.txt          – clean cost matrix  C      (shape n×n)
    <prefix>_C_tilde.txt    – noisy cost matrix  C̃     (shape n×n)
    <prefix>_X_points.txt   – points on first circle    (shape n×2)
    <prefix>_Y_points.txt   – points on second circle   (shape n×2)
    <prefix>_metadata.json  – parameters for reproducibility
"""
from __future__ import annotations
import argparse
import json
import numpy as np


def equally_spaced_circle(radius: float, n: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def cost_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - y[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate two-circle cost matrix with Gaussian noise.")
    p.add_argument("--r1", type=float, required=True, help="radius of first circle")
    p.add_argument("--r2", type=float, required=True, help="radius of second circle")
    p.add_argument("-n", type=int, required=True, help="number of points on each circle")
    p.add_argument("--sigma", type=float, required=True, help="variance of Gaussian noise (σ > 0)")
    p.add_argument("-o", "--output", type=str, required=True, help="output prefix for files")
    p.add_argument("--seed", type=int, default=None, help="optional RNG seed for reproducibility")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # 1. Construct the two point clouds
    X = equally_spaced_circle(args.r1, args.n)
    Y = equally_spaced_circle(args.r2, args.n)

    # 2. Compute clean cost matrix C (n × n)
    C = cost_matrix(X, Y)

    # 3. Add Gaussian white noise with variance σ (std = √σ)
    noise = rng.normal(0.0, np.sqrt(args.sigma), size=C.shape)
    C_tilde = C + noise

    # 4. Save everything as plain-text
    np.savetxt(f"{args.output}_C.txt", C, fmt="%.8f")
    np.savetxt(f"{args.output}_C_tilde.txt", C_tilde, fmt="%.8f")
    np.savetxt(f"{args.output}_X_points.txt", X, fmt="%.8f")
    np.savetxt(f"{args.output}_Y_points.txt", Y, fmt="%.8f")

    # Metadata for later reference
    metadata = {
        "r1": args.r1,
        "r2": args.r2,
        "n": args.n,
        "sigma": args.sigma,
        "seed": args.seed,
        "cost_shape": C.shape,
    }
    with open(f"{args.output}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    main()
