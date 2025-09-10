#!/usr/bin/env python3
"""
Construct two Gaussian mixtures with k means, then build the pair-wise
Euclidean-distance (cost) matrix between their samples.

Steps
-----
1.  Place k means μ₁,…,μ_k as vertices of a regular simplex in ℝ^{k−1}
    scaled so every pair is unit distance apart.
2.  Draw n₁,…,n_k points X from N(μ_i, σ² I).
3.  Perturb each μ_i by N(0, τ² I) → μ'₁,…,μ'_k  (τ = --perturb-scale).
4.  Draw Y from the perturbed mixture with the same counts.
5.  Return C ∈ ℝ^{n×n},  C_{ij} = ‖X_i − Y_j‖₂.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

def regular_simplex(k: int) -> np.ndarray:
    """
    Return k vertices of a regular simplex embedded in ℝ^{k−1}
    with pairwise Euclidean distance exactly 1.

    Construction: take standard basis in ℝ^{k}, subtract the centroid,
    then scale by √(k / (2(k−1))) so ‖μ_i − μ_j‖₂ = 1.
    We finally drop the last coordinate to live in ℝ^{k−1}.
    """
    e = np.eye(k)                       # shape (k, k)
    centroid = np.full((k, 1), 1 / k)
    verts = e - centroid                # centred in ℝ^{k}
    scale = np.sqrt(k / (2.0 * (k - 1)))
    verts *= scale
    return verts[:, :-1]                # drop final coord → ℝ^{k−1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--k", type=int, required=True,
                   help="number of mixture components / means")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="std-dev of each isotropic Gaussian")
    p.add_argument("--perturb-scale", type=float, default=0.2,
                   help="std-dev of mean perturbation (τ)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--counts", type=float, nargs="+",
                       help="explicit n₁ … n_k (list of positive ints)")
    group.add_argument("--n-total", type=int,
                       help="total #points; distributed as ⌊n/k⌋ each, remainder on last")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument("--out", type=str, help="path to write cost matrix (.npy);"
                   " if omitted, matrix is printed as text on stdout")
    return p.parse_args()

def sample_mixture(means: np.ndarray, counts: np.ndarray, sigma: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw samples and return (samples, labels ∈ {0,…,k−1}).
    """
    dim = means.shape[1]
    pts = []
    labels = []
    for idx, n_i in enumerate(counts):
        samp = rng.normal(loc=means[idx], scale=sigma, size=(n_i, dim))
        pts.append(samp)
        labels.append(np.full(n_i, idx, dtype=int))
    return np.vstack(pts), np.concatenate(labels)

def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    k = args.k
    if args.counts is not None:
        if len(args.counts) != k:
            sys.exit("Error: --counts must supply k numbers")
        counts = np.asarray(args.counts, dtype=int)
    else:
        base = args.n_total // k
        counts = np.full(k, base, dtype=int)
        counts[-1] += args.n_total - base * k   # add remainder to last cluster

    means = regular_simplex(k)                 # shape (k, k−1)

    # First mixture: X
    X, labels_X = sample_mixture(means, counts, args.sigma, rng)

    # Perturb means (isotropic Gaussian noise, τ = perturb_scale)
    perturbations = rng.normal(scale=args.perturb_scale, size=means.shape)
    Y, labels_Y = [], []
    for idx in range(X.shape[0]):
        Y_i = X[idx] + perturbations[labels_X[idx]]
        Y.append(Y_i)
        labels_Y.append(labels_X[idx])
    Y = np.array(Y)
    labels_Y = np.array(labels_Y)
    print(Y - X)
    # Cost matrix: pairwise Euclidean distances
    # Broadcasting: X (n, d) -> (n, 1, d), Y (m, d) -> (1, m, d)
    diff = X[:, None, :] - Y[None, :, :]
    C = np.linalg.norm(diff, axis=2) ** 2

    print(labels_X)
    print(labels_Y)
    P = np.zeros_like(C)
    for i, label_i in enumerate(labels_X):
        for j, label_j in enumerate(labels_Y):
            if label_i == label_j:
                P[i, j] = 1.0
                P[j, i] = 1.0

    P = P / P.sum()
    # --- Output ----------------------------------------------------------------
    
    if args.out:
        np.savetxt(args.out + "_cost_matrix.txt", C,  fmt="%.6f")
        print(f"Cost matrix saved to {args.out}_cost_matrix.txt (shape {C.shape})", file=sys.stderr)

        np.savetxt(args.out + "_optimal_plan.txt", P, fmt="%.6f")
        print(f"Optimal plan saved to {args.out}_optimal_plan.txt (shape {P.shape})", file=sys.stderr)

        np.savetxt(args.out + "_X.txt", X, fmt="%.6f")
        print(f"X samples saved to {args.out}_X.txt (shape {X.shape})", file=sys.stderr)    

        np.savetxt(args.out + "_Y.txt", Y, fmt="%.6f")
        print(f"Y samples saved to {args.out}_Y.txt (shape {Y.shape})", file=sys.stderr)

        metadata = {
            "k": k,
            "sigma": args.sigma,
            "perturb_scale": args.perturb_scale,
            "counts": counts.tolist(),
            "seed": args.seed,
            "cost": np.sum((C / C.max()) * P)
        }

        with open(args.out + "_metadata.json", "w") as f:
            import json
            json.dump(metadata, f, indent=4)

        print(f"Metadata saved to {args.out}_metadata.json", file=sys.stderr)

if __name__ == "__main__":
    main()
