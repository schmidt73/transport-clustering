import sys
import math
import random
import torch
import torchdyn

import pandas as pd
import argparse as ap
import numpy as np

from loguru import logger
from torchdyn.datasets import generate_moons
from scipy.spatial import distance
from ott.geometry import pointcloud, geometry

sys.path.append("src")

import FRLC.FRLC as frlc
import convex_lrot as clrot

def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data

def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.5)
    return x0 * 3 - 1

def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.2).float()

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("-n", "--n", type=int, default=100)
    parser.add_argument("-r", "--rank", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("-a", "--algorithm", default="clrot", choices=["clrot", "frlc", "lot"])
    parser.add_argument("--output", type=str, default="results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # set random seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float64

    batch_size1 = args.n
    batch_size2 = args.n

    g1 = np.ones((batch_size1)) / batch_size1
    g2 = np.ones((batch_size2)) / batch_size2

    x0 = sample_8gaussians(batch_size1)
    x1 = sample_moons(batch_size2)

    C = torch.from_numpy(distance.cdist(x0, x1)).to(device)
    C = C / C.max()

    rank = args.rank

    results = []
    if args.algorithm == "clrot":
        gamma = (1.0 / args.n)
        P, objective_lb = clrot.solve_nuclear_ot(
            C.cpu().numpy(), g1, g2, k=rank, gamma=gamma, max_iter=250, tolerance=1e-4, verbose=True
        )

        for i in range(args.restarts):
            L, R = clrot.nonnegative_rounding(P, g1, g2, rank, seed=args.seed + i)
            P_rounded = L @ R
            primal_cost = torch.sum(C * P_rounded)

            logger.info(f"CLROT objective: {objective_lb}, rounded objective: {primal_cost}")
            results.append({
                "objective_cost": float(primal_cost),
                "lower_bound": objective_lb,
                "rank": rank,
                "simulation_seed": args.seed,
                "num_restart": i,
                "algorithm": args.algorithm,
            })
    elif args.algorithm == "frlc":
        for i in range(args.restart):
            P, errs = frlc.FRLC_opt(
                C, device=device, r=rank, max_iter=20, returnFull=True, gamma=70, max_inneriters_balanced=500, max_inneriters_relaxed=500
            )

            primal_cost = torch.sum(C * P)
            logger.info(f"FRLC objective: {primal_cost}")
            results.append({
                "objective_cost": float(primal_cost),
                "lower_bound": None,
                "rank": rank,
                "simulation_seed": args.seed,
                "num_restart": i,
                "algorithm": args.algorithm,
            })

    results = pd.DataFrame(results)
    results.to_csv(f"{args.output}_{args.algorithm}_{args.n}_{args.rank}.csv", index=False)
            
