import sys
import math
import random
import torch
import torchdyn
import jax.numpy as jnp

import pandas as pd
import argparse as ap
import numpy as np

from loguru import logger

sys.path.append("src")

import FRLC.FRLC as frlc
import convex_lrot as clrot

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("cost_matrix", help="Cost matrix.")
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

    C = np.loadtxt(args.cost_matrix)
    C = C / C.max() # normalize cost matrix

    batch_size1 = C.shape[0]
    batch_size2 = C.shape[1]

    g1 = np.ones((batch_size1)) / batch_size1
    g2 = np.ones((batch_size2)) / batch_size2

    rank = args.rank

    results = []
    if args.algorithm == "clrot":
        gamma = (1.0 / min(batch_size1, batch_size2))
        print(gamma * rank)
        C = jnp.array(C)

        P, objective_lb = clrot.solve_nuclear_ot(
            C, jnp.array(g1), jnp.array(g2), k=rank, gamma=gamma, max_iter=500, tolerance=1e-4, verbose=True
        )

        for i in range(args.restarts):
            L, R = clrot.nonnegative_rounding(P, g1, g2, rank, seed=args.seed + i)
            P_rounded = L @ R
            primal_cost = jnp.sum(C * P_rounded)

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
        C = torch.from_numpy(C)
        for i in range(args.restarts):
            P, errs = frlc.FRLC_opt(
                C, device=device, r=rank, max_iter=20, returnFull=True, gamma=70, max_inneriters_balanced=500, max_inneriters_relaxed=500
            )

            print(P.sum(axis=0)) # get deviations
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
    results.to_csv(f"{args.output}", index=False)
            
