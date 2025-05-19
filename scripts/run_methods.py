import sys
import math
import random
import torch
import torchdyn
import jax.numpy as jnp
import time

import pandas as pd
import argparse as ap
import numpy as np

from ott.geometry.geometry import Geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from loguru import logger

sys.path.append("../src")

import FRLC.FRLC as frlc
import convex_lrot as clrot

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("cost_matrix", help="Cost matrix.")
    parser.add_argument("-r", "--rank", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-o", "--output", type=str, default="results")
    parser.add_argument("-a", "--algorithm", default="clrot", choices=["clrot", "frlc", "lot", "fullrank_round"])
    parser.add_argument("--restarts", type=int, default=10)
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
        C = jnp.array(C)

        start_time = time.time()
        P, objective_lb = clrot.solve_nuclear_ot(
            C, jnp.array(g1), jnp.array(g2), k=rank, gamma=gamma, max_iter=500, tolerance=1e-4, verbose=True
        )
        end_time   = time.time()
        solve_time = end_time - start_time

        for i in range(args.restarts):
            start_time = time.time()
            L, R = clrot.nonnegative_rounding(P, g1, g2, rank, seed=args.seed + i)
            end_time   = time.time()
            round_time = end_time - start_time
            P_rounded = L @ R
            primal_cost = jnp.sum(C * P_rounded)

            l1_row_error = jnp.sum(jnp.abs(g1  - P_rounded.sum(axis=0)))
            l1_col_error = jnp.sum(jnp.abs(g2  - P_rounded.sum(axis=1)))
            l1_error     = jnp.sum(jnp.abs(1.0 - P_rounded.sum()))

            logger.info(f"CLROT objective: {objective_lb}, rounded objective: {primal_cost}")
            res = {
                "objective_cost": float(primal_cost),
                "lower_bound": objective_lb,
                "rank": rank,
                "simulation_seed": args.seed,
                "num_restart": i,
                "algorithm": args.algorithm,
                "l1_row_marginal_error": l1_row_error,
                "l1_col_marginal_error": l1_col_error,
                "l1_total_error": l1_error,
                "runtime": solve_time + round_time
            }

            results.append(res)
    elif args.algorithm == "frlc":
        C = torch.from_numpy(C)
        for i in range(args.restarts):
            start_time = time.time()
            P, errs = frlc.FRLC_opt(
                C, device=device, r=rank, max_iter=20, returnFull=True, gamma=70, max_inneriters_balanced=500, max_inneriters_relaxed=500
            )
            end_time   = time.time()
            solve_time = end_time - start_time

            P = P.numpy()
            l1_row_error = np.sum(np.abs(g1  - P.sum(axis=0)))
            l1_col_error = np.sum(np.abs(g2  - P.sum(axis=1)))
            l1_error     = np.sum(np.abs(1.0 - P.sum()))

            primal_cost = torch.sum(C * P)
            logger.info(f"FRLC objective: {primal_cost}")
            res = {
                "objective_cost": float(primal_cost),
                "lower_bound": None,
                "rank": rank,
                "simulation_seed": args.seed,
                "num_restart": i,
                "algorithm": args.algorithm,
                "l1_row_marginal_error": l1_row_error,
                "l1_col_marginal_error": l1_col_error,
                "l1_total_error": l1_error,
                "runtime": solve_time
            }

            results.append(res)
    elif args.algorithm == "lot":
        geom = Geometry(cost_matrix=C)

        ot_prob = linear_problem.LinearProblem(geom, g1, g2)
        start_time = time.time()
        solver = sinkhorn_lr.LRSinkhorn(rank=rank)
        end_time = time.time()
        solve_time = end_time - start_time
        ot_lr = solver(ot_prob)

        P = ot_lr.matrix
        
        primal_cost  = jnp.sum(C * P)
        l1_row_error = jnp.sum(jnp.abs(g1  - P.sum(axis=0)))
        l1_col_error = jnp.sum(jnp.abs(g2  - P.sum(axis=1)))
        l1_error     = jnp.sum(jnp.abs(1.0 - P.sum()))

        res = {
                "objective_cost": float(primal_cost),
                "lower_bound": None,
                "rank": rank,
                "simulation_seed": args.seed,
                "num_restart": 0,
                "algorithm": args.algorithm,
                "l1_row_marginal_error": l1_row_error,
                "l1_col_marginal_error": l1_col_error,
                "l1_total_error": l1_error,
                "runtime": solve_time
        }

        print(res)
        results.append(res)
    elif args.algorithm == "fullrank_round":
        geom = Geometry(cost_matrix=C, epsilon=0.001)

        ot_prob = linear_problem.LinearProblem(geom, g1, g2)
        start_time = time.time()
        solver = sinkhorn.Sinkhorn()
        end_time = time.time()
        solve_time = end_time - start_time
        ot_result = solver(ot_prob)

        P = ot_result.matrix
        
        for i in range(args.restarts):
            start_time = time.time()
            L, R = clrot.nonnegative_rounding(P, g1, g2, rank, seed=args.seed + i)
            end_time   = time.time()
            round_time = end_time - start_time
            P_rounded = L @ R
            primal_cost = jnp.sum(C * P_rounded)

            l1_row_error = jnp.sum(jnp.abs(g1  - P_rounded.sum(axis=0)))
            l1_col_error = jnp.sum(jnp.abs(g2  - P_rounded.sum(axis=1)))
            l1_error     = jnp.sum(jnp.abs(1.0 - P_rounded.sum()))

            res = {
                "objective_cost": float(primal_cost),
                "lower_bound": None,
                "rank": rank,
                "simulation_seed": args.seed,
                "num_restart": i,
                "algorithm": args.algorithm,
                "l1_row_marginal_error": l1_row_error,
                "l1_col_marginal_error": l1_col_error,
                "l1_total_error": l1_error,
                "runtime": solve_time + round_time
            }

            results.append(res)

    results = pd.DataFrame(results)
    results.to_csv(f"{args.output}", index=False)
            
