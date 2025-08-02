import sys
import math
import random
import torch
import torchdyn
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import time

import pandas as pd
import argparse as ap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ott.initializers.linear.initializers_lr import RandomInitializer, KMeansInitializer, Rank2Initializer
from ott.geometry.geometry import Geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from loguru import logger

sys.path.append("../src")

import FRLC.FRLC as frlc
import convex_lrot as clrot

def visualize_transport_matrix(P, algorithm, primal_cost, rank, show=True):
    P_np = P if isinstance(P, np.ndarray) else np.array(P)
        
    plt.figure(figsize=(10, 8))
    plt.imshow(P_np, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Transport Probability')
    plt.title(f'Inferred Transport Plan: {algorithm.upper()} (rank={rank})')
    plt.xlabel(f'$\\langle C, P\\rangle_F = {primal_cost:.3f}$', fontsize=14)
    
    if show:
        plt.show()

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("cost_matrix", help="Cost matrix.")
    parser.add_argument("-r", "--rank", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-o", "--output", type=str, default="results")
    parser.add_argument("-a", "--algorithm", default="clrot", choices=["clrot", "amdlot", "frlc", "lot", "fullrankround", "monge"])
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--visualize", action="store_true", help="Visualize transport matrix.")
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
        L, R = clrot.auglag_convex_monge_sep(C, rank_1=rank, rank_2=rank)
        # L, R = clrot.alternating_mirror_descent_low_rank_ot(
        #     C, jnp.array(g1), jnp.array(g2), rank_1=3 * rank, rho=1.0, max_iter=25, gamma=gamma
        # )
        P = L @ R
        end_time   = time.time()
        solve_time = end_time - start_time
        
        if args.visualize:
            visualize_transport_matrix(P, "CLROT Raw", jnp.sum(C * P), rank, show=False)

            singular_values = jnp.linalg.norm(L, axis=0) * jnp.linalg.norm(R, axis=1)
            singular_values = jnp.sort(singular_values)[::-1]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=range(len(singular_values)), y=singular_values, marker='o', ax=ax)
            ax.axvline(x=rank, color='black', linestyle='--', label=f'Rank = {rank}')
            ax.set_yscale('log')
            ax.set_xlabel('Index')
            ax.set_ylabel('Singular Value')
            
        for i in range(args.restarts):
            start_time = time.time()
            L, R = clrot.nonnegative_rounding(L, R, g1, g2, rank, seed=args.seed + i)
            L, R = jnp.clip(L, 1e-8), jnp.clip(R, 1e-8)  # ensure strictly positive entries
            end_time   = time.time()
            round_time = end_time - start_time
            L, R = clrot.sinkhorn_rescaling(L, R, g1, g2, max_iter=3000, tol=1e-5) # round all solutions to be 1e-5 feasible

            if args.visualize:
                visualize_transport_matrix(L @ R, "CLROT Rounded", jnp.sum(C * (L @ R)), rank, show=False)

            L, R = clrot.alternating_mirror_descent_low_rank_ot(
                C, jnp.array(g1), jnp.array(g2), rank_1=rank, rho=10.0, L_init=L, R_init=R, max_iter=40
            )

            P_rounded = L @ R
            primal_cost = jnp.sum(C * P_rounded)

            if args.visualize:
                visualize_transport_matrix(P_rounded, "CLROT + AMDLOT", primal_cost, rank)

            l1_row_error = jnp.sum(jnp.abs(g1  - (L @ R).sum(axis=0)))
            l1_col_error = jnp.sum(jnp.abs(g2  - (L @ R).sum(axis=1)))
            l1_error     = jnp.sum(jnp.abs(1.0 - (L @ R).sum()))

            logger.info(f"CLROT objective: {primal_cost}")
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
    elif args.algorithm == "amdlot":
        gamma = (1.0 / min(batch_size1, batch_size2))
        C = jnp.array(C)

        for i in range(args.restarts):
            start_time = time.time()
            L, R = clrot.alternating_mirror_descent_low_rank_ot(
                C, jnp.array(g1), jnp.array(g2), args.rank, rho=1.0, seed=args.seed + i, max_iter=20
            ) 

            end_time   = time.time()
            solve_time = end_time - start_time
            round_time = end_time - start_time

            P = L @ R
            P = clrot.sinkhorn_rescaling_P(P, g1, g2, max_iter=3000, tol=1e-5) # round all solutions to be 1e-5 feasible

            primal_cost = jnp.sum(C * P)

            if args.visualize:
                visualize_transport_matrix(P, args.algorithm, primal_cost, rank)   

            l1_row_error = jnp.sum(jnp.abs(g1  - P.sum(axis=0)))
            l1_col_error = jnp.sum(jnp.abs(g2  - P.sum(axis=1)))
            l1_error     = jnp.sum(jnp.abs(1.0 - P.sum()))

            logger.info(f"ADMLOT objective: {primal_cost}")
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
    elif args.algorithm == "frlc":
        C = torch.from_numpy(C).to(device)
        for i in range(args.restarts):
            start_time = time.time()
            P, errs = frlc.FRLC_opt(
                C, device=device, r=rank, max_iter=20, returnFull=True, gamma=70, max_inneriters_balanced=500, max_inneriters_relaxed=500
            )
            end_time   = time.time()
            solve_time = end_time - start_time

            P = P.cpu().numpy()
            P = clrot.sinkhorn_rescaling_P(P, g1, g2, max_iter=3000, tol=1e-5) # round all solutions to be 1e-5 feasible
            np.savetxt("P.txt", P, fmt="%.8f")

            primal_cost = np.sum(C.cpu().numpy() * P)

            if args.visualize:
                visualize_transport_matrix(P, args.algorithm, primal_cost, rank)   

            l1_row_error = np.sum(np.abs(g1  - P.sum(axis=0)))
            l1_col_error = np.sum(np.abs(g2  - P.sum(axis=1)))
            l1_error     = np.sum(np.abs(1.0 - P.sum()))

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
        solver = sinkhorn_lr.LRSinkhorn(rank=rank, initializer=RandomInitializer(rank))
        end_time = time.time()
        solve_time = end_time - start_time
        ot_lr = solver(ot_prob)

        P = ot_lr.matrix
        P = clrot.sinkhorn_rescaling_P(P, g1, g2, max_iter=3000, tol=1e-5) # round all solutions to be 1e-5 feasible
        
        primal_cost  = jnp.sum(C * P)

        if args.visualize:
            visualize_transport_matrix(P, args.algorithm, primal_cost, rank)   
       
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
    elif args.algorithm == "monge":
        geom = Geometry(cost_matrix=C, epsilon=0.0001)

        ot_prob = linear_problem.LinearProblem(geom, g1, g2)
        start_time = time.time()
        solver = sinkhorn.Sinkhorn()
        end_time = time.time()
        solve_time = end_time - start_time
        ot_result = solver(ot_prob)

        P = ot_result.matrix
        np.savetxt("P.txt", P, fmt="%.8f")

        primal_cost = np.sum(C * P)

        if args.visualize:
            visualize_transport_matrix(P, args.algorithm, primal_cost, rank)   

    elif args.algorithm == "fullrankround":
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
            L, R, _ = clrot.nonnegative_rounding_P(P, g1, g2, rank, seed=args.seed + i)
            end_time   = time.time()
            round_time = end_time - start_time
            P_rounded = L @ R
            P_rounded = clrot.sinkhorn_rescaling_P(P_rounded, g1, g2, max_iter=3000, tol=1e-5) # round all solutions to be 1e-5 feasible
            primal_cost = jnp.sum(C * P_rounded)

            if args.visualize:
                visualize_transport_matrix(P, args.algorithm, primal_cost)   

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
            
