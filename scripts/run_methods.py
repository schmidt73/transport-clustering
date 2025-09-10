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

import monge_rotate as mr
import FRLC.FRLC as frlc
import convex_lrot as clrot
from sklearn.cluster import KMeans

def visualize_transport_matrix(P, algorithm, primal_cost, rank, show=True):
    P_np = P if isinstance(P, np.ndarray) else np.array(P)
        
    plt.figure(figsize=(10, 8))
    plt.imshow(P_np, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Transport Probability')
    plt.title(f'Inferred Transport Plan: {algorithm.upper()} (rank={rank})')
    plt.xlabel(f'$\\langle C, P\\rangle_F = {primal_cost:.5f}$', fontsize=14)
    
    if show:
        plt.show()

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("x_points", help="X points.")
    parser.add_argument("y_points", help="Y points.")
    parser.add_argument("-r", "--rank", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-o", "--output", type=str, default="results")
    parser.add_argument("-a", "--algorithm", default="clrot", choices=["clrot", "mr", "frlc", "lot", "monge"])
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

    X = np.loadtxt(args.x_points)
    Y = np.loadtxt(args.y_points)

    C = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)**2 # squared Euclidean cost
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
        Q, R, objective_values = clrot.solve_lrot(C, rank)
        objective_values = objective_values[1000:]
        P = Q @ R.T
        visualize_transport_matrix(P, args.algorithm, jnp.sum(C * P) / batch_size1, rank, show=False)
        logger.info(f"Primal cost is {jnp.sum(C * P) / batch_size1}")
        fig, ax = plt.subplots()
        sns.scatterplot(x=list(range(objective_values.shape[0])), y=objective_values, 
                        ax=ax, color="orange", edgecolor=None, s=20)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective Value")
        fig.tight_layout()
        plt.savefig("objective_versus_iterations.png")
        plt.show()
    elif args.algorithm == "mr":
        C = jnp.array(C)
        Q, R, _, _ = mr.monge_rotation_kmeans(C, X, Y, rank)
        P = Q @ R.T
        visualize_transport_matrix(P, args.algorithm, jnp.sum(C * P) / batch_size1, rank, show=False)
        logger.info(f"Primal cost is {jnp.sum(C * P) / batch_size1}")
        plt.show()
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
            print(res)

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

    results = pd.DataFrame(results)
    results.to_csv(f"{args.output}", index=False)
            
