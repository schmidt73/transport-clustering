import sys
import math
import random
import torch
import torchdyn
import jax

jax.config.update("jax_enable_x64", True)
print(jax.devices())
print(jax.default_backend())

import jax.numpy as jnp
import time

import pandas as pd
import argparse as ap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ott.initializers.linear.initializers_lr import RandomInitializer, KMeansInitializer, Rank2Initializer
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn_lr
from loguru import logger

sys.path.append("../src")

import monge_rotate as mr
import FRLC.FRLC as frlc

def sinkhorn_rescaling(P, g1, g2, max_iter=100, tol=1e-4):
    rescaling_rows = True
    for _ in range(max_iter):
        if rescaling_rows:
            row_sum = P @ jnp.ones(P.shape[1])
            rescaling_matrix = jnp.diag(g1 / row_sum)
            P = rescaling_matrix @ P
            rescaling_rows = False
        else:
            col_sum = P.T @ jnp.ones(P.shape[0])
            rescaling_matrix = jnp.diag(g2 / col_sum)
            P = P @ rescaling_matrix
            rescaling_rows = True

        norm1 = jnp.sum(jnp.abs(P @ jnp.ones(P.shape[1]) - g1))
        norm2 = jnp.sum(jnp.abs(P.T @ jnp.ones(P.shape[0]) - g2))
        if norm1 < tol and norm2 < tol:
            break
    return P

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
    parser.add_argument("-a", "--algorithm", default="clrot", choices=["mr", "frlc", "lot"])
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--visualize", action="store_true", help="Visualize transport matrix.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float64

    X = np.loadtxt(args.x_points)
    Y = np.loadtxt(args.y_points)

    # C = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)**2
    # C = C / C.max()
    batch_size1 = X.shape[0]
    batch_size2 = Y.shape[0]

    g1 = np.ones((batch_size1)) / batch_size1
    g2 = np.ones((batch_size2)) / batch_size2

    rank = args.rank

    results = []
    if args.algorithm == "mr":
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
            P = sinkhorn_rescaling(P, g1, g2, max_iter=3000, tol=1e-5) # round all solutions to be 1e-5 feasible
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
        geom = PointCloud(x=X, y=Y, epsilon=0.001, scale_cost="max_cost")

        ot_prob = linear_problem.LinearProblem(geom, g1, g2)
        start_time = time.time()
        solver = sinkhorn_lr.LRSinkhorn(rank=rank, initializer=RandomInitializer(rank))
        end_time = time.time()
        solve_time = end_time - start_time
        ot_lr = solver(ot_prob)

        P = ot_lr.matrix
        P = sinkhorn_rescaling(P, g1, g2, max_iter=3000, tol=1e-5) # round all solutions to be 1e-5 feasible
        
        primal_cost = ot_lr.primal_cost

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

    results = pd.DataFrame(results)
    results.to_csv(f"{args.output}", index=False)
            
