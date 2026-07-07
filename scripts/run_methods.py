import sys
import math
import random
import torch
import torchdyn
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import time

import json
import pandas as pd
import argparse as ap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ott.initializers.linear.initializers_lr import RandomInitializer
from ott.geometry.pointcloud import PointCloud
from ott.geometry.geometry import Geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn_lr
from loguru import logger

sys.path.append("../src")

import tc as mr
import FRLC.FRLC as frlc
import LatentOT as latentot

def visualize_transport_matrix(P, algorithm, primal_cost, rank, show=True):
    P_np = P if isinstance(P, np.ndarray) else np.array(P)
        
    plt.figure(figsize=(10, 8))
    plt.imshow(P_np, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Transport Probability')
    plt.title(f'Inferred Transport Plan: {algorithm.upper()} (rank={rank})')
    plt.xlabel(f'$\\langle C, P\\rangle_F = {primal_cost:.5f}$', fontsize=14)
    
    if show:
        plt.show()

def compute_sqeuc_cost_matrix(X, Y, *, dtype=jnp.float32, normalize=True):
    x2 = jnp.sum(X * X, axis=1)
    y2 = jnp.sum(Y * Y, axis=1)
    G  = X @ Y.T
    D  = x2[:, None] + y2[None, :] - 2.0 * G
    D  = jnp.maximum(D, 0)

    if normalize:
        D = D / jnp.maximum(jnp.max(D), jnp.finfo(D.dtype).tiny)
    return D

def sinkhorn_rescaling(P, g1, g2, max_iter=100, tol=1e-4):
    """
    Perform Sinkhorn rescaling on P to make it 
    (approximately) satisfy marginals g1, g2.
    """
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

def run_transport_cluster(
        seed : int, 
        g1: jnp.ndarray, 
        g2: jnp.ndarray, 
        X: jnp.ndarray = None, 
        Y: jnp.ndarray = None, 
        C: jnp.ndarray = None
    ):
    """Run the Transport Clustering algorithm.
    """
    if C is None and (X is None or Y is None):
        raise ValueError("Must provide either cost matrix C or both point clouds X and Y.")
    
    if C is None:
        C = compute_sqeuc_cost_matrix(X, Y, dtype=jnp.float32, normalize=True)

    start_time = time.time()
    C = jnp.array(C)
    Q, g, R = mr.transport_cluster(C, rank, random_state=seed, bm_init=False, debug=False)
    P = Q @ jnp.diag(1 / g) @ R.T
    end_time = time.time()
    solve_time = end_time - start_time
    P = sinkhorn_rescaling(P, g1, g2, max_iter=3000, tol=1e-5)  # round all solutions to be 1e-5 feasible
    
    primal_cost = jnp.sum(C * P)
    logger.info(f"Monge Rotation objective: {primal_cost}")
    
    l1_row_error = jnp.sum(jnp.abs(g1 - P.sum(axis=0)))
    l1_col_error = jnp.sum(jnp.abs(g2 - P.sum(axis=1)))
    l1_error = jnp.sum(jnp.abs(1.0 - P.sum()))
    
    result = {
        "objective_cost": float(primal_cost),
        "lower_bound": None,
        "rank": rank,
        "simulation_seed": args.seed,
        "algorithm": args.algorithm,
        "l1_row_marginal_error": float(l1_row_error),
        "l1_col_marginal_error": float(l1_col_error),
        "l1_total_error": float(l1_error),
        "runtime": solve_time
    }
    
    return Q, g, R, result

def run_frlc(
    seed: int,
    g1: jnp.ndarray, 
    g2: jnp.ndarray, 
    X: jnp.ndarray = None, 
    Y: jnp.ndarray = None, 
    C: jnp.ndarray = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32
    ):
    """Run the FRLC algorithm."""
    if C is None and (X is None or Y is None):
        raise ValueError("Must provide either cost matrix C or both point clouds X and Y.")
    
    if C is None:
        C = compute_sqeuc_cost_matrix(X, Y, dtype=jnp.float32, normalize=True)

    C = torch.tensor(np.array(C), device=device, dtype=dtype)
    start_time = time.time()
    Q, R, g, errs = frlc.FRLC_opt(
        C, device=device, r=rank, max_iter=20, returnFull=False, diagonalize_return=True, gamma=70, 
        max_inneriters_balanced=500, max_inneriters_relaxed=500
    )
    Q = Q.cpu().numpy()
    R = R.cpu().numpy()
    g = np.diagonal(g.cpu().numpy())
    end_time   = time.time()
    solve_time = end_time - start_time
    P = jnp.array(Q @ jnp.diag(1.0 / g) @ R.T)
    P = sinkhorn_rescaling(P, g1, g2, max_iter=3000, tol=1e-5) # round all solutions to be 1e-5 feasible

    primal_cost = jnp.sum(C.cpu().numpy() * P)

    l1_row_error = jnp.sum(jnp.abs(g1  - P.sum(axis=0)))
    l1_col_error = jnp.sum(jnp.abs(g2  - P.sum(axis=1)))
    l1_error     = jnp.sum(jnp.abs(1.0 - P.sum()))

    logger.info(f"FRLC objective: {primal_cost}")

    res = {
        "objective_cost": float(primal_cost),
        "lower_bound": None,
        "rank": rank,
        "simulation_seed": args.seed,
        "algorithm": args.algorithm,
        "l1_row_marginal_error": float(l1_row_error),
        "l1_col_marginal_error": float(l1_col_error),
        "l1_total_error": float(l1_error),
        "runtime": solve_time
    }

    return Q, g, R, res

def run_lot(
        seed: int,
        g1: jnp.ndarray, 
        g2: jnp.ndarray, 
        X: jnp.ndarray = None, 
        Y: jnp.ndarray = None, 
        C: jnp.ndarray = None
    ):
    """Run the LOT algorithm."""
    if C is None and (X is None or Y is None):
        raise ValueError("Must provide either cost matrix C or both point clouds X and Y.")
    
    if C is not None:
        geom = Geometry(cost_matrix=C, epsilon=0.001, scale_cost="max_cost")
    else:
        geom = PointCloud(x=X, y=Y, epsilon=0.001, scale_cost="max_cost")

    rng = jax.random.PRNGKey(seed if seed is not None else 0)
    ot_prob = linear_problem.LinearProblem(geom, g1, g2)
    start_time = time.time()
    solver = sinkhorn_lr.LRSinkhorn(rank=rank, initializer=RandomInitializer(rank))
    end_time = time.time()
    solve_time = end_time - start_time
    ot_lr = solver(ot_prob, rng=rng)

    P = ot_lr.matrix
    P = sinkhorn_rescaling(P, g1, g2, max_iter=3000, tol=1e-5) # round all solutions to be 1e-5 feasible
    
    primal_cost = ot_lr.primal_cost
    
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
        "l1_row_marginal_error": float(l1_row_error),
        "l1_col_marginal_error": float(l1_col_error),
        "l1_total_error": float(l1_error),
        "runtime": solve_time
    }

    return ot_lr.q, ot_lr.g, ot_lr.r, res

def run_latent_ot_lin(
    seed: int,
    g1: jnp.ndarray, 
    g2: jnp.ndarray, 
    X: jnp.ndarray, 
    Y: jnp.ndarray 
    ):
    """Run the Lin et al. 2021 algorithm."""

    rng = jax.random.PRNGKey(seed if seed is not None else 0)
    start_time = time.time()
    lot = latentot.LOT(n_source_anchors=rank, n_target_anchors=rank)
    lot.fit(X, Y)
    transported_features_lot = lot.transport(X, Y)
    end_time = time.time()
    solve_time = end_time - start_time

    Q, T, R = lot.Px_, lot.Pz_, lot.Py_
    P = Q @ jnp.diag(1 / Q.sum(axis=0)) @ T @ jnp.diag(1 / R.sum(axis=1)) @ R
    P = sinkhorn_rescaling(P, g1, g2, max_iter=3000, tol=1e-5) # round all solutions to be 1e-5 feasible

    primal_cost = compute_sqeuc_cost_matrix(X, Y) * P
    primal_cost = jnp.sum(primal_cost)
    logger.info(f"Lin et al. objective: {primal_cost}")
    
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
        "l1_row_marginal_error": float(l1_row_error),
        "l1_col_marginal_error": float(l1_col_error),
        "l1_total_error": float(l1_error),
        "runtime": solve_time
    }

    return Q, Q.sum(axis=0), (T @ jnp.diag(1 / R.sum(axis=1)) @ R).T, res

def parse_args():
    parser = ap.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--cost_matrix", help="Path to pre-computed cost matrix.")
    input_group.add_argument("--points", nargs=2, metavar=('X_POINTS', 'Y_POINTS'), 
                          help="Paths to X and Y point sets.")
    parser.add_argument("-r", "--rank", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-o", "--output", type=str, default="algorithm")
    parser.add_argument("-a", "--algorithm", default="clrot", choices=["mr", "frlc", "lot", "lin"])
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--visualize", action="store_true", help="Visualize transport matrix.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    logger.info(f"JAX Devices: {jax.devices()}, default backend: {jax.default_backend()}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    C, X, Y = None, None, None
    if args.points:
        args.x_points, args.y_points = args.points
        X, Y = np.loadtxt(args.x_points), np.loadtxt(args.y_points)
        n, m = X.shape[0], Y.shape[0]
    if args.cost_matrix:
        args.points = None
        C = np.loadtxt(args.cost_matrix)
        C = jnp.array(C) / jnp.max(C)
        n, m = C.shape

    g1 = np.ones((n)) / n
    g2 = np.ones((m)) / m

    rank = args.rank

    result = None
    if args.algorithm == "mr":
        Q, g, R, result = run_transport_cluster(args.seed, g1, g2, X, Y, C)
    elif args.algorithm == "frlc": 
        torch.manual_seed(args.seed)
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype  = torch.float64
        Q, g, R, result = run_frlc(args.seed, g1, g2, X, Y, C, device=torch_device, dtype=torch_dtype)
    elif args.algorithm == "lot":
        Q, g, R, result = run_lot(args.seed, g1, g2, X, Y, C)
    elif args.algorithm == "lin":
        Q, g, R, result = run_latent_ot_lin(args.seed, g1, g2, X, Y)

    P = jnp.array(Q @ jnp.diag(1.0 / g) @ R.T)
    if args.visualize:
        visualize_transport_matrix(P, args.algorithm, result["objective_cost"], rank)
    
    if args.output:
        with open(args.output + "_summary.json", "w") as f:
            json.dump(result, f)
        logger.info(f"Saved summary of results to {args.output}_summary.json")
        np.savetxt(args.output + "_Q.txt", Q)
        np.savetxt(args.output + "_g.txt", g)
        np.savetxt(args.output + "_R.txt", R)
        logger.info(f"Saved factors Q, g, R to {args.output}_Q.txt, {args.output}_g.txt, {args.output}_R.txt")