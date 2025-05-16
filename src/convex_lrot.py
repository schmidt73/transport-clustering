import time
import os
import jax

import jax.numpy as jnp
import cvxpy as cp
import numpy as np
import gurobipy as gp

from ott.geometry import geometry, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from loguru import logger
from sklearn.decomposition import NMF
from jax.experimental.sparse.linalg import lobpcg_standard as lobpcg

def simplex_projection(s):
    """Projection onto the unit simplex."""
    if np.sum(s) <=1 and np.alltrue(s >= 0):
        return s
    u = np.sort(s)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(s-theta, 0)

def nuclear_projection(A):
    """Projection onto nuclear norm ball."""
    U, s, V = np.linalg.svd(A, full_matrices=False)
    s = simplex_projection(s)
    print(s)
    return U.dot(np.diag(s).dot(V))

def solve_linear_ot_cvxpy(
    C: np.ndarray, 
    g1: np.ndarray, 
    g2: np.ndarray, 
    solver: str = "GUROBI"
):
    m, n = C.shape
    if g1.shape != (m,) or g2.shape != (n,):
        raise ValueError("Dimension mismatch between C, g1, and g2.")

    P = cp.Variable((m, n), nonneg=True)

    ones_n = np.ones(n)
    ones_m = np.ones(m)

    constraints = [
        P @ ones_n == g1,
        P.T @ ones_m == g2
    ]

    objective = cp.Minimize(cp.sum(cp.multiply(C, P)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=True)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver did not converge: status = {prob.status}")

    return P.value, prob.value

def solve_euclidean_reg_ot_cvxpy(
    C: np.ndarray, 
    g1: np.ndarray, 
    g2: np.ndarray, 
    P_reg: np.ndarray, 
    rho: float = 0.1,
    solver: str = "MOSEK"
):
    m, n = C.shape
    if g1.shape != (m,) or g2.shape != (n,):
        raise ValueError("Dimension mismatch between C, g1, and g2.")

    P = cp.Variable((m, n), nonneg=True)

    ones_n = np.ones(n)
    ones_m = np.ones(m)

    constraints = [
        P @ ones_n == g1,
        P.T @ ones_m == g2
    ]

    objective = cp.Minimize(cp.sum(cp.multiply(C, P)) + rho * cp.norm(P - P_reg, "fro")**2)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver did not converge: status = {prob.status}")

    return P.value, prob.value

def solve_nuclear_ot(
    C: np.ndarray, 
    g1: np.ndarray, 
    g2: np.ndarray, 
    k: int, gamma: float, 
    max_iter: int = 100, 
    tolerance: float = 1e-4, 
    rho: float = 100,
    verbose: bool = False
):
    P2 = np.zeros_like(C)
    D  = np.zeros_like(C)

    iteration = 0
    while iteration < max_iter:
        start_time_euc_ot = time.time()
        P1, objective = solve_euclidean_reg_ot_cvxpy(C, g1, g2, P2 - D, rho=rho) # can replace with Sinkhorn type solver
        end_time_euc_ot = time.time()

        start_time_nuc_proj = time.time()
        P2 = (gamma * k) * nuclear_projection((P1 + D) / (gamma * k))
        end_time_nuc_proj = time.time()

        R  = P1 - P2
        D  = D + R

        if verbose:
            logger.info(f"Iteration {iteration}")
            logger.info(f"Objective: {np.sum(C * P1)}")
            logger.info(f"Residual Norm: {np.linalg.norm(R)}")
            logger.info(f"Time for Euclidean OT: {end_time_euc_ot - start_time_euc_ot}")
            logger.info(f"Time for Nuclear Projection: {end_time_nuc_proj - start_time_nuc_proj}")

        iteration += 1
        # if np.linalg.norm(R, "fro") < tolerance and iteration > 1:
        #    break

    return P1, np.sum(C * P1)

def entropy_regularized_nuclear_projection(
    C,
    P1: jnp.ndarray,
    P2: jnp.ndarray,
    D: jnp.ndarray,
    nuclear_norm_bound : float,
    iterations: int = 1000,
):
    # MAJOR ISSUE: doesn't preserve feasability
    def loss(P2, P1, D):
        return jnp.sum(D * P2) + jnp.sum(jax.scipy.special.rel_entr(P1, P2))
    loss_grad = jax.value_and_grad(loss)
    # use franke wolfe or nuclear norm ball
    for i in range(iterations):
        obj, G = loss_grad(P2, P1, D)
        step_size = 1e-2
        #print(P1)
        print(f"Q-step Objective Cost: {obj}")
        # G = D - jnp.where(P2 != 0, P1 / P2, 0)  # gradient of the objective with safeguard for division by zero
        U, S, V = jnp.linalg.svd(G)             # TODO: replace with lobpcg to find only 1 singular vector
        #print(G)
        direction = -nuclear_norm_bound * U[:, 0] @ V[0, :]
        P2 = (1 - step_size) * P2 + step_size * direction
        print(f"Linear Objective Cost: {(C * P2).sum()}")
        #1/ 0
    return P2

#@jax.jit
def sink_helper(
    C, g1, g2, epsilon, min_iterations, max_iterations
):
    geom = geometry.Geometry(cost_matrix=C, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom, g1, g2)
    solver = sinkhorn.Sinkhorn(min_iterations=min_iterations, max_iterations=max_iterations)
    return solver(prob).matrix

def solve_nuclear_ot_sinkhorn(
    C: jnp.ndarray,
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    k: jnp.int32, 
    gamma: jnp.float32, 
    max_iter: jnp.int32 = 100, 
    tolerance: jnp.float32 = 1e-4, 
    epsilon: jnp.float32 = 1e-3,
    verbose: bool = False
):
    P2, D = jnp.ones_like(C), jnp.zeros_like(C)
    iteration = 0
    while iteration < max_iter:
        start_time_entropy_ot = time.time()
        P1 = sink_helper(C - D - jnp.log(P2), g1, g2, epsilon, min_iterations=0, max_iterations=500)
        end_time_entropy_ot = time.time()

        if verbose:
            logger.info(f"Time for Entropy Regularized OT: {end_time_entropy_ot - start_time_entropy_ot}")
            logger.info(f"Linear Objective (P1): {jnp.sum(C * P1)}")

        start_time_nuc_proj = time.time()
        P2 = 1 / (C.shape[0] * C.shape[1]) * jnp.ones_like(C)
        P2 = entropy_regularized_nuclear_projection(C, P1, P2, D, gamma * k, iterations=1000)
        end_time_nuc_proj = time.time()

        if verbose:
            logger.info(f"Time for Nuclear Projection: {end_time_nuc_proj - start_time_nuc_proj}")
            logger.info(f"Linear Objective (P2): {jnp.sum(C * P2)}")

        R  = P1 - P2
        D  = D + R

        iteration += 1

def sinkhorn_rescaling(L, R, g1, g2, max_iter=1000, tol=1e-12):
    rescaling_rows = True
    singular_values = []
    for _ in range(max_iter):
        if rescaling_rows:
            row_sum = L @ R @ np.ones(R.shape[1])
            rescaling_matrix = np.diag(g1 / row_sum)
            L = rescaling_matrix @ L
            rescaling_rows = False
        else:
            col_sum = R.T @ L.T @ np.ones(L.shape[0])
            rescaling_matrix = np.diag(g2 / col_sum)
            R = R @ rescaling_matrix
            rescaling_rows = True

        # for understanding
        svs = np.linalg.svd(L @ R, compute_uv=False)
        singular_values.append(svs)

        norm1 = np.linalg.norm(L @ R @ np.ones(R.shape[1]) - g1)
        norm2 = np.linalg.norm(R.T @ L.T @ np.ones(L.shape[0]) - g2)
        if norm1 < tol and norm2 < tol:
            break
    return L, R, singular_values

def nonnegative_rounding(P, g1, g2, k, seed=0):
    model = NMF(n_components=k, init='random', random_state=seed, max_iter=10000, solver='mu', beta_loss='frobenius')
    W = model.fit_transform(P)
    H = model.components_
    L_round, R_round, singular_values = sinkhorn_rescaling(W, H, g1, g2)
    return L_round, R_round, np.array(singular_values)
