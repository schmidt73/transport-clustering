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

    objective = cp.Minimize(cp.sum(cp.multiply(C, P)) + (rho / 2.0) * cp.norm(P, "fro")**2)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver did not converge: status = {prob.status}")

    return P.value, prob.value

def simplex_projection(X: jnp.ndarray, t: jnp.ndarray):
    """
    Given a vector (x_1, \\ldots, x_n) and a scalar t, finds \\theta
    such that \sum_{i=1}^n[x_i - \\theta]_+ = t. 
    
    Rather than taking in a single vector, we take in a vector of 
    vectors X : m-by-n and an n-dimensional vector of scalars t,
    for which we solve the problem independently for each row of X.

    Based upon the algorithm of Duchi et al. 2008, "Efficient 
    Projections onto the \\ell_1 Ball for Learning in High 
    Dimensions".
    """
    X = jnp.sort(X, axis=1)[:, ::-1]
    X_cum = jnp.cumsum(X, axis=1)
    rhos  = X - (1.0 / jnp.arange(1, X_cum.shape[1] + 1)) * (X_cum.T - t).T 
    rho   = jnp.argmax((rhos > 0) * jnp.arange(0, X_cum.shape[1]), axis=1)
    theta = (1.0 / (rho + 1)) * (X_cum[jnp.arange(0, X_cum.shape[0]), rho] - t)
    return theta

@jax.jit
def nuclear_projection(A):
    """Projection onto nuclear norm ball."""
    U, s, V = jnp.linalg.svd(A, full_matrices=False)
    s = jax.lax.cond(
        jnp.sum(s) <= 1,
        lambda x: x,
        lambda x: jnp.maximum(x - simplex_projection(x[None,:], jnp.array([1.0]))[0], 0),
        s
    )
    return U.dot(jnp.diag(s).dot(V))

@jax.jit
def solve_euclidean_reg_ot(
    C: jnp.ndarray, 
    g1: jnp.ndarray, 
    g2: jnp.ndarray, 
    P1_alpha : jnp.ndarray,
    P2_beta : jnp.ndarray,
    rho: float = 100,
    iterations: int = 20
):
    alpha = P1_alpha
    beta  = P2_beta

    def body(_, state):
        alpha, beta = state
        alpha = simplex_projection(-(C + beta),  rho * g1)
        beta  = simplex_projection(-(C.T + alpha), rho * g2)
        return (alpha, beta)

    alpha, beta = jax.lax.fori_loop(0, iterations, body, (alpha, beta))
    P = jnp.maximum(-(C + alpha[:, None] + beta[None, :]), 0) / rho
    return P, alpha, beta

def solve_nuclear_ot(
    C: jnp.ndarray, 
    g1: jnp.ndarray, 
    g2: jnp.ndarray, 
    k: int, gamma: float, 
    max_iter: int = 100, 
    tolerance: float = 1e-4, 
    rho: float = 100,
    verbose: bool = False
):
    P1_alpha = jnp.zeros_like(g1)
    P1_beta  = jnp.zeros_like(g2)
    P2       = jnp.zeros_like(C)
    D        = jnp.zeros_like(C)

    iteration = 0
    while iteration < max_iter:
        start_time_euc_ot = time.time()
        P1, P1_alpha, P1_beta = solve_euclidean_reg_ot(C - rho * (P2 - D), g1, g2, P1_alpha, P1_beta, rho=rho)
        P1.block_until_ready()
        end_time_euc_ot = time.time()

        start_time_nuc_proj = time.time()
        P2 = (gamma * k) * nuclear_projection((P1 + D) / (gamma * k))
        P2.block_until_ready()
        end_time_nuc_proj = time.time()

        R  = P1 - P2
        D  = D + R

        if verbose:
            logger.info(f"Iteration {iteration}")
            logger.info(f"Objective: {jnp.sum(C * P1)}")
            logger.info(f"Residual Norm: {jnp.linalg.norm(R)}")
            logger.info(f"Time for Euclidean OT: {end_time_euc_ot - start_time_euc_ot}")
            logger.info(f"Time for Nuclear Projection: {end_time_nuc_proj - start_time_nuc_proj}")

        iteration += 1

    return P1, np.sum(C * P1)

def sinkhorn_rescaling(L, R, g1, g2, max_iter=1000, tol=1e-12):
    rescaling_rows = True
    for _ in range(max_iter):
        if rescaling_rows:
            row_sum = L @ R @ jnp.ones(R.shape[1])
            rescaling_matrix = jnp.diag(g1 / row_sum)
            L = rescaling_matrix @ L
            rescaling_rows = False
        else:
            col_sum = R.T @ L.T @ jnp.ones(L.shape[0])
            rescaling_matrix = jnp.diag(g2 / col_sum)
            R = R @ rescaling_matrix
            rescaling_rows = True

        norm1 = np.linalg.norm(L @ R @ jnp.ones(R.shape[1]) - g1)
        norm2 = np.linalg.norm(R.T @ L.T @ jnp.ones(L.shape[0]) - g2)
        if norm1 < tol and norm2 < tol:
            break
    return L, R

def nonnegative_rounding(P, g1, g2, k, seed=0):
    model = NMF(n_components=k, init='random', random_state=seed, max_iter=10000, solver='mu', beta_loss='frobenius')
    W = model.fit_transform(P)
    H = model.components_
    L_round, R_round = sinkhorn_rescaling(W, H, g1, g2)
    return L_round, R_round
