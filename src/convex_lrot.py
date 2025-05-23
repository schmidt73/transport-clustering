import time
import os
import jax

import jax.numpy as jnp
import cvxpy as cp
import numpy as np

from ott.geometry import geometry, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from loguru import logger
from sklearn.decomposition import NMF
import optax

def solve_linear_ot_cvxpy(
    C: np.ndarray, 
    g1: np.ndarray, 
    g2: np.ndarray, 
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

def kullback_leibler_divergence(P, Q):
    """
    Computes the Kullback-Leibler divergence between two positive matrices P and Q.
    """
    return jnp.sum(P * jnp.log(P / Q)) + jnp.sum(Q - P)

def initialize_factors(C, g1, g2, rank):
    """
    Uses a Scetbon-style initialization for the factors L and R.
    """
    rng = jax.random.PRNGKey(0)
    n, m = C.shape

    init_q = jnp.abs(jax.random.normal(rng, (n, rank)))
    init_q = g1[:, None] * (init_q / jnp.sum(init_q, axis=1, keepdims=True))

    init_r = jnp.abs(jax.random.normal(rng, (m, rank)))
    init_r = g2[:, None] * (init_r / jnp.sum(init_r, axis=1, keepdims=True))

    init_g = jnp.abs(jax.random.uniform(rng, (rank,))) + 1.0
    init_g =  init_g / jnp.sum(init_g)

    L = init_q @ jnp.diag(1 / jnp.sqrt(init_g))
    R = jnp.diag(1 / jnp.sqrt(init_g)) @ init_r.T
    return L, R

def alternating_mirror_descent_low_rank_ot(
    C: jnp.ndarray, 
    g1: jnp.ndarray, 
    g2: jnp.ndarray, 
    rank : int,
    rho: float = 0.01
):
    n, m = C.shape
    if g1.shape != (n,) or g2.shape != (m,):
        raise ValueError("Dimension mismatch between C, g1, and g2.")
    
    L, R = initialize_factors(C, g1, g2, rank)
    L, R = sinkhorn_rescaling(L, R, g1, g2)

    ones_n = jnp.ones(n).reshape(-1, 1)
    ones_m = jnp.ones(m).reshape(-1, 1)

    logger.info(f"Initial objective: {jnp.sum(C * (L @ R)) + rho * kullback_leibler_divergence(L, L)}")
    print(jnp.sum(jnp.abs((L @ R) @ ones_m - g1.reshape(-1, 1))))
    print(jnp.sum(jnp.abs((R.T @ L.T) @ ones_n - g2.reshape(-1, 1))))

    def compute_L(alpha, beta, L, R):
        """ 
        Computes dual optimal solution L given dual variables alpha and beta,
        where L and R are the current iterates.
        """
        L_res = L * jnp.exp((1 / rho) * (ones_n @ beta.T @ R.T + alpha @ ones_m.T @ R.T - C @ R.T))
        return L_res
    
    @jax.jit
    def compute_loss(params, args):
        alpha, beta = params
        L_prev, R_prev = args
        L = compute_L(alpha, beta, L_prev, R_prev)
        primal_cost = jnp.sum((C @ R_prev.T) * L) + rho * kullback_leibler_divergence(L, L_prev)
        lagrangian_penalty = alpha.T @ (L @ R_prev @ ones_m - g1.reshape(-1, 1)) + beta.T @ (R_prev.T @ L.T @ ones_n - g2.reshape(-1, 1))
        loss = (primal_cost - lagrangian_penalty[0,0])
        return loss
     
    value_and_grad_fn = jax.value_and_grad(compute_loss)
    
    optimizer = optax.chain(
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        optax.scale(0.001)
    )

    alpha = jnp.zeros(n).reshape(-1, 1)
    beta = jnp.zeros(m).reshape(-1, 1)
    init_params = (alpha, beta)
    params = (alpha, beta)
    opt_state = optimizer.init(init_params)

    for i in range(10000): 
        loss, grads = value_and_grad_fn(params, (L, R))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if i % 100 == 0:
            grad_norm = jnp.linalg.norm(grads[0]) + jnp.linalg.norm(grads[1])
            logger.info(f"Iteration {i}, Loss: {loss}, Grad Norm: {grad_norm}")
            if grad_norm < 1e-5:
                logger.info("Converged!")
                break
    
    # Extract optimized dual variables
    alpha, beta = params
    print(alpha)
    print(beta)
    L_new = compute_L(alpha, beta, L, R)
    L, R = sinkhorn_rescaling(L_new, R, g1, g2)
    logger.info(f"Final objective: {jnp.sum(C * (L_new @ R)) + rho * kullback_leibler_divergence(L_new, L)}")
    #print((L_new @ R))
    print(jnp.sum(jnp.abs((L_new @ R) @ ones_m - g1.reshape(-1, 1))))
    print(jnp.sum(jnp.abs((R.T @ L_new.T) @ ones_n - g2.reshape(-1, 1))))
        
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

@jax.jit
def sinkhorn_rescaling(L, R, g1, g2, max_iter=5, tol=1e-12):
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

        # norm1 = jnp.linalg.norm(L @ R @ jnp.ones(R.shape[1]) - g1)
        # norm2 = jnp.linalg.norm(R.T @ L.T @ jnp.ones(L.shape[0]) - g2)
        # if norm1 < tol and norm2 < tol:
        #     break
    return L, R

def sinkhorn_rescaling_P(P, g1, g2, max_iter=1000, tol=1e-6):
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

def nonnegative_rounding(P, g1, g2, k, seed=0):
    U, s, Vt = jnp.linalg.svd(P)
    print(s[:25])
    s = s.at[k:].set(0)
    print(s[:25])
    P_svd = U @ jnp.diag(s) @ Vt
    model = NMF(n_components=k, init='random', random_state=seed, max_iter=10000, solver='mu', beta_loss='frobenius')
    W = model.fit_transform(P)
    H = model.components_
    L_round, R_round = sinkhorn_rescaling(W, H, g1, g2)
    return L_round, R_round, P_svd
