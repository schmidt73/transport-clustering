import time
import os
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import cvxpy as cp
import numpy as np

from loguru import logger
from sklearn.decomposition import NMF
import optax
import enum

class ProximalDescentStep(enum.Enum):
    L_STEP = "L_STEP"
    R_STEP = "R_STEP"

@jax.jit
def row_constrained_projection(
    X: jnp.ndarray, 
    w: jnp.ndarray, 
    min_iter: int=5, 
    max_iter: int=100, 
    convergence_threshold: float=1e-5, 
    tol: float=1e-9
):
    """
    Solves the problem:
        min_{Y} 1/2||X - Y||_F^2
        s.t. \sum_{j=1}^m w_i||y_i||_2 \leq 1
    where X is a matrix of shape (m, n), w is a vector of shape (m,),
    and y_i is the i-th row of Y.
    """

    X_norms = jnp.linalg.norm(X, axis=1)

    def compute_loss(gamma):
        l1 = w * gamma * X_norms - 0.5 * (w * gamma)**2
        l2 = 0.5 * X_norms**2
        return -(jnp.sum(jnp.where(w * gamma <= X_norms, l1, l2)) - gamma)
    
    value_and_grad_fn = jax.value_and_grad(compute_loss)
    
    gamma = jnp.array(1.0)
    optimizer = optax.lbfgs()
    opt_state = optimizer.init(gamma)
    
    def cond_fn(carry):
        i, gamma, opt_state, loss_value, grads, converged = carry
        return (i < max_iter) & (~converged)
    
    def body_fn(carry):
        i, gamma, opt_state, loss_value, grads, converged = carry
        loss_value, grads = value_and_grad_fn(gamma)
        updates, new_opt_state = optimizer.update(grads, opt_state, gamma, value=loss_value, grad=grads, value_fn=compute_loss)
        new_gamma = optax.apply_updates(gamma, updates)
        new_gamma = jnp.maximum(new_gamma, tol) # ensure positivity of gamma
        has_converged = (i > min_iter) & (jnp.abs(grads) < convergence_threshold)
        return i + 1, new_gamma, new_opt_state, loss_value, grads, has_converged
    
    init_loss, init_grads = value_and_grad_fn(gamma)
    init_state = (0, gamma, opt_state, init_loss, init_grads, jnp.array(False))
    
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    
    _, gamma, _, _, _, _ = final_state

    optimal_gamma = gamma
    Y = jnp.where(
        (w[:, None] * optimal_gamma) < X_norms[:, None],
        X * (1 - (w[:, None] * optimal_gamma) / X_norms[:, None]),
        jnp.zeros_like(X)
    )
    
    return Y

def initialize_factors(C: jnp.ndarray, g1: jnp.ndarray, g2: jnp.ndarray, rank: int, seed: int = 0):
    """
    Uses a Scetbon-style random initialization for the factors L and R.
    """
    rng = jax.random.PRNGKey(seed)
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

@jax.jit
def alternating_proximal_descent_compute_L(C, rho, L, R, alpha, beta):
    """ 
    Computes dual optimal solution L^* given dual variables alpha and beta,
    where L and R are the current iterates.
    """
    n, m = L.shape[0], R.shape[1]
    ones_n = jnp.ones(n).reshape(-1, 1)
    ones_m = jnp.ones(m).reshape(-1, 1)
    L_res = (1 / rho) * jnp.maximum((ones_n @ beta.T + alpha @ ones_m.T) @ R.T - C, 0.0)
    return L_res

@jax.jit
def alternating_proximal_descent_compute_R(C, rho, L, R, alpha, beta):
    """ 
    Computes dual optimal solution R^* given dual variables alpha and beta,
    where L and R are the current iterates.
    """
    n, m = L.shape[0], R.shape[1]
    ones_n = jnp.ones(n).reshape(-1, 1)
    ones_m = jnp.ones(m).reshape(-1, 1)
    R_res = (1 / rho) * jnp.maximum(L.T @ (alpha @ ones_m.T + ones_n @ beta.T) - C, 0.0)
    return R_res

def alternating_proximal_descent_compute_loss(step_type, params, args):
    """
    Computes the dual loss for a given set of dual variables (alpha, beta) where
    the current iterates are L and R.
    """
    alpha, beta = params
    L_prev, R_prev, C, g1, g2, rho = args
    n, m = L_prev.shape[0], R_prev.shape[1]
    ones_n = jnp.ones(n).reshape(-1, 1)
    ones_m = jnp.ones(m).reshape(-1, 1)
    
    if step_type == ProximalDescentStep.L_STEP:
        L = alternating_proximal_descent_compute_L(C, rho, L_prev, R_prev, alpha, beta)
        R = R_prev
        primal_cost = jnp.sum(C * L) + (rho / 2) * jnp.sum(L * L)
    else:
        R = alternating_proximal_descent_compute_R(C, rho, L_prev, R_prev, alpha, beta)
        L = L_prev
        primal_cost = jnp.sum(C * R) + (rho / 2) * jnp.sum(R * R)

    lagrangian_penalty_1 = alpha.T @ (L @ R @ ones_m - g1.reshape(-1, 1))
    lagrangian_penalty_2 = beta.T @ (R.T @ L.T @ ones_n - g2.reshape(-1, 1))
    loss = (primal_cost - (lagrangian_penalty_1 + lagrangian_penalty_2)[0,0])
    return -loss

alternating_pd_compute_grad = jax.value_and_grad(
    alternating_proximal_descent_compute_loss,
    argnums=1
)

alternating_pd_linesearch = optax.scale_by_backtracking_linesearch(max_backtracking_steps=20)
alternating_pd_optimizer  = optax.chain(optax.lbfgs(), alternating_pd_linesearch)

def alternating_proximal_descent_single_step(step_type, params, opt_state, args):
    value_fn = lambda p: alternating_proximal_descent_compute_loss(step_type, p, args)
    loss, grads = alternating_pd_compute_grad(step_type, params, args)
    updates, opt_state = alternating_pd_optimizer.update(
        grads, opt_state, params, 
        value=loss, grad=grads, 
        value_fn=value_fn,
        args=args
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state, grads, loss

def alternating_proximal_descent_step(
    C: jnp.ndarray, 
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    rho: float, 
    L_fixed: jnp.ndarray, 
    R_fixed: jnp.ndarray, 
    step_type: ProximalDescentStep,
    max_iter: int = 100,
    convergence_threshold: float = 1e-3
):
    """
    Solves one of the alternating proximal descent steps:
            min_{L : LR \in \Pi_{a,b}} <C, L>_F + (rho / 2) ||L||_F^2
        or  min_{R : LR \in \Pi_{a,b}} <C, R>_F + (rho / 2) ||R||_F^2,
    by solving the unconstrained dual problem for the dual variables 
    alpha and beta.
    """
    n, m = L_fixed.shape[0], R_fixed.shape[1]
    ones_n = jnp.ones(n).reshape(-1, 1)
    ones_m = jnp.ones(m).reshape(-1, 1)
    
    alpha = (1 / n) * jnp.ones(n).reshape(-1, 1)
    beta = (1 / m) * jnp.ones(m).reshape(-1, 1)
    init_params = (alpha, beta)
    params = (alpha, beta)
    opt_state = alternating_pd_optimizer.init(init_params)

    args = (L_fixed, R_fixed, C, g1, g2, rho)
    def cond_fn(carry):
        i, params, opt_state, grads, loss, converged = carry
        return (i < max_iter) & (~converged)
    
    def body_fn(carry):
        i, params, opt_state, grads, loss, _ = carry
        params, opt_state, grads, loss = alternating_proximal_descent_single_step(step_type, params, opt_state, args)
        grad_norm = jnp.linalg.norm(grads[0]) + jnp.linalg.norm(grads[1])
        has_converged = (jnp.max(jnp.abs(grads[0])) < convergence_threshold) & (jnp.max(jnp.abs(grads[1])) < convergence_threshold)
        return i + 1, params, opt_state, grads, loss, has_converged
    
    init_loss = alternating_proximal_descent_compute_loss(step_type, params, args)
    init_grads = (jnp.zeros_like(params[0]), jnp.zeros_like(params[1]))
    init_state = (0, params, opt_state, init_grads, init_loss, jnp.array(False))
    
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    
    _, params, _, _, _, _ = final_state
    return params

alternating_proximal_descent_step = jax.jit(alternating_proximal_descent_step, static_argnums=[6, 7, 8])

def alternating_relaxed_proximal_descent_L_admm(
    C: jnp.ndarray,
    R: jnp.ndarray,
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    rho: float,
    gamma: float = 1.0,
    nu: float = 10.0,  
    min_iter: int = 10,
    max_iter: int = 25,
    convergence_threshold: float = 1e-4
):
    """
    Solves the alternating proximal descent steps:
            min_{L : LR \in \Pi_{a,b}, ||LR||_+ \leq \gamma} <C, L>_F + (rho / 2) ||L||_F^2
    by using an ADMM approach.
    """
    D  = jnp.zeros_like(C)
    L1 = jnp.zeros_like(C)
    L2 = jnp.zeros_like(C)

    for i in range(max_iter):
        C_prime = C - nu * (L2 - D)
        alpha, beta = alternating_proximal_descent_step(C_prime, g1, g2, rho + nu, L1, R, ProximalDescentStep.L_STEP)
        L1 = alternating_proximal_descent_compute_L(C_prime, rho + nu, L1, R, alpha, beta) 
        L2 = gamma * row_constrained_projection((L1 + D).T / gamma, jnp.linalg.norm(R, axis=1)).T
        D = D + (L1 - L2)

        if jnp.max(jnp.abs(L1 - L2)) < convergence_threshold and i >= min_iter:
            break

    return L1

def alternating_relaxed_proximal_descent_R_admm(
    C: jnp.ndarray,
    L: jnp.ndarray,
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    rho: float,
    gamma: float = 1.0,
    nu: float = 10.0,  
    min_iter: int = 10,
    max_iter: int = 30,
    convergence_threshold: float = 1e-4
):
    """
    Solves the alternating proximal descent steps:
            min_{R : LR ]\in \\Pi_{a,b}, ||LR||_+ \\leq \\gamma} <R, L>_F + (rho / 2) ||R||_F^2
    by using an ADMM approach.
    """
    D  = jnp.zeros_like(C)
    R1 = jnp.zeros_like(C)
    R2 = jnp.zeros_like(C)

    for i in range(max_iter):
        C_prime = C - nu * (R2 - D)
        alpha, beta = alternating_proximal_descent_step(C_prime, g1, g2, rho + nu, L, R1, ProximalDescentStep.R_STEP)
        R1 = alternating_proximal_descent_compute_R(C_prime, rho + nu, L, R1, alpha, beta) 
        R2 = gamma * row_constrained_projection((R1 + D) / gamma, jnp.linalg.norm(L, axis=0))
        D = D + (R1 - R2)
        if jnp.max(jnp.abs(R1 - R2)) < convergence_threshold and i >= min_iter:
            break

    return R1

def alternating_mirror_descent_low_rank_ot(
    C: jnp.ndarray, 
    g1: jnp.ndarray, 
    g2: jnp.ndarray, 
    rank_1 : int,
    rank_2 : int = None,
    rho: float = 1.0,
    gamma: float = 1.0,
    seed: int = 0,
    L_init: jnp.ndarray = None,
    R_init: jnp.ndarray = None,
    max_iter: int = 25,
):
    """
    Solves the low-rank optimal transport problem using alternating mirror 
    descent with the low rank factorization P = L @ R, where L and R are
    non-negative matrices of shape (n, rank) and (rank, m) respectively.
    """

    n, m = C.shape
    if g1.shape != (n,) or g2.shape != (m,):
        raise ValueError("Dimension mismatch between C, g1, and g2.")
    
    if L_init is not None and R_init is not None:
        L, R = L_init, R_init
    else:
        L, R = initialize_factors(C, g1, g2, rank_1, seed=seed)

    ones_n = jnp.ones(n).reshape(-1, 1)
    ones_m = jnp.ones(m).reshape(-1, 1)

    logger.info(f"Initial objective: {jnp.sum(C * (L @ R))}")
    
    for i in range(max_iter):
        step_type = ProximalDescentStep.L_STEP if i % 2 == 0 else ProximalDescentStep.R_STEP

        if step_type == ProximalDescentStep.R_STEP:
            C_prime = L.T @ C - rho * R

            if rank_2 is not None:
                R = alternating_relaxed_proximal_descent_R_admm(C_prime, L, g1, g2, rho, nu=1.0, gamma=gamma * rank_2)
            else:
                alpha, beta = alternating_proximal_descent_step(C_prime, g1, g2, rho, L, R, step_type)
                R = alternating_proximal_descent_compute_R(C_prime, rho, L, R, alpha, beta)
        else:
            C_prime = C @ R.T - rho * L

            if rank_2 is not None:
                L = alternating_relaxed_proximal_descent_L_admm(C_prime, R, g1, g2, rho, nu=1.0, gamma=gamma * rank_2)
            else:
                alpha, beta = alternating_proximal_descent_step(C_prime, g1, g2, rho, L, R, step_type)
                L = alternating_proximal_descent_compute_L(C_prime, rho, L, R, alpha, beta)
        
        L, R = sinkhorn_rescaling(L, R, g1, g2)
        logger.info(f"Iteration {i} Objective: {jnp.sum(C * (L @ R))}")

    return L, R

def auglag_cmp_proj(Q, r):
    pos_Q = jnp.maximum(0.0, Q)
    return jnp.sqrt(r) * (pos_Q / jnp.linalg.norm(pos_Q))

def auglag_cmp_constraints(Q, R, rank):
    """
    Computes the constraints for the low-rank factorization.
    Returns a tuple of four constraints:
        1. Q @ Q.T @ ones_n - ones_n
        2. R @ R.T @ ones_n - ones_n
        3. Q @ R.T @ ones_n - ones_n
        4. R @ Q.T @ ones_n - ones_n
    """
    n = Q.shape[0]
    ones_n = jnp.ones(n).reshape(-1, 1)
    diff1 = Q @ Q.T @ ones_n - ones_n
    diff2 = R @ R.T @ ones_n - ones_n
    diff3 = Q @ R.T @ ones_n - ones_n
    diff4 = R @ Q.T @ ones_n - ones_n
    diff5 = jnp.trace(R.T @ R) - rank
    diff6 = jnp.trace(Q.T @ Q) - rank
    return diff1, diff2, diff3, diff4, diff5, diff6

def auglag_cmp_loss(C, Q, R, rank, μ, λ1, λ2):
    """
    C : n x n cost matrix
    Q : n x r matrix
    R : n x r matrix
    μ : scalar, penalty parameter for the primal term
    λ : 4 x n vector, Lagrange multipliers for the constraints
    """
    n = C.shape[0]
    ones_n = jnp.ones(n).reshape(-1, 1) # n x 1 vector of ones
    primal_term = jnp.sum(C * (Q @ R.T))
    diff1, diff2, diff3, diff4, diff5, diff6 = auglag_cmp_constraints(Q, R, rank)
    quad_term = jnp.sum(diff1**2 + diff2**2 + diff3**2 + diff4**2 + diff5**2 + diff6**2) 
    lagrange_term = (λ1[0] @ diff1 + λ1[1] @ diff2 + λ1[2] @ diff3 + λ1[3] @ diff4)[0]
    lagrange_term += λ2[0] * diff5 + λ2[1] * diff6
    return primal_term + (μ / 2) * quad_term - lagrange_term

auglag_cmp_loss_grad = jax.value_and_grad(auglag_cmp_loss, argnums=(1, 2))
auglag_cmp_loss_grad = jax.jit(auglag_cmp_loss_grad)

def auglag_convex_monge_sep(
    C: jnp.ndarray, 
    rank_1: int,
    rank_2: int = None,
    max_iter: int = 50,
    inner_iter: int = 10000,
    tol: float = 1e-4,
    constraint_tol: float = 1e-2,
    μ_increase_factor: float = 1.3,
    μ_init: float = 0.01,
    init_learning_rate: float = 1e-3,
    seed: int = 0
):
    """
    Solve the convex Monge map problem using augmented Lagrangian method.
    
    Args:
        C: Cost matrix (n x n)
        rank_1: Initial rank
        rank_2: Target rank for low-rank constraint (if None, use rank_1)
        max_iter: Maximum number of outer iterations
        inner_iter: Maximum number of inner iterations for projected gradient descent
        tol: Convergence tolerance
        μ_increase_factor: Factor to increase penalty parameter
        learning_rate: Learning rate for gradient descent
        seed: Random seed
    """
    n = C.shape[0]
    if n != C.shape[1]:
        raise ValueError("C must be a square matrix.")
    
    rank_2 = rank_1 if rank_2 is None else rank_2

    # Initialize Q and R randomly with proper normalization
    Q, R = initialize_factors(C, jnp.ones(n), jnp.ones(n), rank_1, seed=seed)
    Q, R = auglag_cmp_proj(Q, rank_1), auglag_cmp_proj(R.T, rank_1)
    
    μ = μ_init
    λ1 = jnp.zeros((4, n))
    λ2 = jnp.zeros(2)
    
    ones_n = jnp.ones(n).reshape(-1, 1)
    best_obj = jnp.inf
    best_Q, best_R = Q, R
    
    for i in range(max_iter):
        # Step 1: Use L-BFGS to optimize Q and R
        learning_rate = init_learning_rate
        j = 0
        while True:
            loss, (Q_grad, R_grad) = auglag_cmp_loss_grad(C, Q, R, rank_1, μ, λ1, λ2)
            Q_new = jnp.maximum(0.0, Q - learning_rate * Q_grad)
            R_new = jnp.maximum(0.0, R - learning_rate * R_grad)
            new_loss = auglag_cmp_loss(C, Q_new, R_new, rank_1, μ, λ1, λ2)
            resid = jnp.linalg.norm(Q_new - Q) + jnp.linalg.norm(R_new - R)

            # Armijo line search condition
            # c is the Armijo coefficient (typically between 0.0001 and 0.1)
            c = 0.0001
            armijo_condition = loss - new_loss >= c * learning_rate * (jnp.sum(Q_grad * (Q - Q_new)) + jnp.sum(R_grad * (R - R_new)))
            if not armijo_condition:
                learning_rate *= 0.9
                continue

            j += 1
            Q, R = Q_new, R_new
            learning_rate = init_learning_rate
            if j % 10 == 0:
                logger.info(f"Inner Iteration {j}, Loss: {loss:.6f}, Residual: {resid:.6f}")

            if resid < tol and j > 100:
                logger.info(f"Inner loop converged at iteration {j}")
                break
        
        # Step 2: Update Lagrange multipliers
        diff1, diff2, diff3, diff4, diff5, diff6 = auglag_cmp_constraints(Q, R, rank_1)
        λ1 = λ1.at[0].set(λ1[0] - μ * diff1[:,0])
        λ1 = λ1.at[1].set(λ1[1] - μ * diff2[:,0])
        λ1 = λ1.at[2].set(λ1[2] - μ * diff3[:,0])
        λ1 = λ1.at[3].set(λ1[3] - μ * diff4[:,0])
        λ2 = λ2.at[0].set(λ2[0] - μ * diff5)
        λ2 = λ2.at[1].set(λ2[1] - μ * diff6)

        # Step 3: Update penalty parameter μ
        constraint_violation = jnp.sqrt(jnp.sum(diff1**2 + diff2**2 + diff3**2 + diff4**2 + diff5**2 + diff6**2))
        if constraint_violation > tol:
            μ = μ * μ_increase_factor
        
        # Step 4: Compute objective and track best solution
        P = Q @ R.T
        obj_value = jnp.sum(C * P)
        
        if obj_value < best_obj and constraint_violation < constraint_tol:
            best_obj = obj_value
            best_Q, best_R = Q, R
        
        # Print progress every few iterations
        logger.info(f"Iteration {i}, Objective: {obj_value:.6f}, Constraint violation: {constraint_violation:.6f}, μ: {μ:.2f}")
        
        # Check for convergence
        if constraint_violation < constraint_tol:
            logger.info(f"Converged at iteration {i}")
            break
    
    # Return the best solution found
    return best_Q, best_R.T

def sinkhorn_rescaling(L, R, g1, g2, max_iter=100, tol=1e-4):
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

        norm1 = jnp.linalg.norm(L @ R @ jnp.ones(R.shape[1]) - g1)
        norm2 = jnp.linalg.norm(R.T @ L.T @ jnp.ones(L.shape[0]) - g2)
        if norm1 < tol and norm2 < tol:
            break
    return L, R

def sinkhorn_rescaling_P(P, g1, g2, max_iter=100, tol=1e-4):
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

def nonnegative_rounding_P(P, g1, g2, k, seed=0):
    U, s, Vt = jnp.linalg.svd(P)
    s = s.at[k:].set(0)
    P_svd = U @ jnp.diag(s) @ Vt
    model = NMF(n_components=k, init='random', random_state=seed, max_iter=10000, solver='mu', beta_loss='frobenius')
    W = model.fit_transform(P)
    H = model.components_
    L_round, R_round = sinkhorn_rescaling(W, H, g1, g2)
    return L_round, R_round, P_svd

def nonnegative_rounding(L, R, g1, g2, k, seed=0):
    """
    Performs non-negative rounding on the low-rank factors L and R.
    """
    singular_values = jnp.linalg.norm(L, axis=0) * jnp.linalg.norm(R, axis=1)
    sorted_indices = jnp.argsort(singular_values)[::-1]
    top_k_indices = sorted_indices[:k]
    L_k = L[:, top_k_indices]
    R_k = R[top_k_indices, :]
    return L_k, R_k