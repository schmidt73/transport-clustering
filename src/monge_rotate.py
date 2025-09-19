
import sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import GKMS.GKMS as gkms
import torch

import ott
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from loguru import logger
from sklearn.cluster import KMeans
import numpy as np

def monge_permutation(C):
    n = C.shape[0]
    a = jnp.ones(n) / n
    b = jnp.ones(n) / n
    geom = geometry.Geometry(cost_matrix=C, epsilon=1e-4)
    problem = linear_problem.LinearProblem(geom, a=a, b=b)
    solver = sinkhorn.Sinkhorn()
    solution = solver(problem)
    P_soft = solution.matrix
    return P_soft

def random_Q_init(n, r, random_state=None):
    key = jax.random.PRNGKey(random_state if random_state is not None else 0)
    Q = jnp.abs(jax.random.normal(key, (n, r)))
    row_sums = jnp.sum(Q, axis=1, keepdims=True)
    Q = Q / (n * row_sums)
    return Q

def stabilize_Q_init(Q, lambda_factor=0.5):
    n, r = Q.shape[0], Q.shape[1]
    eps_Q = random_Q_init(n, r, random_state=0)
    Q_init = (1 - lambda_factor) * Q + lambda_factor * eps_Q
    return Q_init

def tree_where(cond, a, b):
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), a, b)

def sdp_subproblem_bm(
    C: jnp.ndarray, K: int, *, r: int | None = None,
    beta: float = 20.0, alpha: float = 1e-8,
    tol: float = 1e-6, tol_primal: float = 1e-2,
    nu : float = 0.9,  maxiter: int = 50_000, key: jax.Array | None = None,
    verbose: bool = False, log_every: int = 1000
):
    n = C.shape[0]
    r = 2 * K if r is None else r
    if key is None: key = jax.random.PRNGKey(0)

    one = jnp.ones((n, 1))
    norm_one = jnp.linalg.norm(one)
    sqrtK = jnp.sqrt(K)

    def project(V):
        Vp = jnp.maximum(V, 0.0)
        nf = jnp.sqrt(jnp.sum(Vp * Vp))
        scale = jnp.where(nf > 0, sqrtK / nf, 0.0)
        return Vp * scale

    def compute_objective(U, y):
        return (
            jnp.sum((C @ U) * U) + jnp.sum(y * (U @ (U.T @ one) - one)) 
            + (beta / 2) * jnp.sum((U @ (U.T @ one) - one) ** 2)
        )

    compute_obj_and_grad = jax.value_and_grad(compute_objective, argnums=0)

    y = jnp.zeros((n, 1))
    U = project(jax.random.uniform(key, (n, r), minval=0.0, maxval=1.0 / n))

    def body_fun(carry, idx):
        U, y, alpha_prime = carry

        obj, G = compute_obj_and_grad(U, y)
        Unew = project(U - alpha_prime * G)
        resid = (1 / alpha_prime) * jnp.linalg.norm(Unew - U)
        infeas = Unew @ (Unew.T @ one) - one

        cand1 = (U, y, nu * alpha_prime)
        cand2 = (Unew, y, (1.0 / nu) * alpha_prime)
        cand3 = (Unew, y + beta * infeas, alpha_prime)

        cond1 = obj < compute_objective(Unew, y)
        carry_tmp = tree_where(cond1, cand1, cand2)

        cond2 = resid > tol_primal
        new_carry = tree_where(cond2, carry_tmp, cand3)

        return new_carry, obj

    (U, y, alpha_prime), objs = jax.lax.scan(body_fun, (U, y, alpha), jnp.arange(maxiter))
    return U.block_until_ready(), objs.block_until_ready()

def gkms(C, Q_init, gamma_init=20.0, rescale_gamma=True, max_iter=100):
    def compute_loss(Q):
        return jnp.sum((Q.T @ C) * (jnp.diag(1 / jnp.sum(Q, axis=0)) @ Q.T))

    compute_loss_and_grad = jax.value_and_grad(compute_loss)

    n = C.shape[0]
    def body_fun(carry, idx):
        Q, gamma = carry
        loss, grad = compute_loss_and_grad(Q)
        Q_new = Q * jnp.exp(-gamma * grad)
        row_scaling_vector = jnp.sum(Q_new, axis=1)
        Q_new = jnp.diag(1 / (n * row_scaling_vector)) @ Q_new
        if rescale_gamma:
            gamma = gamma_init / (jnp.max(jnp.abs(grad)) ** 2)
        return (Q_new, gamma), loss

    (Q_curr, _), losses = jax.lax.scan(body_fun, (Q_init, gamma_init), jnp.arange(max_iter))
    loss = compute_loss(Q_curr)
    logger.info(f"Final loss: {loss}")
    return Q_curr, losses

def gkms_logdomain(C, Q_init, gamma_init=20.0, rescale_gamma=True, max_iter=500):
    def compute_loss(Q):
        return jnp.sum((Q.T @ C) * (jnp.diag(1 / jnp.sum(Q, axis=0)) @ Q.T))

    compute_loss_and_grad = jax.value_and_grad(compute_loss)

    n = C.shape[0]
    def body_fun(carry, idx):
        log_Q, gamma = carry
        loss, grad = compute_loss_and_grad(jnp.exp(log_Q))
        log_Q_new = log_Q - gamma * grad
        row_scaling_vector = jax.scipy.special.logsumexp(log_Q_new, axis=1)
        log_Q_new = log_Q_new - row_scaling_vector[:,None] - jnp.log(n)
        if rescale_gamma:
            gamma = gamma_init / jnp.max(jnp.abs(grad) ** 2)
        return (log_Q_new, gamma), loss

    log_Q_init = jnp.log(Q_init + 1e-20)
    (log_Q_curr, _), losses = jax.lax.scan(body_fun, (log_Q_init, gamma_init), jnp.arange(max_iter))
    loss = compute_loss(jnp.exp(log_Q_curr))
    return jnp.exp(log_Q_curr), losses

def save_loss_plot(losses, filename):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    n = len(losses)
    losses = losses[(n // 2):]
    sns.scatterplot(x=jnp.arange(n // 2, n), y=losses, ax=ax)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    plt.savefig(filename)

def monge_conjugate(
    C, r, X=None, Y=None, random_state=0, bm_init=False, kmeans_init=False, 
    rescale_gamma=True, gamma_init=20, max_iter=500, lambda_factor=0.5, 
    log_domain=True, debug=False
):
    n = C.shape[0]
    P = monge_permutation(C)

    def bm_initializer(C_tilde):
        U, objective_values = sdp_subproblem_bm(C_tilde, r, verbose=True)
        _, eigvecs = jnp.linalg.eigh(U @ U.T - (1 / n) * jnp.ones((n, n)))
        top_eigvecs = eigvecs[:, -(r-1):]
        kmeans = KMeans(n_clusters=r, n_init=r, random_state=random_state)
        labels = kmeans.fit_predict(np.array(top_eigvecs))
        Q_init = jnp.zeros((n, r))
        Q_init = Q_init.at[jnp.arange(n), labels].set(1.0 / n)
        return stabilize_Q_init(Q_init, lambda_factor=lambda_factor), objective_values
    
    def kmeans_initializer(points):
        kmeans = KMeans(n_clusters=r, n_init=r, random_state=random_state)
        labels = kmeans.fit_predict(np.array(points))
        Q_init = jnp.zeros((n, r))
        Q_init = Q_init.at[jnp.arange(n), labels].set(1.0 / n)
        return stabilize_Q_init(Q_init, lambda_factor=lambda_factor)

    if bm_init:
        Q_init_1, losses = bm_initializer(C @ P.T @ jnp.diag(1 / jnp.sum(P, axis=1)))
        if debug: save_loss_plot(losses, "bm_init1_loss.png")
    elif kmeans_init:
        if X is None: raise ValueError("X must be provided for k-means initialization")
        Q_init_1 = kmeans_initializer(X)
    else:
        Q_init_1 = random_Q_init(n, r, random_state=random_state)

    gkms_f = gkms_logdomain if log_domain else gkms
    logger.info("Running GKMS with CP^T initialization")
    Q1, losses = gkms_f(
        C @ P.T @ jnp.diag(1 / jnp.sum(P, axis=1)), Q_init_1, 
        gamma_init=gamma_init, rescale_gamma=rescale_gamma, max_iter=max_iter
    )
    R1 = P.T @ jnp.diag(1 / jnp.sum(P, axis=1)) @ Q1
    cost1 = jnp.sum(C * (Q1 @ jnp.diag(1 / jnp.sum(Q1, axis=0)) @ R1.T))
    if debug: save_loss_plot(losses, "gkms1_loss.png")

    if bm_init:
        Q_init_2, losses = bm_initializer(jnp.diag(1 / jnp.sum(P, axis=0)) @ P.T @ C)
        if debug: save_loss_plot(losses, "bm_init2_loss.png")
    elif kmeans_init:
        if Y is None: raise ValueError("Y must be provided for k-means initialization")
        Q_init_2 = P @ jnp.diag(1 / jnp.sum(P, axis=0)) @ kmeans_initializer(Y)
    else:
        Q_init_2 = random_Q_init(n, r, random_state=random_state+1)

    logger.info("Running GKMS with P^TC initialization")
    R2, losses = gkms_f(
        jnp.diag(1 / jnp.sum(P, axis=0)) @ P.T @ C, Q_init_2,
        gamma_init=gamma_init, rescale_gamma=rescale_gamma, max_iter=max_iter
    )
    Q2 = P @ jnp.diag(1 / jnp.sum(P, axis=0)) @ R2
    cost2 = jnp.sum(C * (Q2 @ jnp.diag(1 / jnp.sum(Q2, axis=0)) @ R2.T))

    if debug: save_loss_plot(losses, "gkms2_loss.png")

    logger.info(f"Costs: ({cost1}, {cost2})")

    if cost1 < cost2:
        return Q1, jnp.sum(Q1, axis=0), R1
    else:
        return Q2, jnp.sum(Q2, axis=0), R2
    