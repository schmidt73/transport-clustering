
import sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import sys
import GKMS.GKMS as gkms
import torch

import ott
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from loguru import logger
from clustering import lloyds_kmeans
import gurobipy as gp
from gurobipy import GRB
import numpy as np

def squared_euclidean_cost(X, Y):
    X2 = jnp.sum(X**2, axis=1, keepdims=True)
    Y2 = jnp.sum(Y**2, axis=1, keepdims=True).T
    return X2 + Y2 - 2.0 * X @ Y.T

def monge_permutation(C):
    """
    Exactly computes the Monge permutation and 
    transport matrix from cost matrix C using Gurobi.
    """
    
    n = C.shape[0]
    C_np = np.array(C)
    
    m = gp.Model("monge")
    m.setParam('OutputFlag', 1)
    
    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")
    
    obj = gp.quicksum(C_np[i, j] * x[i, j] for i in range(n) for j in range(n))
    m.setObjective(obj, GRB.MINIMIZE)
    
    for i in range(n): m.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
    for j in range(n): m.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1)
    
    m.optimize()
    
    P = jnp.zeros((n, n))
    if m.status == GRB.OPTIMAL:
        for i in range(n):
            for j in range(n):
                if x[i, j].x > 0.5:
                    P = P.at[i, j].set(1.0)
    
    return P

def random_Q_init(n, r, random_state=None):
    key = jax.random.PRNGKey(random_state if random_state is not None else 0)
    Q = jnp.abs(jax.random.uniform(key, (n, r)))
    row_sums = jnp.sum(Q, axis=1, keepdims=True)
    Q = Q / (n * row_sums)
    return Q

def stabilize_Q_init(Q, lambda_factor=0.5):
    n, r = Q.shape[0], Q.shape[1]
    eps_Q = random_Q_init(n, r, random_state=0)
    Q_init = (1 - lambda_factor) * Q + lambda_factor * eps_Q
    return Q_init

def gkms(C, Q_init, gamma_init=2.0, rescale_gamma=True, max_iter=20000):
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
    return Q_curr

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

def monge_rotation_kmeans(C, X, Y, r, lambda_factor=0.1, random_state=0, kmeans_init=False):
    n = C.shape[0]
    P = monge_permutation(C)
    if kmeans_init:
        logger.info("Computed Monge permutation, performing k-means initialization...")
        labels_X, centers_X = lloyds_kmeans(X, r, random_state=random_state)
        labels_Y, centers_Y = lloyds_kmeans(Y, r, random_state=random_state)

        Q1 = jnp.zeros((n, r))
        Q1 = Q1.at[jnp.arange(n), labels_X].set(1.0 / n)
        R1 = P.T @ Q1
        g1 = jnp.sum(Q1, axis=0)

        R2 = jnp.zeros((n, r))
        R2 = R2.at[jnp.arange(n), labels_Y].set(1.0 / n)
        Q2 = P @ R2
        g2 = jnp.sum(Q2, axis=0)

        cost1 = jnp.sum(C * (Q1 @ jnp.diag(1 / g1) @ R1.T))
        cost2 = jnp.sum(C * (Q2 @ jnp.diag(1 / g2) @ R2.T))

        logger.info(f"Initialization Costs: ({cost1}, {cost2})")
        if cost1 < cost2:
            Q = gkms(C @ P.T, stabilize_Q_init(Q1, lambda_factor=lambda_factor))
            return Q, jnp.sum(Q, axis=0), P.T @ Q
        else:
            Q = gkms(P.T @ C, stabilize_Q_init(Q2, lambda_factor=lambda_factor))
            return P @ Q, jnp.sum(Q, axis=0), Q 

    def bm_init(C_tilde):
        U, objective_values = sdp_subproblem_bm(C_tilde, r, verbose=True)
        _, eigvecs = jnp.linalg.eigh(U @ U.T - (1 / n) * jnp.ones((n, n)))
        top_eigvecs = eigvecs[:, -(r-1):]
        labels, _ = lloyds_kmeans(top_eigvecs, r, random_state=random_state)
        Q_init = jnp.zeros((n, r))
        Q_init = Q_init.at[jnp.arange(n), labels].set(1.0 / n)
        return stabilize_Q_init(Q_init, lambda_factor=lambda_factor)

    Q_init_1 = bm_init(C @ P.T)
    Q1 = gkms(C @ P.T, Q_init_1)
    R1 = P.T @ Q1
    cost1 = jnp.sum(C * (Q1 @ jnp.diag(1 / jnp.sum(Q1, axis=0)) @ R1.T))

    Q_init_2 = bm_init(P.T @ C)
    R2 = gkms(P.T @ C, Q_init_2)
    Q2 = P @ R2
    cost2 = jnp.sum(C * (Q2 @ jnp.diag(1 / jnp.sum(Q2, axis=0)) @ R2.T))

    logger.info(f"Costs: ({cost1}, {cost2})")
    if cost1 < cost2:
        return Q1, jnp.sum(Q1, axis=0), R1
    else:
        return Q2, jnp.sum(Q2, axis=0), R2
    