
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

def squared_euclidean_cost(X, Y):
    X2 = jnp.sum(X**2, axis=1, keepdims=True)
    Y2 = jnp.sum(Y**2, axis=1, keepdims=True).T
    return X2 + Y2 - 2.0 * X @ Y.T

def monge_permutation(C):
    """
    Approximately computes the Monge permutation and 
    transport matrix from cost matrix C.
    """
    n = C.shape[0]
    
    a = jnp.ones(n) / n
    b = jnp.ones(n) / n

    geom = geometry.Geometry(cost_matrix=C, epsilon=1e-3)
    problem = linear_problem.LinearProblem(geom, a=a, b=b)
    solver = sinkhorn.Sinkhorn()
    solution = solver(problem)
    P_soft = solution.matrix
    return P_soft

def stabilize_Q_init(Q, lambda_factor=0.5):
    n, r = Q.shape[0], Q.shape[1]
    eps_Q = jnp.ones((n, r)) / (n * r)
    Q_init = (1 - lambda_factor) * Q + lambda_factor * eps_Q
    return Q_init

def gkms(C, Q_init, gamma=50.0, max_iter=100):
    def compute_loss(Q):
        return jnp.sum((Q.T @ C) * (jnp.diag(1 / jnp.sum(Q, axis=0)) @ Q.T))
    compute_loss_and_grad = jax.jit(jax.value_and_grad(compute_loss))

    n = C.shape[0]
    Q_curr = Q_init
    for _ in range(max_iter):
        loss, grad = compute_loss_and_grad(Q_curr)
        Q_new = Q_curr * jnp.exp(-gamma * grad)
        row_scaling_vector = jnp.sum(Q_new, axis=1)
        Q_new = jnp.diag(1 / (n * row_scaling_vector)) @ Q_new
        Q_curr = Q_new
    
    return Q_curr
    
def monge_rotation_kmeans(C, X, Y, r, lambda_factor=0.5, random_state=0):
    n = X.shape[0]
    P = monge_permutation(C) * n

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