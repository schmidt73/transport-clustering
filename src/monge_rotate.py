
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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

def monge_rotation_kmeans(C, X, Y, r, random_state=0):
    P = monge_permutation(C)
    logger.info("Computed Monge permutation, performing k-means initialization...")
    labels_X, centers_X = lloyds_kmeans(X, r, random_state=random_state)
    labels_Y, centers_Y = lloyds_kmeans(Y, r, random_state=random_state)

    n = X.shape[0]
    Q1 = jnp.zeros((n, r))
    Q1 = Q1.at[jnp.arange(n), labels_X].set(1.0)
    R1 = P.T @ Q1

    R2 = jnp.zeros((n, r))
    R2 = R2.at[jnp.arange(n), labels_Y].set(1.0)
    Q2 = P @ R2

    cost1 = jnp.sum(C * (Q1 @ jnp.diag(1 / (Q1.T @ jnp.ones(n))) @ R1.T))
    cost2 = jnp.sum(C * (Q2 @ jnp.diag(1 / (R2.T @ jnp.ones(n))) @ R2.T))

    logger.info(f"Cost 1: {cost1}, Cost 2: {cost2}")

    if cost1 < cost2:
        return Q1, Q1.T @ jnp.ones(n), R1
    else:
        return Q2, R2.T @ jnp.ones(n), R2