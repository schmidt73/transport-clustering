
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import ott
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from loguru import logger

'''
Code for Monge Rotation on Kernelized Costs (squared Euclidean):
    Reduction proceeds as:
        1. Monge Rotate cost: Ctilde = C @ P.T by solving for Monge map P
        2. Take Sym(Ctilde) = Ctilde + Ctilde.T
        3. Grammize Ctilde as -0.5 * J @ Sym(Ctilde) @ J for J = Id - (1/2) 11^T
        4. Embed by identifying G = X X^T and extracting X from decomposition
        5. Run K-means on X
        6. Yield Q from K-means, R = P.T @ Q
'''

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

def _kmeanspp_init(X, k, rng):
    n = X.shape[0]
    centers = jnp.empty((k, X.shape[1]), dtype=X.dtype)
    i0 = rng.integers(n)
    centers = centers.at[0].set(X[i0])
    d2 = jnp.sum((X - centers[0])**2, axis=1)
    for t in range(1, k):
        s = d2.sum()
        probs = d2/s if s > 0 else np.ones(n)/n
        it = rng.choice(n, p=probs)
        centers = centers.at[t].set(X[it])
        d2 = jnp.minimum(d2, jnp.sum((X - centers[t])**2, axis=1))
    return centers

def _lloyds_kmeans(X, k, max_iter=250, tol=1e-6, random_state=0):
    rng = np.random.default_rng(random_state)
    centers = _kmeanspp_init(X, k, rng)
    for _ in range(max_iter):
        d2 = jnp.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
        labels = jnp.argmin(d2, axis=1)
        new_centers = jnp.zeros_like(centers)
        for j in range(k):
            pts = X[labels == j]
            if len(pts) == 0:
                new_centers = new_centers.at[j].set(X[rng.integers(X.shape[0])])
            else:
                new_centers = new_centers.at[j].set(pts.mean(axis=0))
        if jnp.linalg.norm(new_centers - centers) <= tol:
            centers = new_centers
            break
        centers = new_centers
    d2 = jnp.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
    labels = jnp.argmin(d2, axis=1)
    return labels, centers

def monge_rotation_kmeans(C, X, Y, r, random_state=0):
    P = monge_permutation(C)
    logger.info("Computed Monge permutation, performing k-means initialization...")
    labels_X, centers_X = _lloyds_kmeans(X, r, random_state=random_state)
    labels_Y, centers_Y = _lloyds_kmeans(Y, r, random_state=random_state)

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