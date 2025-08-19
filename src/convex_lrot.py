import jax
jax.config.update("jax_enable_x64", True)
from jax import lax, random

import jax.numpy as jnp
import cvxpy as cp
import numpy as np

from loguru import logger
from sklearn.cluster import KMeans

def monge(C):
    """Solve the Monge problem."""
    n = C.shape[0]
    P = cp.Variable((n, n), nonneg=True)
    objective = cp.Minimize(cp.sum(cp.multiply(C, P)))
    constraints = [cp.sum(P, axis=0) == 1, cp.sum(P, axis=1) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=True)
    return P.value

def sdp_subproblem(C, rank):
    """Solve the SDP subproblem."""
    n = C.shape[0]
    P = cp.Variable((n, n), PSD=True)
    objective = cp.Minimize(cp.sum(cp.multiply(C, P)))
    constraints = [
        cp.sum(P, axis=0) == 1, 
        cp.sum(P, axis=1) == 1, 
        cp.trace(P) == rank, 
        P >= 0
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=True)
    return P.value

def solve_lrot(C, rank, key=random.PRNGKey(0), beta=10., alpha=1e-6, tol=1e-6, tol_primal=1e-3, maxiter=50000):
    assert C.shape[0] == C.shape[1], "C must be a square matrix"

    P = monge(C)
    C_tilde = C @ P
    U = sdp_subproblem(C_tilde, rank)

    n = U.shape[0]
    _, eigvecs = jnp.linalg.eigh(U)
    top_eigvecs = eigvecs[:, -rank:]

    kmeans = KMeans(n_clusters=rank, random_state=0).fit(top_eigvecs)
    labels = kmeans.labels_

    Q = jnp.zeros((n, rank))
    for i in range(n):
        Q = Q.at[i, labels[i]].set(1)
    R = P @ Q
    return Q @ jnp.linalg.inv(Q.T @ Q), R
    
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
