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
    constraints = [
        cp.sum(P, axis=0) == 1, 
        cp.sum(P, axis=1) == 1
    ]
    prob = cp.Problem((1 / n) * objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=True)
    return P.value

def tree_where(cond, a, b):
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), a, b)

def sdp_subproblem_bm(
    C: jnp.ndarray, K: int, *, r: int | None = None,
    beta: float = 20.0, alpha: float = 1e-8,
    tol: float = 1e-6, tol_primal: float = 1e-2,
    nu : float = 0.9,  maxiter: int = 100_000, key: jax.Array | None = None,
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

    @jax.jit
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
    prob = cp.Problem((1 / n) * objective, constraints)
    prob.solve(solver=cp.SCS, verbose=True)
    return P.value

def solve_lrot(C, rank):
    assert C.shape[0] == C.shape[1], "C must be a square matrix"
    n = C.shape[0]

    P = monge(C)
    C_tilde = C @ P.T
    logger.info("Solving SDP subproblem with Burer-Monteiro approach")
    U, objective_values = sdp_subproblem_bm(C_tilde, rank, verbose=True)
    logger.info(f"SDP Relaxation Objective: {jnp.sum((C_tilde @ U) * U) / n}")
    
    # only do eigendecomposition when we use SDP without BM
    #U = sdp_subproblem(C_tilde, rank)
    n = U.shape[0] 
    _, eigvecs = jnp.linalg.eigh(U @ U.T - (1 / n) * jnp.ones((n, n)))
    top_eigvecs = eigvecs[:, -(rank-1):]

    logger.info("SDP subproblem solved, proceeding with KMeans clustering")
    kmeans = KMeans(n_clusters=rank, random_state=0).fit(top_eigvecs)
    labels = kmeans.labels_

    Q = jnp.zeros((n, rank))
    for i in range(n):
        Q = Q.at[i, labels[i]].set(1)
    R = P.T @ Q
    return Q @ jnp.linalg.inv(Q.T @ Q), R, objective_values
    
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
