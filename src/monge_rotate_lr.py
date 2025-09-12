import jax
import jax.numpy as jnp

# Enable float64 if you want parity with torch.float64
from jax import config as _jax_cfg
_jax_cfg.update("jax_enable_x64", True)

import ott
from ott.geometry import geometry
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from loguru import logger
from clustering import lloyds_kmeans
import distance_utils as dist_util

def ott_soft_monge_plan_pointcloud(X, Y, epsilon=1e-2):
    """Balanced Sinkhorn on point clouds, returns soft plan P (n x n)."""
    geom = pointcloud.PointCloud(x=X, y=Y, epsilon=epsilon)
    n = X.shape[0]
    a = jnp.ones((n,)) / n
    b = jnp.ones((n,)) / n
    prob = linear_problem.LinearProblem(geom, a=a, b=b)
    sol = sinkhorn.Sinkhorn()(prob)
    return sol.matrix  # (n, n)

def lr_cost(A, B, Q, R):
    """⟨A B^T, Q Λ R^T⟩ without forming C. Q,R: (n,r)."""
    g = jnp.sum(Q, axis=0)                               # (r,)
    SA = Q.T @ A                                         # (r, k)
    RB = R.T @ B                                         # (r, k)
    return jnp.sum(jnp.sum(RB * SA, axis=1) / jnp.clip(g, 1e-18))

def stabilize_Q_init(Q, lambda_factor=0.5):
    n, r = Q.shape[0], Q.shape[1]
    eps_Q = jnp.ones((n, r)) / (n * r)
    Q_init = (1 - lambda_factor) * Q + lambda_factor * eps_Q
    return Q_init

def loss_lr(Q, A, B):
    g = jnp.sum(Q, axis=0)
    SA = Q.T @ A
    SB = Q.T @ B
    return jnp.sum(SB * (SA / jnp.clip(g, 1e-18)[:, None]))

loss_lr_and_grad = jax.value_and_grad(loss_lr)

def gkms_lr(A, B, Q_init, gamma=50.0, max_iter=100, tol=1e-9, min_iter=25):
    
    n, r = Q_init.shape
    Q = Q_init
    
    @jax.jit
    def step(Q):
        val, grad = loss_lr_and_grad(Q, A, B)
        Qn = Q * jnp.exp(-gamma * grad) # Exponential gradient step
        row_scaling_vector = jnp.sum(Qn, axis=1)
        Qn = jnp.diag(1 / (n * row_scaling_vector)) @ Qn # Diagonal projection of the positive kernel
        return val, Qn
    
    last = None
    
    for k in range(max_iter):
        val, Q = step(Q)
        if k >= max(2, min_iter) and last is not None and jnp.abs(val - last) <= tol:
            break
        last = val
    
    g = jnp.sum(Q, axis=0)
    return Q, g

def monge_rotation_kmeans_LR(X, Y, r, lambda_factor=0.5, random_state=0, epsilon=1e-2):
    """
    Low-rank Monge-rotation k-means initializer + GKMS.
    Avoids forming C entirely; rotates LR factors instead of C.
    """
    n = X.shape[0]
    # LR factors for squared Euclidean
    A, B = dist_util.compute_lr_sqeuclidean_factors(X, Y, rescale_cost=False)
    
    # Soft Monge plan via OTT (no dense C; may update to add HiRef)
    P = ott_soft_monge_plan_pointcloud(X, Y, epsilon=epsilon) * n
    
    # Lloyd init
    labels_X, centers_X = lloyds_kmeans(X, r, random_state=random_state)
    labels_Y, centers_Y = lloyds_kmeans(Y, r, random_state=random_state)
    
    # One-hot membership matrices on rows, scaled by 1/n
    Q1 = jnp.zeros((n, r)).at[jnp.arange(n), labels_X].set(1.0 / n)
    R2 = jnp.zeros((n, r)).at[jnp.arange(n), labels_Y].set(1.0 / n)
    
    # Rotate memberships by the Monge plan
    R1 = P.T @ Q1                     # corresponds to right-rotation (C @ P^T)
    Q2 = P @ R2                       # corresponds to left-rotation (P^T @ C)
    
    g1 = jnp.sum(Q1, axis=0)          # (r,)
    g2 = jnp.sum(Q2, axis=0)          # (r,)
    
    # Initial costs via LR formula (no C)
    cost1 = lr_cost(A, B, Q1, R1)
    cost2 = lr_cost(A, B, Q2, R2)
    
    logger.info(f"Initialization Costs: ({cost1}, {cost2})")
    
    if cost1 < cost2:
        # Use C @ P^T  ==  A @ (P B)^T  → keep A, set B_rot = P @ B
        B_rot = P @ B
        Q0 = stabilize_Q_init(Q1, lambda_factor=lambda_factor)
        Q, g = gkms_lr(A, B_rot, Q0)
        return Q, jnp.sum(Q, axis=0), P.T @ Q
    else:
        # Use P^T @ C  ==  (P^T A) @ B^T  → keep B, set A_rot = P^T @ A
        A_rot = P.T @ A
        Q0 = stabilize_Q_init(Q2, lambda_factor=lambda_factor)
        Q, g = gkms_lr(A_rot, B, Q0)
        return P @ Q, jnp.sum(Q, axis=0), Q

