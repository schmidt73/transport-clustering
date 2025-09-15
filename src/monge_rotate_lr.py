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
import sys
sys.path.insert(0, '../src/HiRef/')
import HiRef
import rank_annealing

def ott_soft_monge_plan_pointcloud(X, Y, epsilon=1e-2):
    """Balanced Sinkhorn on point clouds, returns soft plan P (n x n)."""
    geom = pointcloud.PointCloud(x=X, y=Y, epsilon=epsilon)
    n = X.shape[0]
    a = jnp.ones((n,)) / n
    b = jnp.ones((n,)) / n
    prob = linear_problem.LinearProblem(geom, a=a, b=b)
    sol = sinkhorn.Sinkhorn()(prob)
    return sol.matrix  # (n, n)

@jax.jit
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
    
@jax.jit
def loss_lr(Q, A, B):
    g = jnp.sum(Q, axis=0)
    SA = Q.T @ A
    SB = Q.T @ B
    return jnp.sum(SB * (SA / jnp.clip(g, 1e-18)[:, None]))

loss_lr_and_grad = jax.jit(jax.value_and_grad(loss_lr))

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

def monge_rotation_kmeans_LR(X, Y, r, lambda_factor=0.5,
                             random_state=0, epsilon=1e-2, 
                             ot_solver='HiRef', rescale=True):
    """
    Low-rank Monge-rotation k-means initializer + GKMS.
    Avoids forming C; rotates LR factors A,B and (Q1,R2) via P or via a permutation from HiRef.
    """
    n = X.shape[0]
    # LR factors for squared Euclidean: C = A @ B.T
    A, B = dist_util.compute_lr_sqeuclidean_factors(X, Y, rescale_cost=rescale)
    
    # Lloyd init
    labels_X, centers_X = lloyds_kmeans(X, r, random_state=random_state)
    labels_Y, centers_Y = lloyds_kmeans(Y, r, random_state=random_state)

    # One-hot membership matrices on rows, scaled by 1/n
    Q1 = jnp.zeros((n, r)).at[jnp.arange(n), labels_X].set(1.0 / n)
    R2 = jnp.zeros((n, r)).at[jnp.arange(n), labels_Y].set(1.0 / n)

    # Compute Monge “rotation”
    if ot_solver == 'Sinkhorn':
        # soft plan (dense); keep matmuls
        P = ott_soft_monge_plan_pointcloud(X, Y, epsilon=epsilon) * n

        # Rotate memberships by the Monge plan
        R1 = P.T @ Q1              # corresponds to right-rotation (C @ P^T)
        Q2 = P @ R2                # corresponds to left-rotation (P @ R2)

        # Initial costs via LR formula (no C)
        cost1 = lr_cost(A, B, Q1, R1)
        cost2 = lr_cost(A, B, Q2, R2)

        logger.info(f"Initialization Costs: ({cost1}, {cost2})")

        if cost1 < cost2:
            # C @ P^T == A @ (P B)^T  → B_rot = P @ B
            B_rot = P @ B
            Q0 = stabilize_Q_init(Q1, lambda_factor=lambda_factor)
            Q, g = gkms_lr(A, B_rot, Q0)
            return Q, jnp.sum(Q, axis=0), P.T @ Q
        else:
            # P^T @ C == (P^T A) @ B^T → A_rot = P^T @ A
            A_rot = P.T @ A
            Q0 = stabilize_Q_init(Q2, lambda_factor=lambda_factor)
            Q, g = gkms_lr(A_rot, B, Q0)
            return P @ Q, jnp.sum(Q, axis=0), Q

    elif ot_solver == 'HiRef':
        # permutation-like output (sparse); replace matmuls with indexing
        rank_schedule = rank_annealing.optimal_rank_schedule(
            n, hierarchy_depth=6, max_Q=1000, max_rank=100
        )
        with jax.default_device(jax.devices("gpu")[0]):
            XA = jnp.asarray(X)
            YB = jnp.asarray(Y)
        # returns list of (idxX, idxY) leaves
        frontier = HiRef.hiref_lr(
            XA, YB, rank_schedule, iters_per_level=100, gamma=60.0,
            rescale_cost=False, return_coupling=False
        )
        # permutation vectors
        pi, inv_pi = _clusters_to_perm(frontier, n)   # pi[i]=j ; inv_pi[j]=i
        
        # Rotate memberships using permutation (instead of multiplying by P)
        R1 = Q1[inv_pi, :]       # P.T @ Q1
        Q2 = R2[pi, :]           # P @ R2
        
        # Costs via LR factors (no C)
        cost1 = lr_cost(A, B, Q1, R1)
        cost2 = lr_cost(A, B, Q2, R2)
        
        logger.info(f"Initialization Costs: ({cost1}, {cost2})")
        
        if cost1 < cost2:
            # B_rot = P @ B → index rows of B by pi
            B_rot = B[pi, :]
            Q0 = stabilize_Q_init(Q1, lambda_factor=lambda_factor)
            Q, g = gkms_lr(A, B_rot, Q0)
            # P.T @ Q → reorder rows of Q by inv_pi
            return Q, jnp.sum(Q, axis=0), Q[inv_pi, :]
        else:
            # A_rot = P^T @ A → index rows of A by inv_pi
            A_rot = A[inv_pi, :]
            Q0 = stabilize_Q_init(Q2, lambda_factor=lambda_factor)
            Q, g = gkms_lr(A_rot, B, Q0)
            # P @ Q → reorder rows of Q by pi
            return Q[pi, :], jnp.sum(Q, axis=0), Q
    else:
        raise ValueError("ot_solver must be 'Sinkhorn' or 'HiRef'")

def _clusters_to_perm(frontier, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert HiRef frontier (list of (idxX, idxY)) leaves into a permutation mapping.
    Assumes each leaf is 1–1 or we take the first min(|idxX|,|idxY|) pairs in order.
    Returns (pi, inv_pi) where pi[i]=j and inv_pi[j]=i.
    """
    pi = -jnp.ones((n,), dtype=jnp.int32)
    for idxX, idxY in frontier:
        ix = jnp.asarray(idxX, jnp.int32)
        iy = jnp.asarray(idxY, jnp.int32)
        m = int(jnp.minimum(ix.size, iy.size))
        if m == 0: 
            continue
        pi = pi.at[ix[:m]].set(iy[:m])
    # sanity: fill any unmapped with identity (rare; only if leaves weren’t fully refined)
    unmapped = jnp.where(pi < 0, jnp.arange(n, dtype=jnp.int32), pi)
    pi = unmapped
    # inverse
    inv_pi = jnp.empty_like(pi)
    inv_pi = inv_pi.at[pi].set(jnp.arange(n, dtype=jnp.int32))
    return pi, inv_pi


'''
def monge_rotation_kmeans_LR(X, Y, r, lambda_factor=0.5,
                             random_state=0, epsilon=1e-2, ot_solver='Sinkhorn'):
    """
    Low-rank Monge-rotation k-means initializer + GKMS.
    Avoids forming C entirely; rotates LR factors instead of C.
    """
    n = X.shape[0]
    # LR factors for squared Euclidean
    A, B = dist_util.compute_lr_sqeuclidean_factors(X, Y, rescale_cost=False)
    
    # Lloyd init
    labels_X, centers_X = lloyds_kmeans(X, r, random_state=random_state)
    labels_Y, centers_Y = lloyds_kmeans(Y, r, random_state=random_state)

    # One-hot membership matrices on rows, scaled by 1/n
    Q1 = jnp.zeros((n, r)).at[jnp.arange(n), labels_X].set(1.0 / n)
    R2 = jnp.zeros((n, r)).at[jnp.arange(n), labels_Y].set(1.0 / n)
    
    # Soft Monge plan via OTT (no dense C; may update to add HiRef)
    print('Running OT.')
    if 'Sinkhorn':
        P = ott_soft_monge_plan_pointcloud(X, Y, epsilon=epsilon) * n
    elif 'HiRef':
        rank_schedule = rank_annealing.optimal_rank_schedule(30000, 
                                                     hierarchy_depth=6, 
                                                     max_Q=1000, 
                                                     max_rank=100)
        with jax.default_device(jax.devices("gpu")[0]):
            XA = jnp.asarray(X)
            YB = jnp.asarray(Y)
        perm = HiRef.hiref_lr(
            XA, YB, rank_schedule, iters_per_level=100, gamma=60.0,
            rescale_cost=False, return_coupling=False
        )
    
    print('OT Finished. Starting K-Means.')
    
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
'''
