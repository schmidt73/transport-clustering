import torch
import random
from typing import Tuple
import jax
import jax.numpy as jnp
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)


def _cdist(X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    """
    Pairwise Euclidean distances between rows of X (n x d) and Y (m x d).
    Returns (n x m) matrix.
    """
    # (n,1,d) - (1,m,d) -> (n,m,d) -> sum over d -> (n,m) -> sqrt
    return jnp.sqrt(jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1))

def compute_lr_sqeuclidean_factors(X_s: jnp.ndarray,
                                   X_t: jnp.ndarray,
                                   rescale_cost: bool = False,
                                   return_scale: bool = False):
    """
    Returns (A, B) such that C ≈ A @ B.T for squared Euclidean cost.
    A = [||x||^2, 1, -2x],  B = [1, ||y||^2, y]
    Shapes: A: (n, d+2), B: (m, d+2)
    """
    ns, d = X_s.shape
    nt, _ = X_t.shape
    A = jnp.concatenate(
        (jnp.sum(X_s**2, axis=1, keepdims=True), jnp.ones((ns,1), X_s.dtype), -2.0*X_s),
        axis=1,
    )
    B = jnp.concatenate(
        (jnp.ones((nt,1), X_t.dtype), jnp.sum(X_t**2, axis=1, keepdims=True), X_t),
        axis=1,
    )
    if rescale_cost:
        # Optional mild rescaling (mirrors your torch code idea)
        sA = jnp.sqrt(jnp.maximum(jnp.max(jnp.abs(A)), 1.0))
        sB = jnp.sqrt(jnp.maximum(jnp.max(jnp.abs(B)), 1.0))
        A = A / sA
        B = B / sB
        if return_scale:
            return A, B, sA, sB
    return A, B

def low_rank_distance_factorization_jax(
    X1: jnp.ndarray,
    X2: jnp.ndarray,
    r: int,
    eps: float,
    key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JAX version of low_rank_distance_factorization.

    Args:
        X1: (n, d) array
        X2: (m, d) array
        r: target rank
        eps: accuracy parameter (t = int(r/eps))
        key: jax PRNGKey for randomness

    Returns:
        V: (n, r) array (float64)
        U_t_T: (r, m) array (float64)   # i.e., U_t.T in the torch code
    """
    n = X1.shape[0]
    m = X2.shape[0]

    # Indyk '19
    t = int(r / eps)  # same definition as original

    key, k_i, k_j = jax.random.split(key, 3)
    i_star = jax.random.randint(k_i, shape=(), minval=0, maxval=n)
    j_star = jax.random.randint(k_j, shape=(), minval=0, maxval=m)

    # Build p as in the torch code
    d1 = _cdist(X1, X2[j_star : j_star + 1]) ** 2            # (n,1)
    d2 = (_cdist(X1[i_star : i_star + 1], X2[j_star : j_star + 1]) ** 2)  # (1,1)
    d3 = jnp.sum(_cdist(X1[i_star : i_star + 1], X2)) / m     # scalar

    # Broadcast to (n,1), then take [:,0] and square
    p_vec = (d1 + d2 + d3)[:, 0] ** 2                         # (n,)
    p_sum = jnp.sum(p_vec)
    # If p_sum == 0 (degenerate), fall back to uniform to avoid NaNs
    p_dist = jnp.where(p_sum > 0, p_vec / p_sum, jnp.full_like(p_vec, 1.0 / n))

    key, k_p = jax.random.split(key)
    indices_p = jax.random.choice(k_p, n, shape=(t,), replace=True, p=p_dist)
    X1_t = X1[indices_p, :]                                   # (t, d)

    # Frieze '04
    P_t = jnp.sqrt(p_vec[indices_p] * t)                      # (t,)
    S = _cdist(X1_t, X2) / P_t[:, None]                       # (t, m)

    S_frob_sq = jnp.sum(S**2)
    row_norms_sq = jnp.sum(S**2, axis=0)                      # (m,)
    q_raw = row_norms_sq / jnp.where(S_frob_sq > 0, S_frob_sq, 1.0)
    q_sum = jnp.sum(q_raw)
    q = jnp.where(q_sum > 0, q_raw / q_sum, jnp.full_like(q_raw, 1.0 / m))

    key, k_q = jax.random.split(key)
    indices_q = jax.random.choice(k_q, m, shape=(t,), replace=True, p=q)  # (t,)
    S_t = S[:, indices_q]                                     # (t, t)

    Q_t = jnp.sqrt(q[indices_q] * t)                          # (t,)
    W = S_t / Q_t[None, :]                                    # (t, t)

    U, Sig, Vh = jnp.linalg.svd(W, full_matrices=False)       # all (t, t)
    F = U[:, :r]                                              # (t, r)

    denom = jnp.linalg.norm(W.T @ F)                          # scalar
    # Protect against denom == 0
    denom = jnp.where(denom > 0, denom, 1.0)
    U_t = (S.T @ F) / denom                                   # (m, r)

    # Chen & Price '17
    key, k_idx = jax.random.split(key)
    indices = jax.random.choice(k_idx, m, shape=(t,), replace=True)  # (t,)
    X2_t = X2[indices, :]                                     # (t, d)
    D_t = _cdist(X1, X2_t) / jnp.sqrt(t)                      # (n, t)

    Q = U_t.T @ U_t                                           # (r, r)
    Uq, Sig_q, Vh_q = jnp.linalg.svd(Q, full_matrices=False)  # (r, r)
    # Match torch: U = U / Sig  (row-wise divide due to broadcasting)
    Uq = Uq / Sig_q                                           # (r, r)

    U_tSub = U_t[indices, :].T                                # (r, t)
    B = (Uq.T @ U_tSub) / jnp.sqrt(t)                         # (r, t)
    A = jnp.linalg.inv(B @ B.T)                               # (r, r)
    Z = (A @ B) @ D_t.T                                       # (r, n)
    V = Z.T @ Uq                                              # (n, r)

    return V.astype(jnp.float64), U_t.T.astype(jnp.float64)

'''
-----Torch Versions Below-----
'''
def compute_lr_sqeuclidean_matrix(X_s,
                                  X_t,
                                  rescale_cost,
                                  device=None,
                                  dtype=None):
    """
    Adapted from "Section 3.5, proposition 1" in Scetbon, M., Cuturi, M., & Peyré, G. (2021).
    A function for computing a low-rank factorization of a squared Euclidean distance matrix.
    """
    dtype, device = X_s.dtype, X_s.device
    ns, dim = X_s.shape
    nt, _ = X_t.shape
    # First low rank decomposition of the cost matrix (M1)
    # Compute sum of squares for each source sample
    sum_Xs_sq = torch.sum(X_s ** 2, dim=1).reshape(ns, 1)  # Shape: (ns, 1)
    ones_ns = torch.ones((ns, 1), device=device, dtype=dtype)  # Shape: (ns, 1)
    neg_two_Xs = -2 * X_s  # Shape: (ns, dim)
    M1 = torch.cat((sum_Xs_sq, ones_ns, neg_two_Xs), dim=1)  # Shape: (ns, dim + 2)
    # Second low rank decomposition of the cost matrix (M2)
    ones_nt = torch.ones((nt, 1), device=device, dtype=dtype)  # Shape: (nt, 1)
    sum_Xt_sq = torch.sum(X_t ** 2, dim=1).reshape(nt, 1)  # Shape: (nt, 1)
    Xt = X_t  # Shape: (nt, dim)
    M2 = torch.cat((ones_nt, sum_Xt_sq, Xt), dim=1)  # Shape: (nt, dim + 2)
    if rescale_cost:
        # Compute the maximum value in M1 and M2 for rescaling
        max_M1 = torch.max(M1)
        max_M2 = torch.max(M2)
        # Avoid division by zero
        if max_M1 > 0:
            M1 = M1 / torch.sqrt(max_M1)
        if max_M2 > 0:
            M2 = M2 / torch.sqrt(max_M2)
    return (M1, M2.T)



def low_rank_distance_factorization(X1, X2, r, eps, device='cpu', dtype=torch.float64):
    n = X1.shape[0]
    m = X2.shape[0]
    '''
    Indyk '19
    '''
    # low-rank distance matrix factorization of Bauscke, Indyk, Woodruff
    t = int(r/eps) # this is poly(1/eps, r) in general -- this t might not achieve the correct bound tightly
    i_star = random.randint(0, n-1)
    j_star = random.randint(0, m-1)
    # Define probabilities of sampling
    p = (torch.cdist(X1, X2[j_star][None,:])**2 \
            + torch.cdist(X1[i_star,:][None,:], X2[j_star,:][None,:])**2 \
                    + (torch.sum(torch.cdist(X1[i_star][None,:], X2))/m) )[:,0]**2
    p_dist = (p / p.sum())
    # Use random choice to sample rows
    indices_p = torch.from_numpy(np.random.choice(n, size=(t), p=p_dist.cpu().numpy())).to(device)
    X1_t = X1[indices_p, :]
    '''
    Frieze '04
    '''
    P_t = torch.sqrt(p[indices_p]*t)
    S = torch.cdist(X1_t, X2)/P_t[:, None] # t x m
    # Define probabilities of sampling by row norms
    q = torch.norm(S, dim=0)**2 / torch.norm(S)**2 # m x 1
    q_dist = (q / q.sum())
    # Use random choice to sample rows
    indices_q = torch.from_numpy(np.random.choice(m, size=(t), p=q_dist.cpu().numpy())).to(device)
    S_t = S[:, indices_q] # t x t
    Q_t = torch.sqrt(q[indices_q]*t)
    W = S_t[:, :] / Q_t[None, :]
    # Find U
    U, Sig, Vh = torch.linalg.svd(W) # t x t for all
    F = U[:, :r] # t x r
    # U.T for the final return
    U_t = (S.T @ F) / torch.norm(W.T @ F) # m x r
    '''
    Chen & Price '17
    '''
    # Find V for the final return
    indices = torch.from_numpy(np.random.choice(m, size=(t))).to(device)
    X2_t = X2[indices, :] # t x dim
    D_t = torch.cdist(X1, X2_t) / np.sqrt(t) # n x t
    Q = U_t.T @ U_t # r x r
    U, Sig, Vh = torch.linalg.svd(Q)
    U = U / Sig # r x r
    U_tSub = U_t[indices, :].T # t x r
    B = U.T @ U_tSub / np.sqrt(t) # (r x r) (r x t)
    A = torch.linalg.inv(B @ B.T)
    Z = ((A @ B) @ D_t.T) # (r x r) (r x t) (t x n)
    V = Z.T @ U
    return V.double(), U_t.T.double()



