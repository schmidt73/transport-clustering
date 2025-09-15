import jax
import jax.numpy as jnp
from jax import lax
from typing import List, Tuple, Optional

IndexArray = jnp.ndarray  # 1-D integer array

def _cost_matrix(f, g, G, eps):
    return -(G - f[:, None] - g[None, :]) / eps

@jax.jit
def log_sinkhorn_project(G, a, b, eps, max_iter=200, f=None, g=None,
                         recenter_every=30, tol=1e-12):
    """
    Project the positive kernel exp(-(G)/eps) onto {P: P 1 = a, P^T 1 = b}
    in log-domain using duals (f,g). Reuses duals if provided.
    Returns (P, f, g).
    Shapes: G (n,m), a (n,), b (m,)
    """
    n, m = G.shape
    if f is None: f = jnp.zeros((n,), G.dtype)
    if g is None: g = jnp.zeros((m,), G.dtype)
    
    def body_fun(k, carry):
        f_k, g_k = carry
        # update f
        M = _cost_matrix(f_k, g_k, G, eps)
        f_next = f_k + eps * (jnp.log(a) - jax.scipy.special.logsumexp(M, axis=1))
        # update g
        M = _cost_matrix(f_next, g_k, G, eps)
        g_next = g_k + eps * (jnp.log(b) - jax.scipy.special.logsumexp(M, axis=0))
        # recenter (gauge invariance) every few iters
        def recenter(pair):
            ff, gg = pair
            alpha = jnp.mean(ff)
            return ff - alpha, gg + alpha
        f_next, g_next = lax.cond((k % recenter_every) == 0,
                                  recenter, lambda x: x, (f_next, g_next))
        return f_next, g_next
    
    f_new, g_new = lax.fori_loop(0, max_iter, body_fun, (f, g))
    # assemble primal from final duals
    P = jnp.exp(_cost_matrix(f_new, g_new, G, eps))
    return P, f_new, g_new

def initialize_couplings(a, b, g, gamma, max_iter=50, key=None):
    # Initialization of Proposition F.1 in "Low-Rank Optimal Transport through Factor Relaxation with Latent Coupling"
    if key is None:
        key = jax.random.PRNGKey(0)
    kQ, kR = jax.random.split(key, 2)
    N1, N2, r = a.shape[0], b.shape[0], g.shape[0]
    dtype = a.dtype
    
    Cq = jax.random.uniform(kQ, (N1, r), dtype=dtype)
    Cr = jax.random.uniform(kR, (N2, r), dtype=dtype)
    eps = 1.0 / gamma
    
    Q, fQ, gQ_dual = log_sinkhorn_project(Cq, a, g, eps, max_iter=max_iter)
    R, fR, gR_dual = log_sinkhorn_project(Cr, b, g, eps, max_iter=max_iter)
    return Q, R

# =========================
# Two-sided LR-OT with Sinkhorn projection and dual reuse
# =========================
def _loss_lr_two(Q: jnp.ndarray, R: jnp.ndarray,
                 A: jnp.ndarray, B: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    SA = Q.T @ A          # (r,k)
    RB = R.T @ B          # (r,k)
    return jnp.sum(jnp.sum(RB * SA, axis=1) / jnp.clip(g, 1e-18))

def lrot_lr(A: jnp.ndarray,
            B: jnp.ndarray,
            r: int,
            iters: int = 60,
            gamma: float = 60.0,
            key=None):
    """
    Balanced low-rank OT (uniform g). Each iteration:
      1) compute grads wrt Q, R
      2) scale step via normalized gamma_k
      3) project with log-domain Sinkhorn on G = grad - (1/gamma_k) * log(current),
         reusing duals for speed/stability.
    """
    n, m = A.shape[0], B.shape[0]
    a = jnp.full((n,), 1.0 / n, A.dtype)
    b = jnp.full((m,), 1.0 / m, B.dtype)
    g = jnp.full((r,), 1.0 / r, A.dtype)

    # 0) full-rank init via balanced Sinkhorn
    Q, R = initialize_couplings(a, b, g, gamma, max_iter=50, key=key)
    
    # duals (warm-start these across iters)
    fQ = jnp.zeros((n,), A.dtype); gQ_dual = jnp.zeros((r,), A.dtype)
    fR = jnp.zeros((m,), B.dtype); gR_dual = jnp.zeros((r,), B.dtype)

    grad_Q = jax.grad(_loss_lr_two, argnums=0)
    grad_R = jax.grad(_loss_lr_two, argnums=1)
    
    def step(carry, _):
        Qc, Rc, fQ_c, gQ_c, fR_c, gR_c = carry
        # 1) grads
        gq = grad_Q(Qc, Rc, A, B, g)
        gr = grad_R(Qc, Rc, A, B, g)
        
        # 2) normalized step size
        norm = jnp.maximum(jnp.max(jnp.abs(gq)), jnp.max(jnp.abs(gr)))
        gamma_k = gamma / jnp.clip(norm, 1e-18)
        eps = 1.0 / gamma_k

        # 3) log-Sinkhorn projection with reused duals
        #    (mirror descent step encoded via the "costs" below)
        GQ = gq - (1.0 / gamma_k) * jnp.log(jnp.clip(Qc, 1e-32))
        GR = gr - (1.0 / gamma_k) * jnp.log(jnp.clip(Rc, 1e-32))
        
        Qn, fQ_n, gQ_n = log_sinkhorn_project(GQ, a, g, eps, max_iter=10, f=fQ_c, g=gQ_c)
        Rn, fR_n, gR_n = log_sinkhorn_project(GR, b, g, eps, max_iter=10, f=fR_c, g=gR_c)

        return (Qn, Rn, fQ_n, gQ_n, fR_n, gR_n), None

    (Q, R, _, _, _, _), _ = lax.scan(step, (Q, R, fQ, gQ_dual, fR, gR_dual),
                                     xs=None, length=iters)
    return Q, R

# ===== Low-rank squared Euclidean factors: C = A B^T =====
def lr_sqeuclidean_factors(X: jnp.ndarray, Y: jnp.ndarray, rescale: bool = False):
    n, d = X.shape
    A = jnp.concatenate([jnp.sum(X**2, axis=1, keepdims=True),
                         jnp.ones((n,1), X.dtype),
                         -2.0 * X], axis=1)              # (n, d+2)
    B = jnp.concatenate([jnp.ones((n,1), Y.dtype),
                         jnp.sum(Y**2, axis=1, keepdims=True),
                         Y], axis=1)                     # (n, d+2)
    if rescale:
        sA = jnp.sqrt(jnp.maximum(jnp.max(jnp.abs(A)), 1.0))
        sB = jnp.sqrt(jnp.maximum(jnp.max(jnp.abs(B)), 1.0))
        A = A / sA
        B = B / sB
    return A, B

# ===== Stabilizer for one-hot-ish seeds (optional) =====
def stabilize_Q(Q: jnp.ndarray, lam: float = 0.1) -> jnp.ndarray:
    b = Q.sum(axis=1, keepdims=True)
    g = Q.sum(axis=0, keepdims=True)
    eps = (b @ g) / jnp.clip(g.sum(), 1e-18)
    return (1.0 - lam) * Q + lam * eps

# ===== Capacity-aware Top-K assignment (hard split) =====
def _split_by_capacity(scores: jnp.ndarray, capacity: int) -> List[jnp.ndarray]:
    """
    scores: (N, r) soft membership; returns r disjoint index arrays,
    each of size 'capacity', by greedy top-k per column.
    """
    N, r = scores.shape
    remaining = jnp.arange(N)
    taken_mask = jnp.zeros((N,), dtype=bool)
    out = []
    for z in range(r):
        # mask out already taken
        s = jnp.where(taken_mask, -jnp.inf, scores[:, z])
        vals, idxs = lax.top_k(s, capacity)
        out.append(idxs)
        taken_mask = taken_mask.at[idxs].set(True)
    return out  # list of r arrays (capacity,)

# ===== HiRef (serial), LR everywhere, short & sweet =====
def hiref_lr(X: jnp.ndarray,
             Y: jnp.ndarray,
             rank_schedule: List[int],
             base_rank: int = 1,
             iters_per_level: int = 60,
             gamma: float = 60.0,
             rescale_cost: bool = False,
             return_coupling: bool = False):
    """
    Hierarchical Refinement with LR OT subproblems (balanced, uniform g).
    Returns list of (idxX, idxY) clusters or a sparse permutation coupling.
    """
    n = X.shape[0]
    A_full, B_full = lr_sqeuclidean_factors(X, Y, rescale=rescale_cost)

    # frontier of index-pairs
    frontier = [(jnp.arange(n), jnp.arange(n))]
    for level, r in enumerate(rank_schedule, start=1):
        # print(f'Refinement Level {level}')
        new_frontier = []
        for f, (idxX, idxY) in enumerate(frontier):
            # print(f'Frontier {f}/{len(frontier)}')
            NX = idxX.size
            NY = idxY.size
            if min(NX, NY) <= base_rank:
                new_frontier.append((idxX, idxY))
                continue
            
            # sub-factors (no dense sub-costs)
            A = A_full[idxX, :]    # (NX, k)
            B = B_full[idxY, :]    # (NY, k)
            
            # solve LR-OT (two-sided)
            Q, R = lrot_lr(A, B, r=r, iters=iters_per_level, gamma=gamma)
            
            # Capacity per child (greedy; replace with balanced allocator if desired)
            capX = int(NX // r)
            capY = int(NY // r)
            cap = min(capX, capY)
            if cap == 0:
                # Fallback to base-rank split if tiny residue
                new_frontier.append((idxX, idxY))
                continue

            X_parts = _split_by_capacity(Q, cap)
            Y_parts = _split_by_capacity(R, cap)

            for z in range(r):
                new_frontier.append((idxX[X_parts[z]], idxY[Y_parts[z]]))
        frontier = new_frontier

    # Done: either return clusters or assemble sparse permutation
    if not return_coupling:
        return frontier

    # Sparse 0/1 permutation from clusters (base_rank assumed 1 at leaves)
    P_rows = []
    P_cols = []
    for idxX, idxY in frontier:
        # if base_rank>1 you could distribute uniformly inside the block
        # here assume leaves are pairs
        for i, j in zip(idxX.tolist(), idxY.tolist()):
            P_rows.append(i); P_cols.append(j)
    data = jnp.ones((len(P_rows),), X.dtype)
    # Dense (n x n) is optional; keep it small:
    P = jnp.zeros((n, n), X.dtype).at[(jnp.array(P_rows), jnp.array(P_cols))].set(1.0)
    return P / n

def compute_ot_cost(
    monge_clus: List[Tuple[IndexArray, IndexArray]],
    X: jnp.ndarray,
    Y: jnp.ndarray,
    C: Optional[jnp.ndarray] = None,
    sq_euclidean: bool = True,
) -> jnp.ndarray:
    """
    JAX version: compute OT cost from a list of matched index blocks.

    Args:
      monge_clus: list of (idxX, idxY) pairs; each is a 1-D array of equal length.
      X, Y: point clouds with shape (n, d).
      C: optional full cost matrix (n, n). If given, it's used directly.
      sq_euclidean: if C is None, choose squared Euclidean or Euclidean.

    Returns:
      Scalar cost normalized by n (i.e., average per point).
    """
    n = X.shape[0]
    total = jnp.array(0.0, dtype=X.dtype)

    for idxX, idxY in monge_clus:
        ix = jnp.asarray(idxX, dtype=jnp.int32)
        iy = jnp.asarray(idxY, dtype=jnp.int32)
        # Ensure 1–1 inside each block
        if ix.size != iy.size:
            raise ValueError("Each block must be 1–1: idxX and idxY must have equal length.")
        if ix.size == 0:
            continue

        if C is not None:
            # Elementwise gather then sum: C[ix[k], iy[k]]
            total = total + jnp.sum(C[(ix, iy)])
        else:
            # Direct distances for matched pairs
            diff = X[ix] - Y[iy]                  # (k, d)
            if sq_euclidean:
                total = total + jnp.sum(jnp.sum(diff * diff, axis=1))
            else:
                total = total + jnp.sum(jnp.linalg.norm(diff, axis=1))
    
    return total / jnp.array(n, dtype=X.dtype)




