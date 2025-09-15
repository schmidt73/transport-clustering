# hiref_jax.py
# Self-contained HiRef (Option A) — JAX, GPU-friendly.
# Heavy kernels are JIT-compiled; ragged frontier handled on host.
from functools import partial
from typing import List, Tuple, Optional
import jax
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray
IndexArray = jnp.ndarray  # 1-D int array


# =========================
#  Utilities: LR sq-Euclidean factors  C ≈ A @ B^T
# =========================
def lr_sqeuclidean_factors(X: Array, Y: Array, rescale: bool = False) -> Tuple[Array, Array]:
    n, d = X.shape
    A = jnp.concatenate(
        [jnp.sum(X**2, axis=1, keepdims=True), jnp.ones((n, 1), X.dtype), -2.0 * X],
        axis=1,
    )  # (n, d+2)
    B = jnp.concatenate(
        [jnp.ones((n, 1), Y.dtype), jnp.sum(Y**2, axis=1, keepdims=True), Y],
        axis=1,
    )  # (n, d+2)
    if rescale:
        sA = jnp.sqrt(jnp.maximum(jnp.max(jnp.abs(A)), 1.0))
        sB = jnp.sqrt(jnp.maximum(jnp.max(jnp.abs(B)), 1.0))
        A = A / sA
        B = B / sB
    return A, B


# =========================
#  Log-domain Sinkhorn (balanced), with dual reuse
# =========================
def _cost_matrix(f: Array, g: Array, G: Array, eps: float) -> Array:
    return -(G - f[:, None] - g[None, :]) / eps

@jax.jit
def log_sinkhorn_project(
    G: Array, a: Array, b: Array, eps: float, max_iter: int = 200,
    f: Optional[Array] = None, g: Optional[Array] = None,
    recenter_every: int = 30,
) -> Tuple[Array, Array, Array]:
    """Project exp(-G/eps) to {P : P1=a, P^T1=b} via log-domain Sinkhorn. Returns (P, f, g)."""
    n, m = G.shape
    if f is None: f = jnp.zeros((n,), G.dtype)
    if g is None: g = jnp.zeros((m,), G.dtype)

    log_a = jnp.log(a)
    log_b = jnp.log(b)

    def body(k, carry):
        f_k, g_k = carry
        M = _cost_matrix(f_k, g_k, G, eps)
        f_n = f_k + eps * (log_a - jax.scipy.special.logsumexp(M, axis=1))
        M = _cost_matrix(f_n, g_k, G, eps)
        g_n = g_k + eps * (log_b - jax.scipy.special.logsumexp(M, axis=0))
        def recenter(pair):
            ff, gg = pair
            alpha = jnp.mean(ff)
            return ff - alpha, gg + alpha
        f_n, g_n = lax.cond((k % recenter_every) == 0, recenter, lambda x: x, (f_n, g_n))
        return f_n, g_n

    f_new, g_new = lax.fori_loop(0, max_iter, body, (f, g))
    P = jnp.exp(_cost_matrix(f_new, g_new, G, eps))
    return P, f_new, g_new


# =========================
#  Initialization: full-rank Q,R via Sinkhorn
# =========================
def initialize_couplings(a: Array, b: Array, g: Array, gamma: float,
                         max_iter: int = 50, key: Optional[jax.Array] = None) -> Tuple[Array, Array]:
    if key is None:
        key = jax.random.PRNGKey(0)
    kQ, kR = jax.random.split(key, 2)
    N1, N2, r = a.shape[0], b.shape[0], g.shape[0]
    Cq = jax.random.uniform(kQ, (N1, r), dtype=a.dtype)
    Cr = jax.random.uniform(kR, (N2, r), dtype=b.dtype)
    eps = 1.0 / gamma
    Q, _, _ = log_sinkhorn_project(Cq, a, g, eps, max_iter=max_iter)
    R, _, _ = log_sinkhorn_project(Cr, b, g, eps, max_iter=max_iter)
    return Q, R


# =========================
#  Two-sided LR-OT (uniform g), mirror descent + Sinkhorn projection
# =========================
def _loss_lr_two(Q: Array, R: Array, A: Array, B: Array, g: Array) -> Array:
    SA = Q.T @ A  # (r,k)
    RB = R.T @ B  # (r,k)
    return jnp.sum(jnp.sum(RB * SA, axis=1) / jnp.clip(g, 1e-18))

grad_Q = jax.grad(_loss_lr_two, argnums=0)
grad_R = jax.grad(_loss_lr_two, argnums=1)

@jax.jit
def _md_sinkhorn_step(Q: Array, R: Array, A: Array, B: Array,
                      a: Array, b: Array, g: Array, gamma: float,
                      fQ: Array, gQd: Array, fR: Array, gRd: Array) -> Tuple[Tuple, None]:
    gq = grad_Q(Q, R, A, B, g)
    gr = grad_R(Q, R, A, B, g)
    norm = jnp.maximum(jnp.max(jnp.abs(gq)), jnp.max(jnp.abs(gr)))
    gamma_k = gamma / jnp.clip(norm, 1e-18)
    eps = 1.0 / gamma_k
    
    GQ = gq - (1.0 / gamma_k) * jnp.log(jnp.clip(Q, 1e-32))
    GR = gr - (1.0 / gamma_k) * jnp.log(jnp.clip(R, 1e-32))
    
    Qn, fQn, gQn = log_sinkhorn_project(GQ, a, g, eps, max_iter=15, f=fQ, g=gQd)
    Rn, fRn, gRn = log_sinkhorn_project(GR, b, g, eps, max_iter=15, f=fR, g=gRd)
    return (Qn, Rn, fQn, gQn, fRn, gRn), None

@partial(jax.jit, static_argnums=(2, 3))
def lrot_lr(A, B, r, iters=60, gamma=60.0, key=None):
    n, m = A.shape[0], B.shape[0]
    a = jnp.full((n,), 1.0 / n, A.dtype)
    b = jnp.full((m,), 1.0 / m, B.dtype)
    g = jnp.full((r,), 1.0 / r, A.dtype)

    Q0, R0 = initialize_couplings(a, b, g, gamma, max_iter=50, key=key)
    fQ0 = jnp.zeros((n,), A.dtype); gQ0 = jnp.zeros((r,), A.dtype)
    fR0 = jnp.zeros((m,), B.dtype); gR0 = jnp.zeros((r,), B.dtype)

    def scan_body(carry, _):
        Qc, Rc, fQc, gQc, fRc, gRc = carry
        # call step with the *correct* ordering
        (Qn, Rn, fQn, gQn, fRn, gRn), _ = _md_sinkhorn_step(
            Qc, Rc, A, B, a, b, g, gamma, fQc, gQc, fRc, gRc
        )
        return (Qn, Rn, fQn, gQn, fRn, gRn), None

    (Q, R, _, _, _, _), _ = lax.scan(scan_body,
                                     (Q0, R0, fQ0, gQ0, fR0, gR0),
                                     xs=None, length=iters)
    return Q, R


# =========================
#  On-device split-by-capacity (hard split), no Python loops
# =========================
@partial(jax.jit, static_argnames=('cap',))
def split_by_capacity_device(scores: jnp.ndarray, cap: int) -> jnp.ndarray:
    # scores: (N, r) -> top `cap` row indices per column
    # Returns indices of shape (r, cap), dtype int32
    _, idx = lax.top_k(scores.T, k=cap)   # idx: (r, cap) in [0, N)
    return idx.astype(jnp.int32)
'''
@partial(jax.jit, static_argnames=('cap',))
def split_by_capacity_device(scores: jnp.ndarray, cap: int) -> jnp.ndarray:
    # scores: (N, r)
    N, r = scores.shape
    out = jnp.zeros((r, cap), dtype=jnp.int32)

    def body(t, carry):
        live_scores, out = carry
        # winners per column from current live_scores
        winners = jnp.argmax(live_scores, axis=0).astype(jnp.int32)  # (r,)
        out = out.at[:, t].set(winners)
        # set entire selected rows to -inf in one batched scatter update
        rows = winners  # length r, may repeat; repetition is fine (idempotent)
        live_scores = live_scores.at[rows].set(-jnp.inf)
        return (live_scores, out)

    live0 = scores  # no where() each round
    (_, out) = jax.lax.fori_loop(0, cap, body, (live0, out))
    return out'''

'''
@partial(jax.jit, static_argnames=('cap',))
def split_by_capacity_device(scores: jnp.ndarray, cap: int) -> jnp.ndarray:
    N, r = scores.shape
    taken = jnp.zeros((N,), dtype=bool)
    out = jnp.zeros((r, cap), dtype=jnp.int32)

    def body(t, carry):
        taken, out = carry
        masked = jnp.where(taken[:, None], -jnp.inf, scores)  # (N, r)
        winners = jnp.argmax(masked, axis=0).astype(jnp.int32)  # (r,)
        out = out.at[:, t].set(winners)
        taken = taken.at[winners].set(True)
        return taken, out

    taken, out = jax.lax.fori_loop(0, cap, body, (taken, out))
    return out'''

# =========================
#  One block (host ragged slicing, kernels jitted)
# =========================
def _per_block(A_full: Array, B_full: Array,
               idxX: IndexArray, idxY: IndexArray,
               r: int, iters: int, gamma: float):
    Ai = A_full[idxX, :]  # ragged on host
    Bi = B_full[idxY, :]
    Q, R = lrot_lr(Ai, Bi, r=r, iters=iters, gamma=gamma)  # heavy kernels on device
    cap = int(min(int(idxX.size) // r, int(idxY.size) // r))
    if cap <= 0:
        return None
    Xi = split_by_capacity_device(Q, cap)  # cap is static int now
    Yi = split_by_capacity_device(R, cap)
    return Xi, Yi, cap


# =========================
#  HiRef (Option A): frontier on host, all compute jitted/on-GPU
# =========================
def hiref_lr_fast(
    X: Array,
    Y: Array,
    rank_schedule: List[int],
    base_rank: int = 1,
    iters_per_level: int = 60,
    gamma: float = 60.0,
    rescale_cost: bool = False,
    return_coupling: bool = False,
):
    n = int(X.shape[0])
    A_full, B_full = lr_sqeuclidean_factors(X, Y, rescale=rescale_cost)

    frontier: List[Tuple[IndexArray, IndexArray]] = [(jnp.arange(n), jnp.arange(n))]

    for r in rank_schedule:
        work_blocks, leaf_blocks = [], []
        for idxX, idxY in frontier:
            if min(int(idxX.size), int(idxY.size)) <= base_rank:
                leaf_blocks.append((idxX, idxY))
            else:
                work_blocks.append((idxX, idxY))

        new_frontier = list(leaf_blocks)
        for idxX, idxY in work_blocks:
            out = _per_block(A_full, B_full, idxX, idxY, r, iters_per_level, gamma)
            if out is None:
                new_frontier.append((idxX, idxY))
                continue
                
            Xi, Yi, cap = out
            Xi_g = idxX[Xi]  # (r, cap) device gather once
            Yi_g = idxY[Yi]  # (r, cap) device gather once
            for z in range(r):
                new_frontier.append((Xi_g[z], Yi_g[z]))
                
        frontier = new_frontier

    if not return_coupling:
        return frontier

    # assemble dense permutation; leaves are assumed size-1
    P = jnp.zeros((n, n), X.dtype)
    for idxX, idxY in frontier:
        if idxX.size == 1 and idxY.size == 1:
            P = P.at[(int(idxX[0]), int(idxY[0]))].set(1.0)
        else:
            # if a leaf is larger than 1, spread uniformly (optional)
            size = int(min(idxX.size, idxY.size))
            P = P.at[(idxX[:size, None], idxY[:size][None, :])].set(1.0 / size)
    return P / n


# =========================
#  OT cost from leaf pairs (1-1), works with or without C
# =========================
def compute_ot_cost(
    monge_clus: List[Tuple[IndexArray, IndexArray]],
    X: Array,
    Y: Array,
    C: Optional[Array] = None,
    sq_euclidean: bool = True,
) -> Array:
    n = X.shape[0]
    total = jnp.array(0.0, dtype=X.dtype)
    for idxX, idxY in monge_clus:
        ix = jnp.asarray(idxX, dtype=jnp.int32)
        iy = jnp.asarray(idxY, dtype=jnp.int32)
        if ix.size != iy.size:
            raise ValueError("Each block must be 1–1.")
        if ix.size == 0:
            continue
        if C is not None:
            total = total + jnp.sum(C[(ix, iy)])
        else:
            diff = X[ix] - Y[iy]
            if sq_euclidean:
                total = total + jnp.sum(jnp.sum(diff * diff, axis=1))
            else:
                total = total + jnp.sum(jnp.linalg.norm(diff, axis=1))
    return total / jnp.array(n, dtype=X.dtype)


