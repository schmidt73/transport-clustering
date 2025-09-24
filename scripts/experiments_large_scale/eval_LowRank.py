import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI
from scipy.optimize import linear_sum_assignment

def evaluate_factors(
    Q: np.ndarray,
    R: np.ndarray,
    yA: np.ndarray,
    yB: np.ndarray,
    g: np.ndarray | None = None,
    classes: np.ndarray | None = None,
    *,
    return_matrix: bool = False,
    include_purity: bool = False,
    eps: float = 1e-18,
):
    """
    Minimal evaluation for low-rank OT factors.

    Returns
    -------
    dict with keys:
      - 'A_AMI', 'A_ARI' : AMI/ARI between yA and argmax(Q)
      - 'B_AMI', 'B_ARI' : AMI/ARI between yB and argmax(R)
      - 'CTA'            : cross-domain class-transfer accuracy:
                           tr( rho ) / sum(rho)
      - 'classes'        : the class ids used to build the matrix
      - optionally (if flags set):
          * 'A_cluster_purity', 'B_cluster_purity' (if include_purity=True)
          * 'class_mass_matrix' (if return_matrix=True)
    Notes
    -----
    - Q, R are (n, r) soft memberships (nonnegative). g is (r,) (positive).
    - If g is None, assumes uniform g = 1/r.
    - classes defines the shared label set ordering. If None, inferred from yA∪yB.
    """
    Q = np.asarray(Q); R = np.asarray(R)
    n, r = Q.shape
    assert R.shape == (n, r), "Q and R must have shape (n, r)"

    yA = np.asarray(yA); yB = np.asarray(yB)
    if classes is None:
        classes = np.unique(np.concatenate([yA, yB]))
    classes = np.asarray(classes)
    kC = int(classes.size)

    # Hard labels from soft memberships
    zA = Q.argmax(axis=1)
    zB = R.argmax(axis=1)

    out = {
        "A_AMI": float(AMI(yA, zA)),
        "A_ARI": float(ARI(yA, zA)),
        "B_AMI": float(AMI(yB, zB)),
        "B_ARI": float(ARI(yB, zB)),
        "classes": classes,
    }

    # Build class–cluster tables via one-hot * matmul
    # Map raw labels into [0..kC-1] indices consistent with `classes`
    lab2idx = {c: i for i, c in enumerate(classes)}
    yA_idx = np.vectorize(lab2idx.get)(yA)
    yB_idx = np.vectorize(lab2idx.get)(yB)

    Qbar = _one_hot(yA_idx, kC, dtype=Q.dtype)   # (n,kC)
    Rbar = _one_hot(yB_idx, kC, dtype=R.dtype)   # (n,kC)
    
    # Coarsen according to ground-truth labels
    A_CC = Qbar.T @ Q                             # (kC, r)
    B_CC = Rbar.T @ R                             # (kC, r)
    
    # Proper cross-domain CMA from LR factors
    if g is None:
        g = np.full((r,), 1.0 / r, dtype=Q.dtype)
    g = np.asarray(g, dtype=Q.dtype)
    inv_g = 1.0 / np.maximum(g, eps)              # (r,)

    # Class–class mass matrix: M = A_CC diag(1/g) B_CC^T
    M = A_CC @ (inv_g[None, :] * B_CC.T)          # (kC, kC)
    total_mass = M.sum()
    CMA = float(np.trace(M) / (total_mass + eps))
    out["CMA"] = CMA

    if include_purity:
        # Per-side cluster purity (majority mass fraction) — optional
        A_purity = float(A_CC.max(axis=0).sum() / (Q.sum() + eps))
        B_purity = float(B_CC.max(axis=0).sum() / (R.sum() + eps))
        out["A_cluster_purity"] = A_purity
        out["B_cluster_purity"] = B_purity

    if return_matrix:
        out["class_mass_matrix"] = M

    return out

def _one_hot(labels, num_classes=None, dtype=np.float64):
    y = np.asarray(labels)
    if num_classes is None:
        num_classes = int(y.max()) + 1
    Q = np.zeros((y.shape[0], num_classes), dtype=dtype)
    Q[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return Q

def to_np(x, dtype=np.float64):
    if x is None:
        return None
    try:
        import jax.numpy as jnp
        if isinstance(x, jnp.ndarray):
            x = np.array(x)
    except Exception:
        pass
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(x, dtype=dtype)
    return x


