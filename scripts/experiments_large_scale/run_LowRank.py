from typing import Dict, Tuple, Optional
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import jax.numpy as jnp
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI
import os
import sys
sys.path.insert(0, '../src')
import monge_rotate_lr as mr_lr
from loguru import logger
import gc
import random
import jax
import jax.numpy as jnp
import jax.random as jr

'''
def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
'''
def seed_everything(seed: int = 42):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def get_device(device_str: str) -> torch.device:
    if device_str.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def make_generators(seed: int = 42, device: str = 'cpu'):
    gen_cpu = torch.Generator(device='cpu'); gen_cpu.manual_seed(seed)
    gen_cuda = None
    if device.startswith('cuda'):
        gen_cuda = torch.Generator(device=device); gen_cuda.manual_seed(seed)
    return gen_cpu, gen_cuda

def build_model(model_name: str, device, weights_path: str | None = None):
    model_name = model_name.lower()

    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights=None)  # build w/o weights
        embed_dim, size = 512, 224
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(weights=None)
        embed_dim, size = 2048, 224
    elif model_name == "inception_v3":
        model = torchvision.models.inception_v3(weights=None, aux_logits=False)
        embed_dim, size = 2048, 299
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
        # handle wrappers like {'state_dict': ...} or 'module.' prefixes
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)  # load full model (with fc)

    # now remove classifier to get embeddings
    model.fc = nn.Identity()
    model.eval().to(device)

    # transforms (ImageNet norm)
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((size, size), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return model, embed_dim, transform


def get_transforms(model_name: str) -> T.Compose:
    model_name = model_name.lower()
    if model_name == "inception_v3":
        size = 299
        weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
    elif model_name in ("resnet18", "resnet50"):
        size = 224
        weights = (
            torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            if model_name == "resnet18"
            else torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
    else:
        size = 224
        weights = None

    if weights is not None:
        preprocess = weights.transforms(antialias=True)
        return T.Compose([
            T.Resize((size, size), antialias=True),
            preprocess,
        ])
    else:
        return T.Compose([
            T.Resize((size, size), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def load_cifar10(root: str, transform: T.Compose, download: bool = True):
    ds_train = torchvision.datasets.CIFAR10(root=root, train=True, transform=transform, download=download)
    ds_test = torchvision.datasets.CIFAR10(root=root, train=False, transform=transform, download=download)
    return torch.utils.data.ConcatDataset([ds_train, ds_test])


def extract_embeddings(dataset, model: nn.Module, device: torch.device, batch_size: int = 256, num_workers: int = 4):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                images, labels = batch
            else:
                images, labels = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            feats = model(images)
            feats = feats.detach().cpu().numpy()
            all_feats.append(feats)
            all_labels.append(labels.numpy())
    feats_arr = np.concatenate(all_feats, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    return feats_arr, labels_arr


def save_embeddings(out_path: str, feats: np.ndarray, labels: np.ndarray, meta: Dict):
    np.savez_compressed(out_path, feats=feats, labels=labels, meta=json.dumps(meta))


def load_embeddings(path: str):
    data = np.load(path, allow_pickle=True)
    feats = data["feats"]
    labels = data["labels"]
    meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
    return feats, labels, meta


def stratified_equal_halves(labels: np.ndarray, seed: int = 42):
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    A_idx, B_idx = [], []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        n_c = len(idx_c)
        if n_c % 2 == 1:
            idx_c = idx_c[:-1]  # drop one to make even
            n_c -= 1
        half = n_c // 2
        A_idx.append(idx_c[:half])
        B_idx.append(idx_c[half:])

    A_idx = np.concatenate(A_idx)
    B_idx = np.concatenate(B_idx)
    rng.shuffle(A_idx)
    rng.shuffle(B_idx)
    assert len(A_idx) == len(B_idx)
    return A_idx, B_idx

import os, time, numpy as np
import jax, jax.numpy as jnp
import torch

jax.config.update("jax_enable_x64", True)

def as_jax(x, dtype=jnp.float64):
    """Convert to JAX array with required dtype."""
    return jnp.asarray(x, dtype=dtype)

def as_np(x, dtype=np.float64):
    return np.asarray(x, dtype=dtype)

def as_torch(x, device="cpu", dtype=torch.float64):
    return torch.as_tensor(x, device=device, dtype=dtype)

def compute_sqeuclidean_cost(X, Y, normalize=True, jax_dtype=jnp.float64):
    """Squared Euclidean cost with optional normalization, JAX return."""
    X = as_jax(X, dtype=jax_dtype)
    Y = as_jax(Y, dtype=jax_dtype)
    x2 = jnp.sum(X * X, axis=1)
    y2 = jnp.sum(Y * Y, axis=1)
    G  = X @ Y.T
    C  = x2[:, None] + y2[None, :] - 2.0 * G
    C  = jnp.maximum(C, 0)
    if normalize:
        C = C / jnp.maximum(jnp.max(C), jnp.finfo(C.dtype).tiny)
    return C  # jax array

def sinkhorn_rescaling(P, g1, g2, max_iter=1000, tol=1e-6):
    P  = jnp.asarray(P)
    g1 = jnp.asarray(g1, dtype=P.dtype)
    g2 = jnp.asarray(g2, dtype=P.dtype)
    eps = jnp.finfo(P.dtype).tiny

    ones_col = jnp.ones((P.shape[1],), dtype=P.dtype)
    ones_row = jnp.ones((P.shape[0],), dtype=P.dtype)

    def body_fun(state):
        P, toggle = state
        def scale_rows(P):
            row_sum = jnp.maximum(P @ ones_col, eps)
            s = g1 / row_sum                # shape (n,)
            return P * s[:, None]           # broadcast scale rows
        def scale_cols(P):
            col_sum = jnp.maximum(P.T @ ones_row, eps)
            s = g2 / col_sum                # shape (m,)
            return P * s[None, :]           # broadcast scale cols
        P2 = jax.lax.cond(toggle, scale_rows, scale_cols, P)
        return (P2, jnp.logical_not(toggle))

    def done(state):
        P, _ = state
        rerr = jnp.sum(jnp.abs(P @ ones_col - g1))
        cerr = jnp.sum(jnp.abs(P.T @ ones_row - g2))
        return jnp.logical_or(rerr <= tol, cerr <= tol)

    i = 0
    state = (P, True)
    while (not bool(done(state))) and i < max_iter:
        state = body_fun(state)
        i += 1
    return state[0]

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
        sA = jnp.sqrt(jnp.maximum(jnp.max(jnp.abs(A)), 1.0))
        sB = jnp.sqrt(jnp.maximum(jnp.max(jnp.abs(B)), 1.0))
        A = A / sA
        B = B / sB
        if return_scale:
            return A, B, sA, sB
    return A, B

# ---------- HiRef / Monge-rotation K-Means ----------
def run_monge_conj(XA, YB, rank, ot_solver="HiRef",
                   init='default', hiref_iters=300,
                   hiref_max_Q=1000, hiref_max_rank=100,
                   lambda_factor=0.5
                  ):
    
    import monge_rotate_lr as mr_lr
    np_dtype = np.float64  # or float64 if you keep x64
    XA = np.asarray(XA, dtype=np.float64)
    YB = np.asarray(YB, dtype=np.float64)
    
    t0 = time.time()
    Q, g, R = mr_lr.monge_rotation_kmeans_LR(
            XA, YB, rank, lambda_factor=lambda_factor, random_state=0, epsilon=1e-2, ot_solver=ot_solver,
                init=init, hiref_iters=hiref_iters, 
                hiref_max_Q=hiref_max_Q, hiref_max_rank=hiref_max_rank
        )
    t1 = time.time()
    
    _loss_lr_two  # adjust import
    A, B = compute_lr_sqeuclidean_factors(XA, YB)
    cost = float(_loss_lr_two(Q, R, A, B, g))
    
    res = {
        "algorithm": "hiref",
        "rank": int(rank),
        "objective_cost": cost,
        "runtime_sec": t1 - t0,
        "factors": True,
        "plan": False
    }
    return (Q, R, g), res

# ---------- FRLC ----------
def run_frlc(g1, g2, X=None, Y=None, C=None, rank=64, device=None, dtype=None):
    import FRLC.FRLC as frlc
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    dtype  = dtype or (torch.float64 if jax.config.read("jax_enable_x64") else torch.float32)
    
    # Seeding with 42 for reproducibility
    gen_cpu, gen_cuda = make_generators(123, device)
    gen = gen_cuda if device.startswith('cuda') else gen_cpu
    
    if C is None and (X is None or Y is None):
        raise ValueError("FRLC: provide either C or (X,Y).")
    
    if C is None:
        # Normalized squared-Euclidean cost on device/dtype (torch)
        A, B = compute_lr_sqeuclidean_factors(X, Y, rescale_cost=True)
        t0 = time.time()
        A, B = np.array(A), np.array(B)
        A_t, B_t  = as_torch(A, device=device, dtype=dtype), as_torch(B, device=device, dtype=dtype)
        Q_t, R_t, g_t, errs = frlc.FRLC_LR_opt(
            (A_t, B_t.T), 
            (A_t, B_t.T), 
            (A_t, B_t.T), 
            alpha=0.0,
            device=device, r=int(rank),
            max_iter=20, returnFull=False, gamma=70,
            max_inneriters_balanced=500, max_inneriters_relaxed=500,
            diagonalize_return=True,
            seed=None,
            gen=gen
        )
        #,dtype=dtype
        g_t = torch.diag(g_t)
        t1 = time.time()
    else:
        C_np = np.asarray(C, dtype=np.float64 if dtype == torch.float64 else np.float32)
        C_t  = as_torch(C_np, device=device, dtype=dtype)
        t0 = time.time()
        Q_t, R_t, g_t, errs = frlc.FRLC_opt(C, 
            alpha=0.0,
            device=device, r=int(rank),
            max_iter=20, returnFull=False, gamma=70,
            max_inneriters_balanced=500, max_inneriters_relaxed=500,
            diagonalize_return=True
        )
        g_t = torch.diag(g_t)
        t1 = time.time()
    
    # Convert factors to numpy
    Q = Q_t.detach().cpu().numpy()
    R = R_t.detach().cpu().numpy()
    g = g_t.detach().cpu().numpy()
    
    try:
        #from your_lr_costs import lr_sqeuclidean_factors, _loss_lr_two  # replace with your module
        if X is None or Y is None:
            raise RuntimeError("Need X,Y to build A,B factors for LR loss.")
        A, B = compute_lr_sqeuclidean_factors(X, Y)
        cost = float(_loss_lr_two(Q, R, A, B, g))
    except Exception:
        if X is None or Y is None:
            QTCR = Q.T @ (C_np @ R)   # (r, r)
            cost = float(np.sum(np.diag(QTCR) / np.maximum(g, np.finfo(Q.dtype).tiny)))
    
    res = {
        "algorithm": "frlc",
        "rank": int(rank),
        "objective_cost": cost,
        "runtime_sec": t1 - t0,
        "factors": True,
        "plan": False
    }
    return (Q, R, g), res

# ---------- LOT via OTT (low-rank Sinkhorn) ----------
def run_lot(g1, g2, X=None, Y=None, C=None, rank=64, epsilon=None):
    from ott.initializers.linear.initializers_lr import RandomInitializer
    from ott.geometry.pointcloud import PointCloud
    from ott.geometry.geometry import Geometry
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn_lr

    # dtype policy: follow your global JAX flag
    jdtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    eps = as_jax(1e-3 if epsilon is None else epsilon, jdtype)

    # Geometry: pointcloud preferred to avoid dense C
    if C is not None:
        geom = Geometry(cost_matrix=as_jax(C, jdtype), epsilon=eps, scale_cost="max_cost")
    else:
        geom = PointCloud(x=as_jax(X, jdtype), y=as_jax(Y, jdtype),
                          epsilon=eps, scale_cost="max_cost")

    a = as_jax(g1, jdtype)
    b = as_jax(g2, jdtype)
    prob = linear_problem.LinearProblem(geom, a, b)

    t0 = time.time()
    solver = sinkhorn_lr.LRSinkhorn(rank=int(rank), initializer=RandomInitializer(int(rank)))
    ot_lr = solver(prob)  # LRSinkhornOutput
    # force compute to finish before timing
    _ = jax.block_until_ready(ot_lr.costs)
    t1 = time.time()

    # Factors are directly available per your source
    # (they are JAX arrays; bring to host as NumPy)
    Q = np.array(jax.device_get(ot_lr.q))
    R = np.array(jax.device_get(ot_lr.r))
    g = np.array(jax.device_get(ot_lr.g))

    # Use solver's cost if exposed; otherwise compute cheaply from factors
    try:
        A, B = compute_lr_sqeuclidean_factors(X, Y)
        cost = float(_loss_lr_two(Q, R, A, B, g))
    except Exception:
        QTCR = Q.T @ (np.asarray(C, dtype=Q.dtype) @ R)
        cost = float(np.sum(np.diag(QTCR) / np.maximum(g, np.finfo(Q.dtype).tiny)))
    
    res = {
        "algorithm": "lot",
        "rank": int(rank),
        "objective_cost": cost,
        "runtime_sec": t1 - t0,
        "factors": True,
        "plan": False,
    }
    
    return (Q, R, g), res

# ---------- Plain Sinkhorn (dense) with POT ----------
def run_sinkhorn(g1, g2, X=None, Y=None, C=None, reg=0.05):
    import ot as pot
    np_dtype = np.float64 if jax.config.read("jax_enable_x64") else np.float32

    if C is None:
        if X is None or Y is None:
            raise ValueError("Sinkhorn: need C or (X,Y)")
        C = compute_sqeuclidean_cost(X, Y, normalize=True, jax_dtype=(jnp.float64 if np_dtype==np.float64 else jnp.float32))
    C = np.asarray(C, dtype=np_dtype)
    a = np.asarray(g1, dtype=np_dtype); b = np.asarray(g2, dtype=np_dtype)

    t0 = time.time()
    P = pot.sinkhorn(a, b, C, reg=float(np_dtype(reg)), method="sinkhorn")
    t1 = time.time()
    cost = float(np.sum(P * C))
    res = {
        "algorithm": "sinkhorn",
        "rank": None,
        "objective_cost": cost,
        "runtime_sec": t1 - t0,
        "factors": False,
        "plan": True
    }
    return P, res

def run_all_methods(XA, YB, yA, yB, methods, rank=64, reg=0.05, ot_solver="HiRef",
                    device=None, evaluate_factors_fn=None):
    """
    methods: list like ["monge_conj", "frlc", "lot", "sinkhorn"]
    evaluate_factors_fn: callable (Q, R, yA, yB) -> dict
    """
    
    n, m = XA.shape[0], YB.shape[0]
    g1 = np.ones(n, dtype=np.float64) / n
    g2 = np.ones(m, dtype=np.float64) / m

    out = {}

    for method in methods:
        name = method.lower()
        try:
            if name == "monge_conj":
                (Q, R, g), res = run_monge_conj(XA, YB, rank=rank, ot_solver=ot_solver)
                # ensure numpy
                Q, R, g = np.asarray(Q), np.asarray(R), np.asarray(g)
                out[name] = {"result": res} #, "factors": (Q, R, g), "plan": None}

                if evaluate_factors_fn is not None:
                    try:
                        met = evaluate_factors_fn(Q, R, yA, yB)
                        out[name]["metrics"] = met
                    except Exception as e:
                        logger.warning(f"{name} evaluation failed: {e}")
            
            elif name == "frlc":
                
                (Q, R, g), res = run_frlc(g1, g2, X=XA, Y=YB, C=None, rank=rank, device=device)
                Q, R, g = np.asarray(Q), np.asarray(R), np.asarray(g)
                out[name] = {"result": res} #, "factors": (Q, R, g), "plan": None}

                if evaluate_factors_fn is not None:
                    try:
                        met = evaluate_factors_fn(Q, R, yA, yB)
                        out[name]["metrics"] = met
                    except Exception as e:
                        logger.warning(f"{name} evaluation failed: {e}")

            elif name == "lot":
                (Q, R, g), res = run_lot(g1, g2, X=XA, Y=YB, C=None, rank=rank, epsilon=1e-3)
                Q, R, g = np.asarray(Q), np.asarray(R), np.asarray(g)
                out[name] = {"result": res} #, "factors": (Q, R, g), "plan": None}

                if evaluate_factors_fn is not None:
                    try:
                        met = evaluate_factors_fn(Q, R, yA, yB)
                        out[name]["metrics"] = met
                    except Exception as e:
                        logger.warning(f"{name} evaluation failed: {e}")

            elif name == "sinkhorn":
                P, res = run_sinkhorn(g1, g2, X=XA, Y=YB, C=None, reg=reg)
                P = np.asarray(P)
                out[name] = {"result": res} #, "factors": None, "plan": P}

            else:
                logger.warning(f"Unknown method '{method}', skipping.")
                continue

            logger.info(f"{name}: cost={out[name]['result']['objective_cost']:.6f}, "
                        f"time={out[name]['result']['runtime_sec']:.3f}s")
            # Clear memory/cache
            del Q, R, g
            gc.collect(); torch.cuda.is_available() and (torch.cuda.synchronize(), torch.cuda.empty_cache()); jax.clear_caches()
            
        except Exception as e:
            logger.error(f"{name} failed: {e}")
            out[name] = {"error": str(e)}

    return out

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

def _loss_lr_two(Q: jnp.ndarray, R: jnp.ndarray,
                 A: jnp.ndarray, B: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    SA = Q.T @ A          # (r,k)
    RB = R.T @ B          # (r,k)
    return jnp.sum(jnp.sum(RB * SA, axis=1) / jnp.clip(g, 1e-18))

'''
def lowrank_monge_adapter(XA: np.ndarray, YB: np.ndarray, 
                          rank: int = 64, device: str = "cpu",
                          ot_solver='HiRef'
                         ):
    
    XA = np.asarray(XA, dtype=np.float64)
    YB = np.asarray(YB, dtype=np.float64)
    
    try:
        Q, g, R = mr_lr.monge_rotation_kmeans_LR(
            XA, YB, rank, lambda_factor=0.5, random_state=0, epsilon=1e-2, ot_solver=ot_solver
        )
    except Exception as e:
        print(f"Low-rank method failed: {e}")
        return None, None
    
    A, B = lr_sqeuclidean_factors( XA, YB )
    
    SA = Q.T @ A
    RB = R.T @ B
    cost = _loss_lr_two(Q, R, A, B, g)
    
    return (Q, R, g), cost
'''

'''
def evaluate_plan(P: np.ndarray, yA: np.ndarray, yB: np.ndarray) -> Dict:
    yA = np.asarray(yA)
    yB = np.asarray(yB)
    classes = np.unique(np.concatenate([yA, yB]))
    class_to_idx = {c: k for k, c in enumerate(classes)}
    kC = len(classes)

    M = np.zeros((kC, kC), dtype=np.float64)
    for i, ci in enumerate(yA):
        row = P[i]
        for j, cj in enumerate(yB):
            M[class_to_idx[ci], class_to_idx[cj]] += row[j]

    total_mass = P.sum()
    class_mass_accuracy = float(np.trace(M) / (total_mass + 1e-12))

    predA = np.empty_like(yA)
    for i, _ in enumerate(yA):
        mass_per_class = np.zeros(kC, dtype=np.float64)
        for j, cj in enumerate(yB):
            mass_per_class[class_to_idx[cj]] += P[i, j]
        predA[i] = classes[mass_per_class.argmax()]

    predB = np.empty_like(yB)
    for j, _ in enumerate(yB):
        mass_per_class = np.zeros(kC, dtype=np.float64)
        for i, ci in enumerate(yA):
            mass_per_class[class_to_idx[ci]] += P[i, j]
        predB[j] = classes[mass_per_class.argmax()]

    x_AMI = float(AMI(yA, predA))
    x_ARI = float(ARI(yA, predA))
    y_AMI = float(AMI(yB, predB))
    y_ARI = float(ARI(yB, predB))

    return {
        "class_mass_accuracy": class_mass_accuracy,
        "x_AMI": x_AMI,
        "x_ARI": x_ARI,
        "y_AMI": y_AMI,
        "y_ARI": y_ARI,
        "class_mass_matrix": M,
        "classes": classes,
    }
'''
