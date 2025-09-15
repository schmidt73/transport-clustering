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

def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    if device_str.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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

# Optional adapter for your low-rank OT method
def lowrank_monge_adapter(XA: np.ndarray, YB: np.ndarray, rank: int = 64, device: str = "cpu"):
    
    XA = np.asarray(XA, dtype=np.float64)
    YB = np.asarray(YB, dtype=np.float64)
    
    try:
        Q, g, R = mr_lr.monge_rotation_kmeans_LR(
            XA, YB, rank, lambda_factor=0.5, random_state=0, epsilon=1e-2
        )
    except Exception as e:
        print(f"Low-rank method failed: {e}")
        return None, None
    
    A, B = lr_sqeuclidean_factors( XA, YB )
    
    SA = Q.T @ A
    RB = R.T @ B
    cost = _loss_lr_two(Q, R, A, B, g)
    
    return (Q, R, g), cost
