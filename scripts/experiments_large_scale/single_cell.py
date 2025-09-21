import numpy as np
import pandas as pd
from collections import Counter
import scanpy as sc
import run_LowRank
import eval_LowRank
import torch

# --- Choose A/B from subset_adata (two timepoints, two replicates already selected) ---
def extract_pair(subset_adata, t1, t2, label_col="celltype_update", use_pca=True, pca_key="X_pca", n_comps=50):
    ad = subset_adata  # already .to_memory()
    assert label_col in ad.obs, f"Missing label column: {label_col}"
    
    if use_pca and pca_key not in ad.obsm:
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
        sc.tl.pca(ad, n_comps=n_comps, svd_solver="arpack")

    A = ad[ad.obs["day"] == t1]
    B = ad[ad.obs["day"] == t2]
    XA = A.obsm[pca_key] if use_pca else A.X
    YB = B.obsm[pca_key] if use_pca else B.X
    yA = np.asarray(A.obs[label_col])
    yB = np.asarray(B.obs[label_col])

    return XA, YB, yA, yB

# --- Stratify to equal counts per label across A/B (fixed seed) ---
def stratified_equalize(XA, YA, XB, YB, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.intersect1d(np.unique(YA), np.unique(YB))
    idxA, idxB = [], []
    for c in labels:
        a_idx = np.where(YA == c)[0]
        b_idx = np.where(YB == c)[0]
        k = min(len(a_idx), len(b_idx))
        if k == 0:
            continue
        idxA.append(rng.choice(a_idx, size=k, replace=False))
        idxB.append(rng.choice(b_idx, size=k, replace=False))
    idxA = np.concatenate(idxA) if idxA else np.array([], int)
    idxB = np.concatenate(idxB) if idxB else np.array([], int)
    return XA[idxA], YA[idxA], XB[idxB], YB[idxB], labels

# --- Uniform marginals and rank selection ---
def setup_lr_ot(XA, YB, yA, yB, classes, rank=None):
    n = XA.shape[0]; m = YB.shape[0]
    assert n == m, "Expected equalized sizes for balanced LR-OT."
    g1 = np.ones(n) / n
    g2 = np.ones(m) / m
    if rank is None:
        # conservative: min(#classes_A, #classes_B)
        rank = int(min(len(np.unique(yA)), len(np.unique(yB))))
    return g1, g2, rank

# --- Run all LR methods and evaluate ---
def run_pairwise_eval(
    subset_adata, t1, t2,
    methods=("monge_conj","lot","frlc"),
    label_col="celltype_update",
    seed=0, use_pca=True, pca_key="X_pca", n_comps=50,
    max_rank=None, subsample_to_nonprime=True
):
    # 1) Extract
    XA, YB, yA, yB = extract_pair(subset_adata, t1, t2, label_col, use_pca, pca_key, n_comps)
    # 2) Equalize per-label counts so A/B have same label histograms
    XA, yA, YB, yB, classes = stratified_equalize(XA, yA, YB, yB, seed=seed)
    
    if subsample_to_nonprime:
        n_current = XA.shape[0]
        n_target = choose_composite_size(n_current, max_small_factor=1000, search_window=20000)
        XA, yA, YB, yB = subsample_to_size_balanced(XA, yA, YB, yB, classes, n_target, seed=seed)
        print(f"Subsampled to n={XA.shape[0]} from n={n_current} [needed for Hierarchical Refinement]")
    
    rA = np.linalg.norm(XA, axis=1)
    rB = np.linalg.norm(YB, axis=1)
    s  = np.percentile(np.concatenate([rA, rB]), 95) + 1e-12
    
    # Normalize both consistently
    XA = XA / s
    YB = YB / s
    
    # 3) Setup LR-OT
    g1, g2, rank = setup_lr_ot(XA, YB, yA, yB, classes, rank=max_rank)
    
    # 4) Run methods (expects factor outputs)
    results = []
    for method in methods:
        method = method.lower()
        try:
            if method == "monge_conj":
                (Q, R, g), res = run_LowRank.run_monge_conj(XA, YB, rank=rank, ot_solver="HiRef")
            elif method == "lot":
                (Q, R, g), res = run_LowRank.run_lot(g1, g2, X=XA, Y=YB, rank=rank, epsilon=1e-3)
            elif method == "frlc":
                dev_str = "cuda" if torch.cuda.is_available() else "cpu"
                (Q, R, g), res = run_LowRank.run_frlc(g1, g2, X=XA, Y=YB, rank=rank, 
                                                      device=dev_str)
            else:
                continue
            
            # 5) Evaluate factors (uses proper CMA via class–class mass)
            met = eval_LowRank.evaluate_factors(Q, R, yA, yB, g=g, classes=np.asarray(classes), return_matrix=False, include_purity=False)
            
            row = {
                "timepoint_A": t1, "timepoint_B": t2,
                "method": method, "rank": rank,
                "ot_cost": res.get("objective_cost", np.nan),
                "A_AMI": met["A_AMI"], "A_ARI": met["A_ARI"],
                "B_AMI": met["B_AMI"], "B_ARI": met["B_ARI"],
                "CMA": met["CMA"],
                "runtime_sec": res.get("runtime_sec", np.nan),
                "n_cells": XA.shape[0]
            }
            results.append(row)
        
        except Exception as e:
            results.append({
                "timepoint_A": t1, "timepoint_B": t2,
                "method": method, "rank": rank,
                "error": str(e)
            })
    
    return pd.DataFrame(results)



def _num_divisors(m: int) -> int:
    # tau(m): number of positive divisors
    if m < 2:
        return 1
    divs = 1
    x = m
    p = 2
    while p * p <= x:
        if x % p == 0:
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            divs *= (e + 1)
        p += 1 if p == 2 else 2  # check 2, then odd p
    if x > 1:
        divs *= 2
    return divs

def _has_small_factor(m: int, max_factor: int) -> bool:
    # true if m has any nontrivial factor ≤ max_factor
    if m % 2 == 0:
        return True if 2 <= max_factor else False
    p = 3
    while p * p <= m and p <= max_factor:
        if m % p == 0:
            return True
        p += 2
    return False

def choose_composite_size(n: int, *, max_small_factor: int | None = 1000,
                          search_window: int = 10000) -> int:
    """
    Find m <= n with many divisors and not prime.
    Prefers even numbers; searches up to `search_window` steps downward.
    If max_small_factor is not None, require at least one factor <= that cap.
    """
    best_m, best_tau = None, -1
    start = n
    stop = max(2, n - search_window)
    for m in range(start, stop - 1, -1):
        # skip primes quickly: tau(m)==2
        tau = _num_divisors(m)
        if tau <= 2:
            continue
        if max_small_factor is not None and not _has_small_factor(m, max_small_factor):
            continue
        # prefer even & larger tau; break ties by closeness to n (i.e., larger m)
        score = (tau, (m % 2 == 0), m)
        if score > (best_tau, best_m is not None and (best_m % 2 == 0), best_m or -1):
            best_m, best_tau = m, tau
            # early exit for perfect hits
            if tau >= 128 and m % 2 == 0:
                break
    if best_m is None:
        # fallback: make it even and composite
        best_m = n if n % 2 == 0 else n - 1
        if _num_divisors(best_m) <= 2:
            best_m -= 2
    return max(2, best_m)

def subsample_to_size_balanced(
    XA: np.ndarray, yA: np.ndarray,
    YB: np.ndarray, yB: np.ndarray,
    classes: np.ndarray,
    target_n: int,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    # current per-class counts (equal across A/B by construction)
    counts = np.array([(yA == c).sum() for c in classes], dtype=int)
    total = counts.sum()
    if target_n > total:
        raise ValueError(f"target_n={target_n} exceeds available {total}")

    # proportional target per class
    probs = counts / total
    raw = probs * target_n
    floor = np.floor(raw).astype(int)
    rem = target_n - floor.sum()
    # largest remainder allocation
    ranks = np.argsort(-(raw - floor))  # descending fractional part
    floor[ranks[:rem]] += 1
    target_per_class = floor

    # sample indices per class for A and B
    idxA_list, idxB_list = [], []
    for c, k in zip(classes, target_per_class):
        if k == 0:
            continue
        poolA = np.where(yA == c)[0]
        poolB = np.where(yB == c)[0]
        selA = rng.choice(poolA, size=k, replace=False)
        selB = rng.choice(poolB, size=k, replace=False)
        idxA_list.append(selA)
        idxB_list.append(selB)

    idxA = np.concatenate(idxA_list) if idxA_list else np.array([], int)
    idxB = np.concatenate(idxB_list) if idxB_list else np.array([], int)

    # deterministic shuffle to avoid class blocks
    permA = rng.permutation(idxA)
    permB = rng.permutation(idxB)

    return XA[permA], yA[permA], YB[permB], yB[permB]

