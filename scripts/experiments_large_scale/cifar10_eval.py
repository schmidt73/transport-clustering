import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI
from scipy.optimize import linear_sum_assignment

def evaluate_factors(Q: np.ndarray,
                     R: np.ndarray,
                     yA: np.ndarray,
                     yB: np.ndarray,
                     classes: np.ndarray | None = None):
    """
    Evaluate clustering quality directly from factor memberships Q (X-side) and R (Y-side).

    Args
    ----
    Q : (n, r) nonnegative, row-stochastic or approximately so.
    R : (n, r) same for Y-side.
    yA: (n,) ground-truth class labels for X.
    yB: (n,) ground-truth class labels for Y.
    classes: optional array of unique class ids (shared label set). If None, inferred.

    Returns
    -------
    dict with:
      - 'A_AMI', 'A_ARI': AMI/ARI(yA, argmax(Q))
      - 'B_AMI', 'B_ARI': AMI/ARI(yB, argmax(R))
      - 'A_class_mass_accuracy', 'B_class_mass_accuracy': soft majority-mass fractions
      - 'A_cluster_majority', 'B_cluster_majority': majority class per cluster (length r)
      - 'shared_class_set_match': fraction overlap of majority-class sets (naive check)
      - 'best_cluster_label_alignment_acc': best matching accuracy between A/B cluster labels
        under Hungarian (when r is reasonably comparable on both sides)
      - 'A_class_cluster_matrix', 'B_class_cluster_matrix': soft mass tables (kC x r)
    """
    yA = np.asarray(yA)
    yB = np.asarray(yB)

    if classes is None:
        classes = np.unique(np.concatenate([yA, yB]))
    classes = np.asarray(classes)
    kC = len(classes)
    class_to_idx = {c: k for k, c in enumerate(classes)}

    n, r = Q.shape
    assert R.shape[0] == n and R.shape[1] == r, "Q and R must have matching shapes (n,r)."

    # Hard assignments via argmax
    zA = Q.argmax(axis=1)
    zB = R.argmax(axis=1)

    # External clustering metrics (label vs cluster-id)
    A_AMI = float(AMI(yA, zA))
    A_ARI = float(ARI(yA, zA))
    B_AMI = float(AMI(yB, zB))
    B_ARI = float(ARI(yB, zB))

    # Soft class-by-cluster mass tables
    A_CC = np.zeros((kC, r), dtype=np.float64)
    B_CC = np.zeros((kC, r), dtype=np.float64)
    for i in range(n):
        A_CC[class_to_idx[yA[i]]] += Q[i]   # add row Q[i,:] to that class
        B_CC[class_to_idx[yB[i]]] += R[i]

    # Majority class per cluster (per side)
    A_cluster_majority_idx = A_CC.argmax(axis=0)  # (r,)
    B_cluster_majority_idx = B_CC.argmax(axis=0)  # (r,)
    A_cluster_majority = classes[A_cluster_majority_idx]
    B_cluster_majority = classes[B_cluster_majority_idx]

    # Soft "class-mass accuracy": total mass sitting in majority class of each cluster
    # Normalize by total mass (sum of Q) to get a fraction in [0,1].
    A_majority_mass = A_CC.max(axis=0).sum()
    B_majority_mass = B_CC.max(axis=0).sum()
    # Totals: if rows are normalized to 1/n, sum(Q)=1; else compute explicitly
    A_total_mass = Q.sum()
    B_total_mass = R.sum()
    A_class_mass_accuracy = float(A_majority_mass / (A_total_mass + 1e-12))
    B_class_mass_accuracy = float(B_majority_mass / (B_total_mass + 1e-12))

    # Do A-clusters' majority labels "match" B-clusters' majority labels?
    # 1) simple set overlap ratio
    overlap = len(set(A_cluster_majority.tolist()) & set(B_cluster_majority.tolist()))
    shared_class_set_match = float(overlap / max(1, len(set(classes))))

    # 2) best one-to-one alignment of clusters by majority-label agreement (Hungarian)
    #    Build an r x r score matrix: 1 if labels match, else 0.
    S = (A_cluster_majority[:, None] == B_cluster_majority[None, :]).astype(np.float64)
    # Maximize matches ⇒ minimize -S
    row_ind, col_ind = linear_sum_assignment(-S)
    best_cluster_label_alignment_acc = float(S[row_ind, col_ind].sum() / r)

    return {
        "A_AMI": A_AMI,
        "A_ARI": A_ARI,
        "B_AMI": B_AMI,
        "B_ARI": B_ARI,
        "A_class_mass_accuracy": A_class_mass_accuracy,
        "B_class_mass_accuracy": B_class_mass_accuracy,
        "A_cluster_majority": A_cluster_majority,
        "B_cluster_majority": B_cluster_majority,
        "shared_class_set_match": shared_class_set_match,
        "best_cluster_label_alignment_acc": best_cluster_label_alignment_acc,
        "A_class_cluster_matrix": A_CC,
        "B_class_cluster_matrix": B_CC,
        "classes": classes,
    }
