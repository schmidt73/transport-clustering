
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

'''
Code for Monge Rotation on Kernelized Costs (squared Euclidean):
    Reduction proceeds as:
        1. Monge Rotate cost: Ctilde = C @ P.T by solving for Monge map P
        2. Take Sym(Ctilde) = Ctilde + Ctilde.T
        3. Grammize Ctilde as -0.5 * J @ Sym(Ctilde) @ J for J = Id - (1/2) 11^T
        4. Embed by identifying G = X X^T and extracting X from decomposition
        5. Run K-means on X
        6. Yield Q from K-means, R = P.T @ Q
'''

# ----- Utilities reused (compact) -----
def squared_euclidean_cost(X, Y):
    X2 = np.sum(X**2, axis=1, keepdims=True)
    Y2 = np.sum(Y**2, axis=1, keepdims=True).T
    return X2 + Y2 - 2.0 * X @ Y.T

def monge_permutation(C):
    row_ind, col_ind = linear_sum_assignment(C)
    n = C.shape[0]
    P = np.zeros_like(C)
    P[row_ind, col_ind] = 1.0
    return col_ind, P

def symmetrize(M):
    return 0.5 * (M + M.T)

def double_center(D):
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    return J @ D @ J

def gram_from_cross_dist(S):
    Gc = -0.5 * double_center(S)
    w, V = np.linalg.eigh((Gc + Gc.T) * 0.5)
    w = np.clip(w, 0.0, None)
    Gc = -0.5 * double_center(S)
    return (V * w) @ V.T

def embed_from_gram(G, tol=1e-12):
    w, V = np.linalg.eigh((G + G.T) * 0.5)
    keep = w > tol
    if not np.any(keep):
        return np.zeros((G.shape[0], 1))
    return V[:, keep] * np.sqrt(w[keep])

def _kmeanspp_init(X, k, rng):
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=X.dtype)
    i0 = rng.integers(n)
    centers[0] = X[i0]
    d2 = np.sum((X - centers[0])**2, axis=1)
    for t in range(1, k):
        s = d2.sum()
        probs = d2/s if s > 0 else np.ones(n)/n
        it = rng.choice(n, p=probs)
        centers[t] = X[it]
        d2 = np.minimum(d2, np.sum((X - centers[t])**2, axis=1))
    return centers

def _lloyds_kmeans(X, k, max_iter=100, tol=1e-6, random_state=0):
    rng = np.random.default_rng(random_state)
    centers = _kmeanspp_init(X, k, rng)
    for _ in range(max_iter):
        d2 = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
        labels = np.argmin(d2, axis=1)
        new_centers = np.zeros_like(centers)
        for j in range(k):
            pts = X[labels == j]
            if len(pts) == 0:
                new_centers[j] = X[rng.integers(X.shape[0])]
            else:
                new_centers[j] = pts.mean(axis=0)
        if np.linalg.norm(new_centers - centers) <= tol:
            centers = new_centers
            break
        centers = new_centers
    d2 = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
    labels = np.argmin(d2, axis=1)
    return labels, centers

def monge_rotation_kmeans(X, Y, r, random_state=0):
    # Cost and Monge
    C = squared_euclidean_cost(X, Y)
    perm, P = monge_permutation(C)
    # Rotate and symmetrize
    Ctilde = C @ P.T
    S = Ctilde + Ctilde.T
    # Gram and embed
    G = gram_from_cross_dist(S)
    # Returning our embedded points from the Gram matrix
    Z = embed_from_gram(G)
    # k-means on Z
    labels, centers = _lloyds_kmeans(Z, r, random_state=random_state)
    # "Hard" Q with row-sum 1/n
    n = X.shape[0]
    Q_onehot = np.zeros((n, r))
    Q_onehot[np.arange(n), labels] = 1.0
    Q = Q_onehot / n
    R = P.T @ Q
    return Q, R, labels, perm

def plot_coclusters(X, Y, Q, R, title_suffix=""):
    # Argmax labels
    labels_X = np.argmax(Q, axis=1)
    labels_Y = np.argmax(R, axis=1)

    # Plot X with labels_X
    plt.figure()
    for k in np.unique(labels_X):
        pts = X[labels_X == k]
        plt.scatter(pts[:,0], pts[:,1], label=f"cluster {int(k)}", s=18)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"X clusters via argmax(Q){title_suffix}")
    
    for k in np.unique(labels_Y):
        pts = Y[labels_Y == k]
        plt.scatter(pts[:,0], pts[:,1], label=f"cluster {int(k)}", s=18)
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.title(f"Y clusters via argmax(R){title_suffix}")
    
    plt.legend()
    plt.tight_layout()
    plt.show()
