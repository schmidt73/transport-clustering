
import numpy as np
import jax
import jax.numpy as jnp
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
    X2 = jnp.sum(X**2, axis=1, keepdims=True)
    Y2 = jnp.sum(Y**2, axis=1, keepdims=True).T
    return X2 + Y2 - 2.0 * X @ Y.T

def monge_permutation(C):
    row_ind, col_ind = linear_sum_assignment(C)
    n = C.shape[0]
    P = jnp.zeros_like(C)
    P = P.at[row_ind, col_ind].set(1.0)
    return col_ind, P

def symmetrize(M):
    return 0.5 * (M + M.T)

def double_center(D):
    n = D.shape[0]
    J = jnp.eye(n) - jnp.ones((n, n)) / n
    return J @ D @ J

def gram_from_cross_dist(S):
    Gc = -0.5 * double_center(S)
    w, V = jnp.linalg.eigh((Gc + Gc.T) * 0.5)
    w = jnp.clip(w, 0.0, None)
    Gc = -0.5 * double_center(S)
    return (V * w) @ V.T

def embed_from_gram(G, tol=1e-12):
    w, V = jnp.linalg.eigh((G + G.T) * 0.5)
    keep = w > tol
    if not jnp.any(keep):
        return jnp.zeros((G.shape[0], 1))
    return V[:, keep] * jnp.sqrt(w[keep])

def _kmeanspp_init(X, k, rng):
    n = X.shape[0]
    centers = jnp.empty((k, X.shape[1]), dtype=X.dtype)
    i0 = rng.integers(n)
    centers = centers.at[0].set(X[i0])
    d2 = jnp.sum((X - centers[0])**2, axis=1)
    for t in range(1, k):
        s = d2.sum()
        probs = d2/s if s > 0 else np.ones(n)/n
        it = rng.choice(n, p=probs)
        centers = centers.at[t].set(X[it])
        d2 = jnp.minimum(d2, jnp.sum((X - centers[t])**2, axis=1))
    return centers

def _lloyds_kmeans(X, k, max_iter=250, tol=1e-6, random_state=0):
    rng = np.random.default_rng(random_state)
    centers = _kmeanspp_init(X, k, rng)
    for _ in range(max_iter):
        d2 = jnp.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
        labels = jnp.argmin(d2, axis=1)
        new_centers = jnp.zeros_like(centers)
        for j in range(k):
            pts = X[labels == j]
            if len(pts) == 0:
                new_centers = new_centers.at[j].set(X[rng.integers(X.shape[0])])
            else:
                new_centers = new_centers.at[j].set(pts.mean(axis=0))
        if jnp.linalg.norm(new_centers - centers) <= tol:
            centers = new_centers
            break
        centers = new_centers
    d2 = jnp.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
    labels = jnp.argmin(d2, axis=1)
    return labels, centers

def monge_rotation_kmeans(C, X, Y, r, random_state=0):
    perm, P = monge_permutation(C)
    labels_X, centers_X = _lloyds_kmeans(X, r, random_state=random_state)
    labels_Y, centers_Y = _lloyds_kmeans(Y, r, random_state=random_state)

    Ctilde = C @ P.T
    S = Ctilde + Ctilde.T
    G = gram_from_cross_dist(S)
    Z = embed_from_gram(G)
    labels, centers = _lloyds_kmeans(Z, r, random_state=random_state)
    n = C.shape[0]
    Q1 = jnp.zeros((n, r))
    Q1 = Q1.at[jnp.arange(n), labels].set(1.0)
    R1 = P.T @ Q1

    Q2 = jnp.zeros((n, r))
    Q2 = Q2.at[jnp.arange(n), labels_X].set(1.0)
    R2 = P.T @ Q2

    R3 = jnp.zeros((n, r))
    R3 = R3.at[jnp.arange(n), labels_Y].set(1.0)
    Q3 = P @ R3

    P1 = Q1 @ jnp.linalg.inv(Q1.T @ Q1) @ R1.T
    P2 = Q2 @ jnp.linalg.inv(Q2.T @ Q2) @ R2.T
    P3 = Q3 @ jnp.linalg.inv(Q3.T @ Q3) @ R3.T
    print(f"Cost 1: {jnp.sum(C * P1) / n}, Cost 2: {jnp.sum(C * P2) / n}, Cost 3: {jnp.sum(C * P3) / n}")
    return Q1 @ jnp.linalg.inv(Q1.T @ Q1), R1, labels, perm

def plot_coclusters(X, Y, Q, R, title_suffix=""):
    # Argmax labels
    labels_X = jnp.argmax(Q, axis=1)
    labels_Y = jnp.argmax(R, axis=1)

    # Plot X with labels_X
    plt.figure()
    for k in jnp.unique(labels_X):
        pts = X[labels_X == k]
        plt.scatter(pts[:,0], pts[:,1], label=f"cluster {int(k)}", s=18)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"X clusters via argmax(Q){title_suffix}")

    for k in jnp.unique(labels_Y):
        pts = Y[labels_Y == k]
        plt.scatter(pts[:,0], pts[:,1], label=f"cluster {int(k)}", s=18)
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.title(f"Y clusters via argmax(R){title_suffix}")
    
    plt.legend()
    plt.tight_layout()
    plt.show()
