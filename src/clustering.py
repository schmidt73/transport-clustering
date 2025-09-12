import numpy as np
import jax
import jax.numpy as jnp

def _kmeanspp_init(X, k, rng):
    n = X.shape[0]
    centers = jnp.empty((k, X.shape[1]), dtype=X.dtype)
    i0 = rng.integers(n)
    centers = centers.at[0].set(X[i0])
    d2 = jnp.sum((X - centers[0])**2, axis=1)
    for t in range(1, k):
        s = d2.sum()
        probs = d2/s if s > 0 else jnp.ones(n)/n
        it = rng.choice(n, p=probs)
        centers = centers.at[t].set(X[it])
        d2 = jnp.minimum(d2, jnp.sum((X - centers[t])**2, axis=1))
    return centers

def lloyds_kmeans(X, k, max_iter=250, tol=1e-6, random_state=0):
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
