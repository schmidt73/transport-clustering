import argparse as ap
import jax
import numpy as np
import jax.numpy as jnp

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("-n", "--n", type=int, default=100)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-o", "--output", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    eps = -0.1
    positions = jnp.array([
        [0.0, 2.0], # top left
        [1.0 + eps, 2.0], # top middle
        [2.0, 1.0], # middle right
        [0.0, 1.0], # middle left
        [1.0 - eps, 0.0], # bottom middle
        [2.0, 0.0], # bottom right
    ])

    args = parse_args()
    jax.random.PRNGKey(args.seed)
    n = args.n

    sigma = 0.1
    top_left = jax.random.multivariate_normal(jax.random.PRNGKey(0), mean=positions[0], cov=sigma * jnp.eye(2), shape=(n,))
    bottom_right = jax.random.multivariate_normal(jax.random.PRNGKey(1), mean=positions[-1], cov=sigma * jnp.eye(2), shape=(n,))
    top_middle = jax.random.multivariate_normal(jax.random.PRNGKey(2), mean=positions[1], cov=sigma * jnp.eye(2), shape=(n,))
    bottom_middle = jax.random.multivariate_normal(jax.random.PRNGKey(3), mean=positions[4], cov=sigma * jnp.eye(2), shape=(n,))
    middle_left = jax.random.multivariate_normal(jax.random.PRNGKey(4), mean=positions[3], cov=sigma * jnp.eye(2), shape=(n,))
    middle_right = jax.random.multivariate_normal(jax.random.PRNGKey(5), mean=positions[2], cov=sigma * jnp.eye(2), shape=(n,))
    X = jnp.vstack([top_left, bottom_right, top_middle, bottom_middle])
    Y = jnp.vstack([top_left, bottom_right, middle_right, middle_left])

    np.savetxt(args.output + "_X.txt", X, fmt="%.6f")
    np.savetxt(args.output + "_Y.txt", Y, fmt="%.6f")