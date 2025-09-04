import sys
import jax 
import jax.numpy as jnp
import itertools
from loguru import logger
import numpy as np
import networkx as nx
from tqdm import tqdm

def powerset(S):
    if not S: 
        yield S
        return
    
    e = next(iter(S))
    for T in powerset(S - {e}):
        yield T | {e}
        yield T

def enumerate_permutation_matrices(n : int):
    cols = jnp.arange(n)
    for σ in itertools.permutations(range(n)):
        perm = jnp.zeros((n, n))
        perm = perm.at[cols,σ].set(1.0)
        yield perm

def enumerate_factors(n : int):
    cols = jnp.arange(n)
    for S in powerset(set(range(n))):
        Q = jnp.ones((n, 2))
        Q = Q.at[:,0].set(0.0)
        Q = Q.at[list(S),0].set(1.0)
        Q = Q.at[list(S),1].set(0.0)
        yield Q

def enumerate_low_rank_plans(n : int):
    low_rank_plans = []
    for Q in enumerate_factors(n):
        if jnp.any(Q.sum(axis=0) != n / 2): continue
        for R in enumerate_factors(n):
            if jnp.any(R.sum(axis=0) != n / 2): continue
            P = (Q @ R.T) * (2 / n)
            yield (P, Q, R)

def solve_monge(C : jnp.ndarray, perms = None):
    assert C.shape[0] == C.shape[1]
    n = C.shape[0]  
    if perms is None:
        perms = jnp.array(list(enumerate_permutation_matrices(n)))
    
    objs = jnp.sum(perms * C[None, :], axis=[1,2])
    min_obj = jnp.min(objs)
    return perms[objs <= min_obj + 1e-3]

def check_counterexample(
    C : jnp.ndarray, 
    low_rank_P : jnp.ndarray, 
    low_rank_Q : jnp.ndarray, 
    low_rank_R : jnp.ndarray, 
    full_rank_P : jnp.ndarray
):
    P = solve_monge(C, perms=full_rank_P)
    
    # if P.shape[0] > 1: 
    #     return {
    #         "is_counterexample": False
    #     }

    P = P[0]
    # if not jnp.allclose(P, jnp.eye(P.shape[0])):
    #     return {
    #         "is_counterexample": False
    #     }
    objs = jnp.sum(low_rank_P * C[None,:], axis=[1,2])
    monge_separators = jnp.all(P @ low_rank_R == low_rank_Q, axis=(1,2))
    min_separator_obj = jnp.min(objs[monge_separators])
    min_obj = jnp.min(objs)
    if min_obj < min_separator_obj - 1e-3:
        return {
            "is_counterexample": True,
            "indices": jnp.where(objs < min_separator_obj)[0],
            "min_obj": min_obj,
            "min_separator_obj": min_separator_obj
        }
    
    return {
        "is_counterexample": False
    }

def random_partitioned_distance_matrix(
    n: int,
    p: float = 0.2,
    *,
    weight_dist = "uniform",  # "uniform" | "exponential" | "lognormal"
    weight_params = None,
    seed = None,
    directed = False,
    return_graph = False
):
    rng = np.random.default_rng(seed)
    if weight_params is None:
        weight_params = {}

    # Build random graph
    G = nx.gnp_random_graph(2 * n, p, seed=seed, directed=directed)

    # Sample positive weights per edge
    m = G.number_of_edges()
    if weight_dist == "uniform":
        low = weight_params.get("low", 1.0)
        high = weight_params.get("high", 10.0)
        w = rng.uniform(low, high, size=m)
    elif weight_dist == "exponential":
        scale = weight_params.get("scale", 1.0)
        w = rng.exponential(scale, size=m)
    elif weight_dist == "lognormal":
        mean = weight_params.get("mean", 0.0)
        sigma = weight_params.get("sigma", 1.0)
        w = rng.lognormal(mean, sigma, size=m)
    elif weight_dist == "binary":
        probs = weight_params.get("probs", [0.25, 0.25, 0.25, 0.25])
        w = rng.choice([0.25, 0.5, 0.75, 1.0], size=m, p=probs)
    else:
        raise ValueError(f"Unknown weight_dist={weight_dist!r}")

    # Attach weights
    for (edge, wt) in zip(G.edges(), w):
        G.edges[edge]["weight"] = float(wt)

    # Random partition of nodes into two groups of size n
    nodes = np.arange(2 * n)
    rng.shuffle(nodes)
    partA = nodes[:n].tolist()
    partB = nodes[n:].tolist()

    # Compute distances A -> B
    C = np.full((n, n), np.inf, dtype=float)
    for i, src in enumerate(partA):
        # Dijkstra from src
        dist_map = nx.single_source_dijkstra_path_length(G, src, weight="weight")
        # Fill row i for targets in partB
        for j, tgt in enumerate(partB):
            if tgt in dist_map:
                C[i, j] = dist_map[tgt]

    return (C, partA, partB, G) if return_graph else (C, partA, partB)

###################################################################

if __name__ == "__main__":
    n = 4

    full_rank_P = jnp.array(list(enumerate_permutation_matrices(n)))
    low_rank_P, low_rank_Q, low_rank_R = zip(*list(enumerate_low_rank_plans(n)))
    low_rank_P = jnp.array(low_rank_P)
    low_rank_Q = jnp.array(low_rank_Q)
    low_rank_R = jnp.array(low_rank_R)

    cost_type = "random_partitioned"
    rng = np.random.default_rng()
    num_nonzeros = n * n + 1
    opt_C = None
    num_counterexamples = 0
    optimal_gaps = [1.0]
    best_solution = None
    for _ in tqdm(range(1000000)):
        points_X = rng.random((n, 3))
        points_Y = rng.random((n, 3))
        if cost_type == "euclidean":
            C = jnp.array(jnp.sqrt(jnp.sum((points_X[:, None, :] - points_Y[None, :, :]) ** 2, axis=2)), dtype=jnp.float32)
        elif cost_type == "squared_euclidean":
            C = jnp.array(jnp.sum((points_X[:, None, :] - points_Y[None, :, :]) ** 2, axis=2), dtype=jnp.float32)
        elif cost_type == "manhattan":
            C = jnp.array(jnp.sum(jnp.abs(points_X[:, None, :] - points_Y[None, :, :]), axis=2), dtype=jnp.float32)
        elif cost_type == "random_partitioned":
            C, _, _ = random_partitioned_distance_matrix(n, p=0.5, return_graph=False, weight_dist="binary")
            C = jnp.array(C, dtype=jnp.float32)
        
        if jnp.isinf(C).any():
            continue

        result = check_counterexample(C, low_rank_P, low_rank_Q, low_rank_R, full_rank_P)
        if not result["is_counterexample"]: 
            continue

        gap = result["min_separator_obj"] / result["min_obj"]
        logger.info(f"Found counterexample with gap {gap:.4f}, max gap so far {optimal_gaps[-1]:.4f}")
        if gap > optimal_gaps[-1]:
            optimal_gaps.append(gap)
            logger.info(f"New best gap: {optimal_gaps[-1]:.4f}")
            best_solution = (C, points_X, points_Y, result)
        else:
            optimal_gaps.append(optimal_gaps[-1])
        
        num_counterexamples += 1
        if C.sum() < num_nonzeros:
            opt_C = C
            num_nonzeros = C.sum()
        
    logger.info(f"Most parsimonious counterexample is:")
    print(best_solution)
    logger.info(f"Fraction of counterexamples: {num_counterexamples / 10000:.4f}")