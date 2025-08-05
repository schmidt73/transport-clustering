import sys
import jax 
import jax.numpy as jnp
import itertools
from loguru import logger
import numpy as np
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
    return perms[objs == min_obj]

def check_counterexample(
    C : jnp.ndarray, 
    low_rank_P : jnp.ndarray, 
    low_rank_Q : jnp.ndarray, 
    low_rank_R : jnp.ndarray, 
    full_rank_P : jnp.ndarray
):
    P = solve_monge(C, perms=full_rank_P)
    if P.shape[0] > 1: return False
    P = P[0]
    objs = jnp.sum(low_rank_P * C[None,:], axis=[1,2])
    min_obj = jnp.min(objs)
    indices = jnp.nonzero(objs == min_obj)
    return jnp.any(P @ low_rank_R[indices] != low_rank_Q[indices])

###################################################################

if __name__ == "__main__":
    n = 10

    full_rank_P = jnp.array(list(enumerate_permutation_matrices(n)))
    low_rank_P, low_rank_Q, low_rank_R = zip(*list(enumerate_low_rank_plans(n)))
    low_rank_P = jnp.array(low_rank_P)
    low_rank_Q = jnp.array(low_rank_Q)
    low_rank_R = jnp.array(low_rank_R)

    # Generate a random nxn binary matrix C
    rng = np.random.default_rng()
    num_nonzeros = n * n + 1
    opt_C = None
    num_counterexamples = 0
    for _ in tqdm(range(10000)):
        C = jnp.array(rng.integers(0, 2, size=(n, n)), dtype=jnp.float32)
        is_counterexample = check_counterexample(C, low_rank_P, low_rank_Q, low_rank_R, full_rank_P)
        if not is_counterexample: continue
        
        num_counterexamples += 1
        if C.sum() < num_nonzeros:
            opt_C = C
            num_nonzeros = C.sum()
        
    logger.info(f"Most parsimonious counterexample is:")
    print(opt_C)
    logger.info(f"Fraction of counterexamples: {num_counterexamples / 10000:.4f}")