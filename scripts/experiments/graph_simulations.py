#!/usr/bin/env python3
"""
Generate a random weighted graph on 2n nodes, split nodes into X and Y (n each),
compute the bipartite shortest-path cost matrix C where C[i, j] = d_G(x_i, y_j),
and write C to disk.

Usage
-----
python make_cost_matrix.py -n 100 -p 0.05 --w_low 1 --w_high 10 -o costs.csv
python make_cost_matrix.py -n 1000 -p 0.01 -s 42 -o costs.npy
"""
import argparse
import numpy as np
import networkx as nx


def build_random_weighted_graph(n_total: int,
                                p: float = 0.1,
                                w_low: float = 1.0,
                                w_high: float = 10.0,
                                seed: int | None = None) -> nx.Graph:
    """Erdős–Rényi G(n_total, p) with i.i.d. edge weights ∼ U[w_low, w_high)."""
    G = nx.erdos_renyi_graph(n_total, p, seed=seed, directed=False)
    rng = np.random.default_rng(seed)
    for u, v in G.edges():
        G[u][v]["weight"] = rng.uniform(w_low, w_high)
    return G


def random_bipartition(nodes: list[int], n: int, seed: int | None = None) -> tuple[list[int], list[int]]:
    """Randomly split 'nodes' into X and Y of size n each (deterministic w.r.t. seed)."""
    if len(nodes) != 2 * n:
        raise ValueError(f"Expected 2n nodes, got {len(nodes)} for n={n}.")
    rng = np.random.default_rng(None if seed is None else seed + 1)  # offset seed for split
    perm = rng.permutation(nodes)
    X = list(perm[:n])
    Y = list(perm[n:2 * n])
    return X, Y


def bipartite_shortest_path_cost_matrix(G: nx.Graph, X: list[int], Y: list[int]) -> np.ndarray:
    """
    Return C[i, j] = weighted shortest-path distance in G from X[i] to Y[j],
    with np.inf if Y[j] is unreachable from X[i].
    """
    n = len(X)
    m = len(Y)
    C = np.full((n, m), np.inf, dtype=float)
    for i, src in enumerate(X):
        # Dijkstra from a single source to all targets.
        lengths = nx.single_source_dijkstra_path_length(G, source=src, weight="weight")
        for j, dst in enumerate(Y):
            d = lengths.get(dst)
            if d is not None:
                C[i, j] = d
    return C


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate bipartite cost matrix from a random weighted graph on 2n nodes.")
    parser.add_argument("-n", "--n", type=int, required=True,
                        help="Size per side (total vertices = 2n).")
    parser.add_argument("-p", "--prob", type=float, default=0.1,
                        help="Edge probability p for G(2n, p) (default: 0.1).")
    parser.add_argument("--w_low", type=float, default=1.0,
                        help="Inclusive lower bound for edge weights (default: 1).")
    parser.add_argument("--w_high", type=float, default=10.0,
                        help="Exclusive upper bound for edge weights (default: 10).")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Random seed (optional).")
    parser.add_argument("-o", "--output", required=True,
                        help="Destination filename; use .csv, .txt, or .npy extension.")
    args = parser.parse_args()

    n = args.n
    n_total = 2 * n

    # 1) Build random weighted graph on 2n nodes.
    G = build_random_weighted_graph(n_total, args.prob, args.w_low, args.w_high, args.seed)

    # 2) Randomly split nodes into X and Y (n each).
    nodes = list(G.nodes())
    nodes.sort()  # deterministic node ordering before permutation
    X, Y = random_bipartition(nodes, n, args.seed)

    # 3) Compute C[i, j] = d_G(x_i, y_j).
    C = bipartite_shortest_path_cost_matrix(G, X, Y)

    # Save to disk.
    if args.output.endswith(".npy"):
        np.save(args.output, C)
    elif args.output.endswith(".csv"):
        np.savetxt(args.output, C, delimiter=",")
    elif args.output.endswith(".txt"):
        np.savetxt(args.output, C)
    else:
        raise ValueError("Output filename must end with .csv, .txt, or .npy")

    print(f"Built G with {n_total} nodes; |X|=|Y|={n}.")
    print(f"Saved {C.shape[0]}x{C.shape[1]} bipartite cost matrix to '{args.output}'.")
    # If you want to inspect which vertices fell into X and Y, uncomment:
    # print('X:', X)
    # print('Y:', Y)


if __name__ == "__main__":
    main()
