#!/usr/bin/env python3
"""
Generate a random weighted graph on n nodes, compute the all‑pairs shortest‑path
distance matrix C (C[i, j] = d_G(i,j)), and write C to disk.

Usage
-----
python make_cost_matrix.py -n 100 -p 0.05 --w_low 1 --w_high 10 -o costs.csv
python make_cost_matrix.py -n 1000 -p 0.01 -s 42 -o costs.npy
"""
import argparse
import numpy as np
import networkx as nx


def build_random_weighted_graph(n: int,
                                p: float = 0.1,
                                w_low: float = 1.0,
                                w_high: float = 10.0,
                                seed: int | None = None) -> nx.Graph:
    """Erdős–Rényi G(n,p) graph with i.i.d. edge weights ∼ U[w_low, w_high)."""
    G = nx.gnp_random_graph(n, p, seed=seed, directed=False)
    rng = np.random.default_rng(seed)
    for u, v in G.edges():
        G[u][v]["weight"] = rng.uniform(w_low, w_high)
    return G


def shortest_path_cost_matrix(G: nx.Graph) -> np.ndarray:
    """Return C[i,j] = shortest‑path distance between i and j (∞ if disconnected)."""
    n = G.number_of_nodes()
    C = np.full((n, n), np.inf, dtype=float)
    for i, lengths in nx.all_pairs_dijkstra_path_length(G, weight="weight"):
        for j, d in lengths.items():
            C[i, j] = d
    return C


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random weighted graph and dump cost matrix.")
    parser.add_argument("-n", "--nodes", type=int, required=True,
                        help="Number of nodes (n).")
    parser.add_argument("-p", "--prob", type=float, default=0.1,
                        help="Edge probability p for G(n,p) (default: 0.1).")
    parser.add_argument("--w_low", type=float, default=1.0,
                        help="Inclusive lower bound for edge weights (default: 1).")
    parser.add_argument("--w_high", type=float, default=10.0,
                        help="Exclusive upper bound for edge weights (default: 10).")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Random seed (optional).")
    parser.add_argument("-o", "--output", required=True,
                        help="Destination filename; use .csv or .npy extension.")
    args = parser.parse_args()

    G = build_random_weighted_graph(args.nodes, args.prob,
                                    args.w_low, args.w_high, args.seed)
    C = shortest_path_cost_matrix(G)

    if args.output.endswith(".npy"):
        np.save(args.output, C)
    elif args.output.endswith(".txt"):
        np.savetxt(args.output, C)
    else:
        raise ValueError("Output filename must end with .csv or .npy")

    print(f"Saved {C.shape[0]}×{C.shape[1]} cost matrix to '{args.output}'.")


if __name__ == "__main__":
    main()

