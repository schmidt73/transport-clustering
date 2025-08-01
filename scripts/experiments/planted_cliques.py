from __future__ import annotations
import argparse
import sys
import random
import networkx as nx
import numpy as np
from typing import Tuple

def planted_cliques_graph(
    k: int,
    m: int,
    p_intra: float = 0.1,
    p_inter: float = 0.1,
    weight_range: Tuple[float, float] = (1.0, 10.0),
    seed: int | None = None
) -> nx.Graph:
    rng = random.Random(seed)
    G = nx.Graph()

    for c in range(k):
        nodes = [f"C{c}_{i}" for i in range(m)]
        G.add_nodes_from(nodes, clique=c)
        for u in nodes:
            for v in nodes:
                if u != v and rng.random() < p_intra:
                    w = rng.uniform(*weight_range)
                    G.add_edge(u, v, weight=w, intra=True)

    all_nodes = list(G.nodes)
    for i, u in enumerate(all_nodes):
        for v in all_nodes[i + 1 :]:
            if G.nodes[u]["clique"] != G.nodes[v]["clique"] and rng.random() < p_inter:
                w = rng.uniform(*weight_range)
                G.add_edge(u, v, weight=w, intra=False)

    return G

def shortest_path_cost_matrix(G: nx.Graph) -> np.ndarray:
    """Return C[i,j] = shortest-path distance between i and j (∞ if disconnected)."""
    names_to_indices = {name: i for i, name in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, names_to_indices)  # relabel nodes to indices
    n = G.number_of_nodes()
    C = np.full((n, n), np.inf, dtype=float)
    for i, lengths in nx.all_pairs_dijkstra_path_length(G, weight="weight"):
        for j, d in lengths.items():
            C[i, j] = d
    return C, names_to_indices

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate k planted cliques graph")
    p.add_argument("-k", type=int, required=True, help="number of cliques")
    p.add_argument("-m", type=int, required=True, help="size of each clique")
    p.add_argument("--p-intra", type=float, default=0.1, help="probability of intra‑clique edge (default: 0.1)")
    p.add_argument("--p-inter", type=float, default=0.1, help="probability of inter‑clique edge (default: 0.1)")
    p.add_argument("--range", nargs=2, type=float, default=(1.0, 1.0), help="weight range for edge weights")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("-o", "--output", type=str, default=None, help="output prefix")
    return p.parse_args()

def main():
    args = parse_args()
    G = planted_cliques_graph(
        k=args.k,
        m=args.m,
        p_intra=args.p_intra,
        p_inter=args.p_inter,
        weight_range=tuple(args.range),
        seed=args.seed,
    )

    C, names_to_indices = shortest_path_cost_matrix(G)
    
    P = np.zeros_like(C)
    for i in G.nodes:
        for j in G.nodes:
            if G.nodes[i]["clique"] == G.nodes[j]["clique"]:
                P[names_to_indices[i], names_to_indices[j]] = 1.0
                P[names_to_indices[j], names_to_indices[i]] = 1.0

    P = P / P.sum()

    cost = (P * (C / C.max())).sum()
    if args.output:
        np.savetxt(args.output + "_cost_matrix.txt", C, fmt="%.4f")
        np.savetxt(args.output + "_optimal_plan.txt", P, fmt="%.4f")
        metadata = {
            "k": args.k,
            "m": args.m,
            "p_intra": args.p_intra,
            "p_inter": args.p_inter,
            "weight_range": args.range,
            "seed": args.seed,
            "cost": cost
        }
        with open(args.output + "_metadata.json", "w") as f:
            import json
            json.dump(metadata, f, indent=4)
    

if __name__ == "__main__":
    main()

