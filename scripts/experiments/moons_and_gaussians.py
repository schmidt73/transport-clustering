import sys
import math
import random
import torch

import argparse as ap
import numpy as np

from torchdyn.datasets import generate_moons
from scipy.spatial import distance
from ott.geometry import pointcloud, geometry

def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data

def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.5)
    return x0 * 3 - 1

def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.2).float()

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("-n", "--n", type=int, default=100)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-o", "--output", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
 
    batch_size1 = args.n
    batch_size2 = args.n

    g1 = np.ones((batch_size1)) / batch_size1
    g2 = np.ones((batch_size2)) / batch_size2

    x0 = sample_8gaussians(batch_size1)
    x1 = sample_moons(batch_size2)

    C = distance.cdist(x0, x1)
    C = C / C.max()

    if args.output is None:
        np.savetxt(sys.stdout, C, fmt='%.4f')
    else:
        np.savetxt(args.output + "_cost_matrix.txt", C, fmt='%.4f')
        np.savetxt(args.output + "_X_matrix.txt", x0)
        np.savetxt(args.output + "_Y_matrix.txt", x1)
