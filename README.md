# Transport Clustering (TC)

Code for **Transport Clustering: Solving Low-Rank Optimal Transport via Clustering** (Schmidt, Halmos, Raphael; ICML 2026).

Low-rank optimal transport (LR-OT) constrains the (non-negative) rank of a transport plan to reveal
latent structure between two datasets, but the resulting optimization problem is non-convex and
NP-hard. This repository shows that LR-OT reduces to a clustering problem: given the optimal
**full-rank** transport plan (a permutation, or its entropic relaxation), *registering* the cost
matrix through this plan turns the low-rank OT problem into an instance of *generalized K-means*
&mdash; a single symmetric clustering problem instead of a joint optimization over two coupled
factors and a shared marginal. We call this reduction **Transport Clustering** (`TC`):

1. **Transport (registration) step** &mdash; solve the full-rank OT problem for the optimal coupling
   between `X` and `Y` (a Monge permutation via the Hungarian algorithm, or its soft/entropic
   Sinkhorn relaxation for the Kantorovich variant), and use it to register the cost matrix
   `C -> C @ P^T`.
2. **Clustering step** &mdash; solve generalized K-means on the registered cost with mirror descent
   (`GKMS`, exponentiated-gradient / Sinkhorn-style updates) to obtain one low-rank factor `Q`; the
   second factor `R` is recovered for free from `Q` and `P`.

## Repository layout

```
src/
  monge_rotate.py       Monge registration + generalized K-means (JAX): monge_conjugate(),
                         gkms()/gkms_logdomain(), SDP (Burer-Monteiro) and k-means++ initializers
  monge_rotate_lr.py     Low-rank / factored version: avoids materializing C by rotating the
                         low-rank cost factors (A, B) directly; supports Sinkhorn or HiRef
                         (hierarchical refinement) as the registration step for large n
  clustering.py          Lightweight Lloyd's k-means / k-means++ used for initialization
  distance_utils.py      Cost-matrix and low-rank cost-factor utilities (e.g. squared Euclidean)
  GKMS/                  Torch implementation of generalized K-means (mirror descent + log-domain
                         Sinkhorn inner solver) used for the full-rank cost variant
  FRLC/                  Factor Relaxation with Latent Coupling low-rank OT baseline
  LatentOT/              Latent OT baselines: LOT (Lin et al.), OT, FC
  HiRef/                 Hierarchical refinement OT solver, used as a scalable registration
                         step and as a large-scale baseline

scripts/
  run_methods.py         CLI entry point comparing TC (`mr`), FRLC (`frlc`), low-rank Sinkhorn
                         (`lot`), and Lin et al. (`lin`) on a cost matrix or point clouds
  run_methods.nf          Nextflow pipeline sweeping algorithms x ranks x seeds over simulated
                         instances and publishing per-run results
  summarize_results.py    Aggregates per-run JSON summaries produced by run_methods.py/.nf
  experiments/            Synthetic benchmark generators: planted Gaussians, planted cliques,
                         random weighted graphs, concentric circles, moons-and-8-Gaussians,
                         and a hand-built hard instance for squared-Euclidean cost
  experiments_large_scale/ Large-scale evaluation harness (run_LowRank.py, eval_LowRank.py) and
                         a single-cell transcriptomics case study (single_cell.py)
  plots/                  Figure-generation scripts for transport plans and simulation results

notebooks/                Exploratory notebooks: CIFAR-10 evaluation, single-cell evaluation,
                         and Monge-registration sanity checks (mr_LR_test, mr_test_2)

examples/                 Pre-generated cost matrices / optimal plans (planted Gaussians and
                         random weighted graphs at several sizes) used to sanity-check solvers

models/                   Pretrained weights (ResNet-18) used for embedding-based experiments
                         (e.g. CIFAR-10 evaluation)
```

## Getting started

The core routines are under `src/` and are organized so that JAX (`monge_rotate*.py`, `HiRef`,
`LatentOT`) and PyTorch (`GKMS`, `FRLC`) implementations can be run side-by-side and compared
through `scripts/run_methods.py`.

Solve low-rank OT with `TC` on a precomputed cost matrix:

```bash
cd scripts
python run_methods.py --cost_matrix ../examples/graph_n100_cost_matrix.txt -r 10 -a mr -o out
```

or directly on two point clouds:

```python
import jax.numpy as jnp
from monge_rotate import monge_conjugate

Q, g, R = monge_conjugate(C, r=10)          # C: n x n cost matrix, r: target rank
P = Q @ jnp.diag(1 / g) @ R.T                # recovered low-rank transport plan
```

`run_methods.py` also runs the baselines used in the paper for comparison: `FRLC` (`-a frlc`),
low-rank Sinkhorn (`-a lot`), and the Lin et al. latent-anchor method (`-a lin`). Each run writes
the low-rank factors `(Q, g, R)` and a JSON summary (objective cost, marginal errors, runtime).

### 1. Generalized K-Means (GKMS)

`src/GKMS/GKMS.py` (Torch) and the `gkms`/`gkms_logdomain` functions in `src/monge_rotate.py`
(JAX) implement the clustering subroutine that `TC` reduces to: minimizing
`<C, Q diag(1/g) Q^T>` over hard/soft assignment matrices `Q`, via mirror descent with a
log-domain (semi-relaxed) Sinkhorn inner loop. `src/clustering.py` provides a plain Lloyd's
k-means/k-means++ used to initialize `Q`.

### 2. Low-Rank Optimal Transport via Transport Clustering

`src/monge_rotate.py::monge_conjugate` implements **Monge registration**: it computes the optimal
full-rank permutation `P` (via Sinkhorn) and runs `GKMS` on the registered cost `C @ P^T` (and,
symmetrically, `P^T @ C`), returning the better of the two low-rank solutions. `src/monge_rotate_lr.py`
implements the same reduction for large-scale, low-rank-factored costs, using either Sinkhorn or
`HiRef` (hierarchical refinement) for the registration step so that the full `n x n` cost matrix
never needs to be formed.

## Reproducibility

Synthetic benchmarks from the paper (planted Gaussians, planted cliques, random weighted graphs,
concentric circles, moons-and-8-Gaussians) can be regenerated with the scripts in
`scripts/experiments/`; `examples/` contains representative pre-generated instances. The full
sweep over algorithms, ranks, and seeds used for the paper's experiments is defined in
`scripts/run_methods.nf` and can be summarized with `scripts/summarize_results.py`. Large-scale
and single-cell evaluations are under `scripts/experiments_large_scale/`.

## Contact

For questions, discussions, or collaboration inquiries, feel free to reach out at [henri.schmidt@princeton.edu](mailto:henri.schmidt@princeton.edu) or [ph3641@princeton.edu](mailto:ph3641@princeton.edu).

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{schmidt2026transportclustering,
  title     = {Transport Clustering: Solving Low-Rank Optimal Transport via Clustering},
  author    = {Schmidt, Henri and Halmos, Peter and Raphael, Benjamin J.},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
