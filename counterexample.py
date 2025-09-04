import itertools, numpy as np
import seaborn as sns
import pandas as pd

eps = 0.001
X = np.array([
    [-eps, 2 + eps],
    [0.0, 1.0],
    [2 + eps, -eps],
    [2.0, 1.0]
])

Y = np.array([
    [-eps, 2 + eps],
    [1 - eps, 0.0],
    [2 + eps, -eps],
    [1 + eps, 2.0]
])

C = np.array((np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)), dtype=np.float32)

np.savetxt("C.txt", C, fmt='%.4f')
n = C.shape[0]

perm_costs = {perm: sum(C[i, perm[i]] for i in range(n))
              for perm in itertools.permutations(range(n))}
print(perm_costs)
best_cost = min(perm_costs.values())
best_perms = [p for p, c in perm_costs.items() if c == best_cost]
assert len(best_perms) == 1
print("Minimum assignment cost:", best_cost)
print("Unique optimal permutation:", best_perms[0], "\n")

def complement(t):
    return tuple(sorted(set(range(n)) - set(t)))

row_parts = list(itertools.combinations(range(n), 2))
col_parts = row_parts.copy()

couplings = []
for rp in row_parts:
    X1, X2 = set(rp), set(complement(rp))
    for cp in col_parts:
        Y1, Y2 = set(cp), set(complement(cp))
        print(X1, X2, Y1, Y2)
        if len(X1) != len(Y1):
            continue
        block_sum = (1.0 / len(X1)) * sum(C[i, j] for i in X1 for j in Y1) + \
                    (1.0 / len(X2)) * sum(C[i, j] for i in X2 for j in Y2)
        couplings.append((block_sum, rp, cp))
        print(couplings[-1])

min_rank2 = min(c[0] for c in couplings)
mins = [c for c in couplings if abs(c[0] - min_rank2) < 1e-9]

print(f"Minimal rank-2 objective value: {min_rank2:.2f}")
print(f"Number of minimisers: {len(mins)}\n")

for k, (cost, rp, cp) in enumerate(mins, 1):
    X1, X2 = set(rp), set(complement(rp))
    Y1, Y2 = set(cp), set(complement(cp))

    split = [i+1 for i in range(n)        # 1-based indices
             if (i in X1) != (i in Y1)]   # cluster mismatch

    print(f"Minimiser {k}:")
    print("  Row clusters X1,X2 =", tuple(x for x in sorted(X1)),
          tuple(x for x in sorted(X2)))
    print("  Col clusters Y1,Y2 =", tuple(y for y in sorted(Y1)),
          tuple(y for y in sorted(Y2)))
    print("  Monge pairs split at indices:", split, "\n")

import matplotlib.pyplot as plt

# Create the scatter plot
plt.figure(figsize=(4.5, 4.5))
plt.scatter([p[0] for p in X], [p[1] for p in X],
            marker='o', label='$x_i$')
plt.scatter([p[0] for p in Y], [p[1] for p in Y],
            marker='^', label='$y_i$')

# Annotate the points
for i, (x, y) in enumerate(X, 1):
    plt.text(x, y, f'  x{i}', verticalalignment='bottom')
for i, (x, y) in enumerate(Y, 1):
    plt.text(x, y, f'  y{i}', verticalalignment='bottom')

    # Add grid to the plot
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('$x$-coordinate')
plt.ylabel('$y$-coordinate')
# fix scale
plt.xlim(-0.5, 4.0)
plt.ylim(-0.5, 4.0)
#plt.legend()
plt.tight_layout()
plt.show()