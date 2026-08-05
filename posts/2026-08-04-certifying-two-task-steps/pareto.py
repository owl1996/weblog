"""Pareto-layer reduction for the two-constraint mask problem.

Claim. Removing coordinate i shifts both constraint sums by (-a_i, -b_i). If a
removed i is dominated by a kept j (a_j <= a_i and b_j <= b_i), swapping i for j
keeps the cardinality and weakly increases both sums, so it preserves strict
feasibility. Iterating, some optimum has no removed coordinate with a kept
dominator. Any point in Pareto layer l (toward -inf,-inf) has a chain of l-1
distinct dominators, all then forced into the removal set, so |R| >= l. Hence an
optimum lies entirely within the first |R*| layers -- and the greedy's removal
count is an upper bound on |R*|.

Consequence: the ILP only ever needs the first K layers, ~K*ln(d) points under
independence, instead of all d.

Run: python pareto.py
"""
import os, sys
import numpy as np
from bisect import bisect_right

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import biconstrained as B


def pareto_layers(a, b, kmax):
    """Layer index (1-based) of each point, toward (-inf,-inf); 0 = beyond kmax.

    Patience-sorting structure: sweep in increasing a, keep the running minimum
    of b per layer. Those minima are non-decreasing in the layer index, so the
    layer of a new point is found by binary search. O(d log d), dominated by the
    sort. Ties in a are broken by b ascending and the comparison on b is strict,
    which can only push a point to a later layer -- conservative, never unsafe.
    """
    order = np.lexsort((b, a))
    layer = np.zeros(len(a), dtype=np.int32)
    tails = []                                  # tails[l] = min b so far in layer l+1
    for i in order:
        bi = b[i]
        l = bisect_right(tails, -bi, key=lambda t: -t) if False else None
        # tails is non-decreasing; find first l with tails[l] > bi
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] > bi: hi = mid
            else:               lo = mid + 1
        l = lo
        if l >= kmax:
            continue                            # beyond the layers we need
        if l == len(tails): tails.append(bi)
        else:               tails[l] = bi
        layer[i] = l + 1
    return layer


def exact_on_subset(a, b, idx):
    """Exact ILP restricted to `idx`, with the full sums as the right-hand side."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    A, Bs = a.sum(), b.sum()
    if A > 0 and Bs > 0:
        return 0, True
    if len(idx) == 0:
        return None, False
    da = 1e-6 * np.abs(a).sum(); db = 1e-6 * np.abs(b).sum()
    res = milp(c=np.ones(len(idx)),
               constraints=LinearConstraint(np.vstack([a[idx], b[idx]]),
                                            lb=[-np.inf, -np.inf], ub=[A - da, Bs - db]),
               integrality=np.ones(len(idx)), bounds=Bounds(0, 1))
    return (int(round(res.x.sum())), True) if res.success else (None, False)


if __name__ == "__main__":
    inst = [x for s in range(4) for x in B.collect(s)]
    inst = [(a, b) for a, b in inst if not B.degenerate(a, b)]
    print(f"{len(inst)} genuine two-constraint instances, d = {len(inst[0][0])}\n")

    mismatch = 0; sizes = []; full_sizes = []; layers_used = []
    for a, b in inst:
        n_full, ok_full = B.exact(a, b)
        if not ok_full or n_full == 0:
            continue
        n_greedy, ok_g = B.sweep(a, b)
        kbar = n_greedy if ok_g else 64                 # greedy gives the upper bound
        lay = pareto_layers(a, b, kbar)
        idx = np.where((lay > 0) & ((a < 0) | (b < 0)))[0]
        n_sub, ok_sub = exact_on_subset(a, b, idx)
        full_cand = int(((a < 0) | (b < 0)).sum())
        sizes.append(len(idx)); full_sizes.append(full_cand); layers_used.append(kbar)
        if not ok_sub or n_sub != n_full:
            mismatch += 1
            if mismatch <= 3:
                print(f"  MISMATCH: full={n_full} subset={n_sub} kbar={kbar} |idx|={len(idx)}")

    sizes = np.array(sizes); full_sizes = np.array(full_sizes)
    print(f"instances checked: {len(sizes)}")
    print(f"optimum differs when restricted to the first K layers: {mismatch}"
          f"  <-- must be 0\n")
    print(f"candidate set, full  (a<0 or b<0):   median {np.median(full_sizes):8.0f}  "
          f"max {full_sizes.max()}")
    print(f"candidate set, Pareto-restricted:    median {np.median(sizes):8.0f}  "
          f"max {sizes.max()}")
    print(f"reduction factor:                    median {np.median(full_sizes/np.maximum(sizes,1)):.1f}x")
    d = len(inst[0][0])
    print(f"\npredicted ~ K*ln(d) = {np.median(layers_used):.0f} * {np.log(d):.1f}"
          f" = {np.median(layers_used)*np.log(d):.0f};  observed median {np.median(sizes):.0f}")
