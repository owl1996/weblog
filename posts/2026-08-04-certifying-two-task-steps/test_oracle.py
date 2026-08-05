"""Cross-checks on the layer-extraction oracle itself.

test_pareto.py replicates the *enumerator*, not the oracle: both runs call the
same pareto_layers, so a bug there would pass both identically. Two independent
checks:

  1. An O(d^2) naive implementation of the same layer decomposition, compared
     point-by-point against the O(d log d) sweep.
  2. How much the restriction actually restricts. If the candidate set were
     nearly everything, the containment test would pass vacuously and prove
     nothing -- this measures the bite.

Run: python test_oracle.py
"""
import numpy as np
import pareto as P


def naive_layers(a, b, kmax):
    """O(d^2) peeling. j precedes i iff (a_j,b_j) <= (a_i,b_i) componentwise and
    (a_j, b_j, j) < (a_i, b_i, i) lexicographically -- ties broken by index, so
    duplicate points land in successive layers, matching the sweep's convention.
    """
    n = len(a)
    layer = np.zeros(n, dtype=np.int32)
    remaining = list(range(n))
    for l in range(1, kmax + 1):
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if j == i:
                    continue
                if a[j] <= a[i] and b[j] <= b[i] and (a[j], b[j], j) < (a[i], b[i], i):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        if not front:
            break
        for i in front:
            layer[i] = l
        remaining = [i for i in remaining if layer[i] == 0]
        if not remaining:
            break
    return layer


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    D, KMAX = 14, 6

    families = {
        "continuous":        lambda: (rng.normal(0, 1, D), rng.normal(0, 1, D)),
        "integers [-3,3]":   lambda: (rng.integers(-3, 4, D).astype(float),
                                      rng.integers(-3, 4, D).astype(float)),
        "ternary {-1,0,1}":  lambda: (rng.integers(-1, 2, D).astype(float),
                                      rng.integers(-1, 2, D).astype(float)),
        "duplicated points": lambda: (np.tile(rng.integers(-2, 3, D // 2).astype(float), 2),
                                      np.tile(rng.integers(-2, 3, D // 2).astype(float), 2)),
    }

    print("1. sweep vs naive O(d^2) layer decomposition\n")
    print(f"{'family':22s} {'instances':>10s} {'mismatches':>12s}")
    for name, gen in families.items():
        bad = 0
        for _ in range(2000):
            a, b = gen()
            if not np.array_equal(P.pareto_layers(a, b, KMAX), naive_layers(a, b, KMAX)):
                bad += 1
        print(f"{name:22s} {2000:10d} {bad:12d}")

    print("\n2. does the restriction actually restrict?  (candidate set / d)\n")
    print(f"{'family':22s} {'K=1':>8s} {'K=2':>8s} {'K=4':>8s} {'K=6':>8s}")
    for name, gen in families.items():
        row = []
        for K in (1, 2, 4, 6):
            fr = []
            for _ in range(400):
                a, b = gen()
                lay = P.pareto_layers(a, b, K)
                fr.append((lay > 0).mean())
            row.append(np.mean(fr))
        print(f"{name:22s} " + "".join(f"{v:8.1%}" for v in row))

    # the same, at the scale that matters
    print("\n3. restriction at larger d (continuous, K = 6)\n")
    print(f"{'d':>8s} {'candidates':>12s} {'fraction':>10s} {'K*ln(d)':>10s}")
    for d in (100, 1000, 10000, 100000):
        a = rng.normal(0, 1, d); b = rng.normal(0, 1, d)
        lay = P.pareto_layers(a, b, 6)
        print(f"{d:8d} {(lay>0).sum():12d} {(lay>0).mean():10.3%} {6*np.log(d):10.1f}")
