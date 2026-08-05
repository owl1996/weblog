"""Exhaustive validation of the Pareto-layer containment proposition.

The claim is combinatorial and its truth does not depend on d, so it can be
tested by brute force on small instances -- entirely decoupled from the
ill-conditioned real instances and from any MILP solver.

Two families:
  - random continuous coefficients (ties have measure zero)
  - adversarial small-integer coefficients with deliberate ties, which is the
    case the exchange argument's proof sketch waves at: with equalities, a
    removed coordinate can have a kept dominator that is equal rather than
    strictly smaller, and the swap must still be shown to preserve feasibility.

Run: python test_pareto.py
"""
import numpy as np

import pareto as P


D = 14
_MASKS = ((np.arange(1 << D)[:, None] >> np.arange(D)) & 1).astype(np.float64)  # 2^D x D
_POP = _MASKS.sum(1).astype(np.int32)


def brute_optimum(a, b, restrict=None):
    """Smallest |R| with both remaining sums > 0, by enumerating every subset."""
    A, B = a.sum(), b.sum()
    if A > 0 and B > 0:
        return 0, ()
    ok = (A - _MASKS @ a > 0) & (B - _MASKS @ b > 0)
    if restrict is not None:
        allowed = np.zeros(len(a)); allowed[np.asarray(restrict, dtype=int)] = 1
        ok &= (_MASKS @ (1 - allowed) == 0)          # subsets using only allowed indices
    if not ok.any():
        return None, None
    k = int(_POP[ok].min())
    return k, ()


def check(a, b):
    """Does some optimum live in the first |R*| Pareto layers?"""
    kstar, _ = brute_optimum(a, b)
    if kstar is None or kstar == 0:
        return None
    lay = P.pareto_layers(a, b, kstar)          # layers 1..kstar, 0 = beyond
    cand = np.where(lay > 0)[0]
    krestricted, _ = brute_optimum(a, b, cand)
    return kstar, krestricted


def run(name, gen, n):
    ok = bad = skipped = 0
    worst = None
    for t in range(n):
        a, b = gen(t)
        r = check(a, b)
        if r is None:
            skipped += 1; continue
        kstar, krest = r
        if krest == kstar:
            ok += 1
        else:
            bad += 1
            if worst is None:
                worst = (a.copy(), b.copy(), kstar, krest)
    print(f"{name:44s} ok {ok:6d}   VIOLATIONS {bad:4d}   (skipped {skipped})")
    if worst is not None:
        a, b, ks, kr = worst
        print(f"    counterexample: optimum {ks}, restricted {kr}")
        print(f"    a = {np.round(a,3).tolist()}")
        print(f"    b = {np.round(b,3).tolist()}")
    return bad


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    total = 0

    total += run("random continuous, d=14",
                 lambda t: (rng.normal(0, 1, D), rng.normal(0, 1, D)), 4000)

    total += run("continuous, correlated (rho=+0.6)",
                 lambda t: (lambda u, v: (u, 0.6 * u + 0.8 * v))(
                     rng.normal(0, 1, D), rng.normal(0, 1, D)), 4000)

    total += run("small integers in [-3,3] (many ties)",
                 lambda t: (rng.integers(-3, 4, D).astype(float),
                            rng.integers(-3, 4, D).astype(float)), 20000)

    total += run("ternary {-1,0,1} (maximal ties)",
                 lambda t: (rng.integers(-1, 2, D).astype(float),
                            rng.integers(-1, 2, D).astype(float)), 20000)

    def duplicated(t):
        """Deliberate exact duplicates: dominators equal, not strictly smaller."""
        k = D // 2
        a0 = rng.integers(-2, 3, k).astype(float)
        b0 = rng.integers(-2, 3, k).astype(float)
        return np.tile(a0, 2), np.tile(b0, 2)
    total += run("exact duplicate points (equality dominance)", duplicated, 20000)

    def onedim(t):
        """a == b: the two constraints coincide, layers degenerate to a total order."""
        a0 = rng.integers(-3, 4, D).astype(float)
        return a0, a0.copy()
    total += run("a == b (degenerate, single constraint)", onedim, 10000)

    print(f"\nTOTAL VIOLATIONS: {total}")
