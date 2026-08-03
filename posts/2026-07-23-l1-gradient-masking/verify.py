"""
Verification for "Gradient masking for two-task alignment".

Regenerates every number in the post:
  1. the d=3 counterexample,
  2. the gradient-regime table (how often the greedy is suboptimal),
  3. the dimension scaling at rho = -0.95,
  4. the branch-switch split testing the Proposition.

Brute force enumerates all 2^d masks, so d <= 14. Runs on CPU in a few minutes.

    python verify.py
"""

import itertools

import numpy as np

# The experiments below compare combinatorial outcomes on random floats, where
# exact ties have probability zero; the epsilon margin of the pseudo-code is a
# numerical guard for real gradients and is set to 0 here so the comparison
# against brute force is exact.
EPS = 0.0


def greedy(g1, g2, alpha, beta, eps=EPS):
    """The pseudo-code of the post. Returns (mask or None, branch_switched)."""
    d = len(g1)
    s = g1 * g2
    conflicting = s <= 0  # dominance rule (I2)

    sA = s + (alpha / beta) * g1**2  # = a_i / beta
    sB = s + (beta / alpha) * g2**2  # = b_i / alpha
    piA = [i for i in np.argsort(sA, kind="stable") if conflicting[i]]
    piB = [i for i in np.argsort(sB, kind="stable") if conflicting[i]]

    m = np.ones(d)
    S11, S22, S12 = g1 @ g1, g2 @ g2, g1 @ g2
    pA = pB = 0
    branches = []

    while -S12 >= min((alpha / beta) * S11, (beta / alpha) * S22) - eps:
        on_A = (alpha / beta) * S11 <= (beta / alpha) * S22
        branches.append(on_A)
        if on_A:
            while pA < len(piA) and m[piA[pA]] == 0:
                pA += 1
            if pA == len(piA):
                return None, _switched(branches)
            i, score = piA[pA], sA[piA[pA]]
        else:
            while pB < len(piB) and m[piB[pB]] == 0:
                pB += 1
            if pB == len(piB):
                return None, _switched(branches)
            i, score = piB[pB], sB[piB[pB]]

        if score >= 0:  # every remaining removal would increase V
            return None, _switched(branches)

        m[i] = 0
        S11 -= g1[i] ** 2
        S22 -= g2[i] ** 2
        S12 -= g1[i] * g2[i]

    if m.sum() == 0:
        return None, _switched(branches)
    return m, _switched(branches)


def _switched(branches):
    return len(set(branches)) > 1


def brute(g1, g2, alpha, beta):
    """Feasible mask of maximum cardinality, or None."""
    d = len(g1)
    delta = alpha * g1 + beta * g2
    for k in range(d, -1, -1):
        for support in itertools.combinations(range(d), k):
            m = np.zeros(d)
            m[list(support)] = 1
            if (m * delta) @ g1 > 0 and (m * delta) @ g2 > 0:
                return m
    return None


def card(m):
    return -1 if m is None else int(m.sum())


def draw(rng, d, rho):
    """g1 standard Gaussian, g2 correlated at rho (rho=None means independent)."""
    u = rng.normal(size=d)
    v = rng.normal(size=d)
    g2 = v if rho is None else rho * u + np.sqrt(1 - rho**2) * v
    return u, g2, rng.uniform(0.2, 3), rng.uniform(0.2, 3)


def sweep(rng, n, d_range, rho):
    """Returns (frac full mask already feasible, frac suboptimal, frac of those that FAIL)."""
    trivial = sub = fails = 0
    for _ in range(n):
        d = int(rng.integers(*d_range)) if isinstance(d_range, tuple) else d_range
        g1, g2, alpha, beta = draw(rng, d, rho)
        gm, _ = greedy(g1, g2, alpha, beta)
        bm = brute(g1, g2, alpha, beta)
        trivial += card(bm) == d and card(gm) == d
        if card(gm) < card(bm):
            sub += 1
            fails += gm is None
    return trivial / n, sub / n, fails / max(sub, 1)


def main():
    # --- 1. counterexample ---------------------------------------------------
    print("1. Counterexample (d=3, alpha=beta=1)")
    g1 = np.array([-3.0, -2.0, -2.0])
    g2 = np.array([1.0, 1.0, 3.0])
    delta = g1 + g2
    print(f"   Delta = {delta}   a = {delta * g1}   b = {delta * g2}")
    print(f"   a_i + b_i = {delta * g1 + delta * g2}  (= Delta_i^2, identity I1)")
    gm, _ = greedy(g1, g2, 1.0, 1.0)
    bm = brute(g1, g2, 1.0, 1.0)
    print(f"   greedy -> {gm}      brute force -> {bm}")
    lo, hi = -(g1 @ g2) / (g1 @ g1), (g2 @ g2) / -(g1 @ g2)
    print(f"   admissible alpha/beta ratios (eq-ratio): ({lo:.3f}, {hi:.3f})")

    # --- 2. regime table -----------------------------------------------------
    print("\n2. Gradient regimes  (d ~ U[3,14], alpha,beta ~ U(0.2,3), N=20000)")
    print(f"   {'regime':<28}{'full mask ok':>14}{'suboptimal':>13}{'of which FAIL':>15}")
    for label, rho in [("independent", None), ("rho=-0.6", -0.6), ("rho=-0.9", -0.9), ("rho=-0.99", -0.99)]:
        rng = np.random.default_rng(0)
        triv, sub, fail = sweep(rng, 20000, (3, 15), rho)
        print(f"   {label:<28}{100 * triv:13.1f}%{100 * sub:12.3f}%{100 * fail:14.0f}%")

    # --- 3. dimension scaling ------------------------------------------------
    print("\n3. Dimension scaling at rho=-0.95 (N=2500 each)")
    for d in (3, 5, 8, 11, 13):
        rng = np.random.default_rng(7)
        _, sub, _ = sweep(rng, 2500, d, -0.95)
        print(f"   d={d:2d}: {100 * sub:6.2f}% suboptimal")

    # --- 4. the Proposition --------------------------------------------------
    print("\n4. Branch-switch split (rho=-0.95, d ~ U[3,10], N=20000)")
    rng = np.random.default_rng(7)
    tot = {False: 0, True: 0}
    bad = {False: 0, True: 0}
    for _ in range(20000):
        d = int(rng.integers(3, 11))
        g1, g2, alpha, beta = draw(rng, d, -0.95)
        gm, switched = greedy(g1, g2, alpha, beta)
        if card(gm) == d and gm is not None:
            continue  # greedy never ran
        bm = brute(g1, g2, alpha, beta)
        tot[switched] += 1
        bad[switched] += card(gm) < card(bm)
    for switched, name in [(False, "no branch switch")
                           , (True, "branch switched")]:
        n, k = tot[switched], bad[switched]
        print(f"   {name:<20}{k:5d} / {n:5d} suboptimal  ({100 * k / max(n, 1):.1f}%)")

    # --- 5. eq-ratio always non-empty ---------------------------------------
    print("\n5. Does an admissible alpha/beta ratio always exist? (N=20000)")
    rng = np.random.default_rng(11)
    ok = 0
    for _ in range(20000):
        d = int(rng.integers(2, 50))
        u, v = rng.normal(size=d), rng.normal(size=d)
        rho = rng.uniform(-0.999, 0.999)
        g2 = rho * u + np.sqrt(1 - rho**2) * v
        S11, S22, S12 = u @ u, g2 @ g2, u @ g2
        lo = max(0.0, -S12 / S11) if S12 < 0 else 0.0
        hi = S22 / -S12 if S12 < 0 else np.inf
        r = (lo + hi) / 2 if np.isfinite(hi) else lo + 1
        delta = r * u + g2
        ok += delta @ u > 0 and delta @ g2 > 0
    print(f"   full mask feasible at the midpoint ratio: {ok}/20000")


if __name__ == "__main__":
    main()
