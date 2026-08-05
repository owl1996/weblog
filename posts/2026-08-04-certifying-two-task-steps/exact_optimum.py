"""Certified optima for the two-constraint mask problem, trusting no solver.

The problem, with c = -a, e = -b, targets Ta = delta - sum(a) > 0, Te = delta - sum(b) > 0:

    min  sum_i x_i   s.t.   c'x >= Ta,   e'x >= Te,   x in {0,1}^n

Only two constraints, so the LP dual is two-dimensional:

    g(u,v) = u*Ta + v*Te - sum_i max(0, u*c_i + v*e_i - 1),    u,v >= 0

and by weak duality *any* (u, v) >= 0 gives a valid lower bound -- no simplex, no
solver status to trust. Maximise g numerically to pick (u, v), then re-evaluate
it in exact rational arithmetic: float64 values are exact rationals, so the bound
is certified. Round up, compare against a verified feasible solution; when they
meet, optimality is proven.

Run: python exact_optimum.py
"""
import numpy as np
from fractions import Fraction as Fr

import bicontrained as B
import certificate as C


def dual_value(c, e, Ta, Te, u, v, exact=False):
    if exact:
        u, v = Fr(u).limit_denominator(10**9), Fr(v).limit_denominator(10**9)
        tot = u * Ta + v * Te
        for ci, ei in zip(c, e):
            s = u * ci + v * ei - 1
            if s > 0:
                tot -= s
        return tot
    s = u * c + v * e - 1.0
    return u * Ta + v * Te - np.maximum(s, 0.0).sum()


def best_dual(c, e, Ta, Te):
    """Maximise the concave 2-D dual: coarse grid then local refinement."""
    best = (0.0, 0.0, 0.0)
    scale = max(1.0 / max(np.abs(c).max(), 1e-300), 1.0 / max(np.abs(e).max(), 1e-300))
    grid = np.concatenate([[0.0], np.geomspace(scale * 1e-3, scale * 1e3, 40)])
    for u in grid:
        for v in grid:
            g = dual_value(c, e, Ta, Te, u, v)
            if g > best[2]:
                best = (u, v, g)
    u, v, _ = best
    step = max(u, v, 1e-12) * 0.5
    for _ in range(60):                                   # coordinate refinement
        improved = False
        for du, dv in ((step, 0), (-step, 0), (0, step), (0, -step)):
            nu, nv = max(u + du, 0.0), max(v + dv, 0.0)
            g = dual_value(c, e, Ta, Te, nu, nv)
            if g > best[2]:
                best = (nu, nv, g); u, v = nu, nv; improved = True
        if not improved:
            step /= 2
    return best[0], best[1]


def certified(a, b):
    """Returns (upper_bound, certified_lower_bound, proven) using exact arithmetic."""
    A, Bs = a.sum(), b.sum()
    da = 1e-6 * np.abs(a).sum(); db = 1e-6 * np.abs(b).sum()
    c, e = -a, -b
    Ta, Te = da - A, db - Bs                              # need c'x >= Ta, e'x >= Te
    if Ta <= 0 and Te <= 0:
        return 0, 0, True

    n_ub, ok = B.sweep(a, b)                              # any feasible mask is an upper bound
    if not ok:
        return None, None, False
    # verify that upper bound in exact rationals
    A0, B0 = a.sum(), b.sum(); cand = np.where((a < 0) | (b < 0))[0]
    best_set = None
    for th in np.linspace(0, 1, 21):
        order = cand[np.argsort(th * a[cand] + (1 - th) * b[cand])]
        Aa, Bb, rem = A0, B0, []
        for i in order:
            if Aa > 0 and Bb > 0: break
            Aa -= a[i]; Bb -= b[i]; rem.append(i)
        if Aa > 0 and Bb > 0 and (best_set is None or len(rem) < len(best_set)):
            best_set = rem
    ra = sum(Fr(a[i]) for i in best_set); rb = sum(Fr(b[i]) for i in best_set)
    if not (Fr(A) - ra > 0 and Fr(Bs) - rb > 0):
        return None, None, False                          # upper bound not actually feasible
    n_ub = len(best_set)

    u, v = best_dual(c, e, Ta, Te)
    lb = dual_value([Fr(x) for x in c], [Fr(x) for x in e], Fr(Ta), Fr(Te), u, v, exact=True)
    lb_int = int(-((-lb) // 1))                           # ceil of an exact Fraction
    return n_ub, max(lb_int, 0), (max(lb_int, 0) == n_ub)


if __name__ == "__main__":
    raw = [x for s in B.SEEDS for x in B.collect(s)]
    inst = [(a, b) for a, b in raw if not B.degenerate(a, b)]
    print(f"{len(inst)} well-posed two-constraint instances\n")
    print(f"{'#':>4s} {'sweep (UB)':>11s} {'certified LB':>13s} {'proven optimal?':>16s}")
    proven = gaps = failed = 0
    for k, (a, b) in enumerate(inst):
        ub, lb, ok = certified(a, b)
        if ub is None:
            failed += 1; continue
        if ok: proven += 1
        else:  gaps += 1
        if k < 15 or ok is False:
            print(f"{k:4d} {ub:11d} {lb:13d} {str(ok):>16s}")
    print(f"\nproven optimal: {proven}   still a gap: {gaps}   unusable: {failed}")
    print("Every bound above is a weak-duality certificate evaluated in exact")
    print("rational arithmetic -- no solver status is trusted anywhere.")
