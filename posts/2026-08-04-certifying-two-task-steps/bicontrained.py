"""Is the Lagrangian sweep optimal for the two-constraint mask problem?

Under a proper prior with heterogeneous lambda_k, the single violation V no
longer exists and the repair becomes: minimise removals subject to TWO linear
inequalities. That problem is a 2-constraint knapsack -- NP-hard in general, but
the instances here are tiny (~24 removals out of 9000) and solve exactly with
HiGHS in milliseconds. So the sweep-vs-optimum gap is measurable, not a matter
of opinion.

Collects real instances from the digits two-task run, then compares:
  - Lagrangian sweep: sort on theta*a + (1-theta)*b, greedy, best over a grid
  - exact ILP (scipy.optimize.milp / HiGHS)

Run: python bicontrained.py     (~5 min on CPU)
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import milp, LinearConstraint, Bounds

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "2026-08-03-two-dimensional-adam"))
import experiment as E
import certificate as C

THETAS = np.linspace(0, 1, 21)
SEEDS = range(6)


def layer_lambda(model, ghat, sigma):
    """lambda_i = tau^2/(tau^2+sigma_i^2), with tau^2 pooled per parameter tensor."""
    lam = torch.empty_like(ghat); i = 0
    for p in model.parameters():
        n = p.numel(); sl = slice(i, i + n)
        tau2 = (ghat[sl].pow(2).mean() - sigma[sl].pow(2).mean()).clamp(min=0)
        lam[sl] = tau2 / (tau2 + sigma[sl].pow(2) + 1e-12)
        i += n
    return lam


def sweep(a, b):
    """Lagrangian sweep. Returns (n_removed, ok).

    The candidate set is {a<0 or b<0}: removing a coordinate positive in both
    constraints can only hurt. Within it we scan the whole order rather than
    stopping when the combined key turns positive -- a coordinate with key >= 0
    may still have a_i < 0 and help that constraint, and cutting the scan there
    would make this baseline artificially weak.
    """
    A0, B0 = a.sum(), b.sum()
    cand = np.where((a < 0) | (b < 0))[0]
    if len(cand) == 0:
        return None, False
    best = None
    for th in THETAS:
        key = th * a[cand] + (1 - th) * b[cand]
        order = cand[np.argsort(key)]
        A, B, n = A0, B0, 0
        for i in order:
            if A > 0 and B > 0:
                break
            A -= a[i]; B -= b[i]; n += 1
        if A > 0 and B > 0 and (best is None or n < best):
            best = n
    return (best, True) if best is not None else (None, False)


def degenerate(a, b, eps=1e-3):
    """Guard-rail: nothing to certify.

    If one task's empirical-Bayes prior collapses (lambda == 0 across every
    layer) its constraint vector vanishes and sum m_i a_i > 0 is unsatisfiable
    for *any* mask -- vacuously infeasible, not conflicted. Likewise a near-total
    cancellation means the constraint sits at the numerical floor. Neither is a
    conflict, and neither should be handed to a mask solver.
    """
    sa, sb = np.abs(a).sum(), np.abs(b).sum()
    if sa == 0 or sb == 0:
        return True
    return min(abs(a.sum()) / sa, abs(b.sum()) / sb) < eps


def exact(a, b):
    """Exact minimum removals via HiGHS. Returns (n_removed, ok)."""
    A, B = a.sum(), b.sum()
    if A > 0 and B > 0:
        return 0, True
    cand = np.where((a < 0) | (b < 0))[0]        # removing a>0,b>0 coord only hurts
    if len(cand) == 0:
        return None, False
    # delta must sit above the solver's feasibility tolerance *relative to the
    # coefficients*, or the strict inequality is not enforced at all.
    da = 1e-6 * np.abs(a).sum(); db = 1e-6 * np.abs(b).sum()
    res = milp(c=np.ones(len(cand)),
               constraints=LinearConstraint(np.vstack([a[cand], b[cand]]),
                                            lb=[-np.inf, -np.inf],
                                            ub=[A - da, B - db]),
               integrality=np.ones(len(cand)), bounds=Bounds(0, 1))
    if not res.success:
        return None, False
    return int(round(res.x.sum())), True


def collect(seed):
    (Xo, yo), (Xn, yn), *_ = E.make_data(seed)
    gen = torch.Generator().manual_seed(seed)
    base = E.mlp(seed); opt = torch.optim.Adam(base.parameters(), lr=3e-3)
    for _ in range(400):
        i = torch.randint(len(Xo), (E.BATCH,), generator=gen)
        opt.zero_grad(); F.cross_entropy(base(Xo[i]), yo[i]).backward(); opt.step()
    model = E.mlp(seed); model.load_state_dict(base.state_dict())
    outer = torch.optim.Adam(model.parameters(), lr=C.LR)
    g = torch.Generator().manual_seed(seed + 1000)
    al, be = C.ALPHA, C.BETA
    out = []
    for _ in range(C.STEPS):
        g1, s1 = E.task_gradient(model, Xn, yn, g)
        g2, s2 = E.task_gradient(model, Xo, yo, g)
        f = C.focus(g1, s1, g2, s2)
        l1 = layer_lambda(model, g1, s1); l2 = layer_lambda(model, g2, s2)
        cross = g1 * g2
        b_i = (f * l1 * (cross + (al / be) * g1 * g1)).numpy()
        a_i = (f * l2 * (cross + (be / al) * g2 * g2)).numpy()
        if a_i.sum() <= 0 or b_i.sum() <= 0:
            out.append((a_i.copy(), b_i.copy()))
        delta = -f * (al * g1 + be * g2)
        i = 0
        for p in model.parameters():
            n = p.numel(); p.grad = (-delta[i:i + n].view_as(p)).clone(); i += n
        outer.step(); outer.zero_grad(set_to_none=False)
    return out


if __name__ == "__main__":
    raw = [x for s in SEEDS for x in collect(s)]
    total = C.STEPS * len(list(SEEDS))
    inst = [(a, b) for a, b in raw if not degenerate(a, b)]
    print(f"instances where the two-constraint test fails: {len(raw)} / {total} "
          f"({len(raw)/total:.1%})")
    print(f"  of which vacuous (guard-rail: a prior collapsed / total cancellation): "
          f"{len(raw)-len(inst)} ({1-len(inst)/max(len(raw),1):.1%})")
    print(f"  genuine conflict instances:                     {len(inst)}\n")

    gaps, sw, ex, both_fail, sweep_only_fail = [], [], [], 0, 0
    for a, b in inst:
        ns, oks = sweep(a, b)
        ne, oke = exact(a, b)
        if not oke:
            both_fail += 1; continue
        if not oks:
            sweep_only_fail += 1; continue
        sw.append(ns); ex.append(ne); gaps.append(ns - ne)
    gaps = np.array(gaps)
    print(f"ILP infeasible (true FAIL):          {both_fail}")
    print(f"sweep failed where ILP succeeded:    {sweep_only_fail}")
    print(f"solved by both:                      {len(gaps)}\n")
    if len(gaps):
        print(f"  exact optimum removals: median {np.median(ex):.0f}, mean {np.mean(ex):.1f}, max {max(ex)}")
        print(f"  sweep removals:         median {np.median(sw):.0f}, mean {np.mean(sw):.1f}, max {max(sw)}")
        print(f"  gap (sweep - exact):    median {np.median(gaps):.0f}, mean {np.mean(gaps):.2f}, max {gaps.max()}")
        print(f"  sweep exactly optimal:  {(gaps == 0).mean():.1%} of instances")
        print(f"  gap distribution:       " +
              ", ".join(f"{k}:{(gaps==k).sum()}" for k in range(0, min(gaps.max(), 5) + 1)) +
              (f", >4:{(gaps>4).sum()}" if gaps.max() > 4 else ""))
        rel = gaps / np.maximum(np.array(ex), 1)
        print(f"  relative excess:        median {np.median(rel):.1%}, mean {np.mean(rel):.1%}")
    np.savez("bicontrained_results.npz", sweep=np.array(sw), exact=np.array(ex))
