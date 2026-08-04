"""Feasibility certificate V(f) <= -tau for gated two-task steps, and the greedy
l1-minimal repair when it fails.

V(f) <= 0 is exactly "Delta is a descent direction for both *measured* gradients"
-- pure algebra, no prior. This script measures how often it actually fails on a
real two-task problem, how many coordinates the greedy repair removes when it
does, and what the repair costs in accuracy.

Run: python certificate.py    (~4 min on CPU)
"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "2026-08-03-two-dimensional-adam"))
import experiment as E                                   # model, data, gradients

ALPHA, BETA = 0.7, 0.3
STEPS = 600
SEEDS = range(6)
LR = 1e-2


def violation(m, f, g1, g2, al=ALPHA, be=BETA):
    w = m * f
    S11 = torch.dot(w * g1, g1); S22 = torch.dot(w * g2, g2); S12 = torch.dot(w * g1, g2)
    return -S12 - torch.min(al / be * S11, be / al * S22), S11, S22, S12


def greedy_mask(f, g1, g2, tau=0.0, al=ALPHA, be=BETA):
    """l1-minimal binary mask driving V(f*m) <= -tau. Returns (mask, n_removed, ok)."""
    s = f * g1 * g2
    scoreA = s + (al / be) * f * g1 * g1          # branch where (al/be)*S11 is active
    scoreB = s + (be / al) * f * g2 * g2
    piA = torch.argsort(scoreA); piB = torch.argsort(scoreB)
    m = torch.ones_like(f)
    S11 = torch.dot(f * g1, g1); S22 = torch.dot(f * g2, g2); S12 = torch.dot(f * g1, g2)
    pa = pb = 0
    removed = 0
    while -S12 - min(al / be * S11, be / al * S22) > -tau:
        useA = (al / be * S11) <= (be / al * S22)
        pi, p = (piA, pa) if useA else (piB, pb)
        while p < len(pi) and m[pi[p]] == 0:
            p += 1
        if p >= len(pi):
            return m, removed, False                      # no feasible mask
        i = pi[p]
        if useA: pa = p + 1
        else:    pb = p + 1
        m[i] = 0; removed += 1
        S11 -= f[i] * g1[i] * g1[i]; S22 -= f[i] * g2[i] * g2[i]; S12 -= f[i] * g1[i] * g2[i]
    return m, removed, True


def focus(g1, s1, g2, s2):
    c1, c2 = E.shrinkage(g1, s1), E.shrinkage(g2, s2)
    p1 = 2 * E.PHI(c1 * g1 / (s1 + 1e-12)) - 1
    p2 = 2 * E.PHI(c2 * g2 / (s2 + 1e-12)) - 1
    return 0.5 * (1 + p1 * p2)


def run(seed, repair, tau_scale=0.0):
    (Xo, yo), (Xn, yn), (Xote, yote), (Xnte, ynte) = E.make_data(seed)
    gen = torch.Generator().manual_seed(seed)
    base = E.mlp(seed); opt = torch.optim.Adam(base.parameters(), lr=3e-3)
    for _ in range(400):
        i = torch.randint(len(Xo), (E.BATCH,), generator=gen)
        opt.zero_grad(); F.cross_entropy(base(Xo[i]), yo[i]).backward(); opt.step()
    model = E.mlp(seed); model.load_state_dict(base.state_dict())
    outer = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed + 1000)

    n_trigger = n_fail = 0; removed_when_triggered = []; d = None
    for _ in range(STEPS):
        g1, s1 = E.task_gradient(model, Xn, yn, g)
        g2, s2 = E.task_gradient(model, Xo, yo, g)
        f = focus(g1, s1, g2, s2); d = len(f)
        V, S11, S22, _ = violation(torch.ones_like(f), f, g1, g2)
        tau = tau_scale * torch.min(ALPHA / BETA * S11, BETA / ALPHA * S22).item()
        m = torch.ones_like(f)
        if V.item() > -tau:
            n_trigger += 1
            if repair:
                m, nrem, ok = greedy_mask(f, g1, g2, tau)
                removed_when_triggered.append(nrem)
                if not ok: n_fail += 1
            else:
                removed_when_triggered.append(0)
        delta = -(m * f) * (ALPHA * g1 + BETA * g2)
        i = 0
        for p in model.parameters():
            n = p.numel(); p.grad = (-delta[i:i + n].view_as(p)).clone(); i += n
        outer.step(); outer.zero_grad(set_to_none=False)

    acc = lambda X, y: (model(X).argmax(1) == y).float().mean().item()
    with torch.no_grad():
        return dict(new=acc(Xnte, ynte), old=acc(Xote, yote), d=d,
                    trigger=n_trigger / STEPS, fail=n_fail,
                    removed=float(np.mean(removed_when_triggered)) if removed_when_triggered else 0.0)


if __name__ == "__main__":
    print(f"digits two-task adaptation, alpha={ALPHA}, {STEPS} steps, {len(list(SEEDS))} seeds\n")
    print(f"{'configuration':28s} {'trigger':>9s} {'removed/d':>11s} {'FAIL':>6s} "
          f"{'new acc':>16s} {'old acc':>16s}")
    for label, repair, ts in [("focus only (no repair)", False, 0.0),
                              ("hybrid, tau = 0", True, 0.0),
                              ("hybrid, tau = 0.05 budget", True, 0.05),
                              ("hybrid, tau = 0.20 budget", True, 0.20)]:
        r = [run(s, repair, ts) for s in SEEDS]
        agg = lambda k: (np.mean([x[k] for x in r]), np.std([x[k] for x in r]) / np.sqrt(len(r)))
        tr, _ = agg("trigger"); rem, _ = agg("removed"); fl, _ = agg("fail")
        nm, ns = agg("new"); om, os_ = agg("old")
        print(f"{label:28s} {tr:8.1%} {rem/r[0]['d']:11.2%} {fl:6.1f} "
              f"  {nm:.4f} +/- {ns:.4f}  {om:.4f} +/- {os_:.4f}")
