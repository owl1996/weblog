"""2x2 ablation for the Adam2D post: aggregation x gate.

Aggregation (how to combine the two gradients) and gating (which coordinates to
trust) are separate questions, so they get separate factors. Each cell is tuned
over its own (learning rate, alpha) grid; the gated direction is handed to an
outer Adam, as in Algorithm 1 of the paper -- applying it as a raw SGD step
turns the gate's attenuation (mean f ~ 0.4) into an uncontrolled learning-rate
cut and dominates the comparison.

Produces the table in the "Does it help?" section.  Run: python experiment_ablation.py
"""
import numpy as np
import torch
import torch.nn.functional as F

import experiment as E

SEEDS = 12
TUNE_SEEDS = 3
STEPS = 600
LRS = [3e-3, 1e-2]
ALPHAS = [0.5, 0.7]
CELLS = [("sum", False), ("sum", True), ("PCGrad", False), ("PCGrad", True)]


def aggregate(kind, g1, g2, alpha):
    beta = 1 - alpha
    if kind == "sum":
        return alpha * g1 + beta * g2
    a, b = g1, g2
    if torch.dot(g1, g2) < 0:                      # PCGrad: drop conflicting parts
        a = g1 - torch.dot(g1, g2) / g2.pow(2).sum() * g2
        b = g2 - torch.dot(g2, g1) / g1.pow(2).sum() * g1
    return alpha * a + beta * b


def focus(g1, s1, g2, s2):
    c1, c2 = E.shrinkage(g1, s1), E.shrinkage(g2, s2)
    p1 = 2 * E.PHI(c1 * g1 / (s1 + 1e-12)) - 1
    p2 = 2 * E.PHI(c2 * g2 / (s2 + 1e-12)) - 1
    return 0.5 * (1 + p1 * p2)


def run(seed, kind, use_gate, alpha, lr, steps=STEPS):
    (Xo, yo), (Xn, yn), (Xote, yote), (Xnte, ynte) = E.make_data(seed)
    gen = torch.Generator().manual_seed(seed)

    base = E.mlp(seed)                                   # pretrain on old classes
    opt = torch.optim.Adam(base.parameters(), lr=3e-3)
    for _ in range(400):
        i = torch.randint(len(Xo), (E.BATCH,), generator=gen)
        opt.zero_grad(); F.cross_entropy(base(Xo[i]), yo[i]).backward(); opt.step()

    model = E.mlp(seed); model.load_state_dict(base.state_dict())
    outer = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 1000)
    for _ in range(steps):
        g1, s1 = E.task_gradient(model, Xn, yn, g)       # task 1: new classes
        g2, s2 = E.task_gradient(model, Xo, yo, g)       # task 2: retain old
        delta = -aggregate(kind, g1, g2, alpha)
        if use_gate:
            delta = focus(g1, s1, g2, s2) * delta
        i = 0
        for p in model.parameters():
            n = p.numel(); p.grad = (-delta[i:i + n].view_as(p)).clone(); i += n
        outer.step(); outer.zero_grad(set_to_none=False)

    acc = lambda X, y: (model(X).argmax(1) == y).float().mean().item()
    with torch.no_grad():
        return acc(Xnte, ynte), acc(Xote, yote)


if __name__ == "__main__":
    res = {}
    print(f"{'aggregation':12s} {'gate':>5s} {'new acc':>18s} {'old acc':>18s} {'mean':>18s}")
    for kind, ug in CELLS:
        best = max(((np.array([run(s, kind, ug, a, lr) for s in range(TUNE_SEEDS)]).mean(0).sum(), lr, a)
                    for lr in LRS for a in ALPHAS), key=lambda t: t[0])
        _, lr, alpha = best
        v = np.array([run(s, kind, ug, alpha, lr) for s in range(SEEDS)])
        res[(kind, ug)] = v
        se = v.std(0) / np.sqrt(SEEDS); mm = v.mean(1)
        print(f"{kind:12s} {str(ug):>5s} {v[:,0].mean():.4f} +/- {se[0]:.4f}   "
              f"{v[:,1].mean():.4f} +/- {se[1]:.4f}   "
              f"{mm.mean():.4f} +/- {mm.std()/np.sqrt(SEEDS):.4f}   (lr={lr:g}, alpha={alpha})")

    print("\nPaired effect of switching the gate on, same aggregation:")
    for kind in ("sum", "PCGrad"):
        d = res[(kind, True)].mean(1) - res[(kind, False)].mean(1)
        print(f"  {kind:8s} delta = {d.mean():+.4f} +/- {d.std()/np.sqrt(SEEDS):.4f}"
              f"   wins {int((d > 0).sum())}/{SEEDS}")
    np.savez("ablation_results.npz", **{f"{k}_{g}": v for (k, g), v in res.items()})
