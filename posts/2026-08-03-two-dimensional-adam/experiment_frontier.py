"""Fair two-task comparison: every rule is tuned over its own (learning rate,
alpha) grid, and reported as a Pareto frontier over the two test accuracies.
Tuning per method matters here because the focus gate attenuates the step by
mean(f) ~ 0.4, so a shared learning rate would penalise it for free.

Run: python experiment_frontier.py   (~6 min on CPU)
"""
import numpy as np, torch, torch.nn.functional as F
import experiment as E

LRS = [3e-3, 1e-2, 3e-2]
ALPHAS = [0.35, 0.5, 0.7, 0.9]
RULES = ["sum", "PCGrad", "Adam2D (flat prior)", "Adam2D"]
SEEDS = range(4)
STEPS = 600
CURVE_EVERY = 20


def one(seed, rule, alpha, lr, steps=STEPS, curve=False):
    E.ALPHA, E.BETA, E.LR = alpha, 1 - alpha, lr
    (Xo, yo), (Xn, yn), (Xote, yote), (Xnte, ynte) = E.make_data(seed)
    gen = torch.Generator().manual_seed(seed)
    base = E.mlp(seed)
    opt = torch.optim.Adam(base.parameters(), lr=3e-3)
    for _ in range(400):
        i = torch.randint(len(Xo), (E.BATCH,), generator=gen)
        opt.zero_grad(); F.cross_entropy(base(Xo[i]), yo[i]).backward(); opt.step()
    model = E.mlp(seed); model.load_state_dict(base.state_dict())
    g = torch.Generator().manual_seed(seed + 1000)
    acc = lambda X, y: (model(X).argmax(1) == y).float().mean().item()
    hist = []
    for step in range(steps + 1):
        if curve and step % CURVE_EVERY == 0:
            with torch.no_grad():
                hist.append((step, acc(Xnte, ynte), acc(Xote, yote)))
        if step == steps:
            break
        g1, s1 = E.task_gradient(model, Xn, yn, g)
        g2, s2 = E.task_gradient(model, Xo, yo, g)
        E.set_flat_(model, E.direction(rule, g1, s1, g2, s2), E.LR)
    with torch.no_grad():
        return (acc(Xnte, ynte), acc(Xote, yote), np.array(hist))


def pareto(pts):
    """pts: list of (new, old, meta). Keep non-dominated."""
    out = []
    for p in pts:
        if not any((q[0] >= p[0] and q[1] >= p[1] and q[:2] != p[:2]) for q in pts):
            out.append(p)
    return sorted(out)


if __name__ == "__main__":
    grid, curves = {}, {}
    for rule in RULES:
        pts = []
        for lr in LRS:
            for a in ALPHAS:
                r = np.array([one(s, rule, a, lr)[:2] for s in SEEDS])
                pts.append((round(r[:, 0].mean(), 4), round(r[:, 1].mean(), 4), (lr, a)))
        grid[rule] = pts
        best = max(pts, key=lambda p: p[0] + p[1])
        lr, a = best[2]
        curves[rule] = (np.stack([one(s, rule, a, lr, curve=True)[2] for s in SEEDS]), lr, a)
        print(f"{rule:22s} best lr={lr:<6g} alpha={a:<4} -> new {best[0]:.3f}  old {best[1]:.3f}")

    np.savez("frontier_results.npz",
             **{f"grid::{k}": np.array([(p[0], p[1], p[2][0], p[2][1]) for p in v])
                for k, v in grid.items()},
             **{f"curve::{k}": v[0] for k, v in curves.items()},
             **{f"cfg::{k}": np.array(v[1:]) for k, v in curves.items()})

    print("\nPareto frontier (new acc, old acc) per rule")
    for rule in RULES:
        print(f"  {rule:22s} " + "  ".join(f"({p[0]:.2f},{p[1]:.2f})" for p in pareto(grid[rule])))
