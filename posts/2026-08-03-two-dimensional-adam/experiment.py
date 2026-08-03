"""Two-task adaptation experiment for the Adam2D post.

Setting: fit new classes without regressing on old ones. A small MLP is
pretrained on classes 0-4 of sklearn's `digits`, then adapted to classes 5-9
while a retention loss on the old classes is available. Four update rules are
compared on the same gradients.

Noise scales are estimated by micro-batching (the standard error of the batch
mean), not from an EMA along the trajectory -- an EMA would mix sampling noise
with drift as theta moves.

Run: python experiment.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEEDS = range(8)
STEPS = 300
EVAL_EVERY = 10
BATCH = 128
MICRO = 4          # micro-batches per task -> noise estimate on the batch mean
LR = 3e-3
ALPHA, BETA = 0.5, 0.5
PHI = lambda x: 0.5 * (1 + torch.erf(x / np.sqrt(2)))


def make_data(seed):
    d = load_digits()
    X = (d.data - d.data.mean(0)) / (d.data.std(0) + 1e-6)
    Xtr, Xte, ytr, yte = train_test_split(X, d.target, test_size=0.3,
                                          random_state=seed, stratify=d.target)
    t = lambda a, b: (torch.tensor(a, dtype=torch.float32), torch.tensor(b))
    old_tr, new_tr = ytr < 5, ytr >= 5
    old_te, new_te = yte < 5, yte >= 5
    return (t(Xtr[old_tr], ytr[old_tr]), t(Xtr[new_tr], ytr[new_tr]),
            t(Xte[old_te], yte[old_te]), t(Xte[new_te], yte[new_te]))


def mlp(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64),
                         nn.ReLU(), nn.Linear(64, 10))


def flat_grad(model, loss):
    g = torch.autograd.grad(loss, list(model.parameters()), retain_graph=False)
    return torch.cat([x.reshape(-1) for x in g])


def task_gradient(model, X, y, gen, micro=MICRO):
    """Batch gradient plus the standard error of that mean, via micro-batches."""
    idx = torch.randint(len(X), (BATCH,), generator=gen)
    gs = []
    for chunk in idx.chunk(micro):
        gs.append(flat_grad(model, F.cross_entropy(model(X[chunk]), y[chunk])))
    G = torch.stack(gs)
    ghat = G.mean(0)
    sem = G.std(0) / np.sqrt(micro)          # noise on the mean, not on one datum
    return ghat, sem


def set_flat_(model, delta, lr):
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.add_(lr * delta[i:i + n].view_as(p))
            i += n


def shrinkage(ghat, sem):
    """Empirical-Bayes factor c = tau / sqrt(tau^2 + sigma^2), pooled over
    coordinates: tau^2 = E[ghat^2] - E[sigma^2]."""
    tau2 = (ghat.pow(2).mean() - sem.pow(2).mean()).clamp(min=0)
    return torch.sqrt(tau2 / (tau2 + sem.pow(2).mean() + 1e-12))


def direction(rule, g1, s1, g2, s2):
    if rule == "new only":
        return -g1
    if rule == "sum":
        return -(ALPHA * g1 + BETA * g2)
    if rule == "PCGrad":
        a, b = g1.clone(), g2.clone()
        if torch.dot(g1, g2) < 0:
            a = g1 - torch.dot(g1, g2) / g2.pow(2).sum() * g2
            b = g2 - torch.dot(g2, g1) / g1.pow(2).sum() * g1
        return -(ALPHA * a + BETA * b)
    if rule in ("Adam2D", "Adam2D (flat prior)"):
        c1 = c2 = 1.0
        if rule == "Adam2D":
            c1, c2 = shrinkage(g1, s1), shrinkage(g2, s2)
        p1 = 2 * PHI(c1 * g1 / (s1 + 1e-12)) - 1
        p2 = 2 * PHI(c2 * g2 / (s2 + 1e-12)) - 1
        f = 0.5 * (1 + p1 * p2)
        return -f * (ALPHA * g1 + BETA * g2)
    raise ValueError(rule)


RULES = ["new only", "sum", "PCGrad", "Adam2D (flat prior)", "Adam2D"]


def run(seed):
    (Xo, yo), (Xn, yn), (Xote, yote), (Xnte, ynte) = make_data(seed)
    gen = torch.Generator().manual_seed(seed)

    base = mlp(seed)                                   # pretrain on old classes
    opt = torch.optim.Adam(base.parameters(), lr=3e-3)
    for _ in range(400):
        i = torch.randint(len(Xo), (BATCH,), generator=gen)
        opt.zero_grad(); F.cross_entropy(base(Xo[i]), yo[i]).backward(); opt.step()

    acc = lambda m, X, y: (m(X).argmax(1) == y).float().mean().item()
    out = {}
    for rule in RULES:
        model = mlp(seed)
        model.load_state_dict(base.state_dict())
        g = torch.Generator().manual_seed(seed + 1000)
        hist = []
        for step in range(STEPS + 1):
            if step % EVAL_EVERY == 0:
                with torch.no_grad():
                    hist.append((step, acc(model, Xnte, ynte), acc(model, Xote, yote)))
            g1, s1 = task_gradient(model, Xn, yn, g)     # task 1: new classes
            g2, s2 = task_gradient(model, Xo, yo, g)     # task 2: retain old
            set_flat_(model, direction(rule, g1, s1, g2, s2), LR)
        out[rule] = np.array(hist)
    return out


if __name__ == "__main__":
    runs = [run(s) for s in SEEDS]
    res = {r: np.stack([x[r] for x in runs]) for r in RULES}
    np.savez("experiment_results.npz", **res)

    print(f"digits, {len(SEEDS)} seeds, {STEPS} steps, mean +/- std at the final step")
    print(f"{'rule':22s} {'new acc':>16s} {'old acc':>16s} {'mean':>8s}")
    for r in RULES:
        a = res[r][:, -1, :]
        print(f"{r:22s} {a[:,1].mean():.3f} +/- {a[:,1].std():.3f}   "
              f"{a[:,2].mean():.3f} +/- {a[:,2].std():.3f}   "
              f"{(a[:,1]+a[:,2]).mean()/2:.3f}")
