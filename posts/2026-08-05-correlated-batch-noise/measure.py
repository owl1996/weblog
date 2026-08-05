"""Does correlated batch noise actually matter? A measurement on `digits`.

Two shared-batch two-task problems -- the case the Adam2D post's independence
assumption excludes -- on the same architecture and data:

  "retain"     L1 = cross-entropy on the batch
               L2 = KL(base || model) on the SAME batch      (antagonistic)

  "multitask"  L1 = cross-entropy, digit class, head 1
               L2 = cross-entropy, digit parity, head 2, same batch (cooperative)

A shared fraction kappa of the two batches is the knob: kappa = 0 reproduces the
independent case Adam2D assumes, kappa = 1 the fully shared case.

Two things make this decisive rather than assumption-bound:

  * the noise correlation has a REFERENCE value -- the empirical correlation of
    the batch-mean gradients over R independent batch draws at frozen theta --
    so the cheap aligned-micro-batch estimator can be scored against it;
  * the TRUE gradients are computable exactly, the population loss here being
    the full-dataset loss. So "the two true gradients agree in sign" is an
    observable label, and f -- which claims to be the probability of exactly
    that event -- can be scored as a probabilistic forecast.

Run: python measure.py    (~4 min on CPU)
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "2026-08-03-two-dimensional-adam"))
import experiment as E                                # data, flat_grad, shrinkage

from orthant import focus, focus_indep, phi

SEEDS = range(4)
STEPS = 300
BATCH = 128
MICRO = 8                 # aligned micro-batches -> sigma1, sigma2 and c12
CHECKPOINTS = tuple(range(20, 300, 10))     # 28 per run: the tables
SAMPLE_AT = (20, 60, 100, 150, 200, 250, 290)   # subset kept for the figures
R_REF = 200               # batch draws for the reference correlation
LR = 3e-3
ALPHA, BETA = 0.7, 0.3    # same weighting as the certification post
KAPPAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TEMP = 2.0
PROBLEMS = ("retain", "multitask")


# ------------------------------------------------------------------ model ---

class TwoHead(nn.Module):
    """Shared trunk, two heads -- the multitask problem."""

    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.trunk = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                   nn.Linear(64, 64), nn.ReLU())
        self.h1 = nn.Linear(64, 10)
        self.h2 = nn.Linear(64, 2)

    def forward(self, x):
        z = self.trunk(x)
        return self.h1(z), self.h2(z)


# ------------------------------------------------------------------ losses ---

def losses(problem, model, ctx, X, y, idx1, idx2):
    """(L1 on idx1, L2 on idx2) for the chosen problem."""
    if problem == "retain":
        l1 = F.cross_entropy(model(X[idx1]), y[idx1])
        p = F.log_softmax(model(X[idx2]) / TEMP, dim=1)
        q = F.softmax(ctx[idx2] / TEMP, dim=1)
        l2 = F.kl_div(p, q, reduction="batchmean") * TEMP ** 2
    else:
        l1 = F.cross_entropy(model(X[idx1])[0], y[idx1])
        l2 = F.cross_entropy(model(X[idx2])[1], ctx[idx2])
    return l1, l2


def flat_grad(model, loss):
    """Flat gradient, zero-filled for parameters outside this loss's graph.

    In the multitask problem each head is private to one task, so half the
    parameters are genuinely unused by each loss."""
    ps = list(model.parameters())
    g = torch.autograd.grad(loss, ps, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if x is None else x).reshape(-1)
                      for p, x in zip(ps, g)])


def analysed(problem, model):
    """Boolean mask over the flat parameter vector: the coordinates on which
    the gate is defined. Task-private heads carry no conflict by construction,
    so the multitask problem is analysed on its shared trunk."""
    ps = list(model.parameters())
    sizes = [p.numel() for p in ps]
    if problem == "retain":
        return torch.ones(sum(sizes), dtype=torch.bool)
    shared = {id(p) for p in model.trunk.parameters()}
    return torch.cat([torch.full((n,), id(p) in shared, dtype=torch.bool)
                      for p, n in zip(ps, sizes)])


def grads_on(problem, model, ctx, X, y, idx1, idx2):
    l1, l2 = losses(problem, model, ctx, X, y, idx1, idx2)
    return flat_grad(model, l1), flat_grad(model, l2)


def paired_index(gen, n, kappa, micro=MICRO, batch=BATCH):
    """Index sets for the two tasks sharing a fraction kappa *within* every
    micro-batch, so the aligned splits see the same kappa throughout."""
    idx1 = torch.randint(n, (batch,), generator=gen)
    m = batch // micro
    nshare = int(round(kappa * m))
    idx2 = idx1.clone()
    for c in range(micro):
        lo = c * m
        idx2[lo + nshare:lo + m] = torch.randint(n, (m - nshare,),
                                                 generator=gen)
    return idx1, idx2


def aligned_moments(problem, model, ctx, X, y, gen, kappa, micro=MICRO):
    """One batch -> ghat_k, sigma_k and c12, from MICRO aligned splits.

    The micro-batches are the ones already needed to estimate sigma; evaluating
    both losses on each of them makes the cross-covariance free.
    """
    idx1, idx2 = paired_index(gen, len(X), kappa, micro)
    G1, G2 = [], []
    for c1, c2 in zip(idx1.chunk(micro), idx2.chunk(micro)):
        a, b = grads_on(problem, model, ctx, X, y, c1, c2)
        G1.append(a)
        G2.append(b)
    G1, G2 = torch.stack(G1), torch.stack(G2)
    g1, g2 = G1.mean(0), G2.mean(0)
    s1 = G1.std(0) / np.sqrt(micro)
    s2 = G2.std(0) / np.sqrt(micro)
    c12 = ((G1 - g1) * (G2 - g2)).sum(0) / (micro - 1) / micro
    return g1, s1, g2, s2, c12, G1, G2


def reference_rho(problem, model, ctx, X, y, seed, kappa, R=R_REF):
    """Correlation of the batch-mean gradients over R independent batch draws:
    the estimand itself. 2R backward passes -- run at checkpoints only."""
    gen = torch.Generator().manual_seed(seed * 7919 + 13)
    A, B_ = [], []
    for _ in range(R):
        i1, i2 = paired_index(gen, len(X), kappa)
        a, b = grads_on(problem, model, ctx, X, y, i1, i2)
        A.append(a)
        B_.append(b)
    A, B_ = torch.stack(A), torch.stack(B_)
    ca, cb = A - A.mean(0), B_ - B_.mean(0)
    rho = (ca * cb).mean(0) / (ca.std(0) * cb.std(0) + 1e-30)
    return rho.clamp(-1, 1).numpy(), A.std(0).numpy(), B_.std(0).numpy()


def true_gradients(problem, model, ctx, X, y):
    """Full-dataset gradients: the population quantity a batch estimates."""
    n = torch.arange(len(X))
    g1, g2 = grads_on(problem, model, ctx, X, y, n, n)
    return g1.detach().numpy(), g2.detach().numpy()


# ----------------------------------------------------------------- scoring ---

def brier(f, z):
    return float(np.mean((f - z) ** 2))


def ece(f, z, nbin=10):
    """Expected calibration error over equal-mass bins."""
    q = np.quantile(f, np.linspace(0, 1, nbin + 1))
    q[0], q[-1] = -np.inf, np.inf
    tot = 0.0
    for lo, hi in zip(q[:-1], q[1:]):
        m = (f > lo) & (f <= hi)
        if m.sum():
            tot += m.mean() * abs(f[m].mean() - z[m].mean())
    return float(tot)


def violation(f, g1, g2, al=ALPHA, be=BETA):
    S11 = float(np.sum(f * g1 * g1))
    S22 = float(np.sum(f * g2 * g2))
    S12 = float(np.sum(f * g1 * g2))
    return -S12 - min(al / be * S11, be / al * S22), S11, S22, S12


# -------------------------------------------------------------------- run ----

def setup(problem, seed):
    """Model, frozen context (base logits or parity labels), data."""
    (Xo, yo), (Xn, yn), (Xote, yote), (Xnte, ynte) = E.make_data(seed)
    X, y = torch.cat([Xo, Xn]), torch.cat([yo, yn])
    if problem == "retain":
        base = E.mlp(seed)
        opt = torch.optim.Adam(base.parameters(), lr=3e-3)
        gen = torch.Generator().manual_seed(seed)
        for _ in range(400):                       # pretrain on the old classes
            i = torch.randint(len(Xo), (BATCH,), generator=gen)
            opt.zero_grad()
            F.cross_entropy(base(Xo[i]), yo[i]).backward()
            opt.step()
        with torch.no_grad():
            ctx = base(X).detach()
        model = E.mlp(seed)
        model.load_state_dict(base.state_dict())
    else:
        ctx = (y % 2).long()                       # parity: the second task
        model = TwoHead(seed)
    return model, ctx, X, y, (Xote, yote, Xnte, ynte)


def run(problem, seed, kappa):
    model, ctx, X, y, _ = setup(problem, seed)
    outer = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed + 1000)
    ana = analysed(problem, model)

    rows, samples = [], []
    for step in range(STEPS):
        G1f, G2f = None, None
        g1, s1, g2, s2, c12, G1, G2 = aligned_moments(
            problem, model, ctx, X, y, g, kappa)
        g1_full, g2_full = g1, g2
        g1, s1, g2, s2, c12 = (g1[ana], s1[ana], g2[ana], s2[ana], c12[ana])
        G1, G2 = G1[:, ana], G2[:, ana]
        rho_hat = (c12 / (s1 * s2 + 1e-30)).clamp(-1, 1).numpy()
        a0 = (g1 / (s1 + 1e-12)).numpy()
        b0 = (g2 / (s2 + 1e-12)).numpy()
        f0 = focus_indep(a0, b0)

        if step in CHECKPOINTS:
            rho, sd1, sd2 = reference_rho(problem, model, ctx, X, y,
                                          seed, kappa)
            k = ana.numpy()
            rho, sd1, sd2 = rho[k], sd1[k], sd2[k]
            tg1, tg2 = true_gradients(problem, model, ctx, X, y)
            tg1, tg2 = tg1[k], tg2[k]
            z = (tg1 * tg2 > 0).astype(float)
            gg1, gg2 = g1.numpy(), g2.numpy()

            frho = focus(a0, b0, rho)
            fhat = focus(a0, b0, rho_hat)
            c1, c2 = float(E.shrinkage(g1, s1)), float(E.shrinkage(g2, s2))
            f0_sh = focus_indep(c1 * a0, c2 * b0)
            frho_sh = focus(c1 * a0, c2 * b0, rho)

            sub = np.argsort(-np.abs(a0 * b0))[:400]     # 400 loudest coords
            tau = np.array([kendalltau(G1[:, i].numpy(),
                                       G2[:, i].numpy()).statistic
                            for i in sub])
            rho_kendall = np.sin(np.pi * tau / 2)

            V0 = violation(f0, gg1, gg2)
            Vr = violation(frho, gg1, gg2)
            bud = min(ALPHA / BETA * V0[1], BETA / ALPHA * V0[2])
            agg_dir = ALPHA * gg1 + BETA * gg2
            au = (lambda f: float(roc_auc_score(z, f)) if 0 < z.mean() < 1
                  else np.nan)

            rows.append(dict(
                seed=seed, kappa=kappa, step=step,
                mean_absrho=float(np.mean(np.abs(rho))),
                med_absrho=float(np.median(np.abs(rho))),
                q90_absrho=float(np.quantile(np.abs(rho), 0.9)),
                frac_rho_neg=float(np.mean(rho < 0)),
                mean_rho=float(np.mean(rho)),
                rho_hat_bias=float(np.mean(rho_hat - rho)),
                rho_hat_corr=float(np.corrcoef(rho_hat, rho)[0, 1]),
                rho_hat_rmse=float(np.sqrt(np.mean((rho_hat - rho) ** 2))),
                rho_kendall_corr=float(np.corrcoef(rho_kendall,
                                                   rho[sub])[0, 1]),
                sigma_ratio1=float(np.mean(s1.numpy() / (sd1 + 1e-30))),
                sigma_ratio2=float(np.mean(s2.numpy() / (sd2 + 1e-30))),
                mean_absdf=float(np.mean(np.abs(frho - f0))),
                q99_absdf=float(np.quantile(np.abs(frho - f0), 0.99)),
                lemma1_bound=float(np.arcsin(np.mean(np.abs(rho))) / np.pi),
                lemma2_pred=float(np.mean(np.abs(rho) * 2 * phi(a0) * phi(b0))),
                lemma2_ratio=float(np.mean(np.abs(frho - f0))
                                   / (np.mean(np.abs(rho) * 2 * phi(a0)
                                              * phi(b0)) + 1e-30)),
                df_pos_rho=float(np.mean((frho - f0)[rho > 0]))
                if (rho > 0).any() else np.nan,
                df_neg_rho=float(np.mean((frho - f0)[rho < 0]))
                if (rho < 0).any() else np.nan,
                brier_f0=brier(f0, z), brier_frho=brier(frho, z),
                brier_fhat=brier(fhat, z),
                brier_f0_sh=brier(f0_sh, z), brier_frho_sh=brier(frho_sh, z),
                brier_clim=brier(np.full_like(f0, z.mean()), z),
                ece_f0=ece(f0, z), ece_frho=ece(frho, z),
                ece_f0_sh=ece(f0_sh, z), ece_frho_sh=ece(frho_sh, z),
                auc_f0=au(f0), auc_frho=au(frho), auc_fhat=au(fhat),
                err_f0=float(np.mean((f0 > 0.5) != (z > 0.5))),
                err_frho=float(np.mean((frho > 0.5) != (z > 0.5))),
                agree_rate=float(z.mean()),
                obs_agree_pos=float(np.mean((gg1 * gg2 > 0)[(z > 0)
                                                            & (rho > 0)]))
                if ((z > 0) & (rho > 0)).any() else np.nan,
                obs_agree_neg=float(np.mean((gg1 * gg2 > 0)[(z > 0)
                                                            & (rho < 0)]))
                if ((z > 0) & (rho < 0)).any() else np.nan,
                V0=V0[0], Vrho=Vr[0], budget=bud,
                ddir=float(np.linalg.norm((frho - f0) * agg_dir)
                           / (np.linalg.norm(f0 * agg_dir) + 1e-30)),
                mean_f0=float(f0.mean()), mean_frho=float(frho.mean()),
            ))
            if step in SAMPLE_AT:                # kept for the figures only
                samples.append(np.stack(
                    [rho, phi(a0) * phi(b0), f0, frho, z]
                ).astype(np.float32)[:, ::3])

        f_full = torch.ones(len(ana))                 # heads are ungated
        f_full[ana] = torch.from_numpy(f0).float()
        delta = -f_full * (ALPHA * g1_full + BETA * g2_full)   # deployed: rho=0
        i = 0
        for p in model.parameters():
            n = p.numel()
            p.grad = (-delta[i:i + n].view_as(p)).clone()
            i += n
        outer.step()
        outer.zero_grad(set_to_none=False)
    return rows, samples


def agg(rows, key):
    v = np.array([r[key] for r in rows], float)
    v = v[~np.isnan(v)]
    if not len(v):
        return np.nan, np.nan
    return v.mean(), v.std() / np.sqrt(len(v))


def sel(rows, problem, kappa=None):
    return [r for r in rows if r["problem"] == problem
            and (kappa is None or r["kappa"] == kappa)]


if __name__ == "__main__":
    torch.set_num_threads(4)
    all_rows, samples = [], {}

    for prob in PROBLEMS:
        for kap in KAPPAS:
            for s in SEEDS:
                r, smp = run(prob, s, kap)
                for x in r:
                    x["problem"] = prob
                all_rows += r
                if kap == 1.0:
                    samples.setdefault(prob, []).extend(smp)

    print("=" * 79)
    print("A. how much correlation is there? (reference rho, R = 200 draws)")
    print("=" * 79)
    print(f"{'problem':10s} {'kappa':>6s} {'mean|rho|':>10s} {'med|rho|':>9s} "
          f"{'q90|rho|':>9s} {'mean rho':>9s} {'P[rho<0]':>9s} "
          f"{'mean|df|':>9s} {'q99|df|':>8s} {'|dDelta|':>9s}")
    for prob in PROBLEMS:
        for kap in KAPPAS:
            r = sel(all_rows, prob, kap)
            print(f"{prob:10s} {kap:6.2f} {agg(r,'mean_absrho')[0]:10.3f} "
                  f"{agg(r,'med_absrho')[0]:9.3f} "
                  f"{agg(r,'q90_absrho')[0]:9.3f} "
                  f"{agg(r,'mean_rho')[0]:+9.3f} "
                  f"{agg(r,'frac_rho_neg')[0]:9.3f} "
                  f"{agg(r,'mean_absdf')[0]:9.4f} "
                  f"{agg(r,'q99_absdf')[0]:8.4f} "
                  f"{agg(r,'ddir')[0]:9.4f}")

    print("\n" + "=" * 79)
    print("B. the two bounds against the measured deviation (kappa = 1)")
    print("=" * 79)
    print(f"{'problem':10s} {'mean|df|':>9s} {'Lemma 1 (uniform)':>19s} "
          f"{'Lemma 2 (local)':>17s} {'|df| / Lemma 2':>15s}")
    for prob in PROBLEMS:
        r = sel(all_rows, prob, 1.0)
        print(f"{prob:10s} {agg(r,'mean_absdf')[0]:9.4f} "
              f"{agg(r,'lemma1_bound')[0]:19.4f} "
              f"{agg(r,'lemma2_pred')[0]:17.4f} "
              f"{agg(r,'lemma2_ratio')[0]:15.2f}")

    print("\n" + "=" * 79)
    print("C. the free estimator (aligned micro-batches, k = 8) vs reference")
    print("=" * 79)
    for prob in PROBLEMS:
        r = sel(all_rows, prob, 1.0)
        print(f"  {prob}")
        for k, lab in [("rho_hat_corr", "corr(rho_hat, rho_ref)"),
                       ("rho_hat_bias", "bias  E[rho_hat - rho_ref]"),
                       ("rho_hat_rmse", "RMSE"),
                       ("rho_kendall_corr",
                        "corr(sin(pi tau/2), rho_ref), 400 loudest"),
                       ("sigma_ratio1", "sigma1_hat / sigma1_ref"),
                       ("sigma_ratio2", "sigma2_hat / sigma2_ref")]:
            m, s = agg(r, k)
            print(f"     {lab:44s} {m:+.4f} +/- {s:.4f}")

    print("\n" + "=" * 79)
    print("D. f as a forecast of true-gradient sign agreement (kappa = 1)")
    print("=" * 79)
    for prob in PROBLEMS:
        r = sel(all_rows, prob, 1.0)
        print(f"  {prob}:  base rate P[g1 g2 > 0] = "
              f"{agg(r,'agree_rate')[0]:.3f}")
        print(f"     {'gate':32s} {'Brier':>8s} {'ECE':>8s} {'AUC':>8s} "
              f"{'err(f>1/2)':>11s}")
        for bk, ek, ak, xk, lab in [
                ("brier_clim", None, None, None, "constant base rate"),
                ("brier_f0", "ece_f0", "auc_f0", "err_f0",
                 "f^0   flat prior (Adam2D)"),
                ("brier_frho", "ece_frho", "auc_frho", "err_frho",
                 "f^rho flat prior"),
                ("brier_f0_sh", "ece_f0_sh", None, None,
                 "f^0   shrunk prior"),
                ("brier_frho_sh", "ece_frho_sh", None, None,
                 "f^rho shrunk prior"),
                ("brier_fhat", None, "auc_fhat", None,
                 "f^rho, free estimator rho_hat")]:
            g = lambda k, w=8, p=4: (f"{agg(r,k)[0]:{w}.{p}f}" if k
                                     else " " * w)
            print(f"     {lab:32s} {g(bk)} {g(ek)} {g(ak)} {g(xk,11)}")

    print("\n" + "=" * 79)
    print("E. the two directed predictions (kappa = 1)")
    print("=" * 79)
    for prob in PROBLEMS:
        r = sel(all_rows, prob, 1.0)
        p, ps = agg(r, "df_pos_rho")
        n, ns = agg(r, "df_neg_rho")
        oa, _ = agg(r, "obs_agree_pos")
        ob, _ = agg(r, "obs_agree_neg")
        print(f"  {prob}")
        print(f"     mean(f^rho - f^0) where rho > 0: {p:+.4f} +/- {ps:.4f}"
              f"   (rho > 0 ignored  ->  over-masking)")
        print(f"     mean(f^rho - f^0) where rho < 0: {n:+.4f} +/- {ns:.4f}"
              f"   (rho < 0 ignored  ->  under-masking)")
        print(f"     among truly-agreeing coordinates, observed agreement:"
              f"  rho>0 {oa:.3f}   rho<0 {ob:.3f}")

    print("\n" + "=" * 79)
    print("F. effect on the level-(i) certificate (kappa = 1)")
    print("=" * 79)
    for prob in PROBLEMS:
        r = sel(all_rows, prob, 1.0)
        V0 = np.array([x["V0"] for x in r])
        Vr = np.array([x["Vrho"] for x in r])
        print(f"  {prob:10s}  V(f^0) > 0: {np.mean(V0 > 0):6.1%}    "
              f"V(f^rho) > 0: {np.mean(Vr > 0):6.1%}    "
              f"sign disagreement: {np.mean((V0 > 0) != (Vr > 0)):6.1%}")

    np.savez("measure_results.npz",
             **{f"samples_{p}": np.concatenate(v, axis=1)
                for p, v in samples.items()},
             problem=np.array([r["problem"] for r in all_rows]),
             **{k: np.array([r[k] for r in all_rows], float)
                for k in all_rows[0] if k != "problem"})
    print("\nwrote measure_results.npz")
