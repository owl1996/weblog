"""Real shadow softmax clouds: is the Dirichlet the right model on the simplex?

The geometry in the post is exact *given* the Dirichlet parametrisation.  This
script asks what that assumption costs on softmax outputs a real training
pipeline actually produces.

Setup: `sklearn.datasets.load_digits` (1797 points, 64 features, C = 10) with
15% of the labels flipped, so that some points are genuinely memorised and
membership is detectable at all.  M shadow MLPs are trained on independent
random halves; for each attack point the shadows split by themselves into the
side that saw it and the side that did not, which is the standard LiRA
bookkeeping [carlini2022membership].  All M are trained as one batched tensor
program, four chunks at a time.

Measured, per attack point, on shadows held out of the fit:
  * held-out log-density on the simplex under three models -- Dirichlet
    (method of moments), logistic-normal with diagonal covariance, and
    logistic-normal with a Ledoit-Wolf full covariance;
  * whether the fitted member region is bounded, i.e. alpha_x > alpha_xr
    coordinatewise;
  * AUC and TPR at low FPR for the simplex likelihood ratio and for U-LiRA's
    scalar test.

Writes shadow_results.npz.  Run: python shadows.py    (a few minutes on CPU)
"""
import numpy as np
import torch
from scipy.special import gammaln
from sklearn.covariance import LedoitWolf
from sklearn.datasets import load_digits

from llr import kl, entropy, mom, to_alpha, llr_closed_form
from synthetic import auc, tpr_at_fpr, logit, ulira_fit, ulira_score

M_SHADOW = 512
CHUNK = 128
HIDDEN = 64
STEPS = 1500
LR = 3e-3
WD = 1e-4
LABEL_NOISE = 0.15
FRAC_IN = 0.5              # each shadow sees a random half
N_ATTACK = 200
FIT_FRAC = 0.65            # shadows used to fit; the rest are the releases
EPS = 1e-9
SEED = 0


# --------------------------------------------------------------------------
# data and shadow training
# --------------------------------------------------------------------------
def make_data(rng):
    d = load_digits()
    X = d.data.astype(np.float32) / 16.0
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    y = d.target.astype(np.int64).copy()
    n = len(y)
    flip = rng.choice(n, size=int(LABEL_NOISE * n), replace=False)
    y[flip] = (y[flip] + rng.integers(1, 10, size=len(flip))) % 10
    return X, y, flip


def train_chunk(X, y, mask, steps=STEPS, seed=0):
    """Train `mask.shape[0]` MLPs in parallel; return softmax on all of X."""
    g = torch.Generator().manual_seed(seed)
    m, n = mask.shape
    d, C = X.shape[1], int(y.max()) + 1
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    Mt = torch.from_numpy(mask.astype(np.float32))

    def p(*shape, gain):
        return torch.nn.Parameter(
            torch.randn(*shape, generator=g) * gain)

    W1 = p(m, d, HIDDEN, gain=(2.0 / d) ** 0.5)
    b1 = torch.nn.Parameter(torch.zeros(m, 1, HIDDEN))
    W2 = p(m, HIDDEN, C, gain=(2.0 / HIDDEN) ** 0.5)
    b2 = torch.nn.Parameter(torch.zeros(m, 1, C))
    params = [W1, b1, W2, b2]
    opt = torch.optim.Adam(params, lr=LR, weight_decay=WD)

    Xb = Xt.unsqueeze(0)                                  # [1, n, d]
    denom = Mt.sum(1, keepdim=True).clamp(min=1.0)        # [m, 1]
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        h = torch.relu(Xb @ W1 + b1)                      # [m, n, H]
        z = h @ W2 + b2                                   # [m, n, C]
        ce = torch.nn.functional.cross_entropy(
            z.reshape(-1, C), yt.repeat(m), reduction="none").view(m, n)
        ((ce * Mt).sum(1, keepdim=True) / denom).mean().backward()
        opt.step()
    with torch.no_grad():
        h = torch.relu(Xb @ W1 + b1)
        s = torch.softmax(h @ W2 + b2, dim=-1)
    return s.numpy()


CACHE = "shadow_softmax.npz"


def run_shadows(X, y, rng):
    import os
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        print(f"   reusing {CACHE}")
        return z["S"], z["mask"]
    n = len(y)
    mask = np.zeros((M_SHADOW, n), dtype=bool)
    for m in range(M_SHADOW):
        mask[m, rng.choice(n, size=int(FRAC_IN * n), replace=False)] = True
    S = np.empty((M_SHADOW, n, int(y.max()) + 1), dtype=np.float32)
    for i0 in range(0, M_SHADOW, CHUNK):
        i1 = min(i0 + CHUNK, M_SHADOW)
        S[i0:i1] = train_chunk(X, y, mask[i0:i1], seed=SEED + i0)
        acc = (S[i0:i1].argmax(-1) == y[None]).mean(1)
        seen = np.array([(S[i0 + j][mask[i0 + j]].argmax(-1)
                          == y[mask[i0 + j]]).mean() for j in range(i1 - i0)])
        print(f"   shadows {i0:4d}-{i1 - 1:4d}: train acc {seen.mean():.3f}"
              f"  all-data acc {acc.mean():.3f}")
    np.savez_compressed(CACHE, S=S, mask=mask)
    return S, mask


# --------------------------------------------------------------------------
# densities on the simplex
# --------------------------------------------------------------------------
def dirichlet_logpdf(P, alpha):
    P = np.clip(P, EPS, None)
    return (((alpha - 1.0) * np.log(P)).sum(-1)
            + gammaln(alpha.sum()) - gammaln(alpha).sum())


def alr_coords(P):
    P = np.clip(P, EPS, None)
    return np.log(P[..., :-1]) - np.log(P[..., -1:])


def logistic_normal_logpdf(P, mu, prec, logdet_prec):
    """Density on the simplex; the -sum log P_k term is the alr Jacobian."""
    v = alr_coords(P) - mu
    quad = np.einsum("ij,jk,ik->i", v, prec, v)
    k = mu.shape[0]
    return (0.5 * logdet_prec - 0.5 * k * np.log(2 * np.pi) - 0.5 * quad
            - np.log(np.clip(P, EPS, None)).sum(-1))


def fit_logistic_normal(S, full):
    V = alr_coords(S)
    mu = V.mean(0)
    if full:
        cov = LedoitWolf(assume_centered=False).fit(V).covariance_
    else:
        cov = np.diag(V.var(0, ddof=1))
    cov = cov + 1e-6 * np.eye(len(mu))
    sign, logdet = np.linalg.slogdet(cov)
    return mu, np.linalg.inv(cov), -logdet


# --------------------------------------------------------------------------
# per-attack-point evaluation
# --------------------------------------------------------------------------
def aggregate(P, c, comp):
    """Push the C-simplex down to the 3-simplex (correct, competitor, rest).

    A deterministic map between simplices, so data processing applies to it
    too: whatever it keeps is a lower bound on what the full vector keeps.
    """
    a = P[:, c]
    b = P[:, comp]
    return np.stack([a, b, np.clip(1.0 - a - b, EPS, None)], 1)


def lognormal_llr(P, S_in, S_out, full):
    """QDA on alr coordinates. The Jacobian cancels between the two sides."""
    mi, pi, ldi = fit_logistic_normal(S_in, full)
    mo, po, ldo = fit_logistic_normal(S_out, full)
    v = alr_coords(np.clip(P, EPS, None))
    qi = np.einsum("ij,jk,ik->i", v - mi, pi, v - mi)
    qo = np.einsum("ij,jk,ik->i", v - mo, po, v - mo)
    return 0.5 * (ldi - ldo) - 0.5 * (qi - qo)


ATTACKS = ("ulira", "dir3", "dir10", "ln3", "ln10")


def evaluate(S, mask, y, rng, pts):
    rows = []
    scores = {k: ([], []) for k in ATTACKS}
    clouds = {}
    for i in pts:
        idx_in = np.flatnonzero(mask[:, i])
        idx_out = np.flatnonzero(~mask[:, i])
        rng.shuffle(idx_in)
        rng.shuffle(idx_out)
        ki, ko = int(FIT_FRAC * len(idx_in)), int(FIT_FRAC * len(idx_out))
        fit_in, rel_in = idx_in[:ki], idx_in[ki:]
        fit_out, rel_out = idx_out[:ko], idx_out[ko:]

        Sx, Sr = S[fit_in, i].astype(np.float64), S[fit_out, i].astype(np.float64)
        Px, Pr = S[rel_in, i].astype(np.float64), S[rel_out, i].astype(np.float64)
        Sx = Sx / Sx.sum(1, keepdims=True)
        Sr = Sr / Sr.sum(1, keepdims=True)
        Px = Px / Px.sum(1, keepdims=True)
        Pr = Pr / Pr.sum(1, keepdims=True)

        side_x, side_r = mom(Sx, "minka"), mom(Sr, "minka")
        ax, ar = to_alpha(*side_x), to_alpha(*side_r)

        ll = {}
        for name, S_fit, P_hold in (("x", Sx, Px), ("r", Sr, Pr)):
            a = to_alpha(*mom(S_fit, "minka"))
            ll[f"dir_{name}"] = dirichlet_logpdf(P_hold, a).mean()
            for tag, full in (("lnd", False), ("lnf", True)):
                mu, prec, ldp = fit_logistic_normal(S_fit, full)
                ll[f"{tag}_{name}"] = logistic_normal_logpdf(
                    P_hold, mu, prec, ldp).mean()

        c = int(y[i])
        fi, fo = ulira_fit(Sx, c), ulira_fit(Sr, c)
        scores["ulira"][0].append(ulira_score(Px, c, fi, fo))
        scores["ulira"][1].append(ulira_score(Pr, c, fi, fo))
        scores["dir10"][0].append(llr_closed_form(Px, *side_x, *side_r))
        scores["dir10"][1].append(llr_closed_form(Pr, *side_x, *side_r))
        scores["ln10"][0].append(lognormal_llr(Px, Sx, Sr, True))
        scores["ln10"][1].append(lognormal_llr(Pr, Sx, Sr, True))

        comp = int(np.argsort(Sr.mean(0))[-1])
        comp = int(np.argsort(Sr.mean(0))[-2]) if comp == c else comp
        Ax, Ar = aggregate(Sx, c, comp), aggregate(Sr, c, comp)
        Bx, Br = aggregate(Px, c, comp), aggregate(Pr, c, comp)
        s3x, s3r = mom(Ax, "minka"), mom(Ar, "minka")
        scores["dir3"][0].append(llr_closed_form(Bx, *s3x, *s3r))
        scores["dir3"][1].append(llr_closed_form(Br, *s3x, *s3r))
        scores["ln3"][0].append(lognormal_llr(Bx, Ax, Ar, True))
        scores["ln3"][1].append(lognormal_llr(Br, Ax, Ar, True))
        clouds[int(i)] = dict(Ax=Ax, Ar=Ar, c=c, comp=comp,
                              s3x=np.asarray(s3x[0]), n3x=s3x[1],
                              s3r=np.asarray(s3r[0]), n3r=s3r[1])

        rows.append(dict(
            i=i, n_in=len(idx_in), bounded=bool(np.all(ax > ar)),
            nu_x=side_x[1], nu_r=side_r[1],
            frac_beta_neg=float(np.mean(ax <= ar)),
            conf_x=float(Sx[:, c].mean()), conf_r=float(Sr[:, c].mean()),
            **{k: float(v) for k, v in ll.items()}))
    return rows, scores, clouds


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    X, y, flip = make_data(rng)
    print(f"digits: n={len(y)} d={X.shape[1]} C=10, "
          f"{len(flip)} flipped labels")
    print(f"training {M_SHADOW} shadow MLPs "
          f"({X.shape[1]}-{HIDDEN}-10, {STEPS} full-batch Adam steps)")
    S, mask = run_shadows(X, y, rng)

    pts = rng.choice(len(y), size=N_ATTACK, replace=False)
    rows, scores, clouds = evaluate(S, mask, y, rng, pts)

    keys = rows[0].keys()
    arr = {k: np.array([r[k] for r in rows]) for k in keys}
    print(f"\n{N_ATTACK} attack points, "
          f"{int(arr['n_in'].mean())} in-shadows on average, "
          f"{int(FIT_FRAC * 100)}% of each side used for the fit")
    print(f"  bounded member region: {arr['bounded'].mean():.3f} of points; "
          f"median share of coordinates with alpha_x <= alpha_xr: "
          f"{np.median(arr['frac_beta_neg']):.2f}")
    print(f"  concentration: median nu_x {np.median(arr['nu_x']):.1f}, "
          f"nu_r {np.median(arr['nu_r']):.1f}, "
          f"nu_x > nu_r on {np.mean(arr['nu_x'] > arr['nu_r']):.3f}")
    print("\n  held-out log-density on the simplex (higher is better, nats):")
    for side, lab in (("x", "saw z    "), ("r", "did not  ")):
        print(f"    {lab} Dirichlet {arr['dir_' + side].mean():8.2f}"
              f"   log-normal diag {arr['lnd_' + side].mean():8.2f}"
              f"   full {arr['lnf_' + side].mean():8.2f}")
    win = np.mean(arr["dir_x"] > arr["lnf_x"])
    print(f"    Dirichlet beats the full log-normal on "
          f"{win:.3f} of points (seen side)")

    print("\n  attack, pooled over attack points, with a 95% bootstrap "
          "interval resampling attack points (B=1000):")
    out = {}
    boot = np.random.default_rng(7)
    idx_pt = np.arange(len(pts))
    draws = boot.integers(0, len(pts), size=(1000, len(pts)))
    for k in ATTACKS:
        a, b = scores[k]
        ain, aout = np.concatenate(a), np.concatenate(b)
        out[f"auc_{k}"] = auc(ain, aout)
        out[f"tpr5_{k}"] = tpr_at_fpr(ain, aout, 0.05)
        out[f"tpr1_{k}"] = tpr_at_fpr(ain, aout, 0.01)
        out[f"score_in_{k}"], out[f"score_out_{k}"] = ain, aout
        out[f"pointauc_{k}"] = np.array([auc(a[j], b[j]) for j in idx_pt])
        ba = np.array([auc(np.concatenate([a[j] for j in dr]),
                           np.concatenate([b[j] for j in dr]))
                       for dr in draws[:200]])
        bt = np.array([tpr_at_fpr(np.concatenate([a[j] for j in dr]),
                                  np.concatenate([b[j] for j in dr]), 0.01)
                       for dr in draws[:200]])
        out[f"auc_ci_{k}"] = np.quantile(ba, [0.025, 0.975])
        out[f"tpr1_ci_{k}"] = np.quantile(bt, [0.025, 0.975])
        lo, hi = out[f"auc_ci_{k}"]
        tlo, thi = out[f"tpr1_ci_{k}"]
        print(f"    {k:8s} AUC {out['auc_' + k]:.4f} [{lo:.4f},{hi:.4f}]"
              f"   TPR@1%FPR {out['tpr1_' + k]:.4f} [{tlo:.4f},{thi:.4f}]"
              f"   TPR@5% {out['tpr5_' + k]:.4f}")

    print("\n  paired per-point AUC, each attack minus U-LiRA "
          "(mean and 95% bootstrap interval over the 200 points):")
    base = out["pointauc_ulira"]
    for k in ATTACKS:
        if k == "ulira":
            continue
        diff = out[f"pointauc_{k}"] - base
        bs = diff[draws].mean(1)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        out[f"pairedauc_{k}"] = np.array([diff.mean(), lo, hi])
        flag = "" if lo <= 0 <= hi else "   *"
        print(f"    {k:8s} {diff.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]{flag}")

    # one representative cloud for the figure: the point whose fitted
    # 3-simplex sides are best separated
    best = max(clouds, key=lambda i: float(np.linalg.norm(
        clouds[i]["s3x"] - clouds[i]["s3r"])))
    cl = {f"cloud_{k}": v for k, v in clouds[best].items()}
    print(f"\n  representative attack point for the figure: index {best}, "
          f"class {clouds[best]['c']}, competitor {clouds[best]['comp']}")

    np.savez_compressed("shadow_results.npz", **arr, **out, **cl,
                        cloud_index=best, flipped=np.isin(pts, flip))
    print("\nwrote shadow_results.npz")


if __name__ == "__main__":
    main()
