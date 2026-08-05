"""Synthetic evaluation of the 2C construction against the explicit mixture.

Everything is Gaussian and closed-form, so the *true* PMI(y ; r | x) is known
and the two estimators can be scored against it rather than against each other.

Generative truth
    y ~ Uniform{1..C},  s = f with probability eps, else r
    x | (y, s) ~ N(mu[y] + delta * (s == f) * v[y],  Sigma_s)
so the tag is a genuine property of x when delta > 0 -- and, because the
concept direction v[y] is per class, one that genuinely couples y and s. It
carries nothing at all when delta = 0.

Two estimators, both plain LDA, differing only in where the normalisation is:
    2C  : 2C mean vectors, ONE pooled covariance over all n points
    Mix : C means and Sigma_r on the r subset, C means and Sigma_f on the
          f subset -- the small one -- then combined by the prior
and a Mix variant with Ledoit-Wolf shrinkage, as the steelman of the baseline.

Run: python experiment.py       (a couple of minutes on CPU)
"""
import numpy as np
from sklearn.covariance import LedoitWolf

RIDGE = 1e-6            # same for every estimator; numerical only
N_TEST = 4000
MU_SCALE = 0.45         # class separation: tuned so the Bayes rate sits near 0.87
                        # at C = 5, d = 30 -- an easy problem stays at 1.000 and
                        # measures nothing


# --------------------------------------------------------------------------
# generative model
# --------------------------------------------------------------------------
def make_truth(rng, C, d, delta, hetero):
    """Class means, the concept directions, and the per-tag covariances.

    The shift has to be *class-dependent*. A tag that moves every class the
    same way leaves y and s all but conditionally independent given x, so the
    PMI it induces is near zero and there is genuinely nothing for any
    estimator to find -- which the first version of this script measured at
    length before the reason became clear.
    """
    mu = rng.normal(scale=MU_SCALE, size=(C, d))
    v = rng.normal(size=(C, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    A = rng.normal(size=(d, d)) / np.sqrt(d)
    Sigma_r = A @ A.T + 0.5 * np.eye(d)
    if hetero:                                   # the tag also changes the shape
        Bm = rng.normal(size=(d, d)) / np.sqrt(d)
        Sigma_f = 2.0 * (Bm @ Bm.T) + 0.5 * np.eye(d)
    else:                                        # one covariance, as 2C assumes
        Sigma_f = Sigma_r
    means = np.stack([mu, mu + delta * v])       # [s, y, d] with s = 0 (r), 1 (f)
                                                 # v is per class: the concept
                                                 # interacts with the label
    return means, np.stack([Sigma_r, Sigma_f])


def sample(rng, means, Sigmas, n, C, eps):
    y = rng.integers(0, C, size=n)
    s = (rng.random(n) < eps).astype(int)
    L = [np.linalg.cholesky(S) for S in Sigmas]
    x = np.empty((n, means.shape[2]))
    for si in (0, 1):
        m = s == si
        if m.any():
            z = rng.normal(size=(m.sum(), means.shape[2]))
            x[m] = means[si][y[m]] + z @ L[si].T
    return x, y, s


def log_joint(x, means, Sigmas, logpi):
    """log p(x, y, s) for every cell, shape [n, 2, C]."""
    n, d = x.shape
    out = np.empty((n, 2, means.shape[1]))
    for si in range(2):
        S = Sigmas[si] + RIDGE * np.eye(d)
        sign, logdet = np.linalg.slogdet(S)
        P = np.linalg.inv(S)
        for yi in range(means.shape[1]):
            r = x - means[si, yi]
            out[:, si, yi] = (-0.5 * np.einsum("ij,jk,ik->i", r, P, r)
                              - 0.5 * logdet - 0.5 * d * np.log(2 * np.pi)
                              + logpi[si, yi])
    return out


def pmi_from_joint(lj):
    """Delta M = log P_r(y|x) - log P(y|x) = PMI(y ; r | x), shape [n, C]."""
    lse = lambda a, ax: np.log(np.exp(a - a.max(ax, keepdims=True)).sum(ax)) \
        + a.max(ax)
    log_pr_y = lj[:, 0, :] - lse(lj[:, 0, :], 1)[:, None]       # log P(y | x, r)
    log_p_y = lse(lj, 1) - lse(lj.reshape(len(lj), -1), 1)[:, None]
    return log_pr_y - log_p_y


def posterior_y(lj):
    lse = lambda a, ax: np.log(np.exp(a - a.max(ax, keepdims=True)).sum(ax)) \
        + a.max(ax)
    lg = lse(lj, 1)
    return np.exp(lg - lse(lg, 1)[:, None])


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------
def _pooled(x, groups, d, shrink=False):
    """Pooled within-group scatter. Returns (Sigma, degrees of freedom)."""
    res, dof = [], 0
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 2:
            continue
        res.append(x[m] - x[m].mean(0))
        dof += m.sum() - 1
    if not res:
        return np.eye(d) * np.nan, 0
    R = np.vstack(res)
    if shrink:
        return LedoitWolf(assume_centered=True).fit(R).covariance_, dof
    return R.T @ R / max(dof, 1), dof


def fit_2c(x, y, s, C, d):
    """One normalisation over 2C cells: 2C means, one pooled covariance."""
    cell = s * C + y
    means = np.zeros((2, C, d))
    logpi = np.full((2, C), -np.inf)
    for si in range(2):
        for yi in range(C):
            m = (s == si) & (y == yi)
            if m.any():
                means[si, yi] = x[m].mean(0)
                logpi[si, yi] = np.log(m.mean())
    Sigma, dof = _pooled(x, cell, d)
    return means, np.stack([Sigma, Sigma]), logpi, dict(dof_f=dof, dof_r=dof)


def fit_mix(x, y, s, C, d, shrink=False):
    """Two normalisations: one LDA per tag, the f one on the f subset alone."""
    means = np.zeros((2, C, d))
    logpi = np.full((2, C), -np.inf)
    Sig, dofs = [], {}
    for si in range(2):
        ms = s == si
        for yi in range(C):
            m = ms & (y == yi)
            if m.any():
                means[si, yi] = x[m].mean(0)
                logpi[si, yi] = np.log(m.mean())
        S, dof = _pooled(x[ms], y[ms], d, shrink=shrink)
        Sig.append(S)
        dofs["dof_r" if si == 0 else "dof_f"] = dof
    return means, np.stack(Sig), logpi, dofs


def fit_plain(x, y, C, d):
    """The release: ordinary LDA over C classes, blind to the tag."""
    means = np.zeros((C, d))
    logpi = np.zeros(C)
    for yi in range(C):
        m = y == yi
        means[yi] = x[m].mean(0)
        logpi[yi] = np.log(m.mean())
    Sigma, _ = _pooled(x, y, d)
    S = Sigma + RIDGE * np.eye(d)
    P = np.linalg.inv(S)

    def logits(xt):
        return np.stack([-0.5 * np.einsum("ij,jk,ik->i", xt - means[yi], P,
                                          xt - means[yi]) + logpi[yi]
                         for yi in range(C)], axis=1)
    return logits


def cond(S):
    w = np.linalg.eigvalsh(S)
    return w.max() / max(w.min(), 1e-300)


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------
def run_eps(seed, C=5, d=30, n=4000, eps=0.1, delta=1.0, hetero=False):
    """One draw: estimate Delta M with 2C and with Mix, score against truth."""
    rng = np.random.default_rng(seed)
    means, Sigmas = make_truth(rng, C, d, delta, hetero)
    x, y, s = sample(rng, means, Sigmas, n, C, eps)
    xt, yt, st = sample(rng, means, Sigmas, N_TEST, C, eps)

    logpi_true = np.log(np.array([[(1 - eps) / C] * C, [eps / C] * C]))
    pmi_true = pmi_from_joint(log_joint(xt, means, Sigmas, logpi_true))

    # The truth shrinks with eps -- P(f) small means P_r is close to P and the
    # PMI is close to 0 -- so an absolute RMSE rewards an estimator that gives
    # up and predicts zero. Report it relative to the signal, and report how
    # much of the signal's shape survives, which giving up cannot fake.
    scale_true = float(np.sqrt((pmi_true ** 2).mean()))
    out = dict(n_f=int((s == 1).sum()), pmi_rms=scale_true)
    fits = dict(twoc=fit_2c(x, y, s, C, d),
                mix=fit_mix(x, y, s, C, d),
                mix_lw=fit_mix(x, y, s, C, d, shrink=True))
    for name, (m, S, lp, dofs) in fits.items():
        lj = log_joint(xt, m, S, lp)
        pmi = pmi_from_joint(lj)
        rmse = float(np.sqrt(((pmi - pmi_true) ** 2).mean()))
        out[f"rmse_{name}"] = rmse
        out[f"nrmse_{name}"] = rmse / scale_true
        out[f"corr_{name}"] = float(np.corrcoef(pmi.ravel(),
                                                pmi_true.ravel())[0, 1])
        out[f"mag_{name}"] = float(np.sqrt((pmi ** 2).mean())) / scale_true
        out[f"acc_{name}"] = float((posterior_y(lj).argmax(1) == yt).mean())
        out[f"cond_f_{name}"] = float(cond(S[1] + RIDGE * np.eye(d)))
        out[f"dof_f_{name}"] = dofs["dof_f"]

    # the sign test of Eq. (2): tau = E[ softmax(release logits)' Delta M ]
    rel = fit_plain(x, y, C, d)(xt)
    p_rel = np.exp(rel - rel.max(1, keepdims=True))
    p_rel /= p_rel.sum(1, keepdims=True)
    m, S, lp, _ = fits["twoc"]
    dM = pmi_from_joint(log_joint(xt, m, S, lp))
    out["tau"] = float((p_rel * dM).sum(1).mean())
    out["tau_true"] = float((p_rel * pmi_true).sum(1).mean())
    out["acc_plain"] = float((rel.argmax(1) == yt).mean())
    return out


def check_tau_identity(seed=0, C=5, d=30, n=4000, eps=0.1, delta=1.0):
    """tau is a negative KL whenever the corrected model is the proxy's marginal.

    sum_y P(y|x) [log P_r(y|x) - log P(y|x)] = -KL( P(.|x) || P_r(.|x) ) <= 0
    identically, for any pair of distributions. So if the model being corrected
    is the 2C fit's own marginal, tau < 0 carries no information at all: it is
    negative whenever Delta M is not identically zero, signal or no signal.
    """
    rng = np.random.default_rng(seed)
    means, Sigmas = make_truth(rng, C, d, delta, False)
    x, y, s = sample(rng, means, Sigmas, n, C, eps)
    xt, *_ = sample(rng, means, Sigmas, N_TEST, C, eps)
    m, S, lp, _ = fit_2c(x, y, s, C, d)
    lj = log_joint(xt, m, S, lp)
    dM, P = pmi_from_joint(lj), posterior_y(lj)
    tau_self = float((P * dM).sum(1).mean())
    kl = float(-(P * dM).sum(1).mean())          # by the same algebra
    assert tau_self <= 0, tau_self
    assert abs(tau_self + kl) < 1e-12
    return dict(tau_self=tau_self, mean_kl=kl)


def run_sign(seed, C=5, d=30, n=4000, eps=0.1, delta=1.0, n_perm=12):
    """tau against its own permutation null: shuffle the tag, refit, recompute.

    Under a shuffled tag the construction sees a genuinely uninformative s, so
    the spread of tau over permutations is exactly the noise floor of the test.
    """
    rng = np.random.default_rng(seed)
    means, Sigmas = make_truth(rng, C, d, delta, False)
    x, y, s = sample(rng, means, Sigmas, n, C, eps)
    xt, yt, _ = sample(rng, means, Sigmas, N_TEST, C, eps)

    rel = fit_plain(x, y, C, d)(xt)
    p_rel = np.exp(rel - rel.max(1, keepdims=True))
    p_rel /= p_rel.sum(1, keepdims=True)

    def tau_of(tag):
        m, S, lp, _ = fit_2c(x, y, tag, C, d)
        return float((p_rel * pmi_from_joint(log_joint(xt, m, S, lp))).sum(1).mean())

    tau = tau_of(s)
    null = np.array([tau_of(rng.permutation(s)) for _ in range(n_perm)])
    z = (tau - null.mean()) / max(null.std(ddof=1), 1e-300)
    return dict(tau=tau, null_mean=float(null.mean()),
                null_sd=float(null.std(ddof=1)), z=float(z),
                below_null=bool(tau < null.min()))


def run_capacity(seed, C, d=30, n=4000, eps=0.2, delta=1.0):
    """Accuracy on y: 2C marginalised against the tag-blind release."""
    rng = np.random.default_rng(seed)
    means, Sigmas = make_truth(rng, C, d, delta, False)
    x, y, s = sample(rng, means, Sigmas, n, C, eps)
    xt, yt, _ = sample(rng, means, Sigmas, N_TEST, C, eps)
    m, S, lp, _ = fit_2c(x, y, s, C, d)
    acc_2c = float((posterior_y(log_joint(xt, m, S, lp)).argmax(1) == yt).mean())
    acc_plain = float((fit_plain(x, y, C, d)(xt).argmax(1) == yt).mean())
    logpi_true = np.log(np.array([[(1 - eps) / C] * C, [eps / C] * C]))
    acc_bayes = float((posterior_y(log_joint(xt, means, Sigmas, logpi_true))
                       .argmax(1) == yt).mean())
    return dict(C=C, acc_2c=acc_2c, acc_plain=acc_plain, acc_bayes=acc_bayes)


def agg(rows, key):
    a = np.array([r[key] for r in rows], dtype=float)
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))


if __name__ == "__main__":
    SEEDS = range(20)
    EPS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

    def table1(rows, eps, extra=""):
        nf = np.mean([r["n_f"] for r in rows])
        cells = []
        for k in ("twoc", "mix", "mix_lw"):
            n_mu, n_se = agg(rows, f"nrmse_{k}")
            c_mu, _ = agg(rows, f"corr_{k}")
            m_mu, _ = agg(rows, f"mag_{k}")
            cells.append(f"{n_mu:.2f}±{n_se:.2f} {c_mu:>5.2f} {m_mu:>5.2f}")
        cmu, _ = agg(rows, "cond_f_mix")
        pm, _ = agg(rows, "pmi_rms")
        print(f"{eps:>6} {nf:>6.0f} {pm:>6.2f} | {cells[0]} | {cells[1]} | {cells[2]} "
              f"| {cmu:>8.1e}{extra}")

    print("=" * 100)
    print("1. Delta M against the true PMI, as the f subgroup shrinks")
    print("   C=5, d=30, n=4000, delta=1.0, one true covariance")
    print("   per estimator: normalised RMSE (rel. to the RMS of the true PMI),")
    print("   correlation with the true PMI, and relative magnitude")
    print(f"{'eps':>6} {'n_f':>6} {'|PMI|':>6} | {'2C':^21} | {'Mix':^21} "
          f"| {'Mix+LW':^21} | {'cond Sf':>8}")
    hom = {}
    for eps in EPS:
        hom[eps] = [run_eps(s, eps=eps) for s in SEEDS]
        table1(hom[eps], eps)

    print("\n   same, but the tag really does change the covariance (hetero):")
    het = {}
    for eps in (0.2, 0.05, 0.02):
        het[eps] = [run_eps(s, eps=eps, hetero=True) for s in SEEDS]
        table1(het[eps], eps)

    print("\n" + "=" * 78)
    print("2. What the extra cells cost on the original task")
    print("   d=30, n=4000, eps=0.2; accuracy on y, 2C marginalised vs tag-blind")
    print(f"{'C':>4} {'2C cells':>9} | {'acc 2C':>16} {'acc plain':>16} "
          f"{'Bayes':>8}")
    cap = {}
    for C in (2, 3, 5, 8, 12, 20, 32):
        rows = [run_capacity(s, C) for s in SEEDS]
        cap[C] = rows
        a2, s2 = agg(rows, "acc_2c")
        ap, sp = agg(rows, "acc_plain")
        ab, _ = agg(rows, "acc_bayes")
        print(f"{C:>4} {2*C:>9} | {a2:.4f} ± {s2:.4f}  {ap:.4f} ± {sp:.4f}  "
              f"{ab:>8.4f}")

    print("\n" + "=" * 100)
    print("3. The sign test, and what it is worth")
    print("   C=5, d=30, n=4000, eps=0.1; tau against a 12-permutation null")
    print(f"{'delta':>7} | {'tau':>19} {'null mean':>11} {'null sd':>9} "
          f"{'z':>7} | {'P[tau<0]':>9} {'P[below null]':>13}")
    SIGN_SEEDS = range(12)
    DELTAS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
    sign = {}
    for delta in DELTAS:
        rows = [run_sign(s, delta=delta) for s in SIGN_SEEDS]
        sign[delta] = rows
        tm, ts = agg(rows, "tau")
        nm, _ = agg(rows, "null_mean")
        ns, _ = agg(rows, "null_sd")
        zm, _ = agg(rows, "z")
        frac_neg = np.mean([r["tau"] < 0 for r in rows])
        frac_below = np.mean([r["below_null"] for r in rows])
        print(f"{delta:>7} | {tm:>+11.5f}±{ts:.5f} {nm:>+11.5f} {ns:>9.5f} "
              f"{zm:>7.1f} | {frac_neg:>8.0%} {frac_below:>13.0%}")

    keys1 = ("nrmse_twoc", "nrmse_mix", "nrmse_mix_lw", "corr_twoc", "corr_mix",
             "corr_mix_lw", "mag_twoc", "mag_mix", "mag_mix_lw", "n_f",
             "cond_f_mix", "pmi_rms")
    np.savez("experiment_results.npz",
             keys1=np.array(keys1),
             eps=np.array(EPS),
             hom=np.array([[[r[k] for k in keys1] for r in hom[e]] for e in EPS]),
             het_eps=np.array([0.2, 0.05, 0.02]),
             het=np.array([[[r[k] for k in keys1] for r in het[e]]
                           for e in (0.2, 0.05, 0.02)]),
             cap_C=np.array([2, 3, 5, 8, 12, 20, 32]),
             cap=np.array([[[r[k] for k in ("acc_2c", "acc_plain", "acc_bayes")]
                            for r in cap[C]] for C in (2, 3, 5, 8, 12, 20, 32)]),
             sign_delta=np.array(DELTAS),
             sign=np.array([[[r[k] for k in ("tau", "null_mean", "null_sd", "z")]
                             for r in sign[dl]] for dl in DELTAS]))
    print("\nwrote experiment_results.npz")
