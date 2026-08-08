"""The Dirichlet likelihood-ratio test on the simplex: identities and geometry.

Everything the post asserts analytically is checked here against brute force:

  1. the closed form  log L = nu_r KL(s_r||P) - nu KL(s||P) + K  against a
     direct difference of two `scipy.stats.dirichlet.logpdf` calls;
  2. log L is *affine in log P* -- the residual after removing <beta, log P>
     is constant over the simplex;
  3. in additive-log-ratio coordinates log L is concave iff nu >= nu_r, so the
     member region is convex;
  4. the member region is bounded  <=>  alpha_x > alpha_{x,r} coordinatewise,
     tested by ray-marching against the criterion on random parameter pairs;
  5. the method-of-moments estimator: the arithmetic-mean and Minka forms
     recover nu, and the 1/(C-1) normalisation over all C coordinates -- as
     printed in the paper draft -- does not.

Run: python llr.py      (a few seconds)
"""
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import dirichlet

EPS = 1e-12


# --------------------------------------------------------------------------
# the two parametrisations
# --------------------------------------------------------------------------
def to_alpha(sbar, nu):
    """mean-precision (s, nu) -> natural concentration alpha = nu * s."""
    return np.asarray(nu)[..., None] * np.asarray(sbar)


def to_mean_precision(alpha):
    nu = alpha.sum(-1)
    return alpha / nu[..., None], nu


# --------------------------------------------------------------------------
# 1. the closed form
# --------------------------------------------------------------------------
def kl(p, q):
    """KL(p || q) for rows of p, q on the simplex."""
    p = np.clip(p, EPS, None)
    q = np.clip(q, EPS, None)
    return (p * (np.log(p) - np.log(q))).sum(-1)


def entropy(p):
    p = np.clip(p, EPS, None)
    return -(p * np.log(p)).sum(-1)


def llr_constant(sx, nux, sr, nur):
    """K in eq. (dir-cst): everything independent of the release."""
    ax, ar = to_alpha(sx, nux), to_alpha(sr, nur)
    return (nur * entropy(sr) - nux * entropy(sx)
            + gammaln(nux) - gammaln(nur)
            + (gammaln(ar) - gammaln(ax)).sum(-1))


def llr_closed_form(P, sx, nux, sr, nur):
    """log Lambda_x(P) via the two precision-scaled KL divergences."""
    return (nur * kl(sr, P) - nux * kl(sx, P)
            + llr_constant(sx, nux, sr, nur))


def llr_direct(P, sx, nux, sr, nur):
    """log Lambda by brute force: two Dirichlet log-densities."""
    ax, ar = to_alpha(sx, nux), to_alpha(sr, nur)
    P = np.atleast_2d(P)
    out = np.array([dirichlet.logpdf(p / p.sum(), ax)
                    - dirichlet.logpdf(p / p.sum(), ar) for p in P])
    return out


def llr_loglinear(P, sx, nux, sr, nur):
    """log Lambda as an affine function of log P: <beta, log P> + const."""
    beta = to_alpha(sx, nux) - to_alpha(sr, nur)
    const = (gammaln(nux) - gammaln(nur)
             + (gammaln(to_alpha(sr, nur)) - gammaln(to_alpha(sx, nux))).sum(-1))
    return (beta * np.log(np.clip(P, EPS, None))).sum(-1) + const


# --------------------------------------------------------------------------
# 3-4. geometry in additive-log-ratio coordinates
# --------------------------------------------------------------------------
def alr(P):
    """v_k = log(P_k / P_C), k = 1..C-1."""
    P = np.clip(P, EPS, None)
    return np.log(P[..., :-1]) - np.log(P[..., -1:])


def alr_inv(v):
    u = np.concatenate([v, np.zeros(v.shape[:-1] + (1,))], axis=-1)
    return np.exp(u - logsumexp(u, axis=-1, keepdims=True))


def llr_alr(v, sx, nux, sr, nur):
    """log Lambda in alr coordinates: <beta_{1:C-1}, v> - dnu * lse(v, 0) + c.

    Evaluated directly rather than by mapping back to the simplex, which
    underflows long before the asymptotics are visible.  The lse term is
    convex and enters with weight -dnu, so the statistic is concave exactly
    when dnu = nu_x - nu_{x,r} >= 0.
    """
    v = np.atleast_2d(v)
    ax, ar = to_alpha(sx, nux), to_alpha(sr, nur)
    beta = ax - ar
    lse = logsumexp(np.concatenate([v, np.zeros(v.shape[:-1] + (1,))], -1), -1)
    const = gammaln(nux) - gammaln(nur) + (gammaln(ar) - gammaln(ax)).sum(-1)
    return (beta[:-1] * v).sum(-1) - (nux - nur) * lse + const


def region_bounded_by_raymarch(sx, nux, sr, nur, n_dir=20000, t=400.0, seed=0):
    """Is every superlevel set of log Lambda bounded, in alr coordinates?

    Rather than test one threshold -- which a slowly escaping branch can
    survive at any finite radius -- compare the best value on two spheres.
    On a bounded region log Lambda is eventually decreasing along every ray,
    so max_d log Lambda(2t d) < max_d log Lambda(t d); if any branch escapes,
    the max grows linearly instead.  Directions include the coordinate axes,
    which is where the escape happens when a single beta_k is negative.
    """
    C = len(sx)
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n_dir, C - 1))
    axes = np.concatenate([np.eye(C - 1), -np.eye(C - 1),
                           np.ones((1, C - 1)), -np.ones((1, C - 1))])
    d = np.concatenate([d, axes])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    near = llr_alr(t * d, sx, nux, sr, nur).max()
    far = llr_alr(2 * t * d, sx, nux, sr, nur).max()
    return bool(far < near)


# --------------------------------------------------------------------------
# 5. method of moments
# --------------------------------------------------------------------------
def mom(S, variant="arithmetic"):
    """Estimate (s_bar, nu) from shadow softmax rows S of shape [M, C].

    For P ~ Dir(nu s), Var(P_k) = s_k (1 - s_k) / (nu + 1), so every coordinate
    is an estimate of nu + 1.  `arithmetic` averages them, `minka` averages
    their logarithms after subtracting one (the robust form in Minka 2000),
    and `paper` is the 1/(C-1) sum over all C coordinates as printed in the
    draft -- kept only to measure its bias.
    """
    M, C = S.shape
    sbar = S.mean(0)
    var = S.var(0, ddof=1)
    ratio = sbar * (1.0 - sbar) / np.clip(var, 1e-15, None)
    if variant == "arithmetic":
        nu = ratio.mean() - 1.0
    elif variant == "paper":
        nu = ratio.sum() / (C - 1) - 1.0
    elif variant == "minka":
        nu = np.exp(np.log(np.clip(ratio - 1.0, 1e-8, None)).mean())
    else:
        raise ValueError(variant)
    return sbar, max(float(nu), 1e-3)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def rand_side(rng, C, nu_lo=4.0, nu_hi=80.0):
    s = rng.dirichlet(np.full(C, 1.5))
    return s, float(rng.uniform(nu_lo, nu_hi))


def check_closed_form(rng, n_trial=200, n_pt=40):
    err_cf, err_ll = 0.0, 0.0
    for _ in range(n_trial):
        C = int(rng.integers(2, 9))
        sx, nux = rand_side(rng, C)
        sr, nur = rand_side(rng, C)
        P = rng.dirichlet(np.full(C, 1.0), size=n_pt)
        ref = llr_direct(P, sx, nux, sr, nur)
        err_cf = max(err_cf, np.abs(llr_closed_form(P, sx, nux, sr, nur)
                                    - ref).max())
        err_ll = max(err_ll, np.abs(llr_loglinear(P, sx, nux, sr, nur)
                                    - ref).max())
    return err_cf, err_ll


def check_affine(rng, n_trial=200, n_pt=200):
    """Residual of log Lambda - <beta, log P> must not vary over the simplex."""
    spread = 0.0
    for _ in range(n_trial):
        C = int(rng.integers(2, 9))
        sx, nux = rand_side(rng, C)
        sr, nur = rand_side(rng, C)
        P = rng.dirichlet(np.full(C, 0.7), size=n_pt)
        beta = to_alpha(sx, nux) - to_alpha(sr, nur)
        res = (llr_closed_form(P, sx, nux, sr, nur)
               - (beta * np.log(np.clip(P, EPS, None))).sum(-1))
        spread = max(spread, res.max() - res.min())
    return spread


def check_concavity(rng, n_trial=400):
    """Midpoint concavity in alr coordinates, both signs of dnu."""
    worst_ok, worst_bad = np.inf, -np.inf
    for _ in range(n_trial):
        C = int(rng.integers(2, 7))
        sx, nux = rand_side(rng, C)
        sr, nur = rand_side(rng, C)
        v = rng.normal(scale=3.0, size=(400, C - 1))
        w = rng.normal(scale=3.0, size=(400, C - 1))
        f = lambda z: llr_alr(z, sx, nux, sr, nur)
        gap = f(0.5 * (v + w)) - 0.5 * (f(v) + f(w))    # >= 0 iff concave
        if nux >= nur:
            worst_ok = min(worst_ok, gap.min())
        else:
            worst_bad = max(worst_bad, gap.min())
    return worst_ok, worst_bad


def check_boundedness(rng, n_trial=300):
    """alpha_x > alpha_{x,r} coordinatewise  <=>  member region bounded."""
    agree, total, table = 0, 0, {}
    for t in range(n_trial):
        C = int(rng.integers(2, 6))
        sx, nux = rand_side(rng, C)
        sr, nur = rand_side(rng, C)
        # bias the draw so both branches of the criterion occur often
        if t % 2 == 0:
            sx, nux = sr.copy(), nur * float(rng.uniform(1.3, 3.0))
            sx = sx * (1 + 0.15 * rng.normal(size=C))
            sx = np.clip(sx, 1e-3, None)
            sx /= sx.sum()
        crit = bool(np.all(to_alpha(sx, nux) > to_alpha(sr, nur)))
        obs = region_bounded_by_raymarch(sx, nux, sr, nur, seed=t)
        table[(crit, obs)] = table.get((crit, obs), 0) + 1
        agree += int(crit == obs)
        total += 1
    return agree, total, table


def check_mom(rng, n_rep=400, C=6, M=256, nu_true=40.0):
    s = rng.dirichlet(np.full(C, 2.0))
    out = {v: [] for v in ("arithmetic", "minka", "paper")}
    for _ in range(n_rep):
        S = rng.dirichlet(nu_true * s, size=M)
        for v in out:
            out[v].append(mom(S, v)[1])
    return {v: (float(np.mean(a)), float(np.std(a))) for v, a in out.items()}


def main():
    rng = np.random.default_rng(0)
    print("=" * 68)
    print("1-2. closed form and log-linearity vs scipy Dirichlet log-pdf")
    e_cf, e_ll = check_closed_form(rng)
    print(f"     max |KL form   - direct|  = {e_cf:.3e}")
    print(f"     max |log-linear- direct|  = {e_ll:.3e}")
    assert max(e_cf, e_ll) < 1e-8

    print("\n2b.  affine in log P: spread of the residual over the simplex")
    sp = check_affine(rng)
    print(f"     max spread                = {sp:.3e}")
    assert sp < 1e-8

    print("\n3.   concavity in alr coordinates")
    ok, bad = check_concavity(rng)
    print(f"     nu_x >= nu_xr : worst midpoint gap = {ok:+.3e}  (must be >= 0)")
    print(f"     nu_x <  nu_xr : best  midpoint gap = {bad:+.3e}  (must be <= 0)")
    assert ok >= -1e-9 and bad <= 1e-9

    print("\n4.   bounded member region <=> alpha_x > alpha_xr coordinatewise")
    agree, total, table = check_boundedness(rng)
    print(f"     agreement {agree}/{total}")
    for (crit, obs), n in sorted(table.items()):
        print(f"       criterion={str(crit):5s} ray-march={str(obs):5s}  n={n}")
    assert agree == total

    print("\n5.   method of moments for nu (C=6, M=256, true nu = 40)")
    for v, (m, s) in check_mom(rng).items():
        print(f"     {v:11s} mean {m:7.2f}  sd {s:5.2f}"
              f"   bias {m - 40.0:+7.2f}")
    print("=" * 68)
    print("all checks passed")


if __name__ == "__main__":
    main()
