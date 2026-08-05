"""The correlated focus gate: orthant probability, Plackett derivative, bounds.

Everything the post claims analytically is checked here against Monte-Carlo or
against an independent implementation:

  1. the generative claim  Cov(N1, N2) = (kappa/B) Cov_s(d1, d2)
  2. the orthant formula   f = Phi2(a,b;rho) + Phi2(-a,-b;rho)   vs sampling
  3. Owen's-T evaluation   vs scipy's bivariate normal CDF
  4. Plackett             d f / d rho = 2 phi2(a,b;rho) > 0
  5. Lemma 1 (uniform)    |f^rho - f^0| <= arcsin|rho| / pi, tight at a=b=0
  6. Lemma 2 (local)      d f / d rho at rho=0  =  2 phi(a) phi(b)
  7. the degeneracy table  rho=0, rho->1, C->0, Sheppard

Run: python orthant.py   (~20 s)
"""
import numpy as np
from scipy.special import ndtr, owens_t
from scipy.stats import norm, multivariate_normal

SQRT2 = np.sqrt(2.0)
EPS = 1e-9                      # keeps a, b off the Owen's-T singularity at 0


def phi(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / np.sqrt(2 * np.pi)


def phi2(a, b, rho):
    """Standard bivariate normal density."""
    a, b, rho = map(np.asarray, (a, b, rho))
    q = 1.0 - rho ** 2
    z = (a ** 2 - 2 * rho * a * b + b ** 2) / q
    return np.exp(-0.5 * z) / (2 * np.pi * np.sqrt(q))


def _off_zero(x):
    """Replace 0 by +EPS, keeping the sign elsewhere; error on f is O(EPS)."""
    x = np.asarray(x, float)
    return np.where(np.abs(x) < EPS, np.where(x < 0, -EPS, EPS), x)


def focus(a, b, rho):
    """P[g1 g2 > 0 | ghat] = Phi2(a,b;rho) + Phi2(-a,-b;rho), via Owen's T.

    Using Phi2(-a,-b;rho) = 1 - Phi(a) - Phi(b) + Phi2(a,b;rho) and Owen's
    decomposition of Phi2, the marginals cancel and only two T calls remain:

        f = 1{ab > 0} - 2 T(a, (b - rho a) / (a sqrt(1-rho^2)))
                      - 2 T(b, (a - rho b) / (b sqrt(1-rho^2))).

    Vectorised over per-coordinate rho.
    """
    a, b = _off_zero(a), _off_zero(b)
    rho = np.clip(np.asarray(rho, float), -1 + 1e-12, 1 - 1e-12)
    r = np.sqrt(1.0 - rho ** 2)
    ta = owens_t(a, (b - rho * a) / (a * r))
    tb = owens_t(b, (a - rho * b) / (b * r))
    return np.where(a * b > 0, 1.0, 0.0) - 2 * ta - 2 * tb


def focus_indep(a, b):
    """The rho = 0 gate of the Adam2D post: product of two soft signs."""
    p1, p2 = 2 * ndtr(a) - 1, 2 * ndtr(b) - 1
    return 0.5 * (1 + p1 * p2)


# ----------------------------------------------------------------- checks ---

def check_generative(rng, B=64, n_draw=20000, kappa=0.5):
    """Cov(N1,N2) = (kappa/B) Cov_s(d1,d2) for a batch sharing a fraction kappa.

    Per-sample derivatives of the two losses on a single coordinate are drawn
    jointly; task 1 uses samples 0..B-1 of a fresh pool, task 2 reuses the first
    kappa*B of them and draws the rest independently.
    """
    Cs = np.array([[1.0, 0.6], [0.6, 2.0]])                  # per-sample cov
    L = np.linalg.cholesky(Cs)
    nshare = int(round(kappa * B))
    d1m, d2m = np.empty(n_draw), np.empty(n_draw)
    for t in range(n_draw):
        s = (rng.standard_normal((B, 2)) @ L.T)              # shared pool
        extra = (rng.standard_normal((B - nshare, 2)) @ L.T)  # task-2 remainder
        d1m[t] = s[:, 0].mean()
        d2m[t] = np.concatenate([s[:nshare, 1], extra[:, 1]]).mean()
    emp = np.cov(d1m, d2m)[0, 1]
    pred = kappa / B * Cs[0, 1]
    return emp, pred, np.corrcoef(d1m, d2m)[0, 1]


def check_orthant_mc(rng, n=400000):
    """The formula against direct sampling of the posterior."""
    worst = 0.0
    for (a, b, rho) in [(0.3, -0.8, 0.5), (1.5, 2.0, -0.4), (0.0, 0.0, 0.7),
                        (-2.0, 0.1, 0.9), (0.7, 0.7, -0.85), (3.0, -0.2, 0.2)]:
        C = np.array([[1.0, rho], [rho, 1.0]])
        z = rng.multivariate_normal([a, b], C, size=n)
        mc = np.mean(z[:, 0] * z[:, 1] > 0)
        worst = max(worst, abs(mc - focus(a, b, rho)))
    return worst


def check_owens_vs_scipy(rng, n=300):
    """Owen's-T route against scipy's bivariate normal CDF, term by term."""
    a, b = rng.uniform(-3, 3, n), rng.uniform(-3, 3, n)
    rho = rng.uniform(-0.95, 0.95, n)
    ref = np.array([
        multivariate_normal.cdf([ai, bi], mean=[0, 0],
                                cov=[[1, ri], [ri, 1]]) +
        multivariate_normal.cdf([-ai, -bi], mean=[0, 0],
                                cov=[[1, ri], [ri, 1]])
        for ai, bi, ri in zip(a, b, rho)])
    return np.abs(ref - focus(a, b, rho)).max()


def check_plackett(rng, n=2000, h=1e-5):
    """d f / d rho = 2 phi2(a,b;rho), and the sign is positive everywhere.

    The analytic derivative is positive by inspection (a density); the numeric
    one is only informative where it is above float64 cancellation noise, so the
    sign is reported on the points where 2 phi2 > 1e-10.
    """
    a, b = rng.uniform(-3.5, 3.5, n), rng.uniform(-3.5, 3.5, n)
    rho = rng.uniform(-0.9, 0.9, n)
    num = (focus(a, b, rho + h) - focus(a, b, rho - h)) / (2 * h)
    ana = 2 * phi2(a, b, rho)
    ok = ana > 1e-10
    return np.abs(num - ana).max(), num[ok].min(), ana.min(), ok.mean()


def check_lemma1(nrho=19, ngrid=241):
    """sup_{a,b} |f^rho - f^0| against arcsin|rho|/pi, and where it is attained."""
    g = np.linspace(-4, 4, ngrid)
    A, Bq = np.meshgrid(g, g)
    rows = []
    for rho in np.linspace(-0.9, 0.9, nrho):
        dev = np.abs(focus(A, Bq, rho) - focus_indep(A, Bq))
        k = np.unravel_index(dev.argmax(), dev.shape)
        rows.append((rho, dev.max(), np.arcsin(abs(rho)) / np.pi,
                     A[k], Bq[k]))
    return rows


def check_lemma2(rng, n=2000, h=1e-4):
    """d f / d rho at rho = 0 equals 2 phi(a) phi(b)."""
    a, b = rng.uniform(-4, 4, n), rng.uniform(-4, 4, n)
    num = (focus(a, b, h) - focus(a, b, -h)) / (2 * h)
    return np.abs(num - 2 * phi(a) * phi(b)).max()


def check_degeneracies(rng, n=4000):
    a, b = rng.uniform(-3, 3, n), rng.uniform(-3, 3, n)
    out = {}
    out["rho = 0 -> product of soft signs"] = np.abs(
        focus(a, b, 0.0) - focus_indep(a, b)).max()
    out["rho -> 1 -> Phi(min) + 1 - Phi(max)"] = np.abs(
        focus(a, b, 1 - 1e-9)
        - (norm.cdf(np.minimum(a, b)) + 1 - norm.cdf(np.maximum(a, b)))).max()
    out["rho -> -1 -> |Phi(b) - Phi(-a)|"] = np.abs(
        focus(a, b, -1 + 1e-9)
        - np.abs(norm.cdf(b) - norm.cdf(-a))).max()
    # C -> 0 : the hard AND-mask on measured gradients (pointwise, for a b != 0)
    for s in (1e-4, 1e-6):
        out[f"C -> 0 (sigma = {s:.0e}) -> hard AND mask"] = np.abs(
            focus(a / s, b / s, 0.3) - (a * b > 0)).max()
    # Sheppard at a = b = 0
    r = rng.uniform(-0.95, 0.95, 200)
    out["a = b = 0 -> 1/2 + arcsin(rho)/pi"] = np.abs(
        focus(np.zeros_like(r), np.zeros_like(r), r)
        - (0.5 + np.arcsin(r) / np.pi)).max()
    return out


def check_observed_conflict(rng, n_draw=200000):
    """E[ghat1 ghat2] = g1 g2 + c12: negative c12 fabricates observed conflict.

    A coordinate on which the two true gradients agree (g1, g2 > 0) but whose
    sampling noise is negatively correlated is measured in disagreement more
    often than chance.
    """
    g1, g2, sd = 0.4, 0.3, 1.0
    rows = []
    for rho in (-0.8, -0.4, 0.0, 0.4, 0.8):
        C = sd ** 2 * np.array([[1.0, rho], [rho, 1.0]])
        z = rng.multivariate_normal([g1, g2], C, size=n_draw)
        rows.append((rho, np.mean(z[:, 0] * z[:, 1] < 0),
                     np.mean(z[:, 0] * z[:, 1]), g1 * g2 + rho * sd ** 2))
    return rows


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("1. generative model, Cov(N1,N2) = (kappa/B) Cov_s")
    for kap in (0.25, 0.5, 1.0):
        emp, pred, r = check_generative(rng, kappa=kap)
        print(f"   kappa = {kap:4.2f}   empirical {emp:+.5f}   "
              f"predicted {pred:+.5f}   rho = {r:+.3f}")

    print("\n2. orthant formula vs Monte-Carlo posterior")
    print(f"   worst |f_formula - f_MC| over 6 points, 4e5 draws: "
          f"{check_orthant_mc(rng):.2e}")

    print("\n3. Owen's T vs scipy bivariate CDF")
    print(f"   worst absolute discrepancy over 300 random (a,b,rho): "
          f"{check_owens_vs_scipy(rng):.2e}")

    print("\n4. Plackett: d f / d rho = 2 phi2(a,b;rho)")
    err, mn, mn_ana, frac = check_plackett(rng)
    print(f"   worst |numeric - analytic|: {err:.2e}")
    print(f"   min analytic derivative (all 2000 points):      {mn_ana:.3e}")
    print(f"   min numeric derivative where analytic > 1e-10 "
          f"({frac:.0%} of points): {mn:.3e}  "
          f"({'positive everywhere' if mn > 0 else 'SIGN VIOLATION'})")

    print("\n5. Lemma 1: sup_{a,b} |f^rho - f^0| <= arcsin|rho|/pi")
    print(f"   {'rho':>6s} {'sup deviation':>14s} {'bound':>9s} "
          f"{'slack':>10s} {'argmax (a,b)':>18s}")
    for rho, dev, bd, aa, bb in check_lemma1():
        print(f"   {rho:+6.2f} {dev:14.5f} {bd:9.5f} {bd - dev:+10.2e}"
              f"   ({aa:+.2f}, {bb:+.2f})")

    print("\n6. Lemma 2: d f / d rho at rho = 0  =  2 phi(a) phi(b)")
    print(f"   worst |numeric - analytic|: {check_lemma2(rng):.2e}")

    print("\n7. degeneracy table")
    for k, v in check_degeneracies(rng).items():
        print(f"   {k:38s} worst error {v:.2e}")

    print("\n8. negative noise correlation fabricates observed conflict")
    print("   true g1 g2 = +0.12 > 0 (the tasks agree)")
    print(f"   {'rho':>6s} {'P[ghat1 ghat2 < 0]':>20s} "
          f"{'E[ghat1 ghat2]':>16s} {'g1 g2 + c12':>13s}")
    for rho, pneg, emp, pred in check_observed_conflict(rng):
        print(f"   {rho:+6.2f} {pneg:20.4f} {emp:+16.4f} {pred:+13.4f}")
