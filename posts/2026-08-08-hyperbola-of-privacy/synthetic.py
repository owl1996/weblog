"""What the simplex buys over the scalar confidence, and what it costs.

Two experiments, both with Dirichlet ground truth so the *optimal* test is
known and the two attacks can be scored against it rather than against each
other.

  A. blind spot -- a pair of sides whose correct-class marginal is *identical*
     by construction (same Beta(a, nu - a)) but whose residual mass sits on
     different classes.  U-LiRA's statistic is a function of the correct-class
     probability alone, so its AUC is 0.5 exactly; the simplex test is not.
     This is the strict half of the data-processing inequality, made concrete.

  B. shadow budget -- a population of attack points where both signals are
     present, swept over the number of shadow models per side.  The simplex
     test estimates 2C parameters against U-LiRA's 4, so it should lose at
     small M and win at large M.  Where the crossover sits is the question.

Writes synthetic_results.npz.  Run: python synthetic.py   (about a minute)
"""
import numpy as np
from scipy.special import logsumexp

from llr import llr_closed_form, mom, to_alpha, to_mean_precision

EPS = 1e-12
N_REL = 4000          # releases per side, scenario A
N_PT = 500            # attack points, scenario B
N_REL_B = 60          # releases per side per attack point, scenario B
M_GRID = [8, 16, 32, 64, 128, 256, 512, 1024]
C_B = 10


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def auc(score_in, score_out):
    s = np.concatenate([score_in, score_out])
    y = np.concatenate([np.ones(len(score_in)), np.zeros(len(score_out))])
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, r)
    r = (sums / cnt)[inv]
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tpr_at_fpr(score_in, score_out, fpr=0.01):
    """Threshold set on the non-member scores; ties resolved conservatively."""
    tau = np.quantile(score_out, 1.0 - fpr, method="higher")
    return float((score_in > tau).mean())


# --------------------------------------------------------------------------
# the two attacks
# --------------------------------------------------------------------------
def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p) - np.log1p(-p)


def ulira_fit(S, y):
    """Gaussian on the logit-scaled correct-class probability [carlini2022]."""
    phi = logit(S[:, y])
    return float(phi.mean()), float(max(phi.std(ddof=1), 1e-6))


def ulira_score(P, y, fit_in, fit_out):
    mi, si = fit_in
    mo, so = fit_out
    phi = logit(P[:, y])
    return (-0.5 * ((phi - mi) / si) ** 2 - np.log(si)
            + 0.5 * ((phi - mo) / so) ** 2 + np.log(so))


def simplex_score(P, side_in, side_out):
    (sx, nux), (sr, nur) = side_in, side_out
    return llr_closed_form(P, sx, nux, sr, nur)


# --------------------------------------------------------------------------
# A. the blind spot
# --------------------------------------------------------------------------
#   correct class is index 0; both sides give P_0 ~ Beta(12, 8) exactly.
ALPHA_IN_A = np.array([12.0, 5.0, 2.0, 1.0])
ALPHA_OUT_A = np.array([12.0, 1.0, 2.0, 5.0])


def scenario_a(rng):
    Pin = rng.dirichlet(ALPHA_IN_A, size=N_REL)
    Pout = rng.dirichlet(ALPHA_OUT_A, size=N_REL)
    side_in, side_out = to_mean_precision(ALPHA_IN_A), to_mean_precision(ALPHA_OUT_A)

    fit_in, fit_out = ulira_fit(Pin, 0), ulira_fit(Pout, 0)
    u_in = ulira_score(Pin, 0, fit_in, fit_out)
    u_out = ulira_score(Pout, 0, fit_in, fit_out)
    d_in = simplex_score(Pin, side_in, side_out)
    d_out = simplex_score(Pout, side_in, side_out)
    return dict(
        Pin=Pin, Pout=Pout,
        phi_in=logit(Pin[:, 0]), phi_out=logit(Pout[:, 0]),
        d_in=d_in, d_out=d_out,
        auc_ulira=auc(u_in, u_out), auc_simplex=auc(d_in, d_out),
        tpr_ulira=tpr_at_fpr(u_in, u_out), tpr_simplex=tpr_at_fpr(d_in, d_out),
    )


# --------------------------------------------------------------------------
# B. shadow budget
# --------------------------------------------------------------------------
def draw_point(rng, C=C_B):
    """One attack point: a retained side and a seen side that differs both in
    correct-class confidence and in where the residual mass sits."""
    nu_r = float(np.exp(rng.uniform(np.log(12.0), np.log(150.0))))
    c_r = float(rng.beta(6.0, 3.0))                       # correct-class mean
    rest = rng.dirichlet(np.full(C - 1, 0.8))
    s_r = np.concatenate([[c_r], (1 - c_r) * rest])

    gain = float(rng.uniform(0.0, 0.6)) * (1 - c_r)       # memorisation gain
    c_x = c_r + gain
    conf = 1 + int(np.argmax(rest))                       # nearest competitor
    shift = float(rng.uniform(0.2, 0.9))                  # residual reshaping
    rest_x = rest.copy()
    moved = shift * rest_x[conf - 1]
    rest_x[conf - 1] -= moved
    rest_x += moved / (C - 1)
    rest_x /= rest_x.sum()
    s_x = np.concatenate([[c_x], (1 - c_x) * rest_x])

    nu_x = nu_r * float(rng.uniform(1.0, 2.2))            # seen side sharper
    return (s_x, nu_x), (s_r, nu_r)


def scenario_b(rng):
    pts = [draw_point(rng) for _ in range(N_PT)]
    Mmax = max(M_GRID)
    res = {k: [] for k in ("ulira", "simplex", "oracle")}
    boundedness = []

    # per attack point: one pool of shadows per side, nested by M
    pooled = {m: {k: ([], []) for k in res} for m in M_GRID}
    for (side_x, side_r) in pts:
        ax, ar = to_alpha(*side_x), to_alpha(*side_r)
        boundedness.append(bool(np.all(ax > ar)))
        Sx = rng.dirichlet(ax, size=Mmax)
        Sr = rng.dirichlet(ar, size=Mmax)
        Pin = rng.dirichlet(ax, size=N_REL_B)
        Pout = rng.dirichlet(ar, size=N_REL_B)

        o_in = simplex_score(Pin, side_x, side_r)
        o_out = simplex_score(Pout, side_x, side_r)
        for M in M_GRID:
            fi, fo = ulira_fit(Sx[:M], 0), ulira_fit(Sr[:M], 0)
            u_in = ulira_score(Pin, 0, fi, fo)
            u_out = ulira_score(Pout, 0, fi, fo)
            hx, hr = mom(Sx[:M], "minka"), mom(Sr[:M], "minka")
            d_in = simplex_score(Pin, hx, hr)
            d_out = simplex_score(Pout, hx, hr)
            for k, (a, b) in (("ulira", (u_in, u_out)),
                              ("simplex", (d_in, d_out)),
                              ("oracle", (o_in, o_out))):
                pooled[M][k][0].append(a)
                pooled[M][k][1].append(b)

    out = {}
    for k in res:
        out[f"auc_{k}"] = np.array(
            [auc(np.concatenate(pooled[M][k][0]),
                 np.concatenate(pooled[M][k][1])) for M in M_GRID])
        out[f"tpr_{k}"] = np.array(
            [tpr_at_fpr(np.concatenate(pooled[M][k][0]),
                        np.concatenate(pooled[M][k][1])) for M in M_GRID])
    out["M"] = np.array(M_GRID)
    out["frac_bounded"] = float(np.mean(boundedness))
    return out


def main():
    rng = np.random.default_rng(1)
    a = scenario_a(rng)
    print("A. blind spot   (C=4, identical Beta(12,8) marginal on the "
          "correct class)")
    print(f"   U-LiRA   AUC {a['auc_ulira']:.4f}   TPR@1%FPR "
          f"{a['tpr_ulira']:.4f}")
    print(f"   simplex  AUC {a['auc_simplex']:.4f}   TPR@1%FPR "
          f"{a['tpr_simplex']:.4f}")

    b = scenario_b(rng)
    print(f"\nB. shadow budget  (C={C_B}, {N_PT} attack points, "
          f"{N_REL_B} releases per side per point)")
    print(f"   fraction of points with a bounded member region: "
          f"{b['frac_bounded']:.3f}")
    print(f"   {'M':>6} {'AUC uL':>8} {'AUC smx':>8} {'AUC orc':>8}"
          f" {'TPR uL':>8} {'TPR smx':>8} {'TPR orc':>8}")
    for i, M in enumerate(b["M"]):
        print(f"   {M:6d} {b['auc_ulira'][i]:8.4f} {b['auc_simplex'][i]:8.4f}"
              f" {b['auc_oracle'][i]:8.4f} {b['tpr_ulira'][i]:8.4f}"
              f" {b['tpr_simplex'][i]:8.4f} {b['tpr_oracle'][i]:8.4f}")

    np.savez_compressed("synthetic_results.npz",
                        **{f"a_{k}": v for k, v in a.items()},
                        **{f"b_{k}": v for k, v in b.items()},
                        alpha_in_a=ALPHA_IN_A, alpha_out_a=ALPHA_OUT_A)
    print("\nwrote synthetic_results.npz")


if __name__ == "__main__":
    main()
