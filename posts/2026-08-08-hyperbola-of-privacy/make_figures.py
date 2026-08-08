"""Figures for the hyperbola-of-privacy post.

  hyperbola-{light,dark}.png  the decision boundary on the 2-simplex, in the
                              bounded and unbounded regimes, and the same
                              region drawn in alr coordinates
  blindspot-{light,dark}.svg  the constructed pair on which the correct-class
                              statistic is exactly uninformative
  budget-{light,dark}.svg     AUC and TPR at 1% FPR against shadow count
  real-{light,dark}.png       a real shadow cloud, model fit, and the ladder
                              of attacks on digits

The first reads nothing; the others read synthetic_results.npz and
shadow_results.npz -- run `python synthetic.py` and `python shadows.py` first.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from llr import llr_alr, llr_closed_form, to_mean_precision

TOK = {"light": dict(s1="#2a78d6", s2="#eb6834", s3="#5a9e5f", muted="#898781",
                     grid="#e1e0d9", axis="#c3c2b7", ink="#0b0b0b",
                     ink2="#52514e"),
       "dark": dict(s1="#3987e5", s2="#d95926", s3="#4e9b56", muted="#898781",
                    grid="#2c2c2a", axis="#383835", ink="#ffffff",
                    ink2="#c3c2b7")}
mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica Neue", "Helvetica",
                                         "Arial", "DejaVu Sans"],
                     "font.size": 10.5, "svg.fonttype": "none",
                     "figure.dpi": 110})
SQ3 = np.sqrt(3.0) / 2.0


def style(ax, t, spines=("left", "bottom")):
    ax.set_facecolor("none")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in spines)
        if s in spines:
            ax.spines[s].set_color(t["axis"])
            ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=t["muted"], labelsize=9, length=3)
    for l in ax.get_xticklabels() + ax.get_yticklabels():
        l.set_color(t["muted"])
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])


# --------------------------------------------------------------- ternary ---
def bary(P):
    P = np.atleast_2d(P)
    return P[:, 1] + 0.5 * P[:, 2], SQ3 * P[:, 2]


def tri_grid(n=240, eps=2e-4):
    a, b = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    a, b = a.ravel(), b.ravel()
    keep = a + b <= 1.0
    P = np.stack([1 - a[keep] - b[keep], a[keep], b[keep]], 1)
    P = np.clip(P, eps, None)
    return P / P.sum(1, keepdims=True)


def frame(ax, t, labels):
    v = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, SQ3], [0.0, 0.0]])
    ax.plot(v[:, 0], v[:, 1], color=t["axis"], lw=1.1, zorder=5)
    for (x, y), lab, ha, va in zip(v[:3], labels,
                                   ("right", "left", "center"),
                                   ("top", "top", "bottom")):
        ax.text(x + (0.02 if ha == "left" else -0.02 if ha == "right" else 0),
                y + (0.028 if va == "bottom" else -0.03), lab, ha=ha, va=va,
                fontsize=9.5, color=t["ink2"])
    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(-0.10, SQ3 + 0.10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    style(ax, t, spines=())


def ternary_panel(ax, t, ax_par, ar_par, title, rng, n_pt=260, levels=9):
    """Contours of log Lambda on the 2-simplex, plus shadow clouds."""
    sx, nux = to_mean_precision(np.asarray(ax_par, float))
    sr, nur = to_mean_precision(np.asarray(ar_par, float))
    P = tri_grid()
    L = llr_closed_form(P, sx, nux, sr, nur)
    x, y = bary(P)
    tri = mtri.Triangulation(x, y)

    lv = np.append(np.sort(L.max() - np.geomspace(0.4, 40.0, levels - 1)),
                   L.max())
    ax.tricontourf(tri, L, levels=lv, cmap=ramp(t["s1"]), zorder=1,
                   extend="min")
    ax.tricontour(tri, L, levels=lv[1:-1], colors=t["axis"], linewidths=0.5,
                  zorder=2)
    ax.tricontour(tri, L, levels=[0.0], colors=[t["s2"]], linewidths=2.0,
                  zorder=4)

    for a, col, mk, lab in ((ax_par, t["s1"], "o", r"$\bar{s}_x$"),
                            (ar_par, t["ink2"], "s", r"$\bar{s}_{x,r}$")):
        S = rng.dirichlet(np.asarray(a, float), size=n_pt)
        px, py = bary(S)
        ax.scatter(px, py, s=5, color=col, alpha=0.32, lw=0, zorder=3)
        mx, my = bary(np.asarray(a, float) / np.sum(a))
        ax.scatter(mx, my, s=64, marker=mk, facecolor=col,
                   edgecolor=t["ink"], lw=0.9, zorder=6, label=lab)
    frame(ax, t, (r"$P_1$", r"$P_2$", r"$P_3$"))
    ax.legend(frameon=False, fontsize=9, labelcolor=t["ink2"],
              loc="upper right", handletextpad=0.2, borderpad=0.0)
    ax.set_title(title, fontsize=10, color=t["ink2"], pad=6)


def ramp(hue):
    r, g, b = mpl.colors.to_rgb(hue)
    return mpl.colors.LinearSegmentedColormap.from_list(
        "seq", [(r, g, b, 0.0), (r, g, b, 0.30), (r, g, b, 0.85)])


# ------------------------------------------------------------- figure 1 ---
A_BOUNDED = (10.0, 3.2, 2.4)
R_BOUNDED = (4.4, 2.6, 1.6)           # beta = (5.6, 0.6, 0.8) > 0 : an island
A_OPEN = (9.0, 2.2, 4.4)
R_OPEN = (4.6, 4.6, 1.5)              # beta_2 < 0 : a branch escapes


def fig_hyperbola(mode):
    t = TOK[mode]
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.5))
    fig.patch.set_alpha(0.0)

    ternary_panel(axes[0], t, A_BOUNDED, R_BOUNDED,
                  r"$\alpha_x \succ \alpha_{x,r}$: a closed island", rng)
    ternary_panel(axes[1], t, A_OPEN, R_OPEN,
                  r"one $\beta_k < 0$: a branch escapes", rng)

    # panel C: the bounded case in alr coordinates
    ax = axes[2]
    sx, nux = to_mean_precision(np.array(A_BOUNDED))
    sr, nur = to_mean_precision(np.array(R_BOUNDED))
    c = np.log(sx[:2]) - np.log(sx[2])
    g1 = np.linspace(c[0] - 5.5, c[0] + 5.5, 420)
    g2 = np.linspace(c[1] - 5.5, c[1] + 5.5, 420)
    V1, V2 = np.meshgrid(g1, g2)
    L = llr_alr(np.stack([V1.ravel(), V2.ravel()], 1),
                sx, nux, sr, nur).reshape(V1.shape)
    lv = np.append(np.sort(L.max() - np.geomspace(0.4, 40.0, 8)),
                   L.max())
    ax.contourf(V1, V2, L, levels=lv, cmap=ramp(t["s1"]), zorder=1,
                extend="min")
    ax.contour(V1, V2, L, levels=lv[1:-1], colors=t["axis"], linewidths=0.5,
               zorder=2)
    ax.contour(V1, V2, L, levels=[0.0], colors=[t["s2"]], linewidths=2.0,
               zorder=3)
    S = rng.dirichlet(np.array(A_BOUNDED), size=300)
    v = np.log(S[:, :2]) - np.log(S[:, 2:])
    ax.scatter(v[:, 0], v[:, 1], s=5, color=t["s1"], alpha=0.3, lw=0, zorder=4)
    S = rng.dirichlet(np.array(R_BOUNDED), size=300)
    v = np.log(S[:, :2]) - np.log(S[:, 2:])
    ax.scatter(v[:, 0], v[:, 1], s=5, color=t["ink2"], alpha=0.3, lw=0,
               zorder=4)
    style(ax, t)
    ax.grid(color=t["grid"], lw=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"$v_1 = \log(P_1/P_3)$")
    ax.set_ylabel(r"$v_2 = \log(P_2/P_3)$")
    ax.set_title("the same region, alr coordinates", fontsize=10,
                 color=t["ink2"], pad=6)
    fig.tight_layout()
    fig.savefig(f"hyperbola-{mode}.png", transparent=True,
                bbox_inches="tight", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------- figure 2 ---
def roc(s_in, s_out, n=400):
    thr = np.quantile(np.concatenate([s_in, s_out]),
                      np.linspace(0, 1, n)[::-1])
    return ((s_out[None] > thr[:, None]).mean(1),
            (s_in[None] > thr[:, None]).mean(1))


def fig_blindspot(d, mode):
    t = TOK[mode]
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2))
    fig.patch.set_alpha(0.0)
    ain, aout = np.asarray(d["alpha_in_a"]), np.asarray(d["alpha_out_a"])

    ax = axes[0]
    style(ax, t)
    bins = np.linspace(-2.2, 4.2, 46)
    for arr, col, lab in ((d["a_phi_in"], t["s1"], "saw $z$"),
                          (d["a_phi_out"], t["ink2"], "did not")):
        ax.hist(arr, bins=bins, color=col, alpha=0.45, lw=0, density=True,
                label=lab)
    ax.legend(frameon=False, fontsize=9, labelcolor=t["ink2"], loc="upper left")
    ax.set_xlabel(r"U-LiRA statistic  $\phi(P_1)$")
    ax.set_ylabel("density")
    ax.set_title(f"AUC {d['a_auc_ulira']:.3f}", fontsize=10, color=t["s2"],
                 pad=6)
    ax.grid(color=t["grid"], lw=0.7)
    ax.set_axisbelow(True)

    ax = axes[1]
    style(ax, t)
    lo = min(d["a_d_in"].min(), d["a_d_out"].min())
    hi = max(d["a_d_in"].max(), d["a_d_out"].max())
    bins = np.linspace(lo, hi, 46)
    for arr, col in ((d["a_d_in"], t["s1"]), (d["a_d_out"], t["ink2"])):
        ax.hist(arr, bins=bins, color=col, alpha=0.45, lw=0, density=True)
    ax.axvline(0.0, color=t["s2"], lw=1.6)
    ax.set_xlabel(r"simplex statistic  $\log\Lambda_x(\mathbf{P})$")
    ax.set_title(f"AUC {d['a_auc_simplex']:.3f}", fontsize=10, color=t["s2"],
                 pad=6)
    ax.grid(color=t["grid"], lw=0.7)
    ax.set_axisbelow(True)

    ax = axes[2]
    style(ax, t)
    for key, col, lab in ((("a_phi_in", "a_phi_out"), t["ink2"], "U-LiRA"),
                          (("a_d_in", "a_d_out"), t["s1"], "simplex")):
        f, tp = roc(d[key[0]], d[key[1]])
        ax.plot(np.clip(f, 2e-4, 1), np.clip(tp, 2e-4, 1), lw=2.0, color=col,
                label=lab, solid_capstyle="round")
    ax.plot([2e-4, 1], [2e-4, 1], lw=1.0, ls=(0, (4, 3)), color=t["axis"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(1e-3, 1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(frameon=False, fontsize=9, labelcolor=t["ink2"], loc="upper left")
    ax.set_title(rf"$\alpha_x={tuple(int(v) for v in ain)}$, "
                 rf"$\alpha_{{x,r}}={tuple(int(v) for v in aout)}$",
                 fontsize=9.5, color=t["ink2"], pad=6)
    ax.grid(color=t["grid"], lw=0.7, which="both")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"blindspot-{mode}.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- figure 3 ---
def fig_budget(d, mode):
    t = TOK[mode]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    fig.patch.set_alpha(0.0)
    M = d["b_M"]
    series = (("b_auc_ulira", "b_tpr_ulira", t["ink2"], "U-LiRA (1-D)", "s"),
              ("b_auc_simplex", "b_tpr_simplex", t["s1"], "simplex", "o"))
    for ax, j, name in ((axes[0], 0, "AUC"),
                        (axes[1], 1, "TPR at 1% FPR")):
        style(ax, t)
        orc = d["b_auc_oracle" if j == 0 else "b_tpr_oracle"][0]
        ax.axhline(orc, color=t["s2"], lw=1.3, ls=(0, (4, 3)))
        ax.text(M[0], orc, "  simplex, known parameters", fontsize=8.8,
                color=t["s2"], va="bottom", ha="left")
        for ka, kt, col, lab, mk in series:
            ax.plot(M, d[ka if j == 0 else kt], lw=2.0, color=col, marker=mk,
                    ms=5.5, label=lab, solid_capstyle="round", zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(M)
        ax.set_xticklabels([str(int(m)) for m in M])
        ax.set_xlabel("shadow models per side  $M$")
        ax.set_ylabel(name)
        ax.grid(color=t["grid"], lw=0.7)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=t["ink2"],
                   loc="lower right")
    fig.tight_layout()
    fig.savefig(f"budget-{mode}.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- figure 4 ---
LADDER = (("ulira", "U-LiRA\n1-D"), ("dir3", "Dirichlet\n3-simplex"),
          ("ln3", "log-normal\n3-simplex"), ("dir10", "Dirichlet\n10-simplex"),
          ("ln10", "log-normal\n10-simplex"))


def logit_(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p) - np.log1p(-p)


def ulira_moments(pcol):
    phi = logit_(pcol)
    return float(phi.mean()), float(max(phi.std(ddof=1), 1e-6))


def fig_real(d, mode):
    t = TOK[mode]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5),
                             gridspec_kw=dict(width_ratios=[1.0, 1.05, 1.25]))
    fig.patch.set_alpha(0.0)

    # a real cloud in alr coordinates -- softmax clouds pile into a corner of
    # the simplex and are unreadable there, which is itself the argument for
    # working in log-ratios
    ax = axes[0]
    style(ax, t)
    sx, nux = np.asarray(d["cloud_s3x"]), float(d["cloud_n3x"])
    sr, nur = np.asarray(d["cloud_s3r"]), float(d["cloud_n3r"])
    Ax, Ar = np.asarray(d["cloud_Ax"]), np.asarray(d["cloud_Ar"])
    V = np.concatenate([np.log(Ax[:, :2]) - np.log(Ax[:, 2:]),
                        np.log(Ar[:, :2]) - np.log(Ar[:, 2:])])
    pad = 1.2
    g1 = np.linspace(V[:, 0].min() - pad, V[:, 0].max() + pad, 400)
    g2 = np.linspace(V[:, 1].min() - pad, V[:, 1].max() + pad, 400)
    G1, G2 = np.meshgrid(g1, g2)
    Pg = np.stack([np.exp(G1.ravel()), np.exp(G2.ravel()),
                   np.ones(G1.size)], 1)
    Pg /= Pg.sum(1, keepdims=True)
    L = llr_closed_form(Pg, sx, nux, sr, nur).reshape(G1.shape)
    ax.contour(G1, G2, L, levels=[-40, -20, -10, -5, 5, 10, 20],
               colors=[t["axis"]], linewidths=0.7, zorder=1)
    ax.contour(G1, G2, L, levels=[0.0], colors=[t["s2"]], linewidths=2.0,
               zorder=4)
    # what U-LiRA can draw: a level set of P_y alone
    mi, si = ulira_moments(Ax[:, 0])
    mo, so = ulira_moments(Ar[:, 0])
    U = (-0.5 * ((logit_(Pg[:, 0]) - mi) / si) ** 2 - np.log(si)
         + 0.5 * ((logit_(Pg[:, 0]) - mo) / so) ** 2
         + np.log(so)).reshape(G1.shape)
    ax.contour(G1, G2, U, levels=[0.0], colors=[t["ink2"]], linewidths=1.6,
               linestyles=[(0, (5, 3))], zorder=4)
    for A, col, lab, off in ((Ax, t["s1"], "saw $z$", (10, 16)),
                             (Ar, t["ink2"], "did not", (-14, 30))):
        v = np.log(A[:, :2]) - np.log(A[:, 2:])
        ax.scatter(v[:, 0], v[:, 1], s=7, color=col, alpha=0.5, lw=0, zorder=3)
        ax.annotate(lab, np.median(v, 0), textcoords="offset points",
                    xytext=off, fontsize=9.5, color=col, zorder=6)
    ax.text(0.97, 0.03, r"$\log\Lambda_x = 0$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color=t["s2"])
    ax.text(0.97, 0.11, "U-LiRA boundary", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color=t["ink2"])
    ax.set_xlim(g1[0], g1[-1])
    ax.set_ylim(g2[0], g2[-1])
    ax.grid(color=t["grid"], lw=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"$\log(P_y / P_{\mathrm{rest}})$")
    ax.set_ylabel(r"$\log(P_{\mathrm{comp}} / P_{\mathrm{rest}})$")
    ax.set_title(f"digits, attack point {int(d['cloud_index'])}",
                 fontsize=10, color=t["ink2"], pad=6)

    # held-out log-density, Dirichlet against logistic-normal
    ax = axes[1]
    style(ax, t)
    lo = min(d["dir_r"].min(), d["lnf_r"].min())
    hi = max(d["dir_r"].max(), d["lnf_r"].max())
    ax.plot([lo, hi], [lo, hi], lw=1.0, ls=(0, (4, 3)), color=t["axis"],
            zorder=1)
    ax.scatter(d["dir_r"], d["lnf_r"], s=13, color=t["s1"], alpha=0.55, lw=0,
               zorder=3, label="did not see $z$")
    ax.scatter(d["dir_x"], d["lnf_x"], s=13, color=t["s2"], alpha=0.55, lw=0,
               zorder=3, label="saw $z$")
    ax.set_xlabel("Dirichlet, held-out log-density")
    ax.set_ylabel("logistic-normal")
    ax.legend(frameon=False, fontsize=9, labelcolor=t["ink2"],
              loc="upper left")
    ax.grid(color=t["grid"], lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title("above the diagonal: the Dirichlet loses", fontsize=9.5,
                 color=t["ink2"], pad=6)

    # the ladder of attacks
    ax = axes[2]
    style(ax, t)
    pos = np.arange(len(LADDER))
    w = 0.38
    for off, key, col, lab in ((-w / 2, "auc", t["s1"], "AUC"),
                               (w / 2, "tpr1", t["s2"], "TPR at 1% FPR")):
        vals = np.array([float(d[f"{key}_{k}"]) for k, _ in LADDER])
        ci = np.array([d[f"{key}_ci_{k}"] for k, _ in LADDER])
        ax.bar(pos + off, vals, width=w, color=col, alpha=0.8, lw=0, label=lab)
        ax.errorbar(pos + off, vals,
                    yerr=np.abs(ci.T - vals), fmt="none", ecolor=t["ink2"],
                    elinewidth=1.1, capsize=2.5, zorder=4)
        for p, v, hi in zip(pos + off, vals, ci[:, 1]):
            ax.text(p, hi + 0.015, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8.2, color=t["ink2"], rotation=90)
    ax.axhline(0.5, color=t["axis"], lw=1.0)
    ax.set_xticks(pos)
    ax.set_xticklabels([lab for _, lab in LADDER], fontsize=8.4)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=9, labelcolor=t["ink2"], ncol=2,
              loc="lower left", bbox_to_anchor=(0.0, 1.0, 1.0, 0.1),
              mode="expand", borderaxespad=0.0)
    ax.grid(color=t["grid"], lw=0.7, axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"real-{mode}.png", transparent=True,
                bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    for mode in ("light", "dark"):
        fig_hyperbola(mode)
    syn = np.load("synthetic_results.npz")
    for mode in ("light", "dark"):
        fig_blindspot(syn, mode)
        fig_budget(syn, mode)
    try:
        real = np.load("shadow_results.npz")
    except FileNotFoundError:
        print("shadow_results.npz missing -- run shadows.py for figure 4")
        return
    for mode in ("light", "dark"):
        fig_real(real, mode)
    print("wrote hyperbola/real -{light,dark}.png and "
          "blindspot/budget -{light,dark}.svg")


if __name__ == "__main__":
    main()
