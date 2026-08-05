"""Figures for the correlated-noise post.

  correlation-{light,dark}.svg  where the noise correlation sits, against where
                                the gate is sensitive to it (Lemma 2)
  reliability-{light,dark}.svg  f as a forecast of true-gradient sign agreement

Both read measure_results.npz -- run `python measure.py` first.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

TOK = {"light": dict(s1="#2a78d6", s2="#eb6834", muted="#898781",
                     grid="#e1e0d9", axis="#c3c2b7", ink="#0b0b0b",
                     ink2="#52514e"),
       "dark": dict(s1="#3987e5", s2="#d95926", muted="#898781",
                    grid="#2c2c2a", axis="#383835", ink="#ffffff",
                    ink2="#c3c2b7")}
mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica Neue", "Helvetica",
                                         "Arial", "DejaVu Sans"],
                     "font.size": 10.5, "svg.fonttype": "none",
                     "figure.dpi": 110})
PROBS = [("retain", "retention: CE + KD on one batch"),
         ("multitask", "multitask: class + parity, shared trunk")]


def style(ax, t):
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["axis"])
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=t["muted"], labelsize=9, length=3)
    for l in ax.get_xticklabels() + ax.get_yticklabels():
        l.set_color(t["muted"])
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])


def ramp(hue):
    """Single-hue sequential ramp by opacity -- correct over either surface."""
    r, g, b = mpl.colors.to_rgb(hue)
    return LinearSegmentedColormap.from_list(
        "seq", [(r, g, b, 0.0), (r, g, b, 0.35), (r, g, b, 1.0)])


# --------------------------------------------------------------- figure 1 ---

def fig_correlation(d, mode):
    t = TOK[mode]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    fig.patch.set_alpha(0.0)
    im = None
    for ax, (prob, title) in zip(axes, PROBS):
        s = d[f"samples_{prob}"]
        rho, sens = s[1], s[2]
        style(ax, t)
        im = ax.hist2d(rho, sens, bins=(90, 70),
                       range=[[-1, 1], [0, 0.16]],
                       cmap=ramp(t["s1"]), norm=LogNorm(vmin=1, vmax=6000),
                       rasterized=True)[3]
        x = np.linspace(-1, 1, 400)
        for c in (0.05, 0.10, 0.20):
            with np.errstate(divide="ignore"):
                y = c / (2 * np.abs(x))
            ax.plot(x, np.where(y <= 0.163, y, np.nan), lw=1.0,
                    ls=(0, (4, 3)), color=t["s2"])
            ax.text(0.97, c / 2 + 0.003, f"{c:.2f}", fontsize=8.5,
                    color=t["s2"], ha="right", va="bottom")
        ax.axvline(0, color=t["axis"], lw=1.0)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 0.163)
        ax.set_xlabel(r"noise correlation  $\rho_i$")
        ax.set_title(title, fontsize=10, color=t["ink2"], pad=8)
        ax.grid(color=t["grid"], lw=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"sensitivity  $\varphi(a_i)\,\varphi(b_i)$")
    cb = fig.colorbar(im, ax=axes, pad=0.015, fraction=0.035)
    cb.set_label("coordinates per cell", color=t["ink2"], fontsize=9)
    cb.ax.tick_params(colors=t["muted"], labelsize=8.5, length=3)
    cb.outline.set_visible(False)
    fig.savefig(f"correlation-{mode}.png", transparent=True,
                bbox_inches="tight", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------- figure 2 ---

def reliability(f, z, nbin=12):
    q = np.quantile(f, np.linspace(0, 1, nbin + 1))
    q[0], q[-1] = -np.inf, np.inf
    xs, ys = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (f > lo) & (f <= hi)
        if m.sum() > 50:
            xs.append(f[m].mean())
            ys.append(z[m].mean())
    return np.array(xs), np.array(ys)


def fig_reliability(d, mode):
    """Left: the kappa knob. Right: what ignoring rho costs in calibration."""
    t = TOK[mode]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    fig.patch.set_alpha(0.0)

    ax = axes[0]
    style(ax, t)
    kap = d["kappa"]
    for (prob, _), col, mk in zip(PROBS, (t["s1"], t["s2"]), ("o", "s")):
        sel = d["problem"] == prob
        ks = np.unique(kap[sel])
        ys = [d["mean_absrho"][sel & (kap == k)].mean() for k in ks]
        ax.plot(ks, ys, lw=2.0, color=col, marker=mk, ms=6.5,
                solid_capstyle="round", label=prob, zorder=3)
        ax.annotate(prob, (ks[-1], ys[-1]), textcoords="offset points",
                    xytext=(-4, 9), fontsize=9.5, color=col, ha="right")
    floor = d["mean_absrho"][kap == 0.0].mean()
    ax.axhline(floor, lw=1.0, ls=(0, (4, 3)), color=t["muted"], zorder=1)
    ax.text(0.05, floor - 0.045, "estimation floor at $R = 200$ draws",
            fontsize=8.5, color=t["muted"])
    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(0, 0.88)
    ax.set_xlabel(r"shared fraction of the two batches  $\kappa$")
    ax.set_ylabel(r"mean $|\rho_i|$")
    ax.set_title("correlation is made by sharing the batch", fontsize=10,
                 color=t["ink2"], pad=8)
    ax.grid(color=t["grid"], lw=0.7)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    ax = axes[1]
    style(ax, t)
    s = d["samples_retain"]
    f0, frho, z = s[4], s[5], s[6]
    ax.plot([0, 0.7], [0, 0.7], lw=1.0, ls=(0, (4, 3)), color=t["muted"],
            zorder=1)
    for f, col, mk, lab in ((f0, t["s1"], "o", r"$f^0$  (Adam2D)"),
                            (frho, t["s2"], "s", r"$f^\rho$")):
        x, y = reliability(f, z)
        ax.plot(x, y, lw=2.0, color=col, marker=mk, ms=5.5, label=lab,
                zorder=3, solid_capstyle="round")
    ax.set_xlim(-0.02, 0.7)
    ax.set_ylim(-0.02, 0.7)
    ax.set_xlabel(r"predicted agreement  $f_i$")
    ax.set_ylabel("observed frequency of\n" r"$g_{1,i}\,g_{2,i} > 0$")
    ax.set_title(r"$\kappa = 1$, retention problem", fontsize=10,
                 color=t["ink2"], pad=8)
    ax.grid(color=t["grid"], lw=0.7)
    ax.set_axisbelow(True)
    ax.text(0.66, 0.68, "perfect\ncalibration", fontsize=8.5,
            color=t["muted"], ha="right", va="top")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    fig.tight_layout(pad=0.6)
    fig.savefig(f"reliability-{mode}.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    d = np.load("measure_results.npz", allow_pickle=True)
    for prob, _ in PROBS:
        s = d[f"samples_{prob}"]
        print(f"{prob}: {s.shape[1]} coordinate-checkpoints, "
              f"mean rho = {s[1].mean():+.3f}, "
              f"mean |df| = {s[3].mean():.4f}, "
              f"mass with |rho|>0.5 and phi.phi>0.05: "
              f"{np.mean((s[0] > 0.5) & (s[2] > 0.05)):.1%}")
    for m in ("light", "dark"):
        fig_correlation(d, m)
        fig_reliability(d, m)
    print("wrote correlation-{light,dark}.png, reliability-{light,dark}.svg")
