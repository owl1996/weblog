"""Figures for the 2D-Adam post. Run: python make_figures.py"""
import numpy as np
from scipy.stats import norm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- design tokens (see dataviz reference palette) -------------------------
TOK = {
    "light": dict(s1="#2a78d6", s2="#eb6834", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", ink="#0b0b0b", ink2="#52514e",
                  neutral="#f0efec", pole_lo="#d03b3b", pole_hi="#2a78d6"),
    "dark":  dict(s1="#3987e5", s2="#d95926", muted="#898781",
                  grid="#2c2c2a", axis="#383835", ink="#ffffff", ink2="#c3c2b7",
                  neutral="#383835", pole_lo="#e66767", pole_hi="#3987e5"),
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.5,
    "svg.fonttype": "none",
    "figure.dpi": 110,
})


def style_axes(ax, t):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=t["muted"], labelsize=9.5, length=3, width=1.0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(t["muted"])
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])
    ax.title.set_color(t["ink"])


psi_adam = lambda s: s / np.sqrt(1.0 + s ** 2)
psi_phi = lambda s: 2 * norm.cdf(s) - 1


# --- Figure 1: the two squashings ------------------------------------------
def fig_squash(mode):
    t = TOK[mode]
    s = np.linspace(-4.2, 4.2, 900)
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(9.4, 3.9), gridspec_kw=dict(width_ratios=[1.55, 1], wspace=0.28))
    fig.patch.set_alpha(0.0)
    style_axes(ax, t)
    style_axes(ax2, t)

    ax.axhline(0, color=t["grid"], lw=1.0, zorder=0)
    ax.axvline(0, color=t["grid"], lw=1.0, zorder=0)
    for y in (-1, 1):
        ax.axhline(y, color=t["grid"], lw=1.0, ls=(0, (4, 4)), zorder=0)

    ax.plot(s, np.sign(s), color=t["muted"], lw=1.4, ls=(0, (2, 3)), zorder=1)
    ax.plot(s, psi_adam(s), color=t["s1"], lw=2.0, zorder=3)
    ax.plot(s, psi_phi(s), color=t["s2"], lw=2.0, zorder=2)

    ax.annotate(r"$s/\sqrt{1+s^2}$" "\n(Adam)", xy=(2.6, psi_adam(2.6)),
                xytext=(2.15, 0.52), color=t["s1"], fontsize=10, ha="left", va="top")
    ax.annotate(r"$2\Phi(s)-1$", xy=(1.9, psi_phi(1.9)), xytext=(0.62, 1.06),
                color=t["s2"], fontsize=10, ha="left", va="bottom")
    ax.annotate(r"$\mathrm{sign}(s)$", xy=(3.9, 1.0), xytext=(3.55, 0.80),
                color=t["muted"], fontsize=9.5, ha="right", va="top")

    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-1.35, 1.35)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("per-coordinate signal-to-noise ratio  $s$")
    ax.set_ylabel("step multiplier")

    # right panel: how fast each one commits to the sign
    sp = np.linspace(0.05, 4.2, 600)
    ax2.semilogy(sp, 1 - psi_adam(sp), color=t["s1"], lw=2.0)
    ax2.semilogy(sp, 1 - psi_phi(sp), color=t["s2"], lw=2.0)
    ax2.annotate(r"$\sim 1/2s^{2}$", xy=(3.5, 1 - psi_adam(3.5)), xytext=(3.45, 0.11),
                 color=t["s1"], fontsize=10, ha="right", va="center")
    ax2.annotate(r"$\sim e^{-s^{2}/2}$", xy=(3.0, 1 - psi_phi(3.0)), xytext=(2.55, 4e-4),
                 color=t["s2"], fontsize=10, ha="right", va="center")
    ax2.set_ylim(1e-5, 2)
    ax2.set_xlim(0, 4.3)
    ax2.grid(axis="y", color=t["grid"], lw=0.8, which="major")
    ax2.set_axisbelow(True)
    ax2.set_xlabel("$s$")
    ax2.set_ylabel(r"residual to $\mathrm{sign}$:  $1-\psi(s)$")
    fig.tight_layout(pad=0.4)
    fig.savefig(f"squashing-{mode}.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


# --- Figure 2: the 2D gate --------------------------------------------------
def fig_gate(mode):
    t = TOK[mode]
    cmap = LinearSegmentedColormap.from_list(
        "gate", [t["pole_lo"], t["neutral"], t["pole_hi"]], N=256)
    g = np.linspace(-3, 3, 500)
    SU, SC = np.meshgrid(g, g)
    F = 0.5 * (1 + psi_phi(SU) * psi_phi(SC))

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    fig.patch.set_alpha(0.0)
    im = ax.imshow(F, cmap=cmap, vmin=0, vmax=1, origin="lower",
                   extent=[-3, 3, -3, 3], interpolation="bilinear")
    cs = ax.contour(SU, SC, F, levels=[0.1, 0.25, 0.75, 0.9],
                    colors=t["ink"], linewidths=0.7, alpha=0.35)
    ax.clabel(cs, fmt="%.2f", fontsize=8, colors=t["ink2"])
    style_axes(ax, t)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(t["axis"])

    ax.set_xlabel(r"$s_1$   (SNR on task 1)")
    ax.set_ylabel(r"$s_2$   (SNR on task 2)")
    ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[0, 0.25, .5, .75, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    cb.set_label("focus  $f$", color=t["ink2"], fontsize=10)

    ax.text(1.55, 1.55, "agree\n$f\\to1$", ha="center", va="center",
            fontsize=9.5, color=t["ink"], alpha=0.75)
    ax.text(-1.55, 1.55, "conflict\n$f\\to0$", ha="center", va="center",
            fontsize=9.5, color=t["ink"], alpha=0.75)
    ax.text(-1.55, -1.55, "agree\n$f\\to1$", ha="center", va="center",
            fontsize=9.5, color=t["ink"], alpha=0.75)
    ax.text(1.55, -1.55, "conflict\n$f\\to0$", ha="center", va="center",
            fontsize=9.5, color=t["ink"], alpha=0.75)
    fig.tight_layout(pad=0.4)
    fig.savefig(f"gate-{mode}.png", transparent=True, bbox_inches="tight", dpi=220)
    plt.close(fig)


for m in ("light", "dark"):
    fig_squash(m)
    fig_gate(m)
print("done")
