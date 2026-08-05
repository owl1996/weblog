"""Figure for the Pareto-layer reduction (Proposition 2 of the certification post).

Left  — one real two-constraint instance in the (a, b) plane, symlog on both
        axes because the coefficients span twenty orders of magnitude. Points
        are coloured by Pareto layer toward (-inf, -inf); only the first Kbar
        layers can contain a removal, so the whole cloud collapses onto a thin
        staircase along the lower-left frontier.
right — the reduction across every well-posed instance: candidates under the
        dominance prefilter alone against candidates after the layer
        restriction.

Instances are the genuine (non-vacuous) ones collected by biconstrained.py.
Correctness of the extraction is not this script's job: test_pareto.py checks
the containment guarantee by exhaustive enumeration and test_oracle.py checks
the extraction against a naive O(d^2) peeling.

Run: python make_pareto_figure.py     (~1 min on CPU)
"""
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import biconstrained as B
from pareto import pareto_layers

SEEDS = range(4)

TOK = {
    "light": dict(s1="#2a78d6", s2="#eb6834", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", ink="#0b0b0b", ink2="#52514e", faint="#d8d7d0"),
    "dark":  dict(s1="#3987e5", s2="#d95926", muted="#898781", grid="#2c2c2a",
                  axis="#383835", ink="#ffffff", ink2="#c3c2b7", faint="#45443f"),
}
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.5, "svg.fonttype": "none", "figure.dpi": 110,
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


# --- data ------------------------------------------------------------------
raw = [x for s in SEEDS for x in B.collect(s)]
inst = [(a, b) for a, b in raw if not B.degenerate(a, b)]
d = len(inst[0][0])
print(f"{len(inst)} well-posed instances, d = {d}")

rows = []
for a, b in inst:
    n_greedy, ok = B.sweep(a, b)
    kbar = n_greedy if ok else 64            # greedy count upper-bounds |R*|
    lay = pareto_layers(a, b, kbar)
    prefilter = (a < 0) | (b < 0)            # dominance rule (I2) of the l1 post
    keep = (lay > 0) & prefilter
    rows.append(dict(a=a, b=b, kbar=kbar, lay=lay,
                     n_pre=int(prefilter.sum()), n_par=int(keep.sum())))

n_pre = np.array([r["n_pre"] for r in rows])
n_par = np.array([r["n_par"] for r in rows])
print(f"candidates, dominance prefilter only: median {np.median(n_pre):.0f}")
print(f"candidates, + Pareto restriction:     median {np.median(n_par):.0f}")
print(f"reduction: median {np.median(n_pre / np.maximum(n_par, 1)):.1f}x, "
       f"as a fraction of d: {np.median(n_par) / d:.1%}")

# a representative instance: the one whose greedy count is nearest the median
kbars = np.array([r["kbar"] for r in rows])
pick = rows[int(np.argmin(np.abs(kbars - np.median(kbars))))]
print(f"panel A instance: Kbar = {pick['kbar']}, "
      f"{pick['n_par']} candidates out of {d}")


def figure(mode):
    t = TOK[mode]
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(9.8, 4.3), gridspec_kw=dict(wspace=0.30))
    fig.patch.set_alpha(0.0)
    for ax in (axL, axR):
        style_axes(ax, t)

    # ---- left: layers of one instance ------------------------------------
    a, b, lay, kbar = pick["a"], pick["b"], pick["lay"], pick["kbar"]
    scale = max(np.abs(a).max(), np.abs(b).max())
    lin = 1e-12 * scale                      # symlog knee, well below the mass
    beyond = lay == 0
    axL.scatter(a[beyond], b[beyond], s=5, color=t["faint"], zorder=1,
                edgecolors="none", rasterized=True,
                label=f"beyond layer $\\bar K$  ({beyond.sum():,})")
    inside = np.where(lay > 0)[0]
    order = np.argsort(-lay[inside])         # deep layers first, front on top
    sc = axL.scatter(a[inside][order], b[inside][order], s=13,
                     c=lay[inside][order], cmap="viridis", vmin=1, vmax=kbar,
                     zorder=3, edgecolors="none", rasterized=True)
    axL.axhline(0, color=t["grid"], lw=1.0, zorder=0)
    axL.axvline(0, color=t["grid"], lw=1.0, zorder=0)
    axL.set_xscale("symlog", linthresh=lin)
    axL.set_yscale("symlog", linthresh=lin)
    axL.set_xlabel("$a_i$   (constraint 1 coefficient)")
    axL.set_ylabel("$b_i$   (constraint 2 coefficient)")
    axL.set_title(f"one instance: layers 1 to $\\bar K = {kbar}$", fontsize=11, pad=8)
    e_hi = int(np.floor(np.log10(scale)))
    e_lo = e_hi - 6
    ticks = [-10.0 ** e_hi, -10.0 ** e_lo, 0, 10.0 ** e_lo, 10.0 ** e_hi]
    labels = [f"$-10^{{{e_hi}}}$", f"$-10^{{{e_lo}}}$", "$0$",
              f"$10^{{{e_lo}}}$", f"$10^{{{e_hi}}}$"]
    axL.set_xticks(ticks); axL.set_xticklabels(labels)
    axL.set_yticks(ticks); axL.set_yticklabels(labels)
    cb = fig.colorbar(sc, ax=axL, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    cb.set_label("Pareto layer", color=t["ink2"], fontsize=10)
    leg = axL.legend(loc="lower left", frameon=False, fontsize=9,
                     handletextpad=0.3, borderpad=0.1)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    # ---- right: the reduction over all instances --------------------------
    hi = max(n_pre.max(), n_par.max()) * 1.6
    axR.plot([1, hi], [1, hi], color=t["axis"], lw=1.0, ls=(0, (4, 3)), zorder=1)
    axR.annotate("no reduction", xy=(hi * 0.55, hi * 0.55), color=t["muted"],
                 fontsize=9, rotation=45, ha="center", va="bottom")
    axR.scatter(n_pre, n_par, s=34, color=t["s1"], alpha=0.75, zorder=3,
                edgecolors="none")
    med = np.median(n_pre / np.maximum(n_par, 1))
    axR.plot([1, hi], [1 / med, hi / med], color=t["s2"], lw=1.6, zorder=2)
    axR.annotate(f"median  ${med:.1f}\\times$", xy=(hi * 0.5, hi * 0.5 / med),
                 xytext=(hi * 0.62, hi * 0.30 / med), color=t["s2"], fontsize=9.5,
                 ha="center", va="top")
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlim(50, hi); axR.set_ylim(50, hi)
    axR.set_aspect("equal")
    axR.set_xlabel("candidates: dominance prefilter alone")
    axR.set_ylabel("candidates: + Pareto layers")
    axR.set_title(f"{len(rows)} well-posed instances, $d = {d:,}$",
                  fontsize=11, pad=8)

    fig.savefig(f"pareto-{mode}.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


for m in ("light", "dark"):
    figure(m)
print("wrote pareto-light.svg, pareto-dark.svg")
