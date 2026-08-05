"""Figure for "Gradient masking for two-task alignment". Run: python make_figures.py

Draws the d = 3 counterexample in the (a, b) plane:
  left  — the three coordinates, the half-plane a + b >= 0 forced by (I1),
          and the two orders the greedy sorts by;
  right — the running total (A, B), the feasible quadrant, and the three
          single-removal outcomes, so that the greedy's choice and the optimum
          are visible as points rather than as a table.

Numbers are recomputed here from g1, g2 rather than hard-coded; verify.py is
the authority on them and this script asserts agreement with the post's table.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- design tokens (shared with the other posts' figures) ------------------
TOK = {
    "light": dict(s1="#2a78d6", s2="#eb6834", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", ink="#0b0b0b", ink2="#52514e",
                  good="#2f8f5b", bad="#d03b3b", fill="#e1e0d9", fill_a=0.55),
    "dark":  dict(s1="#3987e5", s2="#d95926", muted="#898781",
                  grid="#2c2c2a", axis="#383835", ink="#ffffff", ink2="#c3c2b7",
                  good="#4fb37c", bad="#e66767", fill="#3a3a37", fill_a=0.95),
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


# --- the counterexample of the post ----------------------------------------
g1 = np.array([-3.0, -2.0, -2.0])
g2 = np.array([1.0, 1.0, 3.0])
alpha = beta = 1.0
delta = alpha * g1 + beta * g2
a = delta * g1
b = delta * g2

assert np.allclose(delta, [-2, -1, 1])
assert np.allclose(a, [6, 2, -2]) and np.allclose(b, [-2, -1, 3])
assert np.allclose(a + b, delta ** 2)          # identity (I1)
assert (g1 * g2 < 0).all()                     # dominance prunes nothing here

A0, B0 = a.sum(), b.sum()                      # the unmasked total, (6, 0)
assert (A0, B0) == (6.0, 0.0)
# single removals: the only ones that matter here
totals = {i: (A0 - a[i], B0 - b[i]) for i in range(3)}
assert totals[0] == (0.0, 2.0) and totals[1] == (4.0, 1.0) and totals[2] == (8.0, -3.0)
GREEDY, OPT = 0, 1                             # 0-indexed; the post counts from 1


def figure(mode):
    t = TOK[mode]
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(9.6, 4.3), gridspec_kw=dict(wspace=0.30))
    fig.patch.set_alpha(0.0)
    for ax in (axL, axR):
        style_axes(ax, t)
        ax.axhline(0, color=t["grid"], lw=1.0, zorder=0)
        ax.axvline(0, color=t["grid"], lw=1.0, zorder=0)

    # ---------------- left: the coordinates -------------------------------
    lim = 8.0
    xs = np.linspace(-lim, lim, 200)
    axL.fill_between(xs, -xs, lim, color=t["fill"], alpha=t["fill_a"], zorder=0,
                     linewidth=0)
    axL.plot(xs, -xs, color=t["axis"], lw=1.0, ls=(0, (4, 3)), zorder=1)
    axL.annotate("(I1): every coordinate has\n$a_i + b_i = \\Delta_i^2 \\geq 0$",
                 xy=(-5.4, 5.8), color=t["ink2"], fontsize=9.5, ha="left",
                 va="center")

    for i in range(3):
        colour = t["s2"] if i == GREEDY else (t["good"] if i == OPT else t["muted"])
        axL.scatter([a[i]], [b[i]], s=64, color=colour, zorder=4,
                    edgecolors="none")
        axL.annotate(f"$i={i+1}$", xy=(a[i], b[i]), xytext=(a[i] + 0.42, b[i] + 0.42),
                     color=colour, fontsize=10.5, ha="left", va="bottom", zorder=4)

    # the two orders the greedy may sort by
    for i in range(3):
        axL.plot([a[i], a[i]], [b[i], -lim + 0.55], color=t["s1"], lw=0.8,
                 ls=(0, (1, 2.5)), zorder=2)
        axL.plot([a[i], -lim + 0.55], [b[i], b[i]], color=t["s2"], lw=0.8,
                 ls=(0, (1, 2.5)), zorder=2)
    axL.annotate("branch $A$ sorts by $a_i$", xy=(0, -lim + 0.1),
                 xytext=(-lim + 0.4, -lim + 0.05), color=t["s1"], fontsize=9.5,
                 ha="left", va="bottom")
    axL.annotate("branch $B$\nsorts by $b_i$", xy=(-lim + 0.4, 3.0),
                 xytext=(-lim + 0.4, 3.4), color=t["s2"], fontsize=9.5,
                 ha="left", va="bottom")

    axL.set_xlim(-lim, lim)
    axL.set_ylim(-lim, lim)
    axL.set_aspect("equal")
    axL.set_xlabel("$a_i = \\Delta_i g_{1,i}$")
    axL.set_ylabel("$b_i = \\Delta_i g_{2,i}$")
    axL.set_title("the three coordinates", fontsize=11, pad=8)

    # ---------------- right: the totals -----------------------------------
    lo, hi = -5.0, 10.4
    axR.fill_between([0, hi], 0, hi, color=t["good"], alpha=0.10, zorder=0,
                     linewidth=0)
    axR.annotate("feasible:\n$A>0$ and $B>0$", xy=(7.4, 7.0), color=t["good"],
                 fontsize=9.5, ha="center", va="center")

    axR.scatter([A0], [B0], s=64, color=t["ink"], zorder=5, edgecolors="none")
    axR.annotate("full mask $(6,0)$\ninfeasible", xy=(A0, B0),
                 xytext=(A0 + 0.55, B0 - 0.55), color=t["ink2"], fontsize=9.5,
                 ha="left", va="top")

    labels = {
        GREEDY: ("greedy removes $i=1$\n$(0,2)$ — still infeasible", t["s2"],
                 (-4.6, 3.2), "left"),
        OPT: ("optimum removes $i=2$\n$(4,1)$ — feasible", t["good"],
              (4.6, 4.4), "center"),
        2: ("removing $i=3$\n$(8,-3)$", t["muted"], (8.9, -3.1), "left"),
    }
    for i, (txt, colour, xytext, ha) in labels.items():
        x, y = totals[i]
        axR.annotate("", xy=(x, y), xytext=(A0, B0),
                     arrowprops=dict(arrowstyle="->", color=colour, lw=1.5,
                                     shrinkA=6, shrinkB=6, alpha=0.9))
        axR.scatter([x], [y], s=64, color=colour, zorder=5, edgecolors="none")
        axR.annotate(txt, xy=(x, y), xytext=xytext, color=colour, fontsize=9.5,
                     ha=ha, va="bottom")

    axR.set_xlim(lo, hi)
    axR.set_ylim(lo, hi)
    axR.set_aspect("equal")
    axR.set_xlabel("$A(M) = \\sum_{i \\in M} a_i$")
    axR.set_ylabel("$B(M) = \\sum_{i \\in M} b_i$")
    axR.set_title("what one removal does to the total", fontsize=11, pad=8)

    fig.tight_layout(pad=0.4)
    fig.savefig(f"counterexample-{mode}.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)


for m in ("light", "dark"):
    figure(m)
print("wrote counterexample-light.svg, counterexample-dark.svg")
