"""Figures for the 2C post. Run experiment.py first, then: python make_figures.py

Figure 1 — how much of the true PMI each estimator recovers as the f subgroup
           shrinks (left), and what the doubled label space costs on the
           original task as C grows (right).
Figure 2 — the sign test against its permutation null.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

TOK = {
    "light": dict(s1="#2a78d6", s2="#eb6834", s3="#8a6fbf", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", ink="#0b0b0b", ink2="#52514e",
                  warn="#d03b3b", band="#e1e0d9"),
    "dark":  dict(s1="#3987e5", s2="#d95926", s3="#a08ad0", muted="#898781",
                  grid="#2c2c2a", axis="#383835", ink="#ffffff", ink2="#c3c2b7",
                  warn="#e66767", band="#3a3a37"),
}
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.5, "svg.fonttype": "none", "figure.dpi": 110,
})

R = np.load("experiment_results.npz", allow_pickle=True)
K = {k: i for i, k in enumerate(R["keys1"])}
D = 30                                        # dimension used in experiment.py


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


def mean_se(arr):                             # arr: [config, seed]
    return arr.mean(1), arr.std(1, ddof=1) / np.sqrt(arr.shape[1])


def figure1(mode):
    t = TOK[mode]
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(9.8, 4.1), gridspec_kw=dict(wspace=0.28))
    fig.patch.set_alpha(0.0)
    for ax in (axL, axR):
        style_axes(ax, t)

    # ---- left: how much of the PMI survives -------------------------------
    hom, het = R["hom"], R["het"]
    nf = hom[:, :, K["n_f"]].mean(1)
    series = [("2C: one softmax over $2C$ cells", "corr_twoc", t["s1"], "-", "o"),
              ("Mix: two models, two softmaxes", "corr_mix", t["s2"], "-", "s"),
              ("Mix + Ledoit–Wolf shrinkage", "corr_mix_lw", t["s3"], "--", "^")]
    for label, key, colour, ls, mk in series:
        mu, se = mean_se(hom[:, :, K[key]])
        axL.plot(nf, mu, color=colour, lw=1.8, ls=ls, marker=mk, ms=4.5,
                 label=label, zorder=3)
        axL.fill_between(nf, mu - se, mu + se, color=colour, alpha=0.16,
                         linewidth=0, zorder=2)

    axL.axvspan(nf.min() * 0.75, D, color=t["warn"], alpha=0.08, linewidth=0,
                zorder=0)
    axL.annotate("$n_f < d$: $\\hat\\Sigma_{\\mathrm{f}}$ singular,\nMix returns"
                 " nothing at all", xy=(nf.min() * 1.05, 0.80),
                 color=t["warn"], fontsize=9, ha="left", va="center")
    axL.axhline(0, color=t["grid"], lw=1.0, zorder=0)
    axL.set_xscale("log")
    axL.set_xlim(nf.min() * 0.75, nf.max() * 1.35)
    axL.set_ylim(-0.08, 1.02)
    axL.set_xlabel("size of the $\\mathrm{f}$ subgroup,  $n_f$   (of $n = 4000$)")
    axL.set_ylabel("correlation of $\\Delta M$ with the true PMI")
    axL.set_title("recovering a concept the tag really carries", fontsize=11, pad=8)
    leg = axL.legend(loc="lower right", frameon=False, fontsize=9,
                     handletextpad=0.5, borderpad=0.1)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    # (the covariance-borne variant reverses the ordering entirely; it is a
    #  table in the post rather than a fourth line here)

    # ---- right: what the extra cells cost ---------------------------------
    Cs, cap = R["cap_C"], R["cap"]
    for label, idx, colour, mk in (("2C, marginalised to $y$", 0, t["s1"], "o"),
                                   ("tag-blind LDA", 1, t["muted"], "s")):
        mu, se = mean_se(cap[:, :, idx])
        axR.errorbar(Cs, mu, yerr=se, color=colour, lw=1.8, marker=mk, ms=4.5,
                     capsize=2.5, label=label, zorder=3)
    axR.plot(Cs, cap[:, :, 2].mean(1), color=t["ink2"], lw=1.2, ls=(0, (4, 3)),
             label="Bayes rate", zorder=2)
    axR.set_xscale("log")
    axR.set_xticks(Cs)
    axR.set_xticklabels([str(c) for c in Cs])
    axR.set_xlabel("number of classes $C$   (the fit carries $2C$ cells)")
    axR.set_ylabel("accuracy on $y$")
    axR.set_title("what the doubled label space costs", fontsize=11, pad=8)
    leg = axR.legend(loc="upper right", frameon=False, fontsize=9,
                     handletextpad=0.5, borderpad=0.1)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    # inset: the paired difference, which is what the overlap above hides
    ins = axR.inset_axes([0.10, 0.13, 0.44, 0.30])
    style_axes(ins, t)
    diff = cap[:, :, 0] - cap[:, :, 1]
    dmu, dse = mean_se(diff)
    ins.axhline(0, color=t["grid"], lw=1.0)
    ins.errorbar(Cs, 100 * dmu, yerr=100 * 2 * dse, color=t["s1"], lw=1.2,
                 marker="o", ms=3, capsize=2)
    ins.set_xscale("log")
    ins.set_xticks(Cs); ins.set_xticklabels([])
    ins.tick_params(labelsize=8)
    ins.set_title("2C $-$ blind, points of accuracy ($\\pm2$ se)", fontsize=8,
                  color=t["ink2"], pad=3)

    fig.savefig(f"recovery-{mode}.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


def figure2(mode):
    t = TOK[mode]
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    fig.patch.set_alpha(0.0)
    style_axes(ax, t)

    dl, sg = R["sign_delta"], R["sign"]
    tau, null_m, null_s = sg[:, :, 0], sg[:, :, 1], sg[:, :, 2]
    x = np.arange(len(dl))

    nm = null_m.mean(1)
    ns = null_s.mean(1)
    ax.fill_between(x, nm - 2 * ns, nm + 2 * ns, color=t["band"], alpha=0.85,
                    linewidth=0, zorder=1,
                    label="permutation null, $\\pm 2$ sd")
    ax.plot(x, nm, color=t["muted"], lw=1.2, ls=(0, (4, 3)), zorder=2)
    mu, se = mean_se(tau)
    ax.errorbar(x, mu, yerr=se, color=t["s1"], lw=1.8, marker="o", ms=5,
                capsize=2.5, zorder=4, label="$\\tau$ measured")
    ax.axhline(0, color=t["grid"], lw=1.0, zorder=0)
    ax.annotate("$\\tau < 0$ on every run,\nsignal or no signal",
                xy=(0.05, mu[0]), xytext=(0.05, mu[0] * 0.45),
                color=t["ink2"], fontsize=9, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=t["ink2"], lw=1.0,
                                shrinkA=2, shrinkB=4))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in dl])
    ax.set_xlabel("concept strength $\\delta$   ($0$ = the tag carries nothing)")
    ax.set_ylabel("$\\tau$")
    ax.set_title("the sign is free; the decision is not", fontsize=11, pad=8)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=9,
                    handletextpad=0.5, borderpad=0.1)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    fig.savefig(f"tau-{mode}.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


for m in ("light", "dark"):
    figure1(m)
    figure2(m)
print("wrote recovery-{light,dark}.svg and tau-{light,dark}.svg")
