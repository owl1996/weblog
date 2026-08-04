"""Distribution of the feasibility ratio r = -S12 / min(a/b S11, b/a S22) over
training batches. The step is feasible for the measured gradients iff r <= 1."""
import numpy as np, torch, torch.nn.functional as F
import matplotlib as mpl, matplotlib.pyplot as plt
import certificate as C
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "2026-08-03-two-dimensional-adam"))
import experiment as E

TOK = {"light": dict(s1="#2a78d6", s2="#eb6834", muted="#898781", grid="#e1e0d9",
                     axis="#c3c2b7", ink="#0b0b0b", ink2="#52514e"),
       "dark":  dict(s1="#3987e5", s2="#d95926", muted="#898781", grid="#2c2c2a",
                     axis="#383835", ink="#ffffff", ink2="#c3c2b7")}
mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 10.5, "svg.fonttype": "none", "figure.dpi": 110})


def collect(seed):
    (Xo, yo), (Xn, yn), *_ = E.make_data(seed)
    gen = torch.Generator().manual_seed(seed)
    base = E.mlp(seed); opt = torch.optim.Adam(base.parameters(), lr=3e-3)
    for _ in range(400):
        i = torch.randint(len(Xo), (E.BATCH,), generator=gen)
        opt.zero_grad(); F.cross_entropy(base(Xo[i]), yo[i]).backward(); opt.step()
    model = E.mlp(seed); model.load_state_dict(base.state_dict())
    outer = torch.optim.Adam(model.parameters(), lr=C.LR)
    g = torch.Generator().manual_seed(seed + 1000)
    ratios = []
    for _ in range(C.STEPS):
        g1, s1 = E.task_gradient(model, Xn, yn, g)
        g2, s2 = E.task_gradient(model, Xo, yo, g)
        f = C.focus(g1, s1, g2, s2)
        _, S11, S22, S12 = C.violation(torch.ones_like(f), f, g1, g2)
        budget = torch.min(C.ALPHA / C.BETA * S11, C.BETA / C.ALPHA * S22).item()
        ratios.append((-S12.item()) / budget if budget > 0 else np.nan)
        delta = -f * (C.ALPHA * g1 + C.BETA * g2)
        i = 0
        for p in model.parameters():
            n = p.numel(); p.grad = (-delta[i:i + n].view_as(p)).clone(); i += n
        outer.step(); outer.zero_grad(set_to_none=False)
    return np.array(ratios)


def plot(r, mode):
    t = TOK[mode]
    fig, ax = plt.subplots(figsize=(6.8, 3.4)); fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["axis"]); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=t["muted"], labelsize=9.5, length=3)
    for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_color(t["muted"])

    bins = np.linspace(-2.5, 2.5, 70)
    feas, infeas = r[r <= 1], r[r > 1]
    ax.hist(feas, bins=bins, color=t["s1"], alpha=0.85, label="feasible  ($r \\leq 1$)")
    ax.hist(infeas, bins=bins, color=t["s2"], alpha=0.85, label="needs repair  ($r > 1$)")
    ax.axvline(1.0, color=t["ink"], lw=1.2, ls=(0, (4, 3)))
    ax.text(1.06, ax.get_ylim()[1] * 0.93, "feasibility\nboundary", fontsize=9,
            color=t["ink2"], va="top")
    ax.set_xlabel(r"conflict-to-budget ratio  $r = -\tilde S_{12} \,/\, \min(\frac{\alpha}{\beta}\tilde S_{11}, \frac{\beta}{\alpha}\tilde S_{22})$")
    ax.set_ylabel("batches")
    ax.xaxis.label.set_color(t["ink2"]); ax.yaxis.label.set_color(t["ink2"])
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for txt in leg.get_texts(): txt.set_color(t["ink2"])
    ax.grid(axis="y", color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    fig.savefig(f"violation-{mode}.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    r = np.concatenate([collect(s) for s in range(6)])
    r = r[np.isfinite(r)]
    print(f"batches: {len(r)}   infeasible: {(r>1).mean():.1%}")
    print(f"median r = {np.median(r):+.3f}   90th pct = {np.percentile(r,90):+.3f}   "
          f"max = {r.max():+.3f}")
    print(f"among infeasible: median r = {np.median(r[r>1]):+.3f}, max = {r[r>1].max():+.3f}")
    for m in ("light", "dark"): plot(r, m)
    np.save("violation_ratios.npy", r)
