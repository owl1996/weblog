"""What the correlated gate costs.

The claim to check is that going from f^0 to f^rho is not a new pass over data.
Two things are measured, both scale-free:

  * the elementwise gate, per coordinate: two Owen's T calls against two erf;
  * the cross-covariance accumulation, against the standard-deviation
    accumulation over the same k aligned micro-batch gradients.

Ratios against a *gradient evaluation* are deliberately not reported: they say
more about numpy temporaries versus fused backward kernels than about the
method. What is reported is that the gate stays O(d) elementwise and the
cross-covariance stays O(kd) on buffers already in memory.

Run: python cost.py
"""
import time

import numpy as np

from orthant import focus, focus_indep

D = 2_000_000
K = 8
REP = 12


def timeit(fn, rep=REP):
    fn()
    ts = []
    for _ in range(rep):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def sigma_only(G1, G2):
    return G1.mean(0), G2.mean(0), G1.std(0) / np.sqrt(K), G2.std(0) / np.sqrt(K)


def cross_cov(G1, G2):
    return ((G1 - G1.mean(0)) * (G2 - G2.mean(0))).sum(0) / (K - 1) / K


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.standard_normal(D)
    b = rng.standard_normal(D)
    rho = rng.uniform(-0.9, 0.9, D)
    G1 = rng.standard_normal((K, D))
    G2 = rng.standard_normal((K, D))

    t0 = timeit(lambda: focus_indep(a, b))
    tr = timeit(lambda: focus(a, b, rho))
    ts = timeit(lambda: sigma_only(G1, G2))
    tc = timeit(lambda: cross_cov(G1, G2))

    print(f"d = {D:,}, k = {K} micro-batches, float64 numpy, median of {REP}\n")
    print(f"{'item':46s} {'time':>9s} {'per coord':>11s}")
    for lab, t in [("gate f^0   -- 2 erf, O(d)", t0),
                   ("gate f^rho -- 2 Owen's T, O(d)", tr),
                   ("ghat_k and sigma_k from k splits, O(kd)", ts),
                   ("cross-covariance c12, same splits, O(kd)", tc)]:
        print(f"{lab:46s} {t * 1e3:8.1f}ms {t / D * 1e9:10.1f}ns")
    print(f"\n{'Owen T gate / erf gate':46s} {tr / t0:8.2f}x")
    print(f"{'c12 accumulation / sigma accumulation':46s} {tc / ts:8.2f}x")
    print(f"{'whole correlated gate / whole plain gate':46s} "
          f"{(tr + ts + tc) / (t0 + ts):8.2f}x")
