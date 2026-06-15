"""Statistics helpers: cluster bootstrap CIs, McNemar's paired test, Holm correction.

Kept dependency-light (numpy + math) so it runs anywhere the harness runs.
"""
import math

import numpy as np


def cluster_bootstrap(cluster_indices, stat_fns, n_boot=1000, seed=42, ci=0.95):
    """Bootstrap CIs by resampling CLUSTERS (one cluster = all observations of one
    example, across repeated runs). This respects the correlation between repeated
    runs of the same example, giving an honest sampling CI rather than pretending
    each run is independent.

    cluster_indices : list of 1-D int arrays indexing into the global observation arrays
    stat_fns        : dict name -> fn(idx_array) -> float
    returns         : dict name -> {"point", "lo", "hi"}
    """
    rng = np.random.default_rng(seed)
    all_idx = np.concatenate(cluster_indices) if cluster_indices else np.array([], int)
    point = {k: _safe(fn, all_idx) for k, fn in stat_fns.items()}
    draws = {k: [] for k in stat_fns}
    ncl = len(cluster_indices)
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    for _ in range(n_boot):
        pick = rng.integers(0, ncl, ncl)
        idx = np.concatenate([cluster_indices[i] for i in pick])
        for k, fn in stat_fns.items():
            draws[k].append(_safe(fn, idx))
    out = {}
    for k in stat_fns:
        arr = np.array([d for d in draws[k] if d is not None], float)
        if arr.size == 0:
            out[k] = {"point": point[k], "lo": None, "hi": None}
        else:
            out[k] = {"point": point[k],
                      "lo": float(np.percentile(arr, lo_q)),
                      "hi": float(np.percentile(arr, hi_q))}
    return out


def _safe(fn, idx):
    try:
        v = fn(idx)
        return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None
    except Exception:
        return None


def mcnemar(a_correct, b_correct):
    """Paired test of whether two models differ on per-example binary correctness.
    b = A wrong & B right ; c = A right & B wrong. Continuity-corrected chi-square (1 df).
    """
    a = np.asarray(a_correct, bool)
    b = np.asarray(b_correct, bool)
    n_b = int(np.sum(~a & b))      # B fixes what A got wrong
    n_c = int(np.sum(a & ~b))      # A fixes what B got wrong
    if n_b + n_c == 0:
        return {"b": n_b, "c": n_c, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(n_b - n_c) - 1) ** 2 / (n_b + n_c)
    p = math.erfc(math.sqrt(chi2 / 2)) if chi2 > 0 else 1.0   # chi-square 1df survival
    return {"b": n_b, "c": n_c, "chi2": float(chi2), "p": float(min(1.0, p))}


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (controls family-wise error)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj
