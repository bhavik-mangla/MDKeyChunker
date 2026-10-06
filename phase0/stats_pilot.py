"""Self-contained v2-style statistics for the v3 PILOT data (numpy only).

Input convention (as in results/*_per_question.json):
    values: {cluster_id: [per-question values in fixed order]}
A paired difference is the elementwise A - B within each cluster. Qasper
clusters are papers (mean over questions, papers resampled); FreshStack
clusters are single questions.

Provided:
  paired_diff(pa, pb)                 -> {cluster: [a-b, ...]}
  boot_ci(diffs, levels, n, seed)     -> mean, {level: (lo, hi)}   percentile cluster bootstrap
  signflip_p(diffs, n, seed, shift, alternative)  cluster-level sign-flip permutation test
  tost(diffs, margin, ...)            -> TOST p (max of two one-sided shifted sign-flip tests)
  holm(pvals)                         -> Holm-adjusted p-values (same order)
  verdict(...)                        -> "equivalent" / "different" / "different but equivalent" / "inconclusive"
  analyse(pa, pb, margin)             -> dict with everything above
"""
import numpy as np


def paired_diff(pa: dict, pb: dict) -> dict:
    assert pa.keys() == pb.keys() and all(len(pa[k]) == len(pb[k]) for k in pa), "unpaired"
    return {k: [x - y for x, y in zip(pa[k], pb[k])] for k in pa}


def _cluster_arrays(diffs: dict):
    keys = list(diffs)
    sums = np.array([sum(diffs[k]) for k in keys], dtype=float)
    counts = np.array([len(diffs[k]) for k in keys], dtype=float)
    return sums, counts


def boot_ci(diffs: dict, levels=(0.95, 0.90), n: int = 10_000, seed: int = 0):
    """Mean over questions; percentile CI from resampling clusters with replacement."""
    sums, counts = _cluster_arrays(diffs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n, len(sums)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    out = {}
    for lv in levels:
        a = (1 - lv) / 2
        out[lv] = (float(np.quantile(means, a)), float(np.quantile(means, 1 - a)))
    return float(sums.sum() / counts.sum()), out


def signflip_p(diffs: dict, n: int = 10_000, seed: int = 0, shift: float = 0.0,
               alternative: str = "two-sided") -> float:
    """Paired sign-flip permutation test at the cluster level.

    Statistic: mean over questions of (d + shift). Under H0 each cluster's
    (shifted) differences are symmetric about 0, so all of a cluster's values
    flip sign together. p includes the observed statistic (+1 correction).
    """
    keys = list(diffs)
    s = np.array([sum(x + shift for x in diffs[k]) for k in keys], dtype=float)
    N = sum(len(diffs[k]) for k in keys)
    obs = s.sum() / N
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(s)))
    null = (signs * s).sum(1) / N
    eps = 1e-12
    if alternative == "greater":
        k = (null >= obs - eps).sum()
    elif alternative == "less":
        k = (null <= obs + eps).sum()
    else:
        k = (np.abs(null) >= abs(obs) - eps).sum()
    return float((k + 1) / (n + 1))


def tost(diffs: dict, margin: float, n: int = 10_000, seed: int = 0) -> dict:
    """Two one-sided tests: H0a d <= -margin (test d + margin > 0), H0b d >= +margin."""
    p_lo = signflip_p(diffs, n, seed, shift=+margin, alternative="greater")
    p_hi = signflip_p(diffs, n, seed, shift=-margin, alternative="less")
    return {"p_lower": p_lo, "p_upper": p_hi, "p_tost": max(p_lo, p_hi)}


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank_, i in enumerate(order):
        running = max(running, min(1.0, (m - rank_) * pvals[i]))
        adj[i] = running
    return adj


def verdict(ci95, ci90, margin) -> str:
    diff = ci95[0] > 0 or ci95[1] < 0
    equiv = ci90[0] > -margin and ci90[1] < margin
    if diff and equiv:
        return "different but equivalent"
    if diff:
        return "different"
    if equiv:
        return "equivalent"
    return "inconclusive"


def analyse(pa: dict, pb: dict, margin: float, n: int = 10_000, seed: int = 0) -> dict:
    d = paired_diff(pa, pb)
    mean, cis = boot_ci(d, (0.95, 0.90), n, seed)
    t = tost(d, margin, n, seed)
    return {"diff": mean, "ci95": list(cis[0.95]), "ci90": list(cis[0.90]),
            "p_perm": signflip_p(d, n, seed), **t, "margin": margin,
            "verdict_ci": verdict(cis[0.95], cis[0.90], margin),
            "verdict_perm": verdict_p(signflip_p(d, n, seed), t["p_tost"])}


def verdict_p(p_diff: float, p_tost: float, alpha: float = 0.05) -> str:
    """Same decision language from permutation p-values (unadjusted)."""
    diff, equiv = p_diff < alpha, p_tost < alpha
    return ("different but equivalent" if diff and equiv else "different" if diff
            else "equivalent" if equiv else "inconclusive")
