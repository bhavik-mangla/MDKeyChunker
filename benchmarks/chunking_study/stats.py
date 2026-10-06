"""Paired statistics for the v4 confirmatory analysis (ANALYSIS_PLAN_v2 Sec. 4).

Inputs are per-cluster lists of per-question paired differences (A - B), as in
evaluate.paired(): {cluster_id: [d_q1, d_q2, ...]}. The point estimate is the
mean over all questions (clusters weighted by their size), matching
evaluate.cluster_bootstrap.

- cluster_bootstrap: resample clusters with replacement (optionally within
  strata, e.g. FreshStack topics), 10,000 resamples, percentile and BCa CIs.
  BCa acceleration comes from the leave-one-cluster-out jackknife
  (Efron & Tibshirani 1993, ch. 14).
- sign_flip_test: two-sided paired permutation test flipping the sign of whole
  clusters (exact when 2^C <= n_perm).
- tost: equivalence within +/-margin. Verdict (pre-declared language):
  equivalent if the 90% CI lies inside (-margin, +margin); different if the 95%
  CI excludes 0; inconclusive otherwise. Both can hold ("different but
  equivalent"): verdict is then "equivalent" and label says so. p_tost is
  obtained by inverting the same bootstrap CI, so p_tost < alpha exactly when
  the 90% CI is inside the margin; a cluster-robust t version is also given.
- holm: Holm-Bonferroni step-down adjusted p-values.
- cohen_kappa: unweighted, linear or quadratic weights.
- spearman: rank correlation with average ranks for ties.

Only numpy and the standard library are used (no scipy in the venv).
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Hashable, Mapping, Sequence

import numpy as np

_N = NormalDist()


def _arrays(per_cluster: Mapping[Hashable, Sequence[float]]) -> tuple[list, np.ndarray, np.ndarray]:
    keys = list(per_cluster)
    if not keys:
        raise ValueError("no clusters")
    sums = np.array([float(np.sum(per_cluster[k])) for k in keys])
    cnts = np.array([len(per_cluster[k]) for k in keys], dtype=float)
    if np.any(cnts == 0):
        raise ValueError("empty cluster")
    return keys, sums, cnts


def _boot_means(sums: np.ndarray, cnts: np.ndarray, n: int, rng: np.random.Generator,
                strata: np.ndarray | None) -> np.ndarray:
    C = len(sums)
    if strata is None:
        idx = rng.integers(0, C, size=(n, C))
        return sums[idx].sum(1) / cnts[idx].sum(1)
    tot_s = np.zeros(n)
    tot_c = np.zeros(n)
    for s in np.unique(strata):
        members = np.flatnonzero(strata == s)
        idx = members[rng.integers(0, len(members), size=(n, len(members)))]
        tot_s += sums[idx].sum(1)
        tot_c += cnts[idx].sum(1)
    return tot_s / tot_c


def _jackknife(sums: np.ndarray, cnts: np.ndarray) -> np.ndarray:
    return (sums.sum() - sums) / (cnts.sum() - cnts)


def _bca_params(boot: np.ndarray, theta: float, sums: np.ndarray, cnts: np.ndarray) -> tuple[float, float]:
    B = len(boot)
    prop = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / B
    prop = min(max(prop, 1.0 / (B + 1)), B / (B + 1.0))
    z0 = _N.inv_cdf(prop)
    if len(sums) < 3:
        return z0, 0.0
    jk = _jackknife(sums, cnts)
    d = jk.mean() - jk
    den = 6.0 * (np.sum(d ** 2)) ** 1.5
    a = float(np.sum(d ** 3) / den) if den > 0 else 0.0
    return z0, a


def _bca_quantile(boot: np.ndarray, z0: float, a: float, level: float) -> float:
    z = _N.inv_cdf(level)
    adj = _N.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    return float(np.quantile(boot, adj))


def cluster_bootstrap(per_cluster: Mapping[Hashable, Sequence[float]], n: int = 10_000, seed: int = 0,
                      levels: Sequence[float] = (0.90, 0.95),
                      strata: Mapping[Hashable, Hashable] | None = None) -> dict:
    """Mean with percentile and BCa CIs at each confidence level.

    Returns {"mean", "n_clusters", "n_obs", "percentile": {lvl: (lo, hi)},
    "bca": {lvl: (lo, hi)}, "z0", "a", "boot" (np.ndarray)}."""
    keys, sums, cnts = _arrays(per_cluster)
    theta = float(sums.sum() / cnts.sum())
    st = np.array([strata[k] for k in keys]) if strata is not None else None
    boot = _boot_means(sums, cnts, n, np.random.default_rng(seed), st)
    z0, a = _bca_params(boot, theta, sums, cnts)
    pct, bca = {}, {}
    for lvl in levels:
        lo_q, hi_q = (1 - lvl) / 2, 1 - (1 - lvl) / 2
        pct[lvl] = (float(np.quantile(boot, lo_q)), float(np.quantile(boot, hi_q)))
        if np.ptp(boot) == 0:
            bca[lvl] = (theta, theta)
        else:
            bca[lvl] = (_bca_quantile(boot, z0, a, lo_q), _bca_quantile(boot, z0, a, hi_q))
    return {"mean": theta, "n_clusters": len(keys), "n_obs": int(cnts.sum()), "percentile": pct,
            "bca": bca, "z0": z0, "a": a, "boot": boot}


def _ci_pvalue(boot: np.ndarray, z0: float, a: float, value: float, side: str, method: str) -> float:
    """One-sided p-value from inverting the bootstrap CI.

    side="lower": H0 theta <= value (rejected when the lower bound exceeds value).
    side="upper": H0 theta >= value (rejected when the upper bound is below value)."""
    B = len(boot)
    q = (np.sum(boot < value) + 0.5 * np.sum(boot == value)) / B
    q = min(max(q, 1.0 / (B + 1)), B / (B + 1.0))
    if method == "percentile" or np.ptp(boot) == 0:
        return float(q if side == "lower" else 1 - q)
    w = _N.inv_cdf(q) - z0
    z = w / (1 + a * w) - z0          # solves z0 + (z0+z)/(1-a(z0+z)) = Phi^-1(q)
    return float(_N.cdf(z) if side == "lower" else 1 - _N.cdf(z))


def tost(per_cluster: Mapping[Hashable, Sequence[float]], margin: float, alpha: float = 0.05,
         n: int = 10_000, seed: int = 0, method: str = "bca",
         strata: Mapping[Hashable, Hashable] | None = None, boot: dict | None = None) -> dict:
    """Two one-sided tests for |mean| < margin, decided on the (1 - 2*alpha) CI."""
    if margin <= 0:
        raise ValueError("margin must be positive")
    if method not in ("bca", "percentile"):
        raise ValueError(method)
    lvl_eq, lvl_diff = 1 - 2 * alpha, 1 - alpha
    b = boot or cluster_bootstrap(per_cluster, n=n, seed=seed, levels=(lvl_eq, lvl_diff), strata=strata)
    ci_eq, ci_diff = b[method][lvl_eq], b[method][lvl_diff]
    equivalent = -margin < ci_eq[0] and ci_eq[1] < margin
    different = ci_diff[0] > 0 or ci_diff[1] < 0
    p_lower = _ci_pvalue(b["boot"], b["z0"], b["a"], -margin, "lower", method)
    p_upper = _ci_pvalue(b["boot"], b["z0"], b["a"], margin, "upper", method)
    verdict = "equivalent" if equivalent else ("different" if different else "inconclusive")
    label = "different_but_equivalent" if (equivalent and different) else verdict
    return {"mean": b["mean"], "margin": margin, "method": method, "ci90": ci_eq, "ci95": ci_diff,
            "verdict": verdict, "label": label, "equivalent": equivalent, "different": different,
            "p_lower": p_lower, "p_upper": p_upper, "p_tost": max(p_lower, p_upper),
            "p_tost_t": tost_t(per_cluster, margin)["p_tost"], "n_clusters": b["n_clusters"]}


# --------------------------------------------------------------------------- t-based TOST

def _betacf(a: float, b: float, x: float) -> float:
    # Continued fraction for the incomplete beta (Numerical Recipes, 3rd ed., 6.4).
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_cdf(t: float, df: float) -> float:
    """Student t CDF via the regularized incomplete beta function."""
    x = df / (df + t * t)
    tail = 0.5 * _betai(df / 2, 0.5, x)
    return 1 - tail if t > 0 else tail


def tost_t(per_cluster: Mapping[Hashable, Sequence[float]], margin: float) -> dict:
    """TOST with a cluster-robust (ratio-estimator) standard error and C-1 df."""
    _, sums, cnts = _arrays(per_cluster)
    C = len(sums)
    theta = sums.sum() / cnts.sum()
    if C < 2:
        return {"mean": float(theta), "se": float("nan"), "p_tost": float("nan")}
    resid = sums - theta * cnts
    se = math.sqrt(C / (C - 1) * float(np.sum(resid ** 2))) / cnts.sum()
    if se == 0:
        p = 0.0 if abs(theta) < margin else 1.0
        return {"mean": float(theta), "se": 0.0, "p_lower": p, "p_upper": p, "p_tost": p}
    p_lower = 1 - t_cdf((theta + margin) / se, C - 1)
    p_upper = t_cdf((theta - margin) / se, C - 1)
    return {"mean": float(theta), "se": se, "df": C - 1, "p_lower": p_lower, "p_upper": p_upper,
            "p_tost": max(p_lower, p_upper)}


# --------------------------------------------------------------------------- permutation

def sign_flip_test(per_cluster: Mapping[Hashable, Sequence[float]], n: int = 10_000, seed: int = 0) -> dict:
    """Two-sided paired permutation test; whole clusters flip sign together.

    Statistic: mean over questions. Exact enumeration when 2^C <= n, otherwise
    Monte Carlo with the (1 + #extreme) / (1 + n) correction."""
    _, sums, cnts = _arrays(per_cluster)
    C, N = len(sums), cnts.sum()
    obs = abs(sums.sum() / N)
    eps = 1e-12 * max(1.0, obs)
    if C <= 20 and 2 ** C <= n:
        signs = ((np.arange(2 ** C)[:, None] >> np.arange(C)) & 1) * 2 - 1
        stats = np.abs(signs @ sums) / N
        return {"p": float(np.mean(stats >= obs - eps)), "exact": True, "n_perm": int(2 ** C)}
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, n, 2000):
        m = min(2000, n - start)
        signs = rng.integers(0, 2, size=(m, C)) * 2 - 1
        hits += int(np.sum(np.abs(signs @ sums) / N >= obs - eps))
    return {"p": (1 + hits) / (1 + n), "exact": False, "n_perm": n}


# --------------------------------------------------------------------------- multiplicity, agreement

def holm(pvals: Sequence[float], alpha: float = 0.05) -> dict:
    """Holm step-down: adjusted p-values (in input order) and reject flags."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank_, i in enumerate(order):
        running = max(running, min(1.0, (m - rank_) * pvals[i]))
        adj[i] = running
    return {"p_adj": adj, "reject": [p <= alpha for p in adj]}


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable], labels: Sequence[Hashable] | None = None,
                weights: str | None = None) -> float:
    """Cohen's kappa for two raters. weights: None, "linear" or "quadratic"
    (weighted kappa needs `labels` in their natural order)."""
    if len(a) != len(b) or not a:
        raise ValueError("ratings must be non-empty and paired")
    labels = list(labels) if labels is not None else sorted(set(a) | set(b), key=str)
    ix = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[ix[x], ix[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / max(k - 1, 1)
    elif weights == "quadratic":
        w = ((i - j) / max(k - 1, 1)) ** 2
    else:
        raise ValueError(weights)
    de = float(np.sum(w * exp))
    if de == 0:
        return 1.0 if float(np.sum(w * obs)) == 0 else 0.0
    return 1.0 - float(np.sum(w * obs)) / de


def _ranks(x: Sequence[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):  # average ranks over ties
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        raise ValueError("unpaired")
    if len(x) < 3:
        return float("nan")
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])
