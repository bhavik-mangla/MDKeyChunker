"""Paired statistics for retrieval comparisons, ported from the v4 study.

Source: ``benchmarks/chunking_study/stats.py`` (cluster bootstrap with BCa intervals,
sign-flip test, TOST, Holm) and ``evaluate_v2.compare`` (the paired comparison row).
Zero dependencies. When numpy is installed the bootstrap runs the study's numpy code
exactly, so it reproduces the paper's intervals and p-values; without numpy the same
algorithm runs on Python's ``random`` (same estimator, different Monte Carlo draws).

Inputs are per-cluster lists of per-question paired differences (A - B):
``{cluster_id: [d_q1, d_q2, ...]}``. A cluster is a document when questions are asked
within one document (Qasper: a paper), else a single question. The point estimate is
the mean over all questions (clusters weighted by size).

- ``cluster_bootstrap``: resample clusters with replacement (optionally within strata);
  percentile and BCa CIs, BCa acceleration from the leave-one-cluster-out jackknife
  (Efron & Tibshirani 1993, ch. 14).
- ``sign_flip_test``: two-sided paired permutation test flipping whole clusters (exact
  when 2^C <= n).
- ``tost``: equivalence within +/-margin, decided on the 90% CI (p_tost by inverting the
  same bootstrap CI), plus a cluster-robust t version.
- ``holm``: Holm-Bonferroni adjusted p-values.
- ``min_detectable_difference``: normal-theory MDE at the given power.
- ``compare``: one paired A vs B row as the study reported it.

Example:
    >>> d = {"p1": [0.1, 0.0], "p2": [0.2], "p3": [0.0, 0.1, 0.1], "p4": [0.3]}
    >>> b = cluster_bootstrap(d, n=2000, seed=0)
    >>> round(b.mean, 4), b.n_clusters, b.n_obs
    (0.1143, 4, 7)
    >>> sign_flip_test(d).exact
    True
"""
from __future__ import annotations

import importlib
import math
import random
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

__all__ = [
    "Bootstrap",
    "Comparison",
    "SignFlip",
    "Tost",
    "cluster_bootstrap",
    "compare",
    "holm",
    "min_detectable_difference",
    "paired",
    "sign_flip_test",
    "tost",
    "tost_t",
]

_N = NormalDist()
PerCluster = Mapping[Any, Sequence[float]]  # Any keys: Mapping is invariant in its key type


def _numpy() -> Any:
    try:
        return importlib.import_module("numpy")
    except ImportError:
        return None


def _arrays(per_cluster: PerCluster) -> tuple[list[Hashable], list[float], list[float]]:
    keys = list(per_cluster)
    if not keys:
        raise ValueError("no clusters")
    sums = [math.fsum(float(v) for v in per_cluster[k]) for k in keys]
    cnts = [float(len(per_cluster[k])) for k in keys]
    if any(c == 0 for c in cnts):
        raise ValueError("empty cluster")
    return keys, sums, cnts


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default 'linear' method)."""
    n = len(sorted_vals)
    h = (n - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    return float(sorted_vals[lo] + (h - lo) * (sorted_vals[hi] - sorted_vals[lo]))


# --------------------------------------------------------------------------- resampling

def _boot_means(sums: list[float], cnts: list[float], n: int, seed: int,
                strata: list[Hashable] | None, use_numpy: bool = False) -> list[float]:
    rnd = random.Random(seed)
    if strata is None:
        groups = [list(range(len(sums)))]
    else:
        by: dict[Hashable, list[int]] = {}
        for i, s in enumerate(strata):
            by.setdefault(s, []).append(i)
        groups = [by[s] for s in sorted(by, key=repr)]
    out = []
    for _ in range(n):
        ts = tc = 0.0
        for members in groups:
            for i in rnd.choices(members, k=len(members)):
                ts += sums[i]
                tc += cnts[i]
        out.append(ts / tc)
    return out


def _jackknife(sums: list[float], cnts: list[float]) -> list[float]:
    ts, tc = math.fsum(sums), math.fsum(cnts)
    return [(ts - s) / (tc - c) for s, c in zip(sums, cnts)]


def _bca_params(boot: list[float], theta: float, sums: list[float],
                cnts: list[float]) -> tuple[float, float]:
    b = len(boot)
    tol = 1e-12 * max(1.0, abs(theta))
    below = sum(x < theta - tol for x in boot)
    ties = sum(abs(x - theta) <= tol for x in boot)
    prop = (below + 0.5 * ties) / b
    prop = min(max(prop, 1.0 / (b + 1)), b / (b + 1.0))
    z0 = _N.inv_cdf(prop)
    if len(sums) < 3:
        return z0, 0.0
    jk = _jackknife(sums, cnts)
    m = math.fsum(jk) / len(jk)
    d = [m - x for x in jk]
    den = 6.0 * math.fsum(x * x for x in d) ** 1.5
    a = math.fsum(x ** 3 for x in d) / den if den > 0 else 0.0
    return z0, a


def _bca_quantile(sorted_boot: list[float], z0: float, a: float, level: float) -> float:
    z = _N.inv_cdf(level)
    adj = _N.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    return _quantile(sorted_boot, adj)


@dataclass
class Bootstrap:
    """Result of ``cluster_bootstrap``: mean, CIs per level, BCa parameters, the draws."""

    mean: float
    n_clusters: int
    n_obs: int
    percentile: dict[float, tuple[float, float]]
    bca: dict[float, tuple[float, float]]
    z0: float
    a: float
    boot: list[float] = field(repr=False)

    @property
    def se(self) -> float:
        """Bootstrap standard error of the mean."""
        m = math.fsum(self.boot) / len(self.boot)
        return math.sqrt(math.fsum((x - m) ** 2 for x in self.boot) / (len(self.boot) - 1))


def _bootstrap_numpy(np: Any, per_cluster: PerCluster, n: int, seed: int,
                     levels: Sequence[float], strata: Mapping[Hashable, Hashable] | None
                     ) -> Bootstrap:
    """The study's cluster_bootstrap, operation for operation (so results are identical)."""
    keys = list(per_cluster)
    sums = np.array([float(np.sum(per_cluster[k])) for k in keys])
    cnts = np.array([len(per_cluster[k]) for k in keys], dtype=float)
    theta = float(sums.sum() / cnts.sum())
    rng = np.random.default_rng(seed)
    c = len(sums)
    if strata is None:
        idx = rng.integers(0, c, size=(n, c))
        boot = sums[idx].sum(1) / cnts[idx].sum(1)
    else:
        st = np.array([strata[k] for k in keys])
        tot_s, tot_c = np.zeros(n), np.zeros(n)
        for s in np.unique(st):
            members = np.flatnonzero(st == s)
            idx = members[rng.integers(0, len(members), size=(n, len(members)))]
            tot_s += sums[idx].sum(1)
            tot_c += cnts[idx].sum(1)
        boot = tot_s / tot_c
    prop = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / n
    prop = min(max(prop, 1.0 / (n + 1)), n / (n + 1.0))
    z0 = _N.inv_cdf(float(prop))
    a = 0.0
    if c >= 3:
        jk = (sums.sum() - sums) / (cnts.sum() - cnts)
        d = jk.mean() - jk
        den = 6.0 * (np.sum(d ** 2)) ** 1.5
        a = float(np.sum(d ** 3) / den) if den > 0 else 0.0
    pct: dict[float, tuple[float, float]] = {}
    bca: dict[float, tuple[float, float]] = {}
    for lvl in levels:
        lo_q, hi_q = (1 - lvl) / 2, 1 - (1 - lvl) / 2
        pct[lvl] = (float(np.quantile(boot, lo_q)), float(np.quantile(boot, hi_q)))
        if np.ptp(boot) == 0:
            bca[lvl] = (theta, theta)
        else:
            adj = [_N.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
                   for z in (_N.inv_cdf(lo_q), _N.inv_cdf(hi_q))]
            bca[lvl] = (float(np.quantile(boot, adj[0])), float(np.quantile(boot, adj[1])))
    return Bootstrap(mean=theta, n_clusters=c, n_obs=int(cnts.sum()), percentile=pct, bca=bca,
                     z0=z0, a=a, boot=[float(x) for x in boot])


def cluster_bootstrap(per_cluster: PerCluster, n: int = 10_000, seed: int = 0,
                      levels: Sequence[float] = (0.90, 0.95),
                      strata: Mapping[Hashable, Hashable] | None = None,
                      use_numpy: bool = True) -> Bootstrap:
    """Mean with percentile and BCa CIs at each confidence level.

    Args:
        per_cluster: ``{cluster: [value, ...]}`` (typically paired differences).
        n: bootstrap resamples (study: 10,000).
        seed: random seed.
        levels: confidence levels.
        strata: optional ``{cluster: stratum}``; clusters are resampled within strata
            (the study stratified FreshStack by domain).
        use_numpy: when numpy is installed, run the study's numpy code exactly (identical
            results). The pure-Python path counts resamples within 1e-12 of the estimate
            as ties when estimating the BCa bias, so rounding cannot shift it.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    keys, sums, cnts = _arrays(per_cluster)
    np = _numpy() if use_numpy else None
    if np is not None:
        return _bootstrap_numpy(np, per_cluster, n, seed, levels, strata)
    theta = math.fsum(sums) / math.fsum(cnts)
    st = [strata[k] for k in keys] if strata is not None else None
    boot = _boot_means(sums, cnts, n, seed, st, use_numpy=False)
    z0, a = _bca_params(boot, theta, sums, cnts)
    sb = sorted(boot)
    flat = sb[0] == sb[-1]
    pct: dict[float, tuple[float, float]] = {}
    bca: dict[float, tuple[float, float]] = {}
    for lvl in levels:
        lo_q, hi_q = (1 - lvl) / 2, 1 - (1 - lvl) / 2
        pct[lvl] = (_quantile(sb, lo_q), _quantile(sb, hi_q))
        bca[lvl] = (theta, theta) if flat else (_bca_quantile(sb, z0, a, lo_q),
                                                _bca_quantile(sb, z0, a, hi_q))
    return Bootstrap(mean=theta, n_clusters=len(keys), n_obs=int(sum(cnts)), percentile=pct,
                     bca=bca, z0=z0, a=a, boot=boot)


# --------------------------------------------------------------------------- TOST

def _ci_pvalue(boot: list[float], z0: float, a: float, value: float, side: str,
               method: str) -> float:
    """One-sided p-value from inverting the bootstrap CI.

    side="lower": H0 theta <= value; side="upper": H0 theta >= value."""
    b = len(boot)
    q = (sum(x < value for x in boot) + 0.5 * sum(x == value for x in boot)) / b
    q = min(max(q, 1.0 / (b + 1)), b / (b + 1.0))
    if method == "percentile" or min(boot) == max(boot):
        return float(q if side == "lower" else 1 - q)
    w = _N.inv_cdf(q) - z0
    z = w / (1 + a * w) - z0  # solves z0 + (z0+z)/(1-a(z0+z)) = Phi^-1(q)
    return float(_N.cdf(z) if side == "lower" else 1 - _N.cdf(z))


@dataclass
class Tost:
    """Two one-sided tests for ``|mean| < margin``."""

    mean: float
    margin: float
    method: str
    ci90: tuple[float, float]
    ci95: tuple[float, float]
    verdict: str          # "equivalent", "different" or "inconclusive"
    label: str            # verdict, or "different_but_equivalent"
    equivalent: bool
    different: bool
    p_lower: float
    p_upper: float
    p_tost: float
    p_tost_t: float
    n_clusters: int


def tost(per_cluster: PerCluster, margin: float, alpha: float = 0.05, n: int = 10_000,
         seed: int = 0, method: str = "bca", strata: Mapping[Hashable, Hashable] | None = None,
         boot: Bootstrap | None = None) -> Tost:
    """Equivalence within ``+/-margin``, decided on the (1 - 2*alpha) CI.

    Verdict (the study's pre-declared language): equivalent if the 90% CI lies inside
    (-margin, +margin); different if the 95% CI excludes 0; inconclusive otherwise.
    Both can hold: the verdict is then "equivalent" and ``label`` says so.
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    if method not in ("bca", "percentile"):
        raise ValueError(method)
    lvl_eq, lvl_diff = 1 - 2 * alpha, 1 - alpha
    b = boot or cluster_bootstrap(per_cluster, n=n, seed=seed, levels=(lvl_eq, lvl_diff),
                                  strata=strata)
    cis = b.bca if method == "bca" else b.percentile
    ci_eq, ci_diff = cis[lvl_eq], cis[lvl_diff]
    equivalent = -margin < ci_eq[0] and ci_eq[1] < margin
    different = ci_diff[0] > 0 or ci_diff[1] < 0
    p_lower = _ci_pvalue(b.boot, b.z0, b.a, -margin, "lower", method)
    p_upper = _ci_pvalue(b.boot, b.z0, b.a, margin, "upper", method)
    verdict = "equivalent" if equivalent else ("different" if different else "inconclusive")
    label = "different_but_equivalent" if (equivalent and different) else verdict
    return Tost(mean=b.mean, margin=margin, method=method, ci90=ci_eq, ci95=ci_diff,
                verdict=verdict, label=label, equivalent=equivalent, different=different,
                p_lower=p_lower, p_upper=p_upper, p_tost=max(p_lower, p_upper),
                p_tost_t=tost_t(per_cluster, margin)["p_tost"], n_clusters=b.n_clusters)


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
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x)
           + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_cdf(t: float, df: float) -> float:
    """Student t CDF via the regularized incomplete beta function."""
    x = df / (df + t * t)
    tail = 0.5 * _betai(df / 2, 0.5, x)
    return 1 - tail if t > 0 else tail


def _cluster_se(sums: list[float], cnts: list[float]) -> tuple[float, float]:
    """Mean and cluster-robust (ratio-estimator) standard error."""
    c = len(sums)
    theta = math.fsum(sums) / math.fsum(cnts)
    if c < 2:
        return theta, float("nan")
    resid = [s - theta * n for s, n in zip(sums, cnts)]
    return theta, math.sqrt(c / (c - 1) * math.fsum(r * r for r in resid)) / math.fsum(cnts)


def tost_t(per_cluster: PerCluster, margin: float) -> dict[str, float]:
    """TOST with a cluster-robust (ratio-estimator) standard error and C-1 df."""
    _, sums, cnts = _arrays(per_cluster)
    c = len(sums)
    theta, se = _cluster_se(sums, cnts)
    if c < 2:
        return {"mean": theta, "se": float("nan"), "p_tost": float("nan")}
    if se == 0:
        p = 0.0 if abs(theta) < margin else 1.0
        return {"mean": theta, "se": 0.0, "p_lower": p, "p_upper": p, "p_tost": p}
    p_lower = 1 - t_cdf((theta + margin) / se, c - 1)
    p_upper = t_cdf((theta - margin) / se, c - 1)
    return {"mean": theta, "se": se, "df": float(c - 1), "p_lower": p_lower,
            "p_upper": p_upper, "p_tost": max(p_lower, p_upper)}


# --------------------------------------------------------------------------- permutation

@dataclass
class SignFlip:
    p: float
    exact: bool
    n_perm: int


def sign_flip_test(per_cluster: PerCluster, n: int = 10_000, seed: int = 0,
                   use_numpy: bool = True) -> SignFlip:
    """Two-sided paired permutation test; whole clusters flip sign together.

    Statistic: mean over questions. Exact enumeration when 2^C <= n (and C <= 20),
    otherwise Monte Carlo with the (1 + #extreme) / (1 + n) correction.
    """
    _, sums, cnts = _arrays(per_cluster)
    c, total = len(sums), math.fsum(cnts)
    obs = abs(math.fsum(sums) / total)
    eps = 1e-12 * max(1.0, obs)
    if c <= 20 and 2 ** c <= n:
        hits = 0
        for mask in range(2 ** c):
            s = math.fsum(v if (mask >> i) & 1 else -v for i, v in enumerate(sums))
            hits += abs(s) / total >= obs - eps
        return SignFlip(p=hits / 2 ** c, exact=True, n_perm=2 ** c)
    np = _numpy() if use_numpy else None
    hits = 0
    if np is not None:  # identical draws to the study
        rng = np.random.default_rng(seed)
        s_arr = np.array(sums)
        for start in range(0, n, 2000):
            m = min(2000, n - start)
            signs = rng.integers(0, 2, size=(m, c)) * 2 - 1
            hits += int(np.sum(np.abs(signs @ s_arr) / total >= obs - eps))
    else:
        rnd = random.Random(seed)
        for _ in range(n):
            s = math.fsum(v if rnd.random() < 0.5 else -v for v in sums)
            hits += abs(s) / total >= obs - eps
    return SignFlip(p=(1 + hits) / (1 + n), exact=False, n_perm=n)


# --------------------------------------------------------------------------- multiplicity, MDE

def holm(pvals: Sequence[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Holm step-down: adjusted p-values (in input order) and reject flags.

    Example:
        >>> holm([0.01, 0.04, 0.03])
        ([0.03, 0.06, 0.06], [True, False, False])
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank_, i in enumerate(order):
        running = max(running, min(1.0, (m - rank_) * pvals[i]))
        adj[i] = running
    return adj, [p <= alpha for p in adj]


def min_detectable_difference(per_cluster: PerCluster, alpha: float = 0.05,
                              power: float = 0.80) -> float:
    """Smallest true difference a two-sided test at ``alpha`` detects with ``power``.

    Normal theory with the cluster-robust standard error of the mean difference:
    ``(z_{1-alpha/2} + z_power) * se`` (about 2.8 * se at the defaults). An approximate
    guide to what the data can resolve, not a test. NaN with fewer than two clusters.

    Example:
        >>> round(min_detectable_difference({"a": [0.1], "b": [0.3], "c": [-0.1]}), 2)
        0.32
    """
    _, sums, cnts = _arrays(per_cluster)
    _, se = _cluster_se(sums, cnts)
    return (_N.inv_cdf(1 - alpha / 2) + _N.inv_cdf(power)) * se


# --------------------------------------------------------------------------- comparison row

def paired(pa: PerCluster, pb: PerCluster) -> dict[Hashable, list[float]]:
    """Per-cluster paired differences A - B on the clusters both arms have.

    Clusters whose question counts differ between the arms are dropped (as in the study).
    """
    return {k: [float(x) - float(y) for x, y in zip(pa[k], pb[k])]
            for k in pa if k in pb and len(pa[k]) == len(pb[k])}


@dataclass
class Comparison:
    """A paired A - B comparison, as ``evaluate_v2.compare`` reported it."""

    mean_a: float
    mean_b: float
    diff: float
    n_clusters: int
    n_obs: int
    ci90_bca: tuple[float, float]
    ci95_bca: tuple[float, float]
    ci90_pct: tuple[float, float]
    ci95_pct: tuple[float, float]
    p_perm: float
    perm_exact: bool
    excludes_zero_95: bool
    mde: float
    verdict: str  # "different" / "no_difference_detected", or the TOST verdict with a margin
    margin: float | None = None
    label: str | None = None
    p_tost: float | None = None
    p_tost_t: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def compare(pa: PerCluster, pb: PerCluster, *, margin: float | None = None,
            strata: Mapping[Hashable, Hashable] | None = None, n: int = 10_000, seed: int = 0,
            use_numpy: bool = True) -> Comparison:
    """Paired A - B on the clusters both arms have (same per-cluster question counts).

    With ``margin`` the verdict is the TOST verdict (equivalent / different /
    inconclusive); without, "different" when the 95% BCa CI excludes 0, else
    "no_difference_detected". No multiplicity adjustment is applied here.

    Example:
        >>> a = {"d1": [0.5, 0.6], "d2": [0.7], "d3": [0.4]}
        >>> b = {"d1": [0.4, 0.5], "d2": [0.5], "d3": [0.4]}
        >>> c = compare(a, b, n=2000)
        >>> round(c.diff, 3), c.n_clusters
        (0.1, 3)
    """
    keys = [k for k in pa if k in pb and len(pa[k]) == len(pb[k])]
    if not keys:
        raise ValueError("the two arms share no cluster with matching question counts")
    d = paired(pa, pb)
    st = {k: strata[k] for k in keys} if strata is not None else None
    b = cluster_bootstrap(d, n=n, seed=seed, strata=st, use_numpy=use_numpy)
    perm = sign_flip_test(d, n=n, seed=seed, use_numpy=use_numpy)
    mean_a = math.fsum(float(v) for k in keys for v in pa[k]) / b.n_obs
    mean_b = math.fsum(float(v) for k in keys for v in pb[k]) / b.n_obs
    excl = b.bca[0.95][0] > 0 or b.bca[0.95][1] < 0
    row = Comparison(mean_a=mean_a, mean_b=mean_b, diff=b.mean, n_clusters=b.n_clusters,
                     n_obs=b.n_obs, ci90_bca=b.bca[0.90], ci95_bca=b.bca[0.95],
                     ci90_pct=b.percentile[0.90], ci95_pct=b.percentile[0.95],
                     p_perm=perm.p, perm_exact=perm.exact, excludes_zero_95=excl,
                     mde=min_detectable_difference(d),
                     verdict="different" if excl else "no_difference_detected")
    if margin:
        t = tost(d, margin=margin, boot=b)
        row.margin, row.verdict, row.label = margin, t.verdict, t.label
        row.p_tost, row.p_tost_t = t.p_tost, t.p_tost_t
    return row
