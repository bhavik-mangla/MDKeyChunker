"""mdkeychunker.stats: hand-checked cases, parity with the study's stats.py, and the paper's
primary confirmatory rows reproduced from the committed per-question results."""
import json
import random
import sys
from pathlib import Path

import pytest

from mdkeychunker import stats as mk

STUDY = Path(__file__).resolve().parents[1] / "benchmarks" / "chunking_study"
RESULTS = STUDY / "results" / "v2"


def rand_clusters(seed, n_clusters=None, strata=False):
    rng = random.Random(seed)
    c = n_clusters or rng.randint(3, 40)
    d = {f"c{i}": [rng.gauss(0.03, 0.2) for _ in range(rng.randint(1, 5))] for i in range(c)}
    st = {k: f"s{i % 3}" for i, k in enumerate(d)} if strata else None
    return d, st


# --------------------------------------------------------------------------- hand-checked

def test_holm_hand_values():
    adj, rej = mk.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06]) and rej == [True, False, False]
    assert mk.holm([]) == ([], [])
    assert mk.holm([0.9, 0.9])[0] == [1.0, 1.0]


def test_sign_flip_exact_small_case():
    # 3 clusters, all positive: only the all-+ and all-- flips are as extreme -> p = 2/8
    r = mk.sign_flip_test({"a": [1.0], "b": [1.0], "c": [1.0]}, use_numpy=False)
    assert (r.p, r.exact, r.n_perm) == (0.25, True, 8)
    assert mk.sign_flip_test({"a": [0.0], "b": [0.0]}).p == 1.0


def test_sign_flip_monte_carlo_when_too_many_clusters():
    d, _ = rand_clusters(1, n_clusters=30)
    r = mk.sign_flip_test(d, n=500, use_numpy=False)
    assert not r.exact and r.n_perm == 500 and 1 / 501 <= r.p <= 1.0


def test_bootstrap_mean_is_question_weighted_and_constant_data_is_degenerate():
    b = mk.cluster_bootstrap({"a": [1.0, 1.0, 1.0], "b": [0.0]}, n=200, use_numpy=False)
    assert b.mean == 0.75 and (b.n_clusters, b.n_obs) == (2, 4)
    flat = mk.cluster_bootstrap({"a": [0.2], "b": [0.2], "c": [0.2]}, n=200, use_numpy=False)
    assert flat.bca[0.95] == pytest.approx((0.2, 0.2))
    assert flat.percentile[0.95] == pytest.approx((0.2, 0.2))
    with pytest.raises(ValueError):
        mk.cluster_bootstrap({}, n=10)
    with pytest.raises(ValueError):
        mk.cluster_bootstrap({"a": []}, n=10)


def test_tost_verdicts():
    rng = random.Random(0)
    tight = {f"c{i}": [rng.gauss(0.0, 0.01)] for i in range(80)}
    assert mk.tost(tight, margin=0.05, n=2000).verdict == "equivalent"
    big = {f"c{i}": [0.3 + rng.gauss(0.0, 0.01)] for i in range(80)}
    t = mk.tost(big, margin=0.05, n=2000)
    assert t.verdict == "different" and t.p_tost > 0.5
    noisy = {f"c{i}": [rng.gauss(0.0, 0.5)] for i in range(6)}
    assert mk.tost(noisy, margin=0.05, n=2000).verdict == "inconclusive"
    with pytest.raises(ValueError):
        mk.tost(tight, margin=0.0)


def test_mde_shrinks_with_more_clusters_and_matches_bootstrap_se():
    small, _ = rand_clusters(3, n_clusters=10)
    large, _ = rand_clusters(3, n_clusters=200)
    assert mk.min_detectable_difference(large) < mk.min_detectable_difference(small)
    b = mk.cluster_bootstrap(large, n=4000, use_numpy=False)
    assert mk.min_detectable_difference(large) == pytest.approx(2.8016 * b.se, rel=0.1)
    assert mk.min_detectable_difference({"a": [0.1]}) != mk.min_detectable_difference({"a": [0.1]})


def test_compare_pairs_on_common_clusters_only():
    a = {"d1": [0.5, 0.6], "d2": [0.7], "d3": [0.4], "only_a": [1.0], "len": [1.0, 1.0]}
    b = {"d1": [0.4, 0.5], "d2": [0.5], "d3": [0.4], "len": [0.0]}
    c = mk.compare(a, b, n=1000, use_numpy=False)
    assert (c.n_clusters, c.n_obs) == (3, 4) and c.diff == pytest.approx(0.1)
    assert c.mean_a == pytest.approx(0.55) and c.mean_b == pytest.approx(0.45)
    assert c.verdict in ("different", "no_difference_detected") and c.margin is None
    with pytest.raises(ValueError):
        mk.compare({"x": [1.0]}, {"y": [1.0]})


def test_pure_python_path_works_without_numpy(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", None)
    d, st = rand_clusters(5, n_clusters=25, strata=True)
    b = mk.cluster_bootstrap(d, n=500, strata=st)
    lo, hi = b.bca[0.95]
    assert lo < b.mean < hi
    assert not mk.sign_flip_test(d, n=300).exact


# --------------------------------------------------------------------------- study parity

@pytest.fixture(scope="module")
def study_stats():
    pytest.importorskip("numpy")
    sys.path.insert(0, str(STUDY))
    try:
        import stats as study
    finally:
        sys.path.remove(str(STUDY))
    return study


@pytest.mark.parametrize("seed", range(8))
def test_numpy_path_matches_study_stats(study_stats, seed):
    d, st = rand_clusters(seed, strata=seed % 2 == 1)
    want = study_stats.cluster_bootstrap(d, n=3000, seed=seed, strata=st)
    got = mk.cluster_bootstrap(d, n=3000, seed=seed, strata=st)
    assert got.mean == pytest.approx(want["mean"], abs=1e-12)
    assert (got.z0, got.a) == pytest.approx((want["z0"], want["a"]), abs=1e-9)
    for lvl in (0.90, 0.95):
        assert got.bca[lvl] == pytest.approx(want["bca"][lvl], abs=1e-10)
        assert got.percentile[lvl] == pytest.approx(want["percentile"][lvl], abs=1e-10)
    for n in (64, 3000):  # exact enumeration and Monte Carlo
        assert mk.sign_flip_test(d, n=n, seed=seed).p == study_stats.sign_flip_test(d, n=n,
                                                                                    seed=seed)["p"]
    t_w = study_stats.tost(d, margin=0.05, n=3000, seed=seed, strata=st)
    t_g = mk.tost(d, margin=0.05, n=3000, seed=seed, strata=st)
    assert (t_g.p_tost, t_g.p_tost_t) == pytest.approx((t_w["p_tost"], t_w["p_tost_t"]), abs=1e-9)
    assert t_g.label == t_w["label"]
    assert mk.holm([0.2, 0.01, 0.04])[0] == study_stats.holm([0.2, 0.01, 0.04])["p_adj"]


@pytest.mark.parametrize("seed", range(4))
def test_pure_python_path_agrees_with_study_within_monte_carlo_error(study_stats, seed):
    d, st = rand_clusters(seed, n_clusters=60, strata=seed % 2 == 1)
    want = study_stats.cluster_bootstrap(d, n=10_000, seed=seed, strata=st)
    got = mk.cluster_bootstrap(d, n=10_000, seed=seed, strata=st, use_numpy=False)
    assert got.mean == pytest.approx(want["mean"], abs=1e-12)
    assert got.a == pytest.approx(want["a"], abs=1e-9)  # jackknife: deterministic
    se = got.se  # Monte Carlo sd of a 10k-draw BCa bound is ~0.04 se; allow ~4 sd of the difference
    for lvl in (0.90, 0.95):
        assert got.bca[lvl] == pytest.approx(want["bca"][lvl], abs=0.25 * se)


# --------------------------------------------------------------------------- the paper's rows

PRIMARY = [("qasper", "rec@512t", 0.05), ("freshstack", "prec@1024t", 0.04)]


def _primary_rows():
    rows = json.loads((RESULTS / "stats.json").read_text(encoding="utf-8"))["primary"]
    return {(r["dataset"], r["a"], r["b"]): r for r in rows}


def _per_question(ds):
    raw = json.loads((RESULTS / f"{ds}_per_question.json").read_text(encoding="utf-8"))
    return raw["values"], raw.get("strata")


@pytest.mark.parametrize("ds,metric,margin", PRIMARY)
@pytest.mark.parametrize("b", ["S+T", "CR"])
def test_reproduces_paper_primary_rows_with_numpy(ds, metric, margin, b):
    pytest.importorskip("numpy")
    want = _primary_rows()[(ds, "E", b)]
    vals, strata = _per_question(ds)
    got = mk.compare(vals[f"hybrid|E|{metric}"], vals[f"hybrid|{b}|{metric}"], margin=margin,
                     strata=strata, n=10_000, seed=0)
    assert got.diff == pytest.approx(want["diff"], abs=1e-12)
    assert got.ci95_bca == pytest.approx(want["ci95_bca"], abs=1e-9)
    assert got.ci90_bca == pytest.approx(want["ci90_bca"], abs=1e-9)
    assert got.p_perm == pytest.approx(want["p_perm"], abs=1e-12)
    assert got.p_tost == pytest.approx(want["p_tost"], abs=1e-9)
    assert got.n_clusters == want["n_clusters"]
    assert got.verdict == want["verdict_ci"]  # the TOST verdict before the family's Holm step


def test_paper_headline_row_without_numpy(monkeypatch):
    """Enrichment vs free prefix on Qasper (+5.5 points; confirmatory) from pure Python."""
    want = _primary_rows()[("qasper", "E", "S+T")]
    vals, _ = _per_question("qasper")
    monkeypatch.setitem(sys.modules, "numpy", None)
    got = mk.compare(vals["hybrid|E|rec@512t"], vals["hybrid|S+T|rec@512t"], margin=0.05,
                     n=10_000, seed=0)
    assert got.diff == pytest.approx(want["diff"], abs=1e-12)
    assert got.ci95_bca == pytest.approx(want["ci95_bca"], abs=0.003)
    assert got.verdict == "different" and want["excludes_zero_95"]
