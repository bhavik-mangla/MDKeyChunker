"""RRF and Convex fusion against hand-computed values (RESEARCH.md sec. 4)."""
import pytest

from mdkeychunker.retrieval.fusion import RRF, Convex


def ids(r):
    return [i for i, _ in r]


A = [(0, 10.0), (1, 8.0), (2, 6.0)]
B = [(2, 0.9), (3, 0.8), (0, 0.1)]


def test_rrf_hand_computed():
    got = dict(RRF(k=60).fuse([A, B]))
    assert got[0] == pytest.approx(1 / 61 + 1 / 63)
    assert got[1] == pytest.approx(1 / 62)
    assert got[2] == pytest.approx(1 / 63 + 1 / 61)
    assert got[3] == pytest.approx(1 / 62)
    assert ids(RRF().fuse([A, B])) == [0, 2, 1, 3]


def test_rrf_ties_keep_first_appearance_order_like_the_study():
    # 0 and 2 tie, 1 and 3 tie: order is first appearance (list A, then list B)
    assert ids(RRF().fuse([A, B])) == [0, 2, 1, 3]
    assert ids(RRF().fuse([B, A])) == [2, 0, 3, 1]


def test_rrf_matches_study_function_on_random_lists():
    import random

    def study_rrf(a, b, k=60):  # benchmarks/chunking_study/evaluate_v2.rrf, verbatim
        score = {}
        for r_ in (a, b):
            for pos, idx in enumerate(r_):
                score[idx] = score.get(idx, 0.0) + 1.0 / (k + pos + 1)
        return sorted(score, key=lambda i: -score[i])

    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(1, 30)
        a, b = list(range(n)), list(range(n))
        rng.shuffle(a)
        rng.shuffle(b)
        b = b[: rng.randint(0, n)]
        got = ids(RRF().fuse([[(i, 0.0) for i in a], [(i, 0.0) for i in b]]))
        assert got == study_rrf(a, b)


def test_rrf_disjoint_lists_and_weights():
    got = RRF(k=0).fuse([[(0, 1.0)], [(1, 1.0)]])
    assert dict(got) == {0: 1.0, 1: 1.0} and ids(got) == [0, 1]
    got = RRF(k=0, weights=(1.0, 3.0)).fuse([[(0, 1.0)], [(1, 1.0)]])
    assert ids(got) == [1, 0] and dict(got)[1] == 3.0
    assert ids(RRF(weights=(0.0, 1.0)).fuse([A, B])) == [2, 3, 0, 1]


def test_rrf_single_list_and_empty():
    assert ids(RRF().fuse([A])) == [0, 1, 2]
    assert RRF().fuse([[], []]) == []
    assert RRF().fuse([]) == []


def test_rrf_validation():
    with pytest.raises(ValueError):
        RRF(k=-1)
    with pytest.raises(ValueError):
        RRF(weights=(1.0,)).fuse([A, B])
    with pytest.raises(ValueError):
        RRF(weights=(0.0, 0.0)).fuse([A, B])
    with pytest.raises(ValueError):
        RRF(weights=(-1.0, 2.0)).fuse([A, B])


def test_convex_minmax_hand_computed():
    # A minmax: 0->1, 1->0.5, 2->0; B minmax: 2->1, 3->0.875, 0->0; missing -> floor 0
    got = dict(Convex(alpha=0.5).fuse([A, B]))
    assert got[0] == pytest.approx(0.5 * 1 + 0.5 * 0)
    assert got[1] == pytest.approx(0.5 * 0.5 + 0.5 * 0)
    assert got[2] == pytest.approx(0.5 * 0 + 0.5 * 1)
    assert got[3] == pytest.approx(0.5 * 0 + 0.5 * 0.875)
    assert ids(Convex(alpha=0.0).fuse([A, B]))[0] == 0  # sparse only
    assert ids(Convex(alpha=1.0).fuse([A, B]))[:2] == [2, 3]  # dense only


def test_convex_zscore_hand_computed():
    a = [(0, 3.0), (1, 1.0)]       # z: +1, -1
    b = [(1, 2.0), (2, 0.0)]       # z: +1, -1
    got = dict(Convex(alpha=0.5, norm="zscore").fuse([a, b]))
    assert got[0] == pytest.approx(0.5 * 1 + 0.5 * -1)   # missing from b -> b's min z
    assert got[1] == pytest.approx(0.5 * -1 + 0.5 * 1)
    assert got[2] == pytest.approx(0.5 * -1 + 0.5 * -1)


def test_convex_degenerate_lists():
    flat = [(0, 5.0), (1, 5.0)]
    assert dict(Convex(norm="minmax").fuse([flat])) == {0: 1.0, 1: 1.0}
    assert dict(Convex(norm="zscore").fuse([flat])) == {0: 0.0, 1: 0.0}
    assert ids(Convex().fuse([A])) == [0, 1, 2]
    assert Convex().fuse([[], []]) == []
    # missing items rank below every present item, even when a list is flat
    # (item 1 is missing from the first list)
    assert dict(Convex(alpha=0.5).fuse([[(0, 1.0)], flat]))[1] == pytest.approx(0.5)
    z = dict(Convex(alpha=0.5, norm="zscore").fuse([[(0, 1.0)], flat]))
    assert z[1] == pytest.approx(-0.5) and z[0] > z[1]


def test_convex_weights_and_validation():
    three = [A, B, [(3, 1.0)]]
    with pytest.raises(ValueError, match="weights"):
        Convex().fuse(three)
    assert ids(Convex(weights=(0.0, 0.0, 1.0)).fuse(three))[0] == 3
    for bad in (dict(alpha=1.5), dict(alpha=-0.1), dict(norm="rank")):
        with pytest.raises(ValueError):
            Convex(**bad)
