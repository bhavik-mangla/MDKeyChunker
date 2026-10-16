"""Exploratory (post hoc) analyses for paper v4, computed from existing results and caches only.

EVERYTHING IN THIS FILE IS EXPLORATORY. It was written after every confirmatory, secondary and
mechanism result had been viewed. No build, retrieval run, LLM call or embedding call is made:
inputs are the per-question result dumps under results/v2 and the chunk caches under data/v2.
Statistics reuse the harness (evaluate_v2.compare: paired, cluster bootstrap by paper on Qasper and
by question within domain strata on FreshStack, 10,000 resamples, BCa CIs, sign-flip p). Every
analysis that is run is written out, including nulls.

Sections of results/v2/analysis_v4.json
  lexical       question/gold vocabulary overlap; E - S+T by overlap tertile per retriever; tokens
                that enrichment adds per chunk; "bridging" (question terms absent from the gold
                chunk's S+T text but present in its E metadata) and E - S+T by bridging
  frontier      means of every arm x retriever, the qwen3 / reranker / enrichment paired
                contrasts, indexing LLM tokens per Stage-1 chunk, and the Pareto frontier over
                (indexing tokens, query-time cost class, score)
  heterogeneity Qasper by answer type, evidence count and paper length; FreshStack by domain and
                gold length; win/tie/loss rates; answer F1 by type and conditional on the recall
                change
  budgets       E - S+T, E - CR, S - F256t at every budget / rank metric under five retrievers
  late_chunking per-paper LC-S - S loss against window count, chunk count and cosine collapse
  granularity   mean chunk length against score over arms; merge benefit by gold length
  keys          rolling-dictionary reconstruction: reuse distance, LRU evictions, keys and fields
                found verbatim in the chunk text
  audit         recomputed descriptive counts quoted in the paper

Run:  HF_HUB_OFFLINE=1 .venv-ml/bin/python analysis_v4.py [--boot 10000]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import build_v2
import evaluate
import evaluate_corpus
import evaluate_v2 as ev
import stats

ROOT = Path(__file__).parent
V2R = ROOT / "results" / "v2"
V2D = ROOT / "data" / "v2"
CACHE = build_v2.CACHE_V2
OUT = V2R / "analysis_v4.json"
LABEL = "EXPLORATORY (post hoc, analysis_v4.py); unadjusted 95% BCa CIs"

PM = ev.PRIMARY_METRIC
DATASETS = ev.DATASETS
SEED = 0
BOOT = 10_000

# scikit-learn 1.9.1 ENGLISH_STOP_WORDS, copied so the script runs in either venv
STOP = frozenset("""a about above across after afterwards again against all almost alone along already also although
always am among amongst amoungst amount an and another any anyhow anyone anything anyway anywhere are around as at
back be became because become becomes becoming been before beforehand behind being below beside besides between
beyond bill both bottom but by call can cannot cant co con could couldnt cry de describe detail do done down due
during each eg eight either eleven else elsewhere empty enough etc even ever every everyone everything everywhere
except few fifteen fifty fill find fire first five for former formerly forty found four from front full further
get give go had has hasnt have he hence her here hereafter hereby herein hereupon hers herself him himself his how
however hundred i ie if in inc indeed interest into is it its itself keep last latter latterly least less ltd made
many may me meanwhile might mill mine more moreover most mostly move much must my myself name namely neither never
nevertheless next nine no nobody none noone nor not nothing now nowhere of off often on once one only onto or
other others otherwise our ours ourselves out over own part per perhaps please put rather re same see seem seemed
seeming seems serious several she should show side since sincere six sixty so some somehow someone something
sometime sometimes somewhere still such system take ten than that the their them themselves then thence there
thereafter thereby therefore therein thereupon these they thick thin third this those though three through
throughout thru thus to together too top toward towards twelve twenty two un under until up upon us very via was
we well were what whatever when whence whenever where whereafter whereas whereby wherein whereupon wherever whether
which while whither who whoever whole whom whose why will with within without would yet you your yours yourself
yourselves""".split())

# Arms whose chunk caches exist: label -> (model tag, variant, rendering). RM / RM-TP average seeds.
ARM_CACHE = {**ev.SYSTEMS_V2,
             "M": ("local", "merged", "meta"), "SM": ("local", "smerge", "meta"),
             "E-noRK": ("local", "enr_nork", "meta"), "M-noRK": ("local", "merged_nork", "meta"),
             "M-TP": ("local", "merged_tp", "meta"), "SM-TP": ("local", "smerge_tp", "meta")}
SEEDED = {"RM": [("local", f"rmerge_s{s}", "meta") for s in range(5)],
          "RM-TP": [("local", f"rmerge_tp_s{s}", "meta") for s in range(5)]}
TEXT_FAMILY = ["F512c", "F256t", "FL", "PP", "S"]
META_FAMILY = ["E", "E-noRK", "M", "SM", "RM", "M-noRK", "M-TP", "SM-TP", "RM-TP"]


def content_tokens(s: str) -> set:
    return {w for w in evaluate.words(s) if w not in STOP and len(w) > 1}


# =========================================================================== loading

class Results:
    """All per-question dumps; get() looks a (retriever, arm, metric) up in a fixed source order,
    so the confirmatory files win for the shared arms (as in paper_numbers.write_grids)."""

    FILES = {"qasper": ["qasper_per_question.json", "extra/qasper_per_question.json", "h4_qasper_per_question.json",
                        "h5_qasper_per_question.json", "dial_qasper_per_question.json",
                        "qasper_e2e_per_question.json"],
             "freshstack": ["freshstack_per_question.json", "extra/freshstack_per_question.json",
                            "h4_freshstack_per_question.json", "h5_freshstack_per_question.json",
                            "dial_freshstack_per_question.json"]}

    def __init__(self):
        self.src: dict = {}
        self.strata: dict = {}
        self.qids: dict = {}
        for ds, files in self.FILES.items():
            self.src[ds] = []
            for f in files:
                p = V2R / f
                if p.exists():
                    d = ev.load_per_question(p)
                    self.src[ds].append((f, d["per"]))
                    if f == files[0]:
                        self.qids[ds] = d["question_ids"]
                        self.strata[ds] = d.get("strata")

    def get(self, ds: str, r: str, arm: str, metric: str):
        for _, per in self.src[ds]:
            if (r, arm, metric) in per:
                return per[(r, arm, metric)]
        return None

    def keys(self, ds: str) -> set:
        return {k for _, per in self.src[ds] for k in per}


R: Results
QROWS: dict = {}  # ds -> list of question feature dicts in canonical (cluster, idx) order


def qkey_order(ds: str) -> list[tuple[str, int]]:
    return [(c, i) for c, qs in R.qids[ds].items() for i in range(len(qs))]


def by_q(per: dict) -> dict:
    return {(c, i): v for c, vs in per.items() for i, v in enumerate(vs)}


def sub(per: dict, keep: set | None) -> dict:
    if keep is None:
        return per
    out = {}
    for c, vs in per.items():
        x = [v for i, v in enumerate(vs) if (c, i) in keep]
        if x:
            out[c] = x
    return out


def cmp(ds: str, pa: dict | None, pb: dict | None, keep: set | None = None, margin: float | None = None) -> dict:
    if not pa or not pb:
        return {"status": "not_available"}
    a, b = sub(pa, keep), sub(pb, keep)
    st = R.strata[ds] if ds == "freshstack" else None
    r = ev.compare(a, b, margin=margin, strata=st, n=BOOT, seed=SEED)
    if r.get("status") != "ok":
        return r
    keep_f = ("status", "mean_a", "mean_b", "diff", "ci95_bca", "ci90_bca", "p_perm", "n_clusters", "n_obs",
              "excludes_zero_95", "verdict", "label", "margin", "p_tost")
    return {k: r[k] for k in keep_f if k in r}


def cmp_arms(ds, r, a, b, metric=None, keep=None, rb=None, margin=None):
    metric = metric or PM[ds]
    return {"a": f"{a}|{r}", "b": f"{b}|{rb or r}", "metric": metric,
            **cmp(ds, R.get(ds, r, a, metric), R.get(ds, rb or r, b, metric), keep, margin)}


def mean_of(per: dict | None, keep: set | None = None) -> float | None:
    if not per:
        return None
    v = [x for vs in sub(per, keep).values() for x in vs]
    return float(np.mean(v)) if v else None


def contrast(ds: str, d: dict, g1: set, g2: set) -> dict:
    """Bootstrap of mean(d | g1) - mean(d | g2), d = per-cluster paired differences.

    Clusters (papers / questions within domain strata) are resampled together, so the two
    groups' dependence within a paper is kept. BCa from the leave-one-cluster-out jackknife."""
    keys = [c for c in d if any((c, i) in g1 or (c, i) in g2 for i in range(len(d[c])))]
    if not keys:
        return {"status": "not_available"}
    s1 = np.array([sum(v for i, v in enumerate(d[c]) if (c, i) in g1) for c in keys], float)
    n1 = np.array([sum(1 for i in range(len(d[c])) if (c, i) in g1) for c in keys], float)
    s2 = np.array([sum(v for i, v in enumerate(d[c]) if (c, i) in g2) for c in keys], float)
    n2 = np.array([sum(1 for i in range(len(d[c])) if (c, i) in g2) for c in keys], float)
    if n1.sum() == 0 or n2.sum() == 0:
        return {"status": "not_available"}
    theta = s1.sum() / n1.sum() - s2.sum() / n2.sum()
    rng = np.random.default_rng(SEED)
    C = len(keys)
    if ds == "freshstack":
        st = np.array([R.strata[ds][c] for c in keys])
        idx = np.concatenate([np.flatnonzero(st == s)[rng.integers(0, int((st == s).sum()), size=(BOOT, int((st == s).sum())))]
                              for s in np.unique(st)], axis=1)
    else:
        idx = rng.integers(0, C, size=(BOOT, C))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = s1[idx].sum(1) / n1[idx].sum(1) - s2[idx].sum(1) / n2[idx].sum(1)
        boot = boot[np.isfinite(boot)]
        jk = (s1.sum() - s1) / (n1.sum() - n1) - (s2.sum() - s2) / (n2.sum() - n2)
    jk = jk[np.isfinite(jk)]
    prop = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / len(boot)
    z0 = stats._N.inv_cdf(min(max(prop, 1.0 / (len(boot) + 1)), len(boot) / (len(boot) + 1.0)))
    dd = jk.mean() - jk
    den = 6.0 * (np.sum(dd ** 2)) ** 1.5
    a = float(np.sum(dd ** 3) / den) if den > 0 else 0.0
    lo, hi = stats._bca_quantile(boot, z0, a, 0.025), stats._bca_quantile(boot, z0, a, 0.975)
    return {"status": "ok", "diff": float(theta), "ci95_bca": [lo, hi], "n_g1": int(n1.sum()), "n_g2": int(n2.sum()),
            "excludes_zero_95": bool(lo > 0 or hi < 0)}


def diffs(ds, r, a, b, metric=None, rb=None) -> dict | None:
    metric = metric or PM[ds]
    pa, pb = R.get(ds, r, a, metric), R.get(ds, rb or r, b, metric)
    if not pa or not pb:
        return None
    return {c: [x - y for x, y in zip(pa[c], pb[c])] for c in pa if c in pb and len(pa[c]) == len(pb[c])}


def spearman_boot(x: list, y: list) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = stats.spearman(list(x), list(y))
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(BOOT):
        i = rng.integers(0, len(x), len(x))
        v = stats.spearman(list(x[i]), list(y[i]))
        if not math.isnan(v):
            bs.append(v)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return {"rho": rho, "ci95_pct": [float(lo), float(hi)], "n": int(len(x))}


def tertiles(rows: list[dict], feat: str) -> tuple[dict, list[float]]:
    v = np.array([r[feat] for r in rows if r[feat] is not None and not math.isnan(r[feat])])
    t1, t2 = np.quantile(v, [1 / 3, 2 / 3])
    groups = {"low": set(), "mid": set(), "high": set()}
    for r in rows:
        x = r[feat]
        if x is None or math.isnan(x):
            continue
        groups["low" if x <= t1 else "mid" if x <= t2 else "high"].add(r["key"])
    return groups, [float(t1), float(t2)]


# =========================================================================== per-question features

def chunks_of(arm: str, docs: list[str]) -> list[dict]:
    tag, var, _ = ARM_CACHE[arm]
    out = []
    for d in docs:
        out += json.loads(build_v2.cache_path(tag, var, d, CACHE).read_text())["chunks"]
    return out


def head_text(c: dict, mode: str) -> str:
    """The indexed text that a rendering puts before the chunk text (prefix, fields or CR context)."""
    full = evaluate.render(c, mode)
    assert full.endswith(c["text"])
    return full[: len(full) - len(c["text"])]


def build_qasper_rows() -> list[dict]:
    papers = {p["id"]: p for p in json.loads((V2D / "qasper_v2.json").read_text())}
    rows = []
    for pid, qids in R.qids["qasper"].items():
        p = papers[pid]
        qs = [q for q in p["questions"] if q["evidence"]]
        assert [q["qid"] for q in qs] == qids, pid
        st = chunks_of("S+T", [pid])
        en = chunks_of("E", [pid])
        assert [c["text"] for c in st] == [c["text"] for c in en]
        tok_st = [set(evaluate.words(evaluate.render(c, "tc"))) for c in st]
        tok_txt = [set(evaluate.words(c["text"])) for c in st]
        tok_e = [set(evaluate.words(evaluate.render(c, "meta"))) for c in en]
        new_e = [te - ts for te, ts in zip(tok_e, tok_st)]
        tok_m = [set(evaluate.words(head_text(c, "meta"))) for c in en]
        units = [evaluate.words(c["text"]) for c in st]
        grams = [evaluate.ngrams(u) for u in units]
        n_tok = len(evaluate.ENC.encode(p["markdown"]))
        for i, q in enumerate(qs):
            gold = sorted({j for e in q["evidence"] for j in range(len(st)) if evaluate.covered(e, [units[j]], grams[j])})
            rows.append(dict(key=(pid, i), qid=q["qid"], cluster=pid, **_lex(q["question"], " ".join(q["evidence"]), gold,
                                                                                 tok_st, tok_txt, tok_e, new_e, tok_m),
                             n_evidence=len(q["evidence"]), evidence_words=sum(len(evaluate.words(e)) for e in q["evidence"]),
                             answer_type=_answer_type(q["answers"]),
                             any_unanswerable=any(a["type"] == "none" for a in q["answers"]),
                             paper_chunks=len(st), paper_tokens=n_tok))
    return rows


def _answer_type(answers: list[dict]) -> str:
    """yes/no if any annotator gave a boolean answer; else extractive / abstractive if every
    answerable annotator agrees; else mixed. Annotators who marked the question unanswerable are
    ignored here (their evidence is ignored by the dataset construction too)."""
    t = [a["type"] for a in answers if a["type"] != "none"]
    if "boolean" in t:
        return "yes/no"
    if t and all(x == "extractive" for x in t):
        return "extractive"
    if t and all(x == "abstractive" for x in t):
        return "abstractive"
    return "mixed" if t else "unanswerable"


def _lex(question: str, gold_text: str, gold: list[int], tok_st, tok_txt, tok_e, new_e, tok_m) -> dict:
    q = content_tokens(question)
    gt = set(evaluate.words(gold_text))
    gs = set().union(*(tok_st[j] for j in gold)) if gold else set()
    ge = set().union(*(tok_e[j] for j in gold)) if gold else set()
    missing = q - gs
    bridged = missing & ge
    lost = (q & gs) - ge  # question terms in the gold chunk's S+T text that E's rendering lacks (section path)
    gold_set = set(gold)
    new_q = [len(q & ne) for ne in new_e]
    ng = [new_q[j] for j in range(len(new_e)) if j not in gold_set]
    gq = [new_q[j] for j in gold]
    # question terms that the generated fields (title, summary, keywords, questions) mention at all,
    # new or already in the text: gold chunk(s) against the average other chunk
    mh = [len(q & m) for m in tok_m]
    mg = len(q & set().union(*(tok_m[j] for j in gold))) if gold else float("nan")
    mo = [mh[j] for j in range(len(tok_m)) if j not in gold_set]
    return {"meta_hits_gold": mg, "meta_hits_other": float(np.mean(mo)) if mo else float("nan"),"q_tokens": len(q), "overlap": (len(q & gt) / len(q)) if q else float("nan"),
            "overlap_goldchunk_st": (len(q & gs) / len(q)) if q and gold else float("nan"),
            "n_gold_chunks": len(gold), "n_missing": len(missing), "n_bridged": len(bridged), "n_lost": len(lost),
            "new_q_gold": float(np.mean(gq)) if gq else float("nan"),
            "new_q_other": float(np.mean(ng)) if ng else float("nan")}


def build_fs_rows() -> list[dict]:
    sel = json.loads((V2D / "selection.json").read_text())
    rows = []
    for dom in ev.FS_DOMAINS:
        ids = sel["freshstack"][dom]["doc_ids"]
        keep = set(sel["freshstack"][dom]["qids"])
        qs = [q for q in json.loads((V2D / f"freshstack_{dom}.json").read_text())["questions"] if q["qid"] in keep]
        st = chunks_of("S+T", ids)
        en = chunks_of("E", ids)
        assert [c["text"] for c in st] == [c["text"] for c in en]
        tok_st = [set(evaluate.words(evaluate.render(c, "tc"))) for c in st]
        tok_txt = [set(evaluate.words(c["text"])) for c in st]
        tok_e = [set(evaluate.words(evaluate.render(c, "meta"))) for c in en]
        new_e = [te - ts for te, ts in zip(tok_e, tok_st)]
        tok_m = [set(evaluate.words(head_text(c, "meta"))) for c in en]
        cgrams = [evaluate.ngrams(evaluate.words(c["text"])) for c in st]
        inv = defaultdict(list)
        for j, g in enumerate(cgrams):
            for x in g:
                inv[x].append(j)
        for q in qs:
            gold_grams = set().union(*(evaluate.ngrams(evaluate.words(e["text"])) for e in q["evidence"]))
            hits = Counter()
            for g in gold_grams:
                for j in inv.get(g, ()):
                    hits[j] += 1
            gold = sorted(j for j, h in hits.items() if cgrams[j] and h / len(cgrams[j]) >= 0.5)
            gtext = " ".join(e["text"] for e in q["evidence"])
            rows.append(dict(key=(q["qid"], 0), qid=q["qid"], cluster=q["qid"], domain=dom,
                             **_lex(q["question"], gtext, gold, tok_st, tok_txt, tok_e, new_e, tok_m),
                             gold_grams=len(gold_grams), gold_tokens=len(evaluate.ENC.encode(gtext))))
    order = {k: i for i, k in enumerate(qkey_order("freshstack"))}
    assert set(order) == {r["key"] for r in rows}
    return sorted(rows, key=lambda r: order[r["key"]])


# =========================================================================== 1. lexical mechanism

LEX_RETRIEVERS = ["bm25", "nomic", "mxbai", "hybrid", "qwen3", "hybrid-qwen3", "hybrid+rerank"]


def added_tokens() -> dict:
    """Unique word types that each rendering adds to the chunk text (evaluate.words tokens)."""
    sel = json.loads((V2D / "selection.json").read_text())
    docs = {"qasper": sel["qasper"]["paper_ids"],
            "freshstack": [d for dom in ev.FS_DOMAINS for d in sel["freshstack"][dom]["doc_ids"]]}
    out = {}
    for ds, ids in docs.items():
        res = {}
        for arm, mode in (("S+T", "tc"), ("E", "meta"), ("CR", "cr"), ("E-strong", "meta")):
            n_new, n_new_c, frac, n_txt = [], [], [], []
            for c in chunks_of(arm, ids):
                txt = set(evaluate.words(c["text"]))
                added = set(evaluate.words(head_text(c, mode)))
                new = added - txt
                n_new.append(len(new))
                n_new_c.append(len({w for w in new if w not in STOP and len(w) > 1}))
                frac.append(len(new) / max(len(added), 1))
                n_txt.append(len(txt))
            res[arm] = {"chunks": len(n_new), "new_types_mean": float(np.mean(n_new)),
                        "new_types_median": float(np.median(n_new)),
                        "new_content_types_mean": float(np.mean(n_new_c)),
                        "share_of_added_types_not_in_text": float(np.mean(frac)),
                        "text_types_mean": float(np.mean(n_txt))}
        out[ds] = res
    return out


def lexical() -> dict:
    out = {"added_tokens": added_tokens()}
    for ds in DATASETS:
        rows = QROWS[ds]
        g, cuts = tertiles(rows, "overlap")
        res = {"overlap_cuts": cuts, "overlap_mean": float(np.nanmean([r["overlap"] for r in rows])),
               "bin_sizes": {k: len(v) for k, v in g.items()}, "by_bin": {}, "contrast_low_minus_high": {}}
        for r in LEX_RETRIEVERS:
            if R.get(ds, r, "E", PM[ds]) is None:
                continue
            res["by_bin"][r] = {b: cmp_arms(ds, r, "E", "S+T", keep=g[b]) for b in ("low", "mid", "high")}
            res["by_bin"][r]["all"] = cmp_arms(ds, r, "E", "S+T")
            res["contrast_low_minus_high"][r] = contrast(ds, diffs(ds, r, "E", "S+T"), g["low"], g["high"])
        # robustness: overlap with the gold chunks' S+T rendering instead of the evidence text
        g2, cuts2 = tertiles(rows, "overlap_goldchunk_st")
        res["robust_goldchunk_st"] = {"cuts": cuts2, **{r: {"low": cmp_arms(ds, r, "E", "S+T", keep=g2["low"]),
                                                            "high": cmp_arms(ds, r, "E", "S+T", keep=g2["high"]),
                                                            "contrast": contrast(ds, diffs(ds, r, "E", "S+T"), g2["low"], g2["high"])}
                                                        for r in ("bm25", "nomic", "hybrid")}}
        # robustness: median split
        med = float(np.nanmedian([r["overlap"] for r in rows]))
        lo = {r["key"] for r in rows if r["overlap"] <= med}
        hi = {r["key"] for r in rows if r["overlap"] > med}
        res["robust_median_split"] = {"median": med, **{r: contrast(ds, diffs(ds, r, "E", "S+T"), lo, hi)
                                                        for r in ("bm25", "nomic", "hybrid")}}
        # bridging
        has_gold = [r for r in rows if r["n_gold_chunks"] > 0]
        br1 = {r["key"] for r in has_gold if r["n_bridged"] > 0}
        br0 = {r["key"] for r in has_gold if r["n_bridged"] == 0}
        net_pos = {r["key"] for r in has_gold if r["new_q_gold"] - r["new_q_other"] > 0}
        net_le = {r["key"] for r in has_gold if r["new_q_gold"] - r["new_q_other"] <= 0}
        miss0 = {r["key"] for r in has_gold if r["n_missing"] > 0 and r["n_bridged"] == 0}
        nomiss = {r["key"] for r in has_gold if r["n_missing"] == 0}
        b = {"questions_with_gold_chunk": len(has_gold), "share_bridged": len(br1) / len(has_gold),
             "n_missing_not_bridged": len(miss0), "n_no_missing": len(nomiss),
             "missing_terms_mean": float(np.mean([r["n_missing"] for r in has_gold])),
             "bridged_terms_mean": float(np.mean([r["n_bridged"] for r in has_gold])),
             "share_missing_bridged": float(sum(r["n_bridged"] for r in has_gold) / max(sum(r["n_missing"] for r in has_gold), 1)),
             "lost_terms_mean": float(np.mean([r["n_lost"] for r in has_gold])),
             "new_q_terms_gold_mean": float(np.nanmean([r["new_q_gold"] for r in has_gold])),
             "new_q_terms_other_mean": float(np.nanmean([r["new_q_other"] for r in has_gold])),
             "share_net_positive": len(net_pos) / len(has_gold), "by_retriever": {}}
        for r in LEX_RETRIEVERS:
            if R.get(ds, r, "E", PM[ds]) is None:
                continue
            d = diffs(ds, r, "E", "S+T")
            dq = by_q(d)
            st_base = R.get(ds, r, "S+T", PM[ds])
            b["by_retriever"][r] = {"bridged": cmp_arms(ds, r, "E", "S+T", keep=br1),
                                    "missing_not_bridged": cmp_arms(ds, r, "E", "S+T", keep=miss0),
                                    "no_missing": cmp_arms(ds, r, "E", "S+T", keep=nomiss),
                                    "contrast_bridged_minus_missing_not_bridged": contrast(ds, d, br1, miss0),
                                    "ST_mean": {"bridged": mean_of(st_base, br1), "missing_not_bridged": mean_of(st_base, miss0),
                                                "no_missing": mean_of(st_base, nomiss)},
                                    "share_of_total_gain_bridged": (sum(v for k, v in dq.items() if k in br1) / sum(dq.values()))
                                    if sum(dq.values()) else None,
                                    "not_bridged": cmp_arms(ds, r, "E", "S+T", keep=br0),
                                    "contrast_bridged_minus_not": contrast(ds, d, br1, br0),
                                    "net_positive": cmp_arms(ds, r, "E", "S+T", keep=net_pos),
                                    "net_nonpositive": cmp_arms(ds, r, "E", "S+T", keep=net_le),
                                    "contrast_net": contrast(ds, d, net_pos, net_le)}
        # bridging rate among E wins / ties / losses (BM25 and hybrid)
        for r in ("bm25", "hybrid"):
            d = by_q(diffs(ds, r, "E", "S+T"))
            wl = {"win": [k for k in d if d[k] > 0], "tie": [k for k in d if d[k] == 0], "loss": [k for k in d if d[k] < 0]}
            b[f"bridged_share_by_outcome_{r}"] = {o: (sum(1 for k in ks if k in br1) / max(sum(1 for k in ks if k in br1 | br0), 1),
                                                      len(ks)) for o, ks in wl.items()}
        res["bridging"] = b
        # targeting: question terms in the gold chunk's generated fields minus the average other chunk's
        for r_ in has_gold:
            r_["meta_net"] = r_["meta_hits_gold"] - r_["meta_hits_other"]
        gm, cm = tertiles(has_gold, "meta_net")
        res["targeting"] = {"cuts": cm, "meta_hits_gold_mean": float(np.mean([r["meta_hits_gold"] for r in has_gold])),
                            "meta_hits_other_mean": float(np.mean([r["meta_hits_other"] for r in has_gold])),
                            "share_gold_meta_hits_ge1": float(np.mean([r["meta_hits_gold"] >= 1 for r in has_gold])),
                            "bin_sizes": {k: len(v) for k, v in gm.items()},
                            **{r: {**{lab: cmp_arms(ds, r, "E", "S+T", keep=keep) for lab, keep in gm.items()},
                                   "contrast_high_minus_low": contrast(ds, diffs(ds, r, "E", "S+T"), gm["high"], gm["low"])}
                               for r in LEX_RETRIEVERS if R.get(ds, r, "E", PM[ds]) is not None}}
        out[ds] = res
    return out


# =========================================================================== 2. frontier

FRONT_RET = {"bm25": "lexical", "nomic": "embed", "mxbai": "embed", "qwen3": "embed",
             "hybrid": "RRF", "hybrid-qwen3": "RRF", "hybrid+rerank": "RRF+rerank30"}
QCOST = {"lexical": 0, "embed": 1, "RRF": 2, "RRF+rerank30": 3}
FRONT_ARMS = ["F512c", "F256t", "FL", "PP", "S", "S+T", "E", "CR", "E-strong", "CR-strong"]


def index_tokens() -> dict:
    """LLM tokens per Stage-1 chunk at indexing time, from the caches' server counters."""
    out = {}
    for ds, is_fs in (("qasper", False), ("freshstack", True)):
        res = {}
        for arm, (tag, var) in {"E": ("local", "enr"), "CR": ("local", "cr"), "E-strong": ("strong", "enr"),
                                "CR-strong": ("strong", "cr")}.items():
            t = Counter()
            for p in (CACHE / tag / var).glob("*.json"):
                if ("__" in p.stem) != is_fs:
                    continue
                d = json.loads(p.read_text())
                st = d["stats"]
                t["calls"] += st.get("calls", 0) or 0
                t["in"] += st.get("in_tokens", 0) or 0
                t["out"] += st.get("out_tokens", 0) or 0
                t["chunks"] += len(d["chunks"])
                t["usd"] += st.get("cost_usd", 0) or 0
            if t["chunks"]:
                res[arm] = {"chunks": t["chunks"], "calls": t["calls"], "in_per_chunk": t["in"] / t["chunks"],
                            "out_per_chunk": t["out"] / t["chunks"], "tokens_per_chunk": (t["in"] + t["out"]) / t["chunks"],
                            "usd_total": t["usd"]}
        out[ds] = res
    return out


def frontier() -> dict:
    tok = index_tokens()
    out = {"index_tokens": tok, "query_cost_classes": QCOST,
           "notes": "rerank30 = Qwen3-Reranker-0.6B over the hybrid (BM25+nomic) top-30; it was never run over "
                    "qwen3. qwen3 / jina require re-embedding the corpus; the reranker reuses the existing index."}
    for ds in DATASETS:
        grid = []
        for arm in FRONT_ARMS:
            for r, qc in FRONT_RET.items():
                m = mean_of(R.get(ds, r, arm, PM[ds]))
                if m is None:
                    continue
                grid.append({"arm": arm, "retriever": r, "query_class": qc, "score": m,
                             "index_llm_tokens_per_chunk": tok[ds].get(arm, {}).get("tokens_per_chunk", 0.0)})
        for g in grid:
            g["pareto"] = not any((h["index_llm_tokens_per_chunk"] <= g["index_llm_tokens_per_chunk"]
                                   and QCOST[h["query_class"]] <= QCOST[g["query_class"]] and h["score"] >= g["score"]
                                   and (h["index_llm_tokens_per_chunk"] < g["index_llm_tokens_per_chunk"]
                                        or QCOST[h["query_class"]] < QCOST[g["query_class"]] or h["score"] > g["score"]))
                                  for h in grid)
        pairs = {
            "STq3_minus_STrr": ("S+T", "qwen3", "S+T", "hybrid+rerank"),
            "STq3_minus_Ehyb": ("S+T", "qwen3", "E", "hybrid"),
            "STrr_minus_Ehyb": ("S+T", "hybrid+rerank", "E", "hybrid"),
            "Eq3_minus_STq3": ("E", "qwen3", "S+T", "qwen3"),
            "Err_minus_STrr": ("E", "hybrid+rerank", "S+T", "hybrid+rerank"),
            "Eq3_minus_Err": ("E", "qwen3", "E", "hybrid+rerank"),
            "STq3_minus_SThyb": ("S+T", "qwen3", "S+T", "hybrid"),
            "STrr_minus_SThyb": ("S+T", "hybrid+rerank", "S+T", "hybrid"),
            "Ehyb_minus_SThyb": ("E", "hybrid", "S+T", "hybrid"),
            "STq3_minus_STnomic": ("S+T", "qwen3", "S+T", "nomic"),
            "Snomic_minus_Shyb": ("S", "nomic", "S", "hybrid"),
            "STnomic_minus_SThyb": ("S+T", "nomic", "S+T", "hybrid"),
            "Enomic_minus_Ehyb": ("E", "nomic", "E", "hybrid"),
            "PPrr_minus_STrr": ("PP", "hybrid+rerank", "S+T", "hybrid+rerank"),
            "F256tq3_minus_STq3": ("F256t", "qwen3", "S+T", "qwen3"),
            "Esrr_minus_STq3": ("E-strong", "hybrid+rerank", "S+T", "qwen3"),
            "CRq3_minus_STq3": ("CR", "qwen3", "S+T", "qwen3"),
        }
        comps = {k: cmp_arms(ds, ra, a, b, rb=rb, margin=ev.MARGIN[ds]) for k, (a, ra, b, rb) in pairs.items()}
        best = max(grid, key=lambda g: g["score"])
        best_free = max((g for g in grid if g["index_llm_tokens_per_chunk"] == 0), key=lambda g: g["score"])
        best_free_norr = max((g for g in grid if g["index_llm_tokens_per_chunk"] == 0 and g["query_class"] != "RRF+rerank30"),
                             key=lambda g: g["score"])
        comps["best_minus_best_free"] = ({"status": "same_config"} if best is best_free else
                                         cmp_arms(ds, best["retriever"], best["arm"], best_free["arm"], rb=best_free["retriever"]))
        comps["best_free_minus_best_free_norr"] = cmp_arms(ds, best_free["retriever"], best_free["arm"], best_free_norr["arm"],
                                                           rb=best_free_norr["retriever"])
        out[ds] = {"grid": grid, "pareto": [g for g in grid if g["pareto"]], "comparisons": comps,
                   "best": best, "best_free": best_free, "best_free_no_rerank": best_free_norr}
    return out


# =========================================================================== 3. heterogeneity

def heterogeneity() -> dict:
    out = {}
    # win / tie / loss per retriever (both datasets)
    wtl = {}
    for ds in DATASETS:
        wtl[ds] = {}
        for r in LEX_RETRIEVERS + ["jina"]:
            for a, b in (("E", "S+T"), ("E", "CR")):
                d = diffs(ds, r, a, b)
                if not d:
                    continue
                v = [x for vs in d.values() for x in vs]
                wtl[ds][f"{a}-{b}|{r}"] = {"win": sum(x > 0 for x in v) / len(v), "tie": sum(x == 0 for x in v) / len(v),
                                           "loss": sum(x < 0 for x in v) / len(v), "n": len(v),
                                           "win_given_untied": sum(x > 0 for x in v) / max(sum(x != 0 for x in v), 1)}
    out["win_tie_loss"] = wtl

    rows = QROWS["qasper"]
    groups = {
        "answer_type": {t: {r["key"] for r in rows if r["answer_type"] == t} for t in ("extractive", "abstractive", "mixed", "yes/no")},
        "any_unanswerable": {str(v): {r["key"] for r in rows if r["any_unanswerable"] == v} for v in (False, True)},
        "n_evidence": {"1": {r["key"] for r in rows if r["n_evidence"] == 1}, "2": {r["key"] for r in rows if r["n_evidence"] == 2},
                       "3": {r["key"] for r in rows if r["n_evidence"] == 3}, "4+": {r["key"] for r in rows if r["n_evidence"] >= 4}},
    }
    pc, cuts = tertiles(rows, "paper_chunks")
    groups["paper_chunks"] = pc
    q = {"paper_chunk_cuts": cuts, "groups": {}}
    for gname, gs in groups.items():
        q["groups"][gname] = {}
        for lab, keep in gs.items():
            q["groups"][gname][lab] = {"n": len(keep), **{r: cmp_arms("qasper", r, "E", "S+T", keep=keep)
                                                          for r in ("hybrid", "bm25", "hybrid+rerank")}}
    q["contrast_paper_chunks_high_minus_low"] = {r: contrast("qasper", diffs("qasper", r, "E", "S+T"), pc["high"], pc["low"])
                                                 for r in ("hybrid", "bm25")}
    q["contrast_evidence_4plus_minus_1"] = {r: contrast("qasper", diffs("qasper", r, "E", "S+T"), groups["n_evidence"]["4+"],
                                                        groups["n_evidence"]["1"]) for r in ("hybrid", "bm25")}
    # answer F1 by answer type and conditional on the recall change
    f1 = {}
    for lab, keep in groups["answer_type"].items():
        f1[lab] = {"n": len(keep), "E-S+T": cmp_arms("qasper", "hybrid", "E", "S+T", metric="answer_f1", keep=keep),
                   "means": {a: mean_of(R.get("qasper", "hybrid", a, "answer_f1"), keep) for a in ("S+T", "E", "CR", "E-strong")}}
    drec = by_q(diffs("qasper", "hybrid", "E", "S+T"))
    cond = {"rec_up": {k for k, v in drec.items() if v > 0}, "rec_same": {k for k, v in drec.items() if v == 0},
            "rec_down": {k for k, v in drec.items() if v < 0}}
    f1["conditional_on_recall_change"] = {lab: {"n": len(keep), "dF1": cmp_arms("qasper", "hybrid", "E", "S+T", metric="answer_f1", keep=keep)}
                                          for lab, keep in cond.items()}
    f1["contrast_rec_up_minus_rec_down"] = contrast("qasper", diffs("qasper", "hybrid", "E", "S+T", metric="answer_f1"),
                                                    cond["rec_up"], cond["rec_down"])
    dF = by_q(diffs("qasper", "hybrid", "E", "S+T", metric="answer_f1"))
    ks = sorted(drec)
    f1["spearman_drec_dF1"] = stats.spearman([drec[k] for k in ks], [dF[k] for k in ks])
    # how often does the recall change move answers at all
    f1["share_rec_changed"] = sum(1 for k in ks if drec[k] != 0) / len(ks)
    f1["share_f1_changed"] = sum(1 for k in ks if dF[k] != 0) / len(ks)
    q["answer_f1"] = f1
    out["qasper"] = q

    rows = QROWS["freshstack"]
    gl, cuts = tertiles(rows, "gold_tokens")
    fs = {"gold_token_cuts": cuts, "groups": {"gold_length": {}, "domain": {}}}
    for lab, keep in gl.items():
        fs["groups"]["gold_length"][lab] = {"n": len(keep), **{r: cmp_arms("freshstack", r, "E", "S+T", keep=keep)
                                                               for r in ("hybrid", "bm25", "hybrid+rerank")}}
    for dom in ev.FS_DOMAINS:
        keep = {r["key"] for r in rows if r["domain"] == dom}
        fs["groups"]["domain"][dom] = {"n": len(keep), **{r: cmp_arms("freshstack", r, "E", "S+T", keep=keep)
                                                          for r in ("bm25", "hybrid+rerank", "qwen3")}}
    fs["contrast_gold_long_minus_short"] = {r: contrast("freshstack", diffs("freshstack", r, "E", "S+T"), gl["high"], gl["low"])
                                            for r in ("hybrid", "bm25")}
    out["freshstack"] = fs
    return out


# =========================================================================== 4. budgets

BUDGET_METRICS = {"qasper": ["rec@256t", "rec@512t", "rec@1024t", "hit@1", "hit@5", "rec@5"],
                  "freshstack": ["prec@512t", "prec@1024t", "prec@2048t", "rec@1024t", "hit@1", "hit@5", "mrr@10"]}
BUDGET_RET = ["hybrid", "bm25", "nomic", "qwen3", "hybrid+rerank"]
BUDGET_PAIRS = [("E", "S+T"), ("E", "CR"), ("S", "F256t")]


def budgets() -> dict:
    out = {}
    for ds in DATASETS:
        res, rev = {}, []
        for a, b in BUDGET_PAIRS:
            for r in BUDGET_RET:
                prim = cmp_arms(ds, r, a, b)
                for m in BUDGET_METRICS[ds]:
                    row = prim if m == PM[ds] else cmp_arms(ds, r, a, b, metric=m)
                    res[f"{a}-{b}|{r}|{m}"] = row
                    if row.get("status") == "ok" and prim.get("status") == "ok" and m != PM[ds]:
                        if np.sign(row["diff"]) != np.sign(prim["diff"]) and (row["excludes_zero_95"] or prim["excludes_zero_95"]):
                            rev.append({"pair": f"{a}-{b}", "retriever": r, "metric": m, "diff": row["diff"],
                                        "strict": bool(row["excludes_zero_95"]),
                                        "ci95": row["ci95_bca"], "primary_diff": prim["diff"], "primary_ci95": prim["ci95_bca"]})
        out[ds] = {"rows": res, "sign_reversals": rev}
    return out


# =========================================================================== 5. late chunking

def late_chunking() -> dict:
    import retrieval_extra as rx
    import hashlib
    from lc_similarity import mean_offdiag_cos
    per_doc = {}
    for f in sorted(rx.LC_CACHE.glob("*.npz")):
        doc = f.name.rsplit(".", 2)[0]
        if "__" in doc:
            continue
        z = np.load(f, allow_pickle=True)
        late = z["emb"]
        rep = json.loads(str(z["report"]))
        ch = ev.load_chunks("S", [doc], CACHE)
        texts = [evaluate.render(c, "text") for c in ch]
        paths = [rx.JINA_CACHE / (hashlib.sha1(f"{rx.JinaV3.name}|retrieval.passage|{t}".encode()).hexdigest() + ".npy") for t in texts]
        if len(texts) < 2 or len(late) != len(texts) or not all(p.exists() for p in paths):
            continue
        plain = np.vstack([np.load(p) for p in paths])
        per_doc[doc] = {"n_windows": rep["n_windows"], "n_chunks": len(texts), "cos_plain": mean_offdiag_cos(plain),
                        "cos_late": mean_offdiag_cos(late)}
    d = ev.load_per_question(V2R / "extra" / "qasper_per_question.json")["per"]
    lc, s = d[("jina", "LC-S", "rec@512t")], d[("jina", "S", "rec@512t")]
    papers = [p for p in lc if p in per_doc]
    loss = [float(np.mean(lc[p]) - np.mean(s[p])) for p in papers]
    f = {k: [per_doc[p][k] for p in papers] for k in ("n_windows", "n_chunks", "cos_plain", "cos_late")}
    dcos = [b - a for a, b in zip(f["cos_plain"], f["cos_late"])]
    toks = {p: len(evaluate.ENC.encode(x["markdown"])) for x in json.loads((V2D / "qasper_v2.json").read_text())
            for p in [x["id"]] if p in per_doc}
    out = {"n_papers": len(papers),
           "windows_hist": dict(Counter(f["n_windows"])),
           "rho_loss_vs_dcos": spearman_boot(loss, dcos),
           "rho_loss_vs_cos_late": spearman_boot(loss, f["cos_late"]),
           "rho_loss_vs_n_chunks": spearman_boot(loss, f["n_chunks"]),
           "rho_loss_vs_tokens": spearman_boot(loss, [toks[p] for p in papers]),
           "rho_dcos_vs_n_chunks": spearman_boot(dcos, f["n_chunks"]),
           "rho_plain_S_vs_n_chunks": spearman_boot([float(np.mean(s[p])) for p in papers], f["n_chunks"]),
           "cos_late_mean": float(np.mean(f["cos_late"])), "cos_plain_mean": float(np.mean(f["cos_plain"])),
           "share_papers_cos_late_ge_0_9": float(np.mean([c >= 0.9 for c in f["cos_late"]]))}
    one = {p for p, w in zip(papers, f["n_windows"]) if w == 1}
    multi = {p for p, w in zip(papers, f["n_windows"]) if w > 1}
    keep1 = {(p, i) for p in one for i in range(len(lc[p]))}
    keepm = {(p, i) for p in multi for i in range(len(lc[p]))}
    out["by_windows"] = {"1": {"papers": len(one), **cmp("qasper", lc, s, keep1)},
                         ">1": {"papers": len(multi), **cmp("qasper", lc, s, keepm)},
                         "contrast_multi_minus_one": contrast("qasper", {p: [x - y for x, y in zip(lc[p], s[p])] for p in lc}, keepm, keep1)}
    # by within-paper cosine collapse tertile (question level, cluster = paper)
    dc = dict(zip(papers, dcos))
    t1, t2 = np.quantile(dcos, [1 / 3, 2 / 3])
    gk = {lab: {(p, i) for p in papers if cond(dc[p]) for i in range(len(lc[p]))}
          for lab, cond in (("low", lambda x: x <= t1), ("mid", lambda x: t1 < x <= t2), ("high", lambda x: x > t2))}
    out["by_dcos_tertile"] = {"cuts": [float(t1), float(t2)], **{k: cmp("qasper", lc, s, v) for k, v in gk.items()},
                              "contrast_high_minus_low": contrast("qasper", {p: [x - y for x, y in zip(lc[p], s[p])] for p in lc},
                                                                  gk["high"], gk["low"])}
    # FreshStack documents: collapse only (retrieval is pooled, so no per-document loss)
    fsd = []
    for f_ in sorted(rx.LC_CACHE.glob("*__*.npz")):
        rep = json.loads(str(np.load(f_, allow_pickle=True)["report"]))
        fsd.append(rep["n_windows"])
    out["freshstack_windows_hist"] = dict(Counter(fsd))
    return out


# =========================================================================== 6. granularity

def mean_chunk_tokens(arm: str, ds: str) -> float:
    sel = json.loads((V2D / "selection.json").read_text())
    ids = sel["qasper"]["paper_ids"] if ds == "qasper" else [d for dom in ev.FS_DOMAINS for d in sel["freshstack"][dom]["doc_ids"]]
    specs = SEEDED.get(arm) or [ARM_CACHE[arm]]
    vals = []
    for tag, var, _ in specs:
        n_tok = n = 0
        for d in ids:
            for c in json.loads(build_v2.cache_path(tag, var, d, CACHE).read_text())["chunks"]:
                n_tok += len(evaluate.ENC.encode(c["text"]))
                n += 1
        vals.append(n_tok / n)
    return float(np.mean(vals))


def granularity() -> dict:
    out = {}
    for ds in DATASETS:
        fam = {}
        for name, arms in (("text", TEXT_FAMILY), ("meta", META_FAMILY)):
            pts = []
            for a in arms:
                m = mean_of(R.get(ds, "hybrid", a, PM[ds]))
                if m is None:
                    continue
                pts.append({"arm": a, "mean_tokens": mean_chunk_tokens(a, ds), "score": m,
                            "score_bm25": mean_of(R.get(ds, "bm25", a, PM[ds]))})
            rho = stats.spearman([p["mean_tokens"] for p in pts], [p["score"] for p in pts])
            fam[name] = {"points": pts, "spearman_len_score_hybrid": rho}
        # merge benefit by gold length (per question)
        rows = QROWS[ds]
        feat = "gold_tokens" if ds == "freshstack" else "evidence_words"
        g, cuts = tertiles(rows, feat)
        mb = {"feature": feat, "cuts": cuts}
        for a, b in (("M", "E"), ("M-TP", "E-noRK"), ("RM-TP", "E-noRK"), ("SM-TP", "E-noRK")):
            d = diffs(ds, "hybrid", a, b)
            if not d:
                continue
            mb[f"{a}-{b}"] = {**{lab: cmp_arms(ds, "hybrid", a, b, keep=keep) for lab, keep in g.items()},
                              "contrast_long_minus_short": contrast(ds, d, g["high"], g["low"])}
        fam["merge_by_gold_length"] = mb
        out[ds] = fam
    return out


# =========================================================================== 7. rolling keys

def keys() -> dict:
    sel = json.loads((V2D / "selection.json").read_text())
    docs = {"qasper": sel["qasper"]["paper_ids"],
            "freshstack": [d for dom in ev.FS_DOMAINS for d in sel["freshstack"][dom]["doc_ids"]]}
    cap = 40
    out = {}
    for ds, ids in docs.items():
        res = {}
        for arm, var in (("E", "enr"), ("E-noRK", "enr_nork")):
            dist, visible, evicted_reuse, reuse, n, keyed = [], 0, 0, 0, 0, 0
            docs_evicting, evicted_total, evicted_recur = 0, 0, 0
            key_in_text, key_tok_frac, kw_tok_new, q_tok_new, key_tok_new_st = [], [], [], [], []
            for d in ids:
                ch = json.loads(build_v2.cache_path("local", var, d, CACHE).read_text())["chunks"]
                roll: dict = {}  # key -> last index (mdkeychunker.enricher._update_rolling_keys)
                last: dict = {}
                evicted: set = set()
                ev_doc = False
                for i, c in enumerate(ch):
                    n += 1
                    k = c.get("key", "")
                    txt = evaluate.words(c["text"])
                    stx = set(evaluate.words(evaluate.render(c, "tc")))
                    if k:
                        keyed += 1
                        kw = evaluate.words(k)
                        key_in_text.append(any(txt[j:j + len(kw)] == kw for j in range(len(txt) - len(kw) + 1)) if kw else False)
                        key_tok_frac.append(sum(w in set(txt) for w in kw) / max(len(kw), 1))
                        key_tok_new_st.append(sum(w not in stx for w in kw) / max(len(kw), 1))
                    kws = [w for x in c.get("keywords", []) for w in evaluate.words(x)]
                    if kws:
                        kw_tok_new.append(sum(w not in stx for w in kws) / len(kws))
                    qws = [w for x in c.get("questions", []) for w in evaluate.words(x) if w not in STOP]
                    if qws:
                        q_tok_new.append(sum(w not in stx for w in qws) / len(qws))
                    if not k:
                        continue
                    if k in last:
                        reuse += 1
                        dist.append(i - last[k])
                        if k in roll:
                            visible += 1
                        else:
                            evicted_reuse += 1
                            if k in evicted:
                                evicted_recur += 1
                    last[k] = i
                    roll[k] = i
                    if len(roll) > cap:
                        keepk = sorted(roll.items(), key=lambda x: x[1], reverse=True)[:cap]
                        gone = set(roll) - {x for x, _ in keepk}
                        evicted |= gone
                        evicted_total += len(gone)
                        ev_doc = True
                        roll = dict(keepk)
                docs_evicting += ev_doc
            dist_a = np.array(dist)
            res[arm] = {"chunks": n, "keyed": keyed, "reuses": reuse, "reuse_rate": reuse / max(n, 1),
                        "reuse_distance_median": float(np.median(dist_a)) if len(dist_a) else None,
                        "reuse_distance_mean": float(dist_a.mean()) if len(dist_a) else None,
                        "share_reuse_adjacent": float(np.mean(dist_a == 1)) if len(dist_a) else None,
                        "share_reuse_within_3": float(np.mean(dist_a <= 3)) if len(dist_a) else None,
                        "share_reuse_gt_10": float(np.mean(dist_a > 10)) if len(dist_a) else None,
                        "reuses_of_key_outside_window": evicted_reuse,
                        "docs_with_eviction": docs_evicting, "evicted_keys": evicted_total,
                        "evicted_key_recurred": evicted_recur,
                        "key_verbatim_in_text": float(np.mean(key_in_text)),
                        "key_token_share_in_text": float(np.mean(key_tok_frac)),
                        "key_token_share_not_in_st": float(np.mean(key_tok_new_st)),
                        "keyword_token_share_not_in_st": float(np.mean(kw_tok_new)),
                        "question_token_share_not_in_st": float(np.mean(q_tok_new))}
        out[ds] = res
    return out


# =========================================================================== 8. audit counts

def audit() -> dict:
    papers = {p["id"]: p for p in json.loads((V2D / "qasper_v2.json").read_text())}
    qs = [q for pid in R.qids["qasper"] for q in papers[pid]["questions"] if q["evidence"]]
    nev = Counter(min(len(q["evidence"]), 4) for q in qs)
    nann = Counter(len(q["answers"]) for q in qs)
    tok = sum(len(evaluate.ENC.encode(papers[p]["markdown"])) for p in R.qids["qasper"])
    nchunks = sum(len(ev.load_chunks("S", [p], CACHE)) for p in R.qids["qasper"])
    fsq = Counter(R.strata["freshstack"].values())
    a = {"qasper_questions": len(qs), "qasper_papers": len(R.qids["qasper"]),
         "qasper_evidence_paragraphs": sum(len(q["evidence"]) for q in qs),
         "qasper_evidence_per_question_1_2_3_4plus": [nev[1], nev[2], nev[3], nev[4]],
         "qasper_annotators_hist": dict(sorted(nann.items())), "qasper_markdown_tokens": tok, "qasper_S_chunks": nchunks,
         "fs_questions_by_domain": dict(fsq),
         "pf_questions": sum(len(v) for v in (R.get("qasper", "hybrid", "PF", "rec@512t") or {}).values()),
         "pf_papers": len(R.get("qasper", "hybrid", "PF", "rec@512t") or {})}
    # hybrid recomputed in the extra file equals the confirmatory hybrid
    same = True
    for ds in DATASETS:
        main = dict(R.src[ds][0][1])
        extra = dict(R.src[ds][1][1])
        for k, v in extra.items():
            if k[0] == "hybrid" and k in main:
                same &= all(np.allclose(main[k][c], v[c]) for c in v)
    a["extra_hybrid_equals_confirmatory"] = bool(same)
    # every arm under hybrid on FreshStack (for "M-TP highest point estimate of any arm")
    fsm = {}
    for _, per in R.src["freshstack"]:
        for (r, arm, m), pp in per.items():
            if r == "hybrid" and m == "prec@1024t":
                fsm.setdefault(arm, mean_of(pp))
    a["fs_hybrid_arm_means"] = dict(sorted(fsm.items(), key=lambda x: -x[1]))
    # answer F1 spread (Qasper, hybrid)
    f1 = {arm: mean_of(R.get("qasper", "hybrid", arm, "answer_f1")) for arm in ("F512c", "F256t", "S", "S+T", "E", "CR", "E-strong", "CR-strong")}
    a["qasper_f1_means"] = f1
    return a


# =========================================================================== main

def _clean(o):
    if isinstance(o, dict):
        return {str(k) if not isinstance(k, str) else k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def main(argv=None) -> int:
    global R, BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--only", default="", help="comma list of sections (debugging)")
    args = ap.parse_args(argv)
    BOOT = args.boot
    R = Results()
    QROWS["qasper"] = build_qasper_rows()
    QROWS["freshstack"] = build_fs_rows()
    secs = {"lexical": lexical, "frontier": frontier, "heterogeneity": heterogeneity, "budgets": budgets,
            "late_chunking": late_chunking, "granularity": granularity, "keys": keys, "audit": audit}
    only = [s for s in args.only.split(",") if s]
    res = {"label": LABEL, "settings": {"resamples": BOOT, "seed": SEED, "primary_metric": PM,
                                        "stopwords": "scikit-learn 1.9.1 ENGLISH_STOP_WORDS; tokens = evaluate.words, len > 1"}}
    for name, fn in secs.items():
        if only and name not in only:
            continue
        print(f"[analysis_v4] {name}", flush=True)
        res[name] = fn()
    Path(args.out).write_text(json.dumps(_clean(res), indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
