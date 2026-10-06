"""Analysis 5 (PILOT v3 data): the mxbai truncation hypothesis.

(i) Count indexed strings longer than mxbai's 512-token window using mxbai's own
    WordPiece tokenizer (HF mixedbread-ai/mxbai-embed-large-v1 tokenizer.json,
    downloaded to phase0/mxbai_tokenizer.json; counts include [CLS]/[SEP]).
(ii) Add Qwen3-Embedding-0.6B (32k context; run with num_ctx 8192) as a dense
    retriever and in a BM25+qwen3 RRF hybrid; does E vs S+T change?
(iii) "Metadata-last" rendering of E (text first, generated fields after), which
    keeps generated text from displacing source text under truncation.
"""
import sys
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import (OUT, chunks_of, eval_freshstack, eval_qasper, fs_chunks, load_freshstack,  # noqa: E402
                    load_qasper, mean_of, render, save_json)
from stats_pilot import analyse  # noqa: E402

TOK = Tokenizer.from_file(str(Path(__file__).parent / "mxbai_tokenizer.json"))
TOK.no_truncation()
TOK.no_padding()
LIMIT = 512


def render_x(c, mode):
    if mode == "meta_last":
        m = render(c, "meta")
        head = m[: len(m) - len(c["text"])].strip()
        return c["text"] + "\n\n" + head
    return render(c, mode)


SYSTEMS = {"F256t": ("fixedtok", "text"), "F512c": ("fixed512", "text"), "S": ("struct", "text"),
           "S+T": ("struct", "tc"), "CR": ("cr", "cr"), "E": ("enr_rk", "meta"), "E(meta-last)": ("enr_rk", "meta_last"),
           "M": ("merged_rk", "meta")}
DENSE_SYS = ["S", "S+T", "CR", "E", "E(meta-last)"]
RETR = ["mxbai", "qwen3e", "hybrid_qwen3e", "nomic", "hybrid", "bm25"]
COMPS = [("E", "S+T"), ("E", "CR"), ("E", "S"), ("E(meta-last)", "E"), ("E(meta-last)", "S+T")]


def trunc_stats(chunks, mode):
    n_over, lost, total, src_lost, src_total = 0, 0, 0, 0, 0
    for c in chunks:
        s = render_x(c, mode)
        n = len(TOK.encode(s).ids)  # includes [CLS] and [SEP]
        total += n
        if n > LIMIT:
            n_over += 1
            lost += n - LIMIT
        # source text (c["text"]) tokens falling beyond the window, given the rendering
        t = len(TOK.encode(c["text"], add_special_tokens=False).ids)
        if mode in ("meta_last", "text"):
            head = 0
        else:
            head = len(TOK.encode(s, add_special_tokens=False).ids) - t
        avail = max(0, LIMIT - 2 - head)
        src_lost += max(0, t - avail)
        src_total += t
    return {"chunks": len(chunks), "over_512": n_over, "share_over_512": n_over / len(chunks),
            "share_tokens_truncated": lost / total, "share_source_text_truncated": src_lost / src_total}


def main():
    out = {"label": "PILOT v3 data, exploratory",
           "tokenizer": "HF mixedbread-ai/mxbai-embed-large-v1 tokenizer.json (BERT WordPiece, uncased)"}
    lines = ["# A5 mxbai truncation and a long-context embedder (PILOT)", ""]
    for ds in ("qasper", "freshstack"):
        if ds == "qasper":
            papers = load_qasper()
            metric, margin = "rec@512t", 0.05
            allch = lambda cs: [c for p in papers for c in chunks_of(cs, p["id"])]  # noqa: E731
        else:
            data, ids = load_freshstack()
            metric, margin = "prec@1024t", 0.03
            allch = lambda cs: fs_chunks(cs, ids)  # noqa: E731
        tr = {k: trunc_stats(allch(cs), m) for k, (cs, m) in SYSTEMS.items()}
        per = {}
        for k in DENSE_SYS:
            cs, m = SYSTEMS[k]
            if ds == "qasper":
                res = eval_qasper(papers, lambda pid, cs=cs: chunks_of(cs, pid),
                                  lambda pid, ch, m=m: [render_x(c, m) for c in ch], RETR)
            else:
                ch = allch(cs)
                res = eval_freshstack(data, ch, [render_x(c, m) for c in ch], RETR)
            for (r, mm), v in res.items():
                per[(r, k)] = v
        rows = []
        for r in RETR:
            row = {"retriever": r, "means": {k: mean_of(per[(r, k)]) for k in DENSE_SYS}}
            for a, b in COMPS:
                row[f"{a} - {b}"] = analyse(per[(r, a)], per[(r, b)], margin)
            rows.append(row)
        out[ds] = {"metric": metric, "truncation": tr, "rows": rows}
        lines += [f"## {ds}", "", "| system | chunks | >512 mxbai tokens | share of tokens cut | share of SOURCE text cut |",
                  "|---|---|---|---|---|"]
        for k, t in tr.items():
            lines.append(f"| {k} | {t['chunks']} | {t['over_512']} ({100*t['share_over_512']:.1f}%) | "
                         f"{100*t['share_tokens_truncated']:.1f}% | {100*t['share_source_text_truncated']:.1f}% |")
        lines += ["", f"{metric}:", "", "| retr | " + " | ".join(DENSE_SYS) + " | " + " | ".join(f"{a} - {b}" for a, b in COMPS) + " |",
                  "|---" * (1 + len(DENSE_SYS) + len(COMPS)) + "|"]
        for row in rows:
            c = lambda k: f"{100*row[k]['diff']:+.1f} [{100*row[k]['ci95'][0]:+.1f},{100*row[k]['ci95'][1]:+.1f}]"  # noqa: E731
            lines.append(f"| {row['retriever']} | " + " | ".join(f"{row['means'][k]:.3f}" for k in DENSE_SYS) + " | "
                         + " | ".join(c(f"{a} - {b}") for a, b in COMPS) + " |")
        lines.append("")
    save_json("a5_truncation.json", out)
    (OUT / "a5_truncation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
