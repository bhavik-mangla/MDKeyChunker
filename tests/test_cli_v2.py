"""mdkeychunker CLI: chunk, diagnose, the 0.3 alias and the legacy pipeline route."""
import json

import pytest

from mdkeychunker import cli_v2

GUIDE = ("# Guide\n\n## Install\n\nRun pip install foo on Linux.\n\n"
         "## Pricing\n\nTen dollars a month.\n")


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    (d / "sub").mkdir(parents=True)
    (d / "guide.md").write_text(GUIDE)
    (d / "sub" / "notes.txt").write_text("plain text " * 200)
    (d / "image.png").write_bytes(b"\x89PNG")
    return d


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_help_and_no_args(capsys):
    assert cli_v2.main([]) == 0
    assert "mdkeychunker chunk" in capsys.readouterr().out


def test_chunk_to_stdout_and_file(docs, tmp_path, capsys):
    assert cli_v2.main(["chunk", str(docs / "guide.md")]) == 0
    rows = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert rows and rows[0]["doc_id"] == "guide.md"
    assert rows[0]["embed_text"].startswith("Guide")
    out = tmp_path / "c.jsonl"
    assert cli_v2.main(["chunk", str(docs), "-o", str(out), "--no-prefix"]) == 0
    rows = read_jsonl(out)
    assert {r["doc_id"] for r in rows} == {"guide.md", "sub/notes.txt"}  # .png skipped
    assert all(r["embed_text"] == r["text"] for r in rows if r["doc_id"] == "guide.md")


def test_chunk_windows_and_tokenizer(docs, tmp_path):
    out = tmp_path / "w.jsonl"
    assert cli_v2.main(["chunk", str(docs / "sub" / "notes.txt"), "--windows", "--size", "64",
                        "--overlap", "8", "-o", str(out)]) == 0
    rows = read_jsonl(out)
    assert len(rows) > 3 and all(r["token_count"] <= 64 for r in rows)
    pytest.importorskip("tiktoken")
    assert cli_v2.main(["chunk", str(docs / "guide.md"), "--tokenizer", "cl100k_base",
                        "-o", str(out)]) == 0


def test_chunk_missing_file_is_an_error(tmp_path, capsys):
    assert cli_v2.main(["chunk", str(tmp_path / "nope.md")]) == 2
    assert "not found" in capsys.readouterr().err


def test_bare_file_is_the_chunk_alias_and_writes_file_jsonl(docs, capsys):
    assert cli_v2.main([str(docs / "guide.md")]) == 0
    err = capsys.readouterr().err
    assert "now runs `mdkeychunker chunk FILE`" in err and "no LLM" in err
    assert read_jsonl(docs / "guide.jsonl")[0]["doc_id"] == "guide.md"


def test_llm_flags_route_to_the_legacy_pipeline(docs, monkeypatch, capsys):
    seen = {}

    def fake_main():
        import sys
        seen["argv"] = list(sys.argv)
        return 0

    from mdkeychunker import cli
    monkeypatch.setattr(cli, "main", fake_main)
    with pytest.warns(DeprecationWarning):
        assert cli_v2.main([str(docs / "guide.md"), "--provider", "openai"]) == 0
    assert seen["argv"][1:] == [str(docs / "guide.md"), "--provider", "openai"]
    assert "deprecated 0.3 pipeline" in capsys.readouterr().err
    with pytest.warns(DeprecationWarning):
        assert cli_v2.main(["legacy", str(docs / "guide.md")]) == 0


# --------------------------------------------------------------------------- diagnose

def write_qa(tmp_path, rows):
    p = tmp_path / "qa.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_diagnose_from_docs_dir_with_byo_objects(docs, tmp_path, capsys):
    pytest.importorskip("numpy")  # the built-in dense half ([dense] brings numpy)
    qa = write_qa(tmp_path, [
        {"question": "how do I install foo?", "gold": ["Run pip install foo"],
         "doc_id": "guide.md"},
        {"question": "what does it cost?", "gold": ["Ten dollars a month"], "doc_id": "guide.md"},
    ])
    out_json = tmp_path / "r.json"
    code = cli_v2.main(["diagnose", "--docs", str(docs), "--qa", str(qa),
                        "--checks", "fusion,rerank",
                        "--bm25", "study", "--embedder", "tests.cli_fixtures:flat_embed",
                        "--reranker", "tests.cli_fixtures:null_reranker", "--n-boot", "200",
                        "--retriever", "mine=tests.cli_fixtures:never",
                        "--compare", "mine,bm25", "--json", str(out_json)])
    assert code == 0
    md = capsys.readouterr().out
    assert md.startswith("# Retrieval diagnosis") and "| mine |" in md
    report = json.loads(out_json.read_text())
    pairs = {(f["a"], f["b"]) for f in report["findings"]}
    assert {("hybrid", "dense"), ("mine+rerank", "mine"), ("mine", "bm25")} <= pairs
    assert report["settings"]["baseline"] == "mine"


def test_diagnose_from_chunks_jsonl_and_markdown_output(docs, tmp_path, capsys):
    chunks = tmp_path / "c.jsonl"
    assert cli_v2.main(["chunk", str(docs / "guide.md"), "-o", str(chunks)]) == 0
    qa = write_qa(tmp_path, [{"question": "install?", "gold": ["pip install foo"],
                              "doc_id": "guide.md"}])
    out_md = tmp_path / "r.md"
    assert cli_v2.main(["diagnose", "--chunks", str(chunks), "--qa", str(qa), "--checks", "none",
                        "--retriever", "mine=tests.cli_fixtures:never", "-o", str(out_md)]) == 0
    assert "| mine | 0.0 |" in out_md.read_text()


def test_diagnose_errors_are_reported_not_raised(docs, tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    assert cli_v2.main(["diagnose", "--docs", str(docs), "--qa", str(bad)]) == 2
    assert "not valid JSON" in capsys.readouterr().err
    qa = write_qa(tmp_path, [{"question": "q", "gold": ["g"], "doc_id": "guide.md"}])
    assert cli_v2.main(["diagnose", "--docs", str(docs), "--qa", str(qa),
                        "--retriever", "oops"]) == 2
    assert "NAME=MODULE:ATTR" in capsys.readouterr().err
    assert cli_v2.main(["diagnose", "--docs", str(docs), "--qa", str(qa), "--checks", "vibes"]) == 2
    assert "unknown check" in capsys.readouterr().err
    try:
        import numpy  # noqa: F401
    except ImportError:  # core install: the check that needs [dense] is named
        assert cli_v2.main(["diagnose", "--docs", str(docs), "--qa", str(qa),
                            "--checks", "fusion", "--bm25", "study"]) == 2
        assert "check='fusion' needs a dense embedder" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # argparse: --qa is required
        cli_v2.main(["diagnose", "--docs", str(docs)])
