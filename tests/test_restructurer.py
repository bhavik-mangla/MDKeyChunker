"""Tests for Restructurer."""
import pytest
from mdkeychunker.config import Config
from mdkeychunker.restructurer import Restructurer, _dedupe_entities
from mdkeychunker.models import Chunk


def _chunk(text: str, key: str = "", pos: int = 0) -> Chunk:
    c = Chunk(text=text, key=key, position_index=pos)
    c.generate_id()
    return c


@pytest.fixture
def config():
    return Config(
        merge_by_keys=True,
        max_merged_size=3000,
    )


@pytest.fixture
def restructurer(config):
    return Restructurer(config)


def test_merge_same_key_adjacent(restructurer):
    chunks = [
        _chunk("Alpha text about auth.", key="authentication", pos=0),
        _chunk("Beta text about auth.", key="authentication", pos=1),
        _chunk("Unrelated content.", key="logging", pos=2),
    ]
    result = restructurer.restructure(chunks)
    # The two auth chunks should be merged
    merged = [c for c in result if c.key == "authentication"]
    assert len(merged) == 1
    assert "Alpha text" in merged[0].text
    assert "Beta text" in merged[0].text


def test_no_merge_different_keys(restructurer):
    chunks = [
        _chunk("About authentication.", key="authentication", pos=0),
        _chunk("About database.", key="database", pos=1),
        _chunk("About caching.", key="caching", pos=2),
    ]
    result = restructurer.restructure(chunks)
    assert len(result) == 3


def test_merge_globally_regardless_of_distance(restructurer):
    """Same-key chunks at any distance should merge globally."""
    chunks = [
        _chunk("Auth A.", key="auth", pos=0),
        _chunk("Filler chunk.", key="other", pos=1),
        _chunk("Auth B.", key="auth", pos=2),
    ]
    result = restructurer.restructure(chunks)
    auth_chunks = [c for c in result if c.key == "auth"]
    assert len(auth_chunks) == 1
    assert "Auth A" in auth_chunks[0].text
    assert "Auth B" in auth_chunks[0].text


def test_no_merge_when_disabled():
    config = Config(merge_by_keys=False)
    restructurer = Restructurer(config)
    chunks = [
        _chunk("Auth A.", key="auth", pos=0),
        _chunk("Auth B.", key="auth", pos=1),
    ]
    result = restructurer.restructure(chunks)
    assert len(result) == 2


def test_no_merge_when_combined_exceeds_max():
    config = Config(merge_by_keys=True, max_merged_size=20)
    restructurer = Restructurer(config)
    chunks = [
        _chunk("This is chunk one text.", key="topic", pos=0),
        _chunk("This is chunk two text.", key="topic", pos=1),
    ]
    result = restructurer.restructure(chunks)
    # Combined would be ~48 chars > max_merged_size=20 — no merge
    assert len(result) == 2


def test_keywords_merged_and_deduped(restructurer):
    c1 = _chunk("Alpha.", key="topic", pos=0)
    c1.keywords = ["alpha", "shared"]
    c2 = _chunk("Beta.", key="topic", pos=1)
    c2.keywords = ["beta", "shared"]
    result = restructurer.restructure([c1, c2])
    assert len(result) == 1
    kws = result[0].keywords
    assert "shared" in kws
    assert kws.count("shared") == 1


def test_entities_deduplicated(restructurer):
    c1 = _chunk("A.", key="topic", pos=0)
    c1.entities = [{"name": "Python", "type": "TECH"}]
    c2 = _chunk("B.", key="topic", pos=1)
    c2.entities = [{"name": "Python", "type": "TECH"},
                   {"name": "FastAPI", "type": "TECH"}]
    result = restructurer.restructure([c1, c2])
    entity_names = [e["name"] for e in result[0].entities]
    assert entity_names.count("Python") == 1
    assert "FastAPI" in entity_names


def test_empty_chunks(restructurer):
    assert restructurer.restructure([]) == []


def test_chunks_without_keys_pass_through(restructurer):
    chunks = [_chunk("No key here.", key="", pos=0)]
    result = restructurer.restructure(chunks)
    assert len(result) == 1
    assert result[0].text == "No key here."


def test_dedupe_entities_helper():
    entities = [
        {"name": "Python", "type": "TECH"},
        {"name": "python", "type": "TECH"},
        {"name": "FastAPI", "type": "TECH"},
    ]
    result = _dedupe_entities(entities)
    names = [e["name"] for e in result]
    assert len(names) == 2  # python deduped (case-insensitive)


# ─── New tests (Part 5 of optimization prompt) ─────────────────────────────


def test_global_merge_by_key(restructurer):
    """Same-key chunks at any distance should merge into one."""
    chunks = [
        _chunk("Auth intro.", key="oauth token flow", pos=0),
        _chunk("Database schema.", key="schema design", pos=1),
        _chunk("Caching layer.", key="cache strategy", pos=2),
        _chunk("Auth details.", key="oauth token flow", pos=20),
        _chunk("Auth conclusion.", key="oauth token flow", pos=47),
    ]
    result = restructurer.restructure(chunks)
    auth_chunks = [c for c in result if c.key == "oauth token flow"]
    assert len(auth_chunks) == 1
    assert "Auth intro" in auth_chunks[0].text
    assert "Auth details" in auth_chunks[0].text
    assert "Auth conclusion" in auth_chunks[0].text


def test_school_article_different_keys_stay_separate(restructurer):
    """Chunks about the same broad topic but different specific aspects must NOT merge.
    This is the core design principle — keys are specific subtopics, not document topics."""
    chunks = [
        _chunk("Admissions begin in March.", key="admissions process", pos=0),
        _chunk("Our campus spans 50 acres.", key="campus facilities", pos=1),
        _chunk("IB curriculum framework.", key="curriculum framework", pos=2),
        _chunk("Admissions criteria.", key="admissions process", pos=3),
    ]
    result = restructurer.restructure(chunks)
    keys = [c.key for c in result]
    assert keys.count("admissions process") == 1  # merged
    assert "campus facilities" in keys  # NOT merged with admissions
    assert "curriculum framework" in keys  # NOT merged with admissions
    assert len(result) == 3  # admissions(merged) + campus + curriculum


def test_unrelated_chunks_not_absorbed(restructurer):
    """Chunks with different keys must survive independently (no between-chunk absorption)."""
    chunks = [
        _chunk("Auth intro.", key="oauth flow", pos=0),
        _chunk("Database schema.", key="schema design", pos=1),
        _chunk("Auth details.", key="oauth flow", pos=2),
    ]
    result = restructurer.restructure(chunks)
    keys = [c.key for c in result]
    assert "schema design" in keys  # NOT eaten by oauth merge
    assert keys.count("oauth flow") == 1  # merged
    assert len(result) == 2  # oauth(merged) + schema


def test_orphan_chunk_gets_context():
    """Small keyless chunk should receive context header when below min_orphan_size."""
    config = Config(merge_by_keys=True, max_merged_size=3000,
                    min_orphan_size=200)
    restructurer = Restructurer(config)
    chunks = [
        _chunk("Previous chunk about databases. " *
               5, key="schema design", pos=0),
        _chunk("MIT License", key="", pos=1),  # tiny orphan (< 200 chars)
        _chunk("Next chunk about testing. " * 5, key="test strategy", pos=2),
    ]
    chunks[0].summary = "Database design overview"
    chunks[0].start_line = 0
    chunks[1].start_line = 10
    chunks[2].start_line = 20
    chunks[2].summary = "Testing framework setup"
    result = restructurer.restructure(chunks)
    orphan = [c for c in result if "MIT License" in c.text][0]
    assert orphan.key == ""  # still its own chunk, key unchanged
    # Context header should be prepended
    assert "[Previous:" in orphan.text or "[Next:" in orphan.text or "[Section:" in orphan.text
    # schema + orphan + test (no merging of different keys)
    assert len(result) == 3


def test_same_key_overflow_creates_bins():
    """If merging all same-key chunks exceeds max_merged_size, split into bins."""
    config = Config(merge_by_keys=True, max_merged_size=100)
    restructurer = Restructurer(config)
    chunks = [
        _chunk("A" * 40, key="aspect", pos=0),
        _chunk("B" * 40, key="aspect", pos=1),
        # 40+40+2=82 fits, +40+2=124 > 100
        _chunk("C" * 40, key="aspect", pos=2),
    ]
    chunks[0].start_line = 0
    chunks[1].start_line = 5
    chunks[2].start_line = 10
    result = restructurer.restructure(chunks)
    aspect_chunks = [c for c in result if c.key == "aspect"]
    assert len(aspect_chunks) == 2  # [A+B] and [C]


# ─── Chunk model tests ────────────────────────────────────────────────────


def test_to_dict_includes_empty_fields():
    """to_dict() must preserve empty lists and empty strings (not filter falsy values)."""
    chunk = Chunk(text="test")
    d = chunk.to_dict()
    assert "keywords" in d
    assert "entities" in d
    assert "related_keys" in d
    assert "key" in d
    assert "previous_chunk_id" in d
    assert d["keywords"] == []
    assert d["key"] == ""


def test_chunk_repr():
    """__repr__ should return a compact, readable string."""
    chunk = Chunk(text="hello", key="test key", title="Test Title")
    chunk.chunk_id = "abc123"
    r = repr(chunk)
    assert "abc123" in r
    assert "test key" in r
    assert "Test Title" in r


def test_generate_id_includes_key():
    """generate_id must incorporate the key field for better deduplication."""
    c1 = Chunk(text="same text", section_title="sec",
               key="key-a", position_index=0)
    c2 = Chunk(text="same text", section_title="sec",
               key="key-b", position_index=0)
    c1.generate_id()
    c2.generate_id()
    assert c1.chunk_id != c2.chunk_id  # different keys → different IDs
