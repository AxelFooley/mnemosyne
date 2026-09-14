"""
Tests for persistent recall provenance logging (mnemosyne/core/recall_provenance.py).

When MNEMOSYNE_RECALL_PROVENANCE=1, BeamMemory.recall() appends one JSONL
line per call next to the db, recording query + returned ids/scores. The
flag is read per call; default OFF means no file is ever created.
"""

import json
import tempfile
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.recall_provenance import (
    append_recall_provenance,
    read_recall_provenance,
)


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield db_path


def _provenance_file(db_path: Path) -> Path:
    return db_path.parent / "recall_provenance.jsonl"


def _read_last_record(db_path: Path) -> dict:
    lines = _provenance_file(db_path).read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1])


def test_flag_on_writes_provenance_line(temp_db, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_RECALL_PROVENANCE", "1")
    beam = BeamMemory(session_id="prov-a", db_path=temp_db)
    id1 = beam.remember("pluto alpha provenance fact one", source="test")
    id2 = beam.remember("pluto beta provenance fact two", source="test")

    # Long query (>200 chars) with the shared token early, so the
    # truncated stored query still matches both memories via FTS.
    query = "pluto " + "padding " * 30
    assert len(query) > 200
    results = beam.recall(query, top_k=5)

    prov_file = _provenance_file(temp_db)
    assert prov_file.exists()
    record = _read_last_record(temp_db)
    assert record["query"] == query[:200]
    assert record["top_k"] == 5
    assert isinstance(record["ts"], str)
    returned_ids = {r["id"] for r in record["results"]}
    assert {id1, id2} <= returned_ids
    for entry in record["results"]:
        assert "id" in entry
        assert "tier" in entry
        assert "score" in entry
        assert "importance" in entry
        assert entry["importance"] is not None
        assert "timestamp" in entry
        assert entry["timestamp"] is not None
    # Ids recorded are exactly what recall returned, in order.
    assert [r["id"] for r in record["results"]] == [r["id"] for r in results]


def test_flag_off_by_default_writes_no_file(temp_db, monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_RECALL_PROVENANCE", raising=False)
    beam = BeamMemory(session_id="prov-b", db_path=temp_db)
    beam.remember("pluto gamma provenance fact three", source="test")
    beam.recall("pluto", top_k=5)

    assert not _provenance_file(temp_db).exists()


def test_flag_garbage_value_treated_as_off(temp_db, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_RECALL_PROVENANCE", "yes")
    beam = BeamMemory(session_id="prov-c", db_path=temp_db)
    beam.remember("pluto delta provenance fact four", source="test")
    beam.recall("pluto", top_k=5)

    assert not _provenance_file(temp_db).exists()


def test_read_recall_provenance_newest_first_and_limit(temp_db, monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_RECALL_PROVENANCE", raising=False)

    assert read_recall_provenance(temp_db) == []

    for i in range(4):
        append_recall_provenance(
            str(temp_db), f"query-{i}", [{"id": f"m{i}", "tier": "working",
                                          "score": 1.0}], top_k=5,
        )

    records = read_recall_provenance(temp_db, limit=20)
    assert [r["query"] for r in records] == [
        "query-3", "query-2", "query-1", "query-0",
    ]

    limited = read_recall_provenance(temp_db, limit=2)
    assert len(limited) == 2
    assert [r["query"] for r in limited] == ["query-3", "query-2"]


def test_read_recall_provenance_zero_or_negative_limit_returns_empty(
        temp_db, monkeypatch):
    """limit<=0 must return [], not the whole file (-0 == 0 slicing bug)."""
    monkeypatch.delenv("MNEMOSYNE_RECALL_PROVENANCE", raising=False)
    for i in range(3):
        append_recall_provenance(
            str(temp_db), f"query-{i}", [{"id": f"m{i}", "tier": "working",
                                          "score": 1.0}], top_k=5,
        )

    assert read_recall_provenance(temp_db, limit=0) == []
    assert read_recall_provenance(temp_db, limit=-3) == []


def test_read_recall_provenance_skips_malformed_lines(temp_db, monkeypatch):
    """Malformed JSONL lines are skipped; valid neighbors still returned."""
    monkeypatch.delenv("MNEMOSYNE_RECALL_PROVENANCE", raising=False)
    append_recall_provenance(
        str(temp_db), "good-1", [{"id": "m1", "tier": "working",
                                  "score": 1.0}], top_k=5,
    )
    with open(_provenance_file(temp_db), "a", encoding="utf-8") as f:
        f.write("{not valid json\n")
    append_recall_provenance(
        str(temp_db), "good-2", [{"id": "m2", "tier": "working",
                                  "score": 1.0}], top_k=5,
    )

    records = read_recall_provenance(temp_db, limit=10)
    assert [r["query"] for r in records] == ["good-2", "good-1"]


def test_explain_recall_writes_no_provenance_line(temp_db, monkeypatch):
    """explain=True returns via the explain-trace early return, before the
    provenance hook, so no JSONL line is written even with the flag on."""
    monkeypatch.setenv("MNEMOSYNE_RECALL_PROVENANCE", "1")
    beam = BeamMemory(session_id="prov-e", db_path=temp_db)
    beam.remember("pluto zeta provenance fact six", source="test")

    explained = beam.recall("pluto", top_k=5, explain=True)
    assert explained["engine"] == "linear"
    assert not _provenance_file(temp_db).exists()


def test_enhanced_recall_delegation_writes_no_provenance_line(
        temp_db, monkeypatch):
    """Under MNEMOSYNE_ENHANCED_RECALL=1, the recall_enhanced() cache-miss
    delegation into self.recall() must not log provenance: its expanded
    query + doubled top_k are internal, not the caller's request."""
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_RECALL_PROVENANCE", "1")
    beam = BeamMemory(session_id="prov-f", db_path=temp_db)
    beam.remember("pluto eta provenance fact seven", source="test")

    # First call is always a cache miss: the linear path really runs.
    results = beam.recall_enhanced(
        "pluto", top_k=5,
        use_intent=False, use_synonyms=False, use_weibull=False, use_mmr=False,
    )
    assert results

    assert not _provenance_file(temp_db).exists()


def test_flag_read_per_call(temp_db, monkeypatch):
    """The flag is consulted on every recall call, not cached at init."""
    monkeypatch.delenv("MNEMOSYNE_RECALL_PROVENANCE", raising=False)
    beam = BeamMemory(session_id="prov-d", db_path=temp_db)
    beam.remember("pluto epsilon provenance fact five", source="test")

    beam.recall("pluto", top_k=5)
    assert not _provenance_file(temp_db).exists()

    monkeypatch.setenv("MNEMOSYNE_RECALL_PROVENANCE", "1")
    beam.recall("pluto", top_k=5)
    assert _provenance_file(temp_db).exists()

    monkeypatch.setenv("MNEMOSYNE_RECALL_PROVENANCE", "0")
    beam.recall("pluto", top_k=5)
    lines = _provenance_file(temp_db).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
