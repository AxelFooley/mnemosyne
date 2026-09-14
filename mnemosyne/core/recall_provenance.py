"""Persistent recall provenance (query -> returned memory ids).

`recall_diagnostics` (mnemosyne/core/recall_diagnostics.py) keeps
in-process counters: per-tier hit counts and fallback rates over a
measurement window. Those counters answer "where does recall signal
come from?" but are lost when the process exits and never record
WHICH query pulled WHICH memories.

This module complements them with a persistent, queryable mapping:
when `BeamMemory.recall()` runs with MNEMOSYNE_RECALL_PROVENANCE=1,
one JSONL line is appended to `recall_provenance.jsonl` next to the
database, recording the query and the returned memory ids with their
scores. Operators can later audit which memories shaped a given
answer ("why did the agent say X?"). Records stay lean: ids and
scores only, no content previews.

Coverage note: the hook lives at the single final return of the
LINEAR recall path in beam.py. The `recall_enhanced()` and
polyphonic (MNEMOSYNE_POLYPHONIC_RECALL=1) delegation paths return
before reaching it and are NOT logged.

All writes are wrapped in try/except and logged at debug level:
provenance is audit signal and must NEVER break or slow recall.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


logger = logging.getLogger(__name__)

# Serializes appends from concurrent recall threads so each JSONL
# line stays intact (no interleaved partial writes).
_append_lock = threading.Lock()

# Hard cap on stored query text. Queries can be long prompts; the
# full text lives with the caller, the audit trail only needs to
# identify the call.
_MAX_QUERY_CHARS = 200

RESULT_FIELDS = ("id", "tier", "score", "importance", "timestamp")


def _provenance_path(db_path: Any) -> Path:
    return Path(db_path).parent / "recall_provenance.jsonl"


def append_recall_provenance(db_path: Any, query: str,
                             results: List[Dict], top_k: int) -> None:
    """Append one JSONL provenance record for a recall() call.

    Writes `{"ts", "query", "top_k", "results"}` to
    `<db_path dir>/recall_provenance.jsonl`, where each entry in
    `results` carries only id/tier/score/importance/timestamp (no
    content preview). Never raises: failures are logged at debug
    level and swallowed so recall behavior is unaffected.
    """
    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "query": str(query or "")[:_MAX_QUERY_CHARS],
            "top_k": int(top_k),
            "results": [
                {field: r.get(field) for field in RESULT_FIELDS}
                for r in (results or [])
                if isinstance(r, dict)
            ],
        }
        with _append_lock:
            with open(_provenance_path(db_path), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("recall provenance append failed (non-fatal)", exc_info=True)


def read_recall_provenance(db_path: Any, limit: int = 20) -> List[Dict]:
    """Read provenance records, newest first.

    Returns at most `limit` records taken from the end of the file
    (the most recent calls), reversed so the newest is first.
    Malformed lines are skipped. Missing file -> [].
    """
    path = _provenance_path(db_path)
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if line.strip()]
    except Exception:
        logger.debug("recall provenance read failed (non-fatal)", exc_info=True)
        return []
    records: List[Dict] = []
    for line in lines[-max(0, int(limit)):]:
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    records.reverse()
    return records
