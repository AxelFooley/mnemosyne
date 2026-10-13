"""Fail-loud contract for an unavailable embedding engine (Hermes provider).

Regression for the 2026-10-13 production incident: the engine's optional
dependencies (fastembed, sqlite_vec) were not importable in the gateway's
runtime interpreter, every embedding write was silently skipped by the
``if available():`` guard, and the deployment ran ~30 days with dense/semantic
recall off without a single log line. Modeled on the fail-loud tests in
tests/test_embedding_api_retry.py (the PR #1144 contract): the provider
announces the degraded mode once per process at startup, stating the
consequence and the fix, and never raises.
"""

import logging

import pytest

import mnemosyne_hermes
from mnemosyne_hermes import MnemosyneMemoryProvider


@pytest.fixture(autouse=True)
def _isolate_provider_config(monkeypatch, tmp_path):
    from mnemosyne.core.config import MnemosyneConfig

    core_config_path = tmp_path / "mnemosyne-config.yaml"
    core_config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        MnemosyneConfig,
        "_instance",
        MnemosyneConfig(config_path=core_config_path),
    )


class _OkBeam:
    """Minimal BeamMemory stand-in: accepts init kwargs, does nothing."""

    author_id = None

    def __init__(self, *args, **kwargs):
        pass


def _engine_capability(monkeypatch, *, dense: bool, vec: bool) -> None:
    """Pin the capability predicates; no real dependency is required."""
    from mnemosyne.core import beam as beam_module
    from mnemosyne.core import embeddings as embeddings_module

    # Reset the once-per-process guard so the test is order-independent.
    monkeypatch.setattr(
        mnemosyne_hermes, "_EMBEDDING_ENGINE_UNAVAILABLE_LOGGED", False,
        raising=False,
    )
    monkeypatch.setattr(embeddings_module, "available", lambda: dense)
    monkeypatch.setattr(beam_module, "_SQLITE_VEC_AVAILABLE", vec)


def test_provider_announces_unavailable_embedding_engine_once(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _OkBeam)
    monkeypatch.setattr(MnemosyneMemoryProvider, "_init_audit_log", lambda self: None)
    provider = MnemosyneMemoryProvider()
    kwargs = {"hermes_home": str(tmp_path), "auto_sleep": False}

    _engine_capability(monkeypatch, dense=False, vec=False)
    with caplog.at_level(logging.WARNING, logger="mnemosyne_hermes"):
        provider.initialize("sess", **kwargs)

        # The consequence, the hint and both missing halves are named, once.
        assert caplog.text.count("dense/semantic recall is disabled") == 1
        assert "fastembed" in caplog.text
        assert "sqlite_vec" in caplog.text
        assert "keyword-only" in caplog.text
        assert "extra importable" in caplog.text

        # Once per process: a second provider start must not re-announce.
        MnemosyneMemoryProvider().initialize("sess-2", **kwargs)
        assert caplog.text.count("dense/semantic recall is disabled") == 1

    # Engine available -> no announcement at all.
    _engine_capability(monkeypatch, dense=True, vec=True)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mnemosyne_hermes"):
        provider.initialize("sess-3", **kwargs)
        assert "dense/semantic recall is disabled" not in caplog.text
