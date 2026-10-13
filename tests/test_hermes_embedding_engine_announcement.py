"""Fail-loud contract for an unavailable embedding engine (bundled provider).

Mirrors integrations/hermes/tests/test_embedding_engine_announcement.py for
the standalone bundled copy: the engine's optional dependencies (fastembed,
sqlite_vec) were not importable in a gateway's runtime interpreter, every
embedding write was silently skipped by the ``if available():`` guard, and the
deployment ran ~30 days with dense/semantic recall off without a single log
line. The bundled provider announces the degraded mode once per process at
startup, stating the consequence and the fix, and never raises.
"""

import logging

import pytest

import hermes_memory_provider
from hermes_memory_provider import MnemosyneMemoryProvider


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
        hermes_memory_provider, "_EMBEDDING_ENGINE_UNAVAILABLE_LOGGED", False,
        raising=False,
    )
    monkeypatch.setattr(embeddings_module, "available", lambda: dense)
    monkeypatch.setattr(beam_module, "_SQLITE_VEC_AVAILABLE", vec)
    # Clear the env opt-out aliases so the missing-dependency branch runs.
    for name in ("MNEMOSYNE_NO_EMBEDDINGS", "MNEMOSYNE_SKIP_EMBEDDINGS", "MNEMOSYNE_EMBEDDINGS_OFF"):
        monkeypatch.delenv(name, raising=False)

def test_provider_announces_unavailable_embedding_engine_once(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(hermes_memory_provider, "_get_beam_class", lambda: _OkBeam)
    monkeypatch.setattr(MnemosyneMemoryProvider, "_init_audit_log", lambda self: None)
    provider = MnemosyneMemoryProvider()
    second = None
    kwargs = {"hermes_home": str(tmp_path), "auto_sleep": False}
    try:
        _engine_capability(monkeypatch, dense=False, vec=False)
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
            provider.initialize("sess", **kwargs)

            # The consequence, the hint and both missing halves are named, once.
            assert caplog.text.count("dense/semantic recall is disabled") == 1
            assert "fastembed" in caplog.text
            assert "sqlite_vec" in caplog.text
            assert "keyword-only" in caplog.text
            assert "extra importable" in caplog.text

            # Once per process: a second provider start must not re-announce.
            second = MnemosyneMemoryProvider()
            second.initialize("sess-2", **kwargs)
            assert caplog.text.count("dense/semantic recall is disabled") == 1

        # Engine available -> no announcement at all.
        _engine_capability(monkeypatch, dense=True, vec=True)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
            provider.initialize("sess-3", **kwargs)
            assert "dense/semantic recall is disabled" not in caplog.text
    finally:
        provider.shutdown()
        if second is not None:
            second.shutdown()


def test_provider_announces_missing_fastembed_only(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(hermes_memory_provider, "_get_beam_class", lambda: _OkBeam)
    monkeypatch.setattr(MnemosyneMemoryProvider, "_init_audit_log", lambda self: None)
    provider = MnemosyneMemoryProvider()
    try:
        _engine_capability(monkeypatch, dense=False, vec=True)
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
            provider.initialize("sess", **{"hermes_home": str(tmp_path), "auto_sleep": False})
            assert "missing fastembed" in caplog.text
            assert "sqlite_vec" not in caplog.text
            assert "extra importable" in caplog.text
    finally:
        provider.shutdown()


def test_provider_announces_missing_sqlite_vec_only(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(hermes_memory_provider, "_get_beam_class", lambda: _OkBeam)
    monkeypatch.setattr(MnemosyneMemoryProvider, "_init_audit_log", lambda self: None)
    provider = MnemosyneMemoryProvider()
    try:
        _engine_capability(monkeypatch, dense=True, vec=False)
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
            provider.initialize("sess", **{"hermes_home": str(tmp_path), "auto_sleep": False})
            assert "missing sqlite_vec" in caplog.text
            assert "fastembed" not in caplog.text
            assert "extra importable" in caplog.text
    finally:
        provider.shutdown()


def test_provider_reports_opt_out_without_install_hint(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(hermes_memory_provider, "_get_beam_class", lambda: _OkBeam)
    monkeypatch.setattr(MnemosyneMemoryProvider, "_init_audit_log", lambda self: None)
    provider = MnemosyneMemoryProvider()
    try:
        _engine_capability(monkeypatch, dense=False, vec=False)
        monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
            provider.initialize("sess", **{"hermes_home": str(tmp_path), "auto_sleep": False})
            assert "opt-out env var" in caplog.text
            assert "keyword-only" in caplog.text
            assert "extra importable" not in caplog.text
            assert "missing fastembed" not in caplog.text
            assert "missing sqlite_vec" not in caplog.text
    finally:
        provider.shutdown()


def test_provider_announces_api_backend_without_credentials(monkeypatch, tmp_path, caplog):
    from mnemosyne.core import beam as beam_module
    from mnemosyne.core import embeddings as embeddings_module

    monkeypatch.setattr(hermes_memory_provider, "_get_beam_class", lambda: _OkBeam)
    monkeypatch.setattr(MnemosyneMemoryProvider, "_init_audit_log", lambda self: None)
    provider = MnemosyneMemoryProvider()
    try:
        monkeypatch.setattr(
            hermes_memory_provider, "_EMBEDDING_ENGINE_UNAVAILABLE_LOGGED", False,
            raising=False,
        )
        monkeypatch.setattr(embeddings_module, "_DEFAULT_MODEL", "openai/text-embedding-3-small")
        monkeypatch.setattr(embeddings_module, "_OPENAI_API_KEY", "")
        monkeypatch.setattr(embeddings_module, "_FASTEMBED_AVAILABLE", True)
        monkeypatch.setattr(beam_module, "_SQLITE_VEC_AVAILABLE", True)
        for name in (
            "MNEMOSYNE_NO_EMBEDDINGS",
            "MNEMOSYNE_SKIP_EMBEDDINGS",
            "MNEMOSYNE_EMBEDDINGS_OFF",
            "MNEMOSYNE_EMBEDDING_API_URL",
        ):
            monkeypatch.delenv(name, raising=False)
        with caplog.at_level(logging.WARNING, logger="hermes_memory_provider"):
            provider.initialize("sess", **{"hermes_home": str(tmp_path), "auto_sleep": False})
            assert "embedding API backend has no usable credentials" in caplog.text
            assert "MNEMOSYNE_EMBEDDING_API_KEY" in caplog.text
            assert "missing fastembed" not in caplog.text
            assert "extra importable" not in caplog.text
    finally:
        provider.shutdown()
