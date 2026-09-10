"""Provider fallback + endpoint failover tests (all offline, no network)."""

import pytest

from app.services.agent import provider
from app.services.agent.failover import ainvoke_with_failover


class _Graph:
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = 0

    async def ainvoke(self, payload, config):
        self.calls += 1
        if isinstance(self.behavior, Exception):
            raise self.behavior
        return self.behavior


def _run(coro):
    import asyncio

    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Provider construction (no network: object creation only)
# ---------------------------------------------------------------------------

def test_no_keys_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_FALLBACK", raising=False)
    with pytest.raises(provider.LLMNotConfigured):
        provider.get_chat_model()
    with pytest.raises(provider.LLMNotConfigured):
        provider.get_fallback_model()
    assert not provider.has_fallback()


def test_primary_is_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-or")
    monkeypatch.delenv("OPENROUTER_API_KEY_FALLBACK", raising=False)
    assert not provider.has_fallback()
    model = provider.get_chat_model()
    assert type(model).__name__ == "ChatOpenAI"
    assert "openrouter" in str(model.openai_api_base)


def test_fallback_is_secondary_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY_FALLBACK", "dummy-or-2")
    assert provider.has_fallback()
    model = provider.get_fallback_model()
    assert type(model).__name__ == "ChatOpenAI"
    assert "openrouter" in str(model.openai_api_base)


def test_is_capacity_error():
    assert provider.is_capacity_error(Exception("[503] ResourceExhausted: busy"))
    assert provider.is_capacity_error(Exception("429 rate_limit_exceeded"))
    assert provider.is_capacity_error(Exception("Timeout on reading data from socket"))
    assert not provider.is_capacity_error(ValueError("bad key"))


# ---------------------------------------------------------------------------
# ainvoke_with_failover
# ---------------------------------------------------------------------------

def test_primary_success_no_failover(monkeypatch):
    monkeypatch.setattr(provider, "has_fallback", lambda: True)
    graph = _Graph("RESULT")
    result, used, out_graph = _run(
        ainvoke_with_failover(lambda *a, **k: graph, {"x": 1}, {})
    )
    assert (result, used, out_graph) == ("RESULT", "primary", graph)
    assert graph.calls == 1


def test_failover_on_capacity_error(monkeypatch):
    monkeypatch.setattr(provider, "has_fallback", lambda: True)
    monkeypatch.setattr(provider, "get_fallback_model", lambda: "fb-model")
    seen = []

    def build(model=None):
        seen.append(model)
        if model is None:
            return _Graph(Exception("[503] ResourceExhausted"))
        return _Graph("FB-RESULT")

    result, used, _ = _run(ainvoke_with_failover(build, {"x": 1}, {}))
    assert result == "FB-RESULT" and used == "fallback"
    assert seen == [None, "fb-model"]


def test_non_capacity_error_propagates(monkeypatch):
    monkeypatch.setattr(provider, "has_fallback", lambda: True)
    builds = []

    def build(model=None):
        builds.append(model)
        return _Graph(ValueError("auth failed"))

    with pytest.raises(ValueError, match="auth failed"):
        _run(ainvoke_with_failover(build, {}, {}))
    assert builds == [None]


def test_no_fallback_configured_reraises(monkeypatch):
    monkeypatch.setattr(provider, "has_fallback", lambda: False)
    builds = []

    def build(model=None):
        builds.append(model)
        return _Graph(Exception("ResourceExhausted"))

    with pytest.raises(Exception, match="ResourceExhausted"):
        _run(ainvoke_with_failover(build, {}, {}))
    assert builds == [None]
