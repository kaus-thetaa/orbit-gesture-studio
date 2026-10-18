import socket
import time

import pytest

from orbit.brain.providers import (AuthError, GeminiProvider, HttpClient, LLMRequest, OllamaProvider, ProviderError,
                                   ProviderUnavailable, RateLimited)
from fake_llm_server import FakeServer, STATE


@pytest.fixture
def server():
    with FakeServer() as s:
        yield s


REQ = LLMRequest("Be brief.", [("user", "snacks?")], max_tokens=40)


def gem(server, key="test-key", model="gemini-3.1-flash-lite", timeout=3.0):
    return GeminiProvider(model, lambda: key, server.url, timeout)


def oll(server, model="llama3.2:3b", timeout=3.0):
    return OllamaProvider(model, server.url, "30m", timeout)


# ------------------------------------------------------------------ Gemini
def test_gemini_request_shape_and_reply(server):
    r = gem(server).generate(REQ)
    assert r.text == "Samosa and tea" and r.provider == "gemini" and r.tokens_in == 40 and r.tokens_out == 5
    method, path, headers, body = STATE.requests[-1]
    assert path == "/v1beta/models/gemini-3.1-flash-lite:generateContent"
    assert headers["x-goog-api-key"] == "test-key"                    # key in header, never in the URL
    assert "key=" not in path
    assert body["system_instruction"]["parts"][0]["text"] == "Be brief."
    assert body["contents"] == [{"role": "user", "parts": [{"text": "snacks?"}]}]
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert body["generationConfig"]["maxOutputTokens"] >= 40


def test_gemini_assistant_role_is_mapped_to_model(server):
    gem(server).generate(LLMRequest("s", [("user", "a"), ("assistant", "b"), ("user", "c")]))
    assert [c["role"] for c in STATE.requests[-1][3]["contents"]] == ["user", "model", "user"]


def test_gemini_thinking_variants_fall_back(server):
    STATE.reject_thinking = {"minimal", "low"}
    g = gem(server)
    assert g.generate(REQ).text
    levels = [r[3]["generationConfig"].get("thinkingConfig") for r in STATE.requests if r[0] == "POST"]
    assert levels == [{"thinkingLevel": "minimal"}, {"thinkingLevel": "low"}, {"thinkingBudget": 0}]
    n = len(STATE.requests)
    g.generate(REQ)                                                   # remembers the working variant
    assert len(STATE.requests) == n + 1


def test_gemini_unknown_model_is_auto_discovered(server):
    g = gem(server, model="gemini-1.0-retired")
    assert g.generate(REQ).model == "gemini-3.1-flash-lite"           # picked a flash-lite from ListModels
    assert not any("embedding" in str(r) for r in STATE.requests if r[0] == "POST")


@pytest.mark.parametrize("status,exc", [(429, RateLimited), (403, AuthError), (503, ProviderUnavailable), (400, ProviderError)])
def test_gemini_error_mapping(server, status, exc):
    STATE.gemini_status = status
    with pytest.raises(exc) as ei:
        gem(server).generate(REQ)
    if status == 429:
        assert ei.value.retry_after == 7


def test_gemini_missing_key_never_hits_network(server):
    with pytest.raises(AuthError):
        gem(server, key=None).generate(REQ)
    assert STATE.requests == []


def test_gemini_wrong_key(server):
    with pytest.raises(AuthError):
        gem(server, key="nope").generate(REQ)


def test_gemini_timeout_is_unavailable(server):
    STATE.gemini_delay = 1.0
    t0 = time.perf_counter()
    with pytest.raises(ProviderUnavailable):
        gem(server, timeout=0.3).generate(REQ)
    assert time.perf_counter() - t0 < 0.9


def test_gemini_connection_refused_is_unavailable():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    with pytest.raises(ProviderUnavailable):
        GeminiProvider("m", lambda: "k", f"http://127.0.0.1:{port}", 1.0).generate(REQ)


def test_persistent_connection_is_reused(server):
    g = gem(server)
    for _ in range(4):
        g.generate(REQ)
    assert STATE.connections == 1


def test_reconnects_after_server_drops_connection(server):
    g = gem(server)
    g.generate(REQ)
    g.http._conn.sock.close()             # simulate the server/NAT closing an idle connection
    assert g.generate(REQ).text


# ------------------------------------------------------------------ Ollama
def test_ollama_request_shape(server):
    r = oll(server).generate(REQ)
    assert r.text == "Tea at 4:30" and r.provider == "ollama"
    body = STATE.requests[-1][3]
    assert body["stream"] is False and body["keep_alive"] == "30m" and body["think"] is False
    assert body["options"]["num_predict"] == 40
    assert body["messages"][0] == {"role": "system", "content": "Be brief."}


def test_ollama_think_flag_dropped_if_unsupported(server):
    STATE.ollama_reject_think = True
    o = oll(server)
    assert o.generate(REQ).text
    assert "think" not in STATE.requests[-1][3]
    n = len(STATE.requests)
    o.generate(REQ)
    assert len(STATE.requests) == n + 1 and "think" not in STATE.requests[-1][3]


def test_ollama_missing_model_uses_smallest_installed(server):
    o = oll(server, model="not-installed:9b")
    assert o.generate(REQ).model == "tiny:1b"


def test_ollama_no_models_and_not_running(server):
    STATE.ollama_models = []
    with pytest.raises(ProviderUnavailable):
        oll(server).generate(REQ)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    with pytest.raises(ProviderUnavailable):
        OllamaProvider("m", f"http://127.0.0.1:{port}", timeout=1.0).generate(REQ)


def test_ollama_warm_sends_empty_chat(server):
    oll(server).warm()
    assert STATE.requests[-1][3]["messages"] == []


def test_json_mode_flags(server):
    gem(server).generate(LLMRequest("s", [("user", "x")], json_mode=True))
    assert STATE.requests[-1][3]["generationConfig"]["responseMimeType"] == "application/json"
    oll(server).generate(LLMRequest("s", [("user", "x")], json_mode=True))
    assert STATE.requests[-1][3]["format"] == "json"


def test_http_client_rejects_bad_urls():
    with pytest.raises(ValueError):
        HttpClient("ftp://example.com")


def test_ollama_picks_general_model_over_code_model(server):
    from orbit.brain.providers import pick_ollama_model
    mine = [("qwen2.5-coder:1.5b", 986_000_000), ("qwen2.5-coder:7b", 4_700_000_000), ("qwen2.5-coder:latest", 4_700_000_000), ("llama3.2:latest", 2_000_000_000)]
    assert pick_ollama_model("llama3.2:3b", mine) == "llama3.2:latest"          # same family, different tag
    assert pick_ollama_model("mistral", mine) == "llama3.2:latest"              # smallest GENERAL model, not the 1.5b coder
    assert pick_ollama_model("qwen2.5-coder:7b", mine) == "qwen2.5-coder:7b"    # explicit choice is respected
    assert pick_ollama_model("x", [("qwen2.5-coder:1.5b", 1)]) == "qwen2.5-coder:1.5b"   # only option left
    assert pick_ollama_model("x", []) is None
    STATE.ollama_models = mine
    assert oll(server, model="llama3.2:3b").generate(REQ).model == "llama3.2:latest"
