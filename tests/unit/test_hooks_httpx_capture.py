"""Tests for the httpx body-capture feature in hooks.py.

Covers the pieces added by "capture full HTTP request/response bodies in
API spans":

- ``_parse_sse_response`` — SSE stream reconstruction into a JSON response
- ``_capture_response`` — raw + SSE-parsed storage
- ``_take_captured_bodies`` — thread-keyed retrieval that preserves other
  in-flight threads' captures (concurrency safety)
- ``_install_httpx_interceptor`` — live monkey-patch that buffers + re-wraps
  the response so the caller can still read it, and closes the original
- gating in ``on_pre_api_request`` — only patch httpx when a capture flag is on
- ``on_post_api_request`` — consumes the SSE-parsed body and only its own
  thread's entry
"""

from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import patch

import hermes_otel.hooks as hooks
import pytest
from hermes_otel.hooks import on_post_api_request, on_pre_api_request

# The interceptor targets httpx; skip the whole module if it isn't installed
# (matches the project's optional-dependency convention).
httpx = pytest.importorskip("httpx")


def _sse(*chunks: dict) -> str:
    """Build an SSE body from chunk dicts, terminated by ``[DONE]``."""
    lines = [f"data: {json.dumps(c)}" for c in chunks]
    lines.append("data: [DONE]")
    return "\n".join(lines) + "\n"


@pytest.fixture()
def clean_httpx_state():
    """Isolate the module-global httpx interceptor state per test.

    Saves/restores the (possibly monkey-patched) ``httpx.Client.send`` /
    ``AsyncClient.send`` and resets the ``_httpx_patched`` flag and the
    ``_httpx_bodies`` capture buffer so tests don't leak into each other.
    """
    orig_client_send = httpx.Client.send
    orig_async_send = httpx.AsyncClient.send
    orig_patched = hooks._httpx_patched
    hooks._httpx_patched = False
    hooks._httpx_bodies.clear()
    try:
        yield
    finally:
        httpx.Client.send = orig_client_send
        httpx.AsyncClient.send = orig_async_send
        hooks._httpx_patched = orig_patched
        hooks._httpx_bodies.clear()


@pytest.fixture()
def mock_tracer():
    """Mock tracer with a real SessionState + default config, like the
    fixture in ``test_hooks_callbacks.py``."""
    from unittest.mock import MagicMock

    from hermes_otel.plugin_config import HermesOtelConfig
    from hermes_otel.session_state import SessionState

    tracer = MagicMock()
    tracer.is_enabled = True
    tracer.spans = MagicMock()
    tracer.spans._active_spans = {}
    tracer.sessions = SessionState()
    tracer.config = HermesOtelConfig()
    with patch("hermes_otel.hooks.get_tracer", return_value=tracer):
        yield tracer


def _pre_kwargs(**extra):
    base = dict(
        task_id="t1",
        session_id="s1",
        platform="cli",
        model="gpt-4",
        provider="openai",
        base_url="",
        api_mode="chat",
        api_call_count=1,
        message_count=2,
        tool_count=0,
        approx_input_tokens=10,
        request_char_count=40,
        max_tokens=0,
    )
    base.update(extra)
    return base


def _post_kwargs(**extra):
    base = dict(
        task_id="t1",
        session_id="s1",
        platform="cli",
        model="gpt-4",
        provider="openai",
        base_url="",
        api_mode="chat",
        api_call_count=1,
        api_duration=0.1,
        finish_reason="stop",
        message_count=2,
        response_model="gpt-4",
        usage={},
        assistant_content_chars=5,
        assistant_tool_call_count=0,
    )
    base.update(extra)
    return base


class TestParseSseResponse:
    def test_non_sse_returns_none(self):
        assert hooks._parse_sse_response('{"choices": []}') is None

    def test_empty_or_blank_returns_none(self):
        assert hooks._parse_sse_response("") is None
        assert hooks._parse_sse_response("   \n  ") is None

    def test_reconstructs_content_across_chunks(self):
        sse = _sse(
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
        )
        out = json.loads(hooks._parse_sse_response(sse))
        assert out["choices"][0]["message"]["content"] == "Hello"
        assert out["choices"][0]["finish_reason"] == "stop"

    def test_merges_tool_calls_by_index(self):
        sse = _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "web_", "arguments": '{"q":'},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"name": "search", "arguments": '"x"}'}}
                            ]
                        }
                    }
                ]
            },
        )
        out = json.loads(hooks._parse_sse_response(sse))
        tc = out["choices"][0]["message"]["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "web_search"
        assert tc["function"]["arguments"] == '{"q":"x"}'

    def test_multiple_tool_calls_distinct_indexes_sorted(self):
        sse = _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "b",
                                    "function": {"name": "second", "arguments": "{}"},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "a",
                                    "function": {"name": "first", "arguments": "{}"},
                                }
                            ]
                        }
                    }
                ]
            },
        )
        out = json.loads(hooks._parse_sse_response(sse))
        names = [tc["function"]["name"] for tc in out["choices"][0]["message"]["tool_calls"]]
        assert names == ["first", "second"]  # sorted by index

    def test_captures_model_and_usage(self):
        sse = _sse(
            {
                "model": "gpt-4o",
                "choices": [{"delta": {"content": "hi"}}],
                "usage": {"total_tokens": 7},
            },
        )
        out = json.loads(hooks._parse_sse_response(sse))
        assert out["model"] == "gpt-4o"
        assert out["usage"]["total_tokens"] == 7

    def test_finish_reason_propagates(self):
        sse = _sse({"choices": [{"delta": {"content": "x"}, "finish_reason": "length"}]})
        out = json.loads(hooks._parse_sse_response(sse))
        assert out["choices"][0]["finish_reason"] == "length"

    def test_skips_malformed_data_lines(self):
        sse = "data: not-json\n" 'data: {"choices":[{"delta":{"content":"ok"}}]}\n' "data: [DONE]\n"
        out = json.loads(hooks._parse_sse_response(sse))
        assert out["choices"][0]["message"]["content"] == "ok"

    def test_role_only_delta_yields_none(self):
        # No content and no tool_calls => empty message => None.
        sse = _sse({"choices": [{"delta": {"role": "assistant"}}]})
        assert hooks._parse_sse_response(sse) is None


class TestCaptureResponse:
    def test_plain_json_has_no_sse_key(self, clean_httpx_state):
        hooks._capture_response("T", '{"choices": []}')
        entry = hooks._httpx_bodies["T"]
        assert entry["response"] == '{"choices": []}'
        assert "response_sse_parsed" not in entry

    def test_sse_stores_raw_and_parsed(self, clean_httpx_state):
        sse = _sse({"choices": [{"delta": {"content": "hi"}}]})
        hooks._capture_response("T", sse)
        entry = hooks._httpx_bodies["T"]
        assert entry["response"] == sse  # raw kept
        parsed = json.loads(entry["response_sse_parsed"])
        assert parsed["choices"][0]["message"]["content"] == "hi"

    def test_preserves_existing_request_entry(self, clean_httpx_state):
        hooks._httpx_bodies["T"] = {"request": "the-request"}
        hooks._capture_response("T", '{"ok": 1}')
        assert hooks._httpx_bodies["T"]["request"] == "the-request"
        assert hooks._httpx_bodies["T"]["response"] == '{"ok": 1}'


class TestTakeCapturedBodies:
    def test_returns_and_removes_current_thread_entry(self, clean_httpx_state):
        hooks._httpx_bodies["T"] = {"request": "r"}
        assert hooks._take_captured_bodies("T") == {"request": "r"}
        assert "T" not in hooks._httpx_bodies

    def test_falls_back_to_most_recent_when_missing(self, clean_httpx_state):
        hooks._httpx_bodies["A"] = {"request": "a"}
        hooks._httpx_bodies["B"] = {"request": "b"}  # inserted last == newest
        assert hooks._take_captured_bodies("missing") == {"request": "b"}
        assert "B" not in hooks._httpx_bodies
        assert "A" in hooks._httpx_bodies  # only the consumed one is removed

    def test_preserves_other_threads_entries(self, clean_httpx_state):
        hooks._httpx_bodies["A"] = {"request": "a"}
        hooks._httpx_bodies["B"] = {"request": "b"}
        hooks._take_captured_bodies("A")
        assert list(hooks._httpx_bodies) == ["B"]

    def test_empty_returns_empty_dict(self, clean_httpx_state):
        assert hooks._take_captured_bodies("anything") == {}


class TestHttpxInterceptorLive:
    def test_captures_request_and_response_and_caller_can_read(self, clean_httpx_state):
        def handler(request):
            return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

        hooks._install_httpx_interceptor()
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            resp = client.post(
                "http://test/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "yo"}]},
            )

        # Re-wrapped response is still fully readable by the caller.
        assert resp.json()["choices"][0]["message"]["content"] == "hi"

        entry = hooks._take_captured_bodies(threading.current_thread().name)
        assert json.loads(entry["request"])["messages"][0]["content"] == "yo"
        assert json.loads(entry["response"])["choices"][0]["message"]["content"] == "hi"

    def test_parses_sse_response_and_preserves_raw_for_caller(self, clean_httpx_state):
        sse = _sse(
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
        )

        def handler(request):
            return httpx.Response(
                200, content=sse.encode(), headers={"content-type": "text/event-stream"}
            )

        hooks._install_httpx_interceptor()
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            resp = client.post("http://test", json={"messages": []})

        assert resp.text == sse  # caller still sees the raw SSE stream
        entry = hooks._take_captured_bodies(threading.current_thread().name)
        parsed = json.loads(entry["response_sse_parsed"])
        assert parsed["choices"][0]["message"]["content"] == "Hello"

    def test_closes_original_response(self, clean_httpx_state, monkeypatch):
        closed = {"n": 0}
        real_close = httpx.Response.close

        def spy_close(self):
            closed["n"] += 1
            return real_close(self)

        monkeypatch.setattr(httpx.Response, "close", spy_close)

        def handler(request):
            return httpx.Response(200, json={"ok": True})

        hooks._install_httpx_interceptor()
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.post("http://test", json={"a": 1})

        assert closed["n"] >= 1

    def test_install_is_idempotent(self, clean_httpx_state):
        hooks._install_httpx_interceptor()
        first = httpx.Client.send
        hooks._install_httpx_interceptor()
        assert httpx.Client.send is first

    def test_async_send_captures_request_and_response(self, clean_httpx_state):
        def handler(request):
            return httpx.Response(200, json={"choices": [{"message": {"content": "yo"}}]})

        hooks._install_httpx_interceptor()

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await client.post(
                    "http://test", json={"messages": [{"role": "user", "content": "q"}]}
                )

        resp = asyncio.run(run())
        assert resp.json()["choices"][0]["message"]["content"] == "yo"

        # asyncio.run drives the loop on the current thread.
        entry = hooks._take_captured_bodies(threading.current_thread().name)
        assert json.loads(entry["request"])["messages"][0]["content"] == "q"
        assert json.loads(entry["response"])["choices"][0]["message"]["content"] == "yo"


class TestInterceptorGating:
    """The interceptor must only patch httpx when a capture flag is enabled."""

    def test_not_installed_when_both_flags_off(self, mock_tracer):
        with patch("hermes_otel.hooks._install_httpx_interceptor") as spy:
            on_pre_api_request(**_pre_kwargs())
        spy.assert_not_called()

    def test_installed_when_capture_full_prompts_on(self, mock_tracer):
        from hermes_otel.plugin_config import HermesOtelConfig

        mock_tracer.config = HermesOtelConfig(capture_full_prompts=True)
        with patch("hermes_otel.hooks._install_httpx_interceptor") as spy:
            on_pre_api_request(**_pre_kwargs())
        spy.assert_called_once()

    def test_installed_when_capture_full_responses_on(self, mock_tracer):
        from hermes_otel.plugin_config import HermesOtelConfig

        mock_tracer.config = HermesOtelConfig(capture_full_responses=True)
        with patch("hermes_otel.hooks._install_httpx_interceptor") as spy:
            on_pre_api_request(**_pre_kwargs())
        spy.assert_called_once()


class TestPostApiHttpxCapture:
    def test_uses_sse_parsed_response_for_output(self, mock_tracer, clean_httpx_state):
        from hermes_otel.plugin_config import HermesOtelConfig

        mock_tracer.config = HermesOtelConfig(capture_full_responses=True)
        sse_parsed = json.dumps({"choices": [{"message": {"content": "Hello world"}}]})
        hooks._httpx_bodies[threading.current_thread().name] = {
            "response": "data: ...raw...",
            "response_sse_parsed": sse_parsed,
        }
        on_post_api_request(**_post_kwargs())
        attrs = mock_tracer.end_span.call_args[1]["attributes"]
        assert attrs["llm.output.content"] == "Hello world"
        assert attrs["output.value"] == "Hello world"

    def test_sets_input_messages_from_captured_request(self, mock_tracer, clean_httpx_state):
        from hermes_otel.plugin_config import HermesOtelConfig

        mock_tracer.config = HermesOtelConfig(capture_full_prompts=True)
        hooks._httpx_bodies[threading.current_thread().name] = {
            "request": json.dumps(
                {"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}]}
            )
        }
        on_post_api_request(**_post_kwargs())
        attrs = mock_tracer.end_span.call_args[1]["attributes"]
        assert json.loads(attrs["llm.input_messages"])[0]["content"] == "hi"
        assert json.loads(attrs["llm.request.tools"])[0]["type"] == "function"

    def test_consumes_only_current_thread_and_preserves_others(
        self, mock_tracer, clean_httpx_state
    ):
        from hermes_otel.plugin_config import HermesOtelConfig

        mock_tracer.config = HermesOtelConfig(capture_full_prompts=True)
        hooks._httpx_bodies["other-thread"] = {"request": "{}", "response": "{}"}
        hooks._httpx_bodies[threading.current_thread().name] = {
            "request": json.dumps({"messages": [{"role": "user", "content": "mine"}]})
        }
        on_post_api_request(**_post_kwargs())

        # Another in-flight thread's capture must survive this hook.
        assert "other-thread" in hooks._httpx_bodies
        assert threading.current_thread().name not in hooks._httpx_bodies
