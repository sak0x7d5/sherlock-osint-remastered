import json

import httpx
import pytest

from sherlock_project.ai_config import (
    DEFAULT_GEMINI_BASE_URL,
    AISettings,
    load_ai_settings,
    save_ai_settings,
)
from sherlock_project.ai_provider import (
    AIContentBlockedError,
    AIModelNotFoundError,
    AIProviderAuthenticationError,
    AIProviderUnavailableError,
    AIQuotaExhaustedError,
    AIRateLimitedError,
    LlamaCppProvider,
    OpenAICompatibleProvider,
    ProviderWideError,
    create_provider,
)
from sherlock_project.ai_rate_limit import RequestRateLimiter, backoff_delay


def _settings(**overrides) -> AISettings:
    values: dict[str, object] = {
        "provider": "gemini",
        "base_url": DEFAULT_GEMINI_BASE_URL,
        "model": "gemini-2.5-flash",
    }
    values.update(overrides)
    return AISettings(**values)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _provider(handler, *, settings: AISettings | None = None, key: str | None = "k"):
    clock = FakeClock()
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(
        transport=transport, base_url=DEFAULT_GEMINI_BASE_URL
    )
    resolved = settings or _settings()
    limiter = RequestRateLimiter(
        resolved.requests_per_minute or 10, clock=clock, sleep=clock.sleep
    )
    provider = OpenAICompatibleProvider(
        resolved,
        client=client,
        api_key=key,
        environ={},
        limiter=limiter,
        sleep=clock.sleep,
    )
    return provider, clock


def _chat(content: str = '{"identity_status":"reject"}', finish: str = "stop") -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "completion_tokens_details": {"reasoning_tokens": 12},
        },
    }


def _quota_error(quota_id: str, retry: str | None = "7s") -> list:
    details: list[dict] = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaMetric": "generate_requests", "quotaId": quota_id}],
        }
    ]
    if retry is not None:
        details.append(
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry}
        )
    # Gemini's OpenAI-compatible route wraps the error object in a list.
    return [{"error": {"code": 429, "message": "quota", "details": details}}]


async def _generate(provider, **overrides):
    arguments = {
        "system_prompt": "system",
        "payload": {"site": "x"},
        "max_tokens": 256,
        "json_schema": {"type": "object"},
    }
    arguments.update(overrides)
    return await provider.generate(**arguments)


# -- configuration ---------------------------------------------------------


def test_gemini_settings_round_trip_without_storing_a_key(tmp_path):
    path = tmp_path / "config.toml"
    save_ai_settings(_settings(requests_per_minute=8), path=path, environ={})
    text = path.read_text(encoding="utf-8")
    assert 'provider = "gemini"' in text
    assert "api_key" not in text.replace("api_key_env", "")

    loaded = load_ai_settings(path=path, environ={})
    assert loaded.is_cloud
    assert loaded.requests_per_minute == 8


def test_llama_base_url_override_never_redirects_a_cloud_provider(tmp_path):
    path = tmp_path / "config.toml"
    save_ai_settings(_settings(), path=path, environ={})
    loaded = load_ai_settings(
        path=path, environ={"LLAMA_SERVER_BASE_URL": "http://127.0.0.1:9"}
    )
    assert loaded.base_url == DEFAULT_GEMINI_BASE_URL


def test_factory_picks_the_provider_the_config_names():
    assert isinstance(create_provider(_settings()), OpenAICompatibleProvider)
    local = AISettings(base_url="http://127.0.0.1:8080", model="m.gguf")
    assert isinstance(create_provider(local), LlamaCppProvider)


# -- request shape ---------------------------------------------------------


async def test_request_carries_key_schema_and_reasoning_switch():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_chat())

    provider, _ = _provider(handler)
    completion = await _generate(provider, reasoning_off=True)

    request = seen[0]
    assert request.url.path == "/v1beta/openai/chat/completions"
    assert request.headers["authorization"] == "Bearer k"
    body = json.loads(request.content)
    assert body["model"] == "gemini-2.5-flash"
    assert body["reasoning_effort"] == "none"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is False
    assert "chat_template_kwargs" not in body
    assert completion.final_text == '{"identity_status":"reject"}'
    assert completion.stats.input_tokens == 120
    assert completion.stats.reasoning_tokens == 12


async def test_reasoning_on_asks_for_a_small_budget():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_chat())

    provider, _ = _provider(handler)
    await _generate(provider, reasoning_off=False)
    assert seen[0]["reasoning_effort"] == "low"


async def test_rejected_reasoning_effort_is_dropped_once_and_remembered():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(
                400,
                json={"error": {"message": "reasoning_effort is not supported"}},
            )
        return httpx.Response(200, json=_chat())

    provider, _ = _provider(handler)
    await _generate(provider)
    await _generate(provider)
    assert provider.reasoning_effort_supported is False
    assert ["reasoning_effort" in body for body in seen] == [True, False, False]


async def test_rejected_schema_falls_back_to_json_object():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if body.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(
                400, json={"error": {"message": "Invalid JSON schema field"}}
            )
        return httpx.Response(200, json=_chat())

    provider, _ = _provider(handler)
    await _generate(provider)
    assert provider.structured_output_mode == "json_object"
    assert seen[-1]["response_format"] == {"type": "json_object"}


async def test_an_unexplained_400_is_reported_not_retried():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": {"message": "bad temperature"}})

    provider, _ = _provider(handler)
    with pytest.raises(Exception, match="bad temperature"):
        await _generate(provider)
    assert calls == 1


# -- 429: pace versus quota ------------------------------------------------


async def test_per_minute_429_waits_the_stated_delay_then_succeeds():
    responses = [
        httpx.Response(429, json=_quota_error("GenerateRequestsPerMinutePerProject")),
        httpx.Response(200, json=_chat()),
    ]

    provider, clock = _provider(lambda request: responses.pop(0))
    completion = await _generate(provider)
    assert completion.final_text
    assert clock.slept == [7.0]


async def test_daily_quota_429_stops_at_once_and_is_provider_wide():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            429, json=_quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        )

    provider, _ = _provider(handler)
    with pytest.raises(AIQuotaExhaustedError) as caught:
        await _generate(provider)
    assert calls == 1
    assert isinstance(caught.value, ProviderWideError)


async def test_persistent_per_minute_429_is_one_request_failing_not_the_provider():
    provider, _ = _provider(
        lambda request: httpx.Response(429, headers={"retry-after": "3"}, json={})
    )
    with pytest.raises(AIRateLimitedError) as caught:
        await _generate(provider)
    assert not isinstance(caught.value, ProviderWideError)
    assert caught.value.retry_after == 3.0


async def test_overloaded_503_is_retried_with_backoff():
    responses = [httpx.Response(503, json={}), httpx.Response(200, json=_chat())]
    provider, clock = _provider(lambda request: responses.pop(0))
    await _generate(provider)
    assert len(clock.slept) == 1


async def test_persistent_503_is_unavailable():
    provider, _ = _provider(lambda request: httpx.Response(503, json={}))
    with pytest.raises(AIProviderUnavailableError):
        await _generate(provider)


@pytest.mark.parametrize("status", [401, 403])
async def test_rejected_key_is_an_authentication_error(status: int):
    provider, _ = _provider(lambda request: httpx.Response(status, json={}))
    with pytest.raises(AIProviderAuthenticationError, match="GEMINI_API_KEY"):
        await _generate(provider)


async def test_missing_key_fails_before_any_request():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be sent without a key")

    provider, _ = _provider(handler, key=None)
    with pytest.raises(AIProviderAuthenticationError, match="aistudio"):
        await _generate(provider)


async def test_content_filter_is_one_site_failing():
    provider, _ = _provider(
        lambda request: httpx.Response(200, json=_chat(content="", finish="content_filter"))
    )
    with pytest.raises(AIContentBlockedError) as caught:
        await _generate(provider)
    assert not isinstance(caught.value, ProviderWideError)


# -- models ----------------------------------------------------------------


async def test_models_are_listed_without_the_models_prefix():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta/openai/models"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "models/gemini-2.5-flash", "object": "model"},
                    {"id": "models/gemini-2.5-pro", "object": "model"},
                ]
            },
        )

    provider, _ = _provider(handler)
    models = await provider.list_models()
    assert [model.key for model in models] == ["gemini-2.5-flash", "gemini-2.5-pro"]
    loaded = await provider.ensure_model_loaded()
    assert loaded.key == "gemini-2.5-flash"
    assert not loaded.requires_native_reasoning


async def test_an_unknown_model_is_reported_before_any_site_is_sent():
    provider, _ = _provider(
        lambda request: httpx.Response(200, json={"data": [{"id": "models/other"}]})
    )
    with pytest.raises(AIModelNotFoundError, match="gemini-2.5-flash"):
        await provider.ensure_model_loaded()


# -- limiter ---------------------------------------------------------------


async def test_limiter_admits_a_full_minute_then_waits_for_the_window():
    clock = FakeClock()
    limiter = RequestRateLimiter(3, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        await limiter.acquire()
    assert clock.slept == []
    await limiter.acquire()
    assert clock.slept == [60.0]


async def test_limiter_defer_holds_every_caller_back():
    clock = FakeClock()
    limiter = RequestRateLimiter(100, clock=clock, sleep=clock.sleep)
    limiter.defer(12.0)
    await limiter.acquire()
    assert clock.slept == [12.0]


def test_backoff_grows_and_is_capped():
    assert backoff_delay(1, jitter=lambda: 1.0) == 2.0
    assert backoff_delay(3, jitter=lambda: 1.0) == 8.0
    assert backoff_delay(20, jitter=lambda: 1.0) == 60.0
    assert backoff_delay(3, jitter=lambda: 0.0) == 4.0
