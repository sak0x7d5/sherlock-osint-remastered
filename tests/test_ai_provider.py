import json

import httpx
import pytest

from sherlock_project.ai_config import AISettings
from sherlock_project.ai_provider import (
    AIModelNotFoundError,
    AIProviderAuthenticationError,
    AIProviderProtocolError,
    AIProviderUnavailableError,
    LlamaCppProvider,
)

pytestmark = pytest.mark.asyncio

MODEL_PATH = "D:/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"


def _settings(model: str = "Qwen3-8B-Q4_K_M.gguf") -> AISettings:
    return AISettings(
        base_url="http://llamacpp.test",
        model=model,
        temperature=0.1,
        context_length=8192,
    )


def _props(model_path: str = MODEL_PATH, n_ctx: int = 8192) -> dict:
    return {
        "model_path": model_path,
        "default_generation_settings": {"n_ctx": n_ctx},
    }


def _chat_response(
    *,
    content: str = '{"reasoning":"brief","extraction":{}}',
    reasoning_content: str | None = None,
) -> dict:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        "timings": {"prompt_ms": 400.0, "predicted_per_second": 12.5},
    }


def _provider(handler, *, settings: AISettings | None = None) -> LlamaCppProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://llamacpp.test")
    return LlamaCppProvider(settings or _settings(), client=client)


def _router_props() -> dict:
    """Router mode reports no model of its own; the catalogue is /v1/models."""
    return {
        "role": "router",
        "model_path": "none",
        "default_generation_settings": {"n_ctx": 0},
    }


def _router_entry(name: str, *, loaded: bool = False, quant: str = "Q4_K_M") -> dict:
    return {
        "id": f"{name}-GGUF",
        "object": "model",
        "status": {
            "value": "loaded" if loaded else "unloaded",
            "args": [
                "llama-server.exe",
                "--alias",
                f"{name}-GGUF",
                "--model",
                f"D:/models/{name}-GGUF/{name}-{quant}.gguf",
            ],
        },
    }


def _router_handler(entries: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json=_router_props())
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": entries})
        return httpx.Response(200, json=_chat_response())

    return handler


async def test_single_mode_reports_the_one_loaded_model():
    provider = _provider(lambda _request: httpx.Response(200, json=_props()))
    models = await provider.list_models()

    assert len(models) == 1
    assert models[0].key == "Qwen3-8B-Q4_K_M.gguf"
    assert models[0].display_name == "Qwen3-8B-Q4_K_M"
    assert models[0].quantization == "Q4_K_M"
    assert models[0].max_context_length == 8192
    assert models[0].loaded is True


async def test_router_mode_lists_the_whole_catalogue():
    """`--models-dir` makes this a real picker, not a one-row table.

    Every GGUF under the directory is offered whether loaded or not, because
    naming one in a chat request loads it on demand.
    """
    provider = _provider(
        _router_handler([
            _router_entry("Qwen3-8B", loaded=True),
            _router_entry("gemma-4-E2B-it", quant="Q8_0"),
        ]),
        settings=_settings(model="Qwen3-8B-GGUF"),
    )
    models = await provider.list_models()

    assert [model.key for model in models] == ["Qwen3-8B-GGUF", "gemma-4-E2B-it-GGUF"]
    assert models[0].display_name == "Qwen3-8B"
    assert models[0].loaded is True
    assert models[1].loaded is False
    # The quantization lives only in the launch argv's .gguf path -- the id is
    # the repo directory name and usually does not carry it.
    assert models[1].quantization == "Q8_0"


async def test_router_mode_selects_the_configured_model():
    provider = _provider(
        _router_handler([
            _router_entry("Qwen3-8B"),
            _router_entry("gemma-4-E2B-it"),
        ]),
        settings=_settings(model="gemma-4-E2B-it-GGUF"),
    )
    model = await provider.ensure_model_loaded()

    assert model.key == "gemma-4-E2B-it-GGUF"


async def test_router_mode_reports_a_model_it_does_not_serve():
    """With a real catalogue, a name that is not there is a real mistake.

    Quietly running something else would put the wrong attribution on every
    extraction produced -- `results.ai_extraction_model` would be a lie.
    """
    provider = _provider(
        _router_handler([_router_entry("Qwen3-8B"), _router_entry("gemma-4-E2B-it")]),
        settings=_settings(model="absent/model"),
    )

    with pytest.raises(AIModelNotFoundError, match="Available:"):
        await provider.ensure_model_loaded()


async def test_single_mode_adopts_a_mismatch_rather_than_refusing():
    """One model on offer means the launch flag decided, not the config.

    Relaunching llama-server with a different GGUF is a decision, so the stored
    name is stale bookkeeping rather than an instruction to obey.
    """
    provider = _provider(
        lambda _request: httpx.Response(200, json=_props()),
        settings=_settings(model="something-else.gguf"),
    )
    model = await provider.ensure_model_loaded()

    assert model.key == "Qwen3-8B-Q4_K_M.gguf"


async def test_nothing_served_at_all_is_reported():
    provider = _provider(lambda _request: httpx.Response(200, json={"model_path": ""}))

    with pytest.raises(AIModelNotFoundError, match="serving no models"):
        await provider.ensure_model_loaded()


async def test_router_with_an_empty_directory_is_reported():
    """The `--models-dir` layout trap: one level too deep finds zero models."""
    provider = _provider(_router_handler([]))

    with pytest.raises(AIModelNotFoundError, match="serving no models"):
        await provider.ensure_model_loaded()


async def test_reasoning_probe_detects_a_model_that_cannot_stop_thinking():
    """The only signal there is. llama-server publishes no capability block.

    Measured against b9837: an always-thinking model accepts
    `enable_thinking: false` with HTTP 200 and thinks anyway, with nothing in
    the response admitting it. Behaviour is the only thing that tells them
    apart, and getting it wrong sends an always-thinking model down the
    canonical Pass 1 path where its thinking eats the answer's token budget.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json=_props())
        return httpx.Response(
            200,
            json=_chat_response(reasoning_content="thought anyway"),
        )

    model = await _provider(handler).ensure_model_loaded()

    assert model.reasoning_options == ("on",)
    assert model.supports_reasoning_off is False
    assert model.requires_native_reasoning is True


async def test_reasoning_probe_clears_a_model_that_honours_the_switch():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json=_props())
        return httpx.Response(200, json=_chat_response())

    model = await _provider(handler).ensure_model_loaded()

    assert model.reasoning_options == ("off", "on")
    assert model.requires_native_reasoning is False


async def test_a_failed_probe_leaves_the_model_usable():
    """An unknown capability must not cost the run.

    Falling back to the canonical pair is the documented degraded path; raising
    here would turn "could not measure" into "cannot scan".
    """
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json=_props())
        return httpx.Response(503)

    model = await _provider(handler).ensure_model_loaded()

    assert model.key == "Qwen3-8B-Q4_K_M.gguf"
    assert model.reasoning_options == ()
    assert model.requires_native_reasoning is False


async def test_schema_is_enforced_through_response_format():
    """The schema goes on the wire, not into the prompt.

    This is the whole point of the migration: llama.cpp compiles it to a
    grammar, so a reply that does not match is unrepresentable rather than
    merely discouraged.
    """
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json=_props())
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_chat_response())

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    provider = _provider(handler)
    await provider.generate(
        system_prompt="system",
        payload={"input": "evidence"},
        max_tokens=1024,
        json_schema=schema,
    )

    # bodies[0] would be the reasoning probe that ensure_model_loaded fires
    # lazily on the first generate. The call under test is the last one.
    response_format = bodies[-1]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == schema
    assert response_format["json_schema"]["strict"] is True


async def test_no_schema_means_no_response_format():
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json=_props())
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_chat_response())

    provider = _provider(handler)
    await provider.generate(system_prompt="system", payload={}, max_tokens=10)

    assert "response_format" not in bodies[0]


async def test_reasoning_off_uses_the_only_switch_llama_server_honours():
    """`enable_thinking` and nothing else.

    Measured against llama-server b9837: `reasoning: "off"` and
    `reasoning_budget: 0` are accepted with HTTP 200 and ignored, because they
    are launch flags. `reasoning_format: "none"` is worse than useless -- it
    stops thinking being EXTRACTED, so without a schema it lands inline in
    content and breaks the JSON parse. Asserted negatively too, so a future
    edit reintroducing one of them fails here rather than in a scan.
    """
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json=_props())
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_chat_response())

    provider = _provider(handler)
    await provider.generate(
        system_prompt="system",
        payload={},
        max_tokens=10,
        reasoning_off=True,
    )

    body = bodies[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning" not in body
    assert "reasoning_budget" not in body
    assert "reasoning_format" not in body


async def test_reasoning_on_leaves_thinking_alone():
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json=_props())
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_chat_response())

    provider = _provider(handler)
    await provider.generate(
        system_prompt="system",
        payload={},
        max_tokens=10,
        reasoning_off=False,
    )

    # Last, not first: the reasoning probe runs ahead of it and always sets
    # enable_thinking, which is the whole point of the probe.
    assert "chat_template_kwargs" not in bodies[-1]


async def test_native_reasoning_is_split_away_from_the_answer():
    """Thinking must never reach the text that gets JSON-parsed."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json=_props())
        return httpx.Response(
            200,
            json=_chat_response(
                content='{"identity_status":"reject"}',
                reasoning_content="compared the evidence",
            ),
        )

    provider = _provider(handler)
    completion = await provider.generate(
        system_prompt="system",
        payload={},
        max_tokens=10,
        reasoning_off=False,
    )

    assert completion.final_text == '{"identity_status":"reject"}'
    assert completion.native_reasoning == "compared the evidence"


async def test_stats_come_from_usage_and_timings():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json=_props())
        return httpx.Response(200, json=_chat_response())

    provider = _provider(handler)
    completion = await provider.generate(
        system_prompt="system",
        payload={},
        max_tokens=10,
    )

    assert completion.stats.input_tokens == 100
    assert completion.stats.output_tokens == 20
    assert completion.stats.tokens_per_second == 12.5
    assert completion.stats.time_to_first_token_seconds == 0.4
    # llama-server does not separate reasoning tokens. None, never a guess --
    # an invented number would read as measured in the -v trace.
    assert completion.stats.reasoning_tokens is None


async def test_api_token_is_sent_as_bearer_header():
    authorization: str | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal authorization
        authorization = request.headers.get("Authorization")
        return httpx.Response(200, json=_props())

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://llamacpp.test")
    provider = LlamaCppProvider(_settings(), client=client, api_token="secret-token")

    await provider.list_models()

    assert authorization == "Bearer secret-token"


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (401, AIProviderAuthenticationError),
        # 503 is llama-server still loading the model, which is the normal
        # state for the first seconds after launch -- "try again", not "fail".
        (503, AIProviderUnavailableError),
    ],
)
async def test_http_failures_are_typed(status: int, error_type: type[Exception]):
    provider = _provider(lambda _request: httpx.Response(status))

    with pytest.raises(error_type):
        await provider.list_models()


async def test_connection_failure_is_typed_and_safe():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private connection details", request=request)

    provider = _provider(handler)

    with pytest.raises(AIProviderUnavailableError) as error_info:
        await provider.list_models()

    assert "private connection details" not in str(error_info.value)


async def test_malformed_chat_response_is_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json=_props())
        return httpx.Response(200, json={"wrong": []})

    provider = _provider(handler)

    with pytest.raises(AIProviderProtocolError, match="choices array"):
        await provider.generate(system_prompt="s", payload={}, max_tokens=10)


async def test_non_json_response_is_rejected():
    provider = _provider(lambda _request: httpx.Response(200, text="<html>"))

    with pytest.raises(AIProviderProtocolError, match="non-JSON"):
        await provider.list_models()
