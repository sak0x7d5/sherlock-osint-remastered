import json

import httpx
import pytest

from sherlock_project.ai_config import AISettings
from sherlock_project.ai_provider import (
    AIProviderAuthenticationError,
    AIProviderProtocolError,
    AIProviderUnavailableError,
    LMStudioProvider,
)

pytestmark = pytest.mark.asyncio


def _settings(model: str = "example/model") -> AISettings:
    return AISettings(
        base_url="http://lmstudio.test",
        model=model,
        temperature=0.1,
        context_length=8192,
    )


def _model(
    *,
    key: str = "example/model",
    reasoning: list[str] | None = None,
    loaded: bool = False,
) -> dict:
    capabilities = {}
    if reasoning is not None:
        capabilities["reasoning"] = {
            "allowed_options": reasoning,
            "default": reasoning[-1],
        }
    return {
        "type": "llm",
        "key": key,
        "display_name": "Example Model",
        "quantization": {"name": "Q4_K_M"},
        "params_string": "8B",
        "loaded_instances": ([{"id": key}] if loaded else []),
        "max_context_length": 32768,
        "capabilities": capabilities,
    }


def _provider(handler, *, settings: AISettings | None = None) -> LMStudioProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(
        transport=transport,
        base_url="http://lmstudio.test",
    )
    return LMStudioProvider(settings or _settings(), client=client)


async def test_list_models_filters_non_llms_and_parses_reasoning_capability():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    _model(reasoning=["off", "on"], loaded=True),
                    {"type": "embedding", "key": "embed/model"},
                ]
            },
        )

    provider = _provider(handler)
    models = await provider.list_models()

    assert len(models) == 1
    assert models[0].key == "example/model"
    assert models[0].reasoning_options == ("off", "on")
    assert models[0].supports_reasoning_off is True
    assert models[0].loaded is True


async def test_ensure_model_loaded_posts_expected_configuration():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json={"models": [_model(reasoning=["off", "on"])]})
        return httpx.Response(
            200,
            json={"type": "llm", "instance_id": "example/model", "status": "loaded"},
        )

    provider = _provider(handler)
    model = await provider.ensure_model_loaded()

    assert model.loaded is True
    assert [request.url.path for request in requests] == [
        "/api/v1/models",
        "/api/v1/models/load",
    ]
    load_body = json.loads(requests[1].content)
    assert load_body == {
        "model": "example/model",
        "context_length": 8192,
        "echo_load_config": True,
    }


async def test_loaded_model_is_reused_without_load_request():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={"models": [_model(loaded=True)]})

    provider = _provider(handler)
    await provider.ensure_model_loaded()

    assert paths == ["/api/v1/models"]


async def test_generate_disables_native_reasoning_and_parses_stats():
    request_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return httpx.Response(
                200,
                json={"models": [_model(reasoning=["off", "on"], loaded=True)]},
            )
        request_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "output": [
                    {"type": "reasoning", "content": "unexpected native thought"},
                    {"type": "message", "content": '{"reasoning":"brief","extraction":{}}'},
                ],
                "stats": {
                    "input_tokens": 100,
                    "total_output_tokens": 20,
                    "reasoning_output_tokens": 3,
                    "tokens_per_second": 12.5,
                    "time_to_first_token_seconds": 0.4,
                },
            },
        )

    provider = _provider(handler)
    await provider.ensure_model_loaded()
    completion = await provider.generate(
        system_prompt="system",
        payload={"input": "evidence"},
        max_tokens=1024,
    )

    request = request_bodies[0]
    assert request["reasoning"] == "off"
    assert request["temperature"] == 0.1
    assert request["stream"] is False
    assert request["store"] is False
    assert request["max_output_tokens"] == 1024
    assert completion.native_reasoning == "unexpected native thought"
    assert completion.stats.reasoning_tokens == 3
    assert completion.stats.tokens_per_second == 12.5


async def test_generate_explicitly_enables_supported_native_reasoning():
    request_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return httpx.Response(
                200,
                json={"models": [_model(reasoning=["off", "on"], loaded=True)]},
            )
        request_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "output": [
                    {"type": "reasoning", "content": "compared the evidence"},
                    {"type": "message", "content": '{"identity_status":"reject"}'},
                ],
                "stats": {"reasoning_output_tokens": 4},
            },
        )

    provider = _provider(handler)
    await provider.ensure_model_loaded()
    completion = await provider.generate(
        system_prompt="system",
        payload={"input": "evidence"},
        max_tokens=1024,
        reasoning_off=False,
    )

    assert request_bodies[0]["reasoning"] == "on"
    assert completion.native_reasoning == "compared the evidence"
    assert completion.final_text == '{"identity_status":"reject"}'
    assert completion.stats.reasoning_tokens == 4


async def test_api_token_is_sent_as_bearer_header():
    authorization: str | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal authorization
        authorization = request.headers.get("Authorization")
        return httpx.Response(200, json={"models": []})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(
        transport=transport,
        base_url="http://lmstudio.test",
    )
    provider = LMStudioProvider(
        _settings(),
        client=client,
        api_token="secret-token",
    )

    await provider.list_models()

    assert authorization == "Bearer secret-token"


async def test_non_reasoning_model_omits_reasoning_parameter():
    chat_body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_body
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json={"models": [_model(loaded=True)]})
        chat_body = json.loads(request.content)
        return httpx.Response(200, json={"output": [], "stats": {}})

    provider = _provider(handler)
    await provider.ensure_model_loaded()
    await provider.generate(system_prompt="system", payload={}, max_tokens=10)

    assert "reasoning" not in chat_body


async def test_reasoning_only_model_loads_and_declares_thinking_on():
    """A model that always thinks is usable; the request says so explicitly.

    It used to be refused at load time. Omitting the key instead would leave
    the mode to the model's default; declaring it means LM Studio returns
    thinking as its own output item, which keeps `final_text` clean JSON.
    """
    chat_body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return httpx.Response(
                200,
                json={"models": [_model(reasoning=["on"])]},
            )
        if request.url.path == "/api/v1/chat":
            chat_body.update(json.loads(request.content))
            return httpx.Response(200, json={"output": [], "stats": {}})
        return httpx.Response(200, json={})

    provider = _provider(handler)

    model = await provider.ensure_model_loaded()
    await provider.generate(
        system_prompt="system",
        payload={},
        max_tokens=10,
        reasoning_off=True,
    )

    assert model.requires_native_reasoning is True
    assert chat_body["reasoning"] == "on"


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (401, AIProviderAuthenticationError),
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


async def test_malformed_model_listing_is_rejected():
    provider = _provider(
        lambda _request: httpx.Response(200, json={"wrong": []})
    )

    with pytest.raises(AIProviderProtocolError, match="models array"):
        await provider.list_models()
