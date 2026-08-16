"""Provider-neutral AI completions and the llama.cpp implementation.

`llama-server` serves ONE model, loaded at launch. There is no model catalogue
to browse, nothing to load on demand, and no idle-unload TTL -- the lifecycle
belongs to whoever started the process. So this provider adopts whatever the
server already has rather than managing anything, and `AIModelInfo` is filled
from `/props` plus the filename.

Structured output is ENFORCED here, not requested. The JSON Schema rides in
`response_format`, llama.cpp compiles it to a GBNF grammar, and the model
physically cannot emit anything else. That replaces pasting the schema into the
system prompt and hoping, which is what the LM Studio path did.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import httpx

from sherlock_project.ai_config import AISettings

CONNECT_TIMEOUT_SECONDS = 10.0
READ_TIMEOUT_SECONDS = 600.0

# Quantization is not in any llama-server response, but it is in the filename
# every GGUF ships under. Worth recovering: the setup screen shows it, and
# "which quant am I actually running" is the first question when quality drops.
_QUANT_PATTERN = re.compile(
    r"(?:^|[-_.])((?:IQ|Q)\d+(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4)(?:[-_.]|$)",
    re.IGNORECASE,
)


class AIProviderError(RuntimeError):
    """Base class for safe, provider-facing AI errors."""


class AIProviderUnavailableError(AIProviderError):
    """The provider cannot currently accept requests."""


class AIProviderAuthenticationError(AIProviderError):
    """The configured credentials were rejected."""


class AIProviderProtocolError(AIProviderError):
    """The provider returned an unsupported response shape."""


class AIModelNotFoundError(AIProviderError):
    """The server is reachable but has no model loaded."""


class AIModelCompatibilityError(AIProviderError):
    """The loaded model cannot run the pipeline at all.

    Not raised for models that merely refuse to stop thinking: those are
    degraded but usable. Kept as the error for a genuine incompatibility.
    """


ProviderWideError = (
    AIProviderUnavailableError,
    AIProviderAuthenticationError,
)


@dataclass(frozen=True, slots=True)
class AIModelInfo:
    key: str
    display_name: str
    quantization: str | None
    params: str | None
    loaded: bool
    max_context_length: int | None
    reasoning_options: tuple[str, ...]

    @property
    def supports_reasoning_off(self) -> bool:
        return not self.reasoning_options or "off" in self.reasoning_options

    @property
    def requires_native_reasoning(self) -> bool:
        """True when the model thinks natively and cannot be told not to.

        llama-server exposes no capability block to read this from, so nothing
        populates `reasoning_options` yet and this stays False. The honest
        answer needs a live probe -- send `enable_thinking: false` once and see
        whether `reasoning_content` comes back empty -- which is deferred.
        Until then an always-thinking model runs the canonical Pass 1 pair,
        which is the documented degraded path, not a broken one.
        """
        return bool(self.reasoning_options) and "off" not in self.reasoning_options


@dataclass(frozen=True, slots=True)
class AIGenerationStats:
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    tokens_per_second: float | None = None
    time_to_first_token_seconds: float | None = None
    model_load_time_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class AICompletion:
    final_text: str
    native_reasoning: str
    stats: AIGenerationStats
    elapsed_seconds: float


class AIProvider(Protocol):
    settings: AISettings

    async def list_models(self) -> list[AIModelInfo]: ...

    async def ensure_model_loaded(self) -> AIModelInfo: ...

    async def generate(
        self,
        *,
        system_prompt: str,
        payload: dict[str, object],
        max_tokens: int,
        reasoning_off: bool = True,
        json_schema: dict[str, Any] | None = None,
    ) -> AICompletion: ...

    async def close(self) -> None: ...


def _quantization_from_name(name: str) -> str | None:
    stem = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].removesuffix(".gguf")
    match = _QUANT_PATTERN.search(stem)
    return match.group(1).upper() if match else None


class LlamaCppProvider:
    """llama.cpp `llama-server`, over its OpenAI-compatible route."""

    def __init__(
        self,
        settings: AISettings,
        *,
        client: httpx.AsyncClient | None = None,
        api_token: str | None = None,
    ) -> None:
        self.settings = settings
        token = api_token if api_token is not None else os.getenv("LLAMA_API_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT_SECONDS,
            read=READ_TIMEOUT_SECONDS,
            write=30.0,
            pool=CONNECT_TIMEOUT_SECONDS,
        )
        if client is None:
            self._client = httpx.AsyncClient(
                base_url=settings.base_url,
                headers=headers,
                timeout=timeout,
            )
        else:
            self._client = client
            if token:
                self._client.headers["Authorization"] = f"Bearer {token}"
        self._owns_client = client is None
        self._model_info: AIModelInfo | None = None
        self._closed = False

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, object] | None = None,
    ) -> httpx.Response:
        try:
            response = await self._client.request(method, path, json=json_body)
        except httpx.TimeoutException as error:
            raise AIProviderUnavailableError(
                f"llama-server timed out while calling {path}."
            ) from error
        except httpx.RequestError as error:
            raise AIProviderUnavailableError(
                f"Unable to reach llama-server at {self.settings.base_url}."
            ) from error

        if response.status_code in {401, 403}:
            raise AIProviderAuthenticationError(
                "llama-server rejected the configured API token."
            )
        # 503 is llama-server's "still loading the model" answer, not a crash.
        # It is the normal state for the first few seconds after launch and for
        # the whole of a cold load off a slow disk, so it must read as "try
        # again", never as a failure worth aborting a scan over.
        if response.status_code in {408, 429, 500, 502, 503, 504}:
            raise AIProviderUnavailableError(
                f"llama-server is temporarily unavailable (HTTP {response.status_code})."
            )
        if response.is_error:
            raise AIProviderError(
                f"llama-server returned HTTP {response.status_code} for {path}."
            )
        return response

    @staticmethod
    def _response_json(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as error:
            raise AIProviderProtocolError(
                "llama-server returned a non-JSON response."
            ) from error
        if not isinstance(payload, dict):
            raise AIProviderProtocolError(
                "llama-server returned an invalid top-level response."
            )
        return payload

    async def list_models(self) -> list[AIModelInfo]:
        """Whatever this server has loaded -- one model, or none.

        Kept plural to satisfy the Protocol and to keep the setup screen's
        table code unchanged. It is not a catalogue: llama-server cannot load
        anything it was not launched with.
        """
        model = await self._describe_loaded_model()
        return [model] if model is not None else []

    async def _describe_loaded_model(self) -> AIModelInfo | None:
        props = self._response_json(await self._request("GET", "/props"))
        raw_path = props.get("model_path")
        if not isinstance(raw_path, str) or not raw_path:
            return None

        name = raw_path.replace("\\", "/").rsplit("/", 1)[-1]
        generation = props.get("default_generation_settings")
        context_length = (
            generation.get("n_ctx") if isinstance(generation, dict) else None
        )
        return AIModelInfo(
            key=name,
            display_name=name.removesuffix(".gguf"),
            quantization=_quantization_from_name(name),
            params=None,
            loaded=True,
            max_context_length=(
                context_length
                if isinstance(context_length, int)
                and not isinstance(context_length, bool)
                else None
            ),
            # llama-server publishes no reasoning capability block. Empty means
            # "assume it can be turned off", which is the degraded-but-working
            # default; see AIModelInfo.requires_native_reasoning.
            reasoning_options=(),
        )

    async def ensure_model_loaded(self) -> AIModelInfo:
        """Adopt the server's model. Nothing is loaded or unloaded here.

        The configured `model` is advisory: it is recorded against extractions
        so results say which model produced them, but it cannot select
        anything. A mismatch is not an error -- the user may have relaunched
        llama-server with a different GGUF, and refusing to run would be
        obstruction, not safety.
        """
        model = await self._describe_loaded_model()
        if model is None:
            raise AIModelNotFoundError(
                f"llama-server at {self.settings.base_url} has no model loaded."
            )
        self._model_info = model
        return model

    @staticmethod
    def _number(
        payload: dict[str, Any],
        key: str,
        expected: type[int | float],
    ) -> int | float | None:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return expected(value)

    def _build_stats(self, body: dict[str, Any]) -> AIGenerationStats:
        raw_usage = body.get("usage")
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        raw_timings = body.get("timings")
        timings = raw_timings if isinstance(raw_timings, dict) else {}

        prompt_ms = self._number(timings, "prompt_ms", float)
        return AIGenerationStats(
            input_tokens=self._number(usage, "prompt_tokens", int),  # type: ignore[arg-type]
            output_tokens=self._number(usage, "completion_tokens", int),  # type: ignore[arg-type]
            # llama-server does not break reasoning out of the token count the
            # way LM Studio did. Left None rather than guessed -- a fabricated
            # number here would land in the -v trace looking measured.
            reasoning_tokens=None,
            tokens_per_second=self._number(timings, "predicted_per_second", float),  # type: ignore[arg-type]
            time_to_first_token_seconds=(
                prompt_ms / 1000.0 if prompt_ms is not None else None
            ),
            model_load_time_seconds=None,
        )

    async def generate(
        self,
        *,
        system_prompt: str,
        payload: dict[str, object],
        max_tokens: int,
        reasoning_off: bool = True,
        json_schema: dict[str, Any] | None = None,
    ) -> AICompletion:
        model = self._model_info or await self.ensure_model_loaded()
        request_body: dict[str, object] = {
            "model": model.key,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "stream": False,
            "temperature": self.settings.temperature,
            "max_tokens": max_tokens,
        }
        if json_schema is not None:
            request_body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "sherlock_response",
                    "schema": json_schema,
                    "strict": True,
                },
            }
        if reasoning_off:
            # The ONLY per-request reasoning switch llama-server honours.
            # Measured 2026-08-16 against b9837 on Qwen3-8B: `reasoning: "off"`
            # and `reasoning_budget: 0` are accepted with HTTP 200 and ignored,
            # because those are launch flags whose request-body lookalikes are
            # silently discarded. Do NOT reach for `reasoning_format: "none"`
            # either -- it suppresses EXTRACTION, not thinking, so without a
            # schema attached it returns raw `<think>` inline in content and
            # breaks the JSON parse.
            #
            # This is a chat-template feature, so a model whose template has no
            # such switch ignores it and keeps thinking, with nothing in the
            # response saying so. Harmless here: thinking still arrives in
            # `reasoning_content`, away from the JSON.
            request_body["chat_template_kwargs"] = {"enable_thinking": False}

        started_at = perf_counter()
        response = await self._request(
            "POST",
            "/v1/chat/completions",
            json_body=request_body,
        )
        elapsed = perf_counter() - started_at
        body = self._response_json(response)

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AIProviderProtocolError(
                "llama-server chat response is missing the choices array."
            )
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        if not isinstance(message, dict):
            raise AIProviderProtocolError(
                "llama-server chat response is missing the message object."
            )

        content = message.get("content")
        reasoning = message.get("reasoning_content")
        return AICompletion(
            final_text=content.strip() if isinstance(content, str) else "",
            native_reasoning=reasoning.strip() if isinstance(reasoning, str) else "",
            stats=self._build_stats(body),
            elapsed_seconds=elapsed,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()
