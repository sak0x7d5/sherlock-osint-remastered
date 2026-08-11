"""Provider-neutral AI completions and the LM Studio REST implementation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import httpx

from sherlock_project.ai_config import AISettings

CONNECT_TIMEOUT_SECONDS = 10.0
READ_TIMEOUT_SECONDS = 600.0


class AIProviderError(RuntimeError):
    """Base class for safe, provider-facing AI errors."""


class AIProviderUnavailableError(AIProviderError):
    """The provider cannot currently accept requests."""


class AIProviderAuthenticationError(AIProviderError):
    """The configured credentials were rejected."""


class AIProviderProtocolError(AIProviderError):
    """The provider returned an unsupported response shape."""


class AIModelNotFoundError(AIProviderError):
    """The configured model is not available from the provider."""


class AIModelCompatibilityError(AIProviderError):
    """The configured model cannot run the pipeline at all.

    No longer raised for models that merely refuse to disable native thinking:
    those are degraded but usable, and are warned about at selection instead.
    Kept as the error for a genuine incompatibility.
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

        Distinct from `not supports_reasoning_off` only in intent: this one is
        asked at request time to decide how much output budget to allow, not at
        setup time to decide whether the model is usable at all.
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
    ) -> AICompletion: ...

    async def close(self) -> None: ...


class LMStudioProvider:
    """LM Studio native REST v1 provider."""

    def __init__(
        self,
        settings: AISettings,
        *,
        client: httpx.AsyncClient | None = None,
        api_token: str | None = None,
    ) -> None:
        self.settings = settings
        token = api_token if api_token is not None else os.getenv("LM_API_TOKEN")
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
            response = await self._client.request(
                method,
                path,
                json=json_body,
            )
        except httpx.TimeoutException as error:
            raise AIProviderUnavailableError(
                f"LM Studio timed out while calling {path}."
            ) from error
        except httpx.RequestError as error:
            raise AIProviderUnavailableError(
                f"Unable to reach LM Studio at {self.settings.base_url}."
            ) from error

        if response.status_code in {401, 403}:
            raise AIProviderAuthenticationError(
                "LM Studio rejected the configured API token."
            )
        if response.status_code in {408, 429, 500, 502, 503, 504}:
            raise AIProviderUnavailableError(
                f"LM Studio is temporarily unavailable (HTTP {response.status_code})."
            )
        if response.is_error:
            raise AIProviderError(
                f"LM Studio returned HTTP {response.status_code} for {path}."
            )
        return response

    @staticmethod
    def _response_json(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as error:
            raise AIProviderProtocolError(
                "LM Studio returned a non-JSON response."
            ) from error
        if not isinstance(payload, dict):
            raise AIProviderProtocolError(
                "LM Studio returned an invalid top-level response."
            )
        return payload

    @staticmethod
    def _parse_model(raw: object) -> AIModelInfo | None:
        if not isinstance(raw, dict) or raw.get("type") != "llm":
            return None
        key = raw.get("key")
        if not isinstance(key, str) or not key:
            return None
        display_name = raw.get("display_name")
        quantization = raw.get("quantization")
        capabilities = raw.get("capabilities")
        reasoning: object = None
        if isinstance(capabilities, dict):
            reasoning = capabilities.get("reasoning")
        reasoning_options: tuple[str, ...] = ()
        if isinstance(reasoning, dict):
            options = reasoning.get("allowed_options")
            if isinstance(options, list):
                reasoning_options = tuple(
                    option for option in options if isinstance(option, str)
                )
        quantization_name = (
            quantization.get("name")
            if isinstance(quantization, dict)
            and isinstance(quantization.get("name"), str)
            else None
        )
        loaded_instances = raw.get("loaded_instances")
        max_context_length = raw.get("max_context_length")
        return AIModelInfo(
            key=key,
            display_name=(
                display_name if isinstance(display_name, str) else key
            ),
            quantization=quantization_name,
            params=(
                raw.get("params_string")
                if isinstance(raw.get("params_string"), str)
                else None
            ),
            loaded=isinstance(loaded_instances, list) and bool(loaded_instances),
            max_context_length=(
                max_context_length
                if isinstance(max_context_length, int)
                and not isinstance(max_context_length, bool)
                else None
            ),
            reasoning_options=reasoning_options,
        )

    async def list_models(self) -> list[AIModelInfo]:
        response = await self._request("GET", "/api/v1/models")
        payload = self._response_json(response)
        models = payload.get("models")
        if not isinstance(models, list):
            raise AIProviderProtocolError(
                "LM Studio model listing is missing the models array."
            )
        parsed = [self._parse_model(model) for model in models]
        return [model for model in parsed if model is not None]

    async def ensure_model_loaded(self) -> AIModelInfo:
        models = await self.list_models()
        model = next(
            (item for item in models if item.key == self.settings.model),
            None,
        )
        if model is None:
            raise AIModelNotFoundError(
                f"Configured LM Studio model {self.settings.model!r} is not downloaded."
            )
        # A model that cannot disable native reasoning used to be refused here.
        # It is degraded, not incompatible: `generate` declares the reasoning it
        # will actually get, LM Studio returns thinking as its own output item,
        # and the structured reply still parses. Refusing at load time meant a
        # model setup had accepted could never be used, so the check belongs
        # where the user is warned -- at selection -- not here.
        if not model.loaded:
            await self._request(
                "POST",
                "/api/v1/models/load",
                json_body={
                    "model": model.key,
                    "context_length": self.settings.context_length,
                    "echo_load_config": True,
                },
            )
            model = AIModelInfo(
                key=model.key,
                display_name=model.display_name,
                quantization=model.quantization,
                params=model.params,
                loaded=True,
                max_context_length=model.max_context_length,
                reasoning_options=model.reasoning_options,
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

    async def generate(
        self,
        *,
        system_prompt: str,
        payload: dict[str, object],
        max_tokens: int,
        reasoning_off: bool = True,
    ) -> AICompletion:
        model = self._model_info or await self.ensure_model_loaded()
        request_body: dict[str, object] = {
            "model": model.key,
            "system_prompt": system_prompt,
            "input": json.dumps(payload, ensure_ascii=False),
            "stream": False,
            "store": False,
            "temperature": self.settings.temperature,
            "max_output_tokens": max_tokens,
            "context_length": self.settings.context_length,
        }
        if model.reasoning_options:
            requested_reasoning = "off" if reasoning_off else "on"
            if requested_reasoning not in model.reasoning_options:
                # The model cannot honour what we asked for. Declaring the
                # reasoning we are actually going to get beats omitting the key
                # and leaving it to the model's default: LM Studio then emits
                # thinking as a separate `reasoning` output item, which this
                # parser routes away from `final_text`, so the structured reply
                # stays clean JSON instead of risking inline think markers.
                requested_reasoning = "on" if "on" in model.reasoning_options else ""
            if requested_reasoning:
                request_body["reasoning"] = requested_reasoning

        started_at = perf_counter()
        response = await self._request(
            "POST",
            "/api/v1/chat",
            json_body=request_body,
        )
        elapsed = perf_counter() - started_at
        body = self._response_json(response)
        output = body.get("output")
        if not isinstance(output, list):
            raise AIProviderProtocolError(
                "LM Studio chat response is missing the output array."
            )

        final_parts: list[str] = []
        reasoning_parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, str):
                continue
            if item.get("type") == "reasoning":
                reasoning_parts.append(content)
            elif item.get("type") == "message":
                final_parts.append(content)

        raw_stats = body.get("stats")
        stats_payload = raw_stats if isinstance(raw_stats, dict) else {}
        stats = AIGenerationStats(
            input_tokens=self._number(stats_payload, "input_tokens", int),  # type: ignore[arg-type]
            output_tokens=self._number(stats_payload, "total_output_tokens", int),  # type: ignore[arg-type]
            reasoning_tokens=self._number(stats_payload, "reasoning_output_tokens", int),  # type: ignore[arg-type]
            tokens_per_second=self._number(stats_payload, "tokens_per_second", float),  # type: ignore[arg-type]
            time_to_first_token_seconds=self._number(
                stats_payload,
                "time_to_first_token_seconds",
                float,
            ),  # type: ignore[arg-type]
            model_load_time_seconds=self._number(
                stats_payload,
                "model_load_time_seconds",
                float,
            ),  # type: ignore[arg-type]
        )
        return AICompletion(
            final_text="".join(final_parts).strip(),
            native_reasoning="".join(reasoning_parts).strip(),
            stats=stats,
            elapsed_seconds=elapsed,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()
