"""Provider-neutral AI completions and the llama.cpp implementation.


`llama-server` runs in two shapes and this provider supports both, because the
difference is the whole user experience:

- ROUTER, `--models-dir PATH`: every GGUF under that directory is on offer,
  and naming one in a chat request loads it on demand. Models live wherever
  the user keeps them and switching costs a request rather than a restart.
  This is the mode worth running and the one setup should steer people to.
- SINGLE, `-m FILE`: one model, fixed until the process is restarted.

Neither loads or unloads anything on a timer, so there is no TTL to manage --
the lifecycle belongs to whoever started the process.

Model metadata is thin either way: llama-server publishes nothing like LM
Studio's capability block, so quantization and parameter count are recovered
from the filename and reasoning capability is not detected at all yet.

Structured output is ENFORCED here, not requested. The JSON Schema rides in
`response_format`, llama.cpp compiles it to a GBNF grammar, and the model
physically cannot emit anything else. That replaces pasting the schema into the
system prompt and hoping, which is what the LM Studio path did.

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
# `/props.role` when llama-server was started with `--models-dir`, and absent
# when it was started with `-m`. The only way to tell the two apart.
ROUTER_ROLE = "router"

# Quantization is not in any llama-server response, but it is in the filename
# every GGUF ships under. Worth recovering: the setup screen shows it, and
# "which quant am I actually running" is the first question when quality drops.
_QUANT_PATTERN = re.compile(
    r"(?:^|[-_.])((?:IQ|Q)\d+(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4)(?:[-_.]|$)",
    re.IGNORECASE,
)
# Parameter count, also filename-only: llama-server reports nothing like LM
# Studio's `params_string`, and an all-"?" Size column is worth less than a
# figure recovered from the name every GGUF already carries. Anchored on a
# separator so the "3" of "Qwen3" cannot be read as a size.
# The optional E is Gemma's "effective parameters" naming (E2B, E4B). Reporting
# 2B for an E2B is closer to useful than reporting nothing.
_PARAMS_PATTERN = re.compile(
    r"(?:^|[-_.])E?(\d+(?:\.\d+)?)\s*B(?:[-_.]|$)",
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


def _basename(name: str) -> str:
    return name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].removesuffix(".gguf")


def _quantization_from_name(name: str) -> str | None:
    match = _QUANT_PATTERN.search(_basename(name))
    return match.group(1).upper() if match else None


def _params_from_name(name: str) -> str | None:
    match = _PARAMS_PATTERN.search(_basename(name))
    return f"{match.group(1)}B" if match else None


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
        """Every model this server can serve.

        llama-server runs in one of two shapes and they answer this very
        differently, so the mode is detected rather than assumed:

        - ROUTER (`--models-dir PATH`): a real catalogue. `/v1/models` lists
          every GGUF under that directory, loaded or not, and naming one in a
          chat request loads it on demand -- `--models-autoload` is on by
          default. This is the mode worth running: models live wherever the
          user keeps them and switching costs a request, not a restart.
        - SINGLE (`-m FILE`): one model, fixed at launch. `/v1/models` returns
          it with its full path as the id, which is not a name worth showing,
          so `/props` is the better source and is used instead.

        `/props.role` is the discriminator: "router" there, absent otherwise.

        The directory layout the router expects is `<models-dir>/<repo>/*.gguf`
        -- one level, not two. Pointing it at a tree of publisher directories
        finds nothing and says "Loaded 0 local model presets", which is easy to
        read as "the flag did not work".
        """
        props = self._response_json(await self._request("GET", "/props"))
        if props.get("role") == ROUTER_ROLE:
            return await self._list_router_models()
        model = self._model_from_props(props)
        return [model] if model is not None else []

    async def _list_router_models(self) -> list[AIModelInfo]:
        payload = self._response_json(await self._request("GET", "/v1/models"))
        data = payload.get("data")
        if not isinstance(data, list):
            raise AIProviderProtocolError(
                "llama-server model listing is missing the data array."
            )
        parsed = [self._model_from_router_entry(entry) for entry in data]
        return [model for model in parsed if model is not None]

    @staticmethod
    def _model_from_router_entry(raw: object) -> AIModelInfo | None:
        if not isinstance(raw, dict):
            return None
        key = raw.get("id")
        if not isinstance(key, str) or not key:
            return None
        raw_status = raw.get("status")
        status = raw_status if isinstance(raw_status, dict) else {}
        # The launch argv the router would use carries the real .gguf path,
        # which is the only place the quantization appears. The id is the
        # repo directory name, which usually does not carry it.
        path = ""
        args = status.get("args")
        if isinstance(args, list):
            for index, argument in enumerate(args):
                if argument in {"-m", "--model"} and index + 1 < len(args):
                    candidate = args[index + 1]
                    if isinstance(candidate, str):
                        path = candidate
                    break
        return AIModelInfo(
            key=key,
            display_name=key.removesuffix("-GGUF"),
            quantization=_quantization_from_name(path or key),
            params=_params_from_name(path or key),
            loaded=status.get("value") == "loaded",
            # Unknown until the model is actually loaded: the router reports
            # n_ctx per instance, and an unloaded entry has no instance.
            max_context_length=None,
            reasoning_options=(),
        )

    @staticmethod
    def _model_from_props(props: dict[str, Any]) -> AIModelInfo | None:
        raw_path = props.get("model_path")
        # Router mode with nothing loaded reports the string "none" here, which
        # is why this is only ever reached after the role check above.
        if not isinstance(raw_path, str) or not raw_path or raw_path == "none":
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
            params=_params_from_name(name),
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
        """Pick the configured model, or adopt the only one on offer.

        Nothing is loaded here even in router mode -- naming the model on the
        chat request is what triggers the load, so the cost lands on the first
        real call rather than on a preflight that might be for a model the run
        never uses. Expect that first call to take tens of seconds cold.

        The two shapes want opposite things from a mismatch, so the rule is on
        the COUNT rather than the mode. One model on offer means the server was
        launched with it and the configured name is stale bookkeeping: adopt it
        rather than refuse, because the user changing what they launched is a
        decision, not a mistake. Several on offer means the name genuinely
        selects, and a name that is not there has to be reported -- silently
        running a different model than the one configured would put the wrong
        attribution on every extraction it produced.
        """
        models = await self.list_models()
        if not models:
            raise AIModelNotFoundError(
                f"llama-server at {self.settings.base_url} is serving no models."
            )

        configured = self.settings.model
        selected = next((model for model in models if model.key == configured), None)
        if selected is None:
            if len(models) == 1:
                selected = models[0]
            else:
                available = ", ".join(sorted(model.key for model in models)[:8])
                raise AIModelNotFoundError(
                    f"llama-server is not serving {configured!r}. "
                    f"Available: {available}"
                )
        # Set BEFORE probing: the probe goes through `generate`, which falls
        # back to `ensure_model_loaded` when no model is set, and would call
        # straight back into here.
        self._model_info = selected
        self._model_info = await self._probe_reasoning(selected)
        return self._model_info

    async def _probe_reasoning(self, model: AIModelInfo) -> AIModelInfo:
        """Find out whether this model can actually be told to stop thinking.

        Nothing llama-server publishes answers this. `/v1/models` carries
        modalities and no capability block, and `--reasoning-format auto`
        resolves it internally at load time without surfacing the result. So
        the only honest signal is behavioural: ask for no thinking on a trivial
        prompt and look at whether `reasoning_content` comes back empty.

        `enable_thinking` is a chat-TEMPLATE feature, so a model whose template
        has no such switch ignores it and keeps thinking -- with HTTP 200 and
        nothing in the response admitting it. That silence is exactly why this
        has to be measured rather than assumed.

        Costs one small request, and in router mode forces the model load that
        the first extraction would have paid for anyway. Failure is not fatal:
        an unreachable or odd server leaves the capability unknown, which falls
        back to the canonical Pass 1 pair -- the documented degraded path.
        """
        try:
            completion = await self.generate(
                system_prompt="Reply with the single word ok.",
                payload={"ping": "ok"},
                max_tokens=16,
                reasoning_off=True,
            )
        except AIProviderError:
            return model

        thinks_anyway = bool(completion.native_reasoning)
        return AIModelInfo(
            key=model.key,
            display_name=model.display_name,
            quantization=model.quantization,
            params=model.params,
            loaded=True,
            max_context_length=model.max_context_length,
            # ("on",) alone is what `requires_native_reasoning` reads as "cannot
            # be turned off"; ("off", "on") is the ordinary case.
            reasoning_options=("on",) if thinks_anyway else ("off", "on"),
        )

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
