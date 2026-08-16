from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal, TypeVar
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

from sherlock_project.ai_config import AISettings, load_ai_settings
from sherlock_project.ai_provider import (
    AICompletion,
    AIGenerationStats,
    AIModelInfo,
    AIProvider,
    LlamaCppProvider,
    ProviderWideError,
)
from sherlock_project.profile_synthesis import (
    IdentityStatus,
    InvalidExtraction,
    InvestigationContext,
    ProfileSynthesis,
    SiteExtraction,
    SourceDecision,
    SynthesisEvidence,
    aggregate_synthesis,
    build_profile_provenance,
    canonical_field,
    merge_extraction,
    normalize_value,
    synthesis_warnings,
)

PASS_ONE_PROMPT_PATH = Path(__file__).resolve().parent / "resources" / "pass_one.md"
# Pass 1 for models that always think natively. Same extraction rules as
# `pass_one.md`, differing only in the Output section and the examples: it asks
# for the reasoning in native thinking and forbids it in the JSON, so such a
# model does not narrate the work twice against one output budget. Edits to the
# extraction, skip, or key rules must be made in BOTH files -- they are one
# contract with two output shapes, and only this file's copy is under test.
PASS_ONE_NATIVE_REASONING_PROMPT_PATH = (
    Path(__file__).resolve().parent / "resources" / "pass_one_native_reasoning.md"
)
PASS_TWO_PROMPT_PATH = Path(__file__).resolve().parent / "resources" / "pass_two.md"
PASS_TWO_MAX_INPUT_BYTES = 12_000
PASS_TWO_MAX_OUTPUT_TOKENS = 2_048
SAFE_EXTRACTION_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
PASS_ONE_VALIDATION_POLICY_VERSION = "open-dynamic-profile-keys-v4"
PROFILE_CONTENT_EXTRACTION_POLICY_VERSION = "profile-content-v3"
DEFAULT_STRUCTURED_RESPONSE_MAX_TOKENS = 1024
# Extra output budget for models that cannot be told to stop thinking. Their
# native reasoning is spent from the same `max_output_tokens` allowance as the
# answer, so without headroom the JSON is truncated before it is finished and
# every site fails validation. Additive and applied only to those models, so a
# model that honours reasoning-off is charged nothing for this. Not part of
# `pass_one_contract_hash` -- see the note there; the budget is allowed to vary
# by model precisely because the cache contract does not.
NATIVE_REASONING_TOKEN_ALLOWANCE = 2048


SafeExtractionKey = Annotated[
    str,
    StringConstraints(pattern=SAFE_EXTRACTION_KEY.pattern),
]
ProfileFact = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ProfileFactList = Annotated[list[ProfileFact], Field(min_length=1)]


_PASS_ONE_EXCLUDED_KEYS = frozenset(
    {
        "account_activity",
        "account_age",
        "account_type",
        "ar",
        "ar_counts",
        "avatar",
        "avatar_url",
        "average_wpm",
        "avg_wpm",
        "badstanding",
        "best_race_wpm",
        "car",
        "cover",
        "cover_url",
        "current_course_id",
        "experience_level",
        "followers",
        "following",
        "from_language",
        "friends_invited",
        "games_played",
        "games_won",
        "gamesplayed",
        "gameswon",
        "game_time",
        "gametime",
        "join_date",
        "joined_date",
        "karma",
        "karma_level",
        "last_active",
        "last_login",
        "last_visit",
        "learning_language",
        "level",
        "member_since",
        "online_status",
        "points",
        "profile_url",
        "profile_views",
        "rank",
        "ranking",
        "registration_date",
        "referrals",
        "reputation",
        "reputation_title",
        "score",
        "skill_level",
        "state",
        "statistics",
        "status",
        "streak",
        "supporter",
        "supporter_tier",
        "top_wpm",
        "total_credits",
        "total_followers",
        "total_following",
        "total_posts",
        "total_races",
        "total_threads",
        "total_visits",
        "total_vouches",
        "total_xp",
        "trustscan",
        "views",
        "visitors",
        "web_url",
        "wpm_percentage",
    }
)
_PASS_ONE_TELEMETRY_KEY_PARTS = frozenset(
    {
        "badges",
        "comments",
        "credits",
        "favorites",
        "followers",
        "following",
        "games",
        "karma",
        "likes",
        "points",
        "posts",
        "plurks",
        "rank",
        "ranking",
        "replies",
        "responses",
        "referrals",
        "score",
        "streak",
        "submissions",
        "threads",
        "trustscan",
        "views",
        "visits",
        "visitors",
        "vouches",
        "xp",
    }
)
_PASS_ONE_PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "-",
        "--",
        "?",
        "hidden",
        "false",
        "n/a",
        "na",
        "nil",
        "no",
        "no data",
        "no information",
        "none",
        "not available",
        "not found",
        "not specified",
        "null",
        "private",
        "true",
        "unavailable",
        "unknown",
        "yes",
    }
)
_PASS_ONE_GENERIC_ROLE_VALUES = frozenset(
    {
        "kaskuser",
        "member",
        "members",
        "raw racing recruit",
        "rookie level cybersecurity professional",
        "unranked",
        "user",
        "users",
        "film reviews and lists",
    }
)
_PASS_ONE_GENERIC_UI_VALUES = frozenset(
    {
        "awards",
        "current level",
        "groups",
        "items",
        "online time",
    }
)
_PROFILE_TITLE_PATTERN = re.compile(
    r"(?im)^-\s*(?:title|open graph title|twitter title):\s*"
    r"(?P<name>[^(\r\n]{1,100})\s*\(@?(?P<handle>[^)\r\n]+)\)"
)
_ZERO_VALUE_PATTERN = re.compile(r"^[+-]?0+(?:\.0+)?%?$")
_ACCOUNT_METRIC_VALUE_PATTERN = re.compile(
    r"(?i)^(?:"
    r"[\d,.]+\s*(?:"
    r"awards?|badges?|comments?|credits?|favorites?|followers?|following|"
    r"games?|karma|likes?|points?|posts?|replies|scores?|streaks?|"
    r"submissions?|threads?|views?|visits?|visitors?|vouches?|xp"
    r")"
    r"|(?:global\s+)?rank(?:ed)?\s*[:#-]?\s*[\d,.]+"
    r"|top\s+[\d,.]+%"
    r"|(?:last\s+)?active\s+.+"
    r"|online\s+(?:for\s+)?(?:time\s*)?.+"
    r"|(?:\d+\s+hours?(?:,\s*)?)+"
    r"(?:\d+\s+minutes?(?:,\s*)?)?"
    r"(?:\d+\s+seconds?)?"
    r"|(?:joined|registered|member\s+since|last\s+(?:login|visit))\s+.+"
    r"|no\s+(?:awards?|badges?|certifications?|followers?|following|posts?|"
    r"presentations?|threads?|vouches?)"
    r")$"
)


ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)
AIPhase = Literal["pass_one", "pass_two"]


@dataclass(frozen=True, slots=True)
class AIRequestTrace:
    phase: AIPhase
    username: str
    site_name: str
    site_id: int | None
    attempt: int
    provider: str
    model_key: str
    temperature: float | None
    context_length: int | None
    max_tokens: int
    elapsed_seconds: float
    stats: AIGenerationStats
    native_reasoning: str
    final_text: str
    structured_reasoning: str
    validated_output: dict[str, Any] | None
    validation_error: str | None
    # True when this request was sent to a model that always thinks, so native
    # reasoning is the design rather than a model ignoring reasoning-off. The
    # reporter needs the difference: warning about it on every site of a scan
    # that deliberately relies on it is noise that hides the real warnings.
    native_reasoning_expected: bool = False


AITraceCallback = Callable[[AIRequestTrace], None]


class StructuredResponseError(RuntimeError):
    def __init__(
        self,
        error_context: str,
        *,
        stop_reason: str | None = None,
        predicted_tokens: int | None,
        max_tokens: int,
        parsed_type: str,
        final_content_chars: int,
        validation_error: str,
    ) -> None:
        self.error_context = error_context
        self.stop_reason = stop_reason
        self.predicted_tokens = predicted_tokens
        self.max_tokens = max_tokens
        self.parsed_type = parsed_type
        self.final_content_chars = final_content_chars
        self.validation_error = validation_error
        super().__init__(
            f"llama-server did not return a valid structured {error_context} "
            f"(predicted_tokens="
            f"{predicted_tokens if predicted_tokens is not None else 'unknown'}, "
            f"max_tokens={max_tokens}, parsed_type={parsed_type}, "
            f"final_chars={final_content_chars}, "
            f"validation={validation_error})."
        )

    def safe_diagnostics(self) -> str:
        """Return response-shape diagnostics without exposing model content."""

        predicted = (
            str(self.predicted_tokens)
            if self.predicted_tokens is not None
            else "unknown"
        )
        return (
            f"validation={self.validation_error}, "
            f"output_tokens={predicted}/{self.max_tokens}, "
            f"final_chars={self.final_content_chars}, "
            f"stop={self.stop_reason or 'unknown'}"
        )


def _validation_error_code(error: ValidationError) -> str:
    errors = error.errors()
    if not errors:
        return "validation_error"
    error_type = errors[0].get("type")
    return str(error_type) if error_type else "validation_error"


def _render_structured_reasoning(validated: BaseModel | None) -> str:
    if validated is None:
        return ""
    reasoning = getattr(validated, "reasoning", "")
    if isinstance(reasoning, str):
        return reasoning
    if isinstance(reasoning, BaseModel):
        return json.dumps(
            reasoning.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if isinstance(reasoning, (dict, list)):
        # Only reachable on the native-reasoning variant, which allows extra
        # fields: a model that ignored "no reasoning field" may put anything
        # there. Render it so the trace shows the prompt did not land.
        try:
            return json.dumps(reasoning, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return ""
    return ""


class StrictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OSINTResponse(StrictResponse):
    reasoning: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1200),
    ] = Field(
        description=(
            "One short clause per owner-evidence line, in page order, each "
            "either 'include VALUE as KEY' or 'skip: REASON'. Prose only: no "
            "arrays, objects, or JSON. Transient and never stored."
        )
    )
    extraction: dict[SafeExtractionKey, ProfileFactList] = Field(
        description=(
            "Every value marked include in reasoning, under the key named "
            "there, and nothing else. Keys are snake_case; each value is a "
            "nonempty array of nonempty strings."
        ),
        json_schema_extra={"additionalProperties": False},
    )

    @field_validator("extraction")
    @classmethod
    def deduplicate_extraction_values(
        cls,
        extraction: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        return {
            key: list(dict.fromkeys(values))
            for key, values in extraction.items()
        }


class NativeReasoningOSINTResponse(BaseModel):
    """Pass 1 for a model whose native thinking cannot be turned off.

    `OSINTResponse.reasoning` is scaffolding: writing one clause per evidence
    line, in page order, is what stops a model deciding the whole answer in one
    leap and dropping facts on the way. A model that always thinks natively has
    already made that pass before it starts the JSON, so asking for the field
    again buys no accuracy and spends the output budget the answer needs.

    Deliberately neither a base class nor a subclass of `OSINTResponse`. Sharing
    the field through inheritance would reorder that model's JSON Schema
    properties, and its schema is hashed into `pass_one_contract_hash` -- a
    reordering alone would invalidate every cached extraction on disk.

    `extra="allow"`, not `StrictResponse`'s `extra="forbid"`: a model that emits
    `reasoning` out of habit has still produced a usable extraction, and
    rejecting it would recreate the exact failure this variant exists to remove.
    The stray field rides into the trace instead, so a verbose run shows whether
    the prompt actually landed.
    """

    model_config = ConfigDict(extra="allow")

    extraction: dict[SafeExtractionKey, ProfileFactList] = Field(
        description=(
            "Every fact the page states about the owner, under a snake_case "
            "key; each value is a nonempty array of nonempty strings. The only "
            "permitted field -- reason in your own thinking, not in here."
        ),
        json_schema_extra={"additionalProperties": False},
    )

    @field_validator("extraction")
    @classmethod
    def deduplicate_extraction_values(
        cls,
        extraction: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        return {
            key: list(dict.fromkeys(values))
            for key, values in extraction.items()
        }


PassOneResponse = OSINTResponse | NativeReasoningOSINTResponse


def validate_pass_one_extraction_payload(
    extraction: object,
) -> dict[str, list[str]]:
    """Validate one bare cached extraction using the live value contract."""

    return OSINTResponse.model_validate(
        {
            "reasoning": "Validate the cached extraction against the live schema.",
            "extraction": extraction,
        }
    ).extraction


class PassOneKeyRegistry:
    """Keep validated Pass 1 key names isolated by searched username."""

    def __init__(self) -> None:
        self._keys_by_username: dict[str, dict[str, None]] = {}

    def seed(
        self,
        username: str,
        extractions: Iterable[Mapping[str, object]],
    ) -> None:
        self._keys_by_username[username] = {}
        for extraction in extractions:
            self.add(username, extraction)

    def add(self, username: str, extraction: Mapping[str, object]) -> None:
        registry = self._keys_by_username.setdefault(username, {})
        for key in extraction:
            if SAFE_EXTRACTION_KEY.fullmatch(key):
                registry.setdefault(key, None)

    def names(self, username: str) -> list[str]:
        return list(self._keys_by_username.get(username, {}))


def _identity_token(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _is_excluded_pass_one_key(key: str) -> bool:
    if key in _PASS_ONE_EXCLUDED_KEYS:
        return True
    if key.startswith("total_"):
        return True
    if key.endswith(("_count", "_counts")):
        return True
    return bool(set(key.split("_")) & _PASS_ONE_TELEMETRY_KEY_PARTS)


def _is_current_site_url(value: str, site_name: str) -> bool:
    candidate = value.strip()
    if "://" not in candidate:
        if not re.fullmatch(
            r"(?i)(?:www\.)?[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,}"
            r"(?:/[^\s]*)?",
            candidate,
        ):
            return False
        candidate = f"https://{candidate}"
    parsed = urlsplit(candidate)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return False

    hostname = _identity_token(parsed.hostname or "")
    site_words = {
        _identity_token(word)
        for word in re.findall(r"[A-Za-z0-9]+", site_name)
        if len(_identity_token(word)) >= 4
    }
    site_token = _identity_token(site_name)
    return (
        bool(site_token and site_token in hostname)
        or any(word in hostname for word in site_words)
    )


def _profile_metadata_context(
    site_content: str,
    searched_username: str,
) -> tuple[str | None, str | None]:
    metadata = site_content.split("## Main content", maxsplit=1)[0]
    searched_token = _identity_token(searched_username)
    for match in _PROFILE_TITLE_PATTERN.finditer(metadata):
        if _identity_token(match.group("handle")) != searched_token:
            continue
        candidate = " ".join(match.group("name").split()).strip(" -–—|•")
        candidate_token = _identity_token(candidate)
        if (
            not candidate_token
            or candidate_token == searched_token
            or candidate_token == "profile"
            or candidate.casefold().startswith("profile of ")
            or not any(character.isalpha() for character in candidate)
            or len(candidate) > 100
            or len(candidate.split()) > 8
        ):
            return metadata, None
        return metadata, candidate
    return None, None


def _is_excluded_pass_one_value(
    *,
    key: str,
    value: str,
    searched_username: str,
    site_name: str,
    owner_metadata: str | None,
) -> bool:
    collapsed = " ".join(value.split()).strip()
    casefolded = collapsed.casefold()
    if casefolded in _PASS_ONE_PLACEHOLDER_VALUES:
        return True
    if _ZERO_VALUE_PATTERN.fullmatch(casefolded):
        return True
    if _ACCOUNT_METRIC_VALUE_PATTERN.fullmatch(collapsed):
        return True
    if _identity_token(collapsed) == _identity_token(searched_username):
        return True
    if casefolded.startswith("this user has no "):
        return True
    if casefolded.startswith("no ") and casefolded.endswith(
        (" yet", " at this time", " available")
    ):
        return True
    if key in {"role", "roles", "title"} and (
        casefolded in _PASS_ONE_GENERIC_ROLE_VALUES
    ):
        return True
    if key in {"groups", "status", "trustscan"} and (
        casefolded in _PASS_ONE_GENERIC_UI_VALUES
    ):
        return True

    value_site_token = _identity_token(collapsed.removesuffix(".com"))
    if value_site_token == _identity_token(site_name):
        return True
    if _is_current_site_url(collapsed, site_name):
        return True

    # When metadata explicitly identifies the searched profile owner, treat the
    # main body as a feed. Facts must also occur in the owner metadata so posts,
    # replies, and unrelated feed text cannot become profile attributes.
    if owner_metadata is not None:
        value_token = _identity_token(collapsed)
        if len(value_token) >= 4 and value_token not in _identity_token(owner_metadata):
            return True
    return False


def sanitize_pass_one_extraction(
    extraction: Mapping[str, Sequence[str]],
    *,
    searched_username: str,
    site_name: str,
    site_content: str,
) -> dict[str, list[str]]:
    """Remove explicitly forbidden semantic categories from valid Pass 1 JSON."""

    owner_metadata, metadata_name = _profile_metadata_context(
        site_content,
        searched_username,
    )
    sanitized: dict[str, list[str]] = {}
    for key, values in extraction.items():
        if _is_excluded_pass_one_key(key):
            continue
        retained = [
            value
            for value in values
            if not _is_excluded_pass_one_value(
                key=key,
                value=value,
                searched_username=searched_username,
                site_name=site_name,
                owner_metadata=owner_metadata,
            )
        ]
        if retained:
            sanitized[key] = retained

    if metadata_name is not None and not {
        "display_name",
        "full_name",
        "name",
        "real_name",
    }.intersection(sanitized):
        sanitized["full_name"] = [metadata_name]

    location_values = sanitized.get("location", [])
    language_values = sanitized.get("language", [])
    if (
        location_values
        and location_values == language_values
        and all(re.fullmatch(r"[A-Za-z]{2}", value) for value in location_values)
    ):
        sanitized.pop("location", None)
        sanitized.pop("language", None)
    return sanitized


def finalize_pass_one_response[
    PassOneResponseT: (OSINTResponse, NativeReasoningOSINTResponse)
](
    response: PassOneResponseT,
    *,
    searched_username: str,
    site_name: str,
    site_content: str,
) -> PassOneResponseT:
    """Apply deterministic semantic sanitation to the model's extraction."""

    return response.model_copy(
        update={
            "extraction": sanitize_pass_one_extraction(
                response.extraction,
                searched_username=searched_username,
                site_name=site_name,
                site_content=site_content,
            )
        }
    )


def _compact_model_schema(response_model: type[BaseModel]) -> str:
    return json.dumps(
        response_model.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _pass_one_contract_hash(prompt: str) -> str:
    contract = {
        "content_extraction_policy_version": (
            PROFILE_CONTENT_EXTRACTION_POLICY_VERSION
        ),
        "prompt": prompt,
        "schema": _compact_model_schema(OSINTResponse),
        "validation_policy_version": PASS_ONE_VALIDATION_POLICY_VERSION,
    }
    encoded_contract = json.dumps(
        contract,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded_contract).hexdigest()


def pass_one_contract_hash() -> str:
    """Return the model-independent cache contract for current Pass 1.

    Pinned to the canonical prompt and `OSINTResponse`, never to whichever
    variant a given model was actually sent. Both variants ask for the same
    facts under the same key rules and produce the same stored artifact -- only
    `extraction` is ever written to disk, and its value contract is identical.
    Hashing the variant instead would make the hash model-dependent by the back
    door: configuring an always-thinking model would strand every cached
    extraction, and switching back would strand them again.
    """

    return _pass_one_contract_hash(
        PASS_ONE_PROMPT_PATH.read_text(encoding="utf-8")
    )


class TargetDecision(StrictResponse):
    identity_status: IdentityStatus = Field(
        description=(
            "strong_match for a clear explicit or semantic anchor match, unsure "
            "for partial or ambiguous compatibility, or reject for conflicting "
            "or irrelevant evidence."
        )
    )


class AIService:
    def __init__(
        self,
        *,
        provider: AIProvider | None = None,
        settings: AISettings | None = None,
        trace_callback: AITraceCallback | None = None,
    ) -> None:
        self._provider = provider
        self._settings = settings or (
            provider.settings if provider is not None else None
        )
        self._trace_callback = trace_callback
        self._extraction_prompt = ""
        self._native_reasoning_extraction_prompt = ""
        self._identity_prompt = ""
        self._closed = False
        self._model_info: AIModelInfo | None = None
        self._load_prompts()

    @classmethod
    async def create(
        cls,
        settings: AISettings | None = None,
        *,
        provider: AIProvider | None = None,
        trace_callback: AITraceCallback | None = None,
    ) -> AIService:
        resolved_settings = settings or (
            provider.settings if provider is not None else load_ai_settings()
        )
        resolved_provider = provider or LlamaCppProvider(resolved_settings)
        self = cls(
            provider=resolved_provider,
            settings=resolved_settings,
            trace_callback=trace_callback,
        )
        try:
            self._model_info = await resolved_provider.ensure_model_loaded()
        except BaseException:
            await self.close()
            raise
        return self

    def _load_prompts(self) -> None:
        self._extraction_prompt = PASS_ONE_PROMPT_PATH.read_text(encoding="utf-8")
        self._native_reasoning_extraction_prompt = (
            PASS_ONE_NATIVE_REASONING_PROMPT_PATH.read_text(encoding="utf-8")
        )
        self._identity_prompt = PASS_TWO_PROMPT_PATH.read_text(encoding="utf-8")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._provider is not None:
            await self._provider.close()
        self._provider = None

    async def extract_profile(
        self,
        username: str,
        site_name: str,
        site_content: str,
        *,
        known_profile_keys: Sequence[str],
    ) -> PassOneResponse:
        if self._provider is None:
            raise RuntimeError("Load a model before calling extract_profile().")

        # A model that always thinks gets the variant that asks for the
        # reasoning in its native thinking and forbids it in the JSON, so it is
        # not charged twice for one traversal. Prompt and schema move together:
        # the schema alone would not beat two worked examples, because nothing
        # in this request enforces the schema -- it is sent as prompt text.
        native = self.uses_native_reasoning
        response_model = (
            NativeReasoningOSINTResponse if native else OSINTResponse
        )
        return await self._respond_structured(
            phase="pass_one",
            username=username,
            site_name=site_name,
            site_id=None,
            attempt=1,
            system_prompt=(
                self._native_reasoning_extraction_prompt
                if native
                else self._extraction_prompt
            ),
            payload={
                "searched_username_do_not_extract": username,
                "site_name": site_name,
                "known_profile_keys": list(known_profile_keys),
                "site_content": site_content,
            },
            response_model=response_model,
            error_context="OSINT extraction",
            max_tokens=self.pass_one_max_tokens,
            # Same request either way -- `generate` already falls back to "on"
            # when "off" is not offered. Saying so is honest about the variant
            # relying on that thinking rather than tolerating it.
            reasoning_off=not native,
            native_reasoning_expected=native,
            transform=lambda response: finalize_pass_one_response(
                response,
                searched_username=username,
                site_name=site_name,
                site_content=site_content,
            ),
        )

    @property
    def uses_native_reasoning(self) -> bool:
        """True when the loaded model thinks natively and cannot be stopped.

        False when the model info is unknown, which is the case for a service
        constructed around a provider directly rather than through `create`.
        Pass 1 is designed around reasoning-off, so the conservative reading of
        "unknown" is "behaves like every model this pipeline was tuned for".
        """
        return (
            self._model_info is not None
            and self._model_info.requires_native_reasoning
        )

    @property
    def pass_one_max_tokens(self) -> int:
        """Pass 1 output budget, widened for models that always think."""
        if not self.uses_native_reasoning:
            return DEFAULT_STRUCTURED_RESPONSE_MAX_TOKENS
        return (
            DEFAULT_STRUCTURED_RESPONSE_MAX_TOKENS
            + NATIVE_REASONING_TOKEN_ALLOWANCE
        )

    @property
    def model_key(self) -> str:
        return self._settings.model if self._settings is not None else "unconfigured"

    @property
    def pass_one_contract_hash(self) -> str:
        # Canonical prompt only, whatever this service will actually send --
        # see `pass_one_contract_hash` for why the variant stays out of it.
        return _pass_one_contract_hash(self._extraction_prompt)

    @property
    def synthesis_prompt_fingerprints(self) -> dict[str, str]:
        fingerprints = {
            "identity": sha256(self._identity_prompt.encode("utf-8")).hexdigest(),
            "target_schema": sha256(
                self._compact_schema(TargetDecision).encode("utf-8")
            ).hexdigest(),
            "max_output_tokens": str(PASS_TWO_MAX_OUTPUT_TOKENS),
        }
        if self._settings is not None:
            fingerprints.update(
                {
                    "provider": self._settings.provider,
                    "native_reasoning": "on",
                    "temperature": str(self._settings.temperature),
                    "context_length": str(self._settings.context_length),
                }
            )
        return fingerprints

    async def synthesize(
        self,
        username: str,
        extractions: list[SiteExtraction],
        context: InvestigationContext,
        *,
        input_hash: str,
        pending_site_ids: list[int] | None = None,
        invalid_extractions: list[InvalidExtraction] | None = None,
    ) -> ProfileSynthesis:
        evidence = SynthesisEvidence(
            extractions=extractions,
            pending_site_ids=pending_site_ids or [],
            invalid_extractions=invalid_extractions or [],
        )
        if not context.has_anchors:
            return aggregate_synthesis(
                username=username,
                input_hash=input_hash,
                context=context,
                evidence=evidence,
            )

        if self._provider is None:
            raise RuntimeError("Load a model before anchored profile synthesis.")

        strong_profile: dict[str, Any] = {}
        decisions: dict[int, SourceDecision] = {}
        warnings = synthesis_warnings(evidence)
        provider_unavailable = False

        unsure_extractions: list[SiteExtraction] = []

        for extraction in sorted(extractions, key=lambda item: item.site_id):
            if not extraction.extraction:
                decisions[extraction.site_id] = SourceDecision(
                    site_id=extraction.site_id,
                    site_name=extraction.site_name,
                    site_url=extraction.site_url,
                    disposition="ignored",
                )
                continue
            if provider_unavailable:
                decisions[extraction.site_id] = SourceDecision(
                    site_id=extraction.site_id,
                    site_name=extraction.site_name,
                    site_url=extraction.site_url,
                    disposition="failed",
                )
                continue

            try:
                decision = await self._assess_target(
                    username=username,
                    context=context,
                    strong_profile=strong_profile,
                    extraction=extraction,
                )
            except ProviderWideError:
                provider_unavailable = True
                decisions[extraction.site_id] = SourceDecision(
                    site_id=extraction.site_id,
                    site_name=extraction.site_name,
                    site_url=extraction.site_url,
                    disposition="failed",
                )
                warnings.append(
                    "The AI provider became unavailable during pass two; "
                    "remaining sources were left unresolved."
                )
                continue
            except Exception as error:
                decisions[extraction.site_id] = SourceDecision(
                    site_id=extraction.site_id,
                    site_name=extraction.site_name,
                    site_url=extraction.site_url,
                    disposition="failed",
                )
                diagnostics = (
                    f" [{error.safe_diagnostics()}]"
                    if isinstance(error, StructuredResponseError)
                    else ""
                )
                warnings.append(
                    f"Pass-two decision failed for site id {extraction.site_id} "
                    f"({type(error).__name__}){diagnostics}; "
                    "its facts were not merged."
                )
                continue

            if decision.identity_status == "strong_match":
                disposition = "included"
                merge_extraction(strong_profile, extraction.extraction, username=username)
            elif decision.identity_status == "unsure":
                disposition = "included"
                unsure_extractions.append(extraction)
            else:
                disposition = "excluded"

            decisions[extraction.site_id] = SourceDecision(
                site_id=extraction.site_id,
                site_name=extraction.site_name,
                site_url=extraction.site_url,
                disposition=disposition,
                identity_status=decision.identity_status,
            )

        # Sweep 2
        for extraction in unsure_extractions:
            if provider_unavailable:
                break
            try:
                decision = await self._assess_target(
                    username=username,
                    context=context,
                    strong_profile=strong_profile,
                    extraction=extraction,
                )
            except Exception:
                continue

            if decision.identity_status == "strong_match":
                merge_extraction(strong_profile, extraction.extraction, username=username)
                decisions[extraction.site_id].identity_status = "strong_match"
            elif decision.identity_status == "reject":
                decisions[extraction.site_id].disposition = "excluded"
                decisions[extraction.site_id].identity_status = "reject"

        # Rebuild the two output profiles from final source decisions. An unsure
        # source promoted during sweep two must not remain in both profiles.
        final_strong_profile: dict[str, Any] = {}
        final_unsure_profile: dict[str, Any] = {}
        extractions_by_site_id = {
            extraction.site_id: extraction
            for extraction in extractions
        }
        for site_id in sorted(decisions):
            decision = decisions[site_id]
            extraction = extractions_by_site_id[site_id]
            if decision.identity_status == "strong_match":
                merge_extraction(
                    final_strong_profile,
                    extraction.extraction,
                    username=username,
                )
            elif decision.identity_status == "unsure":
                merge_extraction(
                    final_unsure_profile,
                    extraction.extraction,
                    username=username,
                )

        source_decisions = [decisions[site_id] for site_id in sorted(decisions)]
        visible_profile: dict[str, Any] = {}
        merge_extraction(
            visible_profile,
            final_strong_profile,
            username=username,
        )
        merge_extraction(
            visible_profile,
            final_unsure_profile,
            username=username,
        )
        included_extractions = [
            extractions_by_site_id[decision.site_id]
            for decision in source_decisions
            if decision.identity_status in {"strong_match", "unsure"}
        ]
        included = any(
            decision.disposition == "included" for decision in source_decisions
        )
        failed = any(
            decision.disposition == "failed" for decision in source_decisions
        )
        return ProfileSynthesis(
            username=username,
            input_hash=input_hash,
            mode="anchored",
            resolution_status="resolved" if included else "insufficient_evidence",
            completeness=("partial" if failed else evidence.completeness),
            strong_profile=final_strong_profile,
            unsure_profile=final_unsure_profile,
            provenance=build_profile_provenance(
                visible_profile,
                included_extractions,
            ),
            source_decisions=source_decisions,
            anchors=context.anchors,
            warnings=warnings,
        )

    async def _assess_target(
        self,
        *,
        username: str,
        context: InvestigationContext,
        strong_profile: dict[str, Any],
        extraction: SiteExtraction,
    ) -> TargetDecision:
        payload: dict[str, object] = {
            "username": username,
            "current_site": {
                "site_name": extraction.site_name,
                "extraction": extraction.extraction,
            },
        }
        anchors = self._format_anchors(context)
        if anchors:
            payload["anchors"] = anchors
        if strong_profile:
            payload["strong_profile"] = strong_profile
        request_size = len(
            (
                self._structured_system_prompt(
                    self._identity_prompt,
                    TargetDecision,
                )
                + "\n"
                + json.dumps(payload, ensure_ascii=False)
            ).encode("utf-8")
        )
        if request_size > PASS_TWO_MAX_INPUT_BYTES:
            raise RuntimeError(
                f"Pass-two evidence for site id {extraction.site_id} exceeds "
                f"the {PASS_TWO_MAX_INPUT_BYTES}-byte input budget."
            )
        return await self._respond_structured(
            phase="pass_two",
            username=username,
            site_name=extraction.site_name,
            site_id=extraction.site_id,
            attempt=1,
            system_prompt=self._identity_prompt,
            payload=payload,
            response_model=TargetDecision,
            error_context=f"target decision for site id {extraction.site_id}",
            max_tokens=PASS_TWO_MAX_OUTPUT_TOKENS,
            reasoning_off=False,
        )

    @staticmethod
    def _format_anchors(
        context: InvestigationContext,
    ) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        normalized: dict[str, set[str]] = {}
        for anchor in context.anchors:
            field_name = canonical_field(anchor.field)
            normalized_value = normalize_value(field_name, anchor.value)
            seen = normalized.setdefault(field_name, set())
            if normalized_value in seen:
                continue
            seen.add(normalized_value)
            grouped.setdefault(field_name, []).append(anchor.value)
        return {field_name: grouped[field_name] for field_name in sorted(grouped)}

    @staticmethod
    def _compact_schema(response_model: type[BaseModel]) -> str:
        return _compact_model_schema(response_model)

    @classmethod
    def _structured_system_prompt(
        cls,
        system_prompt: str,
        response_model: type[BaseModel],
    ) -> str:
        """Say what shape is wanted. Do not paste the schema.

        The schema now rides in `response_format`, where llama.cpp compiles it
        to a grammar and the model cannot emit anything else. Sending it here
        as well would spend roughly 700 characters of a budget Pass 1 is
        already short of, to restate a rule that is no longer advisory.

        This also retires a trap worth naming: while the schema was prompt
        text, it could LOSE to the worked examples beneath it, so the two had
        to be edited in lockstep or the model followed the examples and failed
        validation. Enforcement removes that coupling -- the grammar wins
        regardless of what the examples show.

        `response_model` is still taken so callers cannot forget which model
        the reply will be validated against, and so the signature survives if
        a future provider needs the text form back.
        """
        return (
            system_prompt.rstrip()
            + "\n\nReturn exactly one JSON object. "
            + "Do not use Markdown fences or add text outside the object."
        )

    def _emit_trace(self, trace: AIRequestTrace) -> None:
        if self._trace_callback is None:
            return
        try:
            self._trace_callback(trace)
        except Exception:
            # Reporting must not change extraction or synthesis behavior.
            pass

    async def _respond_structured(
        self,
        *,
        phase: AIPhase,
        username: str,
        site_name: str,
        site_id: int | None,
        attempt: int,
        system_prompt: str,
        payload: dict[str, object],
        response_model: type[ResponseModelT],
        error_context: str,
        max_tokens: int = DEFAULT_STRUCTURED_RESPONSE_MAX_TOKENS,
        reasoning_off: bool = True,
        native_reasoning_expected: bool = False,
        transform: Callable[[ResponseModelT], ResponseModelT] | None = None,
    ) -> ResponseModelT:
        if self._provider is None:
            raise RuntimeError("Load a model before requesting a response.")

        completion: AICompletion | None = None
        validation_error_code: str | None = None
        validated: ResponseModelT | None = None
        request_started_at = perf_counter()
        try:
            completion = await self._provider.generate(
                system_prompt=self._structured_system_prompt(
                    system_prompt,
                    response_model,
                ),
                payload=payload,
                max_tokens=max_tokens,
                reasoning_off=reasoning_off,
                json_schema=response_model.model_json_schema(),
            )
            if not completion.final_text:
                validation_error_code = "empty_final_response"
                raise StructuredResponseError(
                    error_context,
                    predicted_tokens=completion.stats.output_tokens,
                    max_tokens=max_tokens,
                    parsed_type="str",
                    final_content_chars=0,
                    validation_error=validation_error_code,
                )
            try:
                validated = response_model.model_validate_json(
                    completion.final_text
                )
            except ValidationError as error:
                validation_error_code = _validation_error_code(error)
                raise StructuredResponseError(
                    error_context,
                    predicted_tokens=completion.stats.output_tokens,
                    max_tokens=max_tokens,
                    parsed_type="str",
                    final_content_chars=len(completion.final_text),
                    validation_error=validation_error_code,
                ) from error
            if transform is not None:
                validated = transform(validated)
            return validated
        except Exception as error:
            if validation_error_code is None:
                validation_error_code = type(error).__name__
            raise
        finally:
            stats = completion.stats if completion is not None else AIGenerationStats()
            self._emit_trace(
                AIRequestTrace(
                    phase=phase,
                    username=username,
                    site_name=site_name,
                    site_id=site_id,
                    attempt=attempt,
                    provider=(
                        self._settings.provider
                        if self._settings is not None
                        else "unconfigured"
                    ),
                    model_key=self.model_key,
                    temperature=(
                        self._settings.temperature
                        if self._settings is not None
                        else None
                    ),
                    context_length=(
                        self._settings.context_length
                        if self._settings is not None
                        else None
                    ),
                    max_tokens=max_tokens,
                    elapsed_seconds=(
                        completion.elapsed_seconds
                        if completion is not None
                        else perf_counter() - request_started_at
                    ),
                    stats=stats,
                    native_reasoning=(
                        completion.native_reasoning if completion is not None else ""
                    ),
                    final_text=(
                        completion.final_text if completion is not None else ""
                    ),
                    structured_reasoning=(
                        _render_structured_reasoning(validated)
                    ),
                    validated_output=(
                        validated.model_dump(mode="json")
                        if validated is not None
                        else None
                    ),
                    validation_error=validation_error_code,
                    native_reasoning_expected=native_reasoning_expected,
                )
            )
