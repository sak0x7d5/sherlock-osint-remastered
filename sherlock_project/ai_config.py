"""Persistent configuration for Sherlock's local AI provider."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import tomli_w
from platformdirs import user_config_path
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# 2 added the [scan] and [output] sections. A file written by an older build
# still loads: missing sections take their defaults. A file written by a NEWER
# build is refused with an explanation rather than a pydantic validation dump,
# because extra="forbid" would otherwise turn "you upgraded elsewhere" into an
# unreadable error.
# 3 added scan.webbrowser. The bump is not optional bookkeeping: extra="forbid"
# means a build that predates the key rejects any file containing it, so
# without the version the older build reports a validation dump instead of
# "this was written by a newer version".
# 4 added ai.unload_after_minutes, for the same reason.
# 5 moved the provider from LM Studio to llama.cpp. `ai.unload_after_minutes`
# is RETIRED -- llama-server does not load or unload anything, so there was no
# behaviour left behind the setting. It is stripped on read rather than
# rejected: extra="forbid" would otherwise meet every config file written by
# builds 4 and earlier with a pydantic dump, for a key that now does nothing.
CONFIG_VERSION = 5
DEFAULT_LLAMACPP_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_AI_TEMPERATURE = 0.1
DEFAULT_AI_CONTEXT_LENGTH = 8192
# Keys accepted and discarded on read, newest first. A key belongs here once
# nothing consumes it, so that upgrading never fails on a value that is merely
# obsolete.
RETIRED_AI_KEYS = ("unload_after_minutes",)
DEFAULT_SCAN_CONCURRENCY = 30
DEFAULT_SCAN_TIMEOUT = 60
DEFAULT_SCAN_WEBBROWSER = True


class AIConfigError(RuntimeError):
    """Raised when Sherlock's AI configuration is missing or invalid."""


class AISettings(BaseModel):
    """Runtime settings for one configured AI provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["llamacpp"] = "llamacpp"
    base_url: str
    # Advisory. llama-server serves whatever GGUF it was launched with and
    # cannot be told to switch, so this selects nothing -- it is recorded
    # against extractions so results say which model produced them, and it is
    # what `setup ai` stores after adopting the running server's model.
    model: str
    temperature: float = Field(
        default=DEFAULT_AI_TEMPERATURE,
        ge=0,
        le=1,
    )
    # Also advisory now, and it did not use to be: LM Studio took a context
    # length per request, llama-server fixes it at launch with `-c`. Kept
    # because the Pass 1 budget arithmetic needs a number to reason against,
    # but it DESCRIBES how the server was started rather than controlling it.
    # Setting it higher than the server's real window does not widen anything.
    context_length: int = Field(
        default=DEFAULT_AI_CONTEXT_LENGTH,
        ge=512,
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("must be an absolute HTTP(S) URL")
        return value

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class ScanSettings(BaseModel):
    """Defaults for what a scan does. These change what a scan FINDS.

    Which is why anything sourced from here is echoed at scan start: a flag is
    visible in the command someone typed, a stored default is not, and an
    investigation whose results depend on invisible state is not reproducible.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    concurrency: int = Field(default=DEFAULT_SCAN_CONCURRENCY, ge=1)
    timeout: int = Field(default=DEFAULT_SCAN_TIMEOUT, ge=1)
    proxy: str | None = None
    nsfw: bool = False
    # Fetch sites with a real browser (default) or with plain HTTPS requests.
    # This is the setting in this file with the largest effect on what a scan
    # FINDS: off means no JavaScript runs, so sites that build their profile
    # page in the browser can report a real account as absent. Default True,
    # and it stays True -- the browser is not overhead, it is the accuracy.
    webbrowser: bool = DEFAULT_SCAN_WEBBROWSER


class OutputSettings(BaseModel):
    """Defaults for how results are shown. These change nothing about them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    color: bool = True
    verbose: bool = False


class SherlockSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = CONFIG_VERSION
    # Optional so scan and output preferences can be stored by someone who
    # never configures a model. It was required when [ai] was the only section.
    ai: AISettings | None = None
    scan: ScanSettings = ScanSettings()
    output: OutputSettings = OutputSettings()

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be 1 or greater")
        if value > CONFIG_VERSION:
            raise ValueError(
                f"is {value}, but this build of Sherlock understands "
                f"{CONFIG_VERSION}. The file was written by a newer version"
            )
        return value


def ai_config_path(
    environ: Mapping[str, str] | None = None,
) -> Path:
    environment = os.environ if environ is None else environ
    override = environment.get("SHERLOCK_CONFIG")
    if override:
        return Path(override).expanduser()
    return user_config_path("sherlock", appauthor=False) / "config.toml"


def _read_settings(path: Path) -> SherlockSettings:
    try:
        with path.open("rb") as config_file:
            payload = tomllib.load(config_file)
    except FileNotFoundError as error:
        raise AIConfigError(
            "AI is not configured. Run `sherlock setup ai` first."
        ) from error
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise AIConfigError(
            f"Unable to read AI configuration at {path}: {error}"
        ) from error

    ai_section = payload.get("ai")
    if isinstance(ai_section, dict):
        for retired in RETIRED_AI_KEYS:
            ai_section.pop(retired, None)

    try:
        return SherlockSettings.model_validate(payload)
    except ValidationError as error:
        raise AIConfigError(
            f"Invalid AI configuration at {path}: {error}"
        ) from error


def load_settings(
    path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> SherlockSettings:
    """Read the whole config file, or raise AIConfigError explaining why not."""
    environment = os.environ if environ is None else environ
    return _read_settings(path or ai_config_path(environment))


def load_settings_or_default(
    path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[SherlockSettings, str | None]:
    """Read the config, and say so when it could not be read.

    Same fallback as `try_load_settings` -- stored preferences must never stop
    a scan -- but the reason comes back with it instead of vanishing. Silence
    was wrong in exactly the case a version bump makes reachable: a file
    written by a newer build fails validation, every setting reverts to its
    default, and the scan echo stays quiet because it only reports
    config-sourced values and nothing is config-sourced any more. The run is
    then not the run the user configured, with nothing on screen saying so.
    """
    try:
        return load_settings(path=path, environ=environ), None
    except AIConfigError as error:
        return SherlockSettings(), str(error)


def try_load_settings(
    path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> SherlockSettings:
    """Read the config, falling back to all-defaults when it cannot be read.

    Scan and output preferences must never stop a scan: an absent or corrupt
    config means "no stored preferences", not "refuse to run". AI settings keep
    the strict loader, because there the file is the only source and a silent
    default would point at the wrong endpoint.

    Prefer `load_settings_or_default` anywhere there is a surface to report on:
    this one discards WHY the file was unusable, which is fine for a screen
    that is about to show the defaults it fell back to, and not fine for a scan
    whose behaviour just changed without saying so.
    """
    return load_settings_or_default(path=path, environ=environ)[0]


def load_ai_settings(
    path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AISettings:
    environment = os.environ if environ is None else environ
    settings = _read_settings(path or ai_config_path(environment)).ai
    if settings is None:
        raise AIConfigError(
            "AI is not configured. Run `sherlock setup ai` first."
        )
    base_url_override = environment.get("LLAMA_SERVER_BASE_URL")
    if base_url_override:
        try:
            settings = settings.model_copy(
                update={
                    "base_url": AISettings.validate_base_url(base_url_override),
                }
            )
        except ValueError as error:
            raise AIConfigError(
                f"Invalid LLAMA_SERVER_BASE_URL: {error}"
            ) from error
    return settings


def try_load_ai_settings(
    path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AISettings | None:
    try:
        return load_ai_settings(path=path, environ=environ)
    except AIConfigError:
        return None


def save_ai_settings(
    settings: AISettings,
    path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Write the AI section, preserving every other section already stored.

    Rebuilding the file from just the AI settings would silently discard a
    user's scan and output preferences every time they changed model.
    """
    destination = path or ai_config_path(environ)
    existing = try_load_settings(path=destination, environ=environ)
    return save_settings(
        existing.model_copy(update={"ai": settings}),
        path=destination,
        environ=environ,
    )


def save_settings(
    settings: SherlockSettings,
    path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    destination = path or ai_config_path(environ)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = settings.model_copy(
        update={"version": CONFIG_VERSION}
    ).model_dump(mode="python", exclude_none=True)
    serialized = tomli_w.dumps(payload)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(serialized, encoding="utf-8", newline="\n")
        temporary.replace(destination)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise AIConfigError(
            f"Unable to save AI configuration at {destination}: {error}"
        ) from error
    return destination
