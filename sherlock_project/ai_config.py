"""Persistent configuration for Sherlock's local AI provider."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import tomllib
from typing import Literal
from urllib.parse import urlsplit

from platformdirs import user_config_path
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
import tomli_w


CONFIG_VERSION = 1
DEFAULT_LM_STUDIO_BASE_URL = "http://127.0.0.1:1234"
DEFAULT_AI_TEMPERATURE = 0.1
DEFAULT_AI_CONTEXT_LENGTH = 8192


class AIConfigError(RuntimeError):
    """Raised when Sherlock's AI configuration is missing or invalid."""


class AISettings(BaseModel):
    """Runtime settings for one configured AI provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["lmstudio"] = "lmstudio"
    base_url: str
    model: str
    temperature: float = Field(
        default=DEFAULT_AI_TEMPERATURE,
        ge=0,
        le=1,
    )
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


class SherlockSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = CONFIG_VERSION
    ai: AISettings


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

    try:
        return SherlockSettings.model_validate(payload)
    except ValidationError as error:
        raise AIConfigError(
            f"Invalid AI configuration at {path}: {error}"
        ) from error


def load_ai_settings(
    path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AISettings:
    environment = os.environ if environ is None else environ
    settings = _read_settings(path or ai_config_path(environment)).ai
    base_url_override = environment.get("LM_STUDIO_BASE_URL")
    if base_url_override:
        try:
            settings = settings.model_copy(
                update={
                    "base_url": AISettings.validate_base_url(base_url_override),
                }
            )
        except ValueError as error:
            raise AIConfigError(
                f"Invalid LM_STUDIO_BASE_URL: {error}"
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
    destination = path or ai_config_path(environ)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = SherlockSettings(ai=settings).model_dump(mode="python")
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
