"""Resolve one run's effective settings from flags, stored config, defaults.

Three layers, highest first: a command-line flag, the stored config file, the
built-in default. (Environment variables sit between flag and config for the
few settings that have them -- the database and config paths, and the LM Studio
endpoint -- but those are resolved where they are consumed, not here.)

The resolver records WHERE each value came from, and that provenance is the
point of the module rather than a diagnostic afterthought. A flag is visible in
the command someone typed; a stored default is not. For a tool whose output is
investigative evidence, a run whose behaviour depends on invisible state is a
run nobody can reproduce -- so the scan reports every value it took from the
config file, and stays silent about the rest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from sherlock_project.ai_config import AISettings, SherlockSettings

SettingSource = Literal["flag", "config", "default"]


class IncompleteSettingsError(ValueError):
    """Edits that cannot be written yet without losing their meaning."""


@dataclass(frozen=True, slots=True)
class ResolvedValue:
    value: Any
    source: SettingSource


def resolve_value(
    *,
    flag: Any | None,
    config: Any,
    default: Any,
) -> ResolvedValue:
    """Pick the highest-priority layer that has an opinion.

    `flag` is None when the option was absent from the command line, which is
    why every flag this resolver feeds on has `default=None` in argparse rather
    than its real default: without that, "the user asked for 30" and "nobody
    asked" are the same value and the config layer can never win.
    """
    if flag is not None:
        return ResolvedValue(flag, "flag")
    if config != default:
        return ResolvedValue(config, "config")
    return ResolvedValue(default, "default")


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """One run's effective settings, each with the layer it came from."""

    concurrency: ResolvedValue
    timeout: ResolvedValue
    proxy: ResolvedValue
    nsfw: ResolvedValue
    color: ResolvedValue
    verbose: ResolvedValue

    def from_config(self) -> dict[str, Any]:
        """Settings taken from the stored file, in display order.

        Only these are worth reporting. A flag is already on screen in the
        command line, and a default is what everyone else gets.
        """
        named = (
            ("concurrency", self.concurrency),
            ("timeout", self.timeout),
            ("proxy", self.proxy),
            ("nsfw sites", self.nsfw),
            ("color", self.color),
            ("verbose", self.verbose),
        )
        return {
            name: resolved.value
            for name, resolved in named
            if resolved.source == "config"
        }


def resolve_runtime_settings(
    *,
    stored: SherlockSettings,
    concurrency: int | None,
    timeout: int | None,
    proxy: str | None,
    nsfw: bool | None,
    no_color: bool | None,
    verbose: bool | None,
) -> RuntimeSettings:
    """Combine parsed flags with stored settings into what this run will use.

    `no_color` is inverted on the way in: the flag expresses the negative
    because that is what a terminal user reaches for, while the stored setting
    expresses the positive because that is what reads correctly in a file.
    """
    scan_defaults = type(stored.scan)()
    output_defaults = type(stored.output)()

    return RuntimeSettings(
        concurrency=resolve_value(
            flag=concurrency,
            config=stored.scan.concurrency,
            default=scan_defaults.concurrency,
        ),
        timeout=resolve_value(
            flag=timeout,
            config=stored.scan.timeout,
            default=scan_defaults.timeout,
        ),
        proxy=resolve_value(
            flag=proxy,
            config=stored.scan.proxy,
            default=scan_defaults.proxy,
        ),
        nsfw=resolve_value(
            flag=nsfw,
            config=stored.scan.nsfw,
            default=scan_defaults.nsfw,
        ),
        color=resolve_value(
            flag=(False if no_color else None),
            config=stored.output.color,
            default=output_defaults.color,
        ),
        verbose=resolve_value(
            flag=verbose,
            config=stored.output.verbose,
            default=output_defaults.verbose,
        ),
    )


# --------------------------------------------------------------------------
# The editable settings surface.
#
# Kept here, as data, deliberately: the settings screen renders these and does
# nothing else, so what a setting IS and what it may hold stays testable
# without a terminal. Anything the screen decides on its own is a decision
# nobody can write a test for.
# --------------------------------------------------------------------------

FieldKind = Literal["spin", "toggle", "text", "model"]


@dataclass(frozen=True, slots=True)
class SettingField:
    section: str
    name: str
    label: str
    kind: FieldKind
    # Ordered candidates for a spinner. Values outside the list are still
    # valid -- a hand-edited config is not wrong just because it picked a
    # number this list does not offer -- so stepping moves to the nearest
    # neighbour rather than rejecting what it finds.
    choices: tuple[Any, ...] = ()
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.section}.{self.name}"


SETTING_FIELDS: tuple[SettingField, ...] = (
    SettingField("ai", "model", "model", "model"),
    SettingField("ai", "base_url", "endpoint", "text"),
    SettingField(
        "ai", "temperature", "temperature", "spin",
        tuple(round(step / 10, 1) for step in range(11)),
    ),
    SettingField(
        "ai", "context_length", "context length", "spin",
        (2048, 4096, 8192, 16384, 32768, 65536, 131072),
    ),
    SettingField(
        "scan", "concurrency", "concurrency", "spin",
        (1, 5, 10, 20, 30, 50, 75, 100),
    ),
    SettingField(
        "scan", "timeout", "timeout", "spin",
        (10, 15, 30, 45, 60, 90, 120),
    ),
    SettingField("scan", "proxy", "proxy", "text"),
    SettingField("scan", "nsfw", "NSFW sites", "toggle"),
    SettingField("output", "color", "colour", "toggle"),
    SettingField("output", "verbose", "verbose", "toggle"),
)


def field_values(settings: SherlockSettings) -> dict[str, Any]:
    """Flatten stored settings into {"section.name": value}.

    An absent [ai] section yields None for every AI field rather than being
    skipped, so the screen can show "not configured" in place instead of
    silently dropping rows and changing its own shape.
    """
    values: dict[str, Any] = {}
    for field in SETTING_FIELDS:
        section = getattr(settings, field.section, None)
        values[field.key] = (
            None if section is None else getattr(section, field.name, None)
        )
    return values


def step_value(field: SettingField, current: Any, delta: int) -> Any:
    """Move one step along a field's candidates.

    Stops at the ends rather than wrapping: arrowing past the maximum and
    landing on the minimum is the kind of surprise that sets concurrency to 1
    when someone meant 100.
    """
    if field.kind == "toggle":
        return not bool(current)
    if field.kind != "spin" or not field.choices:
        return current

    choices = field.choices
    if current in choices:
        index = choices.index(current)
    else:
        # A value the list does not offer -- hand-edited, or from a build with
        # different candidates. Move relative to where it would sit.
        index = len([item for item in choices if item < current])
        if delta > 0:
            return choices[min(index, len(choices) - 1)]
        return choices[max(index - 1, 0)]

    return choices[max(0, min(index + delta, len(choices) - 1))]


def apply_values(
    settings: SherlockSettings,
    values: Mapping[str, Any],
) -> SherlockSettings:
    """Rebuild settings from edited values, leaving untouched sections alone.

    Raises ValidationError for anything the schema refuses, which is the point:
    the screen must not be able to write a config the CLI would reject.
    """
    updates: dict[str, Any] = {}
    for section_name in ("scan", "output"):
        section = getattr(settings, section_name)
        changes = {
            field.name: values[field.key]
            for field in SETTING_FIELDS
            if field.section == section_name and field.key in values
        }
        updates[section_name] = section.model_copy(update=changes)

    ai_changes = {
        field.name: values[field.key]
        for field in SETTING_FIELDS
        if field.section == "ai"
        and field.key in values
        and values[field.key] is not None
    }
    if settings.ai is not None:
        if ai_changes:
            updates["ai"] = settings.ai.model_copy(update=ai_changes)
    elif ai_changes:
        # No [ai] section yet, which is every install that has not run
        # `setup ai`. These edits used to be dropped here in silence while the
        # save still reported success -- the worst possible outcome, because
        # nothing on screen said the AI half had not been written.
        if not (ai_changes.get("base_url") and ai_changes.get("model")):
            raise IncompleteSettingsError(
                "AI needs both an endpoint and a model before it can be saved"
            )
        updates["ai"] = AISettings(**ai_changes)

    # model_copy skips validation, so re-validate the whole thing rather than
    # trusting the copy. A screen that can store concurrency 0 would hang the
    # next scan forever with no error anywhere.
    return SherlockSettings.model_validate(
        settings.model_copy(update=updates).model_dump(mode="python")
    )
