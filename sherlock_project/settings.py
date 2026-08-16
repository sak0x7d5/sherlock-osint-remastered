"""Resolve one run's effective settings from flags, stored config, defaults.

Three layers, highest first: a command-line flag, the stored config file, the
built-in default. (Environment variables sit between flag and config for the
few settings that have them -- the database and config paths, and the llama-server
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

from sherlock_project.ai_config import (
    DEFAULT_LLAMACPP_BASE_URL,
    AISettings,
    OutputSettings,
    ScanSettings,
    SherlockSettings,
)

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
    webbrowser: ResolvedValue
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
            ("web browser", self.webbrowser),
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
    webbrowser: bool | None,
    no_webbrowser: bool | None,
    no_color: bool | None,
    verbose: bool | None,
) -> RuntimeSettings:
    """Combine parsed flags with stored settings into what this run will use.

    `no_color` is inverted on the way in: the flag expresses the negative
    because that is what a terminal user reaches for, while the stored setting
    expresses the positive because that is what reads correctly in a file.

    The transport takes BOTH flags, and they are mutually exclusive at the
    parser. The positive one is not redundant with the default: it is the only
    way to override a stored `webbrowser = false` for one run, on the setting
    where being unable to do that changes whether the answers are right.
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
        webbrowser=resolve_value(
            flag=(True if webbrowser else (False if no_webbrowser else None)),
            config=stored.scan.webbrowser,
            default=scan_defaults.webbrowser,
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

# A field this build ships no value for. Distinct from None, which IS the
# shipped value of `scan.proxy` -- "no proxy" is an answer, while "no model"
# is the absence of one and cannot be reset to.
NO_DEFAULT = object()

# PLACEHOLDER. The page does not exist yet and the repository is still private,
# so this 404s for anyone who follows it. Tracked in TODO; do not ship a release
# pointing at example.com. Every message carrying it states its reason in full
# first, so it is always a "read more" and never the only explanation.
#
# PRINTED AS PLAIN TEXT, never as a terminal hyperlink. That was tried and
# measured on 2026-08-12: Textual does not emit OSC 8 for a Rich `link` style at
# all, so in the settings screen -- the surface that most looked like it wanted
# a link -- it was underlined text pretending to be clickable. Rich itself only
# emits OSC 8 outside the legacy Windows console, so even the scan warning
# worked in Windows Terminal and not in cmd.exe. A plain URL is honest, is
# copyable everywhere, and terminals that autodetect URLs make it clickable
# themselves without us claiming they will.
# EMPTY, deliberately. It held "https://example.com" as a placeholder, which
# meant every surface carrying it -- the settings help line, the plain-text
# listing, the scan warning -- offered a "read more" that went nowhere. An
# obviously fake link is worse than no link: it reads as an oversight in a tool
# whose whole job is telling you what is real. Put the real URL here when the
# page exists; everything that prints it already guards on it being non-empty,
# so nothing needs changing but this line.
TRANSPORT_DOC_URL = ""


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
    doc_url: str = ""
    # Only for fields the schema marks required but the tool still has a
    # conventional starting value for -- the llama-server endpoint is required
    # because there is nothing sane to fall back to at load time, yet
    # "put it back to the usual one" is a real thing to want.
    default: Any = NO_DEFAULT
    # Printed after the value. For the fields whose number means nothing on its
    # own: `‹ 5 ›` does not say five of what, and the label column is too narrow
    # to answer it for every row that needs it.
    unit: str = ""
    # What 0 means, when it means something other than zero-of-the-unit.
    # "unload after 0 min" reads as "immediately" and means the exact opposite,
    # so the sentinel is spelled out rather than shown as a number.
    zero_label: str = ""

    @property
    def key(self) -> str:
        return f"{self.section}.{self.name}"


SETTING_FIELDS: tuple[SettingField, ...] = (
    SettingField("ai", "model", "model", "model"),
    SettingField(
        "ai", "base_url", "endpoint", "text",
        default=DEFAULT_LLAMACPP_BASE_URL,
    ),
    SettingField("ai", "models_dir", "models folder", "text"),
    SettingField(
        "ai", "temperature", "temperature", "spin",
        tuple(round(step / 10, 1) for step in range(11)),
    ),
    SettingField(
        "ai", "context_length", "context length", "spin",
        (2048, 4096, 8192, 16384, 32768, 65536, 131072),
    ),
    # `ai.unload_after_minutes` used to sit here. It is gone with LM Studio:
    # llama-server neither loads nor unloads, so the row would have offered a
    # choice that changed nothing.
    # First in its section because it is the setting here with the largest
    # effect on what a scan finds -- everything below it changes how fast or
    # how broad the scan is, this one changes whether an answer is right.
    SettingField(
        "scan", "webbrowser", "web browser", "toggle",
        doc_url=TRANSPORT_DOC_URL,
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

    An absent [ai] section yields each field's BUILT-IN DEFAULT rather than
    None, and None only where the build genuinely ships no answer -- which is
    `ai.model` alone, because which model is right depends on what the user
    has downloaded.

    It used to yield None for every AI field, on the reasoning that the screen
    should show "not configured" rather than imply a setting nobody chose.
    That read well and looked terrible: before `setup ai`, three of the four
    AI rows were empty, and the two spinners rendered the literal word `None`
    between their arrows -- a value no user picked, cannot mean anything, and
    could not be arrowed away from sensibly. The absence is still visible, on
    the one row where it is true and actionable.
    """
    values: dict[str, Any] = {}
    for field in SETTING_FIELDS:
        section = getattr(settings, field.section, None)
        if section is not None:
            values[field.key] = getattr(section, field.name, None)
            continue
        default = field_default(field)
        values[field.key] = None if default is NO_DEFAULT else default
    return values


SECTION_MODELS: dict[str, type] = {
    "ai": AISettings,
    "scan": ScanSettings,
    "output": OutputSettings,
}


def field_default(field: SettingField) -> Any:
    """What this build ships for a field, or NO_DEFAULT if it ships nothing.

    Read off the pydantic model rather than restated here, so the screen's idea
    of a default can never drift from the one the config loader and the CLI
    use. `ai.model` is the one field with no answer -- which model is right
    depends on what the user has downloaded, so there is nothing to restore.
    """
    if field.default is not NO_DEFAULT:
        return field.default
    info = SECTION_MODELS[field.section].model_fields[field.name]
    return NO_DEFAULT if info.is_required() else info.default


@dataclass(frozen=True, slots=True)
class FieldFlag:
    """A terse label riding on a row, and whether it is the risky side.

    `warning` drives colour rather than the screen guessing from the words:
    "faster; inaccurate" and "slower; accurate" are the two halves of one
    choice, and only one of them should look like an alarm.
    """

    text: str = ""
    warning: bool = False

    def __bool__(self) -> bool:
        return bool(self.text)


def field_note(field: SettingField, value: Any) -> FieldFlag:
    """The two- or three-word label that rides on the row itself.

    BOTH sides of a genuine trade-off are labelled, not just the risky one.
    "slower; accurate" beside the default is what says the default was a
    decision rather than an oversight, and it is the only thing that makes the
    faster mode discoverable to someone who never opens the help line. What
    must not grow labels is every unrelated row: eleven captions is wallpaper,
    and the one worth reading would be lost in it.

    It states the TRADE, never the mechanism -- two words cannot carry
    "client-side rendering" without lying by compression, and the trade is what
    the reader is choosing between anyway. The mechanism belongs in
    `field_description`, which has room to be accurate.

    This short because it has to survive the cursor being on some other row,
    which is exactly when a warning gets missed.
    """
    if field.key == "scan.webbrowser":
        if value:
            return FieldFlag("slower; accurate")
        return FieldFlag("faster; inaccurate", warning=True)
    return FieldFlag()


def field_description(field: SettingField, value: Any) -> str:
    """What the selected setting does, in the terms of someone using the tool.

    Shown for whichever row the cursor is on, so every field can afford a real
    sentence -- printed against all eleven rows at once this would be a wall
    nobody reads, which is why the inline notes stay terse.

    Keyed on the value where the value changes the answer. A transport is not
    "the browser setting", it is either "this is why the results are
    trustworthy" or "this is what you are giving up", and only one of those is
    true at a time.

    Lives here rather than in the screen so the wording is testable without a
    terminal, and so both surfaces read from one source.
    """
    # WORDING, decided 2026-08-12 and worth not re-litigating. "No JavaScript"
    # names the mechanism correctly but sounds like a smaller thing than it is.
    # "Dynamic sites" is the tempting plain-English swap and is WRONG in the
    # direction that matters: a server-rendered site is dynamic and arrives
    # complete, so the fast path handles it fine. The failure is specifically
    # CLIENT-SIDE RENDERING -- the server sends a shell and the page is built
    # afterwards, in the browser. So: describe that, in plain words, and name
    # the term once so anyone who wants to look it up can.
    if field.key == "scan.webbrowser":
        if value:
            return (
                "Loads each page the way a real browser does, including the "
                "code that runs after the page arrives. Slower, and the reason "
                "the results can be trusted."
            )
        return (
            "Fetches only what the server sends back. Sites that assemble "
            "their profile page in the browser afterwards (client-side "
            "rendering) arrive as an empty shell, and some refuse a plain "
            "request outright -- so a real account can be reported as absent."
        )
    return _STATIC_DESCRIPTIONS.get(field.key, "")


# Descriptions that do not depend on the current value. Held apart from
# SETTING_FIELDS so the field table stays a table of what a setting IS, rather
# than growing a paragraph per row.
_STATIC_DESCRIPTIONS: dict[str, str] = {
    "ai.model": (
        "Which local model produced a result, recorded against extractions. "
        "llama-server runs whatever GGUF it was started with, so this reports "
        "rather than chooses -- to change model, restart the server."
    ),
    "ai.base_url": (
        "Where llama-server is listening. The LLAMA_SERVER_BASE_URL "
        "environment variable overrides this for a single run."
    ),
    "ai.models_dir": (
        "Folder holding your models, one directory per model. Sherlock starts "
        "llama-server from it when none is running; a server you started "
        "yourself is used as-is and this is ignored."
    ),
    "ai.temperature": (
        "How much the model varies its wording. Extraction is not a creative "
        "task, so low keeps it literal."
    ),
    "ai.context_length": (
        "How much of a page the model can read at once. llama-server fixes "
        "this at launch with -c, so this describes how you started it rather "
        "than changing it; raising it here does not widen the real window."
    ),
    "scan.concurrency": (
        "How many sites are checked at the same time. Higher is faster until "
        "the network or the sites themselves push back."
    ),
    "scan.timeout": (
        "Seconds to wait for one site before giving up. A site that times out "
        "is recorded as inconclusive -- never as absent."
    ),
    "scan.proxy": (
        "Route every request through this proxy. Applies to both transports."
    ),
    "scan.nsfw": (
        "Include the sites the manifest flags as NSFW. They are skipped by "
        "default."
    ),
    "output.color": "Colour terminal output.",
    "output.verbose": (
        "Show the diagnostic detail behind a run: per-request traces, model "
        "failures, and what was skipped."
    ),
}


def value_label(field: SettingField, value: Any) -> str:
    """A field's value as words, before any surface decorates it.

    Lives here, beside `unit` and `zero_label`, because it is the same decision
    they are: what a stored number MEANS is part of what the setting is, not
    something the screen invents. Both surfaces that print a value read it from
    here, so a setting cannot end up reading two ways in two places.

    Returns `str(value)` untouched for every field that declares neither, which
    is all of them but one.
    """
    if field.zero_label and value == 0 and not isinstance(value, bool):
        return field.zero_label
    if field.unit and value is not None and value != "":
        return f"{value} {field.unit}"
    return str(value)


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
        # Name what is actually missing. The endpoint now arrives pre-filled
        # with its default, so "AI needs both an endpoint and a model" sent
        # people hunting for a second problem that was not there.
        missing = [
            label
            for name, label in (("base_url", "an endpoint"), ("model", "a model"))
            if not ai_changes.get(name)
        ]
        if missing:
            raise IncompleteSettingsError(
                f"AI needs {' and '.join(missing)} before it can be saved"
            )
        updates["ai"] = AISettings(**ai_changes)

    # model_copy skips validation, so re-validate the whole thing rather than
    # trusting the copy. A screen that can store concurrency 0 would hang the
    # next scan forever with no error anywhere.
    return SherlockSettings.model_validate(
        settings.model_copy(update=updates).model_dump(mode="python")
    )
