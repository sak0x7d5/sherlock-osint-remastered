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

from dataclasses import dataclass
from typing import Any, Literal

from sherlock_project.ai_config import SherlockSettings

SettingSource = Literal["flag", "config", "default"]


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
