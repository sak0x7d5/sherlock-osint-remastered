"""The colour themes the TUI can be switched between.

`theme.py` decides what the screens look like in terms of ROLES -- accent,
panel, success -- and never names a colour. This file is the other half: the
handful of palettes those roles can resolve to. Kept apart so a palette can be
added without touching a layout rule, and so the stylesheet stays a description
of structure rather than a paint chart.

**The default is a copy of `textual-dark` called `sherlock`, not a new look.**
Everyone who never opens the setting sees exactly the screen they saw before
themes existed, and `tone()` in `theme.py` leaves its ANSI colours alone for it
-- so the default is not merely close to the old one, it is the old one. It is
re-registered under its own name only so the settings row says "sherlock"
rather than naming the toolkit.

**AMOLED themes are black, not dark grey.** `background` and `surface` are
#000000, so on an OLED panel the pixels behind the screen are off. Panels
(appbar, chips, dialogs) step up to #0A0A0A -- just enough to tell a raised
thing from the page -- and the job grey fills usually do, separating one region
from the next, is done by #1F1F1F rules instead. Those rules are
`$panel-lighten-2` in the stylesheet, which is why that variable is pinned
here: derived from a near-black panel it came out too close to black to read
as a line.

Each AMOLED theme carries two accents. `accent` is what is ACTIVE -- the tab
you are on, SCAN, the progress bar. `secondary` is what is WORKING -- the
spinners. One colour for both would make "this is selected" and "this is
still moving" the same signal.

`primary` is set to the accent hue, not left to differ. The stylesheet records
what a stray `$primary` blue did to the default palette (theme.py, the Button
comment); a fresh theme should not repeat it.

**The palettes are derived, not picked.** Measured in OKLCH -- lightness,
chroma and hue as the eye sees them, which hex does not tell you -- the first
cut of these themes failed in three ways nobody would name but everybody would
feel: error was a whole step darker than success and warning, so the three
statuses did not read as one set; the AMOLED "found" mint and the cyan accent
were both bright blue-greens at nearly the same lightness, so a hit and the
selection read as one family; and the
violet theme's lime spinner was the brightest object on screen and 35 degrees
from success green, so "still working" read as "found". The rules that fix it,
held by `tests/test_themes.py` so a palette edit cannot quietly undo them:

- The three statuses sit in one lightness band and at one chroma. Hue says
  which status; weight is the same for all three, so none shouts.
- `accent` and `secondary` keep at least 45 degrees of hue from every status,
  so an active control is never mistaken for a finding.
- `secondary` is never lighter than `accent`. It is the supporting colour.
- The selected row is the accent as a TINT on black, never the accent itself.
  A full-strength cyan bar put the green count of the row it selected at a
  contrast ratio near 1 -- the number vanished on exactly the row being read.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import replace
from typing import TYPE_CHECKING

from textual.color import Color
from textual.theme import BUILTIN_THEMES, Theme

from sherlock_project.ai_config import DEFAULT_THEME, THEME_CHOICES

if TYPE_CHECKING:
    from textual.app import App, SystemCommand

__all__ = [
    "DEFAULT_THEME",
    "SHERLOCK_THEMES",
    "THEME_CHOICES",
    "apply_theme",
    "oklch",
    "register_themes",
    "resolve_theme",
    "without_theme_command",
]

# The rule colour every AMOLED theme shares. See the module docstring.
_AMOLED_RULE = "#1F1F1F"


def _amoled(
    name: str,
    *,
    accent: str,
    secondary: str,
    success: str,
    warning: str,
    error: str,
    foreground: str = "#EDEDED",
) -> Theme:
    black = Color.parse("#000000")
    tint = Color.parse(accent)
    return Theme(
        name=name,
        primary=accent,
        secondary=secondary,
        accent=accent,
        foreground=foreground,
        background="#000000",
        surface="#000000",
        panel="#0A0A0A",
        success=success,
        warning=warning,
        error=error,
        dark=True,
        variables={
            "panel-lighten-1": "#141414",
            "panel-lighten-2": _AMOLED_RULE,
            "panel-lighten-3": "#2E2E2E",
            # The footer paints itself `$panel` by default -- a strip of not-
            # quite-black under a black screen, which reads as a rendering
            # seam rather than as chrome.
            "footer-background": "#000000",
            # Selection as a tint of the accent, not the accent. See the module
            # docstring for what the solid bar did to the row it selected.
            "block-cursor-background": black.blend(tint, 0.32).hex,
            "block-cursor-foreground": foreground,
            "block-cursor-blurred-background": black.blend(tint, 0.16).hex,
        },
    )


SHERLOCK_THEMES: tuple[Theme, ...] = (
    replace(BUILTIN_THEMES["textual-dark"], name=DEFAULT_THEME),
    # OKLCH targets in the comments; the hex is that target, clipped to sRGB
    # where it fell outside. Vivid statuses: L 0.82 / 0.86 / 0.72, chroma ~0.17,
    # hue 150 / 88 / 25. Red cannot be as light as green without turning pink,
    # so error sits at the bottom of the band rather than level with the rest.
    _amoled(
        "amoled",
        accent="#00E5FF",  # cyan    L .84 C .15 h 210
        secondary="#FF76D8",  # orchid  L .76 C .20 h 340 -- softened magenta
        success="#5CE483",
        warning="#FDCA3A",
        error="#FD736D",
    ),
    _amoled(
        "amoled-violet",
        accent="#BF96FF",  # violet  L .75 C .15 h 301
        secondary="#5CB4EF",  # sky     L .74 C .12 h 240 -- analogous, cooler
        success="#5CE483",
        warning="#FDCA3A",
        error="#FD736D",
    ),
    # No hue in the chrome at all, so the statuses are muted to chroma ~0.12:
    # at full strength they were the only colour on screen and read as alarms.
    _amoled(
        "amoled-mono",
        accent="#FFFFFF",
        secondary="#A4A4A4",
        success="#8FE1A1",
        warning="#F6D480",
        error="#F98F87",
        foreground="#D9D9D9",
    ),
)

# THEME_CHOICES -- the order the settings row steps through -- is declared in
# `ai_config` by name, because the config schema needs it and must not import
# Textual. Every name in it is one of the themes above or a Textual built-in
# (the curated few: the rest are a long list to arrow through, and several, the
# ANSI pair especially, look unfinished against this layout). A test holds the
# two lists together.


def oklch(hex_colour: str) -> tuple[float, float, float]:
    """A hex colour as OKLCH (lightness 0-1, chroma, hue in degrees).

    The space the palettes above are reasoned in. Kept here, beside them,
    because "these three colours are equally loud" is a claim about lightness
    as perceived, and sRGB hex cannot answer it -- #FFFF00 and #0000FF are both
    "full brightness" and differ in perceived lightness by a factor of three.
    """
    red, green, blue = (
        _linear(int(hex_colour[index : index + 2], 16) / 255)
        for index in (1, 3, 5)
    )
    long = (0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue) ** (1 / 3)
    mid = (0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue) ** (1 / 3)
    short = (0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue) ** (
        1 / 3
    )
    lightness = 0.2104542553 * long + 0.7936177850 * mid - 0.0040720468 * short
    a = 1.9779984951 * long - 2.4285922050 * mid + 0.4505937099 * short
    b = 0.0259040371 * long + 0.7827717662 * mid - 0.8086757660 * short
    return lightness, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


def _linear(channel: float) -> float:
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def register_themes(app: App) -> None:
    """Teach an app the custom themes. Built-ins it already knows."""
    for theme in SHERLOCK_THEMES:
        app.register_theme(theme)


def resolve_theme(app: App, name: object) -> str:
    """The theme to actually apply for a stored name.

    Falls back rather than raising. The name comes from a config file someone
    may have edited by hand, or one written by a build with themes this one
    lacks, and an unknown palette is a display problem -- which must never stop
    the app from opening.
    """
    if isinstance(name, str) and name in app.available_themes:
        return name
    return DEFAULT_THEME


def without_theme_command(commands: Iterable[SystemCommand]) -> Iterator[SystemCommand]:
    """The command palette's system commands, minus Textual's theme picker.

    That picker lists every theme Textual ships and saves nothing, so a theme
    chosen there was undone by the next tab switch -- which puts the SAVED
    theme back, because that is how a settings preview is kept from outliving
    the settings tab. Two places to choose a theme, one of which forgets, is
    one too many; the settings row is the one that remembers.
    """
    for command in commands:
        if command.title != "Theme":
            yield command


def apply_theme(app: App, name: object) -> None:
    """Switch the app to a stored theme name, falling back if it is unknown."""
    resolved = resolve_theme(app, name)
    if app.theme != resolved:
        app.theme = resolved
