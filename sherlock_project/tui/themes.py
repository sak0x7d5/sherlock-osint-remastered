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
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import replace
from typing import TYPE_CHECKING

from textual.theme import BUILTIN_THEMES, Theme

from sherlock_project.ai_config import DEFAULT_THEME, THEME_CHOICES

if TYPE_CHECKING:
    from textual.app import App, SystemCommand

__all__ = [
    "DEFAULT_THEME",
    "SHERLOCK_THEMES",
    "THEME_CHOICES",
    "apply_theme",
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
        },
    )


SHERLOCK_THEMES: tuple[Theme, ...] = (
    replace(BUILTIN_THEMES["textual-dark"], name=DEFAULT_THEME),
    _amoled(
        "amoled",
        accent="#00E5FF",
        secondary="#FF2BD6",
        success="#00FF9C",
        warning="#FFD60A",
        error="#FF3B5C",
    ),
    _amoled(
        "amoled-violet",
        accent="#B388FF",
        secondary="#C6FF00",
        success="#69F0AE",
        warning="#FFD740",
        error="#FF5252",
    ),
    _amoled(
        "amoled-mono",
        accent="#FFFFFF",
        secondary="#9E9E9E",
        success="#7CFFB2",
        warning="#FFE066",
        error="#FF6B6B",
        foreground="#D9D9D9",
    ),
)

# THEME_CHOICES -- the order the settings row steps through -- is declared in
# `ai_config` by name, because the config schema needs it and must not import
# Textual. Every name in it is one of the themes above or a Textual built-in
# (the curated few: the rest are a long list to arrow through, and several, the
# ANSI pair especially, look unfinished against this layout). A test holds the
# two lists together.


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
