"""`sherlock-rm settings` -- the standalone host for the settings editor.

The editor itself moved to `tui/settings_pane.py` when the unified UI needed the
same rows as one tab among several. What stays here is everything that is about
*this command* rather than about editing settings: argument parsing, the
plain-text fallback for when there is no terminal, and a one-pane app whose only
job is to close when the pane says it is done.

The names the editor exported are re-exported below. They are imported by the
tests and by other modules, and moving a file is not a reason to break a caller
that never cared which file the function lived in.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Footer

from sherlock_project.ai_config import ai_config_path, try_load_settings
from sherlock_project.settings import (
    SETTING_FIELDS,
    field_description,
    field_note,
    field_values,
)
from sherlock_project.tui.settings_pane import (
    LABEL_WIDTH,
    SECTION_TITLES,
    VALUE_WIDTH,
    ModelPickerScreen,
    SettingsPane,
    TextEditScreen,
    format_context,
    plain_value,
    render_value,
    styled_value,
    thinking_label,
)
from sherlock_project.tui.theme import SHERLOCK_CSS

__all__ = [
    "LABEL_WIDTH",
    "SECTION_TITLES",
    "VALUE_WIDTH",
    "ModelPickerScreen",
    "SettingsApp",
    "SettingsPane",
    "TextEditScreen",
    "build_settings_parser",
    "format_context",
    "plain_value",
    "render_value",
    "run_settings",
    "styled_value",
    "thinking_label",
]


class SettingsApp(App[bool]):
    """The settings screen on its own. Returns True when something was saved.

    A shell around `SettingsPane` and nothing more. The state and the key
    handling live on the pane so the unified UI gets the identical editor rather
    than a second one that drifts -- the properties below exist so callers that
    have always asked the app for its values keep working.
    """

    CSS = SHERLOCK_CSS

    def __init__(self, config_path: Path | None = None) -> None:
        super().__init__()
        self._pane = SettingsPane(config_path=config_path or ai_config_path())

    def compose(self) -> ComposeResult:
        yield self._pane
        yield Footer()

    def on_mount(self) -> None:
        # The pane no longer focuses itself -- see the note on its `on_mount`.
        # Here it is the only thing on screen, so it takes focus at once.
        self._pane.focus()

    # The pane owns this state. Returned by reference, not copied: callers
    # legitimately reach in to set a value before driving the screen, and a copy
    # would silently swallow the write.
    @property
    def _values(self) -> dict[str, Any]:
        return self._pane._values

    @property
    def _status(self) -> str:
        return self._pane._status

    @property
    def _cursor(self) -> int:
        return self._pane._cursor

    @property
    def dirty(self) -> bool:
        return self._pane.dirty

    @on(SettingsPane.Closed)
    def _closed(self, event: SettingsPane.Closed) -> None:
        self.exit(event.saved)


def _print_settings(
    values: Mapping[str, Any],
    *,
    config_path: Path,
    console: Console,
) -> None:
    """Plain-text rendering, for when there is no terminal to draw on."""
    section = None
    for field in SETTING_FIELDS:
        if field.section != section:
            section = field.section
            console.print(f"[bold]{SECTION_TITLES[section]}[/bold]")
        value = values[field.key]
        shown = plain_value(field, value)
        line = Text(f"  {field.label:>15}: ", style="dim").append(Text(shown))
        note = field_note(field, value)
        if note:
            line.append(
                Text(f"  {note.text}", style="yellow" if note.warning else "dim")
            )
        console.print(line, soft_wrap=True)

        # Only the RISKY side gets its reason spelled out here, and it gets it
        # in full. There is no cursor to move on this surface -- it is what a
        # redirected `--show` leaves in a file -- so a warning that reads as
        # two words in the editor has to stand on its own in the listing. The
        # safe side needs no paragraph, and giving every row one would bury the
        # values the listing exists to show.
        if note.warning:
            for extra in (field_description(field, value), field.doc_url):
                if extra:
                    console.print(
                        Text(f"{'':>19}{extra}", style="dim"),
                        soft_wrap=True,
                    )
    console.print()
    console.print(Text(f"config file: {config_path}", style="dim"),
                  soft_wrap=True)


def build_settings_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="sherlock-rm settings",
        description=(
            "Edit stored defaults for scans, output and the local AI model. "
            "A command-line flag always overrides these for a single run."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the stored settings and exit, without opening the editor.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable terminal colors.",
    )
    return parser


async def run_settings(
    argv: Sequence[str],
    *,
    config_path: Path | None = None,
    console: Console | None = None,
    interactive: bool | None = None,
) -> int:
    parser = build_settings_parser()
    args = parser.parse_args(list(argv))
    destination = config_path or ai_config_path()
    output = console or Console(
        no_color=args.no_color,
        color_system=None if args.no_color else "auto",
        highlight=False,
    )

    # stdout, not stdin: this screen draws, and a redirected stdout is the case
    # where drawing is wrong. stdin is checked too because a full-screen app
    # with no key source would simply hang.
    can_draw = (
        (sys.stdout.isatty() and sys.stdin.isatty())
        if interactive is None
        else interactive
    )
    if args.show or not can_draw:
        _print_settings(
            field_values(try_load_settings(path=destination)),
            config_path=destination,
            console=output,
        )
        if not args.show:
            output.print(
                "[dim]No terminal to draw on, so the editor was not opened.[/dim]"
            )
        return 0

    # run_async, never run(). App.run() calls asyncio.run() internally, and
    # this is reached from inside main()'s loop, so the synchronous form dies
    # with "asyncio.run() cannot be called from a running event loop" and
    # leaves an un-awaited coroutine behind.
    await SettingsApp(config_path=destination).run_async()
    return 0
