"""The full-screen settings editor behind `sherlock settings`.

Two interactions, because one is not enough. Bounded values -- concurrency,
timeout, temperature, on/off -- are spun in place with the arrow keys, shown as
`‹ 30 ›`. Open-ended ones -- an endpoint, a proxy, a model key -- cannot be
arrowed to, so Enter opens an editor or a picker for those. Screens that try to
express a URL as a spinner are where this pattern falls apart.

Everything about what a setting IS lives in settings.py as data. This module
renders it and nothing else, so the rules stay testable without a terminal.

Box-drawing and ‹› are safe here in a way they are not in scan output: this
screen only ever runs on a real terminal. It refuses to launch otherwise, which
is the whole reason the false-terminal fix came first -- a full-screen app that
takes over a CI log is worse than the crash it replaces.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from pydantic import ValidationError
from rich.console import Console
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Input, Label, ListItem, ListView, Static

from sherlock_project.ai_config import (
    AIConfigError,
    ai_config_path,
    save_settings,
    try_load_settings,
)
from sherlock_project.ai_provider import AIProviderError, LMStudioProvider
from sherlock_project.database import default_database_path
from sherlock_project.settings import (
    SETTING_FIELDS,
    SettingField,
    apply_values,
    field_values,
    step_value,
)

SECTION_TITLES = {"ai": "AI", "scan": "Scan", "output": "Output"}


def render_value(field: SettingField, value: Any) -> str:
    """One field's value as it appears on screen.

    Spinners and toggles carry their ‹ › brackets so it is obvious which rows
    respond to left and right; the rest read as plain values so it is equally
    obvious which do not.
    """
    if field.kind == "toggle":
        return f"‹ {'on' if value else 'off':^7} ›"
    if field.kind == "spin":
        return f"‹ {value!s:^7} ›"
    if value in (None, ""):
        return "not set"
    return str(value)


class TextEditScreen(ModalScreen[str | None]):
    """Enter on an open-ended field: a single input, Esc to abandon."""

    BINDINGS: ClassVar = [Binding("escape", "cancel", "cancel")]

    def __init__(self, label: str, value: str) -> None:
        super().__init__()
        self._label = label
        self._value = value

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"Set {self._label}")
            yield Input(value=self._value, id="edit")
            yield Label("⏎ accept    esc cancel", classes="dim")

    def on_mount(self) -> None:
        self.query_one("#edit", Input).focus()

    @on(Input.Submitted)
    def accept(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)


class ModelPickerScreen(ModalScreen[str | None]):
    """Enter on `model`: the live list from LM Studio.

    Fetched when the screen opens rather than when the app starts, so a
    stopped server costs nothing until someone actually asks for the list --
    and fails as a message in a dialog instead of a dead settings screen.
    """

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "cancel"),
        Binding("ctrl+r", "reload", "refresh"),
    ]

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self._base_url = base_url

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Choose a model")
            yield Static("Asking LM Studio…", id="picker-status")
            yield ListView(id="models")
            yield Label("↑↓ move   ⏎ select   ^R refresh   esc cancel",
                        classes="dim")

    def on_mount(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        status = self.query_one("#picker-status", Static)
        listing = self.query_one("#models", ListView)
        await listing.clear()

        from sherlock_project.ai_config import AISettings

        try:
            settings = AISettings(base_url=self._base_url, model="listing")
            provider = LMStudioProvider(settings)
        except ValueError as error:
            status.update(f"Endpoint is not usable: {error}")
            return

        try:
            models = await provider.list_models()
        except AIProviderError as error:
            status.update(f"{error}\nStart LM Studio, then press ^R.")
            return
        finally:
            await provider.close()

        if not models:
            status.update("LM Studio has no downloaded LLMs.")
            return

        models.sort(key=lambda item: (item.display_name.casefold(), item.key))
        for model in models:
            marks = [model.params or "?", model.quantization or "?"]
            if model.loaded:
                marks.append("loaded")
            if not model.supports_reasoning_off:
                marks.append("always thinks")
            await listing.append(
                ListItem(Static(f"{model.key}   {'  ·  '.join(marks)}"),
                         name=model.key)
            )
        status.update(f"{len(models)} downloaded")
        listing.focus()

    def action_reload(self) -> None:
        self.query_one("#picker-status", Static).update("Asking LM Studio…")
        self.run_worker(self._load(), exclusive=True)

    @on(ListView.Selected)
    def choose(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.name)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SettingsApp(App[bool]):
    """The settings screen. Returns True when something was saved."""

    CSS = """
    Screen { background: $surface; }
    #rows { height: 1fr; padding: 1 2; }
    .section { color: $accent; text-style: bold; padding-top: 1; }
    .row { padding-left: 2; }
    .row-selected { background: $accent 20%; }
    .dim { color: $text-muted; }
    .paths { padding: 1 2 0 2; color: $text-muted; }
    #dialog {
        background: $panel; border: round $accent;
        padding: 1 2; width: 70; height: auto;
    }
    ModalScreen { align: center middle; }
    """

    BINDINGS: ClassVar = [
        Binding("up", "move(-1)", "move", show=False),
        Binding("down", "move(1)", "move", show=False),
        Binding("left", "step(-1)", "change", show=False),
        Binding("right", "step(1)", "change", show=False),
        Binding("enter", "edit", "edit"),
        Binding("r", "reset", "reset row"),
        Binding("ctrl+s", "save", "save"),
        Binding("escape", "leave", "quit"),
    ]

    def __init__(self, config_path: Path | None = None) -> None:
        super().__init__()
        self._config_path = config_path or ai_config_path()
        self._stored = try_load_settings(path=self._config_path)
        self._values = field_values(self._stored)
        self._saved = dict(self._values)
        self._cursor = 0
        self._status = ""

    @property
    def dirty(self) -> bool:
        return self._values != self._saved

    def compose(self) -> ComposeResult:
        yield Static(id="title")
        with VerticalScroll(id="rows"):
            for index, field in enumerate(SETTING_FIELDS):
                yield Static(id=f"row-{index}", classes="row")
        yield Static(id="paths", classes="paths")
        yield Static(id="status", classes="dim")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#paths", Static).update(
            Text(f"database  {default_database_path()}\n"
                 f"config    {self._config_path}")
        )
        self._redraw()

    def _redraw(self) -> None:
        self.query_one("#title", Static).update(
            Text.assemble(
                " Sherlock settings ",
                ("● unsaved", "bold yellow") if self.dirty else ("saved", "dim"),
            )
        )
        section = None
        for index, field in enumerate(SETTING_FIELDS):
            row = self.query_one(f"#row-{index}", Static)
            heading = ""
            if field.section != section:
                section = field.section
                heading = f"{SECTION_TITLES[field.section]}\n"
            marker = "▸" if index == self._cursor else " "
            line = Text()
            if heading:
                line.append(heading.rstrip("\n") + "\n", style="bold cyan")
            line.append(f"{marker} {field.label:<16}", style="none")
            line.append(render_value(field, self._values[field.key]))
            if field.kind in {"text", "model"}:
                line.append("   ⏎ change", style="dim")
            row.update(line)
            row.set_class(index == self._cursor, "row-selected")
        self.query_one("#status", Static).update(self._status)

    def action_move(self, delta: int) -> None:
        self._cursor = max(0, min(self._cursor + delta, len(SETTING_FIELDS) - 1))
        self._redraw()

    def action_step(self, delta: int) -> None:
        field = SETTING_FIELDS[self._cursor]
        current = self._values[field.key]
        if current is None:
            self._status = f"{field.label} is unset — press ⏎ to set it"
        else:
            self._values[field.key] = step_value(field, current, delta)
            self._status = ""
        self._redraw()

    def action_edit(self) -> None:
        field = SETTING_FIELDS[self._cursor]
        if field.kind == "model":
            endpoint = self._values.get("ai.base_url")
            if not endpoint:
                self._status = "Set the endpoint first."
                self._redraw()
                return
            self.push_screen(
                ModelPickerScreen(str(endpoint)),
                lambda chosen: self._accept(field, chosen),
            )
            return
        if field.kind == "text":
            self.push_screen(
                TextEditScreen(field.label, str(self._values[field.key] or "")),
                lambda edited: self._accept(field, edited or None),
            )
            return
        self.action_step(1)

    def _accept(self, field: SettingField, value: Any) -> None:
        if value is not None or field.kind == "text":
            self._values[field.key] = value
        self._redraw()

    def action_reset(self) -> None:
        field = SETTING_FIELDS[self._cursor]
        self._values[field.key] = self._saved[field.key]
        self._status = f"{field.label} back to its saved value"
        self._redraw()

    def action_save(self) -> None:
        try:
            updated = apply_values(self._stored, self._values)
        except ValidationError as error:
            first = error.errors()[0]
            self._status = f"Cannot save: {'.'.join(map(str, first['loc']))} {first['msg']}"
            self._redraw()
            return
        try:
            save_settings(updated, path=self._config_path)
        except AIConfigError as error:
            self._status = str(error)
            self._redraw()
            return
        self._stored = updated
        self._saved = dict(self._values)
        self._status = f"Saved to {self._config_path}"
        self._redraw()

    def action_leave(self) -> None:
        if self.dirty and not self._status.startswith("Unsaved"):
            self._status = "Unsaved changes — ^S to save, esc again to discard"
            self._redraw()
            return
        self.exit(self._saved != field_values(try_load_settings(path=self._config_path)))


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
        if value is None or value == "":
            shown = "not set"
        elif field.kind == "toggle":
            # "on"/"off" here too: the same setting must not read as True
            # in one surface and on in the other.
            shown = "on" if value else "off"
        else:
            shown = str(value)
        console.print(
            Text(f"  {field.label:>15}: ", style="dim").append(Text(shown)),
            soft_wrap=True,
        )
    console.print()
    console.print(Text(f"config file: {config_path}", style="dim"),
                  soft_wrap=True)


def build_settings_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="sherlock settings",
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

    SettingsApp(config_path=destination).run()
    return 0
