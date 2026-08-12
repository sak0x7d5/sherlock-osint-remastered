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
from textual.widgets import DataTable, Footer, Input, Label, Static

from sherlock_project.ai_config import (
    AIConfigError,
    ai_config_path,
    save_settings,
    try_load_settings,
)
from sherlock_project.ai_provider import (
    AIModelInfo,
    AIProviderError,
    LMStudioProvider,
)
from sherlock_project.database import default_database_path
from sherlock_project.settings import (
    SETTING_FIELDS,
    IncompleteSettingsError,
    SettingField,
    apply_values,
    field_values,
    step_value,
)

SECTION_TITLES = {"ai": "AI", "scan": "Scan", "output": "Output"}
LABEL_WIDTH = 18
VALUE_WIDTH = 26


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


def styled_value(field: SettingField, value: Any) -> Text:
    """`render_value`, with the parts that mean something coloured.

    The brackets are the affordance -- they are what tells you this row answers
    to left and right -- so they carry the accent colour and the value does
    not. Uncoloured, they read as decoration and the row looks identical to the
    ones Enter opens.

    "not set" is dimmed and italic so it reads as an absence rather than as a
    value someone chose.
    """
    plain = render_value(field, value)
    text = Text()
    if field.kind in {"spin", "toggle"}:
        text.append("‹", style="bold cyan")
        text.append(plain[1:-1], style="bold")
        text.append("›", style="bold cyan")
    elif plain == "not set":
        text.append(plain, style="dim italic")
    else:
        text.append(plain, style="bold")
    text.pad_right(max(0, VALUE_WIDTH - text.cell_len))
    return text


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


def format_context(length: int | None) -> str:
    """A context window as people talk about it, not as the API reports it.

    131072 is a number to decode; 128K is a fact you can compare against the
    other rows at a glance.
    """
    if not length:
        return "?"
    if length >= 1024 and length % 1024 == 0:
        return f"{length // 1024}K"
    return str(length)


def thinking_label(model: AIModelInfo) -> str:
    """Three distinct answers, not two.

    A model with no reasoning capability at all is not the same as one that can
    be asked to stop, and neither is the same as one that cannot. Pass one is
    written for reasoning-off, so this column is the difference between a model
    that fits and one that is merely allowed.
    """
    if not model.reasoning_options:
        return "none"
    if model.supports_reasoning_off:
        return "optional"
    return "always"


class ModelPickerScreen(ModalScreen[str | None]):
    """Enter on `model`: the live list from LM Studio, as a real table.

    A table rather than one line per model, because the fields are the whole
    point of the screen -- size against quantisation against context window is
    the comparison someone is here to make, and it cannot be made when the
    columns do not line up. As a flat list the longest name pushed everything
    after it out of alignment and wrapped onto a second line.

    Fetched when the screen opens rather than when the app starts, so a stopped
    server costs nothing until someone actually asks for the list, and fails as
    a message in this dialog instead of a dead settings screen.
    """

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "cancel"),
        Binding("ctrl+r", "reload", "refresh"),
    ]

    def __init__(self, base_url: str, current: str | None = None) -> None:
        super().__init__()
        self._base_url = base_url
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Choose a model", classes="dialog-title")
            yield Static("Asking LM Studio...", id="picker-status")
            yield DataTable(id="models", cursor_type="row", zebra_stripes=True)
            yield Label(
                "▸ will be used   ● already in memory (starts instantly)",
                classes="dim",
            )
            yield Label("up/down move   enter select   ^R refresh   esc cancel",
                        classes="dim")

    def on_mount(self) -> None:
        table = self.query_one("#models", DataTable)
        # A two-glyph marker instead of a "state" column. Loaded-in-memory is
        # worth knowing -- a cold first load was measured at 187s, so choosing
        # a resident model is the difference between starting now and waiting
        # three minutes -- but as a column it was blank for every row whenever
        # nothing happened to be loaded, which is most of the time. Paired with
        # the will-be-used marker it always says something about at least
        # one row.
        #
        # "will be used", not "in use" (nothing is running) and not
        # "selected" (the cursor is what is selected on this screen, and
        # Textual's own event for pressing enter is RowSelected).
        table.add_column("", key="mark", width=2)
        table.add_column("model", key="model")
        table.add_column("size", key="size")
        table.add_column("quant", key="quant")
        # Context is here because it decides how much of a page pass one can
        # see, and it was invisible at the moment the choice is made.
        table.add_column("context", key="context")
        table.add_column("thinking", key="thinking")
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        status = self.query_one("#picker-status", Static)
        table = self.query_one("#models", DataTable)
        table.clear()

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
            table.add_row(
                Text(
                    ("▸" if model.key == self._current else " ")
                    + ("●" if model.loaded else " "),
                    style="bold cyan",
                ),
                model.key,
                model.params or "?",
                model.quantization or "?",
                format_context(model.max_context_length),
                thinking_label(model),
                key=model.key,
            )

        # Start on the configured model rather than at the top, so the list
        # opens showing what is in use instead of making someone find it.
        if self._current is not None:
            for index, model in enumerate(models):
                if model.key == self._current:
                    table.move_cursor(row=index)
                    break

        status.update(f"{len(models)} downloaded")
        table.focus()

    def action_reload(self) -> None:
        self.query_one("#picker-status", Static).update("Asking LM Studio...")
        self.run_worker(self._load(), exclusive=True)

    @on(DataTable.RowSelected)
    def choose(self, event: DataTable.RowSelected) -> None:
        self.dismiss(event.row_key.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SettingsApp(App[bool]):
    """The settings screen. Returns True when something was saved."""

    # height: auto on the body, NOT 1fr. With 1fr the rows container ate every
    # spare line and pushed the paths to the floor of the terminal, leaving a
    # lake of empty space in the middle. Sized to content, each block sits
    # under the one above it and the slack collects at the bottom, where
    # nobody has to look at it.
    CSS = """
    Screen { background: $surface; }
    #title { padding: 1 2 0 2; }
    #rows { height: auto; max-height: 1fr; padding: 0 2; }
    .section { color: $accent; text-style: bold; padding: 1 0 0 1; height: 2; }
    .row { padding: 0 1; height: 1; }
    .row-selected { background: $primary 30%; }
    .dim { color: $text-muted; }
    .paths { padding: 1 3 0 3; color: $text-muted; }
    #status { padding: 1 3 0 3; height: auto; }
    #dialog {
        background: $panel; border: round $accent;
        padding: 1 2; width: 90%; max-width: 96; height: auto;
    }
    .dialog-title { text-style: bold; color: $accent; }
    #models { height: auto; max-height: 20; margin: 1 0; }
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
            # Section headings are their own widgets rather than a newline
            # inside the first row of each group. Embedded, they made that row
            # two lines tall while every other row was one, so the cursor
            # appeared to jump unevenly on the way down the list.
            section = None
            for index, field in enumerate(SETTING_FIELDS):
                if field.section != section:
                    section = field.section
                    yield Static(SECTION_TITLES[section], classes="section")
                yield Static(id=f"row-{index}", classes="row")
        yield Static(id="paths", classes="paths")
        yield Static(id="status")
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
                ("Sherlock settings", "bold cyan"),
                "   ",
                ("● unsaved", "bold yellow") if self.dirty else ("saved", "dim"),
            )
        )
        for index, field in enumerate(SETTING_FIELDS):
            row = self.query_one(f"#row-{index}", Static)
            selected = index == self._cursor
            line = Text()
            line.append("▸ " if selected else "  ", style="bold")
            # Fixed label column, then a fixed value column, so the brackets
            # form a straight edge down the screen instead of stepping in and
            # out with the length of each label.
            line.append(
                f"{field.label:<{LABEL_WIDTH}}",
                style="none" if selected else "dim",
            )
            line.append_text(styled_value(field, self._values[field.key]))
            if field.kind in {"text", "model"}:
                line.append("⏎ change", style="dim")
            row.update(line)
            row.set_class(selected, "row-selected")
        self.query_one("#status", Static).update(self._status)

    def action_move(self, delta: int) -> None:
        self._cursor = max(0, min(self._cursor + delta, len(SETTING_FIELDS) - 1))
        # Clear any leftover message. A "Saved to ..." line still sitting there
        # while the title says unsaved is a contradiction the reader has to
        # untangle, and the answer is always "that message is stale".
        self._status = ""
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
                ModelPickerScreen(
                    str(endpoint),
                    current=self._values.get("ai.model"),
                ),
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
        except IncompleteSettingsError as error:
            self._status = f"Cannot save: {error}"
            self._redraw()
            return
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

    # run_async, never run(). App.run() calls asyncio.run() internally, and
    # this is reached from inside main()'s loop, so the synchronous form dies
    # with "asyncio.run() cannot be called from a running event loop" and
    # leaves an un-awaited coroutine behind.
    await SettingsApp(config_path=destination).run_async()
    return 0
