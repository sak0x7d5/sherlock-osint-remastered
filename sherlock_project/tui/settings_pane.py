"""The settings editor, as a pane two different hosts can mount.

It began as a whole `App` behind `sherlock settings`, which was right while that
was the only way to reach it. The unified UI needs the same editor as one tab
among several, and an `App` cannot be a tab -- it owns the terminal, the event
loop and the exit. So the editor is a widget, and both hosts mount it: the
standalone command wraps it in a one-pane app, the unified UI puts it in a tab.

Nothing about the editing model changed in the move. Two interactions, because
one is not enough. Bounded values -- concurrency, timeout, temperature, on/off
-- are spun in place with the arrow keys, shown as `‹ 30 ›`. Open-ended ones --
an endpoint, a proxy, a model key -- cannot be arrowed to, so Enter opens an
editor or a picker for those. Screens that try to express a URL as a spinner are
where this pattern falls apart.

Everything about what a setting IS lives in settings.py as data. This module
renders it and nothing else, so the rules stay testable without a terminal.

**Leaving is a message, not an exit.** The pane cannot call `App.exit` -- in the
unified UI there is no exit to call, Escape just means "I am done with this
tab". So it posts `SettingsPane.Closed` and each host decides what that means.
The unsaved-changes guard stays here, where the unsaved changes are.

Box-drawing and ‹› are safe here in a way they are not in scan output: this pane
only ever runs on a real terminal. `run_settings` refuses to launch otherwise,
which is the whole reason the false-terminal fix came first -- a full-screen app
that takes over a CI log is worse than the crash it replaces.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import ValidationError
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import DataTable, DirectoryTree, Input, Label, Static, Tree

from sherlock_project.ai_config import (
    AIConfigError,
    ai_config_path,
    save_settings,
    try_load_settings,
)
from sherlock_project.ai_provider import (
    AIModelInfo,
    AIProviderError,
    LlamaCppProvider,
)
from sherlock_project.database import default_database_path
from sherlock_project.llama_server import (
    LlamaServerError,
    ManagedLlamaServer,
    discover_models,
)
from sherlock_project.settings import (
    NO_DEFAULT,
    SETTING_FIELDS,
    IncompleteSettingsError,
    SettingField,
    apply_values,
    field_default,
    field_description,
    field_note,
    field_values,
    step_value,
    value_label,
)

SECTION_TITLES = {"ai": "AI", "scan": "Scan", "output": "Output"}
LABEL_WIDTH = 18
VALUE_WIDTH = 26


def plain_value(field: SettingField, value: Any) -> str:
    """A value as words, without the spinner brackets.

    For anywhere a value appears inside a sentence or a plain-text listing,
    where "‹ on ›" reads as an artefact. on/off rather than True/False for the
    same reason it is spelled that way on screen: one setting must not read two
    different ways on two surfaces.
    """
    if value is None or value == "":
        return "not set"
    if field.kind == "toggle":
        return "on" if value else "off"
    return value_label(field, value)


def render_value(field: SettingField, value: Any) -> str:
    """One field's value as it appears on screen.

    Spinners and toggles carry their ‹ › brackets so it is obvious which rows
    respond to left and right; the rest read as plain values so it is equally
    obvious which do not.
    """
    if field.kind == "toggle":
        return f"‹ {'on' if value else 'off':^7} ›"
    if field.kind == "spin":
        return f"‹ {value_label(field, value):^7} ›"
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


class FolderPickerScreen(ModalScreen[str | None]):
    """Enter on `models folder`: browse to it instead of typing it.

    Nobody should have to recall an absolute path to switch on an optional
    feature, and this row used to open an empty text box that expected exactly
    that.

    A Textual tree rather than the OS folder dialog, deliberately. `tkinter` is
    importable on a typical Windows install, so the native picker LOOKS
    available -- but it is a separate system package on Debian and Ubuntu, and
    it cannot draw at all over SSH, in WSL without an X server, in a container,
    or on a headless box. Those are ordinary places to run an OSINT scan. A
    dialog that works on the developer's desktop and hangs on a user's remote
    session is worse than no dialog, and this widget works wherever the rest of
    the TUI already does.

    The count beside the path is the point of the screen, not decoration: it is
    the only way to tell "this is where my models are" from "this looks right"
    before committing, and it is computed with the same recursive search the
    server will use, so what it reports is what will be served.
    """

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "cancel"),
    ]

    def __init__(self, current: str | None = None) -> None:
        super().__init__()
        self._current = current
        start = Path(current).expanduser() if current else Path.home()
        # Falling back up the tree rather than to the filesystem root: a stored
        # folder on a drive that is not mounted today should still open
        # somewhere recognisable rather than at C:\ or /.
        while not start.is_dir() and start != start.parent:
            start = start.parent
        self._root = start if start.is_dir() else Path.home()

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Choose your models folder", classes="dialog-title")
            yield Static(str(self._root), id="folder-path")
            yield DirectoryTree(str(self._root), id="folders")
            yield Label(
                "up/down move   → open   ← back   enter use this folder   "
                "esc cancel",
                classes="dim",
            )

    def on_mount(self) -> None:
        self.query_one("#folders", DirectoryTree).focus()

    @on(Tree.NodeHighlighted)
    def preview(self, event: Tree.NodeHighlighted) -> None:
        """Say how many models are under whatever the cursor is on.

        Off the event loop because it walks the filesystem, and re-entrant by
        design: `exclusive=True` cancels the previous count, so holding an
        arrow key does not queue a scan per row.
        """
        path = getattr(event.node, "data", None)
        target = getattr(path, "path", None)
        if target is None:
            return
        self.run_worker(self._count(Path(target)), exclusive=True)

    async def _count(self, folder: Path) -> None:
        label = self.query_one("#folder-path", Static)
        label.update(f"{folder}  …")
        if not folder.is_dir():
            label.update(str(folder))
            return
        found = await asyncio.to_thread(discover_models, [folder])
        # Same recursive search the server will run, so this number is what
        # would actually be served rather than an estimate.
        label.update(
            f"{folder}  —  {len(found)} model{'' if len(found) == 1 else 's'}"
        )

    @on(DirectoryTree.DirectorySelected)
    def choose(self, event: DirectoryTree.DirectorySelected) -> None:
        self.dismiss(str(event.path))

    def action_cancel(self) -> None:
        self.dismiss(None)


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """What the picker comes back with.

    Two values because the screen can change two things. Someone who arrives
    with their models in the wrong place changes the folder AND then picks from
    it, and returning only the model key would silently drop the folder they
    just chose -- so the next open would show the old list again.
    """

    model: str | None = None
    models_dir: str | None = None


# Row key for the "change folder" entry. A DataTable row rather than a Button
# below the table: this TUI is keyboard-only, new ctrl+ bindings are banned
# (TODO records a test enforcing it) and function keys are unreliable, so a row
# needs no new binding, reuses the "enter select" hint already on screen, and
# cannot be tabbed past unnoticed the way a button under a long list can.
FOLDER_ROW_KEY = "__change_models_folder__"


class ModelPickerScreen(ModalScreen[ModelChoice]):
    """Enter on `model`: every model that can be used, as a real table.

    The folder lives HERE, as the first row, because choosing where models are
    and choosing which one to use are one task, not two features. Someone who
    downloads a model somewhere new, or who has never set a folder at all,
    fixes it without leaving the screen they are already looking at.

    A table rather than one line per model, because the fields are the whole
    point of the screen -- size against quantisation against context window is
    the comparison someone is here to make, and it cannot be made when the
    columns do not line up. As a flat list the longest name pushed everything
    after it out of alignment and wrapped onto a second line.

    Starts a llama-server if none is running, and stops the one it started when
    the screen closes. It used to instruct the user to go and start one, which
    was the last place in the tool still asking them to run a server by hand --
    and it did so on precisely the first run where the list is most needed.

    The work happens when the screen opens rather than at app start, so nothing
    is spawned until someone actually asks for the list, and a failure is a
    message in this dialog instead of a dead settings screen.
    """

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "cancel"),
        Binding("ctrl+r", "reload", "refresh"),
    ]

    def __init__(
        self,
        base_url: str,
        current: str | None = None,
        models_dir: str | None = None,
    ) -> None:
        super().__init__()
        self._base_url = base_url
        self._current = current
        self._models_dir = models_dir
        self._server: ManagedLlamaServer | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Choose a model", classes="dialog-title")
            yield Static("Looking for models...", id="picker-status")
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
        # No "thinking" column. Whether a model can be told to stop thinking is
        # in nothing llama-server publishes, so filling it would mean loading
        # every model in the list to ask. It is measured once, against the
        # model actually chosen, when a scan starts.
        self.run_worker(self._load(), exclusive=True)

    def _add_folder_row(self) -> None:
        """First row, always, whatever else happened.

        Added before anything that can fail, so the way to fix a wrong or
        missing folder is on screen even when the listing errored -- which is
        exactly the moment it is needed.
        """
        table = self.query_one("#models", DataTable)
        table.add_row(
            Text("▸" if self._models_dir is None else " ", style="bold cyan"),
            Text("Change models folder…", style="bold"),
            Text(self._models_dir or "not set", style="dim"),
            Text(""),
            Text(""),
            key=FOLDER_ROW_KEY,
        )

    async def _load(self) -> None:
        status = self.query_one("#picker-status", Static)
        table = self.query_one("#models", DataTable)
        table.clear()
        self._add_folder_row()

        from sherlock_project.ai_config import AISettings

        if not self._models_dir:
            status.update(
                "No models folder set — press enter on the first row to choose one."
            )
            table.focus()
            return

        try:
            settings = AISettings(
                base_url=self._base_url,
                model="listing",
                models_dir=self._models_dir,
            )
        except ValueError as error:
            status.update(f"Endpoint is not usable: {error}")
            return

        # Start a server rather than telling the user to. This screen used to
        # say "Start llama-server, then press ^R", which handed back a job the
        # rest of the tool had already taken over -- and left the picker empty
        # on exactly the first run where someone needs it.
        self._server = ManagedLlamaServer(settings)
        try:
            await self._server.ensure_running()
        except LlamaServerError as error:
            status.update(str(error))
            return

        provider = LlamaCppProvider(settings)
        try:
            models = await provider.list_models()
        except AIProviderError as error:
            status.update(str(error))
            return
        finally:
            await provider.close()

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
                key=model.key,
            )

        # Start on the configured model rather than at the top, so the list
        # opens showing what is in use instead of making someone find it.
        if self._current is not None:
            for index, model in enumerate(models):
                if model.key == self._current:
                    table.move_cursor(row=index)
                    break

        status.update(f"{len(models)} available")
        table.focus()

    def action_reload(self) -> None:
        self.query_one("#picker-status", Static).update("Looking for models...")
        self.run_worker(self._load(), exclusive=True)

    async def _release_server(self) -> None:
        """Stop the server this screen started, if it started one.

        Picking a model must not leave a process behind. A server that was
        already running is untouched -- `stop()` no-ops unless we spawned it.
        """
        server, self._server = self._server, None
        if server is not None:
            await server.stop()

    @on(DataTable.RowSelected)
    def choose(self, event: DataTable.RowSelected) -> None:
        if event.row_key.value == FOLDER_ROW_KEY:
            self._change_folder()
            return
        self.run_worker(self._release_server())
        self.dismiss(ModelChoice(model=event.row_key.value, models_dir=self._models_dir))

    def _change_folder(self) -> None:
        """Open the folder browser, then relist without leaving the screen.

        The old server is stopped first: it was started against the previous
        folder and its preset names models from there, so keeping it would show
        the old list under the new folder's name.
        """
        def chosen(folder: str | None) -> None:
            if not folder or folder == self._models_dir:
                return
            self._models_dir = folder
            self.run_worker(self._reload_for_new_folder(), exclusive=True)

        self.app.push_screen(FolderPickerScreen(self._models_dir), chosen)

    async def _reload_for_new_folder(self) -> None:
        await self._release_server()
        self.query_one("#picker-status", Static).update("Looking for models...")
        await self._load()

    def action_cancel(self) -> None:
        self.run_worker(self._release_server())
        # The folder still comes back on cancel. Someone who set it and then
        # decided against the models they saw has still told us something
        # true, and making them set it twice would be the screen forgetting.
        self.dismiss(ModelChoice(model=None, models_dir=self._models_dir))


class SettingsPane(Vertical):
    """The settings rows, the help line, and the paths block.

    Focusable, because the bindings live here now rather than on a host app --
    a widget only receives key bindings when it has focus, and this pane is the
    thing being driven. Both hosts focus it on mount.
    """

    # height: auto on the body, NOT 1fr. With 1fr the rows container ate every
    # spare line and pushed the paths to the floor of the terminal, leaving a
    # lake of empty space in the middle. Sized to content, each block sits
    # under the one above it and the slack collects at the bottom, where
    # nobody has to look at it.
    can_focus = True

    BINDINGS: ClassVar = [
        Binding("up", "move(-1)", "move", show=False),
        Binding("down", "move(1)", "move", show=False),
        Binding("left", "step(-1)", "change", show=False),
        Binding("right", "step(1)", "change", show=False),
        Binding("enter", "edit", "edit"),
        Binding("r", "reset", "reset row"),
        Binding("ctrl+s", "save", "save"),
        Binding("escape", "leave", "close"),
    ]

    class Closed(Message):
        """Escape was pressed with nothing unsaved holding it back.

        What that means is the host's decision: the standalone command exits,
        the unified UI just moves focus. The pane does not have an opinion.
        """

        def __init__(self, saved: bool) -> None:
            super().__init__()
            self.saved = saved

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

    @property
    def config_path(self) -> Path:
        return self._config_path

    def compose(self) -> ComposeResult:
        yield Static(id="title")
        with VerticalScroll(id="settings-rows"):
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
        # Directly under the list, not at the bottom of the screen: it
        # describes the row the cursor is on, and a caption that far from what
        # it captions stops being read as connected to it.
        yield Static(id="help", classes="help")
        yield Static(id="paths", classes="paths")
        yield Static(id="status")

    def on_mount(self) -> None:
        self.query_one("#paths", Static).update(
            Text(f"database  {default_database_path()}\n"
                 f"config    {self._config_path}")
        )
        self._redraw()
        # Deliberately does NOT focus itself. Every tab's content mounts when
        # the app starts, not when its tab is first shown, so a pane that grabs
        # focus on mount drags the whole TabbedContent onto its own tab -- the
        # app opened on Scan and then jumped to Settings, and the first
        # keystroke typed into the username field was swallowed by the move.
        # Focus is the host's call: the standalone app takes it immediately,
        # the unified UI hands it over when this tab is actually activated.

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
            # The trade, kept terse. It has to survive the cursor being
            # elsewhere, which is exactly when a warning gets missed -- the
            # prose lives on the help line below. Both sides of the choice are
            # labelled, but only the risky one is coloured like a warning.
            note = field_note(field, self._values[field.key])
            if note:
                if note.warning:
                    tone = "yellow" if selected else "dim yellow"
                else:
                    tone = "dim"
                line.append(note.text, style=tone)
            row.update(line)
            row.set_class(selected, "row-selected")
        self._redraw_help()
        self.query_one("#status", Static).update(self._status)

    def _redraw_help(self) -> None:
        """Describe the row the cursor is on."""
        field = SETTING_FIELDS[self._cursor]
        help_text = Text(
            field_description(field, self._values[field.key]),
            style="dim",
        )
        # The URL rides with the prose rather than the row: it is a "read more"
        # on an explanation, and only worth offering to someone who is on that
        # row deciding. Printed plainly -- see TRANSPORT_DOC_URL for why it is
        # not a terminal hyperlink.
        if help_text.plain and field.doc_url:
            help_text.append(f"  {field.doc_url}", style="dim")
        self.query_one("#help", Static).update(help_text)

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
            models_dir = self._values.get("ai.models_dir")
            self.app.push_screen(
                ModelPickerScreen(
                    str(endpoint),
                    current=self._values.get("ai.model"),
                    models_dir=str(models_dir) if models_dir else None,
                ),
                self._accept_model_choice,
            )
            return
        if field.kind == "folder":
            current = self._values.get(field.key)
            self.app.push_screen(
                FolderPickerScreen(str(current) if current else None),
                lambda chosen: self._accept(field, chosen or None),
            )
            return
        if field.kind == "text":
            self.app.push_screen(
                TextEditScreen(field.label, str(self._values[field.key] or "")),
                lambda edited: self._accept(field, edited or None),
            )
            return
        self.action_step(1)

    def _accept(self, field: SettingField, value: Any) -> None:
        if value is not None or field.kind == "text":
            self._values[field.key] = value
        self._redraw()

    def _accept_model_choice(self, choice: ModelChoice | None) -> None:
        """Take back both values the picker can change.

        The folder is applied even when no model was chosen: someone who fixed
        a wrong folder and then backed out has still told us where their models
        are, and asking again next time would be the screen forgetting.
        """
        if choice is None:
            self._redraw()
            return
        if choice.models_dir:
            self._values["ai.models_dir"] = choice.models_dir
        if choice.model:
            self._values["ai.model"] = choice.model
        self._redraw()

    def action_reset(self) -> None:
        """Restore the value this build ships with -- not the stored one.

        It used to restore the SAVED value, which was a key spent on something
        Esc already does: abandoning unsaved edits is exactly what leaving
        without saving means. The only reset worth its own binding is the one
        nothing else offers -- "what was this before anyone touched it" --
        which is also the way out of a config someone has edited into a corner
        without having to remember what the tool's own number was.
        """
        field = SETTING_FIELDS[self._cursor]
        default = field_default(field)
        if default is NO_DEFAULT:
            self._status = f"{field.label} has no default — it must be set"
        else:
            self._values[field.key] = default
            self._status = (
                f"{field.label} back to the default "
                f"({plain_value(field, default)})"
            )
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
        self.post_message(
            self.Closed(
                self._saved != field_values(try_load_settings(path=self._config_path))
            )
        )
