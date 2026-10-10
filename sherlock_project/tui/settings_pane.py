"""The settings editor, as a pane two different hosts can mount.

It began as a whole `App` behind `sherlock-rm settings`, which was right while that
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
from textual.widgets import DataTable, Input, Label, Static

from sherlock_project.ai_config import (
    DEFAULT_LLAMACPP_BASE_URL,
    AIConfigError,
    AISettings,
    ai_config_path,
    save_settings,
    try_load_settings,
)
from sherlock_project.ai_provider import (
    CLOUD_PRESETS,
    AIModelInfo,
    AIProviderAuthenticationError,
    AIProviderError,
    LlamaCppProvider,
    OpenAICompatibleProvider,
    chat_models,
)
from sherlock_project.ai_secrets import (
    delete_api_key,
    describe_key_source,
    keychain_available,
    keychain_label,
    no_keychain_message,
    resolve_api_key,
    store_api_key,
)
from sherlock_project.database import default_database_path
from sherlock_project.llama_server import (
    LlamaServerError,
    ManagedLlamaServer,
    discover_models,
)
from sherlock_project.settings import (
    NO_DEFAULT,
    PROVIDER_LABELS,
    SETTING_FIELDS,
    IncompleteSettingsError,
    SettingField,
    apply_values,
    field_default,
    current_provider,
    field_applies,
    field_description,
    field_note,
    field_values,
    inapplicable_note,
    step_value,
    value_label,
)
from sherlock_project.tui.confirm_screen import ConfirmScreen

SECTION_TITLES = {
    "ai": "AI",
    "scan": "Scan",
    "output": "Output",
    "update": "Update",
}
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

    def __init__(self, label: str, value: str, *, password: bool = False) -> None:
        super().__init__()
        self._label = label
        self._value = value
        # Masked as typed, for a secret. Pasting a key into a visible box puts
        # it on screen for anyone behind you and in any screen recording.
        self._password = password

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"Set {self._label}")
            yield Input(value=self._value, id="edit", password=self._password)
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


def folder_warning(folder: str | None) -> str:
    """What is wrong with this models folder, or "" if nothing is.

    Checked where it is typed rather than left for the scan to discover: a
    path with a typo, or one pointing at a parent two levels off, is
    indistinguishable from a correct one until something looks inside it.

    Uses the same recursive search the server will run, so a folder that
    passes here is a folder that will serve models -- any layout, since the
    generated preset takes absolute paths.
    """
    if not folder:
        return ""
    path = Path(folder).expanduser()
    if not path.is_dir():
        return f"No such folder: {path}"
    if not discover_models([path]):
        return f"No .gguf models found in {path}"
    return ""


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
                "No models folder set — press enter on the first row to set one."
            )
            table.focus()
            return

        warning = folder_warning(self._models_dir)
        if warning:
            status.update(warning)
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
        """Ask for the folder, then relist without leaving the screen.

        A plain text box. A tree browser lived here and was removed: it needed
        drive enumeration, re-rooting, and a cancellable disk walk to be usable
        at all, and a path can be pasted straight out of a file manager faster
        than it can be arrowed to. What it was really buying was feedback, and
        `folder_warning` gives that without the machinery.

        The old server is stopped first: it was started against the previous
        folder and its preset names models from there, so keeping it would show
        the old list under the new folder's name.
        """
        def chosen(folder: str | None) -> None:
            folder = (folder or "").strip()
            if not folder or folder == self._models_dir:
                return
            self._models_dir = folder
            self.run_worker(self._reload_for_new_folder(), exclusive=True)

        self.app.push_screen(
            TextEditScreen("models folder", self._models_dir or ""),
            chosen,
        )

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


class CloudModelPickerScreen(ModalScreen[ModelChoice]):
    """Enter on `model` with a hosted provider: what the key can use.

    The local picker's job -- start a server, read a folder -- does not exist
    here. What does is the one question a hosted setup has to answer before a
    scan wastes a run on it: does this key work? Listing models is that check,
    so the status line says "key rejected" or "no key" in the place someone
    is already looking, rather than on the first site of the next scan.
    """

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "cancel"),
        Binding("ctrl+r", "reload", "refresh"),
    ]

    def __init__(self, settings: AISettings, current: str | None = None) -> None:
        super().__init__()
        self._settings = settings
        self._current = current

    def compose(self) -> ComposeResult:
        label = CLOUD_PRESETS[self._settings.provider].label
        with Vertical(id="dialog"):
            yield Label(f"Choose a {label} model", classes="dialog-title")
            yield Static("Checking the key...", id="picker-status")
            yield DataTable(id="models", cursor_type="row", zebra_stripes=True)
            yield Label("▸ will be used", classes="dim")
            yield Label("up/down move   enter select   ^R refresh   esc cancel",
                        classes="dim")

    def on_mount(self) -> None:
        table = self.query_one("#models", DataTable)
        table.add_column("", key="mark", width=2)
        table.add_column("model", key="model")
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        status = self.query_one("#picker-status", Static)
        table = self.query_one("#models", DataTable)
        table.clear()
        provider = OpenAICompatibleProvider(self._settings)
        try:
            models = chat_models(await provider.list_models())
        except AIProviderAuthenticationError as error:
            status.update(f"Key problem: {error}")
            return
        except AIProviderError as error:
            status.update(str(error))
            return
        finally:
            await provider.close()

        for model in models:
            table.add_row(
                Text("▸" if model.key == self._current else " ", style="bold cyan"),
                model.key,
                key=model.key,
            )
        for index, model in enumerate(models):
            if model.key == self._current:
                table.move_cursor(row=index)
                break
        status.update(f"Key works. {len(models)} models available.")
        table.focus()

    def action_reload(self) -> None:
        self.query_one("#picker-status", Static).update("Checking the key...")
        self.run_worker(self._load(), exclusive=True)

    @on(DataTable.RowSelected)
    def choose(self, event: DataTable.RowSelected) -> None:
        self.dismiss(ModelChoice(model=event.row_key.value))

    def action_cancel(self) -> None:
        self.dismiss(ModelChoice(model=None))


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
        self._key_status_cache: dict[tuple[str, str], str] = {}

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
            applies = field_applies(field, self._values)
            if field.kind == "secret":
                shown = Text(
                    self._key_status() if applies else "—",
                    style="bold" if applies else "dim",
                )
                shown.pad_right(max(0, VALUE_WIDTH - shown.cell_len))
                line.append_text(shown)
            elif applies:
                line.append_text(styled_value(field, self._values[field.key]))
            else:
                # Drawn, so the layout does not jump when the provider
                # changes, but plainly not in play.
                shown = Text(render_value(field, self._values[field.key]), style="dim")
                shown.pad_right(max(0, VALUE_WIDTH - shown.cell_len))
                line.append_text(shown)
            if applies and field.kind in {"text", "model", "secret"}:
                line.append("⏎ change", style="dim")
            # The trade, kept terse. It has to survive the cursor being
            # elsewhere, which is exactly when a warning gets missed -- the
            # prose lives on the help line below. Both sides of the choice are
            # labelled, but only the risky one is coloured like a warning.
            note = (
                field_note(field, self._values[field.key])
                if applies
                else inapplicable_note(field, self._values)
            )
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

    def _refuse_inapplicable(self, field: SettingField) -> bool:
        if field_applies(field, self._values):
            return False
        self._status = (
            f"{field.label} is {inapplicable_note(field, self._values).text}."
        )
        self._redraw()
        return True

    def action_step(self, delta: int) -> None:
        field = SETTING_FIELDS[self._cursor]
        if self._refuse_inapplicable(field):
            return
        current = self._values[field.key]
        if field.key == "ai.provider":
            self._change_provider(step_value(field, current or "llamacpp", delta))
            return
        if current is None:
            self._status = f"{field.label} is unset — press ⏎ to set it"
        else:
            self._values[field.key] = step_value(field, current, delta)
            self._status = ""
        self._redraw()

    def action_edit(self) -> None:
        field = SETTING_FIELDS[self._cursor]
        if self._refuse_inapplicable(field):
            return
        if field.kind == "secret":
            self._edit_api_key()
            return
        if field.kind == "model" and current_provider(self._values) in CLOUD_PRESETS:
            settings = self._provisional_cloud_settings()
            if settings is None:
                return
            self.app.push_screen(
                CloudModelPickerScreen(settings, current=self._values.get("ai.model")),
                self._accept_model_choice,
            )
            return
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
                TextEditScreen(field.label, str(current) if current else ""),
                lambda edited: self._accept_folder(field, edited),
            )
            return
        if field.kind == "text":
            self.app.push_screen(
                TextEditScreen(field.label, str(self._values[field.key] or "")),
                lambda edited: self._accept(field, edited or None),
            )
            return
        self.action_step(1)

    # -- hosted providers ------------------------------------------------

    def _preset(self):
        return CLOUD_PRESETS.get(current_provider(self._values))

    def _key_env_name(self) -> str:
        preset = self._preset()
        stored = self._stored.ai
        if stored is not None and stored.provider == current_provider(self._values):
            if stored.api_key_env:
                return stored.api_key_env
        return preset.api_key_env if preset is not None else ""

    def _key_status(self) -> str:
        """Where the key comes from, cached per provider.

        Cached because the row redraws on every keypress, and a Secret Service
        lookup is a D-Bus round trip. Invalidated wherever the answer can
        change: a key stored, a key removed.
        """
        preset = self._preset()
        if preset is None:
            return "—"
        env_name = self._key_env_name()
        cache_key = (preset.name, env_name)
        if cache_key not in self._key_status_cache:
            self._key_status_cache[cache_key] = describe_key_source(
                preset.name, env_name
            )
        return self._key_status_cache[cache_key]

    def _provisional_cloud_settings(self) -> AISettings | None:
        """Settings from the rows as they stand, saved or not, for the picker."""
        preset = self._preset()
        try:
            return AISettings(
                provider=preset.name,
                base_url=self._values.get("ai.base_url") or preset.base_url,
                model=self._values.get("ai.model") or "listing",
                api_key_env=self._key_env_name(),
                requests_per_minute=self._values.get("ai.requests_per_minute"),
            )
        except ValueError as error:
            self._status = f"Endpoint is not usable: {error}"
            self._redraw()
            return None

    def _change_provider(self, provider: str) -> None:
        """Switch provider, asking first when the switch sends pages away.

        The confirmation is the point at which "nothing leaves your machine"
        stops being true, so it is a dialog someone has to answer, not a note
        they might not read.
        """
        previous = current_provider(self._values)
        if provider == previous:
            return
        preset = CLOUD_PRESETS.get(provider)
        if preset is None:
            self._apply_provider(provider)
            return

        def answered(confirmed: bool | None) -> None:
            if confirmed:
                self._apply_provider(provider)
            else:
                self._status = (
                    f"Still {PROVIDER_LABELS.get(previous, previous)}; "
                    "nothing changed."
                )
                self._redraw()

        self.app.push_screen(
            ConfirmScreen(
                f"Use {preset.label}?",
                preset.privacy_note
                + "\n\nThe local model stays configured; switch back here "
                "at any time.",
                confirm_label=f"Use {preset.label}",
            ),
            answered,
        )

    def _apply_provider(self, provider: str) -> None:
        previous = current_provider(self._values)
        old_preset = CLOUD_PRESETS.get(previous)
        new_preset = CLOUD_PRESETS.get(provider)
        old_default = old_preset.base_url if old_preset else DEFAULT_LLAMACPP_BASE_URL
        new_default = new_preset.base_url if new_preset else DEFAULT_LLAMACPP_BASE_URL
        self._values["ai.provider"] = provider
        # The endpoint follows the provider only while it is still the old
        # provider's default. One the user typed is theirs, and stays.
        if self._values.get("ai.base_url") in (None, "", old_default):
            self._values["ai.base_url"] = new_default
        # A model name means nothing to the other provider.
        stored = self._stored.ai
        self._values["ai.model"] = (
            stored.model if stored is not None and stored.provider == provider else None
        )
        if new_preset is not None and self._values.get("ai.requests_per_minute") is None:
            self._values["ai.requests_per_minute"] = new_preset.requests_per_minute
        self._status = (
            f"{new_preset.label}: set an API key and pick a model, then ^S."
            if new_preset is not None
            else "Local model: pick a model, then ^S."
        )
        self._redraw()

    def _edit_api_key(self) -> None:
        """Take a key and hand it to the OS keychain immediately.

        Immediately, not on ^S: the key is not part of the config file, and
        holding it as unsaved pane state would keep a secret in memory for no
        reason, and lose it on Esc. The status line says where it went.
        """
        preset = self._preset()
        env_name = self._key_env_name()
        if not keychain_available():
            self._status = no_keychain_message(env_name)
            self._redraw()
            return

        def entered(key: str | None) -> None:
            if key is None:
                self._redraw()
                return
            if not key:
                self._confirm_key_removal()
                return
            self._key_status_cache.clear()
            try:
                store_api_key(preset.name, key, env_name=env_name)
            except AIConfigError as error:
                self._status = str(error)
            else:
                _, source = resolve_api_key(preset.name, env_name)
                self._status = f"Key saved to {keychain_label()}." + (
                    f" ${env_name} is also set and takes precedence."
                    if source == "env"
                    else ""
                )
            self._redraw()

        self.app.push_screen(
            TextEditScreen(f"{preset.label} API key", "", password=True),
            entered,
        )

    def _confirm_key_removal(self) -> None:
        preset = self._preset()

        def answered(confirmed: bool | None) -> None:
            if confirmed:
                self._key_status_cache.clear()
                try:
                    removed = delete_api_key(preset.name)
                except AIConfigError as error:
                    self._status = str(error)
                else:
                    self._status = (
                        f"Key removed from {keychain_label()}."
                        if removed
                        else "No key was stored."
                    )
            self._redraw()

        self.app.push_screen(
            ConfirmScreen(
                f"Remove the stored {preset.label} key?",
                f"It is deleted from {keychain_label()}. A "
                f"${self._key_env_name()} environment variable is not affected.",
                confirm_label="Remove",
            ),
            answered,
        )

    def _accept(self, field: SettingField, value: Any) -> None:
        if value is not None or field.kind == "text":
            self._values[field.key] = value
        self._redraw()

    def _accept_folder(self, field: SettingField, edited: str | None) -> None:
        """Take the typed path, and say so immediately if nothing is in it.

        Stored either way. A folder that is empty today may be where the user
        is about to put models, and refusing to remember what they typed would
        make the warning a rejection -- which it is not.
        """
        value = (edited or "").strip() or None
        self._values[field.key] = value
        warning = folder_warning(value)
        if warning:
            self._status = warning
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
