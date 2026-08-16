"""The anchor editor: facts you already know about the person being scanned.

An anchor is what turns a pile of accounts that happen to share a username into
a profile about one person. Without one, synthesis runs in aggregate mode and
every value it reports carries the caveat that it may describe someone else
entirely -- so this is not a power-user extra, it is the difference between a
result you can act on and a result you have to qualify.

**Why a dialog rather than a row on the scan screen.** Anchors are a
variable-length list of structured records, and every panel on that screen has
a fixed height. Inline, two anchors and five anchors would lay the screen out
differently and push the findings table around before a scan had even started.
A dialog keeps the screen still and holds any number of them -- and it is the
pattern the settings editor already uses for anything a spinner cannot express.

**Reached from the ANCHORS section, which appears only when analysis is on.**
Anchors feed the AI's second pass and nothing else, so with analysis off this is
a control that cannot do anything. The usual objection to hiding a control is
that it cannot then be discovered -- answered here by revealing it with the
toggle that makes it matter, and by its empty state saying what anchors are for.

**There is no trust picker, deliberately.** It offered `verified`, `strong` and
`context`, and the user's objection to it was right on every count:

- The model never sees it. `AIService._format_anchors` reduces anchors to
  `{field: [values]}` before they reach the prompt, so trust cannot influence
  matching -- it is not sent.
- Nothing branches on it. `has_trusted_anchor` is the only code that separates
  `verified`/`strong` from `context`, and it has no callers; every real decision
  runs off `has_anchors`, which is binary. Its own definition treats `verified`
  and `strong` as one thing, which is the codebase admitting they are synonyms.
- Its one live effect is breaking ties between duplicate fields.

So it was a three-way choice, between two words the code itself could not
distinguish, that changed nothing the operator could observe. `AnchorTrust`
stays in the model and `--anchor verified:field=value` still parses -- stored
profiles and existing scripts keep working -- but this screen no longer asks a
question with no consequence.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import ValidationError
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, Label, Static

from sherlock_project.profile_synthesis import IdentityAnchor

# Where an anchor came from, recorded on every one this screen makes. The CLI
# writes "command_line"; keeping them distinct means a profile can still say
# which surface an anchor was typed into months later. Not printed in the
# profile panel -- see INTERNAL_ANCHOR_SOURCES.
ANCHOR_SOURCE = "user_interface"


class AnchorScreen(ModalScreen[list[IdentityAnchor] | None]):
    """Add, edit and remove the anchors for this run.

    Dismisses with the full list, or None if nothing was changed -- the caller
    treats those differently, because replacing a list with an identical copy
    would otherwise look like an edit.
    """

    BINDINGS: ClassVar = [
        Binding("escape", "close", "done"),
        Binding("delete", "remove", "remove"),
    ]

    def __init__(self, anchors: list[IdentityAnchor] | None = None) -> None:
        super().__init__()
        # Copied, not aliased. The caller keeps its list untouched until this
        # screen is dismissed, so leaving with Escape really does leave the run
        # as it was rather than having edited it in place all along.
        self._anchors: list[IdentityAnchor] = list(anchors or [])
        self._dirty = False

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Anchors", classes="dialog-title")
            yield Static(
                Text(
                    "Facts you already know about this person. Without one, "
                    "the profile describes whoever shares the username.",
                    style="dim",
                ),
                id="anchor-blurb",
            )
            yield DataTable(id="anchor-list", cursor_type="row")

            with Grid(id="anchor-form"):
                yield Static("field", classes="anchor-label")
                yield Input(placeholder="full_name", id="anchor-field")
                yield Static("value", classes="anchor-label")
                yield Input(placeholder="Avery Stone", id="anchor-value")

            yield Static(id="anchor-status")
            yield Label(
                "⏎ add    del remove    esc done",
                classes="dim",
            )

    def on_mount(self) -> None:
        table = self.query_one("#anchor-list", DataTable)
        table.add_column("field", key="field", width=16)
        table.add_column("value", key="value")
        self._redraw()
        self.query_one("#anchor-field", Input).focus()

    # -- drawing ------------------------------------------------------------

    def _redraw(self) -> None:
        table = self.query_one("#anchor-list", DataTable)
        table.clear()
        if not self._anchors:
            table.add_row(
                Text("none yet", style="dim italic"),
                Text("the profile will describe anyone with this username",
                     style="dim italic"),
            )
        else:
            for index, anchor in enumerate(self._anchors):
                table.add_row(
                    Text(anchor.field, overflow="ellipsis", no_wrap=True),
                    Text(anchor.value, overflow="ellipsis", no_wrap=True),
                    key=str(index),
                )

    def _status(self, message: str, *, error: bool = False) -> None:
        self.query_one("#anchor-status", Static).update(
            Text(message, style="red" if error else "dim")
        )

    # -- editing ------------------------------------------------------------

    @on(Input.Submitted, "#anchor-field")
    def _field_submitted(self) -> None:
        # Enter in the first box moves to the second rather than adding a
        # half-filled anchor -- the form reads left to right and so should the
        # keyboard.
        self.query_one("#anchor-value", Input).focus()

    @on(Input.Submitted, "#anchor-value")
    def _value_submitted(self) -> None:
        self.action_add()

    def action_add(self) -> None:
        field = self.query_one("#anchor-field", Input).value.strip()
        value = self.query_one("#anchor-value", Input).value.strip()
        try:
            anchor = IdentityAnchor(
                field=field,
                value=value,
                source=ANCHOR_SOURCE,
            )
        except ValidationError:
            # The model already refuses empty text; saying which box is empty
            # is more use than repeating pydantic at someone.
            missing = "field" if not field else "value"
            self._status(f"Fill in {missing} first.", error=True)
            return

        if any(
            existing.field == anchor.field and existing.value == anchor.value
            for existing in self._anchors
        ):
            self._status(f"{anchor.field}={anchor.value} is already listed.")
            return

        self._anchors.append(anchor)
        self._dirty = True
        for box in ("#anchor-field", "#anchor-value"):
            self.query_one(box, Input).value = ""
        self.query_one("#anchor-field", Input).focus()
        self._status(f"Added {anchor.field}={anchor.value}")
        self._redraw()

    def action_remove(self) -> None:
        if not self._anchors:
            return
        table = self.query_one("#anchor-list", DataTable)
        row = table.cursor_row
        if row is None or not (0 <= row < len(self._anchors)):
            return
        removed = self._anchors.pop(row)
        self._dirty = True
        self._status(f"Removed {removed.field}={removed.value}")
        self._redraw()

    def action_close(self) -> None:
        self.dismiss(list(self._anchors) if self._dirty else None)
