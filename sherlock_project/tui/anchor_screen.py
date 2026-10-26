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

import re
from typing import ClassVar

from pydantic import ValidationError
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.widgets import Button, DataTable, Input, Label, Static

from sherlock_project.profile_synthesis import (
    IdentityAnchor,
    canonical_field,
    normalize_value,
)

# Where an anchor came from, recorded on every one this screen makes. The CLI
# writes "command_line"; keeping them distinct means a profile can still say
# which surface an anchor was typed into months later. Not printed in the
# profile panel -- see INTERNAL_ANCHOR_SOURCES.
ANCHOR_SOURCE = "user_interface"

# What the field box completes to, as a person would type it.
#
# The field is free text, not a picker. A picker of five was there so nobody
# had to know the profile's vocabulary -- but every consumer of an anchor runs
# the field through `canonical_field` first, so "employer", "Employer" and
# "organizations" already reach synthesis as the same thing. The picker was
# guarding a door that was not locked, at the cost of an "Other…" detour to a
# second box for anything it did not list.
#
# A blank box has the opposite fault -- nothing says what works, and a typo
# becomes a field nothing else uses -- so the box completes from this list as
# you type, greyed in, accepted with → or Tab. Ordered by how often a person
# knows each about a target.
FIELD_SUGGESTIONS: tuple[str, ...] = (
    "full name",
    "email",
    "location",
    "employer",
    "role",
    "website",
    "phone",
    "alias",
    "username",
    "language",
)

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL = re.compile(r"^(https?://|www\.)\S+$", re.IGNORECASE)


def anchor_label(field: str) -> str:
    """A stored field name as a person would say it: `full_name` -> `full name`."""
    return " ".join(field.replace("_", " ").split())


def infer_field(value: str) -> str | None:
    """The field a value plainly is, when it plainly is one.

    Only the two shapes that cannot be anything else. A bare word could be a
    name, a city or an employer, and guessing there would file a fact under
    the wrong heading without saying so.
    """
    if _EMAIL.match(value):
        return "email"
    if _URL.match(value):
        return "website"
    return None


class AnchorScreen(ModalScreen[list[IdentityAnchor] | None]):
    """Add, edit and remove the anchors for this run.

    Dismisses with the full list, or None if nothing was changed -- the caller
    treats those differently, because replacing a list with an identical copy
    would otherwise look like an edit.
    """

    BINDINGS: ClassVar = [
        Binding("escape", "close", "done"),
    ]

    def __init__(
        self,
        anchors: list[IdentityAnchor] | None = None,
        *,
        username: str | None = None,
    ) -> None:
        super().__init__()
        self._username = username
        # Copied, not aliased. The caller keeps its list untouched until this
        # screen is dismissed, so leaving with Escape really does leave the run
        # as it was rather than having edited it in place all along.
        self._anchors: list[IdentityAnchor] = list(anchors or [])
        self._dirty = False
        self._error_for: tuple[str, str] | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            title = Text("Anchors")
            if self._username:
                title.append(f" for {self._username}")
            yield Label(title, classes="dialog-title")
            yield Static(
                Text(
                    "Facts you already know about this person. The profile "
                    "keeps accounts that agree with them; without any, it "
                    "describes whoever shares the username.",
                    style="dim",
                ),
                id="anchor-blurb",
            )
            yield _AnchorList(id="anchor-list", cursor_type="row")

            # One row: what it is, what it says, and the button that adds it.
            # Add sits WITH the boxes it submits, the same height as them --
            # as a chip in the footer it read as a lesser Done, and it acted on
            # something three rows away.
            with Horizontal(id="anchor-form"):
                field = Input(
                    placeholder="full name, email…",
                    suggester=SuggestFromList(
                        FIELD_SUGGESTIONS, case_sensitive=False
                    ),
                    id="anchor-field",
                )
                # Labels in the border, so they stay readable after typing --
                # a placeholder alone vanishes with the first keystroke.
                field.border_title = "field"
                yield field
                value = Input(placeholder="e.g. Avery Stone", id="anchor-value")
                value.border_title = "value"
                yield value
                yield Button("Add", id="anchor-add")

            # Errors only. Success needs no sentence: the new row appears in
            # the list, highlighted, which says more than "Added x=y" did.
            yield Static(id="anchor-status")
            with Horizontal(id="anchor-buttons"):
                yield Button("Remove selected", id="anchor-remove")
                yield Static(classes="spacer")
                yield Button("Done", variant="primary", id="anchor-done")

    def on_mount(self) -> None:
        table = self.query_one("#anchor-list", DataTable)
        table.add_column("field", key="field", width=16)
        table.add_column("value", key="value")
        self._redraw()
        self._status("")
        # The list when there is something in it, so `del` acts on an anchor
        # straight away. It opened in the text field before, where `del` is the
        # field's own delete-forward -- the one key the dialog advertised for
        # removing an anchor deleted a character instead.
        if self._anchors:
            table.focus()
        else:
            self.query_one("#anchor-field", Input).focus()

    @on(Button.Pressed, "#anchor-add")
    def _add_pressed(self) -> None:
        self.action_add()

    @on(Button.Pressed, "#anchor-remove")
    def _remove_pressed(self) -> None:
        self.action_remove()

    @on(Button.Pressed, "#anchor-done")
    def _done_pressed(self) -> None:
        self.action_close()

    # -- drawing ------------------------------------------------------------

    def _redraw(self, *, select: int | None = None) -> None:
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
                    Text(anchor_label(anchor.field), overflow="ellipsis",
                         no_wrap=True),
                    Text(anchor.value, overflow="ellipsis", no_wrap=True),
                    key=str(index),
                )
            if select is not None:
                table.move_cursor(row=min(select, len(self._anchors) - 1))
        # Nothing to remove is a disabled button, not a button that does
        # nothing when pressed.
        self.query_one("#anchor-remove", Button).disabled = not self._anchors

    def _boxes(self) -> tuple[str, str]:
        return (
            self.query_one("#anchor-field", Input).value,
            self.query_one("#anchor-value", Input).value,
        )

    def _status(self, message: str) -> None:
        # Remembered with what the boxes held, so only an EDIT clears it -- not
        # a Changed event still queued from before the error was raised.
        self._error_for = self._boxes() if message else None
        status = self.query_one("#anchor-status", Static)
        status.update(Text(message, style="red"))
        # No line at all when there is nothing wrong, rather than a blank one
        # holding the footer a row lower than it needs to be.
        status.display = bool(message)

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

    @on(Input.Changed)
    def _typing(self) -> None:
        # An error is about what WAS in the boxes; once they change it is stale.
        if self._error_for is not None and self._error_for != self._boxes():
            self._status("")

    def action_add(self) -> None:
        field_box = self.query_one("#anchor-field", Input)
        value_box = self.query_one("#anchor-value", Input)
        value = value_box.value.strip()
        field = " ".join(field_box.value.split()).casefold()
        if not value:
            self._status("Type what you know in the value box.")
            value_box.focus()
            return
        if not field:
            field = infer_field(value) or ""
        if not field:
            self._status("Say what this is in the field box, e.g. full name.")
            field_box.focus()
            return
        try:
            anchor = IdentityAnchor(field=field, value=value, source=ANCHOR_SOURCE)
        except ValidationError:
            self._status("That anchor could not be read; check both boxes.")
            return

        # Duplicates compared the way synthesis will see them, so "Full name"
        # and "full_name" with the same value are one anchor, not two.
        def key(item: IdentityAnchor) -> tuple[str, str]:
            return (
                canonical_field(item.field),
                normalize_value(item.field, item.value),
            )

        if any(key(existing) == key(anchor) for existing in self._anchors):
            self._status(f"{anchor_label(field)} {value} is already listed.")
            return

        self._anchors.append(anchor)
        self._dirty = True
        field_box.value = ""
        value_box.value = ""
        self._status("")
        self._redraw(select=len(self._anchors) - 1)
        field_box.focus()

    def action_remove(self) -> None:
        if not self._anchors:
            return
        table = self.query_one("#anchor-list", DataTable)
        row = table.cursor_row
        if row is None or not (0 <= row < len(self._anchors)):
            return
        self._anchors.pop(row)
        self._dirty = True
        self._redraw(select=row)
        if not self._anchors:
            self.query_one("#anchor-field", Input).focus()

    def action_close(self) -> None:
        self.dismiss(list(self._anchors) if self._dirty else None)


class _AnchorList(DataTable):
    """The anchor table, with `del` bound where it can actually be heard.

    On the screen the binding lost to `Input`'s own delete-forward whenever a
    field had focus, which was always on open. Here it belongs to the list, so
    it fires exactly when an anchor is selected -- and the footer-less dialog
    names it on the button beside the list instead of in a hint line.
    """

    BINDINGS: ClassVar = [Binding("delete", "remove_anchor", "remove")]

    def action_remove_anchor(self) -> None:
        screen = self.screen
        if isinstance(screen, AnchorScreen):
            screen.action_remove()
